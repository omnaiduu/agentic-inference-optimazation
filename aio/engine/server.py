"""Minimal OpenAI-compatible /v1/completions server for our engine.

Same API as vLLM, so the same load generator and correctness check measure both.
Milestone 1 serves ONE request at a time (requests queue on a lock); batching
many users together is a later milestone.

  python -m aio.engine.server --model-dir DIR --mode graph --port 8001
"""
import argparse
import asyncio
import json
import threading
import time

from aiohttp import web
from transformers import AutoTokenizer

from aio.engine.model import EOS_IDS, Gemma4E2B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--mode", choices=["eager", "graph", "compile"], default="graph")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--quant-dir", default=None, help="Google QAT int4 checkpoint dir")
    ap.add_argument("--chunk", type=int, default=4,
                    help="tokens generated per host sync (graph modes); 1 = sync every token")
    ap.add_argument("--lm-head-bits", type=int, default=None, help="quantize lm_head to 4 or 8 bits")
    args = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = Gemma4E2B(args.model_dir, max_len=args.max_len, quant_dir=args.quant_dir,
                      lm_head_bits=args.lm_head_bits)
    print(f"weights loaded in {time.time() - t0:.1f}s", flush=True)
    if args.mode == "compile":
        model.enable_compile()
    if args.mode in ("graph", "compile"):
        t1 = time.time()
        model.capture_graphs()
        print(f"captured {len(model.graphs)} CUDA graphs in {time.time() - t1:.1f}s", flush=True)
    decode = model.decode_eager if args.mode == "eager" else model.decode_graph
    lock = threading.Lock()

    def generate(prompt_ids, max_tokens, ignore_eos, emit):
        """Runs in a worker thread. Calls emit(token_id) for every generated token."""
        with lock:
            if len(prompt_ids) + max_tokens > args.max_len:
                raise ValueError("prompt + max_tokens exceeds max_len")
            t = model.prefill(prompt_ids)
            emit(t)                       # first token goes out immediately (TTFT)
            done, pos = 1, len(prompt_ids)
            while done < max_tokens and (ignore_eos or t not in EOS_IDS):
                k = min(args.chunk, max_tokens - done)
                if args.mode == "eager" or k == 1:
                    toks = [decode(t, pos)]
                else:
                    toks = model.decode_many(t, pos, k)
                for tt in toks:
                    emit(tt)
                    done += 1
                    if not ignore_eos and tt in EOS_IDS:
                        break
                pos += len(toks)
                t = toks[-1]
                if not ignore_eos and any(x in EOS_IDS for x in toks):
                    break

    async def completions(request: web.Request):
        body = await request.json()
        prompt = body["prompt"]
        prompt_ids = prompt if isinstance(prompt, list) else tok.encode(prompt)
        max_tokens = int(body.get("max_tokens", 16))
        ignore_eos = bool(body.get("ignore_eos", False))
        stream = bool(body.get("stream", False))

        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        fut = loop.run_in_executor(None, generate, prompt_ids, max_tokens, ignore_eos,
                                   lambda t: loop.call_soon_threadsafe(q.put_nowait, t))
        fut.add_done_callback(lambda _: loop.call_soon_threadsafe(q.put_nowait, None))

        out_ids: list[int] = []
        text = ""
        if not stream:
            while (t := await q.get()) is not None:
                out_ids.append(t)
            await fut
            text = tok.decode(out_ids, skip_special_tokens=True)
            return web.json_response({"choices": [{"index": 0, "text": text}],
                                      "usage": {"prompt_tokens": len(prompt_ids),
                                                "completion_tokens": len(out_ids)}})

        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        while (t := await q.get()) is not None:
            out_ids.append(t)
            new_text = tok.decode(out_ids, skip_special_tokens=True)
            delta, text = new_text[len(text):], new_text
            msg = {"choices": [{"index": 0, "text": delta}]}
            await resp.write(f"data: {json.dumps(msg)}\n\n".encode())
        await fut
        usage = {"choices": [], "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(out_ids)}}
        await resp.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())
        await resp.write_eof()
        return resp

    def teacher_forced(prompt_ids, target_ids):
        """Feed the ORACLE's tokens one at a time (decode path) and record what we would have
        predicted at every step. Agreement = fraction of steps where we pick the oracle's token."""
        with lock:
            preds = [model.prefill(prompt_ids)]
            pos = len(prompt_ids)
            for t in target_ids[:-1]:
                preds.append(decode(t, pos))
                pos += 1
            return preds

    async def teacher_forced_route(request: web.Request):
        body = await request.json()
        loop = asyncio.get_running_loop()
        preds = await loop.run_in_executor(None, teacher_forced, body["prompt_ids"], body["target_ids"])
        return web.json_response({"preds": preds})

    async def health(_):
        return web.Response(text="ok")

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.add_routes([web.post("/v1/completions", completions), web.get("/health", health),
                    web.post("/aio/teacher_forced", teacher_forced_route)])
    print(f"serving on :{args.port} (mode={args.mode})", flush=True)
    web.run_app(app, port=args.port, print=None)


if __name__ == "__main__":
    main()
