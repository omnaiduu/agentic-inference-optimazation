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
    ap.add_argument("--mode", choices=["eager", "graph"], default="graph")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--max-len", type=int, default=4096)
    args = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = Gemma4E2B(args.model_dir, max_len=args.max_len)
    print(f"weights loaded in {time.time() - t0:.1f}s", flush=True)
    if args.mode == "graph":
        t1 = time.time()
        model.capture_graphs()
        print(f"captured {len(model.graphs)} CUDA graphs in {time.time() - t1:.1f}s", flush=True)
    decode = model.decode_graph if args.mode == "graph" else model.decode_eager
    lock = threading.Lock()

    def generate(prompt_ids, max_tokens, ignore_eos, emit):
        """Runs in a worker thread. Calls emit(token_id) for every generated token."""
        with lock:
            if len(prompt_ids) + max_tokens > args.max_len:
                raise ValueError("prompt + max_tokens exceeds max_len")
            t = model.prefill(prompt_ids)
            pos = len(prompt_ids)
            for n in range(max_tokens):
                emit(t)
                if (not ignore_eos and t in EOS_IDS) or n == max_tokens - 1:
                    break
                t = decode(t, pos)
                pos += 1

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

    async def health(_):
        return web.Response(text="ok")

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.add_routes([web.post("/v1/completions", completions), web.get("/health", health)])
    print(f"serving on :{args.port} (mode={args.mode})", flush=True)
    web.run_app(app, port=args.port, print=None)


if __name__ == "__main__":
    main()
