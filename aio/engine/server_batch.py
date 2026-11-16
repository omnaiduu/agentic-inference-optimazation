"""OpenAI-compatible server with CONTINUOUS BATCHING for our engine.

One scheduler thread owns the GPU and loops:
  1. admit waiting requests into free slots (prefill each one, stream its first token),
  2. run ONE batched decode chunk (k tokens) for every active request,
  3. stream the new tokens, retire finished requests and compact the slots.
New requests therefore join between chunks instead of waiting for others to finish.

  python -m aio.engine.server_batch --model-dir DIR --quant-dir QDIR --lm-head-bits 4
"""
import argparse
import asyncio
import json
import threading
import time
from dataclasses import dataclass, field

from aiohttp import web
from transformers import AutoTokenizer

from aio.engine.batch import BatchedGemma4
from aio.engine.model import EOS_IDS


@dataclass
class Req:
    prompt: list[int]
    max_tokens: int
    ignore_eos: bool
    emit: callable                       # thread-safe: emit(token or None)
    done: int = 0
    pos: int = 0
    last: int = 0
    finished: bool = False
    out: list[int] = field(default_factory=list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--quant-dir", default=None)
    ap.add_argument("--lm-head-bits", type=int, default=None)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=4)
    args = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = BatchedGemma4(args.model_dir, max_len=args.max_len, quant_dir=args.quant_dir,
                          lm_head_bits=args.lm_head_bits, max_batch=args.max_batch)
    model.enable_compile()
    model.capture_graphs()                # single-sequence graphs (used by teacher-forced eval)
    model.capture_batch_graphs()
    print(f"ready: weights + {len(model.graphs)} single + {len(model.bgraphs)} batch graphs "
          f"in {time.time() - t0:.0f}s", flush=True)

    engine_lock = threading.Lock()
    cond = threading.Condition()
    waiting: list[Req] = []

    def give(r: Req, t: int):
        r.emit(t)
        r.done += 1
        if r.done >= r.max_tokens or (not r.ignore_eos and t in EOS_IDS):
            r.finished = True

    def scheduler():
        active: list[Req] = []
        while True:
            with cond:
                while not waiting and not active:
                    cond.wait()
                new = []
                while waiting and len(active) + len(new) < args.max_batch:
                    new.append(waiting.pop(0))
            with engine_lock:
                for r in new:                                    # admit (prefill) new requests
                    slot = len(active)
                    first = model.prefill_into(slot, r.prompt)
                    r.pos, r.last = len(r.prompt), first
                    give(r, first)
                    if r.finished:
                        r.emit(None)
                    else:
                        active.append(r)
                if not active:
                    continue
                k = args.chunk
                steps = model.decode_batch([r.last for r in active], [r.pos for r in active], k)
                for b, r in enumerate(active):
                    for j in range(k):
                        if not r.finished:
                            give(r, steps[j][b])
                    r.pos += k
                    r.last = steps[k - 1][b]
                i = 0                                            # retire + compact
                while i < len(active):
                    if active[i].finished:
                        active[i].emit(None)
                        last = len(active) - 1
                        if i != last:
                            model.move_slot(last, i)
                            active[i] = active[last]
                        active.pop()
                    else:
                        i += 1

    threading.Thread(target=scheduler, daemon=True).start()

    async def completions(request: web.Request):
        body = await request.json()
        prompt = body["prompt"]
        prompt_ids = prompt if isinstance(prompt, list) else tok.encode(prompt)
        max_tokens = int(body.get("max_tokens", 16))
        if len(prompt_ids) + max_tokens + args.chunk > args.max_len:
            return web.json_response({"error": "too long"}, status=400)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        r = Req(prompt_ids, max_tokens, bool(body.get("ignore_eos", False)),
                lambda t: loop.call_soon_threadsafe(q.put_nowait, t))
        with cond:
            waiting.append(r)
            cond.notify()

        out_ids: list[int] = []
        if not body.get("stream", False):
            while (t := await q.get()) is not None:
                out_ids.append(t)
            return web.json_response({"choices": [{"index": 0, "text": tok.decode(out_ids, skip_special_tokens=True)}],
                                      "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(out_ids)}})
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        text = ""
        while (t := await q.get()) is not None:
            out_ids.append(t)
            new_text = tok.decode(out_ids, skip_special_tokens=True)
            delta, text = new_text[len(text):], new_text
            await resp.write(f"data: {json.dumps({'choices': [{'index': 0, 'text': delta}]})}\n\n".encode())
        usage = {"choices": [], "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(out_ids)}}
        await resp.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())
        await resp.write_eof()
        return resp

    def teacher_forced(prompt_ids, target_ids):
        """Oracle agreement through the BATCHED decode path (batch of 1, slot 0)."""
        with engine_lock:
            preds = [model.prefill_into(0, prompt_ids)]
            pos = len(prompt_ids)
            for t in target_ids[:-1]:
                preds.append(model.decode_batch([t], [pos], 1)[0][0])
                pos += 1
            return preds

    async def teacher_forced_route(request: web.Request):
        body = await request.json()
        preds = await asyncio.get_running_loop().run_in_executor(
            None, teacher_forced, body["prompt_ids"], body["target_ids"])
        return web.json_response({"preds": preds})

    async def health(_):
        return web.Response(text="ok")

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.add_routes([web.post("/v1/completions", completions), web.get("/health", health),
                    web.post("/aio/teacher_forced", teacher_forced_route)])
    web.run_app(app, port=args.port, print=None)


if __name__ == "__main__":
    main()
