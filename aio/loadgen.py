"""Tiny load generator for any OpenAI-compatible /v1/completions server.

We write our own (instead of using vllm bench / AIPerf) so that:
  1. you can read every line and see exactly what TTFT / TPOT mean, and
  2. the SAME client measures vLLM and our own engine later -> fair comparison.

Metrics (per request):
  TTFT  time to first token  = time from sending the request to the first streamed token
  TPOT  time per output token = (end-to-end time - TTFT) / (output_tokens - 1)
  decode tok/s for one user  = 1 / TPOT
Aggregate:
  throughput = total output tokens / wall time of the whole run
"""
import asyncio
import json
import random
import statistics
import time

import aiohttp

WORDS = ("agent tool call result plan step search file code function test error retry "
         "memory context token model server latency request response cache batch kernel "
         "graph layer weight tensor memory bandwidth compute schedule queue stream").split()
TOPICS = ["a robot learning to cook", "how GPUs work", "a city on Mars", "the history of tea",
          "debugging a race condition", "a detective story", "how vaccines work", "ocean currents",
          "a startup's first year", "the rules of chess", "building a treehouse", "black holes"]


def make_prompts(n: int, input_words: int, seed: int = 0) -> list[str]:
    """Deterministic prompts: a long 'context' blob (like an agent's tool output) + an instruction."""
    rng = random.Random(seed)
    prompts = []
    for i in range(n):
        ctx = " ".join(rng.choice(WORDS) for _ in range(input_words))
        user = (f"Context log:\n{ctx}\n\nIgnore the log above. Write a long, detailed essay about "
                f"{TOPICS[i % len(TOPICS)]}.")
        # Gemma 4 chat format (ai.google.dev/gemma/docs/capabilities/text/basic). Without it the
        # instruction-tuned model answers with invisible special tokens only.
        prompts.append(f"<|turn>user\n{user}<turn|>\n<|turn>model\n")
    return prompts


async def one_request(session, url, model, prompt, max_tokens) -> dict:
    body = {
        "model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
        "stream": True, "stream_options": {"include_usage": True},
        "ignore_eos": True,  # force exactly max_tokens so every engine does the same amount of work
    }
    t0 = time.perf_counter()
    ttft = None
    out_tokens = 0
    async with session.post(f"{url}/v1/completions", json=body) as r:
        r.raise_for_status()
        async for raw in r.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            msg = json.loads(line[5:])
            if msg.get("choices") and ttft is None:  # first token arrived (even if it is invisible)
                ttft = time.perf_counter() - t0
            if msg.get("usage"):
                out_tokens = msg["usage"]["completion_tokens"]
    e2e = time.perf_counter() - t0
    tpot = (e2e - ttft) / max(out_tokens - 1, 1)
    return {"ttft": ttft, "tpot": tpot, "e2e": e2e, "out_tokens": out_tokens}


async def run_level(url, model, concurrency, n_requests, input_words, max_tokens) -> dict:
    prompts = make_prompts(n_requests, input_words, seed=concurrency)
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def go(p):
            async with sem:
                return await one_request(session, url, model, p, max_tokens)
        t0 = time.perf_counter()
        rs = await asyncio.gather(*[go(p) for p in prompts])
        wall = time.perf_counter() - t0

    def pct(xs, q):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(q * len(xs)))]
    ttfts = [r["ttft"] for r in rs]
    tpots = [r["tpot"] for r in rs]
    total_out = sum(r["out_tokens"] for r in rs)
    return {
        "concurrency": concurrency, "requests": n_requests,
        "ttft_ms_p50": 1000 * statistics.median(ttfts), "ttft_ms_p90": 1000 * pct(ttfts, 0.9),
        "tpot_ms_p50": 1000 * statistics.median(tpots), "tpot_ms_p90": 1000 * pct(tpots, 0.9),
        "per_user_tok_s": 1 / statistics.median(tpots),
        "throughput_tok_s": total_out / wall,
        "wall_s": wall,
    }


async def run_all(url, model, levels=(1, 8, 32), input_words=700, max_tokens=256) -> list[dict]:
    # warmup (first requests trigger compilation / allocation; we never time them)
    await run_level(url, model, 2, 2, input_words, 16)
    out = []
    for c in levels:
        n = max(4, 3 * c)
        res = await run_level(url, model, c, n, input_words, max_tokens)
        print(f"  conc={c:3d}  TTFT p50 {res['ttft_ms_p50']:7.1f} ms | TPOT p50 {res['tpot_ms_p50']:6.2f} ms "
              f"({res['per_user_tok_s']:6.1f} tok/s/user) | throughput {res['throughput_tok_s']:7.1f} tok/s",
              flush=True)
        out.append(res)
    return out
