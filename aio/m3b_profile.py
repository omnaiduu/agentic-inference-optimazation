"""Milestone 3b, step 0: where does the time go when many users share the GPU?

Measures, in ONE container, with no HTTP server:
  1. prefill: time to process one ~1,100-token prompt, and its kernel breakdown
  2. batched decode: ms per step for B = 1, 8, 32 users (all slots filled to ~1,100 tokens)
  3. kernel breakdown of one ungraphed batched step at B = 32

Ideal for B = 32 on an L40S: ~1.6 ms (int4 weights) + ~1.2 ms (32 users' KV) ~= 3 ms/step.

Run:  modal run -m aio.m3b_profile
"""
import json
import pathlib

import modal

from aio.common import GPU, MODEL_ID, VOLUMES, commit_volumes, gpu_image
from aio.m2_run import QAT_ID, QAT_SHA
from aio.s1_baseline import MODEL_SHA

app = modal.App("aio-m3b-profile")


def kernel_table(prof, reps: int, n: int = 20):
    from collections import defaultdict

    import torch
    by = defaultdict(lambda: [0.0, 0])
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            by[e.name][0] += e.device_time_total / reps
            by[e.name][1] += 1
    total = sum(v[0] for v in by.values())
    top = sorted(by.items(), key=lambda kv: -kv[1][0])[:n]
    for name, (us, cnt) in top:
        print(f"  {us:9.1f} us  {us / total:5.1%}  x{cnt // reps:<4d} {name[:100]}", flush=True)
    return {"total_us": total, "kernels": sum(v[1] for v in by.values()) / reps,
            "top": [(k, us, c // reps) for k, (us, c) in top]}


@app.function(image=gpu_image, gpu=GPU, volumes=VOLUMES, timeout=3600)
def profile(ctx: int = 1100, sizes: tuple = (1, 8, 32)) -> dict:
    import time

    import torch
    from huggingface_hub import snapshot_download

    from aio.engine.batch import BatchedGemma4

    pats = ["*.json", "*.safetensors", "tokenizer*"]
    bf_dir = snapshot_download(MODEL_ID, revision=MODEL_SHA, allow_patterns=pats)
    q_dir = snapshot_download(QAT_ID, revision=QAT_SHA, allow_patterns=pats)
    out = {"gpu": GPU, "ctx": ctx}

    m = BatchedGemma4(bf_dir, quant_dir=q_dir, lm_head_bits=4, max_batch=32)
    m.enable_compile()
    t0 = time.time()
    m.capture_batch_graphs(sizes=sizes, buckets=(2048,))
    out["capture_s"] = time.time() - t0
    print(f"captured {len(m.bgraphs)} graphs in {out['capture_s']:.0f}s", flush=True)
    commit_volumes()

    # 1) prefill
    prompt = [1000 + (i * 7) % 5000 for i in range(ctx)]
    for _ in range(2):
        m.prefill_into(0, prompt)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        m.prefill_into(0, prompt)
    torch.cuda.synchronize()
    out["prefill_ms"] = (time.perf_counter() - t0) / 5 * 1000
    print(f"\n===== prefill {ctx} tokens: {out['prefill_ms']:.1f} ms "
          f"({ctx / out['prefill_ms'] * 1000:.0f} tok/s)", flush=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            m.prefill_into(0, prompt)
        torch.cuda.synchronize()
    out["prefill_kernels"] = kernel_table(prof, 3)

    # 2) batched decode step time (fill every slot with the same prompt)
    for d in range(1, m.max_b):
        m.move_slot(0, d)
    out["decode"] = {}
    for B in sizes:
        toks, poss = [5] * B, [ctx] * B
        m.decode_batch(toks, poss, 32)
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(4):
            m.decode_batch(toks, poss, 32)
        en.record()
        torch.cuda.synchronize()
        ms = st.elapsed_time(en) / (4 * 32)
        out["decode"][B] = ms
        print(f"===== decode B={B:2d}: {ms:.3f} ms/step -> {B / ms * 1000:7.0f} tok/s total", flush=True)

    # 3) kernel breakdown of an ungraphed compiled step at the largest B
    B = max(sizes)
    tok = torch.full((B,), 5, dtype=torch.long, device="cuda")
    pos = torch.full((B,), ctx, dtype=torch.long, device="cuda")
    with torch.inference_mode():
        for _ in range(3):
            m.bstep_fn(tok, pos, 2048)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                m.bstep_fn(tok, pos, 2048)
            torch.cuda.synchronize()
    print(f"\n===== kernel breakdown, decode B={B}", flush=True)
    out["decode_kernels"] = kernel_table(prof, 10)
    commit_volumes()
    return out


@app.local_entrypoint()
def main(prompt_len: int = 1100):
    res = profile.remote(prompt_len)
    pathlib.Path(f"results/m3b_profile_{GPU}.json").write_text(json.dumps(res, indent=1))
