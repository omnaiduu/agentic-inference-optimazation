"""Milestone 3, step 0: measure before optimizing. Everything in ONE container (fair A/B).

For each engine config:
  1. engine-only decode time: replay the CUDA graph N times, time with CUDA events.
     (No HTTP server, no tokenizer. If this is much faster than loadgen's TPOT,
      the server is the bottleneck, not the GPU.)
  2. kernel breakdown: torch.profiler over a few compiled decode steps -> which kernels
     eat the time (our GEMV, cuBLAS, attention, elementwise, argmax...).

Run:  modal run -m aio.m3_profile --configs bf16,qat-lm4
"""
import json
import pathlib

import modal

from aio.common import GPU, MODEL_ID, VOLUMES, commit_volumes, gpu_image
from aio.m2_run import QAT_ID, QAT_SHA
from aio.s1_baseline import MODEL_SHA

app = modal.App("aio-m3-profile")


@app.function(image=gpu_image, gpu=GPU, volumes=VOLUMES, timeout=3600)
def profile(configs: list[str], ctx: int = 1200, steps: int = 200) -> dict:
    import gc
    from collections import defaultdict

    import torch
    from huggingface_hub import snapshot_download

    from aio.engine.model import Gemma4E2B

    pats = ["*.json", "*.safetensors", "tokenizer*"]
    bf_dir = snapshot_download(MODEL_ID, revision=MODEL_SHA, allow_patterns=pats)
    q_dir = snapshot_download(QAT_ID, revision=QAT_SHA, allow_patterns=pats)
    out = {}
    for cfg in configs:
        kw = {}
        if cfg.startswith("qat"):
            kw["quant_dir"] = q_dir
        if cfg.endswith("lm4"):
            kw["lm_head_bits"] = 4
        if cfg.endswith("lm8"):
            kw["lm_head_bits"] = 8
        m = Gemma4E2B(bf_dir, **kw)
        m.enable_compile()
        m.capture_graphs()
        tok = m.prefill(list(range(1000, 1000 + ctx)))  # fill KV cache to a realistic length

        # 1) engine-only decode step time (graph replay, bucket for this context)
        S = m._bucket(ctx)
        m.static_tok.fill_(tok)
        m.static_pos.fill_(ctx)
        g = m.graphs[S]
        for _ in range(20):
            g.replay()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        st.record()
        for _ in range(steps):
            g.replay()
        en.record()
        torch.cuda.synchronize()
        ms = st.elapsed_time(en) / steps
        # 1b) the same, but with the per-token .item() sync the server does
        import time
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(steps):
            m.decode_graph(tok, ctx)
        ms_sync = (time.perf_counter() - t0) / steps * 1000

        # 2) kernel breakdown of the compiled step (not graphed, so every kernel is visible)
        x_tok, x_pos = m.static_tok.clone(), m.static_pos.clone()
        with torch.inference_mode():
            for _ in range(3):
                m.step_fn(x_tok, x_pos, S)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(10):
                    m.step_fn(x_tok, x_pos, S)
                torch.cuda.synchronize()
        by_kernel = defaultdict(lambda: [0.0, 0])
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                by_kernel[e.name][0] += e.device_time_total / 10  # us per step
                by_kernel[e.name][1] += 1
        total_us = sum(v[0] for v in by_kernel.values())
        top = sorted(by_kernel.items(), key=lambda kv: -kv[1][0])[:25]
        n_kernels = sum(v[1] for v in by_kernel.values()) / 10

        print(f"\n===== {cfg}: engine-only {ms:.3f} ms/token ({1000 / ms:.0f} tok/s) | with .item() sync "
              f"{ms_sync:.3f} ms ({1000 / ms_sync:.0f} tok/s) | ungraphed kernel time {total_us / 1000:.3f} ms, "
              f"{n_kernels:.0f} kernels/step", flush=True)
        for name, (us, cnt) in top:
            print(f"  {us:8.1f} us  {us / total_us:5.1%}  x{cnt // 10:<4d} {name[:110]}", flush=True)
        out[cfg] = {"graph_ms": ms, "sync_ms": ms_sync, "kernel_us_total": total_us, "kernels_per_step": n_kernels,
                    "top": [(n, us, c // 10) for n, (us, c) in top]}
        del m, g, prof
        gc.collect()
        torch.cuda.empty_cache()
        commit_volumes()
    return out


@app.local_entrypoint()
def main(configs: str = "bf16,qat-lm4"):
    res = profile.remote(configs.split(","))
    pathlib.Path(f"results/m3_profile_{GPU}.json").write_text(json.dumps(res, indent=1))
