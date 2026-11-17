"""Milestone 3c: kernel microbenchmarks with random data (no model load, fast and cheap).

  (a) decode attention: our Triton flash-decode kernel vs the current einsum path (correctness + time)
  (b) int4 GEMM at M = 32 users: current autotune space vs a wider one (GB/s of weight reads)
  (c) 512-row bf16 GEMMs (the mixed step): cuBLAS vs Inductor max-autotune (TFLOPS)

Run:  modal run -m aio.m3c_bench
"""
import json
import pathlib

import modal

from aio.common import GPU, VOLUMES, commit_volumes, gpu_image

app = modal.App("aio-m3c-bench")

# (N, K) of every Linear in Gemma 4 E2B, and how many layers use it
SHAPES = {
    "qkv_slide": (2560, 1536, 12), "qkv_full": (5120, 1536, 3), "q_slide_shared": (2048, 1536, 16),
    "q_full_shared": (4096, 1536, 4), "o_slide": (1536, 2048, 28), "o_full": (1536, 4096, 7),
    "gate_up": (12288, 1536, 15), "gate_up_wide": (24576, 1536, 20), "down": (1536, 6144, 15),
    "down_wide": (1536, 12288, 20), "ple_gate": (256, 1536, 35), "ple_proj": (1536, 256, 35),
    "lm_head": (262144, 1536, 1),
}


def timeit(fn, reps=20):
    """GPU time per call, measured by replaying a CUDA graph of `reps` calls. (Timing eager calls
    one by one measures the CPU launch overhead instead: ~95 us flat in our first attempt.)"""
    import torch
    for _ in range(3):
        fn()                                   # warm-up: autotuning / compilation happen here
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(5):
        g.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / (5 * reps) * 1000  # us


@app.function(image=gpu_image, gpu=GPU, volumes=VOLUMES, timeout=3600)
def bench() -> dict:
    import torch
    import torch.nn.functional as F
    import triton

    from aio.engine import quant
    from aio.engine.attn import decode_attention
    from aio.engine.quant import QLinear, gemm_small_m, quantize_rtn

    torch.manual_seed(0)
    dev, bf = "cuda", torch.bfloat16
    out = {"attn": {}, "gemm32": {}, "gemm512": {}}

    # ---------------- (a) decode attention ----------------
    def ref_attn(q, K, V, pos, rp, S):
        B = q.shape[0]
        if rp is None:
            Kc, Vc = K[:B, :S], V[:B, :S]
            mask = torch.arange(S, device=dev)[None, :] <= pos[:, None]
        else:
            Kc, Vc = K[:B], V[:B]
            mask = (rp[:B] <= pos[:, None]) & (rp[:B] > pos[:, None] - 512)
        sc = torch.einsum("bhd,bsd->bhs", q, Kc).float().masked_fill(~mask[:, None], float("-inf"))
        return torch.einsum("bhs,bsd->bhd", torch.softmax(sc, -1).to(bf), Vc).reshape(B, -1)

    ref_c = torch.compile(ref_attn, dynamic=False)
    for B in (1, 8, 32):
        pos = torch.randint(1000, 1400, (B,), device=dev)
        for kind, hd, L in (("full", 512, 4096), ("ring", 256, 512)):
            K = torch.randn(33, L, hd, device=dev, dtype=bf)
            V = torch.randn(33, L, hd, device=dev, dtype=bf)
            q = torch.randn(B, 8, hd, device=dev, dtype=bf) * 0.05
            rp = None
            if kind == "ring":   # ring slot s holds the latest position p' <= pos with p' % 512 == s
                s = torch.arange(512, device=dev)[None, :]
                rp = torch.full((33, 512), -10**9, dtype=torch.long, device=dev)
                rp[:B] = pos[:, None] - ((pos[:, None] - s) % 512)
            ours = decode_attention(q, K, V, pos, rp)
            ref = ref_attn(q, K, V, pos, rp, 2048)
            err = (ours.float() - ref.float()).abs().max().item()
            t_ours = timeit(lambda: decode_attention(q, K, V, pos, rp))
            t_ref = timeit(lambda: ref_c(q, K, V, pos, rp, 2048))
            out["attn"][f"{kind}_B{B}"] = {"max_err": err, "ours_us": t_ours, "einsum_compiled_us": t_ref}
            print(f"attn {kind:4s} B={B:2d}: max|diff|={err:.4f}  ours {t_ours:7.1f} us  "
                  f"einsum(compiled, S=2048) {t_ref:7.1f} us", flush=True)
    a = out["attn"]
    print(f"===== attention per decode step at B=32 (7 full + 28 ring layers): ours "
          f"{(7 * a['full_B32']['ours_us'] + 28 * a['ring_B32']['ours_us']) / 1000:.2f} ms, einsum "
          f"{(7 * a['full_B32']['einsum_compiled_us'] + 28 * a['ring_B32']['einsum_compiled_us']) / 1000:.2f} ms",
          flush=True)

    # ---------------- (b) int4 GEMM at M=32 ----------------
    wide = triton.autotune(
        prune_configs_by={"early_config_prune": quant._prune},
        configs=[triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw, num_stages=ns)
                 for bn in (32, 64, 128) for bk in (64, 128, 256) for nw in (4, 8) for ns in (2, 3, 4)],
        key=["M", "N", "K", "KS", "BITS"],
    )(quant._gemm_kernel.fn)
    tot_cur = tot_wide = tot_bytes = 0.0
    for name, (N, K, n_layers) in SHAPES.items():
        w = torch.randn(N, K, device=dev, dtype=bf) * 0.02
        ql = QLinear(*quantize_rtn(w, 4), 4, K)
        del w
        x = torch.randn(32, K, device=dev, dtype=bf)
        t_cur = timeit(lambda: gemm_small_m(x, ql))
        orig = quant._gemm_kernel
        quant._gemm_kernel = wide
        try:
            t_wide = timeit(lambda: gemm_small_m(x, ql))
            cfg = str(wide.best_config)
        finally:
            quant._gemm_kernel = orig
        nbytes = N * K / 2 + N * K / 32 * 2
        tot_cur += t_cur * n_layers
        tot_wide += min(t_cur, t_wide) * n_layers
        tot_bytes += nbytes * n_layers
        out["gemm32"][name] = {"cur_us": t_cur, "wide_us": t_wide, "best": cfg, "MB": nbytes / 1e6}
        print(f"int4 M=32 {name:15s} N={N:6d} K={K:5d}: current {t_cur:7.1f} us ({nbytes / t_cur / 1e3:4.0f} GB/s)"
              f"  wide {t_wide:7.1f} us ({nbytes / t_wide / 1e3:4.0f} GB/s)  {cfg}", flush=True)
        del ql
    print(f"===== int4 GEMMs per decode step at B=32: current {tot_cur / 1000:.2f} ms, best-of-wide "
          f"{tot_wide / 1000:.2f} ms, ideal at 864 GB/s {tot_bytes / 864e3 / 1000:.2f} ms", flush=True)
    commit_volumes()

    # ---------------- (c) 512-row bf16 GEMMs ----------------
    mm_auto = torch.compile(lambda x, w: F.linear(x, w), mode="max-autotune-no-cudagraphs", dynamic=False)
    tot_cub = tot_auto = flops_tot = 0.0
    for name, (N, K, n_layers) in SHAPES.items():
        if name == "lm_head":
            continue
        w = torch.randn(N, K, device=dev, dtype=bf) * 0.02
        x = torch.randn(512, K, device=dev, dtype=bf)
        t_cub = timeit(lambda: F.linear(x, w))
        t_auto = timeit(lambda: mm_auto(x, w))
        fl = 2 * 512 * N * K
        tot_cub += t_cub * n_layers
        tot_auto += min(t_cub, t_auto) * n_layers
        flops_tot += fl * n_layers
        out["gemm512"][name] = {"cublas_us": t_cub, "autotune_us": t_auto}
        print(f"bf16 M=512 {name:15s}: cuBLAS {t_cub:7.1f} us ({fl / t_cub / 1e6:4.0f} TFLOPS)  "
              f"max-autotune {t_auto:7.1f} us ({fl / t_auto / 1e6:4.0f} TFLOPS)", flush=True)
    print(f"===== 512-row GEMMs per mixed step: cuBLAS {tot_cub / 1000:.2f} ms, best-of {tot_auto / 1000:.2f} ms, "
          f"ideal at 362 TFLOPS {flops_tot / 362e6 / 1000:.2f} ms", flush=True)
    commit_volumes()
    return out


@app.local_entrypoint()
def main():
    res = bench.remote()
    pathlib.Path(f"results/m3c_bench_{GPU}.json").write_text(json.dumps(res, indent=1))
