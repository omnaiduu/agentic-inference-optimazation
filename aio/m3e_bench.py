"""Milestone 3e: int4 GEMM at M = 32 users - can we get closer to the 864 GB/s memory floor?

Variants, timed per weight shape of Gemma 4 E2B with CUDA-graph replay (random weights):
  cur    : our current Triton kernel (int -> fp32 convert, subtract 8, scale, cast to bf16)
  magic  : same kernel, but the 4-bit value is OR-ed into the bit pattern of bf16 128.0 (0x4300),
           which IS the number 128+q; subtract 136 -> q-8 exactly. No int->float convert
           instruction (the trick behind FasterTransformer / Marlin, Kim et al. 2022, arXiv 2211.10017)
  magic32: magic bits, then the subtract+scale as one fp32 FMA
  marlin : vLLM's Marlin CUDA kernel (IST-DASLab, arXiv 2408.11743) - the target to beat
  bf16   : cuBLAS on the dense bf16 copy (4x the bytes)
Also checks every variant against the bf16 reference.

Run:  modal run -m aio.m3e_bench
"""
import json
import pathlib

import modal

from aio.common import GPU, VOLUMES, commit_volumes, gpu_image
from aio.m3c_bench import SHAPES, timeit

app = modal.App("aio-m3e-bench")


@app.function(image=gpu_image, gpu=GPU, volumes=VOLUMES, timeout=3600)
def bench(M: int = 32, quick: bool = False) -> dict:
    import inspect

    import torch
    import torch.nn.functional as F

    from aio.engine import quant
    from aio.engine.quant import QLinear, quantize_rtn

    dev, bf = "cuda", torch.bfloat16
    out = {"M": M, "shapes": {}}

    # ---------- Marlin through vLLM (best effort: the API moves between versions) ----------
    marlin = None
    try:
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
        from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
        from vllm.scalar_type import scalar_types
        print("marlin_quantize", inspect.signature(marlin_quantize), flush=True)
        print("apply_gptq_marlin_linear", inspect.signature(mu.apply_gptq_marlin_linear), flush=True)

        def make_marlin(w):                     # w: bf16 [N, K]
            N, K = w.shape
            r = marlin_quantize(w.t().contiguous(), scalar_types.uint4b8, 32)
            w_ref, q_w, s = r[0], r[1], r[2]
            ws = mu.marlin_make_workspace_new(w.device)
            zp = torch.empty(0, dtype=torch.int32, device=w.device)

            def run(x):
                return mu.apply_gptq_marlin_linear(
                    input=x, weight=q_w, weight_scale=s, weight_zp=zp, workspace=ws,
                    wtype=scalar_types.uint4b8, output_size_per_partition=N, input_size_per_partition=K)
            return run, w_ref.t()
        marlin = make_marlin
    except Exception as e:                      # noqa: BLE001
        print(f"marlin unavailable: {type(e).__name__}: {e}", flush=True)

    import itertools
    import math

    def rot(fn, objs):
        """Call fn on a different weight copy each time: 20 calls x a few copies > the 96 MB L2,
        so every call streams its weights from memory, as in a real decode step."""
        it = itertools.cycle(objs)
        return lambda: fn(next(it))

    variants = {"cur": 0, "magic": 1} if quick else {"cur": 0, "magic": 1, "magic32": 2, "pipe": 1}
    import triton
    pipe_kernel = triton.autotune(       # same kernel, loop pipelined with tl.range(num_stages=LS)
        prune_configs_by={"early_config_prune": quant._prune},
        configs=[triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw)
                 for bn in (32, 64) for bk in (64, 128, 256) for nw in (4, 8)],
        key=["M", "N", "K", "KS", "BITS", "DQ"],
    )(quant._gemm_kernel.fn)
    plain_kernel = quant._gemm_kernel
    tot = {k: 0.0 for k in [*variants, "marlin", "bf16"]}
    tot["best_split"] = 0.0
    tot_bytes = 0.0
    for name, (N, K, n_layers) in SHAPES.items():
        w = torch.randn(N, K, device=dev, dtype=bf) * 0.02
        ql = QLinear(*quantize_rtn(w, 4), 4, K)
        x = torch.randn(M, K, device=dev, dtype=bf)
        n_copies = min(20, math.ceil(300e6 / (N * K / 2)))
        copies = [ql] + [QLinear(*quantize_rtn(w + 1e-3 * i, 4), 4, K) for i in range(1, n_copies)]
        ref = F.linear(x.float(), ql.dense.float())
        nbytes = N * K / 2 + N * K / 32 * 2
        tot_bytes += nbytes * n_layers
        row = {"MB": nbytes / 1e6}
        for vname, dq in variants.items():
            quant.DQ = dq
            quant._gemm_kernel = pipe_kernel if vname == "pipe" else plain_kernel
            quant.LS = 3 if vname == "pipe" else 0
            f = quant.gemv if M == 1 else quant.gemm_small_m      # M = 1 runs our GEMV kernel in production
            y = f(x, ql).float()
            err = ((y - ref).abs().max() / ref.abs().max()).item()
            t = timeit(rot(lambda q: f(x, q), copies))
            row[vname] = {"us": t, "GBs": nbytes / t / 1e3, "rel_err": err}
        quant._gemm_kernel = plain_kernel
        quant.LS = 0
        if "pipe" in row:
            row["pipe"]["cfg"] = str(pipe_kernel.best_config)
        quant.DQ = 1
        row["split"] = {}
        for sp in ((1, 2, 4, 8, 16) if not quick and M > 1 else ()):
            if K % (sp * 128) or (K // sp) < 128:
                continue
            row["split"][sp] = timeit(rot(lambda q: quant.gemm_small_m(x, q, split=sp), copies))
        print(f"{name:15s} magic by split: " + "  ".join(f"{k}: {v:6.1f}us" for k, v in row["split"].items()),
              flush=True)
        if row["split"]:
            tot["best_split"] += min(row["split"].values()) * n_layers
        quant.DQ = 0
        row["bf16"] = {"us": timeit(rot(lambda q: F.linear(x, q.dense), copies))}
        if marlin is not None:
            try:
                runs = [marlin(w + 1e-3 * i) for i in range(n_copies)]
                run, w_ref = runs[0]
                ym = run(x).float()
                refm = F.linear(x.float(), w_ref.float())
                err = ((ym - refm).abs().max() / refm.abs().max()).item()
                t = timeit(rot(lambda r: r[0](x), runs))
                row["marlin"] = {"us": t, "GBs": nbytes / t / 1e3, "rel_err": err}
                del run, w_ref, runs
            except Exception as e:              # noqa: BLE001
                print(f"marlin failed on {name}: {type(e).__name__}: {e}", flush=True)
        for k in tot:
            if k in row and "us" in row[k]:
                tot[k] += row[k]["us"] * n_layers
        out["shapes"][name] = row
        print(f"{name:15s} {nbytes / 1e6:6.1f} MB  " + "  ".join(
            f"{k} {row[k]['us']:6.1f}us" + (f" ({row[k]['GBs']:3.0f} GB/s, err {row[k]['rel_err']:.3f})"
                                             if "GBs" in row[k] else "")
            for k in [*variants, "marlin", "bf16"] if k in row), flush=True)
        del w, ql, copies
        torch.cuda.empty_cache()
    out["total_ms"] = {k: v / 1000 for k, v in tot.items()}
    out["floor_ms"] = tot_bytes / 864e3 / 1000
    print("===== per decode step at M=%d: " % M + ", ".join(f"{k} {v / 1000:.2f} ms" for k, v in tot.items())
          + f", floor at 864 GB/s {out['floor_ms']:.2f} ms", flush=True)
    commit_volumes()
    return out


@app.local_entrypoint()
def main(m: int = 32, quick: bool = False):
    res = bench.remote(m, quick)
    pathlib.Path(f"results/m3e_bench_{GPU}_M{m}.json").write_text(json.dumps(res, indent=1))
