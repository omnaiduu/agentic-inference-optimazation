"""Milestone 2, step 0: unit-test the low-bit format and our Triton GEMV kernel.

  1. format : unpacking Google's QAT int4 weights must give ~ the original bf16 weights
  2. kernel : our GEMV must match a plain PyTorch matmul on the dequantized weights
  3. speed  : time per call and achieved memory bandwidth (L40S peak = 864 GB/s)

Run:  modal run -m aio.m2_kernel_test
"""
import json
import pathlib

import modal

from aio.common import GPU, MODEL_ID, VOLUMES, commit_volumes, gpu_image
from aio.s1_baseline import MODEL_SHA

QAT_ID = "google/gemma-4-E2B-it-qat-w4a16-ct"
QAT_SHA = "971342c08f607aa7779983f6b5289778b5d271a7"
app = modal.App("aio-m2-kernel-test")


@app.function(image=gpu_image, gpu=GPU, volumes=VOLUMES, timeout=1800)
def test() -> dict:
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    from aio.engine.quant import QLinear, gemv, quantize_rtn

    out = {}
    bf_dir = snapshot_download(MODEL_ID, revision=MODEL_SHA, allow_patterns=["*.json", "*.safetensors"])
    q_dir = snapshot_download(QAT_ID, revision=QAT_SHA, allow_patterns=["*.json", "*.safetensors"])
    commit_volumes()

    def get(d, name):
        import glob
        for f in glob.glob(f"{d}/*.safetensors"):
            with safe_open(f, "pt", device="cuda") as sf:
                if name in sf.keys():
                    return sf.get_tensor(name)
        raise KeyError(name)

    # ---- 1. format check -------------------------------------------------
    p = "model.language_model.layers.3.mlp.gate_proj."
    w_bf = get(bf_dir, p + "weight").float()
    ql = QLinear(get(q_dir, p + "weight_packed"), get(q_dir, p + "weight_scale"), 4, w_bf.shape[1])
    w_q = ql.dense.float()
    rel = ((w_q - w_bf).norm() / w_bf.norm()).item()
    cos = F.cosine_similarity(w_q.flatten(), w_bf.flatten(), dim=0).item()
    # same packed data read in the WRONG (interleaved) order, as a sanity contrast
    from aio.engine.quant import unpack
    q_int = unpack(ql.packed, 4, ql.K)
    wrong = q_int.view(ql.N, 8, -1).transpose(1, 2).reshape(ql.N, -1).float()
    wrong = (wrong.view(ql.N, -1, 32) * ql.scale.float()[:, :, None]).view(ql.N, -1)
    cos_wrong = F.cosine_similarity(wrong.flatten(), w_bf.flatten(), dim=0).item()
    out["format"] = {"rel_err_vs_bf16": rel, "cosine_vs_bf16": cos, "cosine_if_wrong_order": cos_wrong}
    print(f"[format] QAT int4 vs bf16 weights: cosine {cos:.4f}, rel err {rel:.3f} "
          f"(wrong unpack order would give cosine {cos_wrong:.4f})", flush=True)

    # ---- 2+3. kernel correctness and speed ---------------------------------
    def bench(fn, iters=200):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(iters):
                fn()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record(); g.replay(); en.record(); torch.cuda.synchronize()
        return st.elapsed_time(en) / iters * 1000  # microseconds

    lm = get(bf_dir, "model.language_model.embed_tokens.weight")  # tied lm_head [262144, 1536]
    gate = get(bf_dir, p + "weight")
    cases = {"lm_head [262144x1536]": lm, "mlp gate [6144x1536]": gate,
             "mlp down [1536x6144]": get(bf_dir, "model.language_model.layers.3.mlp.down_proj.weight")}
    rows = []
    for name, w in cases.items():
        x = torch.randn(1, w.shape[1], device="cuda", dtype=torch.bfloat16)
        t_bf = bench(lambda: F.linear(x, w))
        rows.append({"layer": name, "format": "bf16 (cuBLAS)", "us": t_bf,
                     "GB/s": w.numel() * 2 / t_bf / 1e3, "max_abs_err": 0.0})
        for bits in (8, 4):
            packed, scale = quantize_rtn(w, bits)
            ql = QLinear(packed, scale, bits, w.shape[1])
            y_ref = F.linear(x, ql.dense).float()
            y = gemv(x, ql).float()
            err = ((y - y_ref).abs().max() / y_ref.abs().max()).item()
            t = bench(lambda: gemv(x, ql))
            rows.append({"layer": name, "format": f"int{bits} (ours)", "us": t,
                         "GB/s": ql.nbytes_decode() / t / 1e3, "max_abs_err": err})
    for r in rows:
        print(f"[kernel] {r['layer']:24s} {r['format']:14s} {r['us']:8.1f} us  "
              f"{r['GB/s']:6.0f} GB/s  rel.err {r['max_abs_err']:.2e}", flush=True)
    out["kernel"] = rows
    commit_volumes()
    return out


@app.local_entrypoint()
def main():
    res = test.remote()
    pathlib.Path(f"results/m2_kernel_test_{GPU}.json").write_text(json.dumps(res, indent=1))
