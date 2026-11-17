"""Milestone 2: low-bit weights. Every config gets the SAME three tests:
token agreement with the bf16 oracle, GSM8K-100 accuracy, and single-user speed.

Configs (each on its own L40S, all in parallel):
  vllm-bf16     stock vLLM, original weights                (reference quality)
  vllm-qat      stock vLLM, Google's QAT int4 checkpoint    (the fair speed rival)
  ours-bf16     our compiled engine, original weights
  ours-qat      our engine, QAT int4 layers (our Triton kernel), lm_head bf16
  ours-qat-lm8  ... + lm_head quantized to int8
  ours-qat-lm4  ... + lm_head quantized to int4

Run:  modal run -m aio.m2_run                       (all)
      modal run -m aio.m2_run --configs ours-qat-lm4
"""
import asyncio
import json
import pathlib
import subprocess

import modal

from aio.common import GPU, MODEL_ID, VOLUMES, commit_volumes, gpu_image
from aio.s1_baseline import COMMON, MODEL_SHA

QAT_ID = "google/gemma-4-E2B-it-qat-w4a16-ct"
QAT_SHA = "971342c08f607aa7779983f6b5289778b5d271a7"
ALL = ["vllm-bf16", "vllm-qat", "ours-bf16", "ours-qat", "ours-qat-lm8", "ours-qat-lm4"]
# ours-batch: QAT int4 + int4 lm_head + continuous batching (aio/engine/server_batch.py)

app = modal.App("aio-m2")
image = gpu_image


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=3600)
def run_config(name: str, ref: dict) -> dict:
    from huggingface_hub import snapshot_download

    from aio import loadgen, quality
    from aio.harness import wait_ready

    pats = ["*.json", "*.safetensors", "tokenizer*", "*.jinja"]
    bf_dir = snapshot_download(MODEL_ID, revision=MODEL_SHA, allow_patterns=pats)
    q_dir = snapshot_download(QAT_ID, revision=QAT_SHA, allow_patterns=pats)
    commit_volumes()

    engine = name.split("-")[0]
    if engine == "vllm":
        model, rev = (QAT_ID, QAT_SHA) if "qat" in name else (MODEL_ID, MODEL_SHA)
        cmd = ["vllm", "serve", model, "--revision", rev, "--port", "8000", *COMMON]
        url, levels = "http://127.0.0.1:8000", (1, 8, 32)
    elif name.startswith("ours-batch"):          # ours-batch, ours-batch-fp8
        model = MODEL_ID
        cmd = ["python", "-m", "aio.engine.server_batch", "--model-dir", bf_dir, "--quant-dir", q_dir,
               "--lm-head-bits", "4", "--port", "8000"]
        if name.endswith("fp8"):
            cmd.append("--fp8-prefill")
        url, levels = "http://127.0.0.1:8000", (1, 8, 32)
    else:
        model = MODEL_ID
        cmd = ["python", "-m", "aio.engine.server", "--model-dir", bf_dir, "--mode", "compile", "--port", "8000"]
        if "qat" in name:
            cmd += ["--quant-dir", q_dir]
        if name.endswith("lm8"):
            cmd += ["--lm-head-bits", "8"]
        if name.endswith("lm4"):
            cmd += ["--lm-head-bits", "4"]
        url, levels = "http://127.0.0.1:8000", (1,)

    print(f"=== {name}: {' '.join(cmd)}", flush=True)
    log = open("/tmp/server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    res = {"config": name, "gpu": GPU, "cmd": cmd}
    try:
        res["startup_s"] = wait_ready(url, proc, timeout_s=2400)
        commit_volumes()
        print(f"  ready in {res['startup_s']:.0f}s", flush=True)
        res["agreement"] = quality.token_agreement(url, ref, model, engine)
        res["gsm8k"] = quality.gsm8k(url, model, bf_dir)
        res["bench"] = asyncio.run(loadgen.run_all(url, model, levels=levels))
    except Exception as e:
        res["error"] = repr(e)
        print(open("/tmp/server.log").read()[-5000:], flush=True)
    finally:
        proc.terminate()
        proc.wait(timeout=60)
        res["server_stats"] = [l.strip() for l in open("/tmp/server.log") if l.startswith("[sched]")][-6:]
        for l in res["server_stats"]:
            print("  " + l, flush=True)
        commit_volumes()
    return res


@app.local_entrypoint()
def main(configs: str = ",".join(ALL), tag: str = ""):
    ref = json.loads(pathlib.Path("results/s1_reference.json").read_text())
    names = configs.split(",")
    rows = []
    for res in run_config.starmap([(n, ref) for n in names], return_exceptions=True):
        if isinstance(res, Exception):
            print("FAILED:", res)
            continue
        pathlib.Path(f"results/m2_{res['config']}_{GPU}{tag}.json").write_text(json.dumps(res, indent=1))
        rows.append(res)
    print(f"\n{'config':14s} {'agree':>7s} {'GSM8K':>6s} {'tok/s(1 user)':>14s} {'TTFT ms':>8s}")
    for r in rows:
        if "error" in r:
            print(f"{r['config']:14s} ERROR {r['error'][:100]}")
            continue
        b = r["bench"][0]
        print(f"{r['config']:14s} {r['agreement']['agreement']:7.1%} {r['gsm8k']['accuracy']:6.0%} "
              f"{b['per_user_tok_s']:14.1f} {b['ttft_ms_p50']:8.1f}")
