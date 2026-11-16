"""Milestone 1: our own engine, correct (E1) and fast for one user (E2).

One GPU run tests both modes back to back:
  eager : plain PyTorch, the CPU launches every GPU kernel one by one
  graph : the whole decode step is recorded once as a CUDA graph and replayed

Run:  modal run -m aio.m1_engine
"""
import asyncio
import json
import pathlib
import subprocess

import modal

from aio.common import GPU, VOLUMES, commit_volumes, HF_HOME, MODEL_ID, gpu_image, hf_cache
from aio.s1_baseline import MODEL_SHA

app = modal.App("aio-m1-engine")


@app.function(image=gpu_image, gpu=GPU, volumes=VOLUMES, timeout=3600)
def run_engine(ref: dict, modes: list[str]) -> dict:
    from huggingface_hub import snapshot_download

    from aio import loadgen
    from aio.harness import check_correctness, wait_ready

    model_dir = snapshot_download(MODEL_ID, revision=MODEL_SHA,
                                  allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.jinja"])
    results = {}
    for mode in modes:
        print(f"\n=== our engine, mode={mode} ===", flush=True)
        log = open(f"/tmp/engine_{mode}.log", "w")
        proc = subprocess.Popen(["python", "-m", "aio.engine.server", "--model-dir", model_dir,
                                 "--mode", mode, "--port", "8001"], stdout=log, stderr=subprocess.STDOUT)
        url = "http://127.0.0.1:8001"
        try:
            startup = wait_ready(url, proc)
            print(f"  ready in {startup:.0f}s", flush=True)
            corr = check_correctness(url, ref, MODEL_ID)
            bench = asyncio.run(loadgen.run_all(url, MODEL_ID, levels=(1,)))
            results[mode] = {"startup_s": startup, "correctness": corr, "bench": bench}
        except Exception as e:
            print(open(f"/tmp/engine_{mode}.log").read()[-6000:])
            results[mode] = {"error": repr(e)}
        finally:
            proc.terminate()
            proc.wait(timeout=60)
        print(open(f"/tmp/engine_{mode}.log").read()[-1500:], flush=True)
    commit_volumes()
    return {"engine": "aio-gemma4-m1", "gpu": GPU, "results": results}


@app.local_entrypoint()
def main(modes: str = "graph,compile"):
    ref = json.loads(pathlib.Path("results/s1_reference.json").read_text())
    res = run_engine.remote(ref, modes.split(","))
    out = pathlib.Path(f"results/m1_engine_{GPU}.json")
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")
