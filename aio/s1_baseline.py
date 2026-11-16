"""Stage 1: the two things every optimization needs BEFORE you optimize anything.

  1. An accuracy ORACLE: the official Hugging Face transformers implementation,
     greedy-decoding a fixed set of prompts. Any engine we build must produce
     (nearly) the same text, otherwise "faster" is meaningless.
  2. A BASELINE: stock vLLM serving the same model, measured with our load generator.

Run:
  modal run -m aio.s1_baseline --what reference          # oracle, once
  modal run -m aio.s1_baseline --what vllm --config default
  AIO_GPU=H100 modal run -m aio.s1_baseline --what vllm --config default
"""
import json
import pathlib
import subprocess
import time

import modal

from aio.common import GPU, HF_HOME, MODEL_ID, VLLM_VERSION, gpu_image, hf_cache

MODEL_SHA = "4adf15748de9e1a0097b6274cee9c15836c41c37"  # pinned from stage 0

app = modal.App("aio-s1-baseline")

# Short questions + one LONG prompt (> 512 tokens) so the sliding-window
# attention layers are actually exercised by the correctness check.
_long = " ".join(f"Fact {i}: the box number {i} contains {i * 7 % 13} red balls." for i in range(120))
CHECK_PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Write a Python function that checks whether a number is prime.",
    "Explain what a GPU is to a 10 year old.",
    "List three differences between TCP and UDP.",
    "Translate to Spanish: The weather is nice today and I want to go for a walk.",
    "What is 17 * 23? Show your work.",
    "Write a haiku about inference engines.",
    _long + "\n\nHow many red balls are in box number 5? And in box number 100?",
]
CHECK_MAX_TOKENS = 64

# vLLM launch configurations we compare. Each is the extra CLI flags.
COMMON = ["--max-model-len", "4096", "--gpu-memory-utilization", "0.90",
          "--limit-mm-per-prompt", '{"image": 0, "audio": 0}']
CONFIGS = {
    "default": COMMON,
}


@app.function(image=gpu_image, gpu=GPU, volumes={HF_HOME: hf_cache}, timeout=1800)
def reference() -> dict:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_SHA)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_SHA, dtype=torch.bfloat16).to("cuda")
    hf_cache.commit()
    out = []
    for p in CHECK_PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                      tokenize=True, return_dict=True)["input_ids"]
        x = torch.tensor([ids], device="cuda")
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=CHECK_MAX_TOKENS, do_sample=False)
        new = y[0, len(ids):].tolist()
        out.append({"prompt": p[:80], "prompt_ids": ids, "output_ids": new,
                    "output_text": tok.decode(new, skip_special_tokens=True)})
        print(f"[{len(ids):4d} tok prompt] -> {out[-1]['output_text'][:100]!r}")
    return {"model": MODEL_ID, "sha": MODEL_SHA, "transformers": transformers.__version__,
            "torch": torch.__version__, "gpu": GPU, "cases": out}


def _wait_ready(url: str, proc: subprocess.Popen, timeout_s: int = 1200):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"{url}/health", timeout=2)
            return time.time() - t0
        except Exception:
            time.sleep(2)
    raise TimeoutError("server did not become ready")


def check_correctness(url: str, ref: dict) -> dict:
    """Greedy-decode the oracle prompts on the engine and compare text with the oracle."""
    import urllib.request
    rows = []
    for case in ref["cases"]:
        body = json.dumps({"model": MODEL_ID, "prompt": case["prompt_ids"], "max_tokens": CHECK_MAX_TOKENS,
                           "temperature": 0.0}).encode()
        req = urllib.request.Request(f"{url}/v1/completions", body, {"Content-Type": "application/json"})
        got = json.loads(urllib.request.urlopen(req, timeout=300).read())["choices"][0]["text"]
        want = case["output_text"]
        prefix = 0
        while prefix < min(len(got), len(want)) and got[prefix] == want[prefix]:
            prefix += 1
        rows.append({"exact": got == want, "common_prefix_frac": prefix / max(len(want), 1),
                     "got": got[:120], "want": want[:120]})
    n_exact = sum(r["exact"] for r in rows)
    avg_prefix = sum(r["common_prefix_frac"] for r in rows) / len(rows)
    print(f"  correctness: {n_exact}/{len(rows)} exact matches, avg common prefix {avg_prefix:.0%}")
    for r in rows:
        if not r["exact"]:
            print(f"    DIFF after {r['common_prefix_frac']:.0%}:\n      got : {r['got']!r}\n      want: {r['want']!r}")
    return {"exact": n_exact, "total": len(rows), "avg_common_prefix": avg_prefix, "rows": rows}


@app.function(image=gpu_image, gpu=GPU, volumes={HF_HOME: hf_cache}, timeout=3600)
def run_vllm(config: str, ref: dict) -> dict:
    import asyncio

    from aio import loadgen

    args = ["vllm", "serve", MODEL_ID, "--revision", MODEL_SHA, "--port", "8000", *CONFIGS[config]]
    print("launching:", " ".join(args), flush=True)
    log = open("/tmp/server.log", "w")
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    url = "http://127.0.0.1:8000"
    try:
        startup = _wait_ready(url, proc)
        hf_cache.commit()
        print(f"  server ready in {startup:.0f}s", flush=True)
        corr = check_correctness(url, ref)
        bench = asyncio.run(loadgen.run_all(url, MODEL_ID))
    except Exception:
        print(open("/tmp/server.log").read()[-6000:])
        raise
    finally:
        proc.terminate()
        proc.wait(timeout=60)
    tail = [l for l in open("/tmp/server.log").read().splitlines()
            if any(k in l for k in ("KV cache", "CUDA graph", "Maximum concurrency", "attention backend",
                                    "Using", "torch.compile", "Model loading"))]
    return {"engine": f"vllm=={VLLM_VERSION}", "config": config, "args": args, "gpu": GPU,
            "startup_s": startup, "correctness": corr, "bench": bench, "server_log_highlights": tail[-25:]}


@app.local_entrypoint()
def main(what: str = "vllm", config: str = "default"):
    pathlib.Path("results").mkdir(exist_ok=True)
    ref_path = pathlib.Path("results/s1_reference.json")
    if what == "reference":
        ref = reference.remote()
        ref_path.write_text(json.dumps(ref, indent=1))
        print(f"wrote {ref_path}")
        return
    ref = json.loads(ref_path.read_text())
    res = run_vllm.remote(config, ref)
    out = pathlib.Path(f"results/s1_vllm_{config}_{GPU}.json")
    out.write_text(json.dumps(res, indent=1))
    print("\nserver log highlights:\n  " + "\n  ".join(res["server_log_highlights"]))
    print(f"wrote {out}")
