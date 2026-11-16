"""Stage 0: How fast COULD this model possibly run? ("speed of light")

Decoding one token for one user means reading every weight that the token
touches from GPU memory once. Memory bandwidth (GB/s) / bytes read per token
= the theoretical max tokens/sec for a single stream. Real engines get maybe
50-85% of this. It tells us how much room is left to optimize.

Run:  modal run -m aio.s0_speed_of_light
"""
import json
import pathlib
from collections import defaultdict

import modal

from aio.common import MODEL_ID, MODEL_REVISION, cpu_image

app = modal.App("aio-s0-speed-of-light")

# Peak memory bandwidth in GB/s (vendor spec sheets).
GPU_BANDWIDTH_GBS = {
    "T4": 320, "L4": 300, "A10": 600, "L40S": 864,
    "A100-80GB": 2039, "H100": 3350, "H200": 4800, "B200": 8000,
}
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1, "I8": 1, "I32": 4, "I64": 8}


@app.function(image=cpu_image, timeout=600)
def inspect_weights() -> dict:
    from huggingface_hub import HfApi, get_safetensors_metadata, hf_hub_download

    sha = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION).sha
    cfg = json.loads(pathlib.Path(hf_hub_download(MODEL_ID, "config.json", revision=sha)).read_text())
    meta = get_safetensors_metadata(MODEL_ID, revision=sha)
    tensors = {}
    for f in meta.files_metadata.values():
        for name, t in f.tensors.items():
            n = 1
            for d in t.shape:
                n *= d
            tensors[name] = {"dtype": t.dtype, "shape": t.shape, "numel": n}
    return {"sha": sha, "config": cfg, "tensors": tensors}


def group_of(name: str) -> str:
    """Collapse 'model.language_model.layers.12.self_attn.q_proj.weight' -> '...layers.N.self_attn.q_proj'."""
    parts = name.split(".")
    parts = ["N" if p.isdigit() else p for p in parts]
    if parts[-1] in ("weight", "bias"):
        parts = parts[:-1]
    return ".".join(parts)


def speed_of_light(groups: dict) -> dict:
    """Which weights must be streamed from memory for every new token?

    - Every decoder layer weight (attention + MLP + per-layer-input projections): YES
    - lm_head (tied to embed_tokens, [262144 x 1536]): YES, the full matrix is
      multiplied with the hidden state to score all 262K vocab tokens.
    - embed_tokens / embed_tokens_per_layer as INPUT lookups: only 1 row each, ~0 bytes.
    - vision_tower / audio_tower: NO for text-only requests.
    """
    gb = lambda f: sum(v["bytes"] for k, v in groups.items() if f(k)) / 1e9
    lang = lambda k: "language_model" in k
    breakdown = {
        "decoder layers (attn+mlp+PLE proj)": gb(lambda k: lang(k) and ".layers." in k),
        "lm_head (tied embed_tokens, full read)": gb(lambda k: k.endswith("language_model.embed_tokens")),
        "other small (norms, per_layer_model_projection)": gb(
            lambda k: lang(k) and ".layers." not in k and "embed_tokens" not in k),
        "SKIPPED: embed_tokens_per_layer (row lookup)": 0.0,
        "SKIPPED: vision+audio towers (text only)": 0.0,
    }
    total = sum(breakdown.values())
    breakdown["(for reference) PLE table size on disk"] = gb(lambda k: k.endswith("embed_tokens_per_layer"))
    breakdown["(for reference) vision+audio on disk"] = gb(lambda k: "vision" in k or "audio" in k)
    return {
        "bytes_per_token_gb": total,
        "breakdown_gb": breakdown,
        "max_tok_s": {g: bw / total for g, bw in GPU_BANDWIDTH_GBS.items()},
    }


@app.local_entrypoint()
def main():
    info = inspect_weights.remote()
    tensors = info["tensors"]

    groups = defaultdict(lambda: {"count": 0, "params": 0, "bytes": 0, "dtype": set(), "shape": None})
    for name, t in tensors.items():
        g = groups[group_of(name)]
        g["count"] += 1
        g["params"] += t["numel"]
        g["bytes"] += t["numel"] * DTYPE_BYTES.get(t["dtype"], 2)
        g["dtype"].add(t["dtype"])
        g["shape"] = t["shape"]

    print(f"\nModel {MODEL_ID} @ {info['sha']}")
    print(f"{'tensor group':78s} {'n':>4s} {'params(M)':>10s}  shape(one)")
    for k in sorted(groups, key=lambda k: -groups[k]["params"])[:15]:
        g = groups[k]
        print(f"{k[:78]:78s} {g['count']:4d} {g['params']/1e6:10.1f}  {g['shape']} {','.join(g['dtype'])}")

    sol = speed_of_light(groups)
    print("\nBytes READ from GPU memory per decoded token (1 user, short context):")
    for k, v in sol["breakdown_gb"].items():
        print(f"  {k:45s} {v:6.2f} GB")
    print(f"  {'TOTAL (bf16)':45s} {sol['bytes_per_token_gb']:6.2f} GB")
    print("\nTheoretical max single-user decode speed = bandwidth / bytes per token:")
    for gpu, tps in sol["max_tok_s"].items():
        print(f"  {gpu:10s} {GPU_BANDWIDTH_GBS[gpu]:5d} GB/s -> {tps:6.0f} tok/s (bf16)   "
              f"{tps*2:6.0f} (fp8 weights)   {tps*4:6.0f} (4-bit weights)")

    out = pathlib.Path("results/s0_weights.json")
    out.write_text(json.dumps({
        "sha": info["sha"],
        "config": info["config"],
        "speed_of_light": sol,
        "groups": {k: {**v, "dtype": sorted(v["dtype"])} for k, v in groups.items()},
    }, indent=1))
    print(f"\nwrote {out}")
