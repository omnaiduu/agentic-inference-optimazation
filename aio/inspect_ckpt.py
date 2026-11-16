"""Print tensor names / shapes / dtypes of any HF checkpoint (metadata only, no download).

Run:  modal run -m aio.inspect_ckpt --model google/gemma-4-E2B-it-qat-w4a16-ct
"""
import json
import pathlib
from collections import defaultdict

import modal

from aio.common import cpu_image
from aio.s0_speed_of_light import group_of

app = modal.App("aio-inspect")


@app.function(image=cpu_image, timeout=600)
def inspect(model: str) -> dict:
    from huggingface_hub import HfApi, get_safetensors_metadata, hf_hub_download
    sha = HfApi().model_info(model).sha
    cfg = json.loads(pathlib.Path(hf_hub_download(model, "config.json", revision=sha)).read_text())
    meta = get_safetensors_metadata(model, revision=sha)
    t = {n: {"dtype": x.dtype, "shape": x.shape} for f in meta.files_metadata.values() for n, x in f.tensors.items()}
    return {"sha": sha, "config": cfg, "tensors": t}


@app.local_entrypoint()
def main(model: str):
    info = inspect.remote(model)
    groups = defaultdict(lambda: {"n": 0, "dtype": set(), "shape": None})
    for name, x in info["tensors"].items():
        g = groups[group_of(name)]
        g["n"] += 1; g["dtype"].add(x["dtype"]); g["shape"] = x["shape"]
    print("sha", info["sha"])
    for k in sorted(groups):
        if "vision" in k or "audio" in k:
            continue
        g = groups[k]
        print(f"{k[:80]:80s} {g['n']:3d} {g['shape']} {','.join(g['dtype'])}")
    qc = info["config"].get("quantization_config", {})
    qc["ignore"] = [i for i in qc.get("ignore", []) if "vision" not in i and "audio" not in i]
    print(json.dumps(qc, indent=1)[:3000])
    out = pathlib.Path(f"results/ckpt_{model.split('/')[-1]}.json")
    out.write_text(json.dumps({"sha": info["sha"], "quantization_config": qc,
                               "tensors": {k: v for k, v in info["tensors"].items()
                                           if "vision" not in k and "audio" not in k}}, indent=0))
