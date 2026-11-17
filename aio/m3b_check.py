"""Milestone 3b: correctness + speed check of chunked prefill (mixed steps), before the full benchmark.

For each oracle case:
  old path: whole-prompt prefill (verified on day 1) + one eager batched decode step per token
  new path: chunked prefill (256-token chunks) + decode INSIDE mixed steps, while another long
            prompt is being chunked into the staging slot at the same time
Both are teacher-forced on the oracle's output, so we compare predictions token by token.

Run:  modal run -m aio.m3b_check
"""
import json
import pathlib

import modal

from aio.common import GPU, MODEL_ID, VOLUMES, commit_volumes, gpu_image
from aio.m2_run import QAT_ID, QAT_SHA
from aio.s1_baseline import MODEL_SHA

app = modal.App("aio-m3b-check")


@app.function(image=gpu_image, gpu=GPU, volumes=VOLUMES, timeout=3600)
def check(ref: dict) -> dict:
    import time

    import torch
    from huggingface_hub import snapshot_download

    from aio.engine.batch import BatchedGemma4

    pats = ["*.json", "*.safetensors", "tokenizer*"]
    bf_dir = snapshot_download(MODEL_ID, revision=MODEL_SHA, allow_patterns=pats)
    q_dir = snapshot_download(QAT_ID, revision=QAT_SHA, allow_patterns=pats)
    m = BatchedGemma4(bf_dir, quant_dir=q_dir, lm_head_bits=4, max_batch=32)
    m.enable_compile()
    t0 = time.time()
    m.warm_mixed()
    print(f"mixed step compiled in {time.time() - t0:.0f}s", flush=True)
    commit_volumes()

    cases = ref["cases"]
    other = max(cases, key=lambda c: len(c["prompt_ids"]))["prompt_ids"]   # the long prompt
    out = {"cases": []}
    tot = same = hit_old = hit_new = 0
    with torch.inference_mode():
        for c in cases:
            p, t = c["prompt_ids"], c["output_ids"]
            # old path
            old = [m.prefill_into(0, p)]
            pos = len(p)
            for tok in t[:-1]:
                old.append(int(m.bforward(torch.tensor([tok], device="cuda"),
                                          torch.tensor([pos], device="cuda"), m._bbucket(pos))[0]))
                pos += 1
            # new path
            new = [m.prefill_chunked(0, p)]
            pos, off = len(p), 0
            m.start_prefill()
            for tok in t[:-1]:
                if off >= len(other):
                    m.start_prefill()
                    off = 0
                nxt, _ = m.mixed_step([tok], [pos], other[off:off + m.chunk], off)
                off += m.chunk
                new.append(nxt[0])
                pos += 1
            s = sum(a == b for a, b in zip(old, new))
            ho = sum(a == b for a, b in zip(old, t))
            hn = sum(a == b for a, b in zip(new, t))
            tot, same, hit_old, hit_new = tot + len(t), same + s, hit_old + ho, hit_new + hn
            print(f"  prompt {len(p):5d} tok: old==new {s}/{len(t)}  oracle: old {ho} new {hn}", flush=True)
            out["cases"].append({"prompt_len": len(p), "n": len(t), "same": s, "oracle_old": ho, "oracle_new": hn})
    out.update(total=tot, same=same, oracle_old=hit_old, oracle_new=hit_new)
    print(f"===== old==new {same}/{tot} = {same / tot:.1%} | oracle agreement: old {hit_old / tot:.1%}, "
          f"new {hit_new / tot:.1%}", flush=True)

    # speed: 31 decode rows + one 256-token chunk, at ~1,100 tokens of context
    ctx = 1100
    m.prefill_into(0, other[:ctx])
    for d in range(1, m.max_b):
        m.move_slot(0, d)
    out["mixed_ms"] = {}
    for n in (0, 8, 31):
        toks, poss = [5] * n, [ctx] * n
        m.start_prefill()
        m.mixed_step(toks, poss, other[:m.chunk], 0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(20):
            m.mixed_step(toks, poss, other[:m.chunk], 0)
        ms = (time.perf_counter() - t0) / 20 * 1000
        out["mixed_ms"][n] = ms
        print(f"===== mixed step: {n:2d} decode rows + {m.chunk} prompt tokens: {ms:.2f} ms", flush=True)
    commit_volumes()
    return out


@app.local_entrypoint()
def main():
    ref = json.loads(pathlib.Path("results/s1_reference.json").read_text())
    res = check.remote(ref)
    pathlib.Path(f"results/m3b_check_{GPU}.json").write_text(json.dumps(res, indent=1))
