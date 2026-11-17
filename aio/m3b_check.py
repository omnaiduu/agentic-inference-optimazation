"""Milestone 3b/3c: correctness + speed check of the batched engine's new paths, before a full benchmark.

Correctness, for each oracle case (teacher-forced on the oracle's output, token by token):
  old path : whole-prompt prefill (verified on day 1) + one eager batch-of-1 decode step per token
  new path : chunked prefill (480-token chunks) + decode INSIDE mixed steps (32 rows, so the
             full-attention layers use our Triton decode kernel), while another long prompt is
             being chunked into the staging slot at the same time
  run twice: prompt-chunk matmuls in BF16, then in FP8.
Speed: batched decode step (CUDA graph) at B = 32, mixed step with and without FP8.

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

    from aio.engine import quant
    from aio.engine.batch import BatchedGemma4

    pats = ["*.json", "*.safetensors", "tokenizer*"]
    bf_dir = snapshot_download(MODEL_ID, revision=MODEL_SHA, allow_patterns=pats)
    q_dir = snapshot_download(QAT_ID, revision=QAT_SHA, allow_patterns=pats)
    m = BatchedGemma4(bf_dir, quant_dir=q_dir, lm_head_bits=4, max_batch=32)
    m.enable_fp8_prefill()                 # builds the FP8 weight copies; toggled below
    m.enable_compile()
    out = {"variants": {}}
    cases = ref["cases"]
    other = max(cases, key=lambda c: len(c["prompt_ids"]))["prompt_ids"]   # the long prompt

    def teacher_forced_old(p, t):
        quant.FP8_PREFILL = False
        preds = [m.prefill_into(0, p)]
        pos = len(p)
        for tok in t[:-1]:
            preds.append(int(m.bforward(torch.tensor([tok], device="cuda"),
                                        torch.tensor([pos], device="cuda"), m._bbucket(pos))[0]))
            pos += 1
        return preds

    def teacher_forced_new(p, t, fp8):
        quant.FP8_PREFILL = fp8
        preds = [m.prefill_chunked(0, p)]
        pos, off = len(p), 0
        m.start_prefill()
        for tok in t[:-1]:
            if off >= len(other):
                m.start_prefill()
                off = 0
            nxt, _ = m.mixed_step([tok], [pos], other[off:off + m.chunk], off)
            off += m.chunk
            preds.append(nxt[0])
            pos += 1
        return preds

    with torch.inference_mode():
        olds = [teacher_forced_old(c["prompt_ids"], c["output_ids"]) for c in cases]
        n = sum(len(c["output_ids"]) for c in cases)
        hit_old = sum(a == b for o, c in zip(olds, cases) for a, b in zip(o, c["output_ids"]))
        print(f"===== old path: oracle agreement {hit_old}/{n} = {hit_old / n:.1%}", flush=True)
        out["oracle_old"] = hit_old / n
        for fp8 in (False, True):
            t0 = time.time()
            quant.FP8_PREFILL = fp8
            m.capture_mixed_graphs()                 # CUDA graphs of the mixed step for this setting
            news = [teacher_forced_new(c["prompt_ids"], c["output_ids"], fp8) for c in cases]
            same = sum(a == b for o, nw in zip(olds, news) for a, b in zip(o, nw))
            hit = sum(a == b for nw, c in zip(news, cases) for a, b in zip(nw, c["output_ids"]))
            long_same = sum(a == b for a, b in zip(olds[-1], news[-1]))
            name = "fp8" if fp8 else "bf16"
            print(f"===== new path ({name} chunks): same as old {same}/{n} = {same / n:.1%}, oracle agreement "
                  f"{hit / n:.1%}, long prompt {long_same}/{len(news[-1])}  ({time.time() - t0:.0f}s incl. compile)",
                  flush=True)
            out["variants"][name] = {"same_as_old": same / n, "oracle": hit / n}
            commit_volumes()

        # ---- speed: batched decode step at B=32 (CUDA graph, hybrid attention)
        ctx = 1100
        quant.FP8_PREFILL = False
        m.prefill_into(0, other[:ctx])
        for d in range(1, m.max_b):
            m.move_slot(0, d)
        m.capture_batch_graphs(sizes=(32,), buckets=(2048,))
        for d in range(1, m.max_b):        # capture wrote junk into the slots: refill
            m.move_slot(0, d)
        m.prefill_into(0, other[:ctx])
        for d in range(1, m.max_b):
            m.move_slot(0, d)
        toks, poss = [5] * 32, [ctx] * 32
        m.decode_batch(toks, poss, 32)
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(4):
            m.decode_batch(toks, poss, 32)
        en.record()
        torch.cuda.synchronize()
        out["decode32_ms"] = st.elapsed_time(en) / 128
        print(f"===== decode step B=32 (graph): {out['decode32_ms']:.2f} ms (day-2 morning: 6.01 ms)", flush=True)

        # ---- speed: mixed step, 31 decode rows + one 480-token chunk, BF16 vs FP8
        out["mixed_ms"] = {}
        for fp8 in (False, True):
            quant.FP8_PREFILL = fp8
            args = ([5] * 31, [ctx] * 31, other[:m.chunk], 0)
            m.start_prefill()
            m.mixed_step(*args)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                m.mixed_step(*args)
            ms = (time.perf_counter() - t0) / 20 * 1000
            out["mixed_ms"]["fp8" if fp8 else "bf16"] = ms
            print(f"===== mixed step 31 rows + {m.chunk} prompt tokens, {'FP8' if fp8 else 'BF16'}: {ms:.2f} ms "
                  f"(no graph, this morning: BF16 17.85 ms, FP8 33.07 ms)", flush=True)
    commit_volumes()
    return out


@app.local_entrypoint()
def main():
    ref = json.loads(pathlib.Path("results/s1_reference.json").read_text())
    res = check.remote(ref)
    pathlib.Path(f"results/m3c_check_{GPU}.json").write_text(json.dumps(res, indent=1))
