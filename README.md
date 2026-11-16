# Agentic inference optimization: Gemma 4 E2B on a $20 budget

A small reproduction of Baseten's "agentic inference optimization" experiment. Claude Code builds and
optimizes a custom inference engine for `google/gemma-4-E2B-it` on Modal, and races it against vLLM 0.33.

**Best so far (L40S, 1 user):** 343 tok/s, vs vLLM 211 (same QAT int4 weights) and vLLM 128 (bf16).
Accuracy matches the QAT checkpoint.

- `notes/blog-context.md`: the story, results and lessons for the blog
- `notes/engineering-log.md`: detailed, numbered engineering log with evidence
- `notes/lesson-1-concepts.md`: the concepts explained with this project's numbers
- `aio/engine/`: our engine (`model.py`), Triton kernels (`quant.py`), batching (`batch.py`), servers
- `aio/s0_speed_of_light.py`, `aio/s1_baseline.py`, `aio/m2_run.py`, `aio/m3_profile.py`: experiments
- `results/`: raw JSON of every run

Run, for example:

    pip install "modal[api-proxy-support]"
    modal run -m aio.s0_speed_of_light
    modal run -m aio.m2_run --configs vllm-qat,ours-qat-lm4
