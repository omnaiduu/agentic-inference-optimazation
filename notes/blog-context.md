# Blog context: "Agentic inference optimization on a $20 budget"

Raw material for a blog post. The detailed day-by-day evidence is in `engineering-log.md`, and the
concepts are explained in `lesson-1-concepts.md`. All numbers are measured on our own runs, unless
a source link says otherwise.

## The hook
Baseten had Claude Code build a custom inference engine for Qwen 3.6 that beat vLLM by up to 90%
on a B200, with ~1 week and ~$1,000s of compute
([blog](https://www.baseten.co/blog/agentic-inference-optimization-faster-than-sota/)).
**Can the same idea work for an individual, on a small model, with $20 of Modal credit and one day?**

## Setup
| | Baseten | Us |
|---|---|---|
| Model | Qwen 3.6 35B-A3B (MoE, NVFP4) | Gemma 4 E2B-it (dense, effective 2.3B) |
| GPU | 1x B200 | 1x L40S on Modal ($1.95/h) |
| Baseline | vLLM 0.25.1 | vLLM 0.33.0 (bf16, and Google's QAT int4 checkpoint) |
| Agent | Claude Code + Fable 5, mostly autonomous | Claude Code, human-steered step by step |
| Compute spent | ~200 B200-hours | **$6.05** (day 1) |

## Headline results (L40S, single user, ~1,100-token prompt, 256 output tokens)
| Engine | tok/s | vs vLLM bf16 | GSM8K-100 | Token agreement with oracle |
|---|---|---|---|---|
| vLLM 0.33, bf16 | 128 | 1.0x | 73% | 99.5% |
| vLLM 0.33, Google QAT int4 | 211 | 1.6x | 68% | 91.2% |
| ours, bf16 (compiled) | 127–144 | ~1.0–1.1x | 74% | 99.8% |
| **ours, QAT int4 + our int4 lm_head** | **343** | **2.7x** | **70%** | 91.2% |

- **1.63x faster than vLLM on the same 4-bit weights**, with no measurable accuracy loss on top of
  the QAT checkpoint itself.
- Cost per 1M output tokens at single-stream serving: **$4.23 (vLLM bf16) → $1.66 (ours)**.

## The story arc (each step has numbers in engineering-log.md)
1. **Speed of light first.** We counted the bytes read per token from the safetensors metadata.
   - The checkpoint is ~10 GB, but only 4.6 GB is read per token: the 4.7 GB per-layer-embedding
     table is a lookup, and vision and audio aren't used for text.
   - Max on the L40S: 188 tok/s. The lm_head alone is 17.6% of the bytes.
2. **Oracle before optimization.** HF transformers greedy outputs became the answer key.
   - Surprise: the official code gives *different* text on the L4 vs the L40S (near-tie logits).
     Exact match is too strict a gate.
3. **Our own engine in ~250 lines of PyTorch.** We re-derived 9 details from `modeling_gemma4.py`
   that must match exactly: the scale cast to bf16, attention scale 1.0, proportional RoPE,
   KV sharing for 20 layers, PLE, layer_scalar, softcap, and so on.
   - Eager: 21.6 tok/s.
   - **CUDA graphs: 85.3 (4x).**
   - **Fused weights + torch.compile: 144** (beats vLLM bf16's 128).
4. **Quantization plus our first Triton kernel.**
   - Verified Google's int4 packing empirically: cosine 0.992 vs 0.0009 for the wrong order.
   - Wrote a W4A16 GEMV kernel.
   - **Google's checkpoint says `ignore: [lm_head]`, so vLLM reads a 0.81 GB bf16 lm_head every
     token. We quantized it too:** 1,193 µs → 378 µs per token. That's the single biggest edge
     (240 → 327 tok/s).
5. **Profile, then fix.** GEMV 72%, attention 12%, host sync 10%.
   - Ring-buffer KV for the 512-window layers, split-K for skinny matrices, multi-step decode.
   - Result: 342.7 tok/s.
6. **The bug the gate caught.** The ring-buffer mask was built before the current position was
   written, so new tokens couldn't attend to themselves. The text looked fine at a glance, but
   **GSM8K went 74% → 17%**. Fixed in two lines.
   - This is the best illustration of Baseten's warning: an optimizing agent will happily ship
     "fast but broken".
7. **Continuous batching:** a small-batch int4 GEMM on tensor cores, 32 KV slots with compaction,
   and a scheduler thread. Accuracy held (91.2% agreement, GSM8K 66%), and throughput rose 6.7x from
   1 to 32 users, but **vLLM still wins at scale**:
   | Users | Ours total tok/s | vLLM QAT total tok/s |
   |---|---|---|
   | 1 | 281 | 211 |
   | 8 | 1,151 | 1,442 |
   | 32 | 1,888 | 3,757 |
   The cause is visible: our prefill is one request at a time (TTFT 856 ms at 32 users), and our
   attention is not a real decode kernel. vLLM's years of serving work (chunked prefill, paged
   attention) matter exactly here.
8. **Day 2: measure, then copy the right idea.** Profiling showed our batched decode step (6.0 ms
   at 32 users) was already faster than vLLM's; the loss was the scheduler freezing everyone
   while a prompt ran alone. We built vLLM's chunked prefill: one pass = a token for every user +
   a 480-token piece of a new prompt. Profiling the new step found two invisible costs: the
   compiler copying whole 138 MB caches (a view-vs-tensor indexing detail), and 288-row matrix
   multiplies wasting a nearly empty 256-row tile. After fixes:
   | Users | Day 1 | Day 2 | vLLM QAT |
   |---|---|---|---|
   | 8 | 1,151 | **1,407** | 1,442 |
   | 32 | 1,888 | **2,779** | 3,757 |
   Time to first token at 32 users: 856 → 199 ms (vLLM: 202). Accuracy unchanged.

## Quotable lessons
- **Decode is a memory problem, not a maths problem.** At batch 1 the L40S needs 5.3 ms to read
  the weights and 0.013 ms to do the maths (~400x).
- **Measure the ceiling first.** "77% of speed of light" means something; "144 tok/s" alone doesn't.
- **Kernel count is a hidden cost.** Graphs gave 4x and fusion +50%, with no maths changes.
- **Read the checkpoint config.** One line (`ignore: [lm_head]`) pointed at a ~36% speed-up nobody
  was taking.
- **Benchmarks lie in small ways.** Microbenchmarks showed more than 864 GB/s because the test
  matrix stayed in the 96 MB L2 cache.
- **Profile before you build.** We were about to write an attention kernel; the profile said the
  scheduler was the problem. The fix was copying vLLM's idea, not out-engineering it.
- **Compilers hide costs.** One indexing detail made Inductor copy 830 MB per step (20% of the time).
- **Gate every change on accuracy**, with tests that can't cascade (teacher-forced agreement) and
  tests that measure capability (GSM8K).
- **The human's role** is to set the goal, the rules and the budget, and to judge whether a result
  is real. In this run, the human's calls changed the outcome:
  - building our own engine before tuning vLLM
  - asking about kernels
  - insisting on verifying facts with search
  - asking what it saves in dollars

## Honest limitations (put these in the post)
- One GPU type (L40S), one model. We win single-user speed (1.5x) and tie at 8 users; at 32
  users vLLM is still 1.35x ahead on total throughput.
- GSM8K-100 has about ±5 points of noise. The QAT checkpoint costs about 3–6 points vs bf16 (near
  the noise floor). Our int4 lm_head adds nothing measurable on top.
- TTFT is 38–40 ms vs vLLM's 34 ms. Our prefill is eager PyTorch.
- Our engine only supports text, Gemma 4 E2B, prompts up to 4K tokens, and greedy decoding.
- The bf16 run-to-run gap (144 → 125–130 tok/s) is unexplained. Different hosts are the most
  likely cause.
- Infrastructure friction was real:
  - The sandboxed agent needed Modal's new HTTP-proxy support.
  - vLLM's pip install lacked `nvcc` for FlashInfer JIT.
  - The official vLLM image wasn't detected by Modal.
  - The fix was Modal's recipe: CUDA devel image + `add_python`.

## Still to do (day 2)
- Close the rest of the 32-user gap: a real decode-attention kernel, smaller chunks when many
  users are waiting.
- Deeper fusion (toward a per-layer "megakernel").
- An H100 run for the headline numbers.
- Optionally Gemma's MTP speculative decoding (+30% per Google).
