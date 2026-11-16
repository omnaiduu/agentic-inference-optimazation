# Lesson 1: the concepts behind today's results (with our real numbers)

## 1. Memory bandwidth: the real speed limit
- The L40S does ~360 trillion multiply-adds per second (compute), but moves only 864 GB/s from VRAM (bandwidth).
- One token touches 2.3B weights:
  - Maths: 4.6 GFLOP ÷ 360 TFLOPS = **0.013 ms**.
  - Reading: 4.6 GB ÷ 864 GB/s = **5.3 ms**.
  - So reading takes ~400x longer than computing.
- Analogy: a super-fast chef waiting on a slow conveyor belt. To go faster, put fewer vegetables on
  the belt (quantization) or cook for many people per vegetable (batching).

## 2. Speed of light
- Max tokens/s = bandwidth ÷ bytes per token = 864 ÷ 4.6 = **188 tok/s** (bf16 on the L40S).
- vLLM reaches 128 tok/s (68% of it); ours reaches 144 (77%).
- It tells you how good a result is and how much room is left. Any claim above it with the same
  weights is impossible.

## 3. Quantization
- An int4 group of 32 weights, e.g. [0.12, −0.05, 0.31, …]: scale = max|w|/7.5 = 0.041, so each
  weight is stored as a small integer (3, −1, 7, …) and read back as q·scale (0.124 vs 0.12).
- 0.5 bytes + a 2-byte scale shared by 32 weights ≈ **0.56 bytes instead of 2**, i.e. 3.6x fewer bytes.
- PTQ just rounds a finished model. QAT (Google's checkpoint) retrains the model knowing it will be
  rounded.
- Measured: 4.6 → 1.3 GB per token; 144 → 327 tok/s; GSM8K 73% → ~70%.
- The lm_head (262,144 × 1,536 = 0.81 GB) grows from 18% to ~43% of the bytes once the other
  layers are int4.
  - Google's checkpoint and vLLM leave it in bf16.
  - Shrinking it to int4: 1,193 → 378 µs per token (240 → 327 tok/s).

## 4. Kernel
- A small GPU program run by thousands of threads. Our Triton GEMV:
  - each program handles a few output rows;
  - loads int32 words, unpacks 8 nibbles with `(w >> 4i) & 0xF`;
  - multiplies, applies the group scale, sums.
- Parallelism matters: "MLP down" (1,536 rows) gave only ~48 programs for 142 SMs, so it was
  slower than bf16. Fix: split-K.
- Benchmarks can lie: >864 GB/s readings came from the 96 MB L2 cache.

## 5. CUDA graphs and fusion
- One token = ~1,000 kernel launches. Python launches each one at ~5–40 µs of CPU cost.
- **CUDA graph** = record once, replay with one launch. Measured: 46 → 11.7 ms per token
  (21.6 → 85.3 tok/s, 4x).
- **Fusion** = merge tiny kernels so data makes one memory round trip instead of six (e.g. RMSNorm,
  6 kernels → 1, via torch.compile). Measured: 10.5 → 6.9 ms (96 → 144 tok/s).

## 6. Oracle and accuracy tests
- The oracle is the official HF implementation on 8 fixed prompts.
- Exact text match is weak: one early near-tie cascades. The official code itself disagrees with
  itself across GPUs.
- Teacher-forced token agreement: bf16 99.8%, int4 92%.
- GSM8K-100: 73–74% vs ~70% (±5 noise).
- The format test caught a would-be bug: the wrong unpack order gives cosine 0.0009 instead of 0.992.

## 7. Latency vs throughput
- Latency = TTFT (ours 39 ms, vLLM 34 ms) + per-token speed (ours 327 tok/s vs vLLM QAT 211).
- Throughput (vLLM bf16):
  - 1 user: 128 tok/s
  - 8 users: 884 tok/s total
  - 32 users: 2,663 tok/s total (89 each)
- Batching shares one weight read across all users.
- Cost per 1M tokens:
  - $4.23 for vLLM bf16, 1 user
  - **$1.66 for ours, 1 user**
  - $0.14 for vLLM QAT at 32 users (we don't batch yet)
- Latency matters for agents, coding and voice. Throughput matters for bulk offline jobs.
