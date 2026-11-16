# Engineering log: Agentic inference optimization on a $20 budget

A small, cheap reproduction of Baseten's "Agentic inference optimization" experiment
([blog](https://www.baseten.co/blog/agentic-inference-optimization-faster-than-sota/), Oct 2026),
which builds on the MetaInfer paper ([arXiv 2607.12875](https://arxiv.org/abs/2607.12875),
[repo](https://github.com/MetaInfer/MetaInfer)).

| | Baseten | Us |
|---|---|---|
| Model | Qwen 3.6 35B-A3B (MoE), NVFP4 | Gemma 4 E2B-it, BF16 |
| GPU | 1x B200 | L4 for development, H100 for the final showdown |
| Agent | Claude Code + Fable 5, ~1 week, ~1.7B tokens | Claude Code, human-in-the-loop, step by step |
| Budget | ~200 B200-hours (~$1,000s) | $20 of Modal credit |
| Baseline | vLLM 0.25.1 | vLLM 0.33.0 |

Format of each entry: **what we did → what we found → lesson**. The sources were checked with
Exa search while working.

---

## Day 1 (2026-11-16)

### 1. Picking the model
- Gemma 4 has five sizes: E2B, E4B, 12B, 26B-A4B (MoE), 31B. We picked **E2B** because it is
  the cheapest to experiment on.
  Source: [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4), [HF blog](https://huggingface.co/blog/gemma4).
- It is Apache 2.0 licensed, and vLLM has supported it since 0.19.1.
  Source: [vLLM blog](https://vllm.ai/blog/2026-04-02-gemma4), [vLLM recipe](https://recipes.vllm.ai/Google/gemma-4-E2B-it).
- Pinned weights: `google/gemma-4-E2B-it` @ `4adf15748de9e1a0097b6274cee9c15836c41c37`.

### 2. Infrastructure hiccup: driving Modal from a sandboxed cloud agent
- The agent runs in a sandbox where all traffic goes through an HTTPS proxy.
- The Modal client speaks gRPC. At first we thought that couldn't go through a proxy.
- An Exa search found that the Modal client **added HTTP CONNECT proxy support in June 2026**
  ([commit 89e0594](https://github.com/modal-labs/modal-client/commit/89e059452c77a46db57a0af77fcdd21c297a6ed0)).
  It needs `pip install "modal[api-proxy-support]"`.
- The proxy also had to allow these hosts: `api.modal.com` (control plane), plus
  `input-plane.us-east.modal.com` and `input-plane.us-west.modal.com` (function calls).
  The input-plane hosts were only discovered when the first function call failed.
- **Lesson:** an agent's environment is part of the experiment. Allow the smallest set of hosts,
  and find the missing ones from the proxy's failure log instead of guessing.

### 3. Stage 0: speed of light (`aio/s0_speed_of_light.py`)
Before optimizing anything, we asked: *what is the fastest this model could possibly decode on
each GPU?*

To generate one token for one user, the GPU must read every weight that token touches from memory,
once. At batch size 1 the GPU's maths units are mostly idle. **Memory bandwidth is the bottleneck.**

We read the safetensors metadata (no download) and counted the bytes:

| Component | GB read per token (bf16) | Note |
|---|---|---|
| 35 decoder layers (attention + MLP + PLE projections) | 3.76 | Read in full |
| lm_head (tied to embed_tokens, 262,144 × 1,536) | 0.81 | Read in full, every token |
| Norms, small projections | 0.03 | |
| **Total read per token** | **4.60** | |
| Per-Layer Embedding table (262,144 × 8,960) | 4.70 on disk | Only one row is looked up, ~0 GB read |
| Vision + audio towers | 0.95 on disk | Not used for text |

So of a ~10 GB checkpoint, only 4.6 GB is touched per token. **This is exactly what "E2B"
(2.3B effective parameters) means.**

Theoretical max single-user decode speed = bandwidth / 4.6 GB:

| GPU | Bandwidth | bf16 | fp8 weights | 4-bit weights |
|---|---|---|---|---|
| L4 | 300 GB/s | **65 tok/s** | 131 | 261 |
| A10 | 600 GB/s | 131 | 261 | 522 |
| L40S | 864 GB/s | 188 | 376 | 752 |
| H100 | 3,350 GB/s | **729 tok/s** | 1,457 | 2,915 |
| B200 | 8,000 GB/s | 1,740 | 3,480 | 6,961 |

Observations worth a blog paragraph:
- **The lm_head is 17.6% of all the bytes read per token** (0.81 of 4.60 GB). That's because the
  vocabulary is huge (262K). This explains why Google's E2B draft model uses "centroid masking":
  it scores only ~4K candidate tokens instead of all 262K
  ([vLLM Gemma 4 guide](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html)).
- 20 of the 35 layers **reuse the KV cache** of earlier layers (`num_kv_shared_layers: 20`), so
  they compute no K/V projections. Only 7 of the 35 layers use full attention; the other 28 use a
  512-token sliding window. This means the KV cache is tiny, and long contexts stay cheap.
- **Quantization is the biggest lever for single-user speed.** Halving the bytes halves the time.

### 4. GPU choice
- The default is **L4** ($0.80/h): the cheapest GPU that fits the model (24 GB) **and** has FP8
  tensor cores (Ada).
  - A10 and A100 have no FP8.
  - T4 has no BF16 and no modern attention kernels.
- The final head-to-head run is on **H100** ($3.95/h).
- Modal prices were checked on 2026-11-16 ([modal.com/pricing](https://modal.com/pricing)).

### 5. Stage 1: oracle and baseline (`aio/s1_baseline.py`), in progress
- **Oracle:** Hugging Face transformers (the official implementation) greedy-decodes 8 fixed prompts.
  - One prompt is ~1,500 tokens long, so it exercises the 512-token sliding window.
  - Every engine must reproduce these outputs.
  - Why this matters: Baseten's blog warns that without an accuracy gate, an optimizing agent
    converges on `while True: yield "a"`.
- **Baseline:** stock vLLM 0.33.0, measured with our own load generator (`aio/loadgen.py`).
  - Same client for every engine, so the comparison is fair.
  - `ignore_eos` forces an identical amount of work per request.
- *Hiccup:* the first oracle run crashed. `device_map="cuda"` in `from_pretrained` needs the
  `accelerate` package, which we hadn't installed. Fixed with a plain `.to("cuda")`.
  Cost: ~1 minute of container time.
- *Exa check of the oracle setup:*
  - `AutoModelForCausalLM` is a documented way to load Gemma 4 for text
    ([HF Gemma 4 docs](https://huggingface.co/docs/transformers/main/model_doc/gemma4)).
  - The official model card installs `accelerate` for `device_map`
    ([model card](https://huggingface.co/google/gemma-4-E2B-it)), which confirms our crash diagnosis.
  - Pitfall: the chat template has an `enable_thinking` switch, so two engines could format the
    same prompt differently. We avoid this by templating **once** in the oracle and sending the
    exact `prompt_ids` to every engine.
- **Status: still baseline.** Nothing has been optimized yet. Stage 1 only defines "correct"
  (the oracle) and "normal speed" (stock vLLM).

### 6. GPU switch: L4 → L40S
- L4 jobs sat for 10+ minutes with no container allocated. The likely cause is capacity.
- We switched the default to **L40S**: same Ada generation (FP8 support), 864 GB/s, 48 GB, $1.95/h.
- New speed-of-light target: **188 tok/s** in bf16 (from the Stage 0 table).

### 7. Plan change: build our own engine first (user's call)
We skip vLLM tuning for now. Instead:
- answer key (oracle)
- one stock vLLM baseline run
- engine stages:
  - **E1** correct PyTorch engine
  - **E2** CUDA graphs
  - **E3** custom Triton kernels: 4-bit layers, **4-bit lm_head**, fused ops
  - **E4** speculative decoding
  - **E5** batching / serving
- The correctness check runs automatically inside every benchmark run (~10 s).

Kernel targets come from Stage 0:
- Google's QAT w4a16 checkpoint quantizes only `Linear` targets (group size 32), per its
  [config](https://huggingface.co/google/gemma-4-E2B-it-qat-w4a16-ct).
- Gemma's lm_head is tied to the embedding table, so it very likely stays bf16 (to verify on load).
- With 4-bit layers, the bf16 lm_head becomes ~45% of the bytes read per token.

### 8. Oracle saved (L40S)
- `results/s1_reference.json`: 8 prompts, greedy decoding, 64 new tokens, transformers 5.20.2, torch 2.14.
- **Lesson: greedy decoding is not bit-identical across GPUs.** Prompt 2 started "Here are a few ways…"
  on the L4 but "Here is a Python function…" on the L40S, with the same code and weights.
  - Cause: different GPUs order floating-point additions differently, so near-tied logits can flip.
  - Consequence: the correctness gate must allow "close-call" divergences, not demand 100% exact
    text. Baseten made the same allowance (subtle numerical differences, with accuracy at least as
    good as BF16).

### 9. vLLM baseline attempt 1 failed: a missing compiler
- On the first sampling call, vLLM 0.33's FlashInfer sampler JIT-compiles a CUDA kernel.
  That needs `nvcc`, which our slim pip-installed image doesn't have:
  `RuntimeError: Could not find nvcc`.
- Fix attempt 1: use the **official image** `vllm/vllm-openai:v0.33.0` (CUDA 13.0.3, released
  2026-11-02, confirmed via the [release notes](https://github.com/vllm-project/vllm/releases/)).
  Modal rejected it: *"unable to determine the version of Python installed in the Image"*.
  The image keeps its Python somewhere Modal doesn't look.
- Fix attempt 2 (works): follow [Modal's own vLLM guide](https://modal.com/docs/examples/vllm_inference).
  Start from `nvidia/cuda:13.0.3-devel-ubuntu24.04` (the CUDA *devel* image, which includes nvcc),
  add Modal's standalone Python 3.12 with `add_python="3.12"`, then `pip install vllm==0.33.0`.
  CUDA 13.0.3 matches the official vLLM build. Both failed attempts cost $0, because they failed
  before any GPU was allocated.
- We use the same image for our engine, so both sides run identical torch/CUDA.
- **Lesson:** "stock vLLM" should mean the official image, not a pip install. Performance
  libraries increasingly JIT-compile kernels at runtime and need a full CUDA toolkit.

### 10. Milestone 1: our engine (E1 correct + E2 CUDA graphs)
- `aio/engine/model.py` is ~250 lines of PyTorch. It was re-derived by reading
  transformers 5.20.2 `modeling_gemma4.py`. These 9 details must match exactly; any one of them
  wrong gives garbage text:
  1. Embedding scale sqrt(1536), **cast to bf16 first** (39.19 becomes 39.25).
  2. RMSNorm runs in fp32 and multiplies by `w`, not `1 + w`.
  3. q_norm and k_norm are per-head RMSNorms; v_norm has **no weight**.
  4. Attention scale = **1.0**, not 1/sqrt(d).
  5. Sliding layers: head_dim 256, RoPE θ=1e4. Full layers: head_dim **512**, "proportional"
     RoPE θ=1e6 that rotates only 25% of the dims.
  6. Layers 15–34 compute no K/V. They read layer 13's KV (sliding layers) or layer 14's KV
     (full layers).
  7. PLE input = (RMSNorm(proj(embeds)/√1536) + table[tok]·16)/√2, injected after the MLP via
     gate → GELU → ×PLE → proj → norm.
  8. A learned `layer_scalar` multiplies each layer's output.
  9. The lm_head is tied to the embedding table; logits are soft-capped to 30·tanh(x/30).
- Static KV cache: only the 15 non-shared layers need one, and it has 1 KV head.
  That is ~75 MB at 4K context (12 sliding layers × 256 dims + 3 full × 512, for K and V, bf16), tiny compared with the 4.6 GB of weights.
- E2: one CUDA graph per cache-length bucket (256, 512, …, 4096). The decode step replays as
  one launch instead of ~1,000 separate kernel launches.
- `aio/engine/server.py` is an OpenAI-compatible `/v1/completions` endpoint, so the same
  load generator and the same oracle check run against both engines.

### 11. Caching, so we stop paying for the same work twice
| What | Where it is kept | Saves per run |
|---|---|---|
| Container image (CUDA devel + vLLM, ~7 GB) | Modal's image cache (automatic) | minutes |
| Model weights (10 GB) | volume `aio-hf-cache` (`HF_HOME`) | ~1 min download |
| vLLM torch.compile graphs | volume `aio-compile-cache` (`VLLM_CACHE_ROOT`) | ~1–2 min of startup |
| TorchInductor and Triton kernels | `aio-compile-cache` (`TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`) | seconds to minutes |
| FlashInfer JIT kernels (the `nvcc` builds) | `aio-compile-cache` (`FLASHINFER_WORKSPACE_BASE`) | ~30 s+ |

The environment variable names were verified in the source code: `flashinfer/jit/env.py` and `vllm/envs.py`.
The new env layer sits on top of the cached pip layer, so adding it only rebuilds a tiny layer.

### 12. Milestone 1, first results: our engine is CORRECT ✅
| Engine | Oracle check (8 prompts, greedy, 64 tokens) |
|---|---|
| stock vLLM 0.33 | 7/8 exact, 88% average common prefix |
| **our engine (eager)** | 6/8 exact, 84% average common prefix |
| **our engine (CUDA graphs)** | 6/8 exact, 84% (identical to eager, as it should be) |

- Both engines diverge on the **same** prompt (the prime-number one), at the same near-tie.
  That's the prompt where the L4 and L40S runs of the official code also disagreed.
- Our second divergence happens 71% of the way into the answer. It's a late near-tie, not a bug:
  everything before it matches token for token.
- Our engine starts up in **12–16 s**, vs **260 s** for vLLM (torch.compile and graph capture,
  uncached). Capturing all 5 CUDA graphs takes 1.0 s.
- *Bug in our load generator:* the benchmark prompts were raw text with no chat template. The
  instruction-tuned model answered with special tokens only, so no visible text ever arrived and
  TTFT was never set.
  - Fix: wrap the prompts in Gemma's `<|turn>user … <turn|>\n<|turn>model\n` format.
  - Also: TTFT now starts at the first streamed chunk.

### 13. Milestone 1 speed results (L40S, prompt ~1,100 tokens, 256 output tokens)
| Engine | 1 user: TPOT | 1 user: tok/s | TTFT (1 user) | % of speed of light (188 tok/s) |
|---|---|---|---|---|
| stock vLLM 0.33 | 7.78 ms | **128.5** | 33.6 ms | 68% |
| our engine, eager | 46.31 ms | 21.6 | 50.9 ms | 11% |
| our engine, CUDA graphs | 11.73 ms | **85.3** | 51.9 ms | 45% |

vLLM at higher load: 8 users → 884 tok/s total (116 tok/s each); 32 users → 2,667 tok/s total (89 tok/s each).

**Lesson 1: CUDA graphs alone gave a 4x speed-up (21.6 → 85.3 tok/s), with zero maths changes.**
In eager mode, the GPU sits idle waiting for the CPU (Python) to launch the next of ~1,000 tiny kernels.

**Lesson 2: we're still at 66% of vLLM.** The speed of light per token is 4.6 GB ÷ 864 GB/s = 5.3 ms.
vLLM spends 7.8 ms per token, and we spend 11.7 ms. Suspects:
- **Unfused elementwise ops.** Each RMSNorm is ~6 separate kernels (float cast, pow, mean, add, pow,
  mul, cast), and there are ~6 norms × 35 layers. Even inside a CUDA graph, each kernel costs a few
  µs of fixed overhead, and ~1,000 kernels × ~3 µs ≈ 3 ms. That's most of the gap.
- Separate q/k/v and gate/up matmuls: more kernels, and each one reads less data per launch.
- Attention upcasts K/V to fp32 over the whole cache bucket.

### 14. Fusion round: we now BEAT stock vLLM for a single user 🎉
Two changes:
- Fused weights: one qkv matmul instead of three, one gate_up matmul instead of two.
- `torch.compile` (TorchInductor) on the decode step, then our own CUDA graphs on top.

| Engine (L40S, 1 user) | tok/s | TPOT | TTFT | Oracle | % speed of light |
|---|---|---|---|---|---|
| stock vLLM 0.33 | 128.5 | 7.78 ms | **33.6 ms** | 7/8 | 68% |
| ours: CUDA graphs (M1) | 85.3 | 11.73 ms | 51.9 ms | 6/8 | 45% |
| ours: + fused qkv / gate_up | 95.6 | 10.46 ms | 38.3 ms | 5/8 | 51% |
| **ours: + torch.compile** | **144.2** | **6.93 ms** | 38.6 ms | **7/8** | **77%** |

- **+12% faster per token than stock vLLM.** The diagnosis was right: kernel count, not maths, was
  the bottleneck. Inductor fused each RMSNorm, GELU·up and RoPE into single kernels.
- Fusing the weights alone gave +12% (85 → 96). Compile gave a further +51% (96 → 144).
- **Accuracy:** the compiled engine matches the oracle on 7/8 prompts, the same score as vLLM,
  and diverges on the same near-tie prompt.
- **Still behind:** TTFT is 38.6 vs 33.6 ms. Our prefill (reading the prompt) is still uncompiled
  eager PyTorch.
- **Cost:** the first compile took 256 s. It's now cached in the `aio-compile-cache` volume.
- vLLM still wins on throughput with many users. Our engine serves one request at a time; batching
  is a later milestone.

## Milestone 2: low-bit weights and our first custom kernel

### 15. Inside Google's QAT int4 checkpoint (`google/gemma-4-E2B-it-qat-w4a16-ct` @ 971342c)
- Every Linear layer is int4: symmetric, group size 32, packed 8 values per int32 along K
  (`weight_packed` [N, K/8], `weight_scale` [N, K/32] in bf16).
- **`"ignore": ["lm_head"]`**: the lm_head stays BF16, and so do both embedding tables.
  So every engine that serves this checkpoint as-is still reads the 0.81 GB bf16 lm_head on
  every token. That's where we can win.
- Packing order: the checkpoint was written by a pre-release of compressed-tensors (0.15.1.a…),
  so we verified the order empirically instead of trusting the code.
  - Unpacked QAT weights vs the original bf16 weights: **cosine 0.992**.
  - The same bytes read in the wrong (interleaved) order: cosine 0.0009.
  - Relative difference: 30%. QAT weights are re-trained, not just rounded.

### 16. Our Triton W4A16/W8A16 GEMV kernel (`aio/engine/quant.py`)
Each program takes a few output rows. It loads packed int32 words, unpacks 8 nibbles with
shifts, does the dot product per 32-wide group in fp32, and applies the group scale once per group.

Microbenchmark on the L40S (CUDA-graph timed, 200 iterations):

| Layer | bf16 cuBLAS | int8 (ours) | int4 (ours) |
|---|---|---|---|
| lm_head 262144×1536 | 1,193 µs | 635 µs | **378 µs (3.2x)** |
| mlp gate 6144×1536 | 28.1 µs | 7.3 µs | 7.3 µs |
| mlp down 1536×6144 | 10.3 µs | 15.5 µs | 15.3 µs ❌ |

- *Pitfall 1:* the "down" shape has few rows and a long K, so splitting by rows (BLOCK_N ≥ 4)
  creates only ~48 programs for 142 SMs. Fix: allow BLOCK_N = 1–2 and BLOCK_K = 512 in autotune.
- *Pitfall 2:* some numbers exceed the 864 GB/s peak. The microbenchmark re-reads one small matrix
  that stays in the L40S's **96 MB L2 cache**. Only whole-model decode numbers are honest.
- Prefill (T > 1) uses a dequantized bf16 copy on tensor cores. Decode (T = 1) reads only the
  packed bytes through our kernel.

### 17. Better accuracy tests (`aio/quality.py`)
- **Token agreement (teacher-forced).** Walk through the oracle's answers token by token and ask
  "what would you predict here?". One early near-tie can't cascade, because each position is judged
  on its own.
  - Ours uses a `/aio/teacher_forced` endpoint (through the real decode path).
  - vLLM uses `prompt_logprobs` (rank of each oracle token).
- **GSM8K-100.** 100 grade-school math problems, greedy decoding, final answer after `####`.
  This answers "did compression make it dumber?". Expect ±5 points of noise at n=100.

### 18. Milestone 2 results: 2.5x stock vLLM, and +55% over vLLM on the same 4-bit weights 🎉
All runs on an L40S, 1 user, ~1,100-token prompt and 256 output tokens. Agreement is teacher-forced
against the bf16 HF oracle (410 positions). GSM8K is the first 100 test problems, greedy.

| Config | Token agreement | GSM8K-100 | tok/s (1 user) | TTFT | Bytes/token (est.) |
|---|---|---|---|---|---|
| vLLM, bf16 | 99.5% | 73% | 128.4 | 33.9 ms | 4.6 GB |
| vLLM, Google QAT int4 | 91.2% | 68% | 210.8 | 34.1 ms | ~1.9 GB |
| ours, bf16 | 99.8% | 74% | 125.3 ⚠️ | 52.3 ms | 4.6 GB |
| ours, QAT int4 (lm_head bf16) | 91.2% | 65% | 239.5 | 40.1 ms | ~1.9 GB |
| ours, QAT int4 + lm_head int8 | 91.5% | 67% | 283.3 | 54.3 ms | ~1.5 GB |
| **ours, QAT int4 + lm_head int4** | **92.0%** | **70%** | **326.8** | 39.3 ms | **~1.3 GB** |

Takeaways:
- **On identical weights, our engine beats vLLM by 14%** (239.5 vs 210.8 tok/s). That's our Triton
  GEMV plus torch.compile fusion, against vLLM's Marlin kernels.
- **Quantizing the lm_head is the real win.** vLLM serves it in bf16, because the checkpoint says
  `ignore: [lm_head]`.
  - int8 lm_head: +18%. int4 lm_head: +36% on top of the QAT engine.
  - Final: **326.8 tok/s = 2.55x stock vLLM bf16, and 1.55x vLLM on the same QAT checkpoint.**
- **Accuracy:** quantizing the lm_head did **not** measurably hurt. All QAT variants score 91–92%
  agreement and 65–70% on GSM8K.
  - The QAT checkpoint itself is the accuracy cost (−4 to −9 GSM8K points vs bf16, near the ±5
    noise floor at n=100).
  - Our int4 lm_head adds nothing measurable on top of that.
- **Not done yet:**
  - 326.8 tok/s is only ~50% of the int4 speed of light (~655 tok/s at 1.3 GB/token).
  - TTFT is 39–54 ms vs vLLM's 34 ms. Our prefill is still uncompiled eager PyTorch with
    dequantized weights.
  - ⚠️ **Regression to investigate:** ours-bf16 dropped from 144.2 to 125.3 tok/s after the M2
    refactor, while vLLM-bf16 was stable (128.5 → 128.4).
- First-time startup was 10–15 min (Triton autotuning of 54 configs × every layer shape, plus
  Inductor compilation). It's cached in the `aio-compile-cache` volume now.

### 19. What it's worth in production (L40S at $1.95/h, single-stream serving)
| Engine | tok/s (1 user) | tokens per GPU-hour | $ per 1M output tokens |
|---|---|---|---|
| vLLM bf16 | 128 | 461k | $4.23 |
| vLLM QAT int4 | 211 | 760k | $2.57 |
| ours, int4 + int4 lm_head | 327 | 1.18M | **$1.66** |

- Example: 1B latency-sensitive output tokens per day. That's ~$4,230/day on vLLM bf16 vs ~$1,660/day on
  ours, a saving of ~$940k/year for one model.
- An agent doing 50 sequential steps × 300 tokens waits ~117 s on vLLM bf16 vs ~46 s on ours.
- **Honest caveat:** for batched bulk serving, vLLM QAT does 3,757 tok/s at 32 users (~$0.14/M).
  Our engine serves one request at a time, so it only wins in the latency regime until it has batching.

### 20. Ideas confirmed via Exa for the next speed round
- **Split-K GEMV for skinny shapes** (down_proj, o_proj): 1.2–3.5x reported in
  [arXiv 2402.00025](https://arxiv.org/abs/2402.00025) and a
  [W4A16 Triton write-up](https://subhadipmitra.com/blog/2026/w4a16-weight-only-gemm-triton/).
- **Bound sliding-window attention to its window.** vLLM fixed the same waste in one backend in
  [PR #49588](https://github.com/vllm-project/vllm/pull/49588). Our decode attends over a whole
  power-of-two bucket even for the 28 window-512 layers.
- **Load-once int4 unpack:** our kernel already does this.

## Milestone 3a: chasing the speed of light (single user)

### 21. Profile: where does one decode step go? (L40S, context 1,200, `aio/m3_profile.py`)
| Config | Engine-only (graph replay) | + per-token `.item()` sync | Kernels per step |
|---|---|---|---|
| bf16 | 7.67 ms (130 tok/s) | 7.71 ms | 630 |
| QAT int4 + int4 lm_head | **3.01 ms (332 tok/s)** | 3.31 ms (302 tok/s) | 595 |

Int4 step breakdown (ungraphed kernel time, 3.08 ms):
- **Our GEMV: 2.23 ms (72%).** At ~1.3 GB per token that's ~580 GB/s, i.e. ~67% of peak.
  This is the main target (split-K and tuning).
- **Attention: ~0.36 ms (12%):** cutlass bmm 0.25 ms + fused masked softmax 0.09 ms. Target: the
  ring buffer.
- RMSNorms, GELU·up, RoPE: ~0.4 ms (13%), already fused by Inductor.
- argmax + softcap over 262K logits: 0.05 ms.
- **Host sync: +0.3 ms per token (~10%)** from `.item()` after every token. Fix: multi-step decode
  on the device, syncing every N tokens.

Unresolved: bf16 engine-only time is 7.67 ms, vs the 6.93 ms TPOT measured in the M1 run. Most
likely different L40S hosts (we can't control placement), but the cause isn't proven. Not pursued,
because bf16 isn't our target path.

### 22. Three fixes from the profile, plus a bug the accuracy gate caught 🐛
1. **Ring-buffer KV for the 28 sliding-window layers.**
   - Their cache is now exactly 512 slots (slot = position mod 512).
   - Decode attention reads 512 keys instead of a whole power-of-two bucket (up to 4,096).
   - A `ring_pos` tensor records which position each slot holds, and the mask is built from it.
2. **Split-K in the int4 GEMV.**
   - Skinny matrices split their K dimension across 2–8 programs: down_proj 1536×6144 → 8,
     o_proj → 2–4, qkv → 2.
   - Each program writes fp32 partials, and `torch.compile` fuses the partial-sum into the next op.
3. **Multi-step on-device decode.**
   - The captured graph feeds its output token back as the next input and appends it to an
     on-GPU history buffer.
   - The server launches 4 steps and syncs once, removing ~0.3 ms per token of host round-trips.

**The bug.** The first ring-buffer run scored **token agreement 85.9% and GSM8K 17%** (was 99.8% / 74%).
- Cause: the decode mask was built from `ring_pos` **before** the current token's position was
  written into it. So in every sliding layer the new token could not attend to itself.
- The text still *looked* plausible at a glance. Only the gate exposed it.
- Fix: write `ring_pos` first, then build the mask.
- **Lesson:** this is exactly the failure mode Baseten warns about. An optimization that "works"
  and is fast, but quietly breaks the model. Run the accuracy gate on every change, no exceptions.

### 23. Milestone 3a results (L40S, same tests as M2)
| Config | Agreement | GSM8K-100 | tok/s (1 user) | TPOT | TTFT |
|---|---|---|---|---|---|
| ours bf16, M2 | 99.8% | 74% | 125.3 | 7.98 ms | 52.3 ms |
| ours bf16, **M3a** | 99.8% | 74% | 127.2 | 7.86 ms | **40.4 ms** |
| ours int4 + int4 lm_head, M2 | 92.0% | 70% | 326.8 | 3.06 ms | 39.3 ms |
| ours int4 + int4 lm_head, **M3a** | 91.2% | 70% | **342.7** | **2.92 ms** | 38.2 ms |

- Accuracy is unchanged after the fix. The ring buffer is exact, as it should be.
- **+4.9% decode speed** for int4 (326.8 → 342.7 tok/s). That's **1.63x vLLM QAT** (210.8) and
  **2.67x vLLM bf16**.
- bf16 TTFT dropped 23% (52 → 40 ms): prefill no longer touches a 4,096-row bucket.
- Smaller gain than hoped. The int4 GEMV is still ~70% of the step at roughly 2/3 of peak bandwidth.
  We're now at ~52% of the int4 speed of light (~655 tok/s).
  - The remaining gap is spread over ~200 GEMV launches per token, each with a fixed startup tail
    of a few µs.
  - Closing it needs bigger changes: fusing whole layers into one kernel ("megakernels"), or a
    Marlin-style weight layout.
  - That's diminishing returns for this project, so we move on to batching.

## Milestone 3b: continuous batching (many users at once)

### 24. Design (`aio/engine/batch.py`, `aio/engine/server_batch.py`)
Why it works: decode is bandwidth-bound, so with B users in one step the weights are read once
and used B times.

- **Slots.** Up to 32 sequences. Each slot owns its KV:
  - full-attention caches [32, 4096, hd]
  - sliding rings [32, 512, hd]
  - ring positions [32, 512]
- **Compaction.** Active sequences always occupy slots 0..B-1. When one finishes, the last one is
  copied into its slot (~25 MB copy, ~tens of µs), so a step simply processes rows [:B].
- **Prefill** reuses the single-sequence engine (already oracle-tested), then copies that KV into
  the slot.
- **Small-batch W4A16 GEMM kernel** (`_gemm_kernel`):
  - Each program dequantizes a [BLOCK_N, BLOCK_K] int4 tile once in registers.
  - It multiplies that tile with all B rows on tensor cores (`tl.dot`).
  - Split-K is reused for skinny shapes.
  - Dispatch: B = 1 uses the GEMV kernel; 2 ≤ B ≤ 64 uses this GEMM; prefill above 64 uses the
    dense bf16 copy.
- **Scheduler thread (continuous batching).** It loops:
  1. Admit waiting requests into free slots (prefill each, stream its first token).
  2. Run one batched decode chunk (4 tokens) for everyone.
  3. Stream the tokens, retire finished requests, compact the slots.
  New users join between chunks instead of waiting for others to finish.
- **CUDA graphs** per (padded batch size ∈ {1, 2, 4, 8, 16, 32}) × (full-attention bucket ∈ {512, …, 4096}).
  Padded rows write junk only into free slots. `torch.compile(dynamic=None)` compiles a
  shape-generic graph after the 2nd shape, instead of 24 separate compiles.
- **Accuracy checks reuse the gate:**
  - Teacher-forced agreement now runs through the *batched* path (batch of 1).
  - GSM8K sends 16 requests at a time, so it exercises real batching.
