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
- Fix: use the **official image** `vllm/vllm-openai:v0.33.0` (CUDA 13.0.3, released 2026-11-02,
  confirmed via the [release notes](https://github.com/vllm-project/vllm/releases/)).
  We use the same image for our engine, so both sides run identical torch/CUDA.
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
