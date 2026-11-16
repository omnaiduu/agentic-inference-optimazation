# Concepts, in plain English

These are terms we've used so far. We add new ones as they come up.

**Inference engine.** The program that runs a trained model and produces its answers.
Examples: vLLM, SGLang, TensorRT-LLM.
- They are "general" engines: they support hundreds of models on many chips.
- A custom engine supports only one model on one chip, so it can drop all that extra code.

**Token.** A piece of text, roughly ¾ of a word. The model produces text one token at a time.

**Prefill vs decode.** These are the two phases of answering a request.
- **Prefill:** the model reads your whole prompt at once. This is fast and lots of maths in parallel.
- **Decode:** the model writes the answer one token at a time. This is the slow part, because each
  new token needs a full pass through the model.

**TTFT (time to first token).** How long you wait before the answer starts appearing.
Mostly prefill time plus queueing.

**TPOT (time per output token).** The time between tokens once the answer is streaming.
1 / TPOT = tokens per second for one user.

**Throughput.** Total tokens per second across all users at once.
- Batching many users together raises throughput, because the weights are read once and used for
  everyone.
- The trade-off: each individual user gets slightly slower.

**Concurrency.** How many requests are in progress at the same time during a benchmark.

**Memory bandwidth.** How many bytes per second the GPU can move from its memory to its
compute units.
- For one user, decode speed is almost entirely limited by this, not by FLOPs.
- Picture a chef (the compute) who is fast, waiting on a slow conveyor belt of ingredients
  (the memory).

**Speed of light (SOL).** The theoretical best: bandwidth ÷ bytes read per token.
- Real engines reach maybe 50–85% of it.
- The gap between "now" and SOL is the room left to optimize.

**BF16 / FP8 / 4-bit.** Number formats for the weights.
- BF16 = 2 bytes per number, FP8 = 1 byte, 4-bit = ½ byte.
- Fewer bytes means less to read, which means faster decode.
- The risk is lower accuracy, which is why we need an oracle.

**KV cache.** The model's memory of the conversation so far.
- For each past token, the model stores "keys" and "values" so it doesn't have to recompute them.
- Its size grows with context length × number of users.

**Sliding-window attention.** A layer that only looks at the last N tokens (512 for Gemma 4 E2B)
instead of the whole conversation.
- It makes the layer cheaper and its KV cache smaller.
- Gemma 4 E2B: 28 sliding layers and 7 full layers.

**Shared KV cache (Gemma 4).** The last 20 layers reuse the keys and values of earlier layers
instead of computing their own. This saves compute and memory.

**PLE (Per-Layer Embeddings).** Gemma's trick for small models.
- Each layer gets a small extra "hint" vector per token, from a huge lookup table (4.7 GB).
- Only one row is read per token, so it's almost free at runtime.
- That's why "E2B" means 2B **effective** parameters, even though the checkpoint is 5B.

**lm_head.** The last layer. It scores every word in the vocabulary (262,144 of them) to choose
the next token. It's big, and it's read in full for every token.

**Oracle / accuracy gate.** A trusted reference output that the fast engine must match.
Without it, "optimizations" can quietly break the model.

**Greedy decoding.** Always pick the single most likely next token (temperature 0).
It's deterministic, so outputs can be compared exactly.

**CUDA graphs.** A way to record the GPU's whole sequence of work for one step once, then replay it.
- Without them, the CPU launches hundreds of small GPU jobs per token.
- At batch size 1 that launch overhead can be bigger than the actual work.

**Speculative decoding.** A small, fast "draft" model guesses the next few tokens. The big model
checks all of them in one pass.
- If the guesses are right, you get several tokens for the price of one.
- It works best on predictable text.
