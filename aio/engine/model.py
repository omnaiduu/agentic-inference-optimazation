"""Our own Gemma 4 E2B inference engine (text only, one user at a time).

Milestone 1 = E1 (correct) + E2 (fast via CUDA graphs).

Everything here is re-derived from the official transformers implementation
(transformers 5.20.2, models/gemma4/modeling_gemma4.py). The details that MUST match:

  1. Embedding is scaled by sqrt(hidden_size), cast to bf16 first (39.19 -> 39.25!).
  2. RMSNorm computes in float32 and multiplies by `weight` (not 1 + weight).
  3. q_norm / k_norm are RMSNorms over each head; v_norm has NO weight.
  4. Attention scale is 1.0 (the q/k norms already control magnitude), not 1/sqrt(d).
  5. Sliding layers: head_dim 256, normal RoPE (theta 1e4), window 512.
     Full layers:    head_dim 512, "proportional" RoPE (theta 1e6) rotating only 25%
                     of the dimensions (the rest get inv_freq = 0 -> no rotation).
  6. KV sharing: layers 15..34 compute no K/V. They reuse the K/V of the LAST
     non-shared layer of the same type (sliding -> layer 13, full -> layer 14).
  7. Per-Layer Embeddings (PLE): each layer gets a 256-d vector per token =
     (RMSNorm(proj(embeds) / sqrt(1536)) + PLE_table[token] * 16) / sqrt(2),
     injected after the MLP through a gate -> GELU -> multiply -> project -> norm.
  8. Each layer output is multiplied by a learned `layer_scalar`.
  9. lm_head = the embedding matrix (tied); logits soft-capped: 30 * tanh(x / 30).
"""
import glob
import json
import math

import torch
import torch.nn.functional as F
from safetensors import safe_open

from aio.engine.quant import QLinear, linear, quantize_rtn

EOS_IDS = (1, 106)  # <eos>, <turn|> (from generation_config / config eos_token_id)


def rms_norm(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    xf = x.float()
    out = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    if w is not None:
        out = out * w.float()
    return out.to(x.dtype)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class Gemma4E2B:
    def __init__(self, model_dir: str, max_len: int = 4096, device: str = "cuda",
                 quant_dir: str | None = None, lm_head_bits: int | None = None):
        """quant_dir: load Google's QAT int4 checkpoint (compressed-tensors) for all Linear layers.
        lm_head_bits: additionally quantize the lm_head (which the QAT checkpoint keeps in bf16)."""
        self.device = device
        self.max_len = max_len
        cfg = json.load(open(f"{model_dir}/config.json"))["text_config"]
        self.cfg = cfg
        self.L = cfg["num_hidden_layers"]                      # 35
        self.hidden = cfg["hidden_size"]                       # 1536
        self.n_heads = cfg["num_attention_heads"]              # 8 query heads, 1 KV head
        self.eps = cfg["rms_norm_eps"]
        self.window = cfg["sliding_window"]                    # 512
        self.ple_dim = cfg["hidden_size_per_layer_input"]      # 256
        self.softcap = cfg["final_logit_softcapping"]          # 30
        self.types = cfg["layer_types"]
        self.is_full = [t == "full_attention" for t in self.types]
        self.head_dim = [cfg["global_head_dim"] if f else cfg["head_dim"] for f in self.is_full]
        self.first_shared = self.L - cfg["num_kv_shared_layers"]  # 15
        # Which layer's KV cache each layer reads (detail 6).
        self.kv_src = [i if i < self.first_shared else
                       max(j for j in range(self.first_shared) if self.types[j] == self.types[i])
                       for i in range(self.L)]

        # ---- load weights (text decoder only; skip vision/audio towers) ----
        prefix = "model.language_model."
        self.W: dict = {}
        raw: dict[str, torch.Tensor] = {}
        for f in sorted(glob.glob(f"{quant_dir or model_dir}/*.safetensors")):
            with safe_open(f, "pt", device="cpu") as sf:
                for k in sf.keys():
                    if k.startswith(prefix) or k == "lm_head.weight":
                        raw[k.removeprefix(prefix)] = sf.get_tensor(k).to(device)
        for k, v in raw.items():
            if k.endswith(".weight_packed"):       # int4 layer -> QLinear
                base = k.removesuffix(".weight_packed")
                n_out, k_in = raw[base + ".weight_shape"].tolist()
                self.W[base + ".weight"] = QLinear(v, raw[base + ".weight_scale"], 4, k_in)
            elif not k.endswith((".weight_scale", ".weight_shape")):
                self.W[k] = v
        del raw
        self.dtype = self.W["embed_tokens.weight"].dtype
        lm = self.W.pop("lm_head.weight", None)
        self.lm_head = lm if lm is not None else self.W["embed_tokens.weight"]  # tied
        if lm_head_bits:
            packed, scale = quantize_rtn(self.lm_head, lm_head_bits)
            self.lm_head = QLinear(packed, scale, lm_head_bits, self.hidden)
        # Pre-transpose nothing: F.linear(x, W) computes x @ W.T just like nn.Linear.
        self.layer_w = [{k.split(f"layers.{i}.", 1)[1]: v for k, v in self.W.items()
                         if k.startswith(f"layers.{i}.")} for i in range(self.L)]
        # Fusion: one matmul instead of three (q,k,v) / two (gate,up). Fewer, bigger kernels.
        for i, w in enumerate(self.layer_w):
            if i < self.first_shared:
                parts = [w.pop("self_attn.q_proj.weight"), w.pop("self_attn.k_proj.weight"),
                         w.pop("self_attn.v_proj.weight")]
                w["qkv"] = QLinear.cat(parts) if isinstance(parts[0], QLinear) else torch.cat(parts)
            else:
                w["qkv"] = w.pop("self_attn.q_proj.weight")
            parts = [w.pop("mlp.gate_proj.weight"), w.pop("mlp.up_proj.weight")]
            w["gate_up"] = QLinear.cat(parts) if isinstance(parts[0], QLinear) else torch.cat(parts)
        for k in [k for k in self.W if k.startswith("layers.")]:
            del self.W[k]  # the per-layer dicts now own these tensors
        torch.cuda.empty_cache()

        # Detail 1: scales cast to the weight dtype exactly like the reference.
        self.embed_scale = torch.tensor(self.hidden ** 0.5, dtype=self.dtype, device=device)
        self.ple_embed_scale = torch.tensor(self.ple_dim ** 0.5, dtype=self.dtype, device=device)

        # ---- RoPE inverse frequencies (detail 5) ----
        def default_inv(dim, theta):
            return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))

        def proportional_inv(dim, theta, frac):
            n_rot = int(frac * dim // 2)
            rot = 1.0 / (theta ** (torch.arange(0, 2 * n_rot, 2, dtype=torch.float) / dim))
            return torch.cat([rot, torch.zeros(dim // 2 - n_rot)])

        rp = cfg["rope_parameters"]
        self.inv_freq = {
            False: default_inv(cfg["head_dim"], rp["sliding_attention"]["rope_theta"]).to(device),
            True: proportional_inv(cfg["global_head_dim"], rp["full_attention"]["rope_theta"],
                                   rp["full_attention"]["partial_rotary_factor"]).to(device),
        }

        # ---- static KV cache: only for the 15 non-shared layers, 1 KV head each ----
        # Sliding-window layers only ever look at the last 512 tokens, so their cache is a
        # 512-slot RING BUFFER (slot = position % 512). Decode attention then always reads
        # exactly 512 keys instead of a whole power-of-two bucket (up to 4096).
        self.k_cache = [torch.zeros(max_len if self.is_full[i] else self.window, self.head_dim[i],
                                    dtype=self.dtype, device=device)
                        for i in range(self.first_shared)]
        self.ring_pos = torch.full((self.window,), -10**9, dtype=torch.long, device=device)
        self.v_cache = [torch.zeros_like(c) for c in self.k_cache]
        self.kv_positions = torch.arange(max_len, device=device)

        # ---- CUDA graph state (E2) ----
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.static_tok = torch.zeros(1, dtype=torch.long, device=device)
        self.static_pos = torch.zeros(1, dtype=torch.long, device=device)
        self.static_out = torch.zeros(1, dtype=torch.long, device=device)
        # Multi-step decode: each graph replay feeds its output token back in as the next input and
        # appends it to out_hist ON THE GPU, so the CPU can launch N steps and sync only once.
        self.out_hist = torch.zeros(64, dtype=torch.long, device=device)
        self.static_cnt = torch.zeros(1, dtype=torch.long, device=device)
        self.step_fn = self.forward  # replaced by a torch.compile'd version in compile mode

    def enable_compile(self):
        """Let TorchInductor fuse the ~1,000 tiny elementwise kernels per step into a few dozen.
        (e.g. one RMSNorm = cast, square, mean, add, rsqrt, mul, mul, cast -> ONE kernel)."""
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 32
        self.step_fn = torch.compile(self.forward, fullgraph=True, dynamic=False)

    # ------------------------------------------------------------------ math
    def rope(self, positions: torch.Tensor, full: bool):
        freqs = positions[:, None].float() * self.inv_freq[full][None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    def attn_mask(self, positions: torch.Tensor, S: int, full: bool) -> torch.Tensor:
        """Boolean [T, S]: may query at position p look at key position k?"""
        k = self.kv_positions[:S][None, :]
        p = positions[:, None]
        allowed = k <= p
        if not full:
            allowed &= k > p - self.window
        return allowed

    def ring_mask(self, positions: torch.Tensor) -> torch.Tensor:
        """Boolean [T, 512] for decode over the ring: slot holds position ring_pos[s]."""
        k = self.ring_pos[None, :]
        p = positions[:, None]
        return (k <= p) & (k > p - self.window)

    def forward(self, ids: torch.Tensor, positions: torch.Tensor, S: int) -> torch.Tensor:
        """ids, positions: [T]. Writes K/V for these positions, attends over cache[:S].
        Prefill (T > 1) must start at position 0: sliding layers attend over the fresh K/V of
        the whole prompt and only the last 512 entries are written into the ring.
        Returns the next-token id predicted at the LAST position, shape [1]."""
        W, eps, T = self.W, self.eps, ids.shape[0]

        x = F.embedding(ids, W["embed_tokens.weight"]) * self.embed_scale            # [T, 1536]

        # Detail 7: per-layer inputs, computed once for all layers.
        ple_tok = (F.embedding(ids, W["embed_tokens_per_layer.weight"]) * self.ple_embed_scale
                   ).view(T, self.L, self.ple_dim)
        ple_ctx = linear(x, W["per_layer_model_projection.weight"]) * (self.hidden ** -0.5)
        ple_ctx = rms_norm(ple_ctx.view(T, self.L, self.ple_dim), W["per_layer_projection_norm.weight"], eps)
        ple = (ple_ctx + ple_tok) * (2.0 ** -0.5)                                     # [T, 35, 256]

        rope = {f: self.rope(positions, f) for f in (False, True)}
        prefill = T > 1
        if prefill:
            masks = {f: self.attn_mask(positions, T, f) for f in (False, True)}
        else:
            masks = {True: self.attn_mask(positions, S, True), False: self.ring_mask(positions)}
        tail = positions[-self.window:]                    # positions that land in the ring
        self.ring_pos.index_copy_(0, tail % self.window, tail)
        fresh = {}                                         # prefill: full-prompt K/V per source layer

        for i in range(self.L):
            w, full, hd = self.layer_w[i], self.is_full[i], self.head_dim[i]
            cos, sin = rope[full]

            # --- attention block ---
            h = rms_norm(x, w["input_layernorm.weight"], eps)
            qkv = linear(h, w["qkv"])
            q = qkv[:, :self.n_heads * hd].view(T, self.n_heads, hd)
            q = rms_norm(q, w["self_attn.q_norm.weight"], eps)
            q = q * cos[:, None] + rotate_half(q) * sin[:, None]
            if i < self.first_shared:
                k = qkv[:, self.n_heads * hd:(self.n_heads + 1) * hd]
                v = qkv[:, (self.n_heads + 1) * hd:]
                k = rms_norm(k, w["self_attn.k_norm.weight"], eps)
                k = k * cos + rotate_half(k) * sin
                v = rms_norm(v, None, eps)
                if full:
                    self.k_cache[i].index_copy_(0, positions, k)
                    self.v_cache[i].index_copy_(0, positions, v)
                else:
                    slots = tail % self.window
                    self.k_cache[i].index_copy_(0, slots, k[-self.window:])
                    self.v_cache[i].index_copy_(0, slots, v[-self.window:])
                if prefill:
                    fresh[i] = (k, v)
            src = self.kv_src[i]
            if prefill:
                K, V = fresh[src]                                                     # [T, hd]
            elif full:
                K, V = self.k_cache[src][:S], self.v_cache[src][:S]                   # [S, hd]
            else:
                K, V = self.k_cache[src], self.v_cache[src]                           # [512, hd] ring
            scores = torch.einsum("thd,sd->hts", q, K).float()                       # scale = 1.0
            scores = scores.masked_fill(~masks[full][None], float("-inf"))
            probs = torch.softmax(scores, dim=-1).to(self.dtype)
            a = torch.einsum("hts,sd->thd", probs, V).reshape(T, self.n_heads * hd)
            a = linear(a, w["self_attn.o_proj.weight"])
            x = x + rms_norm(a, w["post_attention_layernorm.weight"], eps)

            # --- MLP block (GELU-tanh gated) ---
            h = rms_norm(x, w["pre_feedforward_layernorm.weight"], eps)
            gate, up = linear(h, w["gate_up"]).chunk(2, dim=-1)
            h = linear(F.gelu(gate, approximate="tanh") * up, w["mlp.down_proj.weight"])
            x = x + rms_norm(h, w["post_feedforward_layernorm.weight"], eps)

            # --- per-layer embedding injection ---
            g = F.gelu(linear(x, w["per_layer_input_gate.weight"]), approximate="tanh") * ple[:, i]
            g = linear(g, w["per_layer_projection.weight"])
            x = x + rms_norm(g, w["post_per_layer_input_norm.weight"], eps)

            x = x * w["layer_scalar"]

        # --- final norm + tied lm_head + soft-cap, only for the last position ---
        h = rms_norm(x[-1:], W["norm.weight"], eps)
        logits = linear(h, self.lm_head)
        logits = torch.tanh(logits / self.softcap) * self.softcap
        return torch.argmax(logits, dim=-1)

    # ------------------------------------------------------------ public API
    @torch.inference_mode()
    def prefill(self, prompt_ids: list[int]) -> int:
        T = len(prompt_ids)
        self.ring_pos.fill_(-10**9)                        # new request: forget the old ring
        ids = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
        pos = torch.arange(T, device=self.device)
        return int(self.forward(ids, pos, S=T).item())

    def _bucket(self, pos: int) -> int:
        """Attend over a cache prefix rounded up to a power of two (static shape for CUDA graphs)."""
        S = 256
        while S < pos + 1:
            S *= 2
        return min(S, self.max_len)

    @torch.inference_mode()
    def decode_eager(self, tok: int, pos: int) -> int:
        ids = torch.tensor([tok], device=self.device)
        p = torch.tensor([pos], device=self.device)
        return int(self.forward(ids, p, S=self._bucket(pos)).item())

    def _graph_body(self, S: int):
        out = self.step_fn(self.static_tok, self.static_pos, S)
        self.static_out.copy_(out)
        self.out_hist.index_copy_(0, self.static_cnt, out)
        self.static_tok.copy_(out)       # next step's input = this step's output (stays on GPU)
        self.static_pos.add_(1)
        self.static_cnt.add_(1)

    @torch.inference_mode()
    def capture_graphs(self):
        """E2: record one decode step per cache bucket as a CUDA graph.

        Without a graph, every decode step makes the CPU launch ~1,000 tiny GPU
        kernels one by one; at batch size 1 that launch overhead dominates.
        A graph records them once and replays the whole step with one launch."""
        S = 256
        while True:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):  # warm up allocator / kernels outside the graph
                    self.static_cnt.zero_()
                    self._graph_body(S)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._graph_body(S)
            self.graphs[S] = g
            if S >= self.max_len:
                break
            S *= 2

    @torch.inference_mode()
    def decode_graph(self, tok: int, pos: int) -> int:
        self.static_tok.fill_(tok)
        self.static_pos.fill_(pos)
        self.static_cnt.zero_()
        self.graphs[self._bucket(pos)].replay()
        return int(self.static_out.item())

    @torch.inference_mode()
    def decode_many(self, tok: int, pos: int, n: int) -> list[int]:
        """Generate n tokens with ONE host sync (instead of one per token)."""
        self.static_tok.fill_(tok)
        self.static_pos.fill_(pos)
        self.static_cnt.zero_()
        for i in range(n):
            self.graphs[self._bucket(pos + i)].replay()
        return self.out_hist[:n].tolist()
