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
    def __init__(self, model_dir: str, max_len: int = 4096, device: str = "cuda"):
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
        self.W: dict[str, torch.Tensor] = {}
        for f in sorted(glob.glob(f"{model_dir}/*.safetensors")):
            with safe_open(f, "pt", device="cpu") as sf:
                for k in sf.keys():
                    if k.startswith(prefix):
                        self.W[k[len(prefix):]] = sf.get_tensor(k).to(device)
        self.dtype = self.W["embed_tokens.weight"].dtype
        # Pre-transpose nothing: F.linear(x, W) computes x @ W.T just like nn.Linear.
        self.layer_w = [{k.split(f"layers.{i}.", 1)[1]: v for k, v in self.W.items()
                         if k.startswith(f"layers.{i}.")} for i in range(self.L)]

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
        self.k_cache = [torch.zeros(max_len, self.head_dim[i], dtype=self.dtype, device=device)
                        for i in range(self.first_shared)]
        self.v_cache = [torch.zeros_like(c) for c in self.k_cache]
        self.kv_positions = torch.arange(max_len, device=device)

        # ---- CUDA graph state (E2) ----
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.static_tok = torch.zeros(1, dtype=torch.long, device=device)
        self.static_pos = torch.zeros(1, dtype=torch.long, device=device)
        self.static_out = torch.zeros(1, dtype=torch.long, device=device)

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

    def forward(self, ids: torch.Tensor, positions: torch.Tensor, S: int) -> torch.Tensor:
        """ids, positions: [T]. Writes K/V for these positions, attends over cache[:S].
        Returns the next-token id predicted at the LAST position, shape [1]."""
        W, eps, T = self.W, self.eps, ids.shape[0]

        x = F.embedding(ids, W["embed_tokens.weight"]) * self.embed_scale            # [T, 1536]

        # Detail 7: per-layer inputs, computed once for all layers.
        ple_tok = (F.embedding(ids, W["embed_tokens_per_layer.weight"]) * self.ple_embed_scale
                   ).view(T, self.L, self.ple_dim)
        ple_ctx = F.linear(x, W["per_layer_model_projection.weight"]) * (self.hidden ** -0.5)
        ple_ctx = rms_norm(ple_ctx.view(T, self.L, self.ple_dim), W["per_layer_projection_norm.weight"], eps)
        ple = (ple_ctx + ple_tok) * (2.0 ** -0.5)                                     # [T, 35, 256]

        rope = {f: self.rope(positions, f) for f in (False, True)}
        masks = {f: self.attn_mask(positions, S, f) for f in (False, True)}

        for i in range(self.L):
            w, full, hd = self.layer_w[i], self.is_full[i], self.head_dim[i]
            cos, sin = rope[full]

            # --- attention block ---
            h = rms_norm(x, w["input_layernorm.weight"], eps)
            q = F.linear(h, w["self_attn.q_proj.weight"]).view(T, self.n_heads, hd)
            q = rms_norm(q, w["self_attn.q_norm.weight"], eps)
            q = q * cos[:, None] + rotate_half(q) * sin[:, None]
            if i < self.first_shared:
                k = F.linear(h, w["self_attn.k_proj.weight"]).view(T, hd)
                v = F.linear(h, w["self_attn.v_proj.weight"]).view(T, hd)
                k = rms_norm(k, w["self_attn.k_norm.weight"], eps)
                k = k * cos + rotate_half(k) * sin
                v = rms_norm(v, None, eps)
                self.k_cache[i].index_copy_(0, positions, k)
                self.v_cache[i].index_copy_(0, positions, v)
            K = self.k_cache[self.kv_src[i]][:S]                                      # [S, hd]
            V = self.v_cache[self.kv_src[i]][:S]
            scores = torch.einsum("thd,sd->hts", q.float(), K.float())               # scale = 1.0
            scores = scores.masked_fill(~masks[full][None], float("-inf"))
            probs = torch.softmax(scores, dim=-1).to(self.dtype)
            a = torch.einsum("hts,sd->thd", probs, V).reshape(T, self.n_heads * hd)
            a = F.linear(a, w["self_attn.o_proj.weight"])
            x = x + rms_norm(a, w["post_attention_layernorm.weight"], eps)

            # --- MLP block (GELU-tanh gated) ---
            h = rms_norm(x, w["pre_feedforward_layernorm.weight"], eps)
            h = F.linear(F.gelu(F.linear(h, w["mlp.gate_proj.weight"]), approximate="tanh")
                         * F.linear(h, w["mlp.up_proj.weight"]), w["mlp.down_proj.weight"])
            x = x + rms_norm(h, w["post_feedforward_layernorm.weight"], eps)

            # --- per-layer embedding injection ---
            g = F.gelu(F.linear(x, w["per_layer_input_gate.weight"]), approximate="tanh") * ple[:, i]
            g = F.linear(g, w["per_layer_projection.weight"])
            x = x + rms_norm(g, w["post_per_layer_input_norm.weight"], eps)

            x = x * w["layer_scalar"]

        # --- final norm + tied lm_head + soft-cap, only for the last position ---
        h = rms_norm(x[-1:], W["norm.weight"], eps)
        logits = F.linear(h, W["embed_tokens.weight"])
        logits = torch.tanh(logits / self.softcap) * self.softcap
        return torch.argmax(logits, dim=-1)

    # ------------------------------------------------------------ public API
    @torch.inference_mode()
    def prefill(self, prompt_ids: list[int]) -> int:
        T = len(prompt_ids)
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
                    self.static_out.copy_(self.forward(self.static_tok, self.static_pos, S))
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.static_out.copy_(self.forward(self.static_tok, self.static_pos, S))
            self.graphs[S] = g
            if S >= self.max_len:
                break
            S *= 2

    @torch.inference_mode()
    def decode_graph(self, tok: int, pos: int) -> int:
        self.static_tok.fill_(tok)
        self.static_pos.fill_(pos)
        self.graphs[self._bucket(pos)].replay()
        return int(self.static_out.item())
