"""Batched decode for many users at once (continuous batching).

Why batching wins: decode is limited by READING THE WEIGHTS (bandwidth), not by maths.
With B users in one step, the weights are read once and used B times -> the GPU produces
~B tokens for (almost) the price of one, until the maths or the KV cache become the limit.

Design:
  * MAX_B slots. Active sequences are always COMPACTED into slots 0..B-1 (when one finishes,
    the last one is moved into its slot), so a step simply processes rows [:B].
  * Each slot owns its KV: full-attention caches [MAX_B, max_len, hd], sliding rings
    [MAX_B, 512, hd] and ring positions [MAX_B, 512].
  * Prefill reuses the single-sequence engine (already tested against the oracle) and then
    copies that sequence's KV into its slot.
  * Decode step = the same math as the single engine with a batch dimension; CUDA graphs are
    captured per (padded batch size, full-attention bucket).
"""
import torch
import torch.nn.functional as F

from aio.engine.model import Gemma4E2B, rms_norm, rotate_half
from aio.engine.quant import linear

BATCH_SIZES = (1, 2, 4, 8, 16, 32)


class BatchedGemma4(Gemma4E2B):
    def __init__(self, *args, max_batch: int = 32, **kw):
        super().__init__(*args, **kw)
        self.max_b = max_batch
        dev, dt = self.device, self.dtype
        self.bk = [torch.zeros(max_batch, self.max_len if self.is_full[i] else self.window, self.head_dim[i],
                               dtype=dt, device=dev) for i in range(self.first_shared)]
        self.bv = [torch.zeros_like(c) for c in self.bk]
        self.bring_pos = torch.full((max_batch, self.window), -10**9, dtype=torch.long, device=dev)
        self.rows = torch.arange(max_batch, device=dev)
        # graph I/O buffers
        self.b_tok = torch.zeros(max_batch, dtype=torch.long, device=dev)
        self.b_pos = torch.zeros(max_batch, dtype=torch.long, device=dev)
        self.b_hist = torch.zeros(64, max_batch, dtype=torch.long, device=dev)
        self.b_cnt = torch.zeros(1, dtype=torch.long, device=dev)
        self.bgraphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.bstep_fn = self.bforward

    # ------------------------------------------------------------------ slots
    @torch.inference_mode()
    def prefill_into(self, slot: int, prompt_ids: list[int]) -> int:
        tok = self.prefill(prompt_ids)               # single-sequence path -> self.k_cache etc.
        T = len(prompt_ids)
        for i in range(self.first_shared):
            if self.is_full[i]:
                self.bk[i][slot, :T].copy_(self.k_cache[i][:T])
                self.bv[i][slot, :T].copy_(self.v_cache[i][:T])
            else:
                self.bk[i][slot].copy_(self.k_cache[i])
                self.bv[i][slot].copy_(self.v_cache[i])
        self.bring_pos[slot].copy_(self.ring_pos)
        return tok

    @torch.inference_mode()
    def move_slot(self, src: int, dst: int):
        for i in range(self.first_shared):
            self.bk[i][dst].copy_(self.bk[i][src])
            self.bv[i][dst].copy_(self.bv[i][src])
        self.bring_pos[dst].copy_(self.bring_pos[src])

    # ------------------------------------------------------------------ math
    def bforward(self, tok: torch.Tensor, pos: torch.Tensor, S: int) -> torch.Tensor:
        """One decode step for B rows (row b = slot b). tok, pos: [B]. Returns next tokens [B]."""
        W, eps, B = self.W, self.eps, tok.shape[0]
        rows = self.rows[:B]
        x = F.embedding(tok, W["embed_tokens.weight"]) * self.embed_scale
        ple_tok = (F.embedding(tok, W["embed_tokens_per_layer.weight"]) * self.ple_embed_scale
                   ).view(B, self.L, self.ple_dim)
        ple_ctx = linear(x, W["per_layer_model_projection.weight"]) * (self.hidden ** -0.5)
        ple_ctx = rms_norm(ple_ctx.view(B, self.L, self.ple_dim), W["per_layer_projection_norm.weight"], eps)
        ple = (ple_ctx + ple_tok) * (2.0 ** -0.5)

        rope = {f: self.rope(pos, f) for f in (False, True)}
        slot = pos % self.window
        self.bring_pos[rows, slot] = pos                         # record BEFORE building masks
        p = pos[:, None]
        rp = self.bring_pos[:B]
        masks = {True: self.kv_positions[:S][None, :] <= p,      # [B, S]
                 False: (rp <= p) & (rp > p - self.window)}      # [B, 512]

        for i in range(self.L):
            w, full, hd = self.layer_w[i], self.is_full[i], self.head_dim[i]
            cos, sin = rope[full]
            h = rms_norm(x, w["input_layernorm.weight"], eps)
            qkv = linear(h, w["qkv"])
            q = qkv[:, :self.n_heads * hd].view(B, self.n_heads, hd)
            q = rms_norm(q, w["self_attn.q_norm.weight"], eps)
            q = q * cos[:, None] + rotate_half(q) * sin[:, None]
            if i < self.first_shared:
                k = rms_norm(qkv[:, self.n_heads * hd:(self.n_heads + 1) * hd], w["self_attn.k_norm.weight"], eps)
                k = k * cos + rotate_half(k) * sin
                v = rms_norm(qkv[:, (self.n_heads + 1) * hd:], None, eps)
                idx = pos if full else slot
                self.bk[i][rows, idx] = k
                self.bv[i][rows, idx] = v
            src = self.kv_src[i]
            if full:
                K, V = self.bk[src][:B, :S], self.bv[src][:B, :S]
            else:
                K, V = self.bk[src][:B], self.bv[src][:B]
            scores = torch.einsum("bhd,bsd->bhs", q, K).float()
            scores = scores.masked_fill(~masks[full][:, None, :], float("-inf"))
            probs = torch.softmax(scores, dim=-1).to(self.dtype)
            a = torch.einsum("bhs,bsd->bhd", probs, V).reshape(B, self.n_heads * hd)
            x = x + rms_norm(linear(a, w["self_attn.o_proj.weight"]), w["post_attention_layernorm.weight"], eps)
            h = rms_norm(x, w["pre_feedforward_layernorm.weight"], eps)
            gate, up = linear(h, w["gate_up"]).chunk(2, dim=-1)
            h = linear(F.gelu(gate, approximate="tanh") * up, w["mlp.down_proj.weight"])
            x = x + rms_norm(h, w["post_feedforward_layernorm.weight"], eps)
            g = F.gelu(linear(x, w["per_layer_input_gate.weight"]), approximate="tanh") * ple[:, i]
            x = x + rms_norm(linear(g, w["per_layer_projection.weight"]), w["post_per_layer_input_norm.weight"], eps)
            x = x * w["layer_scalar"]

        h = rms_norm(x, W["norm.weight"], eps)
        logits = linear(h, self.lm_head)
        logits = torch.tanh(logits / self.softcap) * self.softcap
        return torch.argmax(logits, dim=-1)

    # ------------------------------------------------------------------ graphs
    def enable_compile(self):
        super().enable_compile()
        # dynamic=None: after the 2nd distinct (B, S) Dynamo compiles a shape-generic graph,
        # instead of one full compile for each of the 24 (batch size, bucket) combinations.
        torch._dynamo.config.cache_size_limit = 64
        self.bstep_fn = torch.compile(self.bforward, fullgraph=True, dynamic=None)

    def _bgraph_body(self, B: int, S: int):
        out = self.bstep_fn(self.b_tok[:B], self.b_pos[:B], S)
        self.b_hist.index_copy_(0, self.b_cnt, F.pad(out, (0, self.max_b - B))[None])
        self.b_tok[:B].copy_(out)
        self.b_pos[:B].add_(1)
        self.b_cnt.add_(1)

    @torch.inference_mode()
    def capture_batch_graphs(self, sizes=BATCH_SIZES, buckets=(512, 1024, 2048, 4096)):
        for B in sizes:
            for S in buckets:
                self.b_tok.zero_(); self.b_pos.zero_()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(2):
                        self.b_cnt.zero_(); self.b_pos.zero_()
                        self._bgraph_body(B, S)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                self.b_cnt.zero_(); self.b_pos.zero_()
                with torch.cuda.graph(g):
                    self._bgraph_body(B, S)
                self.bgraphs[(B, S)] = g
        self.bring_pos.fill_(-10**9)

    @staticmethod
    def pad_b(n: int) -> int:
        return next(b for b in BATCH_SIZES if b >= n)

    def _bbucket(self, pos: int) -> int:
        S = 512
        while S < pos + 1:
            S *= 2
        return min(S, self.max_len)

    @torch.inference_mode()
    def decode_batch(self, toks: list[int], poss: list[int], k: int) -> list[list[int]]:
        """Advance the first len(toks) slots by k tokens each, with ONE host sync.
        Returns k lists of len(toks) tokens."""
        n = len(toks)
        Bp = self.pad_b(n)
        self.b_tok[:n] = torch.tensor(toks, device=self.device)
        self.b_pos[:n] = torch.tensor(poss, device=self.device)
        self.b_tok[n:Bp] = 0
        self.b_pos[n:Bp] = 0           # padded rows write junk into FREE slots only
        self.b_cnt.zero_()
        top = max(poss)
        for j in range(k):
            self.bgraphs[(Bp, self._bbucket(top + j))].replay()
        return self.b_hist[:k, :n].tolist()
