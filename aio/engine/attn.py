"""Decode attention kernel (Triton): flash-decoding for a batch of users, 8 query heads : 1 KV head.

Why: the generic path (einsum -> mask -> softmax -> einsum) pads every user's context to a
power-of-two bucket (e.g. 2,048 keys when the user has 1,200), materializes the score matrix in
memory, and runs 3+ kernels per layer. This kernel:
  * packs the 8 query heads that share the one KV head into the M dimension of tl.dot (padded to
    16 rows), so each K/V tile is loaded ONCE for all 8 heads (the GQA trick, as in vLLM's
    triton_unified_attention),
  * reads only the keys a user really has (0..pos for full layers; the 512-slot ring for sliding
    layers, masked by the positions stored in the ring),
  * splits long caches over several programs (flash-decoding, Dao et al. 2023) and merges the
    partial results with their log-sum-exp.
Scale is 1.0 (Gemma 4 normalizes q and k instead).
"""
import torch
import triton
import triton.language as tl

HP = 16  # query heads padded to the minimum tl.dot tile height


@triton.jit
def _decode_attn_kernel(q_ptr, k_ptr, v_ptr, pos_ptr, ring_ptr, o_ptr, lse_ptr,
                        slot_stride, ring_stride, L, WINDOW,
                        H: tl.constexpr, HPAD: tl.constexpr, HD: tl.constexpr, RING: tl.constexpr,
                        NSPLIT: tl.constexpr, BLOCK_N: tl.constexpr):
    b = tl.program_id(0)
    sp = tl.program_id(1)
    p = tl.load(pos_ptr + b)
    if RING:
        n = L
    else:
        n = p + 1
    per = tl.cdiv(tl.cdiv(n, NSPLIT), BLOCK_N) * BLOCK_N
    start = sp * per
    end = tl.minimum(start + per, n)

    rh = tl.arange(0, HPAD)
    rd = tl.arange(0, HD)
    hmask = rh < H
    q = tl.load(q_ptr + (b * H + rh[:, None]) * HD + rd[None, :], mask=hmask[:, None], other=0.0)
    m_i = tl.full((HPAD,), float("-inf"), tl.float32)
    l_i = tl.zeros((HPAD,), tl.float32)
    acc = tl.zeros((HPAD, HD), tl.float32)
    kbase = k_ptr + b * slot_stride
    vbase = v_ptr + b * slot_stride
    for j0 in range(start, end, BLOCK_N):
        rj = j0 + tl.arange(0, BLOCK_N)
        jm = rj < end
        k = tl.load(kbase + rj[:, None] * HD + rd[None, :], mask=jm[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k))                                    # [HPAD, BLOCK_N] fp32
        if RING:
            rp = tl.load(ring_ptr + b * ring_stride + rj, mask=jm, other=-1000000000)
            valid = jm & (rp <= p) & (rp > p - WINDOW)
        else:
            valid = jm
        s = tl.where(valid[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        pm = tl.exp(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(pm, 1)
        v = tl.load(vbase + rj[:, None] * HD + rd[None, :], mask=jm[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(pm.to(tl.bfloat16), v)    # probs in bf16, like the reference
        m_i = m_new
    empty = l_i == 0.0
    out = acc / tl.where(empty, 1.0, l_i)[:, None]
    lse = tl.where(empty, float("-inf"), m_i + tl.log(tl.where(empty, 1.0, l_i)))
    ob = (b * NSPLIT + sp) * H
    tl.store(o_ptr + (ob + rh[:, None]) * HD + rd[None, :], out, mask=hmask[:, None])
    tl.store(lse_ptr + ob + rh, lse, mask=hmask)


def decode_attention(q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, pos: torch.Tensor,
                     ring_pos: torch.Tensor | None = None, window: int = 512) -> torch.Tensor:
    """q: [B, H, HD] (row b = slot b), K/V: [n_slots, L, HD] (whole cache of one layer),
    pos: [B] current position of each row. ring_pos: [n_slots, L] for sliding (ring) layers.
    Returns [B, H*HD] bf16."""
    B, H, HD = q.shape
    ring = ring_pos is not None
    L = K.shape[1]
    if ring:
        nsplit = max(1, min(8, 64 // B))
    else:
        nsplit = max(1, min(16, 128 // B))
    o = torch.empty((B, nsplit, H, HD), dtype=torch.float32, device=q.device)
    lse = torch.empty((B, nsplit, H), dtype=torch.float32, device=q.device)
    _decode_attn_kernel[(B, nsplit)](
        q.contiguous(), K, V, pos, ring_pos if ring else pos, o, lse,
        K.stride(0), ring_pos.stride(0) if ring else 0, L, window,
        H=H, HPAD=HP, HD=HD, RING=ring, NSPLIT=nsplit, BLOCK_N=64 if HD <= 256 else 32,
        num_warps=4, num_stages=2)
    m = lse.amax(1, keepdim=True)
    w = torch.exp(lse - m)                                             # [B, nsplit, H]
    out = (o * w[..., None]).sum(1) / w.sum(1)[..., None]
    return out.to(torch.bfloat16).reshape(B, H * HD)
