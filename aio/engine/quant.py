"""Low-bit weights + our first custom GPU kernel (Triton).

Why: at batch size 1, decoding is limited by how many BYTES of weights the GPU
reads per token. bf16 = 2 bytes/weight. int4 = 0.5 byte/weight (+ a bf16 scale
per group of 32 weights = +0.0625) -> ~3.6x fewer bytes -> up to ~3.6x faster.

Storage format (compressed-tensors "pack-quantized", same as Google's QAT checkpoint):
  - symmetric integers q in [-2^(b-1), 2^(b-1)-1], stored with +2^(b-1) offset as unsigned
  - packed along the input dim K: 32/b values per int32, value i at bits [b*i, b*i+b)
  - one bf16 scale per (output row, group of 32 inputs):  w = q * scale
"""
import torch
import triton
import triton.language as tl

GROUP = 32


def unpack(packed: torch.Tensor, bits: int, K: int) -> torch.Tensor:
    """int32 [N, K*bits/32] -> int8 [N, K] (signed)."""
    per = 32 // bits
    shifts = torch.arange(per, device=packed.device, dtype=torch.int32) * bits
    q = (packed[:, :, None] >> shifts) & ((1 << bits) - 1)
    return (q.reshape(packed.shape[0], -1)[:, :K] - (1 << (bits - 1))).to(torch.int8)


def pack(q: torch.Tensor, bits: int) -> torch.Tensor:
    """int8 [N, K] (signed) -> int32 [N, K*bits/32]."""
    per = 32 // bits
    u = (q.to(torch.int32) + (1 << (bits - 1))).reshape(q.shape[0], -1, per)
    shifts = torch.arange(per, device=q.device, dtype=torch.int32) * bits
    return (u << shifts).sum(-1, dtype=torch.int32)


def quantize_rtn(w: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Round-to-nearest symmetric group quantization of a bf16 [N, K] weight."""
    N, K = w.shape
    g = w.float().view(N, K // GROUP, GROUP)
    qmax = (1 << (bits - 1)) - 1
    scale = (g.abs().amax(-1, keepdim=True) / (qmax + 0.5)).clamp(min=1e-8)
    q = torch.round(g / scale).clamp(-qmax - 1, qmax).to(torch.int8).view(N, K)
    return pack(q, bits), scale.squeeze(-1).to(torch.bfloat16)


class QLinear:
    """A low-bit linear layer. Decode (1 token) uses the Triton GEMV kernel and reads only the
    packed bytes. Prefill (many tokens) uses a dequantized bf16 copy on tensor cores."""

    def __init__(self, packed: torch.Tensor, scale: torch.Tensor, bits: int, K: int):
        self.packed, self.scale, self.bits = packed.contiguous(), scale.contiguous(), bits
        self.N, self.K = packed.shape[0], K
        q = unpack(self.packed, bits, K).float().view(self.N, K // GROUP, GROUP)
        self.dense = (q * self.scale.float()[:, :, None]).view(self.N, K).to(torch.bfloat16)

    @staticmethod
    def cat(layers: list["QLinear"]) -> "QLinear":
        out = QLinear.__new__(QLinear)
        out.packed = torch.cat([l.packed for l in layers]).contiguous()
        out.scale = torch.cat([l.scale for l in layers]).contiguous()
        out.dense = torch.cat([l.dense for l in layers])
        out.bits, out.K, out.N = layers[0].bits, layers[0].K, out.packed.shape[0]
        return out

    def nbytes_decode(self) -> int:
        return self.packed.numel() * 4 + self.scale.numel() * 2


def linear(x: torch.Tensor, w) -> torch.Tensor:
    if isinstance(w, torch.Tensor):
        return torch.nn.functional.linear(x, w)
    if x.shape[0] == 1:
        return gemv(x, w)
    return torch.nn.functional.linear(x, w.dense)


def _prune(configs, named_args, **kwargs):
    """Drop tile shapes that do not divide K (e.g. BLOCK_K=512 for the K=256 PLE projection)."""
    K = named_args["K"]
    return [c for c in configs if K % c.kwargs["BLOCK_K"] == 0] or configs[:1]


@triton.autotune(
    prune_configs_by={"early_config_prune": _prune},
    configs=[triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw)
             for bn in (1, 2, 4, 8, 16, 32) for bk in (128, 256, 512) for nw in (1, 2, 4)],
    key=["N", "K", "BITS"],
)
@triton.jit
def _gemv_kernel(x_ptr, wp_ptr, s_ptr, y_ptr, N, K,
                 BITS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    PER: tl.constexpr = 32 // BITS                # values per int32
    MASK: tl.constexpr = (1 << BITS) - 1
    OFF: tl.constexpr = 1 << (BITS - 1)
    G: tl.constexpr = 32                          # group size
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)    # output rows handled by this program
    nmask = rn < N
    KW = K // PER                                 # int32 words per row
    KG = K // G                                   # groups per row
    shifts = tl.arange(0, PER) * BITS
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        # 1) packed weights: [BLOCK_N, BLOCK_K/PER] int32 -> unpack to [BLOCK_N, BLOCK_K]
        rw = k0 // PER + tl.arange(0, BLOCK_K // PER)
        wp = tl.load(wp_ptr + rn[:, None] * KW + rw[None, :], mask=nmask[:, None], other=0)
        q = (wp[:, :, None] >> shifts[None, None, :]) & MASK
        q = tl.reshape(q, (BLOCK_N, BLOCK_K // G, G)).to(tl.float32) - OFF
        # 2) activations for this K block, viewed as groups
        x = tl.load(x_ptr + k0 + tl.arange(0, BLOCK_K)).to(tl.float32)
        x = tl.reshape(x, (1, BLOCK_K // G, G))
        # 3) per-group dot products, then scale once per group
        part = tl.sum(q * x, axis=2)                                          # [BLOCK_N, groups]
        rg = k0 // G + tl.arange(0, BLOCK_K // G)
        s = tl.load(s_ptr + rn[:, None] * KG + rg[None, :], mask=nmask[:, None], other=0).to(tl.float32)
        acc += tl.sum(part * s, axis=1)
    tl.store(y_ptr + rn, acc.to(tl.bfloat16), mask=nmask)


def gemv(x: torch.Tensor, w: QLinear) -> torch.Tensor:
    """y[1, N] = x[1, K] @ W^T with W stored in `bits`-bit groups. Reads ~bits/16 of bf16 bytes."""
    y = torch.empty((1, w.N), dtype=torch.bfloat16, device=x.device)
    grid = lambda meta: (triton.cdiv(w.N, meta["BLOCK_N"]),)
    _gemv_kernel[grid](x, w.packed, w.scale, y, w.N, w.K, BITS=w.bits)
    return y
