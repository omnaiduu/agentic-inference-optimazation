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


# FP8 for big (prompt-chunk) matmuls: the L40S does 733 dense FP8 TFLOPS vs 362 for BF16.
# Row-wise scales: one per activation row (token) and one per weight row (output channel);
# torch._scaled_mm supports this on sm89 since PyTorch 2.7 (pytorch PR #145676).
# Measured on the L40S (aio/m3c_bench.py --parts fp8, 512 rows): row-wise FP8 is 2.5x SLOWER than
# BF16 without fast accumulation and ~1.2-1.6x faster with it; small matrices stay faster in BF16.
FP8_PREFILL = False
F8 = torch.float8_e4m3fn
F8_MAX = 448.0
FP8_MIN_NK = 4096 * 1536            # only matrices at least this big get an FP8 copy


def make_fp8(w: "QLinear"):
    if w.N * w.K < FP8_MIN_NK:
        return
    d = w.dense.float()
    w.s8 = (d.abs().amax(1).clamp(min=1e-8) / F8_MAX).contiguous()          # [N]
    w.w8 = (d / w.s8[:, None]).to(F8)                                        # [N, K] row-major


def linear_fp8(x: torch.Tensor, w: "QLinear") -> torch.Tensor:
    xf = x.float()
    sx = xf.abs().amax(1, keepdim=True).clamp(min=1e-8) / F8_MAX             # [M, 1]
    xq = (xf / sx).to(F8)
    return torch._scaled_mm(xq, w.w8.t(), scale_a=sx, scale_b=w.s8[None, :], out_dtype=torch.bfloat16,
                            use_fast_accum=True)


# Marlin (IST-DASLab, arXiv 2408.11743), the hand-written CUDA int4 kernel that ships inside vLLM.
# Measured at M = 32 users on the L40S with weights streamed from memory (aio/m3e_bench.py): Marlin
# 3.01 ms per decode step vs 3.6-3.9 ms for every Triton variant we tried (magic-number dequant,
# split-K sweep, tl.range pipelining). It wins through an offline weight layout that matches the
# tensor-core fragments, and cp.async multi-stage pipelining. Our checkpoint packs 8 int4 values
# per int32 along K as [N, K/8] (+8 offset) = GPTQ's [K/8, N] layout transposed, so it converts.
MARLIN_MIN_M = 2          # rows from which Marlin is used for every matrix (benchmarks at M = 1 / 8 / 32)
MARLIN_M1_MAX_K = 1536    # at M = 1 Marlin wins only on K = hidden (qkv, gate_up, lm_head); our GEMV on o/down
SMALL_DENSE_NK = 1 << 20  # matrices smaller than this (the PLE gate/proj) run bf16 cuBLAS at M <= 64
_marlin = {}


def make_marlin(w: "QLinear"):
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
    from vllm.scalar_type import scalar_types
    if w.N * w.K < SMALL_DENSE_NK:
        w.small_dense = True
        return
    if w.bits != 4 or w.N % 64 or w.K % 128:
        return
    if not _marlin:
        _marlin.update(mu=mu, wtype=scalar_types.uint4b8,
                       ws=mu.marlin_make_workspace_new(w.packed.device),
                       zp=torch.empty(0, dtype=torch.int32, device=w.packed.device))
    w.mq = ops.gptq_marlin_repack(w.packed.t().contiguous(), w.K, w.N, 4)
    w.ms = mu.marlin_permute_scales(w.scale.t().contiguous(), w.K, w.N, GROUP)


def linear_marlin(x: torch.Tensor, w: "QLinear") -> torch.Tensor:
    m = _marlin
    return m["mu"].apply_gptq_marlin_linear(
        input=x, weight=w.mq, weight_scale=w.ms, weight_zp=m["zp"], workspace=m["ws"], wtype=m["wtype"],
        output_size_per_partition=w.N, input_size_per_partition=w.K)


def linear(x: torch.Tensor, w) -> torch.Tensor:
    if isinstance(w, torch.Tensor):
        return torch.nn.functional.linear(x, w)
    M = x.shape[0]
    if M <= 64:
        if getattr(w, "small_dense", False):      # PLE gate/proj: 0.2 MB, cuBLAS bf16 wins at every M
            return torch.nn.functional.linear(x, w.dense)
        if getattr(w, "mq", None) is not None and (M >= MARLIN_MIN_M or w.K <= MARLIN_M1_MAX_K):
            return linear_marlin(x, w)
    if M == 1:
        return gemv(x, w)
    if M <= 64:
        return gemm_small_m(x, w)
    if FP8_PREFILL and getattr(w, "w8", None) is not None and x.shape[0] % 16 == 0:
        return linear_fp8(x, w)
    return torch.nn.functional.linear(x, w.dense)


def _prune(configs, named_args, **kwargs):
    """Drop tile shapes that do not divide this program's K slice (e.g. BLOCK_K=512 when K=256)."""
    KS = named_args["KS"]
    return [c for c in configs if KS % c.kwargs["BLOCK_K"] == 0] or configs[:1]


def split_for(N: int, K: int) -> int:
    """Split-K factor. A skinny matrix (few output rows, long K) gives too few programs to keep
    all SMs busy, so each row's dot product is cut into SPLIT slices summed afterwards.
    (PyTorch/IBM arXiv 2402.00025; vLLM uses the same idea in FlashDecoding.)"""
    if N >= 4096:
        return 1
    for split in (8, 4, 2):
        if K >= 768 * split and K % (split * 128) == 0:
            return split
    return 1


@triton.autotune(
    prune_configs_by={"early_config_prune": _prune},
    configs=[triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw)
             for bn in (1, 2, 4, 8, 16, 32) for bk in (128, 256, 512) for nw in (1, 2, 4)],
    key=["N", "K", "KS", "BITS"],
)
@triton.jit
def _gemv_kernel(x_ptr, wp_ptr, s_ptr, y_ptr, N, K, KS,
                 BITS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Program (pid_n, pid_k) computes rows [pid_n*BLOCK_N, ...) over K slice [pid_k*KS, (pid_k+1)*KS)
    and writes an fp32 partial sum to y[pid_k, rows]."""
    PER: tl.constexpr = 32 // BITS                # values per int32
    MASK: tl.constexpr = (1 << BITS) - 1
    OFF: tl.constexpr = 1 << (BITS - 1)
    G: tl.constexpr = 32                          # group size
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)    # output rows handled by this program
    nmask = rn < N
    KW = K // PER                                 # int32 words per row
    KG = K // G                                   # groups per row
    shifts = tl.arange(0, PER) * BITS
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(pid_k * KS, (pid_k + 1) * KS, BLOCK_K):
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
    tl.store(y_ptr + pid_k * N + rn, acc, mask=nmask)


def gemv(x: torch.Tensor, w: QLinear) -> torch.Tensor:
    """y[1, N] = x[1, K] @ W^T with W stored in `bits`-bit groups. Reads ~bits/16 of bf16 bytes."""
    split = split_for(w.N, w.K)
    y = torch.empty((split, w.N), dtype=torch.float32, device=x.device)
    grid = lambda meta: (triton.cdiv(w.N, meta["BLOCK_N"]), split)
    _gemv_kernel[grid](x, w.packed, w.scale, y, w.N, w.K, w.K // split, BITS=w.bits)
    # Summing the partials (and the bf16 cast) is fused by torch.compile into the next kernel.
    return y.sum(0, keepdim=True).to(torch.bfloat16)


# ---------------------------------------------------------------------------------------------
# Small-batch W4A16 GEMM (batching milestone): M = 2..64 rows (one per active user).
# Each program dequantizes a [BLOCK_N, BLOCK_K] weight tile ONCE in registers and multiplies it
# with all M input rows on tensor cores (tl.dot). The weight bytes are read once per step and
# shared by every user in the batch - that is the whole point of batching.
# ---------------------------------------------------------------------------------------------
@triton.jit
def _w4_tile(acc, k0, x_ptr, wp_ptr, s_ptr, rm, rn, mmask, nmask, K, KW, KG, shifts,
             BITS: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr, DQ: tl.constexpr):
    """One BLOCK_K step of the small-M int4 GEMM: load packed weights, dequantize, tensor-core dot."""
    PER: tl.constexpr = 32 // BITS
    MASK: tl.constexpr = (1 << BITS) - 1
    OFF: tl.constexpr = 1 << (BITS - 1)
    G: tl.constexpr = 32
    rw = k0 // PER + tl.arange(0, BLOCK_K // PER)
    wp = tl.load(wp_ptr + rn[:, None] * KW + rw[None, :], mask=nmask[:, None], other=0)
    q = (wp[:, :, None] >> shifts[None, None, :]) & MASK
    rg = k0 // G + tl.arange(0, BLOCK_K // G)
    s = tl.load(s_ptr + rn[:, None] * KG + rg[None, :], mask=nmask[:, None], other=0)
    if DQ == 0:       # int -> fp32 convert (slow I2F), subtract the offset, scale in fp32
        q = tl.reshape(q, (BLOCK_N, BLOCK_K // G, G)).to(tl.float32) - OFF
        w = tl.reshape(q * s.to(tl.float32)[:, :, None], (BLOCK_N, BLOCK_K)).to(tl.bfloat16)
    else:             # magic number: bits 0x4300|q ARE the bf16 number 128+q (ulp is 1 at 128)
        f = (q | 0x4300).to(tl.int16).to(tl.bfloat16, bitcast=True)
        f = tl.reshape(f, (BLOCK_N, BLOCK_K // G, G))
        if DQ == 1:   # bf16: (128+q) - (128+OFF) is exact, then one bf16 multiply = same rounding as before
            w = (f - (128 + OFF)) * s[:, :, None]
        else:         # fp32 FMA, one cast
            w = (f.to(tl.float32) - (128 + OFF)) * s.to(tl.float32)[:, :, None]
        w = tl.reshape(w, (BLOCK_N, BLOCK_K)).to(tl.bfloat16)
    rk = k0 + tl.arange(0, BLOCK_K)
    x = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mmask[:, None], other=0)
    return acc + tl.dot(x, tl.trans(w))                                        # [BLOCK_M, BLOCK_N]


@triton.autotune(
    prune_configs_by={"early_config_prune": _prune},
    configs=[triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw, num_stages=ns)
             for bn in (16, 32, 64) for bk in (64, 128, 256) for nw in (2, 4) for ns in (2, 3)],
    key=["M", "N", "K", "KS", "BITS", "DQ"],
)
@triton.jit
def _gemm_kernel(x_ptr, wp_ptr, s_ptr, y_ptr, M, N, K, KS,
                 BITS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                 DQ: tl.constexpr = 0, LS: tl.constexpr = 0):
    PER: tl.constexpr = 32 // BITS
    MASK: tl.constexpr = (1 << BITS) - 1
    OFF: tl.constexpr = 1 << (BITS - 1)
    G: tl.constexpr = 32
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    nmask = rn < N
    mmask = rm < M
    KW = K // PER
    KG = K // G
    shifts = tl.arange(0, PER) * BITS
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if LS == 0:   # kernel-level num_stages: Triton only pipelines loads that feed tl.dot directly
        for k0 in range(pid_k * KS, (pid_k + 1) * KS, BLOCK_K):
            acc = _w4_tile(acc, k0, x_ptr, wp_ptr, s_ptr, rm, rn, mmask, nmask, K, KW, KG, shifts,
                           BITS, BLOCK_K, BLOCK_N, DQ)
    else:         # tl.range(num_stages=LS) asks Triton to prefetch ALL loads LS-1 iterations ahead,
        # including the packed weights that go through shifts/masks before the dot (triton PR #3092)
        for k0 in tl.range(pid_k * KS, (pid_k + 1) * KS, BLOCK_K, num_stages=LS):
            acc = _w4_tile(acc, k0, x_ptr, wp_ptr, s_ptr, rm, rn, mmask, nmask, K, KW, KG, shifts,
                           BITS, BLOCK_K, BLOCK_N, DQ)
    tl.store(y_ptr + (pid_k * M + rm[:, None]) * N + rn[None, :], acc, mask=mmask[:, None] & nmask[None, :])


DQ = 0   # dequantization method of the small-M kernel (see _gemm_kernel; aio/m3e_bench.py measures them)


def gemm_small_m(x: torch.Tensor, w: QLinear, split: int | None = None) -> torch.Tensor:
    """y[M, N] = x[M, K] @ W^T for 2 <= M <= 64, reading the low-bit weights once for all rows."""
    M = x.shape[0]
    split = split or getattr(w, "split", None) or split_for(w.N, w.K)
    block_m = 16 if M <= 16 else (32 if M <= 32 else 64)
    y = torch.empty((split, M, w.N), dtype=torch.float32, device=x.device)
    grid = lambda meta: (triton.cdiv(w.N, meta["BLOCK_N"]), split, triton.cdiv(M, block_m))
    _gemm_kernel[grid](x.contiguous(), w.packed, w.scale, y, M, w.N, w.K, w.K // split,
                       BITS=w.bits, BLOCK_M=block_m, DQ=DQ)
    return y.sum(0).to(torch.bfloat16)
