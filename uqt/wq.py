"""Weight-only quantized matmul for memory-bound decode (GB10: ~273 GB/s, so weight bytes dominate).

    out[M, N] = A[M, K] (bf16) @ dequant(W)[N, K]^T

Formats:
* "fp8": W in float8_e4m3fn with one fp32 scale per output channel (row of W). 1 byte / weight.
* "fp4": W in e2m1 (NVFP4 element format: values 0, 0.5, 1, 1.5, 2, 3, 4, 6 and negatives), two per
  byte along K (low nibble = even k), with one bf16 scale per group of 32 weights along K.
  0.5 byte / weight (+ 1/16 byte of scale).

Dequantization happens in registers inside the Triton kernel; tensor-core math stays bf16. Large-M
calls (prompt prefill) fall back to the bf16 weights, where the work is compute-bound anyway.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP4_GROUP = 32
MAX_ROWS = 128  # above this the work is compute-bound: bf16 cuBLAS wins (measured on GB10)


class QW:
    """A quantized weight. Keeps the bf16 original for large-M calls."""

    def __init__(self, w: torch.Tensor, kind: str):
        assert w.dim() == 2 and w.is_cuda
        self.kind = kind
        self.N, self.K = w.shape
        self.bf16 = w.contiguous()
        wf = w.float()
        if kind == "fp8":
            s = wf.abs().amax(dim=1).clamp(min=1e-8) / 448.0
            self.data = (wf / s[:, None]).to(torch.float8_e4m3fn).contiguous()
            self.scale = s.contiguous()
        elif kind == "fp4":
            assert self.K % FP4_GROUP == 0
            g = wf.view(self.N, self.K // FP4_GROUP, FP4_GROUP)
            s = (g.abs().amax(dim=2).clamp(min=1e-8) / 6.0)
            q = g / s[..., None]
            mags = torch.tensor(E2M1, device=w.device)
            idx = (q.abs()[..., None] - mags).abs().argmin(dim=-1)  # nearest magnitude
            code = idx | ((q < 0).to(idx.dtype) << 3)
            code = code.view(self.N, self.K).to(torch.uint8)
            self.data = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
            self.scale = s.to(torch.bfloat16).contiguous()  # [N, K / 32]
        else:
            raise ValueError(kind)

    def dequant(self) -> torch.Tensor:
        if self.kind == "fp8":
            return (self.data.float() * self.scale[:, None]).to(torch.bfloat16)
        lut = torch.tensor(E2M1 + tuple(-v for v in E2M1), device=self.data.device)
        lo, hi = self.data & 0xF, self.data >> 4
        code = torch.stack([lo, hi], dim=-1).view(self.N, self.K).long()
        vals = lut[code].view(self.N, self.K // FP4_GROUP, FP4_GROUP) * self.scale.float()[..., None]
        return vals.view(self.N, self.K).to(torch.bfloat16)

    @property
    def nbytes(self):
        return self.data.numel() * self.data.element_size() + self.scale.numel() * self.scale.element_size()


@triton.jit
def _e2m1(v):
    """4-bit code -> bf16 by building the bits directly (no float math).

    e2m1 = sign | 2-bit exponent e | 1-bit mantissa m; value (1 + m/2) * 2^(e-1), or m/2 when e == 0.
    bf16: exponent field e - 1 + 127 = e + 126 at bit 7, mantissa bit m at bit 6; 0.5 = (126 << 7).
    """
    v = v.to(tl.int32)
    e = (v >> 1) & 3
    m = v & 1
    bits = tl.where(e == 0, m * (126 << 7), ((e + 126) << 7) | (m << 6)) | (((v >> 3) & 1) << 15)
    return bits.to(tl.int16).to(tl.bfloat16, bitcast=True)


_CFG = [triton.Config({"BN": bn, "BK": bk}, num_warps=w, num_stages=st)
        for bn in (32, 64) for bk in (128, 256) for w in (4,) for st in (3, 4)]


@triton.autotune(configs=_CFG, key=["MB", "N", "K"])
@triton.jit
def _fp8_mm_kernel(a_ptr, w_ptr, s_ptr, c_ptr, M, N, K, stride_am, stride_cm, MB,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * stride_am + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :], mask=(rn[:, None] < N) & (rk[None, :] < K), other=0.0)
        acc = tl.dot(a, tl.trans(w.to(tl.bfloat16)), acc)
    s = tl.load(s_ptr + rn, mask=rn < N, other=0.0)
    acc = acc * s[None, :]
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :], acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


_CFG4 = [triton.Config({"BN": bn, "BK": bk}, num_warps=w, num_stages=st)
         for bn in (32, 64) for bk in (128, 256) for w in (4,) for st in (3, 4)]


@triton.autotune(configs=_CFG4, key=["MB", "N", "K"])
@triton.jit
def _fp4_mm_kernel(a_ptr, w_ptr, s_ptr, c_ptr, M, N, K, stride_am, stride_cm, MB,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr):
    """W: uint8 [N, K/2] (low nibble = even k), scales bf16 [N, K/G]. BK counts unpacked k.

    Decoded low/high nibbles are interleaved back into k order so one contiguous dot covers BK."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    HB: tl.constexpr = BK // 2
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rj = tl.arange(0, HB)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        j0 = k0 // 2
        packed = tl.load(w_ptr + rn[:, None] * (K // 2) + (j0 + rj)[None, :], mask=rn[:, None] < N, other=0)
        grp = (k0 + 2 * rj) // G
        sc = tl.load(s_ptr + rn[:, None] * (K // G) + grp[None, :], mask=rn[:, None] < N, other=0.0)
        lo = _e2m1(packed & 0xF) * sc
        hi = _e2m1(packed >> 4) * sc
        w = tl.interleave(lo, hi)  # [BN, BK] in k order
        rk = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * stride_am + rk[None, :], mask=rm[:, None] < M, other=0.0)
        acc = tl.dot(a, tl.trans(w), acc)
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :], acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


def _mb(M):
    return 16 if M <= 16 else (32 if M <= 32 else (64 if M <= 64 else 128))


@torch.library.custom_op("uqt::wq_mm", mutates_args=())
def wq_mm(a: torch.Tensor, data: torch.Tensor, scale: torch.Tensor, kind: int, N: int, K: int) -> torch.Tensor:
    """a [M, K] bf16 -> [M, N] bf16. kind 0 = fp8, 1 = fp4."""
    a = a.contiguous()
    M = a.shape[0]
    out = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
    MB = _mb(M)
    BM = min(MB, 64)
    grid = lambda meta: (triton.cdiv(N, meta["BN"]), triton.cdiv(M, BM))  # noqa: E731
    if kind == 0:
        _fp8_mm_kernel[grid](a, data, scale, out, M, N, K, a.stride(0), out.stride(0), MB, BM=BM)
    else:
        _fp4_mm_kernel[grid](a, data, scale, out, M, N, K, a.stride(0), out.stride(0), MB, BM=BM, G=FP4_GROUP)
    return out


@wq_mm.register_fake
def _(a, data, scale, kind, N, K):
    return a.new_empty(a.shape[0], N)


def lin(x: torch.Tensor, w) -> torch.Tensor:
    """F.linear(x, w) for a bf16 tensor or a QW (quantized kernel for <= MAX_ROWS rows)."""
    if not isinstance(w, QW):
        return torch.nn.functional.linear(x, w)
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if x2.shape[0] > MAX_ROWS:
        return torch.nn.functional.linear(x, w.bf16)
    out = wq_mm(x2, w.data, w.scale, 0 if w.kind == "fp8" else 1, w.N, w.K)
    return out.reshape(*lead, w.N)
