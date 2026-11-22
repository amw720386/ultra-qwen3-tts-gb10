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

import os

import torch
import triton
import triton.language as tl

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP4_GROUP = 32
FP8_BIG = os.environ.get("UQT_FP8_BIG", "1") == "1"  # FP8 kernel above MAX_ROWS for the shapes where it wins
BIG_MAX_ROWS = int(os.environ.get("UQT_FP8_BIG_MAX", "320"))  # batched decode sizes; prefill stays bf16
MAX_ROWS = 128  # above this the work is compute-bound: bf16 cuBLAS wins (measured on GB10)


class QW:
    """A quantized weight. Keeps the bf16 original for large-M calls."""

    def __init__(self, w: torch.Tensor, kind: str):
        assert w.dim() == 2 and w.is_cuda
        hyb = kind == "hyb"  # FP8 for prefill / batches plus an FP4 copy for 1-2 row decode GEMVs
        kind = "fp8" if hyb else kind
        self.kind = kind
        self.data4 = self.scale4 = None
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
        if hyb:
            f4 = QW(w, "fp4")
            self.data4, self.scale4 = f4.data, f4.scale

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


# Small-M decode GEMMs are a few microseconds of device time, below the launch overhead, so configs are
# be benchmarked inside CUDA graphs with narrow tiles (UQT_WQ_GRAPH_TUNE=1); measured no end-to-end gain.
_GRAPH_TUNE = os.environ.get("UQT_WQ_GRAPH_TUNE", "0") == "1"  # measured: -4% GEMM time, no end-to-end change
_CFG = [triton.Config({"BN": bn, "BK": bk}, num_warps=w, num_stages=st)
        for bn in (16, 32, 64) for bk in (128, 256, 512) for w in (2, 4) for st in (2, 3, 4)
        if bn * bk <= 16384] if _GRAPH_TUNE else [
        triton.Config({"BN": bn, "BK": bk}, num_warps=w, num_stages=st)
        for bn in (32, 64) for bk in (128, 256) for w in (4,) for st in (3, 4)]


@triton.autotune(configs=_CFG, key=["MB", "N", "K"], use_cuda_graph=_GRAPH_TUNE)
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
         for bn in (16, 32, 64, 128) for bk in (128, 256, 512) for w in (2, 4, 8) for st in (2, 3, 4)
         if bn * bk <= 32768] if _GRAPH_TUNE else [
         triton.Config({"BN": bn, "BK": bk}, num_warps=w, num_stages=st)
         for bn in (32, 64, 128) for bk in (128, 256) for w in (4, 8) for st in (3, 4)]


@triton.autotune(configs=_CFG4, key=["MB", "N", "K"], use_cuda_graph=_GRAPH_TUNE)
@triton.jit
def _fp4_mm_kernel(a_ptr, w_ptr, s_ptr, c_ptr, M, N, K, stride_am, stride_cm, MB,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr):
    """W: uint8 [N, K/2] (low nibble = even k), scales bf16 [N, K/G]. BK counts unpacked k.

    Low (even k) and high (odd k) nibbles each meet the matching even / odd activation columns in
    their own dot, so nothing is interleaved; group scales load once per (row, group) and broadcast
    over the group, not once per weight (ultra-qwen3-tts)."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    HB: tl.constexpr = BK // 2
    NG: tl.constexpr = BK // G
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rj = tl.arange(0, HB)
    rg = tl.arange(0, NG)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        j0 = k0 // 2
        packed = tl.load(w_ptr + rn[:, None] * (K // 2) + (j0 + rj)[None, :], mask=rn[:, None] < N, other=0)
        sc = tl.load(s_ptr + rn[:, None] * (K // G) + (k0 // G + rg)[None, :], mask=rn[:, None] < N, other=0.0)
        sc3 = sc[:, :, None]
        lo = tl.reshape(tl.reshape(_e2m1(packed & 0xF), (BN, NG, G // 2)) * sc3, (BN, HB))
        hi = tl.reshape(tl.reshape(_e2m1(packed >> 4), (BN, NG, G // 2)) * sc3, (BN, HB))
        ae = tl.load(a_ptr + rm[:, None] * stride_am + (k0 + 2 * rj)[None, :], mask=rm[:, None] < M, other=0.0)
        ao = tl.load(a_ptr + rm[:, None] * stride_am + (k0 + 2 * rj + 1)[None, :], mask=rm[:, None] < M, other=0.0)
        acc = tl.dot(ae, tl.trans(lo), acc)
        acc = tl.dot(ao, tl.trans(hi), acc)
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :], acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


def _mb(M):
    return 16 if M <= 16 else (32 if M <= 32 else (64 if M <= 64 else 128))


_CFG_BIG = [triton.Config({"BM": bm, "BN": bn, "BK": bk}, num_warps=w, num_stages=st)
            for bm in (64, 128) for bn in (64, 128) for bk in (64, 128) for w in (4, 8) for st in (3, 4)
            if (bm * bk * 2 + bn * bk * 3) * st <= 96000]


@triton.autotune(configs=_CFG_BIG, key=["MB", "N", "K"])
@triton.jit
def _fp8_mm_big_kernel(a_ptr, w_ptr, s_ptr, c_ptr, M, N, K, stride_am, stride_cm, MB,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """FP8 weight-only GEMM for many rows (ultra-qwen3-tts): 1-D grid with the row tiles fastest, so the
    row tiles sharing a weight tile run together and the weight is read from DRAM about once."""
    nm = tl.cdiv(M, BM)
    pid = tl.program_id(0)
    pid_m = pid % nm
    pid_n = pid // nm
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


@torch.library.custom_op("uqt::fp8_mm_big", mutates_args=())
def fp8_mm_big(a: torch.Tensor, data: torch.Tensor, scale: torch.Tensor, N: int, K: int) -> torch.Tensor:
    a = a.contiguous()
    M = a.shape[0]
    out = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),)  # noqa: E731
    mb = 1 << max(7, (M - 1).bit_length())
    _fp8_mm_big_kernel[grid](a, data, scale, out, M, N, K, a.stride(0), out.stride(0), mb)
    return out


@fp8_mm_big.register_fake
def _(a, data, scale, N, K):
    return a.new_empty(a.shape[0], N)


# ---- 1-2 row decode GEMVs (ultra-qwen3-tts): multiply-reduce on CUDA cores instead of tl.dot with the rows
# padded to 16. Measured on GB10 (bench/gb10/gemv_micro.py) FP4 reaches ~190-260 GB/s at one row, so with half
# the bytes it is 1.6-2.6x faster than the FP8 tensor-core kernel; slower from 4 rows on.
GEMV_MAX_ROWS = int(os.environ.get("UQT_GEMV_MAX_ROWS", "2"))
GEMV8 = os.environ.get("UQT_GEMV8", "1") == "1"  # FP8 weights also take the GEMV at 1-2 rows (~5-10%)
_GEMV_CFG = [triton.Config({"BN": bn, "BK": bk}, num_warps=w)
             for bn in (4, 8, 16, 32) for bk in (256, 512, 1024) for w in (2, 4, 8) if bn * bk <= 16384]


@triton.autotune(configs=_GEMV_CFG, key=["N", "K", "M"], use_cuda_graph=True)
@triton.jit
def _gemv8_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K: tl.constexpr, MP: tl.constexpr, BN: tl.constexpr,
                  BK: tl.constexpr):
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    rk0 = tl.arange(0, BK)
    for m in tl.static_range(MP):
        if m < M:
            acc = tl.zeros([BN], dtype=tl.float32)
            for k0 in tl.static_range(0, K, BK):
                x = tl.load(x_ptr + m * K + k0 + rk0).to(tl.float32)
                w = tl.load(w_ptr + rn[:, None] * K + (k0 + rk0)[None, :]).to(tl.float32)
                acc += tl.sum(w * x[None, :], axis=1)
            tl.store(y_ptr + m * N + rn, (acc * tl.load(s_ptr + rn)).to(tl.bfloat16))


@triton.autotune(configs=_GEMV_CFG, key=["N", "K", "M"], use_cuda_graph=True)
@triton.jit
def _gemv4_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K: tl.constexpr, MP: tl.constexpr, BN: tl.constexpr,
                  BK: tl.constexpr, G: tl.constexpr):
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    HB: tl.constexpr = BK // 2
    NG: tl.constexpr = BK // G
    rj = tl.arange(0, HB)
    rg = tl.arange(0, NG)
    for m in tl.static_range(MP):
        if m < M:
            acc = tl.zeros([BN], dtype=tl.float32)
            for k0 in tl.static_range(0, K, BK):
                p = tl.load(w_ptr + rn[:, None] * (K // 2) + (k0 // 2 + rj)[None, :])
                xe = tl.load(x_ptr + m * K + k0 + 2 * rj).to(tl.float32)
                xo = tl.load(x_ptr + m * K + k0 + 2 * rj + 1).to(tl.float32)
                part = _e2m1(p & 0xF).to(tl.float32) * xe[None, :] + _e2m1(p >> 4).to(tl.float32) * xo[None, :]
                sc = tl.load(s_ptr + rn[:, None] * (K // G) + (k0 // G + rg)[None, :]).to(tl.float32)
                acc += tl.sum(tl.sum(tl.reshape(part, (BN, NG, G // 2)), axis=2) * sc, axis=1)
            tl.store(y_ptr + m * N + rn, acc.to(tl.bfloat16))


@torch.library.custom_op("uqt::wq_gemv", mutates_args=())
def wq_gemv(a: torch.Tensor, data: torch.Tensor, scale: torch.Tensor, kind: int, N: int, K: int) -> torch.Tensor:
    """a [M <= 2, K] bf16 -> [M, N] bf16. kind 0 = fp8, 1 = fp4."""
    a = a.contiguous()
    M = a.shape[0]
    out = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
    grid = lambda meta: (triton.cdiv(N, meta["BN"]),)  # noqa: E731
    if kind == 0:
        _gemv8_kernel[grid](a, data, scale, out, M, N, K=K, MP=2)
    else:
        _gemv4_kernel[grid](a, data, scale, out, M, N, K=K, MP=2, G=FP4_GROUP)
    return out


@wq_gemv.register_fake
def _(a, data, scale, kind, N, K):
    return a.new_empty(a.shape[0], N)


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
        # Measured on GB10 (bench/gb10/wq_big_test.py): FP8 wins for every shape up to ~192 rows, and
        # for the widest projections (N >= 8192) up to 320; otherwise bf16 cuBLAS is as fast or faster.
        m = x2.shape[0]
        if FP8_BIG and w.kind == "fp8" and (m <= 192 or (w.N >= 8192 and m <= BIG_MAX_ROWS)):
            return fp8_mm_big(x2, w.data, w.scale, w.N, w.K).reshape(*lead, w.N)
        return torch.nn.functional.linear(x, w.bf16)
    if x2.shape[0] <= GEMV_MAX_ROWS and w.K % 1024 == 0:
        if w.data4 is not None:
            return wq_gemv(x2, w.data4, w.scale4, 1, w.N, w.K).reshape(*lead, w.N)
        if w.kind == "fp8" and GEMV8:
            return wq_gemv(x2, w.data, w.scale, 0, w.N, w.K).reshape(*lead, w.N)
    out = wq_mm(x2, w.data, w.scale, 0 if w.kind == "fp8" else 1, w.N, w.K)
    return out.reshape(*lead, w.N)
