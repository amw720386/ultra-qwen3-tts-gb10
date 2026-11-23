"""Custom ops used by the engine.

* KV-cache writes as opaque in-place ops: inductor otherwise functionalizes `index_copy_` on a
  slice of the big cache into a full copy of that layer's cache every step.
* A single-kernel top-k sampler (Triton): exact k-th-largest threshold by bisection over
  order-preserving fp32 bit keys, then Gumbel-max over the kept logits. Replaces
  torch.topk (radix sort) + softmax + multinomial. Ties at the threshold are kept, so the
  candidate set can exceed k only when logits tie exactly.
  Bisection/Gumbel approach follows vLLM-Omni's `_cp_sample_kernel` (Apache-2.0).
"""
import os

import torch
import triton
import triton.language as tl

# torch.tensor(list, device=cuda) is a blocking copy: it waits for everything queued on the stream, so a
# launch meant to overlap with host work stalls until the step it just launched is done. Off until proven.
ASYNC_H2D = [int(os.environ.get("UQT_ASYNC_H2D", "0"))]


def h2d(data, dev, dtype=None):
    """Host list -> device tensor; with ASYNC_H2D through pinned memory without waiting for the stream."""
    if not ASYNC_H2D[0]:
        return torch.tensor(data, dtype=dtype, device=dev)
    return torch.tensor(data, dtype=dtype).pin_memory().to(dev, non_blocking=True)


@torch.library.custom_op("uqt::kv_write_dyn", mutates_args=("kc", "vc"))

def kv_write_dyn(kc: torch.Tensor, vc: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 pos: torch.Tensor, layer: int) -> None:
    """kc/vc [L, n_kv, S, D]; k/v [1, n_kv, D]; pos [1] long."""
    kc[layer].index_copy_(1, pos, k.transpose(0, 1))
    vc[layer].index_copy_(1, pos, v.transpose(0, 1))


@kv_write_dyn.register_fake
def _(kc, vc, k, v, pos, layer):
    return None


@torch.library.custom_op("uqt::kv_write_static", mutates_args=("kc", "vc"))
def kv_write_static(kc: torch.Tensor, vc: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    layer: int, start: int) -> None:
    """k/v [T, n_kv, D] written to positions [start, start + T)."""
    T = k.shape[0]
    kc[layer, :, start:start + T].copy_(k.transpose(0, 1))
    vc[layer, :, start:start + T].copy_(v.transpose(0, 1))


@kv_write_static.register_fake
def _(kc, vc, k, v, layer, start):
    return None


@triton.jit
def _topk_sample_kernel(logits_ptr, u_ptr, out_ptr, V, inv_temp, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < V
    x = tl.load(logits_ptr + offs, mask=m, other=float("-inf")).to(tl.float32) * inv_temp
    bits = x.to(tl.uint32, bitcast=True)
    neg = (bits >> 31) == 1
    key = tl.where(neg, ~bits, bits | 0x80000000)
    key = tl.where(m, key, 0)
    thr = tl.zeros((), dtype=tl.uint32)
    for b in tl.static_range(31, -1, -1):
        cand = thr | (1 << b)
        cnt = tl.sum((key >= cand).to(tl.int32), axis=0)
        thr = tl.where(cnt >= TOPK, cand, thr)
    keep = (key >= thr) & m & (x > float("-inf"))
    u = tl.load(u_ptr + offs, mask=m, other=0.5)
    u = tl.minimum(tl.maximum(u, 1e-10), 1.0 - 1e-7)
    g = -tl.log(-tl.log(u))
    score = tl.where(keep, x + g, float("-inf"))
    idx = tl.argmax(score, axis=0)
    tl.store(out_ptr, idx.to(tl.int64))


@torch.library.custom_op("uqt::topk_sample", mutates_args=())
def topk_sample(logits: torch.Tensor, u: torch.Tensor, inv_temp: float, top_k: int) -> torch.Tensor:
    """logits [V] (any float dtype), u [V] fp32 uniforms in (0,1) -> [1] long."""
    V = logits.shape[-1]
    out = torch.empty(1, dtype=torch.long, device=logits.device)
    _topk_sample_kernel[(1,)](logits, u, out, V, inv_temp, TOPK=top_k,
                              BLOCK=triton.next_power_of_2(V), num_warps=8)
    return out


@topk_sample.register_fake
def _(logits, u, inv_temp, top_k):
    return logits.new_empty(1, dtype=torch.long)


# ----------------------------------------------------------------------------- batched (rows)

@torch.library.custom_op("uqt::kv_write_rows", mutates_args=("kc", "vc"))
def kv_write_rows(kc: torch.Tensor, vc: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  pos: torch.Tensor, layer: int) -> None:
    """kc/vc [L, S, n_kv, T, D]; k/v [B, n_kv, D]; pos [B] long. Row b writes slot b at pos[b]."""
    B, H, D = k.shape
    k, v = k.contiguous(), v.contiguous()  # [B, n_kv, D] is tiny; compiled graphs can hand in strided views
    if _KV_TRITON and kc.is_contiguous() and vc.is_contiguous():
        L, S, _, T, _ = kc.shape
        _kv_write_rows_kernel[(B, H)](kc, vc, k, v, pos, layer, S, T, H=H, D=D, num_warps=1)
        return
    rows = torch.arange(B, device=k.device)
    kc[layer][rows, :, pos] = k.to(kc.dtype)
    vc[layer][rows, :, pos] = v.to(vc.dtype)


_KV_TRITON = os.environ.get("UQT_KV_TRITON", "1") == "1"


@triton.jit
def _kv_write_rows_kernel(kc_ptr, vc_ptr, k_ptr, v_ptr, pos_ptr, layer, S, T, H: tl.constexpr, D: tl.constexpr):
    """One launch for both caches (ultra-qwen3-tts): row b, kv head h -> cache[layer, b, h, pos[b], :]."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    p = tl.load(pos_ptr + b).to(tl.int64)
    lb = (layer * S + b).to(tl.int64)  # cache is > 2^31 elements: index in int64
    dst = ((lb * H + h) * T + p) * D + d
    src = (b * H + h) * D + d
    # the cache may be FP8 (UQT_KV=fp8): cast on store
    tl.store(kc_ptr + dst, tl.load(k_ptr + src).to(kc_ptr.dtype.element_ty))
    tl.store(vc_ptr + dst, tl.load(v_ptr + src).to(vc_ptr.dtype.element_ty))


@kv_write_rows.register_fake
def _(kc, vc, k, v, pos, layer):
    return None


@torch.library.custom_op("uqt::rows_write", mutates_args=("buf",))
def rows_write(buf: torch.Tensor, idx: torch.Tensor, val: torch.Tensor) -> None:
    """buf [S, T, ...]; idx [B] long; val [B, ...]: buf[b, idx[b]] = val[b]."""
    B = val.shape[0]
    buf[torch.arange(B, device=val.device), idx] = val


@rows_write.register_fake
def _(buf, idx, val):
    return None


@triton.jit
def _topk_sample_rows_kernel(logits_ptr, u_ptr, out_ptr, V, inv_temp, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    m = offs < V
    x = tl.load(logits_ptr + row * V + offs, mask=m, other=float("-inf")).to(tl.float32) * inv_temp
    bits = x.to(tl.uint32, bitcast=True)
    neg = (bits >> 31) == 1
    key = tl.where(neg, ~bits, bits | 0x80000000)
    key = tl.where(m, key, 0)
    thr = tl.zeros((), dtype=tl.uint32)
    for b in tl.static_range(31, -1, -1):
        cand = thr | (1 << b)
        cnt = tl.sum((key >= cand).to(tl.int32), axis=0)
        thr = tl.where(cnt >= TOPK, cand, thr)
    keep = (key >= thr) & m & (x > float("-inf"))
    u = tl.load(u_ptr + row * V + offs, mask=m, other=0.5)
    u = tl.minimum(tl.maximum(u, 1e-10), 1.0 - 1e-7)
    score = tl.where(keep, x - tl.log(-tl.log(u)), float("-inf"))
    tl.store(out_ptr + row, tl.argmax(score, axis=0).to(tl.int64))


@torch.library.custom_op("uqt::topk_sample_rows", mutates_args=())
def topk_sample_rows(logits: torch.Tensor, u: torch.Tensor, inv_temp: float, top_k: int) -> torch.Tensor:
    """logits [B, V] contiguous, u [B, V] fp32 -> [B] long."""
    B, V = logits.shape
    out = torch.empty(B, dtype=torch.long, device=logits.device)
    _topk_sample_rows_kernel[(B,)](logits, u, out, V, inv_temp, TOPK=top_k,
                                   BLOCK=triton.next_power_of_2(V), num_warps=8)
    return out


@topk_sample_rows.register_fake
def _(logits, u, inv_temp, top_k):
    return logits.new_empty(logits.shape[0], dtype=torch.long)


# ----------------------------------------------------------------------------- batched decode attention

@triton.jit
def _bdec_attn_kernel(q_ptr, kc_ptr, vc_ptr, pos_ptr, o_ptr, scale, sk_b, sk_h, sk_t,
                      H: tl.constexpr, HD: tl.constexpr, BLK: tl.constexpr, G: tl.constexpr):
    """One program per (row b, kv head): online softmax over that row's live keys [0, pos[b]] only."""
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    pos = tl.load(pos_ptr + b).to(tl.int32)
    d = tl.arange(0, HD)
    hq = kvh * G + tl.arange(0, G)
    q = tl.load(q_ptr + b * H * HD + hq[:, None] * HD + d[None, :]).to(tl.float32)  # [G, HD]
    base = b.to(tl.int64) * sk_b + kvh.to(tl.int64) * sk_h
    m = tl.full([G], float("-inf"), tl.float32)
    l = tl.zeros([G], tl.float32)
    acc = tl.zeros([G, HD], tl.float32)
    for t0 in range(0, pos + 1, BLK):
        t = t0 + tl.arange(0, BLK)
        tm = t <= pos
        k = tl.load(kc_ptr + base + t[:, None] * sk_t + d[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
        s = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * scale  # [G, BLK]
        s = tl.where(tm[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m - m_new)
        l = l * alpha + tl.sum(p, axis=1)
        v = tl.load(vc_ptr + base + t[:, None] * sk_t + d[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v[None, :, :], axis=1)
        m = m_new
    out = acc / l[:, None]
    tl.store(o_ptr + b * H * HD + hq[:, None] * HD + d[None, :], out.to(o_ptr.dtype.element_ty))


@triton.jit
def _bdec_attn_dot_kernel(q_ptr, kc_ptr, vc_ptr, pos_ptr, o_ptr, scale, sk_b, sk_h, sk_t,
                          H: tl.constexpr, HD: tl.constexpr, BLK: tl.constexpr, G: tl.constexpr, GP: tl.constexpr):
    """Tensor-core variant (from the H100 backport): the G query heads of one kv head padded to GP >= 16
    rows, tl.dot per key block."""
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    pos = tl.load(pos_ptr + b).to(tl.int32)
    d = tl.arange(0, HD)
    gi = tl.arange(0, GP)
    gm = gi < G
    q = tl.load(q_ptr + b * H * HD + (kvh * G + gi)[:, None] * HD + d[None, :], mask=gm[:, None], other=0.0)
    base = b.to(tl.int64) * sk_b + kvh.to(tl.int64) * sk_h
    m = tl.full([GP], float("-inf"), tl.float32)
    l = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, HD], tl.float32)
    for t0 in range(0, pos + 1, BLK):
        t = t0 + tl.arange(0, BLK)
        tm = t <= pos
        k = tl.load(kc_ptr + base + t[:, None] * sk_t + d[None, :], mask=tm[:, None], other=0.0).to(q.dtype)
        s = tl.dot(q, tl.trans(k)) * scale  # [GP, BLK]
        s = tl.where(tm[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m - m_new)
        l = l * alpha + tl.sum(p, axis=1)
        v = tl.load(vc_ptr + base + t[:, None] * sk_t + d[None, :], mask=tm[:, None], other=0.0).to(q.dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)
        m = m_new
    out = acc / l[:, None]
    tl.store(o_ptr + b * H * HD + (kvh * G + gi)[:, None] * HD + d[None, :], out.to(o_ptr.dtype.element_ty),
             mask=gm[:, None])


# "dot" = tensor-core kernel, "fma" = CUDA-core kernel. (BLK, warps, stages) for the dot kernel.
BATTN_KIND = os.environ.get("UQT_BATTN_KIND", "dot")  # dot: +1-3% engine throughput on GB10 (results/gb10/scan-battn-*.json)
BATTN_DOT_CFG = tuple(int(x) for x in os.environ.get("UQT_BATTN_DOT_CFG", "64,8,1").split(","))  # GB10 sweep: bench/gb10/battn_micro.py


@torch.library.custom_op("uqt::bdec_attn", mutates_args=())
def bdec_attn(q: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor, pos: torch.Tensor, scale: float) -> torch.Tensor:
    """q [B, H, HD]; kc/vc [S, KVH, T, HD] (one layer, row b = slot b); pos [B] -> [B, H * HD]."""
    B, H, HD = q.shape
    KVH = kc.shape[1]
    q = q.contiguous()
    out = torch.empty(B, H * HD, dtype=q.dtype, device=q.device)
    G = H // KVH
    if BATTN_KIND == "dot":
        blk, nw, st = BATTN_DOT_CFG
        _bdec_attn_dot_kernel[(B, KVH)](q, kc, vc, pos, out, scale, kc.stride(0), kc.stride(1), kc.stride(2),
                                        H=H, HD=HD, BLK=blk, G=G, GP=max(16, G), num_warps=nw, num_stages=st)
        return out
    _bdec_attn_kernel[(B, KVH)](q, kc, vc, pos, out, scale, kc.stride(0), kc.stride(1), kc.stride(2),
                                H=H, HD=HD, BLK=32, G=G, num_warps=4)
    return out


@bdec_attn.register_fake
def _(q, kc, vc, pos, scale):
    return q.new_empty(q.shape[0], q.shape[1] * q.shape[2])


# ----------------------------------------------------------------------------- INT8 KV cache (UQT_KV=int8)
# Each (token, kv head) row of K and V is stored as int8 with its own fp32 absmax scale: half the bytes of
# a BF16 cache. Rounding error per format on the real cache: bench/gb10/kv_real_error.py.

def quant_rows(x: torch.Tensor):
    """x [..., D] -> (int8 [..., D], fp32 scale [...]) with per-row absmax scaling."""
    s = x.float().abs().amax(-1).clamp_min(1e-8) / 127.0
    return torch.round(x.float() / s[..., None]).clamp_(-127, 127).to(torch.int8), s


@triton.jit
def _kv_write_rows_q_kernel(kc_ptr, vc_ptr, ks_ptr, vs_ptr, k_ptr, v_ptr, pos_ptr, layer, S, T,
                            H: tl.constexpr, D: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    p = tl.load(pos_ptr + b).to(tl.int64)
    row = ((layer * S + b).to(tl.int64) * H + h) * T + p
    src = (b * H + h) * D + d
    k = tl.load(k_ptr + src).to(tl.float32)
    v = tl.load(v_ptr + src).to(tl.float32)
    sk = tl.maximum(tl.max(tl.abs(k), axis=0), 1e-8) / 127.0
    sv = tl.maximum(tl.max(tl.abs(v), axis=0), 1e-8) / 127.0
    kq = tl.minimum(tl.maximum(k / sk, -127.0), 127.0)
    vq = tl.minimum(tl.maximum(v / sv, -127.0), 127.0)
    kq = tl.where(kq >= 0, tl.floor(kq + 0.5), tl.ceil(kq - 0.5))
    vq = tl.where(vq >= 0, tl.floor(vq + 0.5), tl.ceil(vq - 0.5))
    tl.store(kc_ptr + row * D + d, kq.to(tl.int8))
    tl.store(vc_ptr + row * D + d, vq.to(tl.int8))
    tl.store(ks_ptr + row, sk)
    tl.store(vs_ptr + row, sv)


@torch.library.custom_op("uqt::kv_write_rows_q", mutates_args=("kc", "vc", "ks", "vs"))
def kv_write_rows_q(kc: torch.Tensor, vc: torch.Tensor, ks: torch.Tensor, vs: torch.Tensor, k: torch.Tensor,
                    v: torch.Tensor, pos: torch.Tensor, layer: int) -> None:
    """kc/vc int8 [L, S, n_kv, T, D]; ks/vs fp32 [L, S, n_kv, T]; k/v [B, n_kv, D]; pos [B]."""
    B, H, D = k.shape
    _, S, _, T, _ = kc.shape
    _kv_write_rows_q_kernel[(B, H)](kc, vc, ks, vs, k.contiguous(), v.contiguous(), pos, layer, S, T,
                                    H=H, D=D, num_warps=1)


@kv_write_rows_q.register_fake
def _(kc, vc, ks, vs, k, v, pos, layer):
    return None


@triton.jit
def _bdec_attn_q_kernel(q_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pos_ptr, o_ptr, scale, sk_b, sk_h, sk_t, ss_b, ss_h,
                        H: tl.constexpr, HD: tl.constexpr, BLK: tl.constexpr, G: tl.constexpr, GP: tl.constexpr):
    """_bdec_attn_dot_kernel over an int8 cache: int8 -> bf16 is exact, the per-key scales are applied
    to the logits (K) and to the probabilities (V)."""
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    pos = tl.load(pos_ptr + b).to(tl.int32)
    d = tl.arange(0, HD)
    gi = tl.arange(0, GP)
    gm = gi < G
    q = tl.load(q_ptr + b * H * HD + (kvh * G + gi)[:, None] * HD + d[None, :], mask=gm[:, None], other=0.0)
    base = b.to(tl.int64) * sk_b + kvh.to(tl.int64) * sk_h
    sbase = b.to(tl.int64) * ss_b + kvh.to(tl.int64) * ss_h
    m = tl.full([GP], float("-inf"), tl.float32)
    l = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, HD], tl.float32)
    for t0 in range(0, pos + 1, BLK):
        t = t0 + tl.arange(0, BLK)
        tm = t <= pos
        k = tl.load(kc_ptr + base + t[:, None] * sk_t + d[None, :], mask=tm[:, None], other=0).to(q.dtype)
        kscl = tl.load(ks_ptr + sbase + t, mask=tm, other=0.0)
        s = tl.dot(q, tl.trans(k)) * (kscl * scale)[None, :]
        s = tl.where(tm[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m - m_new)
        l = l * alpha + tl.sum(p, axis=1)
        v = tl.load(vc_ptr + base + t[:, None] * sk_t + d[None, :], mask=tm[:, None], other=0).to(q.dtype)
        vscl = tl.load(vs_ptr + sbase + t, mask=tm, other=0.0)
        acc = acc * alpha[:, None] + tl.dot((p * vscl[None, :]).to(q.dtype), v)
        m = m_new
    out = acc / l[:, None]
    tl.store(o_ptr + b * H * HD + (kvh * G + gi)[:, None] * HD + d[None, :], out.to(o_ptr.dtype.element_ty),
             mask=gm[:, None])


@torch.library.custom_op("uqt::bdec_attn_q", mutates_args=())
def bdec_attn_q(q: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor, ks: torch.Tensor, vs: torch.Tensor,
                pos: torch.Tensor, scale: float) -> torch.Tensor:
    """q [B, H, HD]; kc/vc int8 [S, KVH, T, HD]; ks/vs fp32 [S, KVH, T]; pos [B] -> [B, H * HD]."""
    B, H, HD = q.shape
    KVH = kc.shape[1]
    q = q.contiguous()
    out = torch.empty(B, H * HD, dtype=q.dtype, device=q.device)
    G = H // KVH
    blk, nw, st = BATTN_DOT_CFG
    _bdec_attn_q_kernel[(B, KVH)](q, kc, vc, ks, vs, pos, out, scale, kc.stride(0), kc.stride(1), kc.stride(2),
                                  ks.stride(0), ks.stride(1), H=H, HD=HD, BLK=blk, G=G, GP=max(16, G),
                                  num_warps=nw, num_stages=st)
    return out


@bdec_attn_q.register_fake
def _(q, kc, vc, ks, vs, pos, scale):
    return q.new_empty(q.shape[0], q.shape[1] * q.shape[2])
