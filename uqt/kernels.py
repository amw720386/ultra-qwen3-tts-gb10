"""Custom ops used by the engine.

* KV-cache writes as opaque in-place ops: inductor otherwise functionalizes `index_copy_` on a
  slice of the big cache into a full copy of that layer's cache every step.
* A single-kernel top-k sampler (Triton): exact k-th-largest threshold by bisection over
  order-preserving fp32 bit keys, then Gumbel-max over the kept logits. Replaces
  torch.topk (radix sort) + softmax + multinomial. Ties at the threshold are kept, so the
  candidate set can exceed k only when logits tie exactly.
  Bisection/Gumbel approach follows vLLM-Omni's `_cp_sample_kernel` (Apache-2.0).
"""
import torch
import triton
import triton.language as tl


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
