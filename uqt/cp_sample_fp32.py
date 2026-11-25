"""Proposal C (h2h-together FINDINGS.md): the fused code predictor's sampler with fp32 scaled logits and a 32-bit top-k.

uqt.fused_cp._cp_sample_kernel rounds the temperature-scaled logits back to bf16 and finds the top-k threshold by bisection
over 16-bit keys of those bf16 values (ties kept). This variant keeps the scaled logits in fp32 and bisects over
order-preserving 32-bit keys (as uqt.kernels.topk_sample does for the Talker). Same signature, same Gumbel race, same NaN
rule, same PDL handling, so it drops in for both call sites: install() replaces the module global before the engine is built.
"""
import triton
import triton.language as tl
from triton.language.extra import libdevice as tldevice


@triton.jit
def _cp_sample_kernel_fp32(
    logits_ptr, u_ptr, u_stride, codes_ptr, step, G: tl.constexpr, table_ptr, next_ptr, inv_temperature,
    V: tl.constexpr, TOPK: tl.constexpr, HID: tl.constexpr, HAS_NEXT: tl.constexpr, ss_ptr=None, PDL: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    idx = tl.arange(0, V)
    u = tl.load(u_ptr + row * u_stride + idx)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    value = tl.load(logits_ptr + row * V + idx)
    scaled = value.to(tl.float32) * inv_temperature  # stays fp32 (the original rounds back to bf16 here)
    if TOPK > 0:
        bits = scaled.to(tl.uint32, bitcast=True)
        neg = (bits >> 31) == 1
        key = tl.where(neg, bits ^ 0xFFFFFFFF, bits | 0x80000000)  # order-preserving 32-bit keys (xor == ~)
        thr = tl.zeros((), dtype=tl.uint32)
        for b in tl.static_range(31, -1, -1):
            cand = thr | (1 << b)
            cnt = tl.sum((key >= cand).to(tl.int32), axis=0)
            thr = tl.where(cnt >= TOPK, cand, thr)
        sf = tl.where(key >= thr, scaled, -float("inf"))
    else:
        sf = scaled
    score = sf - tldevice.log(-tldevice.log(u))
    max_score = tl.max(score, 0)
    chosen = tl.min(tl.where(score == max_score, idx, V), 0)
    first_nan = tl.min(tl.where(score != score, idx, V), 0)
    code = tl.where(first_nan < V, first_nan, chosen)
    tl.store(codes_ptr + row * G + step, code.to(tl.int64))
    if HAS_NEXT:
        c = tl.arange(0, HID)
        e = tl.load(table_ptr + code.to(tl.int64) * HID + c)
        tl.store(next_ptr + row * HID + c, e)
        if PDL:
            ef = e.to(tl.float32)
            tl.store(ss_ptr + row, tl.sum(ef * ef, axis=0))


def install():
    """Use the fp32 sampler at both call sites of uqt.fused_cp (call before Engine/FusedCodePredictor is built)."""
    import uqt.fused_cp as fcp
    fcp._cp_sample_kernel = _cp_sample_kernel_fp32
