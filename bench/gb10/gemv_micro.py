"""M=1..4 weight GEMVs on GB10: plain multiply-reduce GEMV kernels (no tensor cores) for FP8 and FP4 weights vs
the production uqt.wq kernels (tl.dot with the rows padded to 16), at talker / code-predictor shapes, timed
inside CUDA graphs with weights cycled so they stream from DRAM.

    python -m bench.gb10.gemv_micro
"""
import itertools
import json

import torch
import triton
import triton.language as tl

from uqt.wq import QW, wq_mm, _e2m1  # noqa: F401

SHAPES = {"talker_qkv": (4096, 2048), "talker_o": (2048, 2048), "talker_gu": (12288, 2048),
          "talker_down": (2048, 6144), "cp_qkv": (4096, 1024), "cp_o": (1024, 2048), "cp_gu": (6144, 1024),
          "cp_down": (1024, 3072)}


@triton.jit
def _gemv8(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K: tl.constexpr, MP: tl.constexpr, BN: tl.constexpr,
           BK: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rk0 = tl.arange(0, BK)
    for m in tl.static_range(MP):
        if m < M:
            acc = tl.zeros([BN], dtype=tl.float32)
            for k0 in tl.static_range(0, K, BK):
                x = tl.load(x_ptr + m * K + k0 + rk0).to(tl.float32)
                w = tl.load(w_ptr + rn[:, None] * K + (k0 + rk0)[None, :]).to(tl.float32)
                acc += tl.sum(w * x[None, :], axis=1)
            tl.store(y_ptr + m * N + rn, (acc * tl.load(s_ptr + rn)).to(tl.bfloat16))


@triton.jit
def _gemv4(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K: tl.constexpr, MP: tl.constexpr, BN: tl.constexpr,
           BK: tl.constexpr, G: tl.constexpr):
    """W uint8 [N, K/2] (low nibble = even k), scales bf16 [N, K/G]."""
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
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


def graph_time(fn, iters=50):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        g.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


def main():
    out = {}
    REP = 8  # chain of REP calls on different weight copies per graph replay, so weights stream from DRAM
    with torch.inference_mode():
        for name, (N, K) in SHAPES.items():
            w = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
            q8 = [QW(w, "fp8") for _ in range(REP)]
            q4 = [QW(w, "fp4") for _ in range(REP)]
            row = {}
            for M in (1, 2, 4):
                x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                y = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
                r8 = x @ q8[0].dequant().T
                r4 = x @ q4[0].dequant().T
                t_prod8 = graph_time(lambda: [wq_mm(x, q.data, q.scale, 0, N, K) for q in q8]) / REP
                t_prod4 = graph_time(lambda: [wq_mm(x, q.data, q.scale, 1, N, K) for q in q4]) / REP
                best8 = best4 = None
                for BN, BK, nw in itertools.product((4, 8, 16, 32), (256, 512, 1024), (2, 4, 8)):
                    if BK > K or BN * BK > 16384:
                        continue
                    f8 = lambda: [_gemv8[(triton.cdiv(N, BN),)](x, q.data, q.scale, y, M, N, K=K, MP=4, BN=BN, BK=BK,  # noqa: E731
                                                              num_warps=nw) for q in q8]
                    f4 = lambda: [_gemv4[(triton.cdiv(N, BN),)](x, q.data, q.scale, y, M, N, K=K, MP=4, BN=BN, BK=BK,  # noqa: E731
                                                              G=32, num_warps=nw) for q in q4]
                    try:
                        t8 = graph_time(f8, iters=20) / REP
                        if best8 is None or t8 < best8[1]:
                            best8 = ((BN, BK, nw), t8)
                        t4 = graph_time(f4, iters=20) / REP
                        if best4 is None or t4 < best4[1]:
                            best4 = ((BN, BK, nw), t4)
                    except Exception:  # noqa: BLE001
                        continue
                c8, c4 = best8[0], best4[0]
                _gemv8[(triton.cdiv(N, c8[0]),)](x, q8[0].data, q8[0].scale, y, M, N, K=K, MP=4, BN=c8[0], BK=c8[1], num_warps=c8[2])
                e8 = float((y.float() - r8.float()).norm() / r8.float().norm())
                _gemv4[(triton.cdiv(N, c4[0]),)](x, q4[0].data, q4[0].scale, y, M, N, K=K, MP=4, BN=c4[0], BK=c4[1], G=32, num_warps=c4[2])
                e4 = float((y.float() - r4.float()).norm() / r4.float().norm())
                row[M] = {"prod_fp8_us": round(t_prod8 * 1e3, 1), "gemv_fp8_us": round(best8[1] * 1e3, 1), "cfg8": c8,
                          "prod_fp4_us": round(t_prod4 * 1e3, 1), "gemv_fp4_us": round(best4[1] * 1e3, 1), "cfg4": c4,
                          "gemv_fp8_GBps": round(N * K / best8[1] / 1e6), "gemv_fp4_GBps": round(N * K / 2 / best4[1] / 1e6),
                          "err8": round(e8, 4), "err4": round(e4, 4)}
            out[name] = row
            print(name, json.dumps(row), flush=True)
    json.dump(out, open("/root/results/gemv_micro.json", "w"), indent=1)


if __name__ == "__main__":
    main()
