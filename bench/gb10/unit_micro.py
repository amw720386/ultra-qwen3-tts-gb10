"""Microbenchmark one decoder residual unit per width on GB10 shapes (256 streams x 2 frames):
old path (cuDNN / dilated implicit GEMM / im2col + cuBLAS, then addmm) vs the fused unit kernel and
the dilated implicit GEMM across tile configs.

    python -m bench.gb10.unit_micro --frames 512
"""
import argparse
import itertools
import json

import torch
import triton

import uqt.streaming_codec as sc

WIDTHS = [(768, 32), (384, 160), (192, 640), (96, 1920)]  # C, rows per frame


def bench(fn, iters=15):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=512)
    ap.add_argument("--widths", default="768,384")
    args = ap.parse_args()
    dev, dt = "cuda", torch.bfloat16
    B = 256
    S = B + 1
    out = {}
    with torch.inference_mode():
        for C, rpf in WIDTHS:
            if str(C) not in args.widths.split(","):
                continue
            T = (args.frames // B) * rpf
            rows = B * T
            res = {}
            for D in (1, 3, 9):
                H = 6 * D
                x = torch.randn(rows, C, device=dev, dtype=dt)
                w1 = torch.randn(7 * C, C, device=dev, dtype=dt) * 0.02
                w2 = torch.randn(C, C, device=dev, dtype=dt) * 0.05
                hist = torch.randn(2, S, H, C, device=dev, dtype=dt) * 0.1
                pb, b1 = torch.randn(C, device=dev, dtype=dt) * 0.1, torch.randn(C, device=dev, dtype=dt) * 0.1
                a1, ib1, a2, ib2 = [torch.rand(C, device=dev) + 0.5 for _ in range(4)]
                slots = torch.arange(B, device=dev, dtype=torch.int32)
                pos = torch.full((B,), 40, device=dev, dtype=torch.int32)
                par = torch.zeros(S, device=dev, dtype=torch.int32)

                def ext_pass():
                    ext = torch.empty(B * (H + T), C, device=dev, dtype=dt)
                    sc._im2col_kernel[(B, triton.cdiv(H + T, 32), triton.cdiv(C, 128))](
                        x, pb, a1, ib1, hist, slots, pos, par, ext, T, S, C=C, K=7, D=D, H=H,
                        HAS_BIAS=True, HAS_SNAKE=True, BU=32, BC=128, EXT=True)
                    return ext
                ext = ext_pass()
                r = {"ext_pass": bench(ext_pass)}

                def igemm(cfg):
                    bm, bn, bc, nw, stg = cfg
                    o = torch.empty(rows, C, device=dev, dtype=dt)

                    def f():
                        sc._dilated_conv_snake_kernel[(triton.cdiv(rows, bm), triton.cdiv(C, bn))](
                            ext, w1, o, b1, a2, ib2, T, rows, C=C, N=C, K=7, D=D, BM=bm, BN=bn, BC=bc,
                            num_warps=nw, num_stages=stg)
                        return o
                    return f

                def cublas_path():
                    col = torch.empty(rows, 7 * C, device=dev, dtype=dt)
                    sc._im2col_kernel[(B, triton.cdiv(H + T, 32), triton.cdiv(C, 128))](
                        x, pb, a1, ib1, hist, slots, pos, par, col, T, S, C=C, K=7, D=D, H=H,
                        HAS_BIAS=True, HAS_SNAKE=True, BU=32, BC=128)
                    return torch.mm(col, w1)
                r["im2col+cublas_conv"] = bench(cublas_path)
                r["addmm_w2"] = bench(lambda: torch.addmm(x, x, w2))
                best = None
                for cfg in itertools.product((64, 128), (64, 128, 256), (32, 64), (4, 8), (2, 3, 4)):
                    bm, bn, bc, nw, stg = cfg
                    if C % bc or bn > C or (bm * bc + bc * bn) * 2 * stg > 98000:
                        continue
                    try:
                        ms = bench(igemm(cfg), iters=8)
                    except Exception:  # noqa: BLE001
                        continue
                    if best is None or ms < best[1]:
                        best = (cfg, ms)
                r["igemm_best"] = best
                if C in (96, 192, 384):
                    bestf = None
                    for cfg in itertools.product((32, 64, 128), (32, 64), (4, 8), (2, 3)):
                        bm, bc, nw, stg = cfg
                        if C % bc:
                            continue
                        o = torch.empty_like(x)

                        def f(cfg=cfg, o=o):
                            bm, bc, nw, stg = cfg
                            sc._res_unit_kernel[(triton.cdiv(rows, bm),)](
                                x, hist, slots, pos, par, pb, a1, ib1, w1, b1, a2, ib2, w2, o, T, S, rows,
                                C=C, K=7, D=D, BM=bm, BC=bc, num_warps=nw, num_stages=stg)
                        try:
                            ms = bench(f, iters=8)
                        except Exception:  # noqa: BLE001
                            continue
                        if bestf is None or ms < bestf[1]:
                            bestf = (cfg, ms)
                    r["fused_best"] = bestf
                res[f"d{D}"] = r
                print(C, D, json.dumps(r), flush=True)
            out[str(C)] = res
    json.dump(out, open("/root/results/unit_micro.json", "w"), indent=1)


if __name__ == "__main__":
    main()
