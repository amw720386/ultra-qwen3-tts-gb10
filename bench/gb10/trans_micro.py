"""Microbenchmark of the block-entry transposed conv on GB10 shapes: unfused (snake + cuBLAS + col2im)
vs the fused Triton kernel (with/without in-kernel SnakeBeta) across tile configs.

    python -m bench.gb10.trans_micro --frames 512
"""
import argparse
import itertools
import json

import torch
import triton

import uqt.streaming_codec as sc

BLOCKS = [(1536, 768, 8, 4), (768, 384, 5, 32), (384, 192, 4, 160), (192, 96, 3, 640)]  # CI, CO, R, rows/frame


def bench(fn, iters=20):
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
    args = ap.parse_args()
    out = {}
    dev, dt = "cuda", torch.bfloat16
    B = 256
    T_f = args.frames // B
    with torch.inference_mode():
        for CI, CO, R, rpf in BLOCKS:
            T = T_f * rpf
            rows = B * T
            x = torch.randn(rows, CI, device=dev, dtype=dt)
            pb = torch.randn(CI, device=dev) * 0.1
            a = torch.rand(CI, device=dev) + 0.5
            ib = torch.rand(CI, device=dev) + 0.5
            w = torch.randn(CI, 2 * R * CO, device=dev, dtype=dt) * 0.02
            bias = torch.randn(CO, device=dev, dtype=dt)
            sh = torch.zeros(2, B + 1, 1, CI, device=dev, dtype=dt)
            prev = torch.zeros(2, B + 1, R, CO, device=dev, dtype=dt)
            slots = torch.arange(B, device=dev, dtype=torch.int32)
            pos = torch.full((B,), 40, device=dev, dtype=torch.int32)
            par = torch.zeros(B + 1, device=dev, dtype=torch.int32)
            res = {}

            def snake_pass():
                s = torch.empty_like(x)
                bu, bc = 32, min(128, triton.next_power_of_2(CI))
                one = torch.zeros(1, device=dev, dtype=torch.int32)
                sc._im2col_kernel[(1, triton.cdiv(rows, bu), triton.cdiv(CI, bc))](
                    x, pb, a, ib, x, one, one, par, s, rows, B + 1, C=CI, K=1, D=1, H=0,
                    HAS_BIAS=True, HAS_SNAKE=True, BU=bu, BC=bc)
                return s

            def unfused():
                s = snake_pass()
                zt = torch.mm(s, w)
                o = torch.empty(rows * R, CO, device=dev, dtype=dt)
                bc = min(128, triton.next_power_of_2(CO))
                br = max(1, 2048 // bc)
                sc._col2im_kernel[(triton.cdiv(rows, br), R, triton.cdiv(CO, bc))](
                    zt, bias, prev, slots, pos, par, o, T, B + 1, rows, R=R, C=CO, BR=br, BC=bc)
                return o
            res["unfused"] = bench(unfused)
            res["snake_pass"] = bench(snake_pass)
            s_pre = snake_pass()
            ref = unfused().float()
            best = {}
            for snake in (True, False):
                for bm, bn, bk, nw, stg in itertools.product((64, 128), (32, 64, 128), (32, 64), (4, 8), (2, 3)):
                    if bn > CO or CI % bk or (bm * bk + bk * bn) * 2 * stg > 96000:
                        continue
                    o = torch.empty(rows * R, CO, device=dev, dtype=dt)
                    src = x if snake else s_pre

                    def fused():
                        sc._trans_kernel[(triton.cdiv(rows, bm) * R * triton.cdiv(CO, bn),)](
                            src, pb, a, ib, sh, slots, pos, par, w, bias, o, T, B + 1, rows,
                            CI=CI, CO=CO, R=R, BM=bm, BN=bn, BK=bk, SNAKE=snake, num_warps=nw, num_stages=stg)
                    try:
                        ms = bench(fused, iters=10)
                    except Exception as e:  # noqa: BLE001
                        continue
                    key = (snake, bm, bn, bk, nw, stg)
                    if snake not in best or ms < best[snake][1]:
                        err = float((o.float() - ref).abs().max())
                        best[snake] = (key, ms, err)
            res["fused_snake"] = best.get(True)
            res["fused_presnaked_gemm_only"] = best.get(False)
            out[f"{CI}->{CO}x{R}"] = res
            print(CI, CO, R, json.dumps(res), flush=True)
    json.dump(out, open("/root/results/trans_micro.json", "w"), indent=1)


if __name__ == "__main__":
    main()
