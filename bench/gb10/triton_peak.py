"""Can a Triton bf16 GEMM beat cuBLAS on GB10 (cuBLAS reaches ~100 TFLOPS, mma.sync peak ~213)?
Sweeps tile configs of a plain Triton matmul at large and codec-like shapes.

    python -m bench.gb10.triton_peak
"""
import itertools
import json

import torch
import triton
import triton.language as tl


@triton.jit
def _mm(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    npm, npn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    gid = pid // (GM * npn)
    fm = gid * GM
    gsz = min(npm - fm, GM)
    pm = fm + (pid % (GM * npn)) % gsz
    pn = (pid % (GM * npn)) // gsz
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None].to(tl.int64) * K + (k0 + rk)[None, :], mask=rm[:, None] < M, other=0.0)
        b = tl.load(b_ptr + (k0 + rk)[:, None] * N + rn[None, :], mask=rn[None, :] < N, other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + rm[:, None].to(tl.int64) * N + rn[None, :], acc.to(tl.bfloat16),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


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
    out = {}
    shapes = [(8192, 8192, 8192), (327680, 192, 576), (81920, 384, 1536), (16384, 768, 3840), (512, 1536, 12288),
              (256, 2048, 4096), (256, 2048, 12288)]
    with torch.inference_mode():
        for M, K, N in shapes:
            a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
            fl = 2 * M * N * K
            tc = bench(lambda: a @ b)
            best = None
            for bm, bn, bk, nw, st in itertools.product((64, 128, 256), (64, 128, 256), (32, 64), (4, 8), (2, 3, 4)):
                if (bm * bk + bk * bn) * 2 * st > 98000 or bm * bn > 128 * 256:
                    continue
                c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
                try:
                    t = bench(lambda: _mm[(triton.cdiv(M, bm) * triton.cdiv(N, bn),)](
                        a, b, c, M, N, K, BM=bm, BN=bn, BK=bk, GM=8, num_warps=nw, num_stages=st), iters=8)
                except Exception:  # noqa: BLE001
                    continue
                if best is None or t < best[1]:
                    best = ((bm, bn, bk, nw, st), t)
            r = {"cublas_tflops": round(fl / tc / 1e9, 1), "triton_tflops": round(fl / best[1] / 1e9, 1), "cfg": best[0]}
            out[f"{M}x{K}x{N}"] = r
            print(M, K, N, json.dumps(r), flush=True)
    json.dump(out, open("/root/results/triton_peak.json", "w"), indent=1)


if __name__ == "__main__":
    main()
