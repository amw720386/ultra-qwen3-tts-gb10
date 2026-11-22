"""W8A8 Triton GEMM on GB10: bf16 activations quantized to e4m3 per row per K-block inside the kernel,
FP8 x FP8 tensor-core dot (fp32 accumulate, block scale applied per K-block), per-channel weight scale.
Compared with bf16 cuBLAS and the FP8 weight-only kernel at decode batch sizes; weights cycled so they
stream from DRAM.

    python -m bench.gb10.w8a8_triton_test
"""
import itertools
import json

import torch
import triton
import triton.language as tl

from uqt.wq import fp8_mm_big

SHAPES = {"talker_qkv": (4096, 2048), "talker_o": (2048, 2048), "talker_gu": (12288, 2048),
          "talker_down": (2048, 6144), "cp_qkv": (4096, 1024), "cp_gu": (6144, 1024), "cp_down": (1024, 3072)}


@triton.jit
def _w8a8_kernel(a_ptr, w_ptr, ws_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    nm = tl.cdiv(M, BM)
    pid = tl.program_id(0)
    pid_m = pid % nm
    pid_n = pid // nm
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :], mask=rm[:, None] < M, other=0.0).to(tl.float32)
        sa = tl.maximum(tl.max(tl.abs(a), axis=1), 1e-12) / 448.0
        aq = (a / sa[:, None]).to(tl.float8e4nv)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :], mask=rn[:, None] < N, other=0.0)
        acc += tl.dot(aq, tl.trans(w)) * sa[:, None]
    ws = tl.load(ws_ptr + rn, mask=rn < N, other=0.0)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], (acc * ws[None, :]).to(tl.bfloat16),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def w8a8(a, w, ws, N, K, cfg):
    bm, bn, bk, nw, st = cfg
    M = a.shape[0]
    out = torch.empty(M, N, device=a.device, dtype=torch.bfloat16)
    _w8a8_kernel[(triton.cdiv(M, bm) * triton.cdiv(N, bn),)](a, w, ws, out, M, N, K, BM=bm, BN=bn, BK=bk,
                                                           num_warps=nw, num_stages=st)
    return out


def bench(fn, iters=30):
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
    cfgs = [c for c in itertools.product((64, 128), (64, 128, 256), (64, 128), (4, 8), (2, 3, 4))
            if (c[0] * c[2] + c[1] * c[2]) * c[4] <= 98000]
    with torch.inference_mode():
        for name, (N, K) in SHAPES.items():
            w = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
            ws = w.float().abs().amax(dim=1).clamp(min=1e-8) / 448.0
            q = (w.float() / ws[:, None]).to(torch.float8_e4m3fn).contiguous()
            n = max(4, int(256e6 // (N * K * 2)))
            bfs = [w.clone() for _ in range(n)]
            qs = [q.clone() for _ in range(n)]
            i = {"k": 0}

            def nxt(lst):
                i["k"] = (i["k"] + 1) % len(lst)
                return lst[i["k"]]
            row = {}
            for M in (64, 128, 192, 256):
                x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                ref = (x.float() @ w.float().T)
                best = None
                for cfg in cfgs:
                    try:
                        t = bench(lambda: w8a8(x, nxt(qs), ws, N, K, cfg), iters=10)
                    except Exception:  # noqa: BLE001
                        continue
                    if best is None or t < best[1]:
                        best = (cfg, t)
                got = w8a8(x, q, ws, N, K, best[0])
                rel = float((got.float() - ref).norm() / ref.norm())
                rel_wo = float((fp8_mm_big(x, q, ws, N, K).float() - ref).norm() / ref.norm())
                tb = bench(lambda: torch.nn.functional.linear(x, nxt(bfs)))
                tw = bench(lambda: fp8_mm_big(x, nxt(qs), ws, N, K))
                row[M] = {"w8a8_ms": round(best[1], 4), "cfg": best[0], "bf16_ms": round(tb, 4), "fp8wo_ms": round(tw, 4),
                          "speedup_vs_bf16": round(tb / best[1], 2), "rel_err": round(rel, 4), "rel_err_wo": round(rel_wo, 4)}
            out[name] = row
            print(name, json.dumps(row), flush=True)
            del bfs, qs
            torch.cuda.empty_cache()
    json.dump(out, open("/root/results/w8a8_triton_test.json", "w"), indent=1)


if __name__ == "__main__":
    main()
