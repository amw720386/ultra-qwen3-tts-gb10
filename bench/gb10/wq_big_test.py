"""FP8 weight-only GEMM for 128-512 rows (row-tile-fastest grid) vs bf16 cuBLAS on GB10 talker /
code-predictor shapes; weights cycled through many copies so they stream from DRAM.

    python -m bench.gb10.wq_big_test
"""
import json

import torch

from uqt.wq import fp8_mm_big

SHAPES = {"talker_qkv": (4096, 2048), "talker_o": (2048, 2048), "talker_gu": (12288, 2048),
          "talker_down": (2048, 6144), "cp_qkv": (4096, 1024), "cp_gu": (6144, 1024), "cp_down": (1024, 3072)}


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
    with torch.inference_mode():
        for name, (N, K) in SHAPES.items():
            w = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
            sc = w.float().abs().amax(dim=1).clamp(min=1e-8) / 448.0
            q = (w.float() / sc[:, None]).to(torch.float8_e4m3fn).contiguous()
            n = max(4, int(256e6 // (N * K * 2)))
            bfs = [w.clone() for _ in range(n)]
            qs = [q.clone() for _ in range(n)]
            i = {"k": 0}

            def nxt(lst):
                i["k"] = (i["k"] + 1) % len(lst)
                return lst[i["k"]]
            row = {}
            for M in (160, 192, 256, 320, 512):
                x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                ref = x @ (q.float() * sc[:, None]).to(torch.bfloat16).T
                got = fp8_mm_big(x, q, sc, N, K)
                rel = float((got.float() - ref.float()).norm() / ref.float().norm())
                t8 = bench(lambda: fp8_mm_big(x, nxt(qs), sc, N, K))
                tb = bench(lambda: torch.nn.functional.linear(x, nxt(bfs)))
                row[M] = {"rel": round(rel, 5), "fp8_ms": round(t8, 4), "bf16_ms": round(tb, 4), "speedup": round(tb / t8, 2)}
            out[name] = row
            print(name, json.dumps(row), flush=True)
            del bfs, qs
            torch.cuda.empty_cache()
    json.dump(out, open("/root/results/wq_big_test.json", "w"), indent=1)


if __name__ == "__main__":
    main()
