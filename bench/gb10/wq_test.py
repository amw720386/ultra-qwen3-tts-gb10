"""Correctness and speed of uqt.wq kernels vs bf16 cuBLAS at Qwen3-TTS layer shapes.

    python -m bench.gb10.wq_test
"""
import json

import torch

from uqt.wq import QW, lin

SHAPES = {  # name: (N, K)
    "talker_qkv": (4096, 2048), "talker_o": (2048, 2048), "talker_gu": (12288, 2048), "talker_down": (2048, 6144),
    "head": (3072, 2048), "cp_qkv": (4096, 1024), "cp_gu": (6144, 1024), "cp_down": (1024, 3072),
}


def bench(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters  # ms


def main():
    torch.manual_seed(0)
    out = {}
    with torch.inference_mode():
        for name, (N, K) in SHAPES.items():
            w = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
            row = {}
            # cycle through enough copies (> 128 MB) that weights stream from DRAM as in real decode
            ncopy = max(4, int(256e6 // (N * K * 2)))
            ws = [w.clone() for _ in range(ncopy)]
            it = {"i": 0}

            def nxt(lst):
                it["i"] = (it["i"] + 1) % len(lst)
                return lst[it["i"]]
            for kind in ("fp8", "fp4"):
                q = QW(w, kind)
                qs = [QW(x, kind) for x in ws]
                wd = q.dequant()
                for M in (1, 8, 32, 128):
                    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                    ref = a @ wd.T
                    got = lin(a, q)
                    rel = float((got.float() - ref.float()).norm() / ref.float().norm())
                    qerr = float((ref.float() - (a @ w.T).float()).norm() / (a @ w.T).float().norm())
                    tq = bench(lambda: lin(a, nxt(qs)))
                    tb = bench(lambda: torch.nn.functional.linear(a, nxt(ws)))
                    row[f"{kind}_M{M}"] = {"kernel_vs_dequant_rel": round(rel, 5), "quant_err_vs_bf16": round(qerr, 4),
                                           "ms": round(tq, 4), "bf16_ms": round(tb, 4), "speedup": round(tb / tq, 2),
                                           "GBps": round(q.nbytes / tq / 1e6, 1)}
            out[name] = row
            del ws, qs
            torch.cuda.empty_cache()
            print(name, json.dumps({k: (v["kernel_vs_dequant_rel"], v["speedup"], v["GBps"]) for k, v in row.items()}), flush=True)
    json.dump(out, open("/root/results/wq_test.json", "w"), indent=1)


if __name__ == "__main__":
    main()
