"""Does GB10 (sm_121) run FP8 x FP8 tensor-core GEMMs (torch._scaled_mm, rowwise scales), and how
fast vs bf16 cuBLAS at batched decode sizes? Weights stream from DRAM (cycled copies).

    python -m bench.gb10.w8a8_test
"""
import json

import torch

SHAPES = {"talker_qkv": (4096, 2048), "talker_gu": (12288, 2048), "talker_down": (2048, 6144), "cp_gu": (6144, 1024)}


def w8a8(x, wq, ws):
    sa = x.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-6) / 448.0
    xq = (x.float() / sa).to(torch.float8_e4m3fn)
    return torch._scaled_mm(xq, wq.t(), scale_a=sa, scale_b=ws.view(1, -1), out_dtype=torch.bfloat16)


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
            ws_ = w.float().abs().amax(dim=1).clamp(min=1e-8) / 448.0
            wq = (w.float() / ws_[:, None]).to(torch.float8_e4m3fn).contiguous()
            ncopy = max(4, int(256e6 // (N * K * 2)))
            bf = [w.clone() for _ in range(ncopy)]
            q8 = [wq.clone() for _ in range(ncopy)]
            i = {"k": 0}

            def nxt(lst):
                i["k"] = (i["k"] + 1) % len(lst)
                return lst[i["k"]]
            row = {}
            for M in (64, 128, 256, 512):
                x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                try:
                    ref = x @ w.T
                    got = w8a8(x, wq, ws_)
                    rel = float((got.float() - ref.float()).norm() / ref.float().norm())
                    t8 = bench(lambda: w8a8(x, nxt(q8), ws_))
                    tb = bench(lambda: torch.nn.functional.linear(x, nxt(bf)))
                    row[M] = {"rel_err_vs_bf16": round(rel, 4), "w8a8_ms": round(t8, 3), "bf16_ms": round(tb, 3), "speedup": round(tb / t8, 2)}
                except Exception as e:
                    row[M] = {"error": repr(e)[:200]}
            out[name] = row
            print(name, json.dumps(row), flush=True)
            del bf, q8
            torch.cuda.empty_cache()
    json.dump(out, open("/root/results/w8a8_test.json", "w"), indent=1)


if __name__ == "__main__":
    main()
