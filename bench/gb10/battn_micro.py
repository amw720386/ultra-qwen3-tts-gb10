"""Batched decode attention on GB10: CUDA-core kernel vs the tensor-core (tl.dot) kernel ported from the
H100 backport, across tile configs, at talker shapes (16 q heads, 8 kv heads, head dim 128).

    python -m bench.gb10.battn_micro
"""
import itertools
import json

import torch

import uqt.kernels as K


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
    H, KVH, HD, T = 16, 8, 128, 512
    out = {}
    with torch.inference_mode():
        for B, L in itertools.product((1, 16, 64, 128, 256), (64, 200, 400)):
            S = max(B, 1)
            kc = torch.randn(S, KVH, T, HD, device="cuda", dtype=torch.bfloat16)
            vc = torch.randn_like(kc)
            q = torch.randn(B, H, HD, device="cuda", dtype=torch.bfloat16)
            pos = torch.randint(L // 2, L, (B,), device="cuda", dtype=torch.int64)
            K.BATTN_KIND = "fma"
            ref = K.bdec_attn(q, kc, vc, pos, HD ** -0.5)
            row = {"fma_ms": round(bench(lambda: K.bdec_attn(q, kc, vc, pos, HD ** -0.5)), 4)}
            K.BATTN_KIND = "dot"
            best = None
            for cfg in itertools.product((32, 64, 128), (4, 8), (1, 2, 3)):
                if 2 * cfg[0] * HD * 2 * cfg[2] > 98000:
                    continue
                K.BATTN_DOT_CFG = cfg
                try:
                    got = K.bdec_attn(q, kc, vc, pos, HD ** -0.5)
                    t = bench(lambda: K.bdec_attn(q, kc, vc, pos, HD ** -0.5), iters=15)
                except Exception:  # noqa: BLE001
                    continue
                err = float((got.float() - ref.float()).abs().max())
                if best is None or t < best[1]:
                    best = (cfg, round(t, 4), round(err, 4))
            row["dot_best"] = best
            row["speedup"] = round(row["fma_ms"] / best[1], 2) if best else None
            out[f"B{B}-L{L}"] = row
            print(B, L, json.dumps(row), flush=True)
    json.dump(out, open("/root/results/battn_micro.json", "w"), indent=1)


if __name__ == "__main__":
    main()
