"""Batched decode attention with a BF16 vs an FP8 (e4m3) KV cache at talker shapes, per tile config.

    python -m bench.gb10.kv8_micro
"""
import itertools
import json

import torch

import uqt.kernels as K
from bench.gb10.battn_micro import bench


def main():
    H, KVH, HD, T = 16, 8, 128, 512
    out = {}
    K.BATTN_KIND = "dot"
    with torch.inference_mode():
        for B, L in itertools.product((64, 128, 256), (200, 400)):
            kc = torch.randn(B, KVH, T, HD, device="cuda", dtype=torch.bfloat16)
            vc = torch.randn_like(kc)
            q = torch.randn(B, H, HD, device="cuda", dtype=torch.bfloat16)
            pos = torch.randint(L // 2, L, (B,), device="cuda", dtype=torch.int64)
            k8, v8 = kc.to(torch.float8_e4m3fn), vc.to(torch.float8_e4m3fn)
            K.BATTN_DOT_CFG = (64, 8, 1)
            ref = K.bdec_attn(q, kc, vc, pos, HD ** -0.5)
            row = {"bf16_ms": round(bench(lambda: K.bdec_attn(q, kc, vc, pos, HD ** -0.5)), 4)}
            best = None
            for cfg in itertools.product((32, 64, 128), (4, 8), (1, 2, 3)):
                K.BATTN_DOT_CFG = cfg
                try:
                    got = K.bdec_attn(q, k8, v8, pos, HD ** -0.5)
                    t = bench(lambda: K.bdec_attn(q, k8, v8, pos, HD ** -0.5), iters=15)
                except Exception:  # noqa: BLE001
                    continue
                err = float((got.float() - ref.float()).abs().max())
                if best is None or t < best[1]:
                    best = (cfg, round(t, 4), round(err, 4))
            row["fp8_best"] = best
            row["speedup"] = round(row["bf16_ms"] / best[1], 2)
            out[f"B{B}-L{L}"] = row
            print(B, L, json.dumps(row), flush=True)
    json.dump(out, open("/root/results/kv8_micro.json", "w"), indent=1)


if __name__ == "__main__":
    main()
