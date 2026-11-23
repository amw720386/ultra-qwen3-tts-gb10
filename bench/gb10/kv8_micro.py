"""Batched decode attention over BF16, plain FP8 (e4m3) and per-row-scaled INT8 KV caches at talker
shapes (16 q heads, 8 kv heads, head dim 128). Keys are scaled to the talker's range (|k| up to ~270,
where plain e4m3 error on K moves the logits by whole units).

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
    K.BATTN_DOT_CFG = (64, 8, 1)
    with torch.inference_mode():
        for B, L in itertools.product((64, 128, 256), (200, 400)):
            kc = (torch.randn(B, KVH, T, HD, device="cuda") * 30).to(torch.bfloat16)
            vc = torch.randn(B, KVH, T, HD, device="cuda", dtype=torch.bfloat16)
            q = (torch.randn(B, H, HD, device="cuda") * 0.3).to(torch.bfloat16)
            pos = torch.randint(L // 2, L, (B,), device="cuda", dtype=torch.int64)
            sc = HD ** -0.5
            ref = K.bdec_attn(q.float().bfloat16(), kc, vc, pos, sc).float()
            k8, v8 = kc.to(torch.float8_e4m3fn), vc.to(torch.float8_e4m3fn)
            (kq, ks), (vq, vs) = K.quant_rows(kc), K.quant_rows(vc)
            row = {"bf16_ms": round(bench(lambda: K.bdec_attn(q, kc, vc, pos, sc)), 4),
                   "fp8_ms": round(bench(lambda: K.bdec_attn(q, k8, v8, pos, sc)), 4),
                   "int8_ms": round(bench(lambda: K.bdec_attn_q(q, kq, vq, ks, vs, pos, sc)), 4)}
            den = ref.norm()
            row["fp8_relerr"] = round(float((K.bdec_attn(q, k8, v8, pos, sc).float() - ref).norm() / den), 4)
            row["int8_relerr"] = round(float((K.bdec_attn_q(q, kq, vq, ks, vs, pos, sc).float() - ref).norm() / den), 4)
            out[f"B{B}-L{L}"] = row
            print(B, L, json.dumps(row), flush=True)
    json.dump(out, open("/root/results/kv8_micro.json", "w"), indent=1)


if __name__ == "__main__":
    main()
