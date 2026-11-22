"""Where does a batched step's time go on the box? Kernel-level profile of one eager (ungraphed)
step at batch B, grouped into categories.

    python -m bench.gb10.profile_batch --B 256 --weights fp8
"""
import argparse
import json
import re
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

CATS = [
    ("attention (bmm/softmax)", r"bmm|softmax|gemv.*batched|batched"),
    ("fp8/fp4 weight GEMM (uqt.wq)", r"_fp8_mm|_fp4_mm"),
    ("cuBLAS GEMM", r"gemm|cutlass|nvjet|sm\d+_xmma|cublas"),
    ("code predictor attention/sample", r"_cp_attention|_cp_sample|_silu_mul_rms|_rms_rows"),
    ("codec", r"_im2col|_rvq|_dwconv|_col2im|_conv_out|_attend|_rope_kv|_rms_norm_kernel|_silu_mul_kernel|_bias_gelu|_flip|conv"),
    ("elementwise/fused (inductor)", r"triton_|elementwise|vectorized|reduce|index|copy|fill|cat"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--weights", default="fp8")
    ap.add_argument("--Lb", type=int, default=256)
    args = ap.parse_args()
    from qwen_tts import Qwen3TTSModel

    from uqt.batch_engine import BatchEngine
    from uqt.engine import Engine

    torch.set_grad_enabled(False)
    tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0",
                                        dtype=torch.bfloat16, attn_implementation="sdpa")
    with torch.inference_mode():
        eng = Engine(tts, max_len=2048, streaming_codec=False, autotune=False, weights=args.weights)
        be = BatchEngine(eng, max_slots=args.B, max_len=512)
        p = eng.build_prompt("custom", text="The quick brown fox jumps over the lazy dog near the riverbank.",
                             language="English", speaker="Ryan")
        be.admit_many([(s, p, 10_000, {"id": 0}) for s in range(args.B)])
        for _ in range(3):  # compile/autotune outside the profile
            be._bframe(args.B, args.Lb)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                be._bframe(args.B, args.Lb)
            torch.cuda.synchronize()
        ev = [e for e in prof.key_averages() if e.device_type.name == "CUDA"]
        tot = sum(e.self_device_time_total for e in ev) / 3 / 1000
        cat = defaultdict(float)
        for e in ev:
            name = e.key
            c = next((c for c, rx in CATS if re.search(rx, name, re.I)), "other")
            cat[c] += e.self_device_time_total / 3 / 1000
        top = sorted(ev, key=lambda e: -e.self_device_time_total)[:15]
        out = {"B": args.B, "Lb": args.Lb, "weights": args.weights, "gpu_ms_per_frame": tot,
               "by_category_ms": {k: round(v, 2) for k, v in sorted(cat.items(), key=lambda kv: -kv[1])},
               "top": [(e.key[:80], e.count // 3, round(e.self_device_time_total / 3 / 1000, 3)) for e in top]}
        print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
