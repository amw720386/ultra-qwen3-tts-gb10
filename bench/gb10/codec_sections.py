"""Per-section GPU time of one eager streaming codec call (CUDA events between stages).

    python -m bench.gb10.codec_sections --cases 256x2 --modes ref,fused
"""
import argparse
import json
from collections import defaultdict

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="256x2")
    ap.add_argument("--modes", default="ref,fused")
    args = ap.parse_args()
    from qwen_tts import Qwen3TTSModel

    import uqt.streaming_codec as sc

    torch.set_grad_enabled(False)
    tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0",
                                        dtype=torch.bfloat16, attn_implementation="sdpa")
    dec = tts.model.speech_tokenizer.model.decoder
    out = {}
    with torch.inference_mode():
        sd = sc.StreamingCodecDecoder(dec, num_slots=256)
        for case in args.cases.split(","):
            B, F = (int(x) for x in case.split("x"))
            codes = torch.randint(0, 2048, (B, F, 16), device="cuda", dtype=torch.int32)
            slots = torch.arange(B, device="cuda", dtype=torch.int32)
            pos = torch.full((B,), 40, device="cuda", dtype=torch.int32)
            for mode in args.modes.split(","):
                sd.res_unit = sd.fused_trans = mode == "fused"
                for _ in range(3):
                    sd(codes, slots, pos)
                acc = defaultdict(float)
                n = 5
                for _ in range(n):
                    sd.marks = []
                    sd(codes, slots, pos)
                    torch.cuda.synchronize()
                    m = sd.marks
                    for (a, ea), (b, eb) in zip(m, m[1:]):
                        acc[b] += ea.elapsed_time(eb) / n
                sd.marks = None
                tot = sum(acc.values())
                out[f"{case}/{mode}"] = {"total_ms": round(tot, 3), **{k: round(v, 3) for k, v in acc.items()}}
                print(case, mode, json.dumps(out[f"{case}/{mode}"]), flush=True)
    json.dump(out, open("/root/results/codec_sections.json", "w"), indent=1)


if __name__ == "__main__":
    main()
