"""Profile the stateful streaming codec decoder (uqt.streaming_codec) on the box.

For each (B streams, F frames per call): time per call (graphed), us per stream-frame, and a kernel
breakdown of one eager call (grouped by kernel name).

    python -m bench.gb10.profile_codec --cases 64x2,256x2,64x16,256x16
"""
import argparse
import json
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="64x2,256x2,64x16,128x8")
    ap.add_argument("--top", type=int, default=14)
    args = ap.parse_args()
    from qwen_tts import Qwen3TTSModel

    from uqt.streaming_codec import StreamingCodecDecoder, StreamingDecodeGraphs

    torch.set_grad_enabled(False)
    tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0",
                                        dtype=torch.bfloat16, attn_implementation="sdpa")
    dec = tts.model.speech_tokenizer.model.decoder
    print("decoder dtype", next(dec.parameters()).dtype, flush=True)
    out = {}
    with torch.inference_mode():
        sd = StreamingCodecDecoder(dec, num_slots=256)
        for case in args.cases.split(","):
            B, F = (int(x) for x in case.split("x"))
            codes = torch.randint(0, 2048, (B, F, 16), device="cuda", dtype=torch.int32)
            slots = torch.arange(B, device="cuda", dtype=torch.int32)
            pos = torch.full((B,), 40, device="cuda", dtype=torch.int32)
            g = StreamingDecodeGraphs(sd, [B], frames=F)
            for _ in range(3):
                g(codes, slots, pos)
            torch.cuda.synchronize()
            st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            st.record()
            for _ in range(10):
                g(codes, slots, pos)
            en.record()
            torch.cuda.synchronize()
            ms = st.elapsed_time(en) / 10
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                sd(codes, slots, pos)
                torch.cuda.synchronize()
            ev = [e for e in prof.key_averages() if e.device_type.name == "CUDA"]
            tot = sum(e.self_device_time_total for e in ev) / 1000
            top = sorted(ev, key=lambda e: -e.self_device_time_total)[: args.top]
            out[case] = {"graph_ms": ms, "us_per_stream_frame": ms * 1000 / (B * F), "gpu_ms_eager": tot,
                         "top": [(e.key[:90], e.count, round(e.self_device_time_total / 1000, 3)) for e in top]}
            print(case, json.dumps(out[case], indent=1), flush=True)
    json.dump(out, open("/root/results/profile_codec.json", "w"), indent=1)


if __name__ == "__main__":
    main()
