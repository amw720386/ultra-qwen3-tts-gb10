"""Upper bound on what prompt riding can save, before porting it: the extra GPU time of P more rows in one
1-frame step, against the packed prefill that those rows would replace (both as serving runs them).

    python -m bench.gb10.ride_bound
"""
import json
import time

import torch


def main():
    from qwen_tts import Qwen3TTSModel

    from uqt.batch_engine import BatchEngine
    from uqt.engine import Engine

    torch.set_grad_enabled(False)
    tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0",
                                        dtype=torch.bfloat16, attn_implementation="sdpa")
    man = json.load(open("/root/manifest.json"))
    out = {"step1_ms": {}, "pack_ms": {}, "tokens": []}
    with torch.inference_mode():
        E = Engine(tts, max_len=2048, streaming_codec=False, autotune=False, weights="hyb")
        be = BatchEngine(E, max_slots=512, max_len=512)
        E.build_prompt("custom", text=man[0]["target_text"], language="English", speaker="Ryan")
        out["tokens"] = [int(E.build_prompt("custom", text=m["target_text"], language="English",
                                            speaker="Ryan").embeds.shape[0]) for m in man[:60]]
        be.pos.fill_(150)
        for B in (32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384):
            g = be._graph(1, B, 512)
            torch.cuda.synchronize()
            ts = []
            for i in range(30):
                be.pos[:B].fill_(150)
                torch.cuda.synchronize()
                t = time.perf_counter()
                g.replay()
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t)
            out["step1_ms"][B] = round(1000 * sorted(ts)[len(ts) // 2], 2)
            print("step1", B, out["step1_ms"][B], flush=True)
        be._init_pack()
        be.pk_seg.zero_()
        be.warm_pack()
        for Tb, g in be.pk_graphs.items():
            ts = []
            for i in range(30):
                torch.cuda.synchronize()
                t = time.perf_counter()
                g.replay()
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t)
            out["pack_ms"][Tb] = round(1000 * sorted(ts)[len(ts) // 2], 2)
            print("pack", Tb, out["pack_ms"][Tb], flush=True)
    print("tokens per prompt: median", sorted(out["tokens"])[30], "max", max(out["tokens"]), flush=True)
    json.dump(out, open("/root/results/ride_bound.json", "w"), indent=1)


if __name__ == "__main__":
    main()
