"""Benchmark the uqt engines directly on a rented box (no Modal): single-stream timings and a
batched step-time scan. Prints one JSON line per measurement and writes results to --out.

    python -m bench.gb10.bench_local --variant CustomVoice --out /root/results/gb10-single.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch

TEXTS = [
    "Hello, how can I help you today?",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Please confirm your appointment for Thursday at three thirty in the afternoon, or call us to reschedule.",
    "Speech synthesis has improved dramatically, and modern systems can now produce natural, expressive voices in real time on a single GPU.",
]


def sync():
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="CustomVoice")
    ap.add_argument("--out", default="/root/results/gb10-bench.json")
    ap.add_argument("--batch", default="1,8,32,64,128,256")
    ap.add_argument("--autotune", type=int, default=0)
    ap.add_argument("--skip-single", type=int, default=0)
    args = ap.parse_args()
    from qwen_tts import Qwen3TTSModel

    from uqt.engine import Engine

    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cap": torch.cuda.get_device_capability(),
           "pdl": not os.environ.get("UQT_NO_PDL")}
    torch.set_grad_enabled(False)
    tts = Qwen3TTSModel.from_pretrained(f"Qwen/Qwen3-TTS-12Hz-1.7B-{args.variant}", device_map="cuda:0",
                                        dtype=torch.bfloat16, attn_implementation="sdpa")
    with torch.inference_mode():
        if not args.skip_single:
            t0 = time.time()
            eng = Engine(tts, max_len=2048, autotune=bool(args.autotune))
            p = eng.build_prompt("custom", text=TEXTS[1], language="English", speaker="Ryan")
            eng.warmup(p, Ks=(1, 2, 4, 8), max_bucket=1024, prefill_max=512)
            for _ in range(2):
                for _w in eng.stream(p):
                    pass
            out["single_warmup_s"] = time.time() - t0
            gen = {}
            for K in (1, 2, 8):
                eng.prefill(p); sync(); t = time.perf_counter(); n = 0
                while n < 48:
                    eng.step_frames(K); n += K
                sync(); gen[K] = (time.perf_counter() - t) * 1000 / n
            out["gen_ms_per_frame"] = gen
            sync(); t = time.perf_counter()
            for _ in range(10):
                eng.prefill(p)
            sync(); out["prefill_ms"] = (time.perf_counter() - t) * 100
            ttfa, rtf = [], []
            for rep in range(3):
                for tx in TEXTS:
                    sync(); t = time.perf_counter()
                    pr = eng.build_prompt("custom", text=tx, language="English", speaker="Ryan")
                    first, n = None, 0
                    for w in eng.stream(pr):
                        if first is None:
                            first = time.perf_counter() - t
                        n += len(w)
                    tot = time.perf_counter() - t
                    ttfa.append(first * 1000); rtf.append(tot / (n / 24000))
            out["single_ttfa_ms_p50"] = float(np.median(ttfa))
            out["single_rtf_p50"] = float(np.median(rtf))
            print("single", json.dumps({k: out[k] for k in ("gen_ms_per_frame", "prefill_ms", "single_ttfa_ms_p50", "single_rtf_p50")}), flush=True)
            del eng
            torch.cuda.empty_cache()
        # batched step-time scan
        from uqt.batch_engine import BatchEngine
        eng = Engine(tts, max_len=2048, streaming_codec=False, autotune=False)
        Bs = [int(b) for b in args.batch.split(",")]
        be = BatchEngine(eng, max_slots=max(Bs), max_len=512)
        t0 = time.time()
        be.warmup(Ks=(2,), Bs=[b for b in (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512) if b <= max(Bs)])
        out["batch_warmup_s"] = time.time() - t0
        p0 = eng.build_prompt("custom", text=TEXTS[1], language="English", speaker="Ryan")
        scan = {}
        for B in Bs:
            for s in range(be.S):
                be.info[s] = None
            be.admit_many([(s, p0, 10_000, {"id": 0}) for s in range(B)])
            sync()
            g, d = [], []
            for it in range(10):
                t1 = time.perf_counter(); slots, chunk, valid = be.generate(2); t2 = time.perf_counter()
                be.decode(slots, chunk, [2] * len(slots)); t3 = time.perf_counter()
                if it >= 2:
                    g.append(t2 - t1); d.append(t3 - t2)
                for s in slots:
                    be.info[s]["done"] = False
            gm, dm = float(np.median(g)) * 1000, float(np.median(d)) * 1000
            scan[B] = {"gen_ms_step": gm, "dec_ms_step": dm, "audio_s_per_s": B * 0.16 / ((gm + dm) / 1000)}
            print("scan", B, json.dumps(scan[B]), flush=True)
        out["scan"] = scan
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
