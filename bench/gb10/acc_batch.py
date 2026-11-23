"""Accuracy of the batched serving path on the box: all 300 texts through BatchEngine at full
concurrency (default 256 slots, FP8 weights incl. the large-batch FP8 GEMMs and fused codec), then
Whisper large-v3 WER on the streamed int16 PCM.

    python -m bench.gb10.acc_batch --slots 256 --n 300
"""
import argparse
import json
import os
import time

import numpy as np
import soundfile as sf
import torch


def generate(eng, be, man, d):
    os.makedirs(d, exist_ok=True)
    pend = list(reversed(man))
    owner, pcm = {}, {}
    handle, t0, done = None, time.time(), 0
    while pend or handle is not None or any(it is not None for it in be.info):
        batch = []
        while pend and len(batch) < 16:
            slot = next((i for i, it in enumerate(be.info) if it is None and i not in owner), None)
            if slot is None:
                break
            it = pend.pop()
            owner[slot] = it["id"]
            pcm[it["id"]] = []
            batch.append((slot, eng.build_prompt("custom", text=it["target_text"], language="English",
                                                 speaker="Ryan"), 400, {"id": it["id"]}))
        if batch:
            be.admit_many(batch)
        if handle is not None or be.active():
            be.collect(handle) if handle is not None else be.generate(2)
            handle = be.launch(2) if be.active() else None
        for s, b, fin in be.decode_due():
            if b:
                pcm[owner[s]].append(b)
            if fin:
                rid = owner.pop(s)
                wav = np.frombuffer(b"".join(pcm.pop(rid)), dtype=np.int16).astype(np.float32) / 32767.0
                sf.write(f"{d}/{rid}.wav", wav, 24000)
                be.release(s)
                done += 1
    print(f"generated {done} clips in {time.time() - t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=256)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--weights", default="fp8")
    ap.add_argument("--manifest", default="/root/manifest.json")
    ap.add_argument("--out", default="/root/results/acc_batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="", help="comma list; generates all seeds with one warm engine")
    ap.add_argument("--eval-only", action="store_true", help="re-score wavs already in the output dir")
    args = ap.parse_args()
    from qwen_tts import Qwen3TTSModel

    from uqt.batch_engine import BatchEngine
    from uqt.engine import Engine
    from uqt.evaluation import run_eval

    man = json.load(open(args.manifest))[: args.n]
    torch.set_grad_enabled(False)
    # one compiled engine serves every seed; --seeds writes one directory per seed ({out}-s{seed})
    runs = [(int(x), f"{args.out}-s{x}") for x in args.seeds.split(",")] if args.seeds else [(args.seed, args.out)]
    if not args.eval_only:
        tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0",
                                            dtype=torch.bfloat16, attn_implementation="sdpa")
        with torch.inference_mode():
            eng = Engine(tts, max_len=2048, streaming_codec=False, autotune=False, weights=args.weights)
            be = BatchEngine(eng, max_slots=args.slots, max_len=512)
            be.warmup(Ks=(2,))
            for seed, out in runs:
                torch.manual_seed(seed)
                generate(eng, be, man, f"{out}/{args.weights}-S{args.slots}")
            del be, eng
        del tts
    torch.cuda.empty_cache()
    for seed, out in runs:
        d = f"{out}/{args.weights}-S{args.slots}"
        r = run_eval(d, man, with_sim=False)
        bad = [x for x in r["rows"] if x["errors"] / max(x["n_ref"], 1) > 0.5]
        tot_e = sum(x["errors"] for x in r["rows"] if x not in bad)
        tot_n = sum(x["n_ref"] for x in r["rows"] if x not in bad)
        summ = {"slots": args.slots, "weights": args.weights, "n": r["n_eval"], "wer_corpus": r["wer_corpus"],
                "catastrophic": len(bad), "wer_excl_catastrophic": tot_e / max(tot_n, 1)}
        json.dump({"summary": summ, **r}, open(f"{d}/eval.json", "w"), indent=1)
        summ.update(kv=os.environ.get("UQT_KV", "default"), seed=seed)
        json.dump(summ, open(f"{out}-summary.json", "w"), indent=1)  # one file per run directory
        print("eval", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
