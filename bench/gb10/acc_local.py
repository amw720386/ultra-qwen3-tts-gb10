"""Accuracy on the box: generate the 300-text set with the single-stream engine for several weight
formats, then Whisper large-v3 WER for each (same seeds per text across formats).

    python -m bench.gb10.acc_local --weights bf16,fp8 --n 300 --manifest /root/manifest.json
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="bf16,fp8")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--manifest", default="/root/manifest.json")
    ap.add_argument("--out", default="/root/results/acc")
    args = ap.parse_args()
    from qwen_tts import Qwen3TTSModel

    from uqt.common import seed_for
    from uqt.engine import Engine
    from uqt.evaluation import run_eval

    man = json.load(open(args.manifest))[: args.n]
    torch.set_grad_enabled(False)
    tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0",
                                        dtype=torch.bfloat16, attn_implementation="sdpa")
    summary = {}
    for w in args.weights.split(","):
        d = f"{args.out}/{w}"
        os.makedirs(d, exist_ok=True)
        with torch.inference_mode():
            eng = Engine(tts, max_len=2048, weights=w)
            p = eng.build_prompt("custom", text=man[0]["target_text"], language="English", speaker="Ryan")
            eng.warmup(p, Ks=(1, 2, 4, 8), max_bucket=1024, prefill_max=512)
            for it in man:
                torch.manual_seed(seed_for(it["id"]))
                pr = eng.build_prompt("custom", text=it["target_text"], language="English", speaker="Ryan")
                wav = np.concatenate(list(eng.stream(pr)) or [np.zeros(0, np.float32)])
                sf.write(f"{d}/{it['id']}.wav", wav, 24000)
            del eng
            torch.cuda.empty_cache()
        print("generated", w, flush=True)
    del tts
    torch.cuda.empty_cache()
    for w in args.weights.split(","):
        r = run_eval(f"{args.out}/{w}", man, with_sim=False)
        bad = sum(1 for x in r["rows"] if x["errors"] / max(x["n_ref"], 1) > 0.5)
        summary[w] = {"n": r["n_eval"], "wer": r["wer_corpus"], "catastrophic": bad}
        json.dump(r, open(f"{args.out}/eval-{w}.json", "w"), indent=1)
        print("eval", w, json.dumps(summary[w]), flush=True)
    json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
