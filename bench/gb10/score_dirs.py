"""Whisper large-v3 scoring of many clip directories in one process (the model is loaded once).

    python -m bench.gb10.score_dirs /root/results/garble     # every sub-directory with wavs and no eval.json
"""
import glob
import json
import os
import sys

import jiwer
import librosa
import torch
from transformers import pipeline

root, BS = sys.argv[1], int(os.environ.get("BS", "48"))
man = {x["id"]: x["target_text"] for x in json.load(open(os.environ.get("MANIFEST", "/root/manifest.json")))[:300]}
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float16, device="cuda:0")
norm = asr.tokenizer.normalize
gk = {"language": "english", "task": "transcribe", "num_beams": 1}
for d in sorted(glob.glob(f"{root}/*/")):
    if os.path.exists(f"{d}/eval.json"):
        continue
    ids = [i for i in man if os.path.exists(f"{d}/{i}.wav")]
    au = [librosa.load(f"{d}/{i}.wav", sr=16000)[0] for i in ids]
    short = [k for k, a in enumerate(au) if len(a) <= 30 * 16000]
    hyp = [None] * len(au)
    for k, h in zip(short, asr([{"raw": au[k], "sampling_rate": 16000} for k in short], batch_size=BS, generate_kwargs=gk)):
        hyp[k] = h
    for k in (k for k in range(len(au)) if hyp[k] is None):  # over Whisper's window: one at a time, chunked
        hyp[k] = asr({"raw": au[k], "sampling_rate": 16000}, chunk_length_s=30, batch_size=1, generate_kwargs=gk)
    rows = []
    for i, h in zip(ids, hyp):
        r, t = norm(man[i]), norm(h["text"])
        o = jiwer.process_words(r, t)
        rows.append({"id": i, "ref": r, "hyp": t, "n_ref": len(r.split()), "errors": o.substitutions + o.deletions + o.insertions})
    bad = [x for x in rows if x["errors"] / max(x["n_ref"], 1) > 0.5]
    w = sum(x["errors"] for x in rows) / max(sum(x["n_ref"] for x in rows), 1)
    json.dump({"summary": {"n": len(rows), "wer": w, "catastrophic": len(bad)}, "n_eval": len(rows), "wer_corpus": w, "rows": rows},
              open(f"{d}/eval.json", "w"))
    print(f"{os.path.basename(d.rstrip('/'))} n {len(rows)} wer {100 * w:.2f} garbled {len(bad)}", flush=True)
