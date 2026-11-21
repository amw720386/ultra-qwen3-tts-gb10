"""Closed-loop multi-stream load test on one H100.

C clients each send requests back to back (staggered start). A stream is real-time if every
audio chunk reaches the client before playback (started at the first chunk) needs it.

    modal run bench/load_test.py --variant CustomVoice --concs 1,8,16,32,64
"""
import json
import os
import time

import modal

app = modal.App("uqt-ms-load")
data_vol = modal.Volume.from_name("uqt-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("uqt-hf-cache", create_if_missing=True)
VOLS = {"/data": data_vol, "/root/.cache/huggingface": hf_vol}
CU128 = "https://download.pytorch.org/whl/cu128"
img = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("sox", "libsox-dev", "ffmpeg", "libsndfile1")
    .pip_install("torch==2.9.1", "torchaudio==2.9.1", extra_index_url=CU128)
    .pip_install("qwen-tts==0.1.1", "soundfile", "numpy")
    .add_local_python_source("uqt")
)


def _pct(v, q):
    import numpy as np
    return float(np.percentile(v, q)) if len(v) else None


@app.function(image=img, gpu="H100!", volumes=VOLS, timeout=7200)
def load(variant: str = "CustomVoice", concs: str = "1,8,16,32,64", reqs_per_client: int = 3, K: int = 2,
         max_slots: int = 64, max_len: int = 1024, admit_per_iter: int = 4, save_wavs: int = 0):
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    from uqt.batch_engine import BatchEngine
    from uqt.engine import Engine

    out = {"variant": variant, "K": K, "max_slots": max_slots, "max_len": max_len}
    with torch.inference_mode():
        tts = Qwen3TTSModel.from_pretrained(f"Qwen/Qwen3-TTS-12Hz-1.7B-{variant}", device_map="cuda:0",
                                            dtype=torch.bfloat16, attn_implementation="sdpa")
        eng = Engine(tts, max_len=2048, streaming_codec=False, autotune=False)
        t = time.perf_counter()
        be = BatchEngine(eng, max_slots=max_slots, max_len=max_len)
        man = json.load(open("/data/manifest.json"))
        texts = [m["target_text"] for m in man]
        voices = []
        if variant == "Base":
            for m in man[:4]:
                voices.append(tts.create_voice_clone_prompt(ref_audio=m["prompt_wav"], ref_text=m["prompt_text"]))

        def prompt_for(i, text):
            if variant == "Base":
                return eng.build_prompt("clone", text=text, language="English", voice_clone_prompt=voices[i % len(voices)])
            return eng.build_prompt("custom", text=text, language="English", speaker="Ryan")

        # warm prefill graphs for every prompt-length bucket in the test set
        seen_b = set()
        for i, tx in enumerate(texts):
            p = prompt_for(i, tx)
            b = eng._bucket(p.embeds.shape[0], eng.PF_BUCKETS)
            if b not in seen_b:
                seen_b.add(b)
                eng.prefill(p)
            if variant == "Base" and i < len(voices):
                be._prime(0, p.ref_codes)
        out["warmup_s"] = be.warmup(Ks=(K,)) + (time.perf_counter() - t)

        # ---- step-time scan: B identical streams, generation and decode separately
        scan = {}
        p0 = prompt_for(0, texts[0])
        for B in [b for b in (1, 2, 4, 8, 16, 32, 64, 128) if b <= max_slots]:
            for s in range(max_slots):
                be.info[s] = None
            for s in range(B):
                be.admit(s, p0, max_frames=10_000)
            torch.cuda.synchronize()
            gen_t, dec_t = [], []
            for it in range(12):
                t1 = time.perf_counter()
                slots, chunk, valid = be.generate(K)
                t2 = time.perf_counter()
                be.decode(slots, chunk, [K] * len(slots))
                t3 = time.perf_counter()
                if it >= 2:
                    gen_t.append(t2 - t1); dec_t.append(t3 - t2)
                for s in slots:  # keep streams alive
                    be.info[s]["done"] = False
            scan[B] = {"gen_ms_per_step": 1000 * float(np.median(gen_t)), "dec_ms_per_step": 1000 * float(np.median(dec_t)),
                       "ms_per_frame_all_streams": 1000 * float(np.median(gen_t) + np.median(dec_t)) / K,
                       "rt_capacity_streams": B * (0.08 * K) / float(np.median(gen_t) + np.median(dec_t))}
            print("scan", B, scan[B])
        for s in range(max_slots):
            be.info[s] = None
        out["scan"] = scan

        # ---- closed-loop runs
        results = {}
        for C in [int(c) for c in concs.split(",")]:
            for s in range(max_slots):
                be.info[s] = None
            rng = np.random.default_rng(C)
            order = rng.permutation(len(texts))
            stagger = 6.0 / max(C, 1)
            pending = [(i * stagger, i, 0) for i in range(C)]  # (arrival, client, nth request)
            recs, nxt_text = [], 0
            t0 = time.perf_counter()
            iters, iter_ms = 0, []
            while pending or be.active():
                now = time.perf_counter() - t0
                pending.sort()
                admitted = 0
                while pending and pending[0][0] <= now and admitted < admit_per_iter:
                    slot = be.free_slot()
                    if slot is None:
                        break
                    arr, client, nth = pending.pop(0)
                    ti = int(order[nxt_text % len(order)]); nxt_text += 1
                    rec = {"client": client, "nth": nth, "arrival": arr, "text_idx": ti, "chunks": []}
                    be.admit(slot, prompt_for(ti, texts[ti]), max_frames=400, meta=rec)
                    rec["admitted"] = time.perf_counter() - t0
                    admitted += 1
                if not be.active():
                    if pending:
                        time.sleep(max(0.0, min(0.002, pending[0][0] - (time.perf_counter() - t0))))
                    continue
                ts = time.perf_counter()
                slots, chunk, valid = be.generate(K)
                wavs = be.decode(slots, chunk, valid)
                tn = time.perf_counter() - t0
                iter_ms.append((time.perf_counter() - ts) * 1000)
                iters += 1
                for s, w in zip(slots, wavs):
                    it = be.info[s]
                    rec = it["meta"]
                    if len(w):
                        rec["chunks"].append((tn, len(w)))
                        if save_wavs and len(recs) < save_wavs:
                            rec.setdefault("audio", []).append(w)
                    if it["done"]:
                        rec["done"] = tn
                        recs.append(rec)
                        be.release(s)
                        if rec["nth"] + 1 < reqs_per_client:
                            pending.append((tn, rec["client"], rec["nth"] + 1))
            wall = time.perf_counter() - t0
            ttfa, rtf, late, under = [], [], [], 0
            tot_audio = 0.0
            for r in recs:
                if not r["chunks"]:
                    continue
                t_first = r["chunks"][0][0]
                ttfa.append(t_first - r["arrival"])
                dur = sum(n for _, n in r["chunks"]) / 24000
                tot_audio += dur
                rtf.append((r["done"] - t_first) / max(dur, 1e-6))
                acc, worst = 0.0, -1e9
                for tc, n in r["chunks"]:
                    worst = max(worst, tc - (t_first + acc))
                    acc += n / 24000
                late.append(worst)
                under += worst > 0.0
            res = {"C": C, "requests": len(recs), "wall_s": wall, "audio_s": tot_audio,
                   "throughput_audio_s_per_s": tot_audio / wall,
                   "ttfa_ms_p50": 1000 * _pct(ttfa, 50), "ttfa_ms_p90": 1000 * _pct(ttfa, 90), "ttfa_ms_p99": 1000 * _pct(ttfa, 99),
                   "rtf_p50": _pct(rtf, 50), "rtf_p90": _pct(rtf, 90),
                   "underrun_frac": under / max(len(late), 1), "max_lateness_ms_p90": 1000 * _pct(late, 90),
                   "iter_ms_p50": _pct(iter_ms, 50), "iter_ms_p90": _pct(iter_ms, 90)}
            print("load", json.dumps(res))
            results[C] = res
            if save_wavs:
                os.makedirs(f"/data/gen/ms-{variant}-C{C}", exist_ok=True)
                for r in recs[:save_wavs]:
                    if r.get("audio"):
                        sf.write(f"/data/gen/ms-{variant}-C{C}/{man[r['text_idx']]['id']}.wav", np.concatenate(r["audio"]), 24000)
        out["closed_loop"] = results
    data_vol.commit()
    return out


@app.local_entrypoint()
def main(variant: str = "CustomVoice", concs: str = "1,8,16,32,64", reqs_per_client: int = 3, frames: int = 2,
         max_slots: int = 64, max_len: int = 1024, tag: str = ""):
    K = frames
    r = load.remote(variant, concs, reqs_per_client, K, max_slots, max_len)
    os.makedirs("results/phase3", exist_ok=True)
    name = f"results/phase3/load-{variant}-K{K}-S{max_slots}{('-' + tag) if tag else ''}.json"
    json.dump(r, open(name, "w"), indent=1)
    print(json.dumps({k: v for k, v in r.items() if k != "closed_loop"}, indent=1))
    for C, res in r["closed_loop"].items():
        print(C, json.dumps(res))
