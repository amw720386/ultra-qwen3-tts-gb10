"""Baseline speed + accuracy for Qwen3-TTS-12Hz-1.7B on one H100.

Systems: official `qwen-tts` (reference) and `faster-qwen3-tts` (CUDA graphs).
Variants: Base (voice clone, LibriSpeech test-clean prompts) and CustomVoice (preset speaker).
Accuracy: Whisper large-v3 WER (Whisper English normalizer) and a WavLM speaker-similarity proxy.

    modal run bench/baseline.py              # everything
    modal run bench/baseline.py --n 20       # smaller accuracy set
"""
import hashlib
import json
import os
import time

import modal

app = modal.App("uqt-baseline")
data_vol = modal.Volume.from_name("uqt-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("uqt-hf-cache", create_if_missing=True)
VOLS = {"/data": data_vol, "/root/.cache/huggingface": hf_vol}
GPU = "H100!"  # "!" stops Modal from silently upgrading to H200
CU128 = "https://download.pytorch.org/whl/cu128"

base = modal.Image.debian_slim(python_version="3.12").apt_install(
    "sox", "libsox-dev", "ffmpeg", "libsndfile1"
)
ref_img = (
    base.pip_install("torch==2.8.0", "torchaudio==2.8.0", extra_index_url=CU128)
    .pip_install("qwen-tts==0.1.1", "soundfile", "numpy")
)
fast_img = (
    base.pip_install("torch==2.10.0", "torchaudio==2.10.0", extra_index_url=CU128)
    # transformers 5.20 turned Mimi's `upsample` into a read-only property, breaking qwen-tts-hf.
    .pip_install("faster-qwen3-tts==0.5.4", "transformers==5.15.1", "soundfile", "numpy")
)
eval_img = (
    base.pip_install("torch==2.8.0", "torchaudio==2.8.0", extra_index_url=CU128)
    .pip_install("transformers==4.57.3", "accelerate", "jiwer==3.1.0", "librosa", "soundfile", "numpy")
)
prep_img = base.pip_install("soundfile", "numpy")

LS_URLS = [
    "https://us.openslr.org/resources/12/test-clean.tar.gz",
    "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "https://openslr.trmal.net/resources/12/test-clean.tar.gz",
]

# Fixed sentences for latency measurement (short -> long).
TIMING_TEXTS = [
    "Hello, how can I help you today?",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Please confirm your appointment for Thursday at three thirty in the afternoon, or call us to reschedule.",
    "Speech synthesis has improved dramatically, and modern systems can now produce natural, expressive voices in real time on a single GPU.",
    "When the storm finally passed, the villagers walked down to the harbor, counted the boats that had survived the night, and began repairing the ones that had not.",
]


def seed_for(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def tts_text(s: str) -> str:
    s = s.strip().lower()
    return (s[:1].upper() + s[1:] + ".") if s else s


def to_np(x):
    import numpy as np

    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)


def first_voiced_sample(wav, sr, thresh_db=-40.0, win_ms=10):
    """Index of the first 10 ms window whose RMS is above thresh_db dBFS."""
    import numpy as np

    win = max(1, int(sr * win_ms / 1000))
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if peak == 0.0:
        return None
    thr = 10 ** (thresh_db / 20)
    for i in range(0, len(wav) - win + 1, win):
        if float(np.sqrt(np.mean(wav[i : i + win] ** 2))) >= thr:
            return i
    return None


# --------------------------------------------------------------------------- data


@app.function(image=prep_img, volumes=VOLS, timeout=3600, cpu=4)
def prep(n_pairs: int = 60, seed: int = 0):
    """LibriSpeech test-clean voice-clone pairs (same speaker, different utterance)."""
    import glob
    import random
    import tarfile
    import urllib.request

    import soundfile as sf

    man_path = "/data/manifest.json"
    if os.path.exists(man_path):
        man = json.load(open(man_path))
        if len(man) >= n_pairs:
            return f"manifest exists ({len(man)} pairs)"
    root = "/tmp/ls"
    os.makedirs(root, exist_ok=True)
    tgz = "/tmp/test-clean.tar.gz"
    last = None
    for url in LS_URLS:
        try:
            urllib.request.urlretrieve(url, tgz)
            with tarfile.open(tgz) as tf:
                tf.extractall(root)
            break
        except Exception as e:
            last = e
            print(f"download failed from {url}: {e!r}")
    else:
        raise RuntimeError(f"all LibriSpeech mirrors failed: {last!r}")
    utts = {}
    for trans in glob.glob(f"{root}/LibriSpeech/test-clean/*/*/*.trans.txt"):
        d = os.path.dirname(trans)
        for line in open(trans):
            uid, text = line.strip().split(" ", 1)
            path = f"{d}/{uid}.flac"
            utts.setdefault(uid.split("-")[0], []).append(
                {"uid": uid, "text": text, "path": path, "dur": sf.info(path).duration}
            )
    rng = random.Random(seed)
    speakers = sorted(utts)
    rng.shuffle(speakers)
    os.makedirs("/data/ls/prompts", exist_ok=True)
    os.makedirs("/data/ls/gt", exist_ok=True)
    pairs = []
    rnd = 0
    while len(pairs) < n_pairs and rnd < 10:
        for spk in speakers:
            if len(pairs) >= n_pairs:
                break
            u = sorted(utts[spk], key=lambda x: x["uid"])
            targets = [x for x in u if 4.0 <= x["dur"] <= 10.0]
            prompts = [x for x in u if 3.0 <= x["dur"] <= 8.0]
            if len(targets) <= rnd or len(prompts) < 2:
                continue
            tgt = targets[(rnd * 7) % len(targets)]
            cands = [p for p in prompts if p["uid"] != tgt["uid"] and p["uid"] not in {q["prompt_uid"] for q in pairs}]
            if not cands or any(q["id"] == tgt["uid"] for q in pairs):
                continue
            pr = cands[rng.randrange(len(cands))]
            pw, gw = f"/data/ls/prompts/{pr['uid']}.wav", f"/data/ls/gt/{tgt['uid']}.wav"
            a, sr = sf.read(pr["path"]); sf.write(pw, a, sr)
            a, sr = sf.read(tgt["path"]); sf.write(gw, a, sr)
            pairs.append({
                "id": tgt["uid"], "speaker": spk,
                "prompt_uid": pr["uid"], "prompt_wav": pw, "prompt_text": tts_text(pr["text"]),
                "target_text": tts_text(tgt["text"]), "gt_wav": gw, "gt_dur": tgt["dur"],
            })
        rnd += 1
    json.dump(pairs, open(man_path, "w"), indent=1)
    data_vol.commit()
    return f"wrote {len(pairs)} pairs from {len({p['speaker'] for p in pairs})} speakers"


# --------------------------------------------------------------------------- generation


def _run_accuracy_set(tag, synth, man, out_dir):
    import soundfile as sf
    import torch

    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for it in man:
        torch.manual_seed(seed_for(it["id"]))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            wav, sr = synth(it)
        except Exception as e:  # keep going; failures count against the system
            rows.append({"id": it["id"], "error": repr(e)[:300]})
            continue
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        sf.write(f"{out_dir}/{it['id']}.wav", wav, sr)
        rows.append({"id": it["id"], "wall_s": dt, "audio_s": len(wav) / sr})
    ok = [r for r in rows if "error" not in r]
    tot_wall = sum(r["wall_s"] for r in ok)
    tot_audio = sum(r["audio_s"] for r in ok)
    return {
        "tag": tag, "n": len(man), "n_ok": len(ok), "errors": [r for r in rows if "error" in r][:5],
        "accuracy_set_wall_s": tot_wall, "accuracy_set_audio_s": tot_audio,
        "accuracy_set_rtf": tot_wall / max(tot_audio, 1e-9),  # wall / audio, lower = faster
        "rows": rows,
    }


def _gpu_info():
    import torch

    return {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda}


@app.function(image=ref_img, gpu=GPU, volumes=VOLS, timeout=7200)
def gen_ref(variant: str, n: int):
    """Official qwen-tts. Non-streaming, so time to first audio == time to full audio."""
    import inspect

    import numpy as np
    import torch
    from qwen_tts import Qwen3TTSModel

    mid = f"Qwen/Qwen3-TTS-12Hz-1.7B-{variant}"
    t_load = time.perf_counter()
    m = Qwen3TTSModel.from_pretrained(mid, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    t_load = time.perf_counter() - t_load
    man = json.load(open("/data/manifest.json"))[:n]
    voice = man[0]

    def synth(it):
        if variant == "Base":
            wavs, sr = m.generate_voice_clone(
                text=it["target_text"], language="English", ref_audio=it["prompt_wav"], ref_text=it["prompt_text"]
            )
        else:
            wavs, sr = m.generate_custom_voice(text=it["target_text"], language="English", speaker="Ryan")
        return to_np(wavs[0]), sr

    synth(voice)  # warmup
    timing = []
    for rep in range(2):
        for i, text in enumerate(TIMING_TEXTS):
            it = dict(voice, target_text=text, id=f"t{i}")
            torch.manual_seed(seed_for(f"t{i}-{rep}"))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            wav, sr = synth(it)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            v = first_voiced_sample(wav, sr)
            timing.append({"text_idx": i, "rep": rep, "full_s": dt, "audio_s": len(wav) / sr,
                           "ttfa_ms": dt * 1000, "lead_silence_ms": (v or 0) / sr * 1000})
    acc = _run_accuracy_set(f"ref-{variant}", synth, man, f"/data/gen/ref-{variant}")
    res = {
        "system": "qwen-tts (official)", "variant": variant, "model": mid, "attn": "sdpa", "load_s": t_load,
        "sigs": {k: str(inspect.signature(getattr(m, k))) for k in ("generate_voice_clone", "generate_custom_voice") if hasattr(m, k)},
        **_gpu_info(), "timing": timing,
        "ttfa_ms_median": float(np.median([t["ttfa_ms"] for t in timing])),
        "rtf_median": float(np.median([t["full_s"] / t["audio_s"] for t in timing])),
        **acc,
    }
    os.makedirs("/data/results", exist_ok=True)
    json.dump(res, open(f"/data/results/gen-ref-{variant}.json", "w"), indent=1)
    data_vol.commit()
    return res


@app.function(image=fast_img, gpu=GPU, volumes=VOLS, timeout=7200)
def gen_fast(variant: str, n: int):
    """faster-qwen3-tts (static cache + CUDA graphs), streaming for latency."""
    import inspect

    import numpy as np
    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    mid = f"Qwen/Qwen3-TTS-12Hz-1.7B-{variant}"
    t_load = time.perf_counter()
    m = FasterQwen3TTS.from_pretrained(mid, device="cuda", dtype=torch.bfloat16, attn_implementation="eager", max_seq_len=2048)
    if hasattr(m, "warmup"):
        try:
            m.warmup(prefill_len=100)
        except Exception as e:
            print("warmup failed:", e)
    t_load = time.perf_counter() - t_load
    sigs = {k: str(inspect.signature(getattr(m, k))) for k in dir(m) if k.startswith("generate_")}
    print(json.dumps(sigs, indent=1))
    man = json.load(open("/data/manifest.json"))[:n]
    voice = man[0]

    speaker = "Ryan"
    if variant == "CustomVoice" and hasattr(m, "get_supported_speakers"):
        try:
            sp = list(m.get_supported_speakers())
            speaker = next((s for s in sp if s.lower() == "ryan"), sp[0])
        except Exception as e:
            print("get_supported_speakers failed:", e)

    def kw(it):
        if variant == "Base":
            return dict(text=it["target_text"], language="English", ref_audio=it["prompt_wav"], ref_text=it["prompt_text"])
        return dict(text=it["target_text"], language="English", speaker=speaker)

    gen_name = "generate_voice_clone" if variant == "Base" else "generate_custom_voice"

    def synth(it):
        wavs, sr = getattr(m, gen_name)(**kw(it))
        return to_np(wavs[0] if isinstance(wavs, (list, tuple)) else wavs), sr

    def stream(it, chunk_size):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        arrivals, chunks, sr = [], [], None
        for audio_chunk, sr, _timing in getattr(m, gen_name + "_streaming")(**kw(it), chunk_size=chunk_size):
            torch.cuda.synchronize()
            arrivals.append(time.perf_counter() - t0)
            chunks.append(to_np(audio_chunk))
        total = time.perf_counter() - t0
        wav = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        v = first_voiced_sample(wav, sr)
        ttfa_voiced = None
        if v is not None:
            acc = 0
            for a, c in zip(arrivals, chunks):
                acc += len(c)
                if acc > v:
                    ttfa_voiced = a * 1000
                    break
        return {"ttfa_ms": arrivals[0] * 1000 if arrivals else None, "ttfa_voiced_ms": ttfa_voiced,
                "full_s": total, "audio_s": len(wav) / sr if sr else 0, "n_chunks": len(chunks),
                "lead_silence_ms": (v or 0) / sr * 1000 if sr else None}

    synth(voice)  # warmup (also caches the voice prompt)
    stream(dict(voice, target_text=TIMING_TEXTS[0]), 8)
    timing = []
    for cs in (2, 4, 8):
        for rep in range(2):
            for i, text in enumerate(TIMING_TEXTS):
                torch.manual_seed(seed_for(f"t{i}-{rep}"))
                r = stream(dict(voice, target_text=text), cs)
                timing.append({"chunk_size": cs, "text_idx": i, "rep": rep, **r})
    # Non-streaming throughput on the timing texts too
    nonstream = []
    for i, text in enumerate(TIMING_TEXTS):
        torch.manual_seed(seed_for(f"t{i}-ns"))
        torch.cuda.synchronize(); t0 = time.perf_counter()
        wav, sr = synth(dict(voice, target_text=text))
        torch.cuda.synchronize()
        nonstream.append({"text_idx": i, "full_s": time.perf_counter() - t0, "audio_s": len(wav) / sr})
    acc = _run_accuracy_set(f"fast-{variant}", synth, man, f"/data/gen/fast-{variant}")

    def med(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.median(vals)) if vals else None

    by_cs = {}
    for cs in (2, 4, 8):
        rows = [t for t in timing if t["chunk_size"] == cs]
        by_cs[cs] = {"ttfa_ms_median": med(rows, "ttfa_ms"), "ttfa_voiced_ms_median": med(rows, "ttfa_voiced_ms"),
                     "rtf_median": float(np.median([r["full_s"] / r["audio_s"] for r in rows if r["audio_s"]]))}
    res = {
        "system": "faster-qwen3-tts 0.5.4", "variant": variant, "model": mid, "speaker": speaker, "load_s": t_load,
        "sigs": sigs, **_gpu_info(), "timing": timing, "nonstream": nonstream, "by_chunk_size": by_cs,
        "nonstream_rtf_median": float(np.median([r["full_s"] / r["audio_s"] for r in nonstream])),
        **acc,
    }
    os.makedirs("/data/results", exist_ok=True)
    json.dump(res, open(f"/data/results/gen-fast-{variant}.json", "w"), indent=1)
    data_vol.commit()
    return res


# --------------------------------------------------------------------------- evaluation


@app.function(image=eval_img, gpu="L40S", volumes=VOLS, timeout=3600)
def evaluate(tag: str, n: int):
    """Whisper large-v3 WER + WavLM x-vector speaker similarity (proxy) for one generated set."""
    import jiwer
    import librosa
    import numpy as np
    import torch
    from transformers import AutoFeatureExtractor, WavLMForXVector, pipeline

    data_vol.reload()
    man = json.load(open("/data/manifest.json"))[:n]
    wav_dir = "/data/ls/gt" if tag == "gt" else f"/data/gen/{tag}"
    items = [it for it in man if os.path.exists(f"{wav_dir}/{it['id']}.wav")]
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3",
                   torch_dtype=torch.float16, device="cuda:0")
    norm = asr.tokenizer.normalize
    audios = [librosa.load(f"{wav_dir}/{it['id']}.wav", sr=16000)[0] for it in items]
    hyps = asr([{"raw": a, "sampling_rate": 16000} for a in audios], batch_size=16,
               generate_kwargs={"language": "english", "task": "transcribe", "num_beams": 1})
    refs_n = [norm(it["target_text"]) for it in items]
    hyps_n = [norm(h["text"]) for h in hyps]
    per = [jiwer.wer(r, h) if r else 0.0 for r, h in zip(refs_n, hyps_n)]

    sims = None
    if not tag.endswith("CustomVoice"):
        fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").cuda().eval()

        def emb(a):
            x = fe(a, sampling_rate=16000, return_tensors="pt").to("cuda")
            with torch.no_grad():
                e = sv(**x).embeddings
            return torch.nn.functional.normalize(e, dim=-1)[0]

        sims = []
        for it, a in zip(items, audios):
            p = librosa.load(it["prompt_wav"], sr=16000)[0]
            sims.append(float(torch.dot(emb(a), emb(p))))
    res = {
        "tag": tag, "asr": "openai/whisper-large-v3 (greedy, en, Whisper EnglishTextNormalizer)",
        "sim_model": None if sims is None else "microsoft/wavlm-base-plus-sv (proxy, not Seed-TTS-eval's WavLM-large)",
        "n_eval": len(items), "n_missing": len(man) - len(items),
        "wer_corpus": jiwer.wer(refs_n, hyps_n), "wer_mean_utt": float(np.mean(per)),
        "n_utt_wer_gt_20pct": int(sum(p > 0.2 for p in per)),
        "sim_mean": None if sims is None else float(np.mean(sims)),
        "rows": [{"id": it["id"], "ref": r, "hyp": h, "wer": w, **({"sim": s} if sims else {})}
                 for it, r, h, w, s in zip(items, refs_n, hyps_n, per, sims or [None] * len(items))],
    }
    os.makedirs("/data/results", exist_ok=True)
    json.dump(res, open(f"/data/results/eval-{tag}.json", "w"), indent=1)
    data_vol.commit()
    return res


# --------------------------------------------------------------------------- driver


@app.local_entrypoint()
def main(n: int = 60, systems: str = "ref-Base,ref-CustomVoice,fast-Base,fast-CustomVoice"):
    print(prep.remote(n_pairs=n))
    gt_eval = evaluate.spawn("gt", n)
    calls = {}
    for s in systems.split(","):
        kind, variant = s.split("-", 1)
        calls[s] = (gen_ref if kind == "ref" else gen_fast).spawn(variant, n)
    os.makedirs("results/h100/baseline", exist_ok=True)
    evals = {"gt": gt_eval}
    for s, c in calls.items():
        try:
            r = c.get()
        except Exception as e:
            print(f"[{s}] generation FAILED: {e!r}")
            continue
        json.dump(r, open(f"results/h100/baseline/gen-{s}.json", "w"), indent=1)
        print(f"[{s}] done: n_ok={r['n_ok']}/{r['n']} acc-set RTF={r['accuracy_set_rtf']:.3f}")
        evals[s] = evaluate.spawn(s, n)
    for s, c in evals.items():
        try:
            r = c.get()
        except Exception as e:
            print(f"[{s}] eval FAILED: {e!r}")
            continue
        json.dump(r, open(f"results/h100/baseline/eval-{s}.json", "w"), indent=1)
        print(f"[{s}] WER={r['wer_corpus']*100:.2f}% sim={r['sim_mean']}")
