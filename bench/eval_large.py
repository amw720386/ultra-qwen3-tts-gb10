"""Large multi-seed accuracy comparison: official qwen-tts vs uqt engine.

    modal run bench/baseline.py::prep --n-pairs 300
    modal run bench/eval_large.py --n 300 --seeds 1,2
"""
import json
import os

import modal

app = modal.App("uqt-eval-large")
data_vol = modal.Volume.from_name("uqt-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("uqt-hf-cache", create_if_missing=True)
VOLS = {"/data": data_vol, "/root/.cache/huggingface": hf_vol}
CU128 = "https://download.pytorch.org/whl/cu128"
base = modal.Image.debian_slim(python_version="3.12").apt_install("sox", "libsox-dev", "ffmpeg", "libsndfile1")
img = (base.pip_install("torch==2.9.1", "torchaudio==2.9.1", extra_index_url=CU128)
       .pip_install("qwen-tts==0.1.1", "soundfile", "numpy").add_local_python_source("uqt"))
eval_img = (base.pip_install("torch==2.8.0", "torchaudio==2.8.0", extra_index_url=CU128)
            .pip_install("transformers==4.57.3", "accelerate", "jiwer==3.1.0", "librosa", "soundfile", "numpy")
            .add_local_python_source("uqt"))
SHARD = 50


PREFIX = os.environ.get("UQT_TAG_PREFIX", "L")


def tag_of(system, variant, seed, prefix=None):
    return f"{prefix or PREFIX}-s{seed}-{system}-{variant}"


@app.function(image=img, gpu="H100!", volumes=VOLS, timeout=3600)
def gen_shard(system: str, variant: str, seed: int, start: int, end: int, prefix: str = "L"):
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    from uqt.common import seed_for

    with torch.inference_mode():
        tts = Qwen3TTSModel.from_pretrained(f"Qwen/Qwen3-TTS-12Hz-1.7B-{variant}", device_map="cuda:0",
                                            dtype=torch.bfloat16, attn_implementation="sdpa")
        man = json.load(open("/data/manifest.json"))[start:end]
        out_dir = f"/data/gen/{tag_of(system, variant, seed, prefix)}"
        os.makedirs(out_dir, exist_ok=True)
        eng = None
        if system == "eng":
            from uqt.engine import Engine
            eng = Engine(tts, max_len=2048)
        done = 0
        for it in man:
            if variant == "Base":
                call = dict(text=it["target_text"], language="English",
                            voice_clone_prompt=tts.create_voice_clone_prompt(ref_audio=it["prompt_wav"], ref_text=it["prompt_text"]))
            else:
                call = dict(text=it["target_text"], language="English", speaker="Ryan")
            torch.manual_seed(seed * 1_000_003 + seed_for(it["id"]))
            if eng is None:
                fn = tts.generate_voice_clone if variant == "Base" else tts.generate_custom_voice
                wavs, sr = fn(**call)
                w = wavs[0]
                w = (w.detach().float().cpu().numpy() if hasattr(w, "detach") else np.asarray(w, np.float32)).reshape(-1)
            else:
                kind = "clone" if variant == "Base" else "custom"
                pr = eng.build_prompt(kind, **call)
                w = np.concatenate(list(eng.stream(pr, schedule=(1, 2, 4, 8))) or [np.zeros(0, np.float32)])
                sr = 24000
            sf.write(f"{out_dir}/{it['id']}.wav", w, sr)
            done += 1
        data_vol.commit()
        return done


@app.function(image=eval_img, gpu="L40S", volumes=VOLS, timeout=3600)
def evaluate(tag: str, n: int, with_sim: bool):
    from uqt.evaluation import run_eval

    data_vol.reload()
    man = json.load(open("/data/manifest.json"))[:n]
    res = run_eval(f"/data/gen/{tag}", man, with_sim)
    res["tag"] = tag
    os.makedirs("/data/results", exist_ok=True)
    json.dump(res, open(f"/data/results/eval-{tag}.json", "w"), indent=1)
    data_vol.commit()
    return res


@app.local_entrypoint()
def main(n: int = 300, seeds: str = "1,2", systems: str = "ref,eng", variants: str = "Base,CustomVoice",
         prefix: str = "L", ref_prefix: str = ""):
    from uqt.evaluation import paired_bootstrap

    combos = [(s, v, int(sd)) for s in systems.split(",") for v in variants.split(",") for sd in seeds.split(",")]
    gens = {c: [gen_shard.spawn(*c, a, min(a + SHARD, n), prefix) for a in range(0, n, SHARD)] for c in combos}
    evs = {}
    for c, calls in gens.items():
        k = sum(x.get() for x in calls)
        print(f"generated {tag_of(*c, prefix)}: {k}")
        evs[c] = evaluate.spawn(tag_of(*c, prefix), n, c[1] == "Base")
    res = {c: e.get() for c, e in evs.items()}
    os.makedirs("results/phase2", exist_ok=True)
    summary = {}
    for c, r in res.items():
        json.dump(r, open(f"results/phase2/eval-{tag_of(*c, prefix)}.json", "w"), indent=1)
        summary[tag_of(*c, prefix)] = {"wer": r["wer_corpus"], "sim": r["sim_mean"], "n": r["n_eval"]}
    # pool seeds: rows keyed by (seed, id) so the bootstrap pairs ref/eng on the same utterance and seed
    if ref_prefix:  # reuse earlier official-implementation results from disk
        for v in variants.split(","):
            for sd in seeds.split(","):
                res[("ref", v, int(sd))] = json.load(open(f"results/phase2/eval-{tag_of('ref', v, int(sd), ref_prefix)}.json"))
    syss = set(systems.split(",")) | ({"ref"} if ref_prefix else set())
    for v in variants.split(","):
        pooled = {}
        for s in syss:
            rows = []
            for sd in seeds.split(","):
                rows += [dict(r, id=f"{sd}:{r['id']}") for r in res[(s, v, int(sd))]["rows"]]
            pooled[s] = rows
        if "ref" in pooled and "eng" in pooled:
            summary[f"{v}: eng - ref (pooled seeds)"] = paired_bootstrap(pooled["ref"], pooled["eng"])
    json.dump(summary, open(f"results/phase2/eval-large-summary-{prefix}.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
