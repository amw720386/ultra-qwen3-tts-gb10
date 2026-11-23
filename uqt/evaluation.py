"""Whisper large-v3 WER + WavLM speaker-similarity proxy for a directory of generated wavs."""
import json
import os


def run_eval(wav_dir: str, items: list, with_sim: bool, batch_size: int = 16) -> dict:
    import jiwer
    import librosa
    import numpy as np
    import torch
    from transformers import AutoFeatureExtractor, WavLMForXVector, pipeline

    items = [it for it in items if os.path.exists(f"{wav_dir}/{it['id']}.wav")]
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3",
                   torch_dtype=torch.float16, device="cuda:0")
    norm = asr.tokenizer.normalize
    audios = [librosa.load(f"{wav_dir}/{it['id']}.wav", sr=16000)[0] for it in items]
    gk = {"language": "english", "task": "transcribe", "num_beams": 1}
    # clips over Whisper's 30 s window go through the chunked path one at a time: mixed in a batch with
    # short clips the pipeline fails ("elements of the batch contain different keys")
    short = [i for i, a in enumerate(audios) if len(a) <= 30 * 16000]
    hyps = [None] * len(audios)
    for i, h in zip(short, asr([{"raw": audios[i], "sampling_rate": 16000} for i in short],
                               batch_size=batch_size, generate_kwargs=gk)):
        hyps[i] = h
    for i in (i for i in range(len(audios)) if hyps[i] is None):
        hyps[i] = asr({"raw": audios[i], "sampling_rate": 16000}, chunk_length_s=30, batch_size=1,
                      generate_kwargs=gk)
    refs_n = [norm(it["target_text"]) for it in items]
    hyps_n = [norm(h["text"]) for h in hyps]
    rows = []
    for it, r, h in zip(items, refs_n, hyps_n):
        o = jiwer.process_words(r, h)
        rows.append({"id": it["id"], "ref": r, "hyp": h, "n_ref": len(r.split()),
                     "errors": o.substitutions + o.deletions + o.insertions})
    sims = None
    if with_sim:
        fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").cuda().eval()

        def emb(a):
            x = fe(a, sampling_rate=16000, return_tensors="pt").to("cuda")
            with torch.no_grad():
                e = sv(**x).embeddings
            return torch.nn.functional.normalize(e, dim=-1)[0]

        sims = []
        for row, it, a in zip(rows, items, audios):
            s = float(torch.dot(emb(a), emb(librosa.load(it["prompt_wav"], sr=16000)[0])))
            row["sim"] = s
            sims.append(s)
    tot_err = sum(r["errors"] for r in rows)
    tot_ref = sum(r["n_ref"] for r in rows)
    return {"n_eval": len(rows), "wer_corpus": tot_err / max(tot_ref, 1),
            "sim_mean": None if sims is None else float(np.mean(sims)), "rows": rows}


def paired_bootstrap(a_rows, b_rows, key="errors", n_boot=10000, seed=0):
    """Corpus-WER difference (b - a) with a 95% bootstrap CI over utterances paired by id."""
    import random

    A = {r["id"]: r for r in a_rows}
    B = {r["id"]: r for r in b_rows}
    ids = sorted(set(A) & set(B))
    rng = random.Random(seed)

    def wer(rows_, sel):
        e = sum(rows_[i][key] for i in sel)
        n = sum(rows_[i]["n_ref"] for i in sel)
        return e / max(n, 1)

    base = wer(B, ids) - wer(A, ids)
    diffs = []
    for _ in range(n_boot):
        s = [ids[rng.randrange(len(ids))] for _ in ids]
        diffs.append(wer(B, s) - wer(A, s))
    diffs.sort()
    return {"n": len(ids), "diff": base, "ci95": (diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)])}


if __name__ == "__main__":  # quick local self-test of the bootstrap (pure python)
    a = [{"id": str(i), "errors": i % 3, "n_ref": 10} for i in range(50)]
    b = [{"id": str(i), "errors": (i % 3) + (1 if i % 10 == 0 else 0), "n_ref": 10} for i in range(50)]
    print(json.dumps(paired_bootstrap(a, b, n_boot=2000)))
