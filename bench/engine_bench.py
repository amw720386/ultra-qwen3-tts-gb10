"""Correctness + speed of the uqt engine on one H100.

    modal run bench/engine_bench.py::parity_all           # greedy code parity vs official qwen-tts
    modal run bench/engine_bench.py::bench_all --n 60     # latency + accuracy-set generation
    modal run bench/baseline.py::evaluate --tag eng-Base --n 60   # Whisper WER / SIM
"""
import json
import os
import time

import modal

app = modal.App("uqt-engine-bench")
data_vol = modal.Volume.from_name("uqt-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("uqt-hf-cache", create_if_missing=True)
VOLS = {"/data": data_vol, "/root/.cache/huggingface": hf_vol}
GPU = "H100!"
SCHEDULES = {"ramp1": (1, 2, 4, 8), "ramp2": (2, 4, 8), "fixed2_8": (2, 8), "fixed4": (4,)}
CU128 = "https://download.pytorch.org/whl/cu128"

img = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("sox", "libsox-dev", "ffmpeg", "libsndfile1")
    .pip_install("torch==2.9.1", "torchaudio==2.9.1", extra_index_url=CU128)
    .pip_install("qwen-tts==0.1.1", "soundfile", "numpy")
    .run_commands("python -c \"import torch, triton; print(torch.__version__, triton.__version__)\"")
    .add_local_python_source("uqt")
)


def _load(variant, dtype_name, attn="sdpa"):
    import torch
    from qwen_tts import Qwen3TTSModel

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name]
    return Qwen3TTSModel.from_pretrained(f"Qwen/Qwen3-TTS-12Hz-1.7B-{variant}", device_map="cuda:0",
                                         dtype=dtype, attn_implementation=attn)


def _cases(variant, man, k):
    if variant == "Base":
        return [dict(text=it["target_text"], language="English", ref_audio=it["prompt_wav"], ref_text=it["prompt_text"])
                for it in man[:k]]
    from uqt.common import TIMING_TEXTS
    return [dict(text=t, language="English", speaker="Ryan") for t in TIMING_TEXTS[:k]]


@app.function(image=img, gpu=GPU, volumes=VOLS, timeout=3600)
def parity(variant: str, dtype_name: str, k: int = 3, max_new: int = 120):
    """Greedy decoding: codes from the engine (eager and graphed) vs official generate()."""
    import torch
    from uqt.engine import Engine, SamplingCfg

    tts = _load(variant, dtype_name)
    man = json.load(open("/data/manifest.json"))
    eng = Engine(tts, max_len=2048, sampling=SamplingCfg(greedy=True, cp_greedy=True))
    dec = tts.model.speech_tokenizer
    out = []
    for i, case in enumerate(_cases(variant, man, k)):
        kind = "clone" if variant == "Base" else "custom"
        extra = {}
        if kind == "clone":
            items = tts.create_voice_clone_prompt(ref_audio=case["ref_audio"], ref_text=case["ref_text"])
            call = dict(text=case["text"], language="English", voice_clone_prompt=items)
            R = int(items[0].ref_code.shape[0])
        else:
            call = dict(case)
            R = 0
        # reference codes: intercept the decoder input
        got = {}
        orig = dec.decode

        def spy(enc, _orig=orig):
            got["codes"] = enc[0]["audio_codes"] if isinstance(enc, list) else enc["audio_codes"]
            return _orig(enc)

        dec.decode = spy
        try:
            fn = tts.generate_voice_clone if kind == "clone" else tts.generate_custom_voice
            fn(**call, do_sample=False, subtalker_dosample=False, max_new_tokens=max_new)
        finally:
            dec.decode = orig
        ref = got["codes"][R:].to("cuda")
        prompt = eng.build_prompt(kind, **call)
        e_eager = eng.generate_codes(prompt, max_frames=max_new - 1, chunk=4, use_graph=False)
        e_graph = eng.generate_codes(prompt, max_frames=max_new - 1, chunk=4, use_graph=True)

        def cmp(a, b):
            n = min(len(a), len(b))
            if n == 0:
                return {"common": 0}
            eq = (a[:n] == b[:n]).all(-1)
            bad = (~eq).nonzero()
            return {"common": n, "len_a": len(a), "len_b": len(b),
                    "first_mismatch_frame": int(bad[0]) if len(bad) else None,
                    "frac_frames_equal": float(eq.float().mean()),
                    "frac_codes_equal": float((a[:n] == b[:n]).float().mean())}

        out.append({"case": i, "ref_vs_eager": cmp(ref, e_eager), "eager_vs_graph": cmp(e_eager, e_graph)})
        print(json.dumps(out[-1]))
    return {"variant": variant, "dtype": dtype_name, "results": out}


@app.local_entrypoint()
def parity_all():
    calls = {(v, d): parity.spawn(v, d) for v in ("Base", "CustomVoice") for d in ("fp32", "bf16")}
    os.makedirs("results/phase1", exist_ok=True)
    res = {}
    for key, c in calls.items():
        try:
            res["-".join(key)] = c.get()
        except Exception as e:
            res["-".join(key)] = {"error": repr(e)}
        print(key, json.dumps(res["-".join(key)])[:2000])
    json.dump(res, open("results/phase1/parity.json", "w"), indent=1)


@app.function(image=img, gpu=GPU, volumes=VOLS, timeout=7200)
def bench(variant: str, n: int, left: int = 25, tag: str = ""):
    import numpy as np
    import soundfile as sf
    import torch
    from uqt.common import TIMING_TEXTS, first_voiced_sample, seed_for
    from uqt.engine import Engine

    tts = _load(variant, "bf16")
    man = json.load(open("/data/manifest.json"))[:n]
    eng = Engine(tts, max_len=2048)
    kind = "clone" if variant == "Base" else "custom"
    voice_cache = {}

    def call_for(it, text):
        if kind == "clone":
            key = it["prompt_wav"]
            if key not in voice_cache:
                voice_cache[key] = tts.create_voice_clone_prompt(ref_audio=it["prompt_wav"], ref_text=it["prompt_text"])
            return dict(text=text, language="English", voice_clone_prompt=voice_cache[key])
        return dict(text=text, language="English", speaker="Ryan")

    voice = man[0]
    t0 = time.perf_counter()
    warm_prompt = eng.build_prompt(kind, **call_for(voice, TIMING_TEXTS[0]))
    eng.warmup(warm_prompt, Ks=(1, 2, 4, 8), max_bucket=1024, prefill_max=512,
               dec_windows=[k + left for k in (1, 2, 4, 8)])
    warm_s = time.perf_counter() - t0
    for sch in SCHEDULES.values():  # capture decoder windows used by each schedule
        for _ in eng.stream(warm_prompt, schedule=sch, left=left):
            pass

    # 1) pure generation speed (no decoder): ms per frame by chunk size
    gen = {}
    for K in (1, 2, 4, 8):
        eng.prefill(warm_prompt)
        torch.cuda.synchronize()
        t = time.perf_counter()
        steps = 0
        while steps < 64:
            eng.step_frames(K)
            steps += K
        torch.cuda.synchronize()
        gen[K] = (time.perf_counter() - t) * 1000 / steps
    # prefill timing
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(5):
        p = eng.build_prompt(kind, **call_for(voice, TIMING_TEXTS[2]))
    torch.cuda.synchronize(); build_ms = (time.perf_counter() - t) * 200
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(5):
        eng.prefill(p)
    torch.cuda.synchronize(); prefill_ms = (time.perf_counter() - t) * 200
    # decoder timing for one 8-frame window with 25 left context
    codes = eng.generate_codes(p, max_frames=64)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(5):
        eng.decode_window(codes, 33, 41, left)
    torch.cuda.synchronize(); dec8_ms = (time.perf_counter() - t) * 200
    for _ in range(5):
        eng.decode_window(codes, 33, 35, left)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(5):
        eng.decode_window(codes, 33, 35, left)
    torch.cuda.synchronize(); dec2_ms = (time.perf_counter() - t) * 200

    # 2) end-to-end streaming latency, prompt build included
    timing = []
    for first, sch in SCHEDULES.items():
        for rep in range(2):
            for i, text in enumerate(TIMING_TEXTS):
                torch.manual_seed(seed_for(f"t{i}-{rep}"))
                torch.cuda.synchronize()
                t = time.perf_counter()
                pr = eng.build_prompt(kind, **call_for(voice, text))
                arrivals, chunks = [], []
                for wav in eng.stream(pr, schedule=sch, left=left):
                    arrivals.append(time.perf_counter() - t)
                    chunks.append(wav)
                total = time.perf_counter() - t
                w = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
                v = first_voiced_sample(w, 24000)
                tv = None
                if v is not None:
                    acc = 0
                    for a, c in zip(arrivals, chunks):
                        acc += len(c)
                        if acc > v:
                            tv = a * 1000
                            break
                timing.append({"first_chunk": first, "text_idx": i, "rep": rep, "ttfa_ms": arrivals[0] * 1000 if arrivals else None,
                               "ttfa_voiced_ms": tv, "full_s": total, "audio_s": len(w) / 24000})

    # 3) accuracy set (streaming path, chunk 8, first chunk 2)
    tag = tag or f"eng-{variant}"
    out_dir = f"/data/gen/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for it in man:
        torch.manual_seed(seed_for(it["id"]))
        torch.cuda.synchronize()
        t = time.perf_counter()
        pr = eng.build_prompt(kind, **call_for(it, it["target_text"]))
        w = np.concatenate(list(eng.stream(pr, schedule=SCHEDULES['ramp1'], left=left)) or [np.zeros(0, np.float32)])
        dt = time.perf_counter() - t
        sf.write(f"{out_dir}/{it['id']}.wav", w, 24000)
        rows.append({"id": it["id"], "wall_s": dt, "audio_s": len(w) / 24000})

    def med(rows_, key):
        vals = [r[key] for r in rows_ if r.get(key) is not None]
        return float(np.median(vals)) if vals else None

    by_first = {f: {"ttfa_ms_median": med([t for t in timing if t["first_chunk"] == f], "ttfa_ms"),
                    "ttfa_voiced_ms_median": med([t for t in timing if t["first_chunk"] == f], "ttfa_voiced_ms"),
                    "rtf_median": float(np.median([t["full_s"] / t["audio_s"] for t in timing if t["first_chunk"] == f and t["audio_s"]]))}
                for f in SCHEDULES}
    res = {"system": "uqt engine", "variant": variant, "tag": tag, "left_context": left,
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "warmup_s": warm_s, "gen_ms_per_frame": gen, "build_prompt_ms": build_ms, "prefill_ms": prefill_ms,
           "decode_ms_8frames": dec8_ms, "decode_ms_2frames": dec2_ms, "prompt_len": int(p.embeds.shape[0]),
           "by_first_chunk": by_first, "timing": timing,
           "accuracy_set_rtf": sum(r["wall_s"] for r in rows) / max(sum(r["audio_s"] for r in rows), 1e-9),
           "rows": rows}
    os.makedirs("/data/results", exist_ok=True)
    json.dump(res, open(f"/data/results/gen-{tag}.json", "w"), indent=1)
    data_vol.commit()
    return res


@app.local_entrypoint()
def bench_all(n: int = 60, variants: str = "Base,CustomVoice"):
    calls = {v: bench.spawn(v, n) for v in variants.split(",")}
    os.makedirs("results/phase1", exist_ok=True)
    for v, c in calls.items():
        try:
            r = c.get()
        except Exception as e:
            print(f"[{v}] FAILED: {e!r}")
            continue
        json.dump(r, open(f"results/phase1/gen-{r['tag']}.json", "w"), indent=1)
        print(f"[{v}] gen ms/frame={r['gen_ms_per_frame']} prefill={r['prefill_ms']:.1f}ms build={r['build_prompt_ms']:.1f}ms "
              f"dec8={r['decode_ms_8frames']:.1f}ms dec2={r['decode_ms_2frames']:.1f}ms")
        print(f"[{v}] {json.dumps(r['by_first_chunk'])} acc-set RTF={r['accuracy_set_rtf']:.3f}")


@app.function(image=img, gpu=GPU, volumes=VOLS, timeout=3600)
def profile(variant: str = "CustomVoice"):
    """Per-frame breakdown: code predictor vs Talker, graphed timing and kernel-level GPU time."""
    import torch
    with torch.inference_mode():
        return _profile_body(variant)


def _profile_body(variant):
    import torch
    from torch.profiler import ProfilerActivity, profile as tprof
    from uqt.common import TIMING_TEXTS
    from uqt.engine import Engine

    tts = _load(variant, "bf16")
    eng = Engine(tts, max_len=2048)
    man = json.load(open("/data/manifest.json"))
    kind = "clone" if variant == "Base" else "custom"
    call = (dict(text=TIMING_TEXTS[3], language="English",
                 voice_clone_prompt=tts.create_voice_clone_prompt(ref_audio=man[0]["prompt_wav"], ref_text=man[0]["prompt_text"]))
            if kind == "clone" else dict(text=TIMING_TEXTS[3], language="English", speaker="Ryan"))
    p = eng.build_prompt(kind, **call)
    out = {"prompt_len": int(p.embeds.shape[0])}
    cp_c = torch.compile(eng._frame_cp, fullgraph=True, dynamic=False)
    tk_c = torch.compile(eng._frame_talker, fullgraph=True, dynamic=False)
    parts = {"cp_compiled": cp_c, "talker512_compiled": lambda: tk_c(512),
             "cp_eager": eng._frame_cp, "talker512_eager": lambda: eng._frame_talker(512)}
    for name, fn in parts.items():
        eng.prefill(p)
        g = eng._capture(("prof", name), fn)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(200):
            g.replay()
        torch.cuda.synchronize()
        out[f"{name}_graph_ms"] = (time.perf_counter() - t) * 1000 / 200
    # kernel-level view (compiled, not graphed)
    for name, fn in (("cp_compiled", cp_c), ("talker512_compiled", lambda: tk_c(512))):
        eng.prefill(p)
        fn(); torch.cuda.synchronize()
        with tprof(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
        ev = [e for e in prof.key_averages() if e.device_type.name == "CUDA"]
        tot = sum(e.self_device_time_total for e in ev) / 10 / 1000
        cnt = sum(e.count for e in ev) / 10
        top = sorted(ev, key=lambda e: -e.self_device_time_total)[:12]
        out[f"{name}_kernels_per_call"] = cnt
        out[f"{name}_gpu_busy_ms"] = tot
        out[f"{name}_top"] = [(e.key[:90], e.count // 10, round(e.self_device_time_total / 10 / 1000, 4)) for e in top]
    print(json.dumps(out, indent=1))
    return out


@app.function(image=img, gpu=GPU, timeout=1800)
def sampler_test(n: int = 40000):
    """Total-variation distance between the Triton top-k sampler and torch topk+multinomial."""
    import torch
    from uqt import kernels

    torch.manual_seed(0)
    out = {}
    for V, k, temp in ((2048, 50, 0.9), (3072, 50, 0.9), (2048, 5, 1.3)):
        logits = torch.randn(V, device="cuda") * 3
        ref = torch.zeros(V, device="cuda")
        v, idx = torch.topk(logits / temp, k)
        p = torch.softmax(v, -1)
        exact = torch.zeros(V, device="cuda").scatter_(0, idx, p)
        draws = torch.multinomial(p, n, replacement=True)
        ref.scatter_add_(0, idx[draws], torch.ones(n, device="cuda"))
        tri = torch.zeros(V, device="cuda")
        for _ in range(n):
            u = torch.rand(V, device="cuda")
            tri[kernels.topk_sample(logits, u, 1.0 / temp, k)] += 1
        ref, tri = ref / n, tri / n
        out[f"V{V}_k{k}_t{temp}"] = {"tv_triton_vs_exact": float(0.5 * (tri - exact).abs().sum()),
                                     "tv_torch_vs_exact": float(0.5 * (ref - exact).abs().sum()),
                                     "support_ok": bool(((tri > 0) <= (exact > 0)).all())}
    print(json.dumps(out, indent=1))
    return out


@app.function(image=img, gpu=GPU, volumes=VOLS, timeout=3600)
def fcp_test(variant: str = "CustomVoice", frames: int = 120, autotune: bool = False):
    """Fused (vLLM-Omni port) vs reference code predictor: greedy agreement on real frames + timing."""
    import torch
    with torch.inference_mode():
        from uqt.common import TIMING_TEXTS
        from uqt.engine import Engine, SamplingCfg

        tts = _load(variant, "bf16")
        eng = Engine(tts, max_len=2048, sampling=SamplingCfg(cp_greedy=True), autotune=autotune)
        assert eng.fcp is not None
        man = json.load(open("/data/manifest.json"))
        kind = "clone" if variant == "Base" else "custom"
        call = (dict(text=TIMING_TEXTS[4], language="English",
                     voice_clone_prompt=tts.create_voice_clone_prompt(ref_audio=man[0]["prompt_wav"], ref_text=man[0]["prompt_text"]))
                if kind == "clone" else dict(text=TIMING_TEXTS[4], language="English", speaker="Ryan"))
        p = eng.build_prompt(kind, **call)
        eng.prefill(p, use_graph=False)
        eq_codes, eq_rows, n = 0, 0, 0
        first_mis = {"fused": [], "compiled": []}
        cp_c = torch.compile(eng._frame_cp, fullgraph=True, dynamic=False)

        def first_mismatch(a, b):
            bad = (a != b).nonzero()
            return int(bad[0]) if len(bad) else 16

        for f in range(frames):
            cp_c()                               # compiled reference (calibration of bf16 noise)
            comp = eng.codes[f].clone()
            eng._frame_cp()                      # eager reference (writes self.codes[fidx], self.inp)
            ref = eng.codes[f].clone()
            eng._fcp_step()                      # fused, same inputs (code0, hidden unchanged)
            fused = eng.fcodes[0].clone()
            first_mis["fused"].append(first_mismatch(ref, fused))
            first_mis["compiled"].append(first_mismatch(ref, comp))
            eq_codes += int((ref == fused).sum()); eq_rows += int((ref == fused).all()); n += 1
            eng._frame_talker(eng._bucket(eng.pos_host + 1))   # advance with the reference codes
            eng.pos_host += 1; eng.frames_host += 1
            if int(eng.code0) == eng.eos:
                break
        out = {"frames": n, "frac_codes_equal": eq_codes / (16 * n), "frac_frames_equal": eq_rows / n}
        for k, v in first_mis.items():
            # P(code i equal | codes < i equal), i = 1..15 (code 0 is shared input)
            cond = []
            for i in range(1, 16):
                reach = sum(1 for m in v if m >= i)
                cond.append(round(sum(1 for m in v if m > i) / reach, 3) if reach else None)
            out[f"{k}_cond_agree_by_pos"] = cond
            out[f"{k}_frames_equal"] = sum(1 for m in v if m == 16) / len(v)
        # timing (sampling mode)
        eng.s.cp_greedy = False
        tk = eng._talker_fused_c
        for name, fn in {"fcp": eng._fcp_step, "talker_fused512": lambda: tk(512),
                         "frame_fused512": lambda: (eng._fcp_step(), tk(512))}.items():
            eng.prefill(p)
            g = eng._capture(("t", name), fn)
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(200):
                g.replay()
            torch.cuda.synchronize()
            out[f"{name}_graph_ms"] = (time.perf_counter() - t) * 1000 / 200
        out["pdl"] = bool(eng.fcp._pdl_configs)
        print(json.dumps(out, indent=1))
        return out


@app.function(image=img, gpu=GPU, volumes=VOLS, timeout=1800)
def prompt_test():
    """Fast prompt builder vs the official builder: identical embeddings, and timing."""
    import torch
    from uqt.common import TIMING_TEXTS
    from uqt.engine import Engine

    man = json.load(open("/data/manifest.json"))
    out = {}
    with torch.inference_mode():
        for variant in ("Base", "CustomVoice"):
            for dname in ("fp32", "bf16"):
                tts = _load(variant, dname)
                eng = Engine(tts, max_len=2048, compile=False, fused_cp=False, graph_decoder=False)
                kind = "clone" if variant == "Base" else "custom"
                diffs = []
                calls = []
                for i, text in enumerate(TIMING_TEXTS + [man[1]["target_text"]]):
                    if kind == "clone":
                        it = man[i % 3]
                        vcp = tts.create_voice_clone_prompt(ref_audio=it["prompt_wav"], ref_text=it["prompt_text"])
                        calls.append(dict(text=text, language="English", voice_clone_prompt=vcp))
                    else:
                        calls.append(dict(text=text, language="English", speaker=["Ryan", "Vivian", "Aiden"][i % 3]))
                for c in calls:
                    a = eng.build_prompt_official(kind, **c)
                    b = eng.build_prompt(kind, **c)
                    same_shape = a.embeds.shape == b.embeds.shape and a.trailing.shape == b.trailing.shape
                    d = {"shape_ok": same_shape, "T": int(a.embeds.shape[0]), "Tt": int(a.trailing.shape[0])}
                    if same_shape:
                        d["max_diff_embeds"] = float((a.embeds.float() - b.embeds.float()).abs().max())
                        d["max_diff_trailing"] = float((a.trailing.float() - b.trailing.float()).abs().max())
                        d["max_diff_pad"] = float((a.pad.float() - b.pad.float()).abs().max())
                    diffs.append(d)
                tim = {}
                for name, fn in (("official", eng.build_prompt_official), ("fast", eng.build_prompt)):
                    fn(kind, **calls[2]); torch.cuda.synchronize()
                    t = time.perf_counter()
                    for _ in range(20):
                        fn(kind, **calls[2])
                    torch.cuda.synchronize()
                    tim[f"{name}_ms"] = (time.perf_counter() - t) * 1000 / 20
                out[f"{variant}-{dname}"] = {"cases": diffs, **tim}
                del eng, tts
                torch.cuda.empty_cache()
    print(json.dumps(out, indent=1))
    return out


@app.function(image=img, gpu=GPU, volumes=VOLS, timeout=1800)
def codec_test():
    """Streaming (stateful) and windowed decoders vs the official full-utterance decode."""
    import numpy as np
    import torch
    from uqt.common import TIMING_TEXTS
    from uqt.engine import Engine

    man = json.load(open("/data/manifest.json"))
    out = {}

    def snr(ref, x):
        n = min(len(ref), len(x))
        ref, x = ref[:n], x[:n]
        return float(10 * np.log10((ref ** 2).sum() / max(((ref - x) ** 2).sum(), 1e-20))), float(np.abs(ref - x).max())

    with torch.inference_mode():
        for variant in ("Base", "CustomVoice"):
            tts = _load(variant, "bf16")
            eng = Engine(tts, max_len=2048)
            kind = "clone" if variant == "Base" else "custom"
            res = []
            for i in range(3):
                it = man[i]
                call = (dict(text=TIMING_TEXTS[3 + i % 2], language="English",
                             voice_clone_prompt=tts.create_voice_clone_prompt(ref_audio=it["prompt_wav"], ref_text=it["prompt_text"]))
                        if kind == "clone" else dict(text=TIMING_TEXTS[2 + i], language="English", speaker="Ryan"))
                p = eng.build_prompt(kind, **call)
                gen = eng.generate_codes(p, max_frames=400)
                ref = p.ref_codes if p.ref_codes is not None else gen[:0]
                R = ref.shape[0]
                allc = torch.cat([ref, gen], 0)
                off = tts.model.speech_tokenizer.decode([{"audio_codes": allc}])[0][0]
                off = np.asarray(off.float().cpu() if hasattr(off, "float") else off, np.float32)[R * 1920:]
                # streaming codec, chunk ramp 1,2,4,8...
                eng.voice_dec_state.clear()
                base = eng._sdec_prime(ref if R else None)
                sched, pos, chunks = [1, 2, 4, 8], 0, []
                j = 0
                while pos < gen.shape[0]:
                    k = min(sched[min(j, 3)], gen.shape[0] - pos)
                    chunks.append(eng._sdec_call(gen[pos:pos + k], base + pos).cpu().numpy().copy())
                    pos += k; j += 1
                st = np.concatenate(chunks)
                # windowed decode (original non-stateful path), left 25
                full = torch.cat([ref, gen], 0)
                pos, j, wch = 0, 0, []
                while pos < gen.shape[0]:
                    k = min(sched[min(j, 3)], gen.shape[0] - pos)
                    wch.append(eng.decode_window(full, R + pos, R + pos + k, 25).float().cpu().numpy())
                    pos += k; j += 1
                wd = np.concatenate(wch)
                res.append({"frames": int(gen.shape[0]), "R": int(R),
                            "streaming_snr_db_maxabs": snr(off, st), "windowed_snr_db_maxabs": snr(off, wd),
                            "len_off": len(off), "len_stream": len(st)})
            # timing per chunk size
            tim = {}
            codes = eng.generate_codes(p, max_frames=64)
            for k in (1, 2, 4, 8):
                with torch.cuda.stream(eng.dec_stream):
                    eng._sdec_call(codes[:k], 8)
                    torch.cuda.synchronize(); t = time.perf_counter()
                    for _ in range(50):
                        eng._sdec_call(codes[:k], 8)
                    torch.cuda.synchronize(); tim[f"stream_k{k}_ms"] = (time.perf_counter() - t) * 20
                eng.decode_window(codes, 33, 33 + k, 25)
                torch.cuda.synchronize(); t = time.perf_counter()
                for _ in range(50):
                    eng.decode_window(codes, 33, 33 + k, 25)
                torch.cuda.synchronize(); tim[f"window_k{k}_ms"] = (time.perf_counter() - t) * 20
            eng.voice_dec_state.clear()
            torch.cuda.synchronize(); t = time.perf_counter()
            if kind == "clone":
                eng._sdec_prime(p.ref_codes)
            torch.cuda.synchronize(); tim["prime_cold_ms"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            if kind == "clone":
                eng._sdec_prime(p.ref_codes)
            torch.cuda.synchronize(); tim["prime_cached_ms"] = (time.perf_counter() - t) * 1000
            out[variant] = {"cases": res, **tim}
            del eng, tts
            torch.cuda.empty_cache()
    print(json.dumps(out, indent=1))
    return out


@app.function(image=img, gpu=GPU, volumes=VOLS, timeout=3600)
def ftk_test(variant: str = "CustomVoice", frames: int = 60):
    """Fused Talker chain vs reference Talker step: logits/hidden agreement on real frames + timing."""
    import torch
    import torch.nn.functional as F
    with torch.inference_mode():
        from uqt.common import TIMING_TEXTS
        from uqt.engine import Engine, SamplingCfg

        tts = _load(variant, "bf16")
        eng = Engine(tts, max_len=2048, sampling=SamplingCfg(greedy=True, cp_greedy=True), fused_talker=True)
        print("gemm configs:", eng.ftk.cfg)
        kind = "clone" if variant == "Base" else "custom"
        p = eng.build_prompt(kind, text=TIMING_TEXTS[4], language="English", speaker="Ryan")
        eng.prefill(p, use_graph=False)
        stats = []
        for f in range(frames):
            eng._frame_cp()
            Lb = eng._bucket(eng.pos_host + 1)
            lf = eng.ftk.step(eng.inp, Lb)[0].float().clone()
            hf = eng.ftk.final_hidden().float().clone()
            cos = eng.t_cos.index_select(0, eng.pos); sin = eng.t_sin.index_select(0, eng.pos)
            mask = torch.zeros(1, Lb, device=eng.dev, dtype=eng.dtype).masked_fill(eng.arange[:Lb] > eng.pos, float("-inf"))
            hr = eng.talker.forward(eng.inp, cos, sin, eng.kc, eng.vc, eng._write_dyn, Lb, mask)
            lr = F.linear(hr, eng.head)[0].float()
            lr_m = lr.clone(); lr_m[eng.suppress] = -1e9; lf_m = lf.clone(); lf_m[eng.suppress] = -1e9
            stats.append({"logit_maxdiff": float((lf - lr).abs().max()), "logit_cos": float(F.cosine_similarity(lf, lr, 0)),
                          "argmax_eq": bool(lf_m.argmax() == lr_m.argmax()),
                          "hidden_maxdiff": float((hf - hr.float()).abs().max()),
                          "hidden_cos": float(F.cosine_similarity(hf.reshape(-1), hr.float().reshape(-1), 0))})
            eng._frame_talker(Lb)
            eng.pos_host += 1; eng.frames_host += 1
            if int(eng.code0) == eng.eos:
                break
        n = len(stats)
        out = {"frames": n, "argmax_agree": sum(s["argmax_eq"] for s in stats) / n,
               "logit_cos_min": min(s["logit_cos"] for s in stats), "logit_maxdiff_max": max(s["logit_maxdiff"] for s in stats),
               "hidden_cos_min": min(s["hidden_cos"] for s in stats), "hidden_maxdiff_max": max(s["hidden_maxdiff"] for s in stats),
               "cfg": {k: list(v) for k, v in eng.ftk.cfg.items()}}
        # timing
        eng.s.greedy = False; eng.s.cp_greedy = False
        for name, fn in {"ftk_step512": lambda: eng.ftk.step(eng.inp, 512),
                         "frame_fusedtalker512": lambda: eng._run_frames(1, 512),
                         "ftk_step1024": lambda: eng.ftk.step(eng.inp, 1024)}.items():
            eng.prefill(p)
            g = eng._capture(("t", name), fn)
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(200):
                g.replay()
            torch.cuda.synchronize()
            out[f"{name}_graph_ms"] = (time.perf_counter() - t) * 1000 / 200
        print(json.dumps(out, indent=1))
        return out
