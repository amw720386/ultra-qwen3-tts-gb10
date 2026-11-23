"""Head-to-head: sglang-omni serving Qwen3-TTS-12Hz-1.7B-CustomVoice on one H100, measured with bench/h2h/client.py.

    modal run bench/h2h/sglang_omni.py                 # full run -> results/h100/h2h/sglang-omni.json
    modal run bench/h2h/sglang_omni.py --quick         # smoke test (few texts, C<=8), not saved as the result
    modal run bench/h2h/sglang_omni.py --backend default   # CuTeDSL off (the CI latency-gate configuration)

Install mirrors sglang-omni's own docker/Dockerfile at a pinned commit: the pinned lmsysorg/sglang base image
(SGLang 0.5.23 stack), the project's pyproject dependencies with its protobuf overrides, qwen-tts 0.1.1 --no-deps,
flashinfer-python 0.7.2 reinstalled, plus the `qwen3-tts-cutedsl` extra (nvidia-cutlass-dsl 4.8.0).
Skipped from that Dockerfile: the UCX source build (multi-node transfer only) and the audar/cosyvoice/minicpm extras.

Server config = the CI "qwen3-tts-custom-voice" preset (tests/test_model/tts_ci_config.py, the source of the
15.3 ms 1-rps first-playable reference): --tts_engine.engine.max_running_requests 64, cuda_graph_max_bs 64,
torch_compile_max_bs 64, SGLANG_OMNI_TORCH_COMPILE_DEFAULT=0, default stateful incremental codec and 1->2->4 chunk
ramp; plus SGLANG_OMNI_QTTS_BACKEND=cutedsl (optional sm90 kernels) unless --backend default.
Deviation: max_queued_requests raised 64 -> 256 so closed-loop C=128/256 queue instead of getting HTTP 503.

Endpoint: POST /v1/audio/speech {"stream": true, "response_format": "pcm"} -> headerless s16le mono PCM; the sample
rate / bit depth response headers are checked against the client's 24 kHz int16 assumption before measuring.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

APP = "uqt-h2h-sgl"
app = modal.App(APP)
data_vol = modal.Volume.from_name("uqt-data")
hf_vol = modal.Volume.from_name("uqt-hf-cache")
VOLS = {"/data": data_vol, "/hf": hf_vol}  # base image already populates /root/.cache/huggingface; HF_HOME=/hf
GPU = "H100!"

SGL_OMNI_COMMIT = "777c17fb25fca032a6003b0c85b5a3deb30181eb"  # sgl-project/sglang-omni main, 2026-11-21
SGLANG_IMAGE = "lmsysorg/sglang@sha256:f7876bef696ccf9cf75a2bf7ccb82970f5fac577f082fb8ca95c9a59d2160eb5"  # docker/Dockerfile pin
PY = "/opt/sglang/bin/python3"
REPO = "/opt/sglang-omni"
MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
PORT = 8000
SERVER_ARGS = [
    "--tts_engine.engine.max_running_requests", "64",
    "--tts_engine.engine.cuda_graph_max_bs", "64",
    "--tts_engine.engine.torch_compile_max_bs", "64",
    "--tts_engine.engine.max_queued_requests", "256",
]
H2H = Path(__file__).parent

img = (
    modal.Image.from_registry(SGLANG_IMAGE)
    .apt_install("sox", "git", "libsndfile1")
    .run_commands(
        f"git clone --filter=blob:none https://github.com/sgl-project/sglang-omni.git {REPO}"
        f" && git -C {REPO} checkout {SGL_OMNI_COMMIT}",
        f"{PY} -c 'import tomllib; print(chr(10).join(tomllib.load(open(\"{REPO}/pyproject.toml\", \"rb\"))"
        f"[\"tool\"][\"uv\"][\"override-dependencies\"]))' > /tmp/omni-overrides.txt",
        f"cd {REPO} && uv pip install --python {PY} --no-build-isolation --overrides /tmp/omni-overrides.txt"
        f" -e '.[qwen3-tts-cutedsl]'",
        f"uv pip install --python {PY} --no-deps qwen-tts==0.1.1 einops",
        f"uv pip install --python {PY} --no-deps --reinstall flashinfer-python==0.7.2",
        f"uv pip install --python {PY} httpx soundfile numpy",
        f"{PY} -c \"import torch, sglang, sglang_omni, cutlass; print(torch.__version__, sglang.__version__)\"",
    )
    .env({"FLASHINFER_WORKSPACE_BASE": "/root", "FLASHINFER_JIT_DEBUG": "0", "HF_HOME": "/hf"})
    .add_local_dir(str(H2H), "/root/h2h", ignore=["*.pyc", "__pycache__"])
)


def _versions():
    out = {"sglang_omni_commit": SGL_OMNI_COMMIT, "base_image": SGLANG_IMAGE}
    code = ("import importlib.metadata as m, json; ps=['sglang-omni','sglang','torch','flashinfer-python','transformers',"
            "'qwen-tts','nvidia-cutlass-dsl','triton']; d={}\n"
            "for p in ps:\n  try: d[p]=m.version(p)\n  except Exception as e: d[p]=None\nprint(json.dumps(d))")
    try:
        out.update(json.loads(subprocess.check_output([PY, "-c", code], text=True).strip().splitlines()[-1]))
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)
    out["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                          "--format=csv,noheader"], text=True).strip()
    return out


def _tail(path, n=80):
    try:
        return "".join(open(path, errors="replace").readlines()[-n:])
    except Exception:  # noqa: BLE001
        return ""


@app.function(image=img, gpu=GPU, volumes=VOLS, timeout=5400, cpu=16.0, memory=65536)
def run(backend: str = "cutedsl", quick: bool = False, tag: str = "", wer: int = 0):
    sys.path.insert(0, "/root/h2h")
    import httpx
    import numpy as np
    import soundfile as sf
    from client import latency, load, wait_ready

    env = dict(os.environ)
    env.update({"SGLANG_OMNI_TORCH_COMPILE_DEFAULT": "0", "HF_HUB_DISABLE_XET": "0",
                "TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache"})
    env.pop("PYTHONPATH", None)  # /root holds this file, named sglang_omni.py: keep it from shadowing the package
    if backend == "cutedsl":
        env["SGLANG_OMNI_QTTS_BACKEND"] = "cutedsl"
    else:
        env.pop("SGLANG_OMNI_QTTS_BACKEND", None)
    cmd = ["/opt/sglang/bin/sgl-omni", "serve", "--model-path", MODEL,
           "--config", "examples/configs/qwen3_tts_1_7b_customvoice.yaml", "--port", str(PORT), *SERVER_ARGS]
    log_path = "/tmp/server.log"
    t_launch = time.time()
    srv = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{PORT}"
    url = f"{base}/v1/audio/speech"

    def probe():
        if srv.poll() is not None:
            raise SystemExit(f"server exited rc={srv.returncode}\n{_tail(log_path, 200)}")
        return httpx.get(f"{base}/health", timeout=5).status_code == 200

    try:
        try:
            wait_ready(f"{base}/health", timeout_s=1800, probe=lambda: _safe(probe))
        except TimeoutError:
            raise SystemExit(f"not ready\n{_tail(log_path, 200)}")
        startup_s = time.time() - t_launch
        print(f"server ready in {startup_s:.0f}s", flush=True)
        voices = httpx.get(f"{base}/v1/audio/voices", timeout=30).text
        print("voices:", voices[:500], flush=True)

        def make_payload(text):
            return {"model": MODEL, "input": text, "voice": "Ryan", "language": "English",
                    "stream": True, "response_format": "pcm"}

        # format check: the client assumes headerless s16le mono 24 kHz
        with httpx.stream("POST", url, json=make_payload("Format check."), timeout=120) as r:
            hdr = dict(r.headers)
            body = r.read()
        print("headers:", hdr, "bytes:", len(body), flush=True)
        assert r.status_code == 200, body[:300]
        sr = int(hdr.get("x-sample-rate", 0))
        assert sr == 24000, f"sample rate {sr}"
        bd = hdr.get("x-bit-depth") or hdr.get("x-bits-per-sample")
        assert bd in (None, "16"), f"bit depth {bd}"
        assert body[:4] != b"RIFF", "got a WAV header, expected raw pcm"

        man = json.load(open("/data/manifest.json"))
        n_lat = wer or (6 if quick else 60)  # wer: only generate the first `wer` clips for scoring
        lat_texts = [m["target_text"] for m in man[:n_lat]]
        lat_ids = [m["id"] for m in man[:n_lat]]
        all_texts = [m["target_text"] for m in man]
        concs = (1, 8) if quick else (1, 8, 32, 64, 128, 256)
        out_dir = "/data/gen/h2h-sgl-CustomVoice" + ("-quick" if quick else "") + ("" if backend == "cutedsl" else f"-{backend}") + (f"-{tag}" if tag else "")

        # extra warmup beyond the server's own startup warmup: sequential + batched shapes (discarded)
        t_w = time.time()
        latency(url, make_payload, all_texts[100:110], warmup=0)
        if not quick and not wer:
            load(url, make_payload, all_texts[110:], concs=(16, 64, 128), reqs_per_client=1)
        print(f"warmup {time.time() - t_w:.0f}s", flush=True)

        lat = latency(url, make_payload, lat_texts, ids=lat_ids, out_dir=out_dir, warmup=3)
        print("latency:", json.dumps(lat["summary"]), flush=True)
        data_vol.commit()

        # sanity check of one output wav
        wav, wsr = sf.read(f"{out_dir}/{lat_ids[0]}.wav")
        sanity = {"id": lat_ids[0], "text": lat_texts[0], "sr": wsr, "dur_s": len(wav) / wsr,
                  "rms_dbfs": float(20 * np.log10(np.sqrt(np.mean(wav ** 2)) + 1e-12)),
                  "peak": float(np.max(np.abs(wav))), "words": len(lat_texts[0].split())}
        sanity["ok"] = bool(sanity["dur_s"] > 0.15 * sanity["words"] and sanity["rms_dbfs"] > -45)
        print("sanity:", sanity, flush=True)

        if wer:
            return {"out_dir": out_dir, "latency": lat, "sanity": sanity, "versions": _versions()}
        ld = load(url, make_payload, all_texts, concs=concs, reqs_per_client=4)
        res = {
            "system": "sglang-omni", "model": MODEL, "backend": backend, "quick": quick, "tag": tag,
            "container": {"cpu": 16.0, "memory_mib": 65536, "nproc": os.cpu_count()},
            "versions": _versions(),
            "config": {"cmd": cmd, "env": {k: env[k] for k in env if k.startswith(("SGLANG_OMNI", "TORCHINDUCTOR"))},
                       "payload": make_payload("<text>"), "response_headers": hdr, "startup_s": startup_s},
            "latency": lat, "sanity": sanity, "load": {str(k): v for k, v in ld.items()},
            "server_log_tail": _tail(log_path, 40),
            "server_log_backend_lines": [l.rstrip() for l in open(log_path, errors="replace")
                                         if any(k in l.lower() for k in ("cutedsl", "cute dsl", "qtts_backend", "stateful", "codec_state", "incremental"))][:40],
        }
        os.makedirs("/data/results", exist_ok=True)
        name = "h2h-sgl" + ("-quick" if quick else "") + ("" if backend == "cutedsl" else f"-{backend}") + (f"-{tag}" if tag else "")
        json.dump(res, open(f"/data/results/{name}.json", "w"), indent=1)
        subprocess.run(["cp", log_path, f"/data/results/{name}-server.log"])
        data_vol.commit()
        return res
    finally:
        srv.terminate()
        try:
            srv.wait(30)
        except Exception:  # noqa: BLE001
            srv.kill()
        print("--- server log tail ---\n" + _tail(log_path, 60), flush=True)


def _safe(fn):
    try:
        return fn()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        return False


@app.local_entrypoint()
def main(backend: str = "cutedsl", quick: bool = False, tag: str = "", wer: int = 0, runs: int = 1):
    if wer:
        for c in [run.spawn(backend=backend, tag=f"{tag or 'wer'}{i + 1}", wer=wer) for i in range(runs)]:
            o = c.get()
            print(o["out_dir"], json.dumps(o["latency"]["summary"]), o["sanity"])
        return
    res = run.remote(backend=backend, quick=quick, tag=tag)
    os.makedirs("results/h100/h2h", exist_ok=True)
    name = "sglang-omni" + ("-quick" if quick else "") + ("" if backend == "cutedsl" else f"-{backend}") + (f"-{tag}" if tag else "")
    json.dump(res, open(f"results/h100/h2h/{name}.json", "w"), indent=1)
    print("latency", json.dumps(res["latency"]["summary"]))
    print("sanity", res["sanity"])
    for c, s in res["load"].items():
        print(f"C={c:>4} ttfa p50/p90/p99 {s['ttfa_ms_p50']:.1f}/{s['ttfa_ms_p90']:.1f}/{s['ttfa_ms_p99']:.1f} ms"
              f"  rtf_p90 {s['rtf_p90']:.3f}  underrun {s['underrun_frac']:.3f}"
              f"  thr {s['throughput_audio_s_per_s']:.1f} audio-s/s  ok {s['n_ok']}/{s['n']}")
