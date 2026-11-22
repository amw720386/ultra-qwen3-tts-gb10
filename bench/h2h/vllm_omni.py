"""Head-to-head: vLLM-Omni serving Qwen3-TTS-12Hz-1.7B-CustomVoice on one H100, measured with bench/h2h/client.py.

The server (official `vllm/vllm-omni:v0.32.0` image = vllm 0.32.0 + vllm-omni 0.32.0) runs in a subprocess
on localhost inside the same container as the shared client and streams raw PCM
(`stream=true, stream_format="audio", response_format="pcm"`; in v0.32 `stream=true` alone selects SSE).

Deploy profiles (all bundled with vllm-omni 0.32.0, vllm_omni/deploy/):
  default   qwen3_tts.yaml                          shipped default, two-stage, MRV2, "verified on 1x H100"
  fused     qwen3_tts_fused_single_gpu.yaml         single-stage Talker + in-Talker streaming codec decoder
  fused_fa  fused + talker_stream_first_audio=true  documented overlay for lower first-audio latency
  hc1       qwen3_tts_high_concurrency_mrv2_single_gpu.yaml (+ --api-server-count 4), two-stage, MPS
  hc1_nomps hc1 with cuda_mps: false (documented overlay)

    modal run bench/h2h/vllm_omni.py --cfg fused            # full protocol for one profile
    modal run bench/h2h/vllm_omni.py --cfg fused --final    # also publish as the canonical h2h-vllm result
"""
import json
import os
import subprocess
import sys
import time

import modal

APP_NAME = "uqt-h2h-vllm"
app = modal.App(APP_NAME)
data_vol = modal.Volume.from_name("uqt-data")
hf_vol = modal.Volume.from_name("uqt-hf-cache")
VOLS = {"/data": data_vol, "/root/.cache/huggingface": hf_vol}
GPU = "H100!"
IMAGE_TAG = "vllm/vllm-omni:v0.32.0"
MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
PORT = 8091
H2H = os.path.dirname(os.path.abspath(__file__))

img = (
    modal.Image.from_registry(IMAGE_TAG, setup_dockerfile_commands=["RUN ln -sf /usr/bin/python3 /usr/bin/python"])
    .entrypoint([])
    .run_commands("uv pip install --system httpx soundfile numpy || python -m pip install httpx soundfile numpy")
    .add_local_file(os.path.join(H2H, "client.py"), "/opt/h2h/client.py")
)

OVERLAYS = {
    "fused_fa": """base_config: qwen3_tts_fused_single_gpu.yaml
stages:
  - stage_id: 0
    additional_config:
      talker_stream_decode: true
      talker_stream_first_audio: true
      ref_code_context_frames: 72
      code_predictor_kv_cache: true
      code_predictor_fused_sampling: true
      code_predictor_fused: true
""",
    # sensitivity check, NOT a documented profile: fused_fa with 256 Talker slots so C=256 does not queue
    "fused_fa_s256": """base_config: h2h_fused_first_audio.yaml
stages:
  - stage_id: 0
    max_num_seqs: 256
    gpu_memory_utilization: 0.7
""",
    "hc1_nomps": """base_config: qwen3_tts_high_concurrency_mrv2_single_gpu.yaml
cuda_mps: false
""",
}
CFGS = {
    "default": ("qwen3_tts.yaml", []),
    "fused": ("qwen3_tts_fused_single_gpu.yaml", []),
    "fused_fa": ("h2h_fused_first_audio.yaml", []),
    "fused_fa_s256": ("h2h_fused_fa_s256.yaml", []),
    "hc1": ("qwen3_tts_high_concurrency_mrv2_single_gpu.yaml", ["--api-server-count", "4"]),
    "hc1_nomps": ("h2h_hc1_nomps.yaml", ["--api-server-count", "4"]),
}
OVERLAY_FILES = {"fused_fa": "h2h_fused_first_audio.yaml", "fused_fa_s256": "h2h_fused_fa_s256.yaml", "hc1_nomps": "h2h_hc1_nomps.yaml"}


def _clean_env():
    """Subprocess env without Modal's PYTHONPATH (it contains /root, where this file shadows `vllm_omni`)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60, cwd="/tmp", env=_clean_env()).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return repr(e)


@app.function(image=img, gpu=GPU, volumes=VOLS, timeout=5400, cpu=16.0, memory=65536)
def run(cfg: str = "fused", n_lat: int = 60, concs: str = "1,8,32,64,128,256", reqs_per_client: int = 4,
        final: bool = False, tag: str = ""):
    import httpx
    import numpy as np
    import soundfile as sf

    sys.path.insert(0, "/opt/h2h")
    from client import latency, load

    # NB: this script is mounted as /root/vllm_omni.py and shadows the package when imported from /root,
    # so resolve the installed package (and run the server) from a neutral cwd.
    deploy_dir = subprocess.run([sys.executable, "-c", "import os, vllm_omni; print(os.path.join("
                                 "os.path.dirname(vllm_omni.__file__), 'deploy'))"], cwd="/tmp", capture_output=True,
                                text=True, check=True, env=_clean_env()).stdout.strip()
    for k, txt in OVERLAYS.items():  # overlays must sit beside the bundled files (relative base_config)
        with open(os.path.join(deploy_dir, OVERLAY_FILES[k]), "w") as f:
            f.write(txt)
    yaml_name, extra = CFGS[cfg]
    deploy_path = os.path.join(deploy_dir, yaml_name)
    cmd = ["vllm", "serve", MODEL, "--omni", "--host", "127.0.0.1", "--port", str(PORT),
           "--deploy-config", deploy_path, "--trust-remote-code",
           "--stage-init-timeout", "1800", "--init-timeout", "2400", *extra]
    deploy_text = {"yaml": open(deploy_path).read()}
    base = yaml_name
    while True:  # record the resolved chain of base_config files verbatim
        txt = open(os.path.join(deploy_dir, base)).read()
        nxt = next((ln.split(":", 1)[1].strip() for ln in txt.splitlines() if ln.startswith("base_config:")), None)
        if not nxt:
            break
        deploy_text[nxt] = open(os.path.join(deploy_dir, nxt)).read()
        base = nxt

    info = {
        "system": "vllm-omni", "cfg": cfg, "cmd": " ".join(cmd), "deploy_config": deploy_text,
        "image": IMAGE_TAG,
        "gpu": _sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"),
        "versions": _sh("python -c \"import importlib.metadata as m; print({p: m.version(p) for p in "
                        "['vllm','vllm-omni','torch','transformers','flashinfer-python']})\" 2>/dev/null"),
        "mps_control": _sh("which nvidia-cuda-mps-control"),
    }
    print(json.dumps({k: v for k, v in info.items() if k != "deploy_config"}, indent=1), flush=True)

    os.makedirs("/data/results/h2h-vllm-logs", exist_ok=True)
    log_path = f"/tmp/server-{cfg}.log"
    logf = open(log_path, "w")
    t_launch = time.time()
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd="/tmp", env=_clean_env())
    base_url = f"http://127.0.0.1:{PORT}"
    url = f"{base_url}/v1/audio/speech"

    def tail(n=80):
        logf.flush()
        return "".join(open(log_path, errors="replace").readlines()[-n:])

    def probe():
        if proc.poll() is not None:
            raise RuntimeError(f"server exited rc={proc.returncode}\n{tail()}")
        return httpx.get(f"{base_url}/v1/models", timeout=5).status_code == 200

    def finish(res):
        import shutil
        shutil.copy(log_path, f"/data/results/h2h-vllm-logs/{tag or cfg}.log")
        data_vol.commit()
        return res

    try:
        # own poll loop (client.wait_ready swallows exceptions, so a crashed server would burn the full timeout)
        while not _safe(probe):
            if time.time() - t_launch > 2700:
                print(tail(200), flush=True)
                raise TimeoutError("server not ready")
            time.sleep(3)
        info["startup_s"] = time.time() - t_launch
        print(f"ready after {info['startup_s']:.0f}s", flush=True)

        voices = httpx.get(f"{base_url}/v1/audio/voices", timeout=30).json()
        info["voices"] = voices
        vlist = voices.get("voices", voices) if isinstance(voices, dict) else voices
        names = [v if isinstance(v, str) else (v.get("name") or v.get("id")) for v in vlist]
        voice = next(n for n in names if str(n).lower() == "ryan")
        info["voice"] = voice

        def make_payload(text):
            return {"model": MODEL, "input": text, "voice": voice, "language": "English",
                    "stream": True, "stream_format": "audio", "response_format": "pcm"}

        info["payload_example"] = make_payload("<text>")

        # format check: headers must say 24 kHz mono, body must be int16 PCM (not WAV/SSE)
        with httpx.stream("POST", url, json=make_payload("Hello, this is a format check."), timeout=600) as r:
            body = r.read()
            hdr = {k: v for k, v in r.headers.items() if k.startswith("x-audio") or k == "content-type"}
        info["format_check"] = {"status": r.status_code, "headers": hdr, "n_bytes": len(body),
                                "starts_with": body[:16].hex()}
        print("format check", info["format_check"], flush=True)
        assert r.status_code == 200, body[:300]
        assert hdr.get("x-audio-sample-rate") == "24000" and hdr.get("x-audio-channels", "1") == "1", hdr
        assert not body.startswith(b"RIFF") and not body.startswith(b"data:") and not body.startswith(b"event"), body[:64]

        man = json.load(open("/data/manifest.json"))
        all_texts = [m["target_text"] for m in man]

        # warm the full pipeline (docs: "Warm the full pipeline before measuring") incl. batched graph buckets
        t = time.time()
        load(url, make_payload, all_texts[-64:], concs=(1, 16, 128), reqs_per_client=1)
        info["warmup_s"] = time.time() - t

        out_dir = f"/data/gen/h2h-vllm-CustomVoice-{tag or cfg}"
        lat = latency(url, make_payload, all_texts[:n_lat], ids=[m["id"] for m in man[:n_lat]], out_dir=out_dir)
        print("latency", json.dumps(lat["summary"]), flush=True)

        # sanity-check saved wavs: duration plausible vs text length, not silence
        chk = []
        for m in man[:3]:
            w, sr = sf.read(f"{out_dir}/{m['id']}.wav")
            chk.append({"id": m["id"], "sr": sr, "dur_s": len(w) / sr, "chars": len(m["target_text"]),
                        "chars_per_s": len(m["target_text"]) / max(len(w) / sr, 1e-6),
                        "rms_dbfs": float(20 * np.log10(np.sqrt(np.mean(w ** 2)) + 1e-12)),
                        "peak": float(np.max(np.abs(w)))})
        info["wav_check"] = chk
        print("wav check", chk, flush=True)

        ld = load(url, make_payload, all_texts, concs=tuple(int(c) for c in concs.split(",")),
                  reqs_per_client=reqs_per_client)
        res = {**info, "tag": tag, "latency": lat, "load": {str(k): v for k, v in ld.items()}, "gen_dir": out_dir}
        os.makedirs("/data/results", exist_ok=True)
        with open(f"/data/results/h2h-vllm-{tag or cfg}.json", "w") as f:
            json.dump(res, f, indent=1)
        if final:
            import shutil
            dst = "/data/gen/h2h-vllm-CustomVoice"
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(out_dir, dst)
            res["gen_dir"] = dst
            with open("/data/results/h2h-vllm.json", "w") as f:
                json.dump(res, f, indent=1)
        print("server log tail:\n" + tail(30), flush=True)
        return finish(res)
    finally:
        proc.terminate()
        try:
            proc.wait(60)
        except Exception:  # noqa: BLE001
            proc.kill()


def _safe(fn):
    try:
        return fn()
    except RuntimeError:
        raise
    except Exception:  # noqa: BLE001
        return False


@app.function(image=modal.Image.debian_slim(python_version="3.12"), volumes={"/data": data_vol}, timeout=600)
def publish(cfg: str):
    """Copy an already-measured profile to the canonical h2h-vllm paths."""
    import shutil
    res = json.load(open(f"/data/results/h2h-vllm-{cfg}.json"))
    dst = "/data/gen/h2h-vllm-CustomVoice"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(res["gen_dir"], dst)
    res["gen_dir"] = dst
    json.dump(res, open("/data/results/h2h-vllm.json", "w"), indent=1)
    data_vol.commit()
    return res


@app.local_entrypoint()
def main(cfg: str = "fused", n_lat: int = 60, concs: str = "1,8,32,64,128,256", reqs_per_client: int = 4,
         final: bool = False, publish_only: bool = False, tag: str = ""):
    res = publish.remote(cfg) if publish_only else run.remote(cfg, n_lat, concs, reqs_per_client, final, tag)
    os.makedirs("results/h2h", exist_ok=True)
    name = "vllm-omni.json" if (final or publish_only) else f"vllm-omni-{tag or cfg}.json"
    with open(f"results/h2h/{name}", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["latency"]["summary"], indent=1))
    for c, s in res["load"].items():
        print(c, {k: s.get(k) for k in ("ttfa_ms_p50", "ttfa_ms_p90", "ttfa_ms_p99", "rtf_p90", "underrun_frac",
                                         "throughput_audio_s_per_s", "n_ok")})
