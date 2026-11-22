"""Head-to-head: the uqt engines behind their HTTP server, measured with the shared client.

    modal run bench/h2h/uqt_h2h.py --mode single
    modal run bench/h2h/uqt_h2h.py --mode batch
"""
import json
import os
import subprocess
import sys
import time

import modal

app = modal.App("uqt-h2h-uqt")
data_vol = modal.Volume.from_name("uqt-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("uqt-hf-cache", create_if_missing=True)
VOLS = {"/data": data_vol, "/root/.cache/huggingface": hf_vol}
CU128 = "https://download.pytorch.org/whl/cu128"
HERE = os.path.dirname(os.path.abspath(__file__))
img = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("sox", "libsox-dev", "ffmpeg", "libsndfile1")
    .pip_install("torch==2.9.1", "torchaudio==2.9.1", extra_index_url=CU128)
    .pip_install("qwen-tts==0.1.1", "soundfile", "numpy", "httpx==0.28.1", "starlette==0.46.2", "uvicorn==0.34.3")
    .add_local_python_source("uqt")
    .add_local_file(os.path.join(HERE, "client.py"), "/root/h2h/client.py")
)
CONCS = (1, 8, 32, 64, 128, 256)


@app.function(image=img, gpu="H100!", volumes=VOLS, timeout=7200)
def run(mode: str = "batch", variant: str = "CustomVoice", concs: str = ",".join(map(str, CONCS))):
    import torch

    sys.path.insert(0, "/root/h2h")
    from client import latency, load, wait_ready

    port = 8091
    srv = subprocess.Popen([sys.executable, "-m", "uqt.server", "--mode", mode, "--variant", variant,
                            "--port", str(port), "--max-slots", "256", "--max-len", "1024"],
                           cwd="/root", stdout=sys.stdout, stderr=sys.stderr)
    try:
        t0 = time.time()
        wait_ready(f"http://127.0.0.1:{port}/health", timeout_s=3600)
        startup = time.time() - t0
        url = f"http://127.0.0.1:{port}/v1/audio/speech"
        make = lambda t: {"input": t, "voice": "Ryan", "language": "English", "response_format": "pcm", "stream": True}  # noqa: E731
        man = json.load(open("/data/manifest.json"))
        texts = [m["target_text"] for m in man]
        res = {"system": f"uqt-{mode}", "variant": variant, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "startup_s": startup}
        res["latency"] = latency(url, make, texts[:60], ids=[m["id"] for m in man[:60]],
                                 out_dir=f"/data/gen/h2h-uqt-{mode}-{variant}")
        print("h2h-latency", json.dumps(res["latency"]["summary"]), flush=True)
        if mode == "batch":
            res["load"] = load(url, make, texts, concs=tuple(int(c) for c in concs.split(",")), reqs_per_client=4)
        os.makedirs("/data/results", exist_ok=True)
        json.dump(res, open(f"/data/results/h2h-uqt-{mode}.json", "w"), indent=1)
        data_vol.commit()
        return res
    finally:
        srv.terminate()


@app.local_entrypoint()
def main(mode: str = "batch", variant: str = "CustomVoice", concs: str = ",".join(map(str, CONCS))):
    r = run.remote(mode, variant, concs)
    os.makedirs("results/h100/h2h", exist_ok=True)
    json.dump(r, open(f"results/h100/h2h/uqt-{mode}.json", "w"), indent=1)
    print(json.dumps(r["latency"]["summary"], indent=1))
    for C, s in (r.get("load") or {}).items():
        print(C, json.dumps(s))
