"""Head-to-head: uqt GPU engine process + Go gateway, measured with the shared client.

    modal run bench/h2h/uqt_go_h2h.py
"""
import json
import os
import subprocess
import sys
import time

import modal

app = modal.App("uqt-h2h-uqtgo")
data_vol = modal.Volume.from_name("uqt-data", create_if_missing=True)
hf_vol = modal.Volume.from_name("uqt-hf-cache", create_if_missing=True)
VOLS = {"/data": data_vol, "/root/.cache/huggingface": hf_vol}
CU128 = "https://download.pytorch.org/whl/cu128"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
img = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("sox", "libsox-dev", "ffmpeg", "libsndfile1", "golang-go")
    .add_local_dir(os.path.join(ROOT, "gateway"), "/root/gateway", copy=True)
    .run_commands("cd /root/gateway && go version && go build -o /usr/local/bin/uqt-gw .")
    .pip_install("torch==2.9.1", "torchaudio==2.9.1", extra_index_url=CU128)
    .pip_install("qwen-tts==0.1.1", "soundfile", "numpy", "httpx==0.28.1")
    .add_local_python_source("uqt")
    .add_local_file(os.path.join(HERE, "client.py"), "/root/h2h/client.py")
)
CONCS = "1,8,32,64,128,256,384,512"


@app.function(image=img, gpu="H100!", volumes=VOLS, timeout=7200, cpu=16.0, memory=65536)
def run(concs: str = CONCS, max_slots: int = 512, max_len: int = 512, tag: str = "uqtgo"):
    import torch

    sys.path.insert(0, "/root/h2h")
    from client import latency, load, wait_ready

    for k in ("net.core.somaxconn", "net.ipv4.tcp_max_syn_backlog"):
        try:
            print(k, open("/proc/sys/" + k.replace(".", "/")).read().strip(), flush=True)
        except Exception as e:
            print(k, "unreadable", e, flush=True)
    sock = "/tmp/uqt.sock"
    env = dict(os.environ, UQT_STATS="1")
    eng = subprocess.Popen([sys.executable, "-m", "uqt.engine_proc", "--sock", sock, "--max-slots", str(max_slots),
                            "--max-len", str(max_len)], cwd="/root", stdout=sys.stdout, stderr=sys.stderr, env=env)
    gw = subprocess.Popen(["/usr/local/bin/uqt-gw", "-sock", sock, "-addr", "127.0.0.1:8091"],
                          stdout=sys.stdout, stderr=sys.stderr)
    try:
        t0 = time.time()
        wait_ready("http://127.0.0.1:8091/health", timeout_s=3600)
        startup = time.time() - t0
        url = "http://127.0.0.1:8091/v1/audio/speech"
        make = lambda t: {"input": t, "voice": "Ryan", "language": "English", "response_format": "pcm", "stream": True}  # noqa: E731
        man = json.load(open("/data/manifest.json"))
        texts = [m["target_text"] for m in man]
        # discard a warm pass (first-use costs in the gateway and engine)
        load(url, make, texts, concs=(1, 16, 128), reqs_per_client=1)
        res = {"system": "uqt engine process + Go gateway", "variant": "CustomVoice", "max_slots": max_slots,
               "max_len": max_len, "gpu": torch.cuda.get_device_name(0), "startup_s": startup}
        res["latency"] = latency(url, make, texts[:60], ids=[m["id"] for m in man[:60]],
                                 out_dir=f"/data/gen/h2h-{tag}-CustomVoice")
        print("h2h-latency", json.dumps(res["latency"]["summary"]), flush=True)
        res["load"] = load(url, make, texts, concs=tuple(int(c) for c in concs.split(",")), reqs_per_client=4)
        os.makedirs("/data/results", exist_ok=True)
        json.dump(res, open(f"/data/results/h2h-{tag}.json", "w"), indent=1)
        data_vol.commit()
        return res
    finally:
        gw.terminate()
        eng.terminate()


@app.local_entrypoint()
def main(concs: str = CONCS, max_slots: int = 512, max_len: int = 512, tag: str = "uqtgo"):
    r = run.remote(concs, max_slots, max_len, tag)
    os.makedirs("results/h2h", exist_ok=True)
    json.dump(r, open(f"results/h2h/{tag}.json", "w"), indent=1)
    print(json.dumps(r["latency"]["summary"], indent=1))
    for C, s in r["load"].items():
        print(C, json.dumps(s))
