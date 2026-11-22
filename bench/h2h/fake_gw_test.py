"""Isolate HTTP/connection behaviour: shared client vs the Go gateway in fake mode (no GPU, no engine).

    modal run bench/h2h/fake_gw_test.py
"""
import json
import os
import subprocess
import sys

import modal

app = modal.App("uqt-h2h-fakegw")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
img = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("golang-go")
    .add_local_dir(os.path.join(ROOT, "gateway"), "/root/gateway", copy=True)
    .run_commands("cd /root/gateway && go build -o /usr/local/bin/uqt-gw .")
    .pip_install("numpy", "soundfile", "httpx==0.28.1")
    .add_local_file(os.path.join(HERE, "client.py"), "/root/h2h/client.py")
)


@app.function(image=img, cpu=16.0, memory=16384, timeout=1800)
def run(concs: str = "64,256,512", frame_ms: int = 20):
    sys.path.insert(0, "/root/h2h")
    from client import load, wait_ready

    gw = subprocess.Popen(["/usr/local/bin/uqt-gw", "-fake", str(frame_ms), "-addr", "127.0.0.1:8091"])
    try:
        wait_ready("http://127.0.0.1:8091/health", timeout_s=60)
        make = lambda t: {"input": t, "voice": "Ryan"}  # noqa: E731
        return load("http://127.0.0.1:8091/v1/audio/speech", make, ["x"] * 10,
                    concs=tuple(int(c) for c in concs.split(",")), reqs_per_client=4)
    finally:
        gw.terminate()


@app.local_entrypoint()
def main(concs: str = "64,256,512", frame_ms: int = 20):
    r = run.remote(concs, frame_ms)
    for C, s in r.items():
        print(C, json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in s.items()
                             if k in ("ttfa_ms_p50", "ttfa_ms_p90", "hdr_ms_p50", "hdr_ms_p90", "throughput_audio_s_per_s", "underrun_frac")}))
