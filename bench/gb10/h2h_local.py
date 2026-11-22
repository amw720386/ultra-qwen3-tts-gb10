"""Head-to-head on the GB10 box itself (client co-located): our engine process + Go gateway, or any
already-running competitor server, measured with the shared multi-process client.

    python -m bench.gb10.h2h_local --system uqt --weights fp8              # starts engine + gateway
    python -m bench.gb10.h2h_local --system vllm --url http://127.0.0.1:8091/v1/audio/speech
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "h2h"))
from client import latency, load, wait_ready  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="uqt")
    ap.add_argument("--url", default="http://127.0.0.1:8091/v1/audio/speech")
    ap.add_argument("--health", default="")
    ap.add_argument("--weights", default="hyb")
    ap.add_argument("--max-slots", type=int, default=256)
    ap.add_argument("--concs", default="1,8,32,64,128,256")
    ap.add_argument("--reqs", type=int, default=3)
    ap.add_argument("--voice", default="Ryan")
    ap.add_argument("--extra", default="{}", help="extra JSON fields for the request payload")
    ap.add_argument("--manifest", default="/root/manifest.json")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    procs = []
    if args.system == "uqt":
        sock = "/tmp/uqt.sock"
        env = dict(os.environ, UQT_STATS="0", UQT_NO_PDL="1")
        procs.append(subprocess.Popen([sys.executable, "-m", "uqt.engine_proc", "--sock", sock, "--max-slots",
                                       str(args.max_slots), "--max-len", "512", "--weights", args.weights],
                                      env=env))
        procs.append(subprocess.Popen(["/root/uqt-gw", "-sock", sock, "-addr", "127.0.0.1:8091"]))
        health = "http://127.0.0.1:8091/health"
    else:
        health = args.health
    try:
        t0 = time.time()
        if health:
            wait_ready(health, timeout_s=3600)
        startup = time.time() - t0
        extra = json.loads(args.extra)
        make = lambda t: {"input": t, "voice": args.voice, "language": "English", "response_format": "pcm",  # noqa: E731
                          "stream": True, **extra}
        man = json.load(open(args.manifest))
        texts = [m["target_text"] for m in man]
        load(args.url, make, texts, concs=(1, 16), reqs_per_client=1)  # warm pass, discarded
        res = {"system": args.system, "weights": args.weights if args.system == "uqt" else None,
               "startup_s": startup, "extra": extra}
        res["latency"] = latency(args.url, make, texts[:40], ids=[m["id"] for m in man[:40]],
                                 out_dir=f"/root/gen/h2h-{args.system}")
        print("latency", json.dumps(res["latency"]["summary"]), flush=True)
        res["load"] = load(args.url, make, texts, concs=tuple(int(c) for c in args.concs.split(",")),
                           reqs_per_client=args.reqs)
        for c, s in res["load"].items():
            print("load", c, json.dumps(s), flush=True)
        out = args.out or f"/root/results/h2h-{args.system}.json"
        json.dump(res, open(out, "w"), indent=1)
    finally:
        for p in procs[::-1]:
            p.terminate()


if __name__ == "__main__":
    main()
