"""A/B test of serving settings inside one warm server: the variants alternate round by round, so machine
and startup differences cancel (needs a server started with UQT_TUNE_FILE).

    python -m bench.gb10.ab_live --variants variants.json --concs 32,64,128,256 --rounds 4 --out ab.json

variants.json: {"base": {"DEADLINE": 0, ...}, "x": {...}}; every variant must set every key it varies.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "h2h"))
from client import load, wait_ready  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True)
    ap.add_argument("--concs", default="32,64,128,256")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--reqs", type=int, default=3)
    ap.add_argument("--manifest", default="/root/manifest.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    variants = json.load(open(args.variants))
    tune = "/tmp/uqt-tune.json"
    json.dump(next(iter(variants.values())), open(tune, "w"))
    env = dict(os.environ, UQT_NO_PDL="1", UQT_TUNE_FILE=tune)
    sock = "/tmp/uqt-ab.sock"
    procs = [subprocess.Popen([sys.executable, "-m", "uqt.engine_proc", "--sock", sock, "--max-slots", "256",
                               "--max-len", "512", "--weights", "hyb"], env=env),
             subprocess.Popen(["/root/uqt-gw", "-sock", sock, "-addr", "127.0.0.1:8091"])]
    url = "http://127.0.0.1:8091/v1/audio/speech"
    try:
        wait_ready("http://127.0.0.1:8091/health", timeout_s=3600)
        texts = [m["target_text"] for m in json.load(open(args.manifest))]
        make = lambda t: {"input": t, "voice": "Ryan", "language": "English",  # noqa: E731
                          "response_format": "pcm", "stream": True}
        load(url, make, texts, concs=(1, 16, 64, 256), reqs_per_client=1)  # warm pass, discarded
        rows = []
        for r in range(args.rounds):
            names = list(variants) if r % 2 == 0 else list(variants)[::-1]
            for c in (int(x) for x in args.concs.split(",")):
                for n in names:
                    json.dump(variants[n], open(tune, "w"))
                    time.sleep(1.0)
                    load(url, make, texts, concs=(1,), reqs_per_client=1)  # makes the server pick it up
                    s = load(url, make, texts, concs=(c,), reqs_per_client=args.reqs)
                    s = s[c] if c in s else s[str(c)]
                    rows.append({"round": r, "variant": n, **s})
                    json.dump({"variants": variants, "rows": rows}, open(args.out, "w"), indent=0)
                    print("ab", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()
                                            if k in ("round", "variant", "C", "n_ok", "n", "throughput_audio_s_per_s",
                                                     "ttfa_ms_p50", "ttfa_ms_p90", "heard_ms_p50", "rtf_p50",
                                                     "underrun_frac", "late_ms_p50", "late_ms_p90")}), flush=True)
    finally:
        for p in procs[::-1]:
            p.terminate()


if __name__ == "__main__":
    main()
