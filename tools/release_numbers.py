"""Summarize the raw GB10 runs into results/gb10/release.json, the file the README, charts and video read.

    python tools/release_numbers.py

Head-to-head: one entry per system and client count, medians over the measured runs (the warm pass, one request per
client, is discarded). uqt: final2-uqt.jsonl. sglang-omni: comp2-sglang.jsonl at 8-64 clients (same day as uqt),
comp-sglang.jsonl at 1, 128 and 256. vLLM-Omni: comp-vllm.jsonl.
"""
import json
import math
import pathlib
from statistics import median

R = pathlib.Path(__file__).resolve().parent.parent / "results" / "gb10"
SYSTEMS = [("uqt", "This engine", {"final2-uqt.jsonl": None}),
           ("sglang", "sglang-omni", {"comp2-sglang.jsonl": {8, 16, 32, 64}, "comp-sglang.jsonl": {1, 128, 256}}),
           ("vllm", "vLLM-Omni", {"comp-vllm.jsonl": None})]
LOADS = [1, 8, 16, 32, 64, 128, 256]


def runs(files):
    out = {}
    for name, keep in files.items():
        for line in (R / name).read_text().splitlines():
            r = json.loads(line)
            if r["n"] == r["C"]:      # warm pass
                continue
            if keep is None or r["C"] in keep:
                out.setdefault(r["C"], []).append(r)
    return out


def rnd(x, d):
    return round(x, d)


def load_rows():
    rows = []
    for key, label, files in SYSTEMS:
        by = runs(files)
        for c in LOADS:
            rs = by.get(c)
            if not rs:
                continue
            m = lambda k, s=1.0, d=1: rnd(median(r[k] for r in rs) * s, d)  # noqa: E731
            thr = [r["throughput_audio_s_per_s"] for r in rs]
            rows.append({
                "id": f"{key}-{c}", "system": label, "clients": c, "runs": len(rs),
                "throughput": m("throughput_audio_s_per_s"), "throughput_min": rnd(min(thr), 1), "throughput_max": rnd(max(thr), 1),
                "first_byte_p50_ms": m("ttfa_ms_p50", d=0), "first_byte_p90_ms": m("ttfa_ms_p90", d=0), "heard_p50_ms": m("heard_ms_p50", d=0),
                "rtf_p50": m("rtf_p50", d=2), "late_pct": m("underrun_frac", 100, 0), "late_p90_ms": m("late_ms_p90", d=0),
                "failed": sum(r["n"] - r["n_ok"] for r in rs), "requests": sum(r["n"] for r in rs),
            })
    return rows


def ratio(rows, a, b, c):
    get = {r["id"]: r["throughput"] for r in rows}
    return get[f"{a}-{c}"] / get[f"{b}-{c}"] if f"{b}-{c}" in get else None


def wer(pattern):
    """Pooled corpus WER over the eval files matching a glob; clips with more than half their words wrong are counted apart."""
    e = n = e2 = n2 = bad = clips = 0
    for p in sorted(R.glob(pattern)):
        for r in json.loads(p.read_text(encoding="utf-8"))["rows"]:
            clips += 1
            e += r["errors"]
            n += r["n_ref"]
            if r["errors"] > 0.5 * r["n_ref"]:
                bad += 1
            else:
                e2 += r["errors"]
                n2 += r["n_ref"]
    return {"clips": clips, "wer_pct": rnd(100 * e / n, 2), "over_half_wrong": bad, "wer_pct_without_those": rnd(100 * e2 / n2, 2)}


def single(name):
    d = json.loads((R / f"{name}.json").read_text())
    return {"ms_per_frame": rnd(d["gen_ms_per_frame"]["1"], 2), "rtf": rnd(d["single_rtf_p50"], 3), "ttfa_ms": rnd(d["single_ttfa_ms_p50"], 1)}


def main():
    rows = load_rows()
    vs = {}
    for other in ("sglang", "vllm"):
        rs = [x for c in LOADS if (x := ratio(rows, "uqt", other, c))]
        vs[other] = {"geomean": rnd(math.exp(sum(map(math.log, rs)) / len(rs)), 2), "min": rnd(min(rs), 2), "max": rnd(max(rs), 2), "loads": len(rs),
                     "at": {str(c): rnd(x, 2) for c in LOADS if (x := ratio(rows, "uqt", other, c))}}
    get = {r["id"]: r for r in rows}

    def cell(sys_, c, k):
        r = get.get(f"{sys_}-{c}")
        return None if r is None else int(r[k])

    latency = [{"clients": c, **{f"{s}_{n}": cell(s, c, k) for s in ("uqt", "sglang", "vllm")
                                 for n, k in (("first", "first_byte_p50_ms"), ("heard", "heard_p50_ms"))}} for c in LOADS]
    out = {
        "head_to_head": rows,
        "latency": latency,
        "throughput_vs": vs,
        "single_stream": [{"weights": w, **single(f)} for w, f in (("BF16", "baseline-bf16"), ("FP8", "fp8-battn-codec"), ("FP8 + FP4", "gemv-hyb"))],
        # the same 300 texts through each system's own server on one GB10, final default settings
        "server_wer": {s: wer(f"wer3/{s}-c*/eval.json") for s in ("uqt", "sglang", "vllm")},
    }
    (R / "release.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in out.items() if k != "head_to_head"}, indent=1))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
