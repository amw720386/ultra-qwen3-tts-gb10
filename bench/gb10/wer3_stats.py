"""Pooled WER per system (uqt, sglang-omni, vLLM-Omni) with a bootstrap over texts of the differences.

    python bench/gb10/wer3_stats.py results/gb10/wer3   # dirs <system>-c<clients>-r<run> with eval.json and gen.json
"""
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict

root = sys.argv[1]
E = defaultdict(lambda: defaultdict(list))  # text id -> system -> [(errors, words, seconds, clients)]
md5, failed, per = defaultdict(Counter), Counter(), defaultdict(list)
for d in sorted(glob.glob(f"{root}/*/")):
    s, c, r = os.path.basename(d.rstrip("/\\")).split("-")
    ev, g = json.load(open(f"{d}/eval.json")), json.load(open(f"{d}/gen.json"))
    failed[s] += len(g["failed"])
    for k, h in g["md5"].items():
        md5[s][k, h] += 1
    for x in ev["rows"]:
        E[x["id"]][s].append((x["errors"], x["n_ref"], g["audio_s"][x["id"]], c))
    per[s].append((c, r, 100 * ev["wer_corpus"], ev["summary"]["catastrophic"]))
S = sorted(per)
ids = sorted(E)


def wer(s, ii, excl=False, c=None):
    e = n = 0
    for i in ii:
        for a, b, _, cc in E[i][s]:
            if (excl and a / max(b, 1) > 0.5) or (c and cc != c):
                continue
            e, n = e + a, n + b
    return 100 * e / n


for s in S:
    rows = [x for i in ids for x in E[i][s]]
    print(f"{s}: clips {len(rows)} failed {failed[s]} distinct audio {len(md5[s])} | WER {wer(s, ids):.2f} "
          f"excluding garbled {wer(s, ids, True):.2f} garbled {sum(a / max(b, 1) > 0.5 for a, b, _, _ in rows)} "
          f"over 25 s {sum(x[2] > 25 for x in rows)} max {max(x[2] for x in rows):.1f} s mean {sum(x[2] for x in rows) / len(rows):.2f} s")
    print("   by clients", {c: round(wer(s, ids, c=c), 2) for c in ("c32", "c64")},
          "per run", [(c, r, round(w, 2), k) for c, r, w, k in per[s]])
random.seed(0)
for a in S:
    for b in S:
        if a != "uqt" or b == a:
            continue
        for ex in (False, True):
            d = sorted(wer(a, ii, ex) - wer(b, ii, ex) for ii in (random.choices(ids, k=len(ids)) for _ in range(4000)))
            print(f"{a} - {b}{' excluding garbled' if ex else ''}: {wer(a, ids, ex) - wer(b, ids, ex):+.2f} "
                  f"95% CI [{d[100]:+.2f}, {d[3899]:+.2f}]")
