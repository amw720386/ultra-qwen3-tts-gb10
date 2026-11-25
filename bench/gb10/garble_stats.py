"""Garbled clips (more than half the words wrong) per setting, with intervals.

    python bench/gb10/garble_stats.py results/gb10/garble [out.json]   # dirs <setting>-c<clients>-r<run> with eval.json and gen.json

Counts are given with and without one text ("munny my iron ...") that sits near the 50% line in every system.
Rates: Wilson interval (clips independent) and a bootstrap over texts (the same 300 texts repeat in every run).
"""
import glob
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

root, NEAR = sys.argv[1], "2094-142345-0024"
E = defaultdict(lambda: defaultdict(list))  # setting -> text -> [(errors, words, seconds, run)]
failed, md5, runs = Counter(), defaultdict(set), defaultdict(int)
for d in sorted(glob.glob(f"{root}/*/")):
    s = os.path.basename(os.path.normpath(d)).split("-")[0]
    ev, g = json.load(open(f"{d}/eval.json")), json.load(open(f"{d}/gen.json"))
    failed[s] += len(g["failed"])
    runs[s] += 1
    for k, h in g["md5"].items():
        md5[s].add((k, h))
    for x in ev["rows"]:
        E[s][x["id"]].append((x["errors"], x["n_ref"], g["audio_s"][x["id"]]))


def wilson(k, n, z=1.96):
    p, q = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / q, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / q
    return 100 * max(c - h, 0), 100 * (c + h)


def fisher(a, n, b, m):
    k, t = a + b, n + m
    lg = lambda x: math.lgamma(x + 1)
    pr = lambda x: math.exp(lg(n) - lg(x) - lg(n - x) + lg(m) - lg(k - x) - lg(m - k + x) - lg(t) + lg(k) + lg(t - k))
    p0 = pr(a)
    return min(1.0, sum(p for p in (pr(x) for x in range(max(0, k - m), min(k, n) + 1)) if p <= p0 * (1 + 1e-9)))


def cell(s, drop, kind):  # per text: (hits, clips, errors, words)
    out = {}
    for i, rows in E[s].items():
        if drop and i == NEAR:
            continue
        bad = [a / max(b, 1) > 0.5 for a, b, _ in rows]
        if kind == "garbled":
            out[i] = (sum(bad), len(rows))
        elif kind == "runaway":
            out[i] = (sum(t > 25 for _, _, t in rows), len(rows))
        else:
            keep = [r for r, x in zip(rows, bad) if not (x and kind == "wer_ex")]
            out[i] = (sum(r[0] for r in keep), sum(r[1] for r in keep))
    return out


def rate(c, ids):
    n = sum(c[i][1] for i in ids)
    return 100 * sum(c[i][0] for i in ids) / n


def boot(cs, ids, f, B=4000):
    rng = random.Random(0)
    v = sorted(f(*[rate(c, ii) for c in cs]) for ii in (rng.choices(ids, k=len(ids)) for _ in range(B)))
    return v[int(B * 0.025)], v[int(B * 0.975) - 1]


out, S = {"settings": {}, "differences": {}}, sorted(E)
for s in S:
    o = {"runs": runs[s], "failed": failed[s], "distinct_audio": len(md5[s]),
         "mean_s": sum(t for r in E[s].values() for _, _, t in r) / sum(map(len, E[s].values()))}
    for kind, drop in (("garbled", 0), ("garbled", 1), ("runaway", 0), ("wer", 0), ("wer_ex", 0)):
        c = cell(s, drop, kind)
        ids = sorted(c)
        k, n = sum(c[i][0] for i in ids), sum(c[i][1] for i in ids)
        o[kind + ("_ex_near" if drop else "")] = {
            "k": k, "n": n, "pct": 100 * k / n, "wilson": wilson(k, n), "boot_texts": boot([c], ids, lambda a: a),
            **({"texts_hit": sum(v[0] > 0 for v in c.values())} if "wer" not in kind else {})}
    out["settings"][s] = o
    print(f"{s}: runs {o['runs']} failed {o['failed']} distinct audio {o['distinct_audio']} mean {o['mean_s']:.2f} s")
    for kind in ("garbled", "garbled_ex_near", "runaway", "wer", "wer_ex"):
        v = o[kind]
        ci = v["boot_texts"] if "wer" in kind else v["wilson"]
        print(f"   {kind:16s} {v['k']}/{v['n']} = {v['pct']:.3f}% [{ci[0]:.3f}, {ci[1]:.3f}]"
              + (f" texts hit {v['texts_hit']}" if "texts_hit" in v else ""))
    per = Counter({i: v[0] for i, v in cell(s, 0, "garbled").items() if v[0]})
    print("   garbled by text", per.most_common(12))
    o["garbled_by_text"] = dict(per)
for x, a in enumerate(S):
    for b in S[x + 1:]:
        for kind, drop in (("garbled", 0), ("garbled", 1), ("runaway", 0), ("wer", 0), ("wer_ex", 0)):
            ca, cb = cell(a, drop, kind), cell(b, drop, kind)
            ids = sorted(set(ca) & set(cb))
            ka, na, kb, nb = (sum(c[i][j] for i in ids) for c, j in ((ca, 0), (ca, 1), (cb, 0), (cb, 1)))
            r = {"diff_pct": rate(ca, ids) - rate(cb, ids), "boot_texts": boot([ca, cb], ids, lambda p, q: p - q)}
            if "wer" not in kind:
                r["fisher_p"] = fisher(ka, na, kb, nb)
            out["differences"][f"{a} - {b} {kind}{'_ex_near' if drop else ''}"] = r
            print(f"{a} - {b} {kind}{' ex near' if drop else ''}: {r['diff_pct']:+.3f} points [{r['boot_texts'][0]:+.3f}, "
                  f"{r['boot_texts'][1]:+.3f}]" + (f" Fisher p={r['fisher_p']:.3f}" if "fisher_p" in r else ""))
if len(sys.argv) > 2:
    json.dump(out, open(sys.argv[2], "w"), indent=1)
