"""Speed with ("wer") and without ("old") the cuBLAS workspace fix, every load pass of every server start.

    python bench/gb10/fixspeed_stats.py results/gb10/fixspeed
"""
import glob
import itertools
import json
import re
import statistics as st
import sys
from collections import defaultdict

A = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{sys.argv[1]}/fixspeed-*.log")):
    side = re.search(r"fixspeed-(old|wer)\d", f).group(1)
    for m in re.finditer(r"h2h-load (\{.*\})", open(f, encoding="utf-8", errors="replace").read()):
        x = json.loads(m.group(1))
        A[side][x["C"]].append(x)


def perm(a, b):  # permutation test on the difference of means, passes exchangeable
    d, p = abs(st.mean(a) - st.mean(b)), a + b
    if len(p) <= 18:
        c = [abs(sum(s) / len(a) - (sum(p) - sum(s)) / len(b)) >= d - 1e-9 for s in itertools.combinations(p, len(a))]
    else:
        import random
        r, c = random.Random(0), []
        for _ in range(20000):
            r.shuffle(p)
            c.append(abs(st.mean(p[:len(a)]) - st.mean(p[len(a):])) >= d - 1e-9)
    return sum(c) / len(c)


for c in sorted(A["old"]):
    if c == 1:
        continue
    for k in ("throughput_audio_s_per_s", "ttfa_ms_p50", "heard_ms_p50", "rtf_p50"):
        a, b = ([x[k] for x in A[s][c]] for s in ("old", "wer"))
        print(f"{c:4d} {k:25s} without {st.mean(a):7.2f} [{min(a):.2f}-{max(a):.2f}] n={len(a)}  with {st.mean(b):7.2f} "
              f"[{min(b):.2f}-{max(b):.2f}] n={len(b)}  {100 * (st.mean(b) / st.mean(a) - 1):+.1f}%  p={perm(a, b):.2f}")
print("failed", sum(x["n"] - x["n_ok"] for s in A.values() for r in s.values() for x in r))
