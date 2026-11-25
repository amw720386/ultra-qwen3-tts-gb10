"""Two CUDA graphs captured in separate pools, replayed at the same time on two streams: does a GEMM in one still give
its serial result? PyTorch keeps one cuBLAS workspace per (handle, stream), and every torch.cuda.graph capture uses the
same capture stream, so both graphs bake in the same workspace; split-K GEMMs keep partial sums and arrival counters there.

    python -m bench.gb10.cublas_ws_race [--clear]   # --clear: drop cached workspaces around each capture
"""
import argparse
import sys
import threading
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--clear", action="store_true")
ap.add_argument("--reps", type=int, default=4)
ap.add_argument("--only", default="", help="m,k,n")
ap.add_argument("--iters", type=int, default=400)
a = ap.parse_args()
dev = torch.device("cuda")
g = torch.Generator(dev).manual_seed(0)
R = lambda *s: torch.randn(*s, device=dev, dtype=torch.bfloat16, generator=g)  # noqa: E731
shapes = [(m, k, n) for m in (16, 48, 64, 128, 256) for k, n in ((2048, 12288), (6144, 2048), (2048, 4096), (1024, 6144), (3072, 1024))]

if a.only:
    shapes = [tuple(map(int, a.only.split(",")))]
shapes = shapes * a.reps


def clear():
    torch.cuda.synchronize()
    torch._C._cuda_clearCublasWorkspaces()


def capture(xs, ws):
    op = (lambda x, w: torch.nn.functional.linear(x, w))  # weights [N, K] as in the engine
    outs = [torch.empty(x.shape[0], w.shape[0], device=dev, dtype=torch.bfloat16) for x, w in zip(xs, ws)]
    for x, w in zip(xs, ws):
        op(x, w)
    if a.clear:
        clear()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr, pool=torch.cuda.graph_pool_handle()):
        for o, x, w in zip(outs, xs, ws):
            o.copy_(op(x, w))
    if a.clear:
        clear()
    return gr, outs


last, stage, bad, done_it = [time.time()], ["setup"], {}, [0]


def dog():
    while True:
        time.sleep(5)
        if time.time() - last[0] > 30:
            print(f"RESULT hang in stage {stage[0]!r} after {done_it[0]} good pairs; wrong outputs before it: {sum(bad.values())} {bad}", flush=True)
            import os
            os._exit(3)


threading.Thread(target=dog, daemon=True).start()
with torch.inference_mode():
    xs = [R(m, k) for m, k, n in shapes]
    ws = [R(n, k) for m, k, n in shapes]
    ga, oa = capture(xs, ws)
    gb, ob = capture(xs[::-1], ws[::-1])
    ga.replay(); torch.cuda.synchronize(); ra = [o.clone() for o in oa]
    gb.replay(); torch.cuda.synchronize(); rb = [o.clone() for o in ob]
    def wall(conc):
        torch.cuda.synchronize(); t = time.time()
        for _ in range(20):
            if conc:
                with torch.cuda.stream(s1):
                    ga.replay()
                with torch.cuda.stream(s2):
                    gb.replay()
            else:
                ga.replay(); gb.replay()
        torch.cuda.synchronize(); return time.time() - t
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    ser = 0
    stage[0], last[0] = "serial", time.time()
    ws_ = wall(False)
    last[0] = time.time()
    for _ in range(20):  # serial replays must repeat exactly
        ga.replay(); torch.cuda.synchronize(); ser += sum(not torch.equal(o, r) for o, r in zip(oa, ra))
        gb.replay(); torch.cuda.synchronize(); ser += sum(not torch.equal(o, r) for o, r in zip(ob, rb))
    worst = 0.0
    stage[0], last[0] = "concurrent", time.time()
    for it in range(a.iters):
        with torch.cuda.stream(s1):
            ga.replay()
        with torch.cuda.stream(s2):
            gb.replay()
        torch.cuda.synchronize()
        last[0] = time.time()
        done_it[0] += 1
        for sh, o, r in list(zip(shapes, oa, ra)) + list(zip(shapes[::-1], ob, rb)):
            if not torch.equal(o, r):
                bad[sh] = bad.get(sh, 0) + 1
                worst = max(worst, (o.float() - r.float()).abs().max().item() / r.float().abs().mean().item())
stage[0] = "timing"
wc_ = wall(True)
print(f"20 pairs: one stream {ws_:.2f} s, two streams {wc_:.2f} s")
print("torch", torch.__version__, "clear", a.clear, "| serial mismatches", ser, "| concurrent", a.iters, "pairs x", len(shapes), "GEMMs:",
      sum(bad.values()), "wrong outputs, worst error / mean |out|", round(worst, 3))
print("by (rows, K, N):", dict(sorted(bad.items())))
