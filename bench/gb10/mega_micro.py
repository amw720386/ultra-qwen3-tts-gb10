"""Megakernel feasibility on GB10: the code predictor's 5 layers x 4 weight GEMVs (M=1, FP8 weights) as
(a) 20 separate GEMV kernels replayed from a CUDA graph, vs (b) one persistent kernel that walks the same
20 GEMVs with a software grid barrier (atomic counter) between dependent GEMVs. Reports time per chain,
effective weight bandwidth and barrier cost, and checks both give the same result.

    python -m bench.gb10.mega_micro
"""
import itertools
import json

import torch
import triton
import triton.language as tl

# (N, K) of the code predictor's GEMVs per layer: qkv, o, gate_up, down. Each consumes the first K values
# of the previous output (timing stand-in for attention / activation in between).
LAYER = [(4096, 1024), (1024, 2048), (6144, 1024), (1024, 3072)]
NL = 5


@triton.jit
def _gemv_rows(x_ptr, w_ptr, s_ptr, y_ptr, N, K: tl.constexpr, pid, nprog, BN: tl.constexpr, BK: tl.constexpr):
    """y[n] = s[n] * sum_k w[n, k] x[k] for this program's row tiles n0 = (pid + j * nprog) * BN."""
    rn0 = tl.arange(0, BN)
    rk0 = tl.arange(0, BK)
    for t in range(pid, tl.cdiv(N, BN), nprog):
        rn = t * BN + rn0
        acc = tl.zeros([BN], dtype=tl.float32)
        for k0 in tl.static_range(0, K, BK):
            x = tl.load(x_ptr + k0 + rk0).to(tl.float32)
            w = tl.load(w_ptr + rn[:, None] * K + (k0 + rk0)[None, :]).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)
        tl.store(y_ptr + rn, (acc * tl.load(s_ptr + rn)).to(tl.bfloat16))


@triton.jit
def _gemv_kernel(x_ptr, w_ptr, s_ptr, y_ptr, N, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    _gemv_rows(x_ptr, w_ptr, s_ptr, y_ptr, N, K, tl.program_id(0), tl.num_programs(0), BN, BK)


@triton.jit
def _grid_sync(cnt_ptr, target):
    """All programs arrive, then wait until `target` arrivals are visible (acq_rel atomics, gpu scope)."""
    tl.atomic_add(cnt_ptr, 1)
    v = tl.atomic_add(cnt_ptr, 0)
    while v < target:
        v = tl.atomic_add(cnt_ptr, 0)


@triton.jit
def _chain_kernel(x_ptr, a_ptr, b_ptr, w0, w1, w2, w3, s0, s1, s2, s3, cnt_ptr,
                  NL: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    ph = 0
    src = x_ptr
    for li in tl.static_range(NL):
        # ping-pong a/b; weights of all layers share one set here (timing: they stream from DRAM each time
        # because the chain's weights (~15 MB per layer) are re-read every layer and exceed what stays hot)
        _gemv_rows(src, w0 + li * 4096 * 1024, s0 + li * 4096, a_ptr, 4096, 1024, pid, nprog, BN, BK)
        ph += 1
        _grid_sync(cnt_ptr, ph * nprog)
        _gemv_rows(a_ptr, w1 + li * 1024 * 2048, s1 + li * 1024, b_ptr, 1024, 2048, pid, nprog, BN, BK)
        ph += 1
        _grid_sync(cnt_ptr, ph * nprog)
        _gemv_rows(b_ptr, w2 + li * 6144 * 1024, s2 + li * 6144, a_ptr, 6144, 1024, pid, nprog, BN, BK)
        ph += 1
        _grid_sync(cnt_ptr, ph * nprog)
        _gemv_rows(a_ptr, w3 + li * 1024 * 3072, s3 + li * 1024, b_ptr, 1024, 3072, pid, nprog, BN, BK)
        ph += 1
        _grid_sync(cnt_ptr, ph * nprog)
        src = b_ptr


@triton.jit
def _sync_only_kernel(cnt_ptr, NPH: tl.constexpr):
    nprog = tl.num_programs(0)
    for p in tl.static_range(NPH):
        _grid_sync(cnt_ptr, (p + 1) * nprog)


def graph_time(fn, iters=50):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        g.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


def main():
    dev = "cuda"
    out = {}
    with torch.inference_mode():
        Ws = [torch.randn(NL * n, k, device=dev).mul(0.02).to(torch.float8_e4m3fn) for n, k in LAYER]
        Ss = [torch.rand(NL * n, device=dev) * 0.1 + 0.01 for n, k in LAYER]
        wbytes = sum(NL * n * k for n, k in LAYER)
        x = torch.randn(4096, device=dev, dtype=torch.bfloat16)
        a = torch.zeros(6144, device=dev, dtype=torch.bfloat16)
        b = torch.zeros(6144, device=dev, dtype=torch.bfloat16)
        cnt = torch.zeros(1, device=dev, dtype=torch.int32)
        nsm = torch.cuda.get_device_properties(0).multi_processor_count
        out["sms"] = nsm

        def separate(BN, BK, nprog):
            def f():
                src = x
                for li in range(NL):
                    for j, (n, k) in enumerate(LAYER):
                        dst = a if j % 2 == 0 else b
                        _gemv_kernel[(min(nprog, triton.cdiv(n, BN)),)](
                            src, Ws[j][li * n:], Ss[j][li * n:], dst, n, K=k, BN=BN, BK=BK, num_warps=4)
                        src = dst
            return f

        def chain(BN, BK, nprog, nw):
            def f():
                cnt.zero_()
                _chain_kernel[(nprog,)](x, a, b, *Ws, *[s for s in Ss], cnt, NL=NL, BN=BN, BK=BK, num_warps=nw)
            return f

        res = {}
        for BN, BK, nprog in itertools.product((4, 8, 16), (256, 512), (96, 192, 384)):
            t = graph_time(separate(BN, BK, nprog))
            if "sep" not in res or t < res["sep"][1]:
                res["sep"] = ((BN, BK, nprog), t)
        ref = (separate(*res["sep"][0])(), b.clone())[1]
        for BN, BK, nmul, nw in itertools.product((4, 8, 16), (256, 512), (1, 2, 4), (4, 8)):
            nprog = nsm * nmul
            try:
                t = graph_time(chain(BN, BK, nprog, nw))
            except Exception as e:  # noqa: BLE001
                print("chain fail", (BN, BK, nprog, nw), repr(e)[:100], flush=True)
                continue
            if "chain" not in res or t < res["chain"][1]:
                res["chain"] = ((BN, BK, nprog, nw), t)
        chain(*res["chain"][0])()
        torch.cuda.synchronize()
        err = float((b.float() - ref.float()).abs().max() / ref.float().abs().max())
        for nmul in (1, 2, 4):
            def so(nmul=nmul):
                cnt.zero_()
                _sync_only_kernel[(nsm * nmul,)](cnt, NPH=100, num_warps=4)
            res[f"barrier_us_nprog{nsm * nmul}"] = graph_time(so) * 1000 / 100
        out.update({k: v for k, v in res.items()})
        out["sep_GBps"] = wbytes / res["sep"][1] / 1e6
        out["chain_GBps"] = wbytes / res["chain"][1] / 1e6
        out["rel_err"] = err
        out["weight_MB"] = wbytes / 1e6
        print(json.dumps(out, indent=1), flush=True)
    json.dump(out, open("/root/results/mega_micro.json", "w"), indent=1)


if __name__ == "__main__":
    main()
