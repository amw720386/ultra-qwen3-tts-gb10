"""Fused Talker decode step (batch 1) as a programmatic-dependent-launch kernel chain.

Per layer: QKV GEMM (input RMSNorm weight folded, row scale applied in the epilogue), a split-key
attention kernel (q/k RMSNorm, RoPE, this step's K/V write, attention over [0, pos]) plus a
combine kernel, O GEMM accumulated into the residual stream in place, gate/up GEMM with a SiLU
epilogue (post-attention RMSNorm folded), down GEMM accumulated in place. The final RMSNorm is
folded into the codec head. GEMMs reuse the chain kernels ported from vLLM-Omni
(`uqt/fused_cp.py`); every kernel loads its weights (and, for attention, the already-written
past keys/values) before `gdc_wait`, so it streams memory while the previous kernel runs.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from uqt.fused_cp import _PDL_GU_BN, _cp_gemm_kernel, _cp_gemm_stream_kernel, _head_norm_rope


@triton.jit
def _talker_attn_kernel(
    qkv_ptr, kc_ptr, vc_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr, pos_ptr,
    ws_m_ptr, ws_l_ptr, ws_o_ptr, eps, scale,
    H: tl.constexpr, KVH: tl.constexpr, HD: tl.constexpr, S: tl.constexpr,
    BLK: tl.constexpr, NSPLIT: tl.constexpr,
):
    kvh = tl.program_id(0)
    sp = tl.program_id(1)
    G: tl.constexpr = H // KVH
    HALF: tl.constexpr = HD // 2
    W: tl.constexpr = (H + 2 * KVH) * HD
    d = tl.arange(0, HALF)
    j = sp * BLK + tl.arange(0, BLK)
    jm = j < S
    cache = (kvh * S).to(tl.int64) * HD
    # constants and already-written past K/V: loaded before the wait
    qw1 = tl.load(qw_ptr + d).to(tl.float32)
    qw2 = tl.load(qw_ptr + HALF + d).to(tl.float32)
    kw1 = tl.load(kw_ptr + d).to(tl.float32)
    kw2 = tl.load(kw_ptr + HALF + d).to(tl.float32)
    kp = kc_ptr + cache + j[:, None] * HD + d[None, :]
    vp = vc_ptr + cache + j[:, None] * HD + d[None, :]
    k1 = tl.load(kp, mask=jm[:, None], other=0.0).to(tl.float32)
    k2 = tl.load(kp + HALF, mask=jm[:, None], other=0.0).to(tl.float32)
    v1 = tl.load(vp, mask=jm[:, None], other=0.0).to(tl.float32)
    v2 = tl.load(vp + HALF, mask=jm[:, None], other=0.0).to(tl.float32)
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    pos = tl.load(pos_ptr).to(tl.int32)
    c1 = tl.load(cos_ptr + pos * HD + d).to(tl.float32)
    c2 = tl.load(cos_ptr + pos * HD + HALF + d).to(tl.float32)
    s1 = tl.load(sin_ptr + pos * HD + d).to(tl.float32)
    s2 = tl.load(sin_ptr + pos * HD + HALF + d).to(tl.float32)
    kx1 = tl.load(qkv_ptr + (H + kvh) * HD + d).to(tl.float32)
    kx2 = tl.load(qkv_ptr + (H + kvh) * HD + HALF + d).to(tl.float32)
    vn1 = tl.load(qkv_ptr + (H + KVH + kvh) * HD + d)
    vn2 = tl.load(qkv_ptr + (H + KVH + kvh) * HD + HALF + d)
    kn1, kn2 = _head_norm_rope(kx1, kx2, kw1, kw2, c1, c2, s1, s2, eps, HD)
    dtc = kc_ptr.dtype.element_ty
    wmask = (d < HALF) & (sp == 0)
    tl.store(kc_ptr + cache + pos * HD + d, kn1.to(dtc), mask=wmask)
    tl.store(kc_ptr + cache + pos * HD + HALF + d, kn2.to(dtc), mask=wmask)
    tl.store(vc_ptr + cache + pos * HD + d, vn1.to(dtc), mask=wmask)
    tl.store(vc_ptr + cache + pos * HD + HALF + d, vn2.to(dtc), mask=wmask)
    sel = (j == pos)[:, None]
    k1 = tl.where(sel, kn1.to(dtc).to(tl.float32)[None, :], k1)
    k2 = tl.where(sel, kn2.to(dtc).to(tl.float32)[None, :], k2)
    v1 = tl.where(sel, vn1.to(tl.float32)[None, :], v1)
    v2 = tl.where(sel, vn2.to(tl.float32)[None, :], v2)
    valid = j <= pos
    for gi in tl.static_range(G):
        h = kvh * G + gi
        qx1 = tl.load(qkv_ptr + h * HD + d).to(tl.float32)
        qx2 = tl.load(qkv_ptr + h * HD + HALF + d).to(tl.float32)
        q1, q2 = _head_norm_rope(qx1, qx2, qw1, qw2, c1, c2, s1, s2, eps, HD)
        s = (tl.sum(k1 * q1[None, :], axis=1) + tl.sum(k2 * q2[None, :], axis=1)) * scale
        s = tl.where(valid, s, -float("inf"))
        m = tl.max(s, axis=0)
        e = tl.where(valid, tl.exp(s - m), 0.0)
        l = tl.sum(e, axis=0)
        o1 = tl.sum(v1 * e[:, None], axis=0)
        o2 = tl.sum(v2 * e[:, None], axis=0)
        base = h * NSPLIT + sp
        tl.store(ws_m_ptr + base, m)
        tl.store(ws_l_ptr + base, l)
        tl.store(ws_o_ptr + base * HD + d, o1)
        tl.store(ws_o_ptr + base * HD + HALF + d, o2)


@triton.jit
def _talker_attn_combine_kernel(ws_m_ptr, ws_l_ptr, ws_o_ptr, out_ptr, HD: tl.constexpr, NSPLIT: tl.constexpr,
                                NSP2: tl.constexpr):
    h = tl.program_id(0)
    sidx = tl.arange(0, NSP2)
    smask = sidx < NSPLIT
    d = tl.arange(0, HD)
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    m = tl.load(ws_m_ptr + h * NSPLIT + sidx, mask=smask, other=-float("inf"))
    l = tl.load(ws_l_ptr + h * NSPLIT + sidx, mask=smask, other=0.0)
    M = tl.max(m, axis=0)
    w = tl.where(smask & (m > -float("inf")), tl.exp(m - M), 0.0)
    o = tl.load(ws_o_ptr + (h * NSPLIT + sidx)[:, None] * HD + d[None, :], mask=smask[:, None], other=0.0)
    acc = tl.sum(o * w[:, None], axis=0) / tl.sum(l * w, axis=0)
    tl.store(out_ptr + h * HD + d, acc.to(out_ptr.dtype.element_ty))


# Register-panel GEMM configs (BN, BK, NCH, BM, warps, MT, MSTAGES) to try per Talker shape.
# Streaming-K kernel configs: ("s", BN, BK, SK, BM, warps, STAGES).
_REG = [(16, 256, 2, 16, 4, 1, 1), (16, 256, 4, 16, 4, 1, 1), (32, 256, 2, 16, 4, 1, 1), (32, 256, 4, 16, 4, 1, 1),
        (32, 256, 1, 16, 4, 1, 2), (64, 256, 2, 16, 4, 1, 1), (64, 128, 4, 16, 4, 1, 1), (128, 256, 2, 16, 8, 1, 1)]
_STREAM = [("s", BN, BK, SK, 16, w, st) for BN in (16, 32, 64) for BK in (128, 256) for SK in (1, 2, 4)
           for w in (4,) for st in (3, 4)]
_CANDIDATES = {k: _REG + _STREAM for k in ("qkv", "o", "gu", "down", "head")}


def _fits(cfg, n, k):
    if cfg[0] == "s":
        _, bn, bk, sk = cfg[:4]
        return n % bn == 0 and k % (bk * sk) == 0 and (k // sk) % 64 == 0 and (k // sk) >= bk
    bn, bk, nch = cfg[:3]
    return n % bn == 0 and k % (bk * nch) == 0 and nch in (1, 2, 4)


class FusedTalker:
    """Built from the engine's `_Stack` Talker weights; runs one decode step for batch 1."""

    RMAX = 16

    def __init__(self, eng, attn_block: int = 64):
        T = eng.talker
        dev = eng.dev
        assert eng.dtype == torch.bfloat16
        self.eng = eng
        self.H, self.KVH, self.HD = T.nh, T.nkv, T.hd
        self.eps = T.eps
        self.hid = eng.t_emb.shape[1]
        self.scale = self.HD ** -0.5
        self.S = eng.max_len
        self.blk = attn_block
        bf = torch.bfloat16
        half = _PDL_GU_BN // 2

        def fold(w, ln):  # [N, K] with the RMSNorm weight folded into its columns
            return (w.float() * ln.float()[None, :]).to(bf).contiguous()

        self.layers = []
        for p in T.L:
            gu = fold(p["wgu"], p["ln2"])
            inter = gu.shape[0] // 2
            gu = torch.stack((gu[:inter].reshape(-1, half, self.hid), gu[inter:].reshape(-1, half, self.hid)), 1)
            self.layers.append(dict(qkv=fold(p["wqkv"], p["ln1"]), o=p["wo"].contiguous(),
                                    gu=gu.reshape(-1, self.hid).contiguous(), down=p["wd"].contiguous(),
                                    qn=p["qn"], kn=p["kn"]))
        self.head = fold(eng.head, T.norm)
        self.inter = self.layers[0]["gu"].shape[0] // 2
        R = self.RMAX
        self.x = torch.zeros(R, self.hid, device=dev, dtype=bf)
        self.qkv = torch.zeros(R, self.layers[0]["qkv"].shape[0], device=dev, dtype=bf)
        self.att = torch.zeros(R, self.H * self.HD, device=dev, dtype=bf)
        self.act = torch.zeros(R, self.inter, device=dev, dtype=bf)
        self.logits = torch.zeros(R, self.head.shape[0], device=dev, dtype=bf)
        self.ss = [torch.zeros(self.hid // 16, R, device=dev, dtype=torch.float32) for _ in range(2)]
        widest = max(self.qkv.shape[1], 2 * self.inter, self.head.shape[0], self.hid)
        self.ws = torch.zeros(16 * R * widest, device=dev, dtype=torch.float32)
        self.cnt = torch.zeros(16384, device=dev, dtype=torch.int32)
        self.nsplit_max = triton.cdiv(self.S, self.blk)
        self.ws_m = torch.zeros(self.H * self.nsplit_max, device=dev, dtype=torch.float32)
        self.ws_l = torch.zeros_like(self.ws_m)
        self.ws_o = torch.zeros(self.H * self.nsplit_max * self.HD, device=dev, dtype=torch.float32)
        self.cfg = {}
        self._tune()

    # ---------------------------------------------------------------- launches

    def _gemm(self, cfg, a, w, out, ss_in, nt_in, ss_out, epi, eps=0.0):
        N, K = w.shape
        if cfg[0] == "s":
            _, BN, BK, SK, BM, warps, stages = cfg
            _cp_gemm_stream_kernel[(N // BN, SK, 1)](
                a, 1, 0, w, self.ws, self.cnt, out, ss_in, ss_out, 1, eps,
                K=K, N=N, HID=self.hid, RMAX=self.RMAX, BN=BN, BK=BK, SK=SK, BM=BM, EPI=epi,
                NT_IN=nt_in, STAGES=stages, NLP=triton.next_power_of_2(K // SK // 64),
                num_warps=warps, launch_pdl=True,
            )
            return self.hid // BN if epi == 2 else 0
        BN, BK, NCH, BM, warps, MT, mst = cfg
        _cp_gemm_kernel[(N // BN, K // (BK * NCH), 1)](
            a, 1, 0, w, self.ws, self.cnt, out, ss_in, ss_out, 1, eps,
            K=K, N=N, HID=self.hid, RMAX=self.RMAX, BN=BN, BK=BK, NCH=NCH, BM=BM, EPI=epi,
            NT_IN=nt_in, MSTAGES=mst, num_warps=warps, launch_pdl=True,
        )
        return self.hid // BN if epi == 2 else 0

    def _attn(self, li, Lb):
        e = self.eng
        nsplit = triton.cdiv(Lb, self.blk)
        L = self.layers[li]
        _talker_attn_kernel[(self.KVH, nsplit)](
            self.qkv, e.kc[li], e.vc[li], L["qn"], L["kn"], e.t_cos, e.t_sin, e.pos,
            self.ws_m, self.ws_l, self.ws_o, self.eps, self.scale,
            H=self.H, KVH=self.KVH, HD=self.HD, S=self.S, BLK=self.blk, NSPLIT=nsplit,
            num_warps=4, launch_pdl=True,
        )
        _talker_attn_combine_kernel[(self.H,)](
            self.ws_m, self.ws_l, self.ws_o, self.att, HD=self.HD, NSPLIT=nsplit,
            NSP2=triton.next_power_of_2(nsplit), num_warps=4, launch_pdl=True,
        )

    def step(self, inp: torch.Tensor, Lb: int) -> torch.Tensor:
        """inp [1, hid] bf16 (Talker input embedding). Writes K/V at eng.pos. Returns logits [1, V] bf16."""
        self.x[:1].copy_(inp)
        ss_cur, ss_nxt = self.ss
        torch.sum(self.x[:1].float().square(), dim=1, out=ss_cur[0, :1])
        nt = 1
        c = self.cfg
        for li, L in enumerate(self.layers):
            self._gemm(c["qkv"], self.x, L["qkv"], self.qkv, ss_cur, nt, None, 0, self.eps)
            self._attn(li, Lb)
            nt = self._gemm(c["o"], self.att, L["o"], self.x, None, 0, ss_nxt, 2)
            ss_cur, ss_nxt = ss_nxt, ss_cur
            self._gemm(c["gu"], self.x, L["gu"], self.act, ss_cur, nt, None, 1, self.eps)
            nt = self._gemm(c["down"], self.act, L["down"], self.x, None, 0, ss_nxt, 2)
            ss_cur, ss_nxt = ss_nxt, ss_cur
        self._gemm(c["head"], self.x, self.head, self.logits, ss_cur, nt, None, 0, self.eps)
        self.ss = [ss_cur, ss_nxt]
        return self.logits[:1]

    def final_hidden(self) -> torch.Tensor:
        """Post-final-norm hidden state (the code predictor's input), computed like HF RMSNorm."""
        x = self.x[:1]
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.eng.talker.norm * xf.to(x.dtype)

    # ---------------------------------------------------------------- tuning

    def _tune(self):
        """Pick the fastest fitting config per GEMM shape (isolated timing)."""
        L = self.layers[0]
        shapes = {"qkv": (L["qkv"], self.x, self.qkv, 0), "o": (L["o"], self.att, self.x, 2),
                  "gu": (L["gu"], self.x, self.act, 1), "down": (L["down"], self.act, self.x, 2),
                  "head": (self.head, self.x, self.logits, 0)}
        saved_x = self.x.clone()
        for name, (w, a, out, epi) in shapes.items():
            N, K = w.shape
            best = None
            for cfg in _CANDIDATES[name]:
                bn = cfg[1] if cfg[0] == "s" else cfg[0]
                if not _fits(cfg, N, K) or (name == "gu" and bn != _PDL_GU_BN) or (epi == 2 and self.hid // bn > self.hid // 16):
                    continue
                try:
                    run = lambda: self._gemm(cfg, a, w, out, self.ss[0], 1, self.ss[1], epi, self.eps)
                    for _ in range(3):
                        run()
                    torch.cuda.synchronize()
                    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    st.record()
                    for _ in range(50):
                        run()
                    en.record()
                    torch.cuda.synchronize()
                    t = st.elapsed_time(en) / 50
                except Exception as ex:  # config does not compile for this shape
                    print(f"[uqt] talker gemm {name} cfg {cfg} failed: {ex!r}"[:300])
                    continue
                if best is None or t < best[0]:
                    best = (t, cfg)
            assert best is not None, f"no GEMM config fits {name} {tuple(w.shape)}"
            self.cfg[name] = best[1]
            print(f"[uqt] talker gemm {name} {tuple(w.shape)}: {best[1]} {best[0] * 1000:.1f} us")
        self.x.copy_(saved_x)
        self.x.zero_()
        self.act.zero_()
        self.att.zero_()
