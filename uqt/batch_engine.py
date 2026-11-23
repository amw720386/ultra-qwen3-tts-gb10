"""Multi-stream (continuous batching) decode engine for Qwen3-TTS-12Hz.

Every active stream owns a slot. One graphed step advances all slots [0, B) by K frames:
fused code predictor over B rows (vLLM-Omni Triton port, batched), then a compiled batched Talker
step (one GEMM per projection for all rows, per-row RoPE / KV write / masked attention), per-row
logits processing and top-k sampling. New requests are prefilled by the single-stream engine
(graphed per prompt-length bucket) and their KV/state copied into a free slot between steps.
Audio is produced by the stateful streaming codec, batched over slots.
"""
from __future__ import annotations

import time

import os

import torch
import torch.nn.functional as F

from uqt import kernels
from uqt.engine import _rms, _rot_half
from uqt.wq import lin


B_BUCKETS = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512)


def _bucket_b(n, cap):
    for b in B_BUCKETS:
        if b >= n and b <= cap:
            return b
    return cap


class BatchEngine:
    LB = (128, 256, 384, 512, 768, 1024, 1536, 2048)

    def __init__(self, eng, max_slots: int = 64, max_len: int = 1024):
        from uqt.fused_cp import FusedCodePredictor
        from uqt.streaming_codec import StreamingCodecDecoder

        self.e = eng
        dev, dt = eng.dev, eng.dtype
        assert dt == torch.bfloat16
        T = eng.talker
        self.S, self.L = max_slots, max_len
        assert max_len <= eng.max_len
        nl, D, V = len(T.L), eng.t_emb.shape[1], eng.V
        self.dev, self.dt, self.D, self.V = dev, dt, D, V
        # KV cache format (UQT_KV): bf16; int8 = per-(token, head) absmax-scaled int8, half the attention
        # reads at large batch; fp8 = plain e4m3 (kept for comparison)
        self.kv = os.environ.get("UQT_KV", "int8")
        self.kv_dt = {"bf16": dt, "fp8": torch.float8_e4m3fn, "int8": torch.int8}[self.kv]
        self.kc = torch.zeros(nl, max_slots, T.nkv, max_len, T.hd, device=dev, dtype=self.kv_dt)
        self.vc = torch.zeros_like(self.kc)
        if self.kv == "int8":
            self.ks = torch.zeros(nl, max_slots, T.nkv, max_len, device=dev, dtype=torch.float32)
            self.vs = torch.zeros_like(self.ks)
        self.code0 = torch.zeros(max_slots, dtype=torch.long, device=dev)
        self.hidden = torch.zeros(max_slots, D, device=dev, dtype=dt)
        self.pos = torch.zeros(max_slots, dtype=torch.long, device=dev)
        self.step = torch.zeros_like(self.pos)
        self.fidx = torch.zeros_like(self.pos)
        self.seen = torch.zeros(max_slots, V, dtype=torch.bool, device=dev)
        self.trail = torch.zeros(max_slots, max_len, D, device=dev, dtype=dt)
        self.codes = torch.zeros(max_slots, max_len, 16, dtype=torch.long, device=dev)
        self.fcodes = torch.zeros(max_slots, 16, dtype=torch.long, device=dev)
        self.ones = torch.ones(max_slots, dtype=torch.bool, device=dev)
        self.arange = torch.arange(max_len, device=dev)
        cp = eng.m.talker.code_predictor
        self.fcp = FusedCodePredictor(cp, max_batch=max_slots)
        if getattr(eng, "weights", "bf16") != "bf16":
            self.fcp.quantize(eng.weights)
        self.fcp_u = torch.empty(max_slots, 15, cp.config.vocab_size, device=dev, dtype=torch.float32)
        self.t_u = torch.empty(max_slots, V, device=dev, dtype=torch.float32)
        self.battn = os.environ.get("UQT_BATTN", "1") == "1"
        self._btalker_c = torch.compile(self._btalker, fullgraph=True, dynamic=True)
        self.graphs: dict = {}
        self._pool = None
        self.info: list = [None] * max_slots
        self.sdec = StreamingCodecDecoder(eng.m.speech_tokenizer.model.decoder, num_slots=max_slots)
        self.voice_state: dict = {}
        self.dec_graphs: dict = {}
        self._dec_pool = None
        self.dec_stream = torch.cuda.Stream()

    # ------------------------------------------------------------------ batched step

    def _btalker(self, B: int, Lb: int):
        E, Tk, s = self.e, self.e.talker, self.e.s
        c = self.fcodes[:B]
        parts = [F.embedding(c[:, 0], E.t_emb)] + [F.embedding(c[:, i], E.cp_emb[i - 1]) for i in range(1, 16)]
        rows = torch.arange(B, device=self.dev)
        kernels.rows_write(self.codes, self.fidx[:B].clamp(max=self.L - 1), c)
        x = torch.stack(parts, 0).sum(0) + self.trail[rows, self.step[:B].clamp(max=self.L - 1)]
        pos = self.pos[:B].clamp(max=self.L - 1)
        cos, sin = E.t_cos[pos][:, None], E.t_sin[pos][:, None]
        mask = torch.zeros(B, Lb, device=self.dev, dtype=self.dt).masked_fill(
            self.arange[:Lb][None, :] > pos[:, None], float("-inf"))[:, None, None, :]
        nh, nkv, hd = Tk.nh, Tk.nkv, Tk.hd
        g = nh // nkv
        for i, p in enumerate(Tk.L):
            h = _rms(x, p["ln1"], Tk.eps)
            q, k, v = lin(h, p["wqkv"]).split([Tk.nq, Tk.nk, Tk.nk], dim=-1)
            q = _rms(q.view(B, nh, hd), p["qn"], Tk.eps)
            k = _rms(k.view(B, nkv, hd), p["kn"], Tk.eps)
            v = v.view(B, nkv, hd)
            q = q * cos + _rot_half(q) * sin
            k = k * cos + _rot_half(k) * sin
            if self.kv == "int8":
                kernels.kv_write_rows_q(self.kc, self.vc, self.ks, self.vs, k, v, pos, i)
                o = kernels.bdec_attn_q(q, self.kc[i], self.vc[i], self.ks[i], self.vs[i], pos, hd ** -0.5)
            elif self.battn:  # per-row live-length decode attention (one kernel, no padded reads)
                kernels.kv_write_rows(self.kc, self.vc, k, v, pos, i)
                o = kernels.bdec_attn(q, self.kc[i], self.vc[i], pos, hd ** -0.5)
            else:
                kernels.kv_write_rows(self.kc, self.vc, k, v, pos, i)
                Kc = self.kc[i, :B, :, :Lb]
                Vc = self.vc[i, :B, :, :Lb]
                sc = torch.matmul(q.view(B, nkv, g, hd), Kc.transpose(-1, -2)) * (hd ** -0.5) + mask
                pr = torch.softmax(sc, dim=-1, dtype=torch.float32).to(self.dt)
                o = torch.matmul(pr, Vc).reshape(B, nh * hd)
            x = x + lin(o, p["wo"])
            h = _rms(x, p["ln2"], Tk.eps)
            ga, up = lin(h, p["wgu"]).chunk(2, dim=-1)
            x = x + lin(F.silu(ga) * up, p["wd"])
        h = _rms(x, Tk.norm, Tk.eps)
        lg = lin(h, E.head).float()
        rp = s.repetition_penalty
        seen = self.seen[:B]
        if rp != 1.0:
            lg = torch.where(seen, torch.where(lg < 0, lg * rp, lg / rp), lg)
        lg = lg.masked_fill(E.suppress[None, :], float("-inf"))
        ban = (self.fidx[:B] < (s.min_new_tokens - 1))[:, None] & E.eos_mask[None, :]
        lg = lg.masked_fill(ban, float("-inf"))
        if s.greedy:
            nc = lg.argmax(-1)
        else:
            u = self.t_u[:B].uniform_()
            nc = kernels.topk_sample_rows(lg.contiguous(), u, 1.0 / s.temperature, s.top_k)
        kernels.rows_write(self.seen, nc, self.ones[:B])
        self.hidden[:B].copy_(h)
        self.code0[:B].copy_(nc)
        self.pos[:B].add_(1)
        self.step[:B].add_(1)
        self.fidx[:B].add_(1)

    def _bframe(self, B, Lb):
        s = self.e.s
        if s.cp_greedy:
            self.fcp_u[:B].fill_(0.5)
            inv_t, k = 1.0, 0
        else:
            self.fcp_u[:B].uniform_()
            inv_t, k = 1.0 / s.cp_temperature, s.cp_top_k
        e0 = F.embedding(self.code0[:B], self.e.t_emb)
        self.fcodes[:B].copy_(self.fcp(self.code0[:B], e0, self.hidden[:B], inv_t, k, self.fcp_u[:B]))
        self._btalker_c(B, Lb)

    _STATE = ("code0", "hidden", "pos", "step", "fidx", "seen")

    def _graph(self, K, B, Lb):
        key = (K, B, Lb)
        g = self.graphs.get(key)
        if g is None:
            snap = [getattr(self, n).clone() for n in self._STATE]

            def run():
                for _ in range(K):
                    self._bframe(B, Lb)
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                run()
                run()
            torch.cuda.current_stream().wait_stream(st)
            for n, v in zip(self._STATE, snap):
                getattr(self, n).copy_(v)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._pool):
                run()
            self._pool = g.pool()
            for n, v in zip(self._STATE, snap):
                getattr(self, n).copy_(v)
            self.graphs[key] = g
        return g

    # ------------------------------------------------------------------ slots

    def _kv_put(self, slot, k, v):
        """Copy one prompt's K/V ([layers, n_kv, T, D], bf16, from the single-stream engine) into `slot`."""
        T = k.shape[2]
        if self.kv == "int8":
            kq, ksc = kernels.quant_rows(k)
            vq, vsc = kernels.quant_rows(v)
            self.kc[:, slot, :, :T].copy_(kq)
            self.vc[:, slot, :, :T].copy_(vq)
            self.ks[:, slot, :, :T].copy_(ksc)
            self.vs[:, slot, :, :T].copy_(vsc)
        else:
            self.kc[:, slot, :, :T].copy_(k)
            self.vc[:, slot, :, :T].copy_(v)

    def free_slot(self):
        for i, it in enumerate(self.info):
            if it is None:
                return i
        return None

    def admit(self, slot: int, prompt, max_frames: int, meta=None):
        """Prefill with the single-stream engine, then copy KV/state into `slot`."""
        E = self.e
        E.prefill(prompt)
        T = E.pos_host
        if T + max_frames + 8 >= self.L:
            max_frames = self.L - T - 9
        self._kv_put(slot, E.kc[:, :, :T], E.vc[:, :, :T])
        self.code0[slot:slot + 1].copy_(E.code0)
        self.hidden[slot:slot + 1].copy_(E.hidden)
        self.pos[slot:slot + 1].copy_(E.pos)
        self.step[slot] = 0
        self.fidx[slot] = 0
        self.seen[slot].copy_(E.seen)
        self.trail[slot].copy_(E.trail[:self.L])
        # the decoder slot may still be in use by the previous occupant's last decode on the decode stream
        self.dec_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.dec_stream):
            R = self._prime(slot, prompt.ref_codes)
        self.info[slot] = dict(pos=T, frames=0, dec_pos=R, max_frames=max_frames, done=False, meta=meta,
                                  valid=0, dec=0, ci=0)

    # ---- packed batched prefill: several prompts in one forward (positions restart per prompt,
    #      block-diagonal causal mask), then each prompt's K/V is copied into its slot.

    PACK_BUCKETS = (128, 256, 384, 512, 768, 1024, 1536, 2048)
    PACK_MAX = 16

    def _init_pack(self):
        E = self.e
        n, Tm = self.PACK_MAX, self.PACK_BUCKETS[-1]
        self.pk_emb = torch.zeros(Tm, self.D, device=self.dev, dtype=self.dt)
        self.pk_pos = torch.zeros(Tm, dtype=torch.long, device=self.dev)
        self.pk_seg = torch.full((Tm,), -1, dtype=torch.long, device=self.dev)
        self.pk_last = torch.zeros(n, dtype=torch.long, device=self.dev)
        self.pk_code0 = torch.zeros(n, dtype=torch.long, device=self.dev)
        self.pk_hidden = torch.zeros(n, self.D, device=self.dev, dtype=self.dt)
        self.pk_u = torch.empty(n, self.V, device=self.dev, dtype=torch.float32)
        self.pk_ar = torch.arange(Tm, device=self.dev)
        self._pack_c = torch.compile(self._pack_core, fullgraph=True, dynamic=False)
        self.pk_graphs: dict = {}

    def _pack_core(self, Tb: int):
        E, Tk, s = self.e, self.e.talker, self.e.s
        pos, seg = self.pk_pos[:Tb], self.pk_seg[:Tb]
        ar = self.pk_ar[:Tb]
        ok = (seg[:, None] == seg[None, :]) & (ar[None, :] <= ar[:, None])
        ok = ok | (ar[:, None] == ar[None, :])  # padding rows attend to themselves (no empty softmax)
        mask = torch.zeros(Tb, Tb, device=self.dev, dtype=self.dt).masked_fill(~ok, float("-inf"))
        h = Tk.forward(self.pk_emb[:Tb], E.t_cos[pos], E.t_sin[pos], E.kc, E.vc,
                       E._write_static(E.kc, E.vc, 0), Tb, mask)
        hl = h.index_select(0, self.pk_last)
        lg = lin(hl, E.head).float()
        lg = lg.masked_fill(E.suppress[None, :], float("-inf"))
        if s.min_new_tokens > 0:
            lg = lg.masked_fill(E.eos_mask[None, :], float("-inf"))
        if s.greedy:
            nc = lg.argmax(-1)
        else:
            nc = kernels.topk_sample_rows(lg.contiguous(), self.pk_u.uniform_(), 1.0 / s.temperature, s.top_k)
        self.pk_code0.copy_(nc)
        self.pk_hidden.copy_(hl)

    def warm_pack(self):
        """Compile and capture the packed-prefill graph for every bucket (a lazy compile costs ~30 s)."""
        if not hasattr(self, "pk_emb"):
            self._init_pack()
        for Tb in self.PACK_BUCKETS:
            if Tb not in self.pk_graphs:
                self.pk_pos[:Tb].copy_(self.pk_ar[:Tb])
                self.pk_seg[:Tb].zero_()
                self.pk_last.fill_(Tb - 1)
                self.pk_graphs[Tb] = self.e._capture(("pack", Tb), lambda Tb=Tb: self._pack_c(Tb))
        torch.cuda.synchronize()

    def admit_many(self, reqs):
        """reqs: list of (slot, prompt, max_frames, meta). One packed prefill for all of them."""
        if not reqs:
            return
        if not hasattr(self, "pk_emb"):
            self._init_pack()
        out = []
        i = 0
        while i < len(reqs):  # split into packs that fit the largest bucket / PACK_MAX prompts
            group, tot = [], 0
            while i < len(reqs) and len(group) < self.PACK_MAX and tot + reqs[i][1].embeds.shape[0] <= self.PACK_BUCKETS[-1]:
                group.append(reqs[i]); tot += reqs[i][1].embeds.shape[0]; i += 1
            if not group:
                raise ValueError("prompt longer than the largest pack bucket")
            self._admit_pack(group, tot)

    def _admit_pack(self, group, tot):
        E = self.e
        Tb = next(b for b in self.PACK_BUCKETS if b >= tot)
        offs, off = [], 0
        for j, (_, p, _, _) in enumerate(group):
            T = p.embeds.shape[0]
            self.pk_emb[off:off + T].copy_(p.embeds)
            offs.append(off)
            off += T
        lens = [p.embeds.shape[0] for _, p, _, _ in group]
        pos_h = torch.cat([torch.arange(T) for T in lens] + [torch.zeros(Tb - tot, dtype=torch.long)])
        seg_h = torch.cat([torch.full((T,), j) for j, T in enumerate(lens)] + [torch.full((Tb - tot,), -1)])
        last_h = torch.tensor([o + T - 1 for o, T in zip(offs, lens)] + [0] * (self.PACK_MAX - len(group)))
        self.pk_pos[:Tb].copy_(pos_h.to(self.dev, non_blocking=True))
        self.pk_seg[:Tb].copy_(seg_h.to(self.dev, non_blocking=True))
        self.pk_last.copy_(last_h.to(self.dev, non_blocking=True))
        self.pk_emb[tot:Tb].zero_()
        g = self.pk_graphs.get(Tb)
        if g is None:
            g = E._capture(("pack", Tb), lambda: self._pack_c(Tb))
            self.pk_graphs[Tb] = g
        g.replay()
        for j, (slot, p, max_frames, meta) in enumerate(group):
            T, o = lens[j], offs[j]
            if T + max_frames + 8 >= self.L:
                max_frames = self.L - T - 9
            self._kv_put(slot, E.kc[:, :, o:o + T], E.vc[:, :, o:o + T])
            self.code0[slot:slot + 1].copy_(self.pk_code0[j:j + 1])
            self.hidden[slot:slot + 1].copy_(self.pk_hidden[j:j + 1])
            self.pos[slot] = T
            self.step[slot] = 0
            self.fidx[slot] = 0
            self.seen[slot].zero_()
            self.seen[slot].index_fill_(0, self.pk_code0[j:j + 1], True)
            Tt = p.trailing.shape[0]
            self.trail[slot].copy_(p.pad.expand(self.L, -1))
            self.trail[slot, :Tt].copy_(p.trailing)
            self.dec_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.dec_stream):
                R = self._prime(slot, p.ref_codes)
            self.info[slot] = dict(pos=T, frames=0, dec_pos=R, max_frames=max_frames, done=False, meta=meta,
                                  valid=0, dec=0, ci=0)
        self._e_ev = torch.cuda.Event()
        self._e_ev.record()

    # ---- fast lane: the newest request's prefill and first frames run on the single-stream engine on
    #      its own CUDA stream, concurrently with the batched step already in flight; its first chunk
    #      is decoded and returned at once, then its state joins a batch slot.

    def admit_fast(self, slot, prompt, max_frames, meta, K0=2):
        """Returns (int16 PCM bytes of the first chunk, finished). Call after launching the batched step."""
        E = self.e
        if not hasattr(self, "fast_stream"):
            self.fast_stream = torch.cuda.Stream()
        fs, main = self.fast_stream, torch.cuda.current_stream()
        with torch.cuda.stream(fs):
            if getattr(self, "_e_ev", None) is not None:
                fs.wait_event(self._e_ev)  # single engine's KV scratch is shared with packed admission
            E.prefill(prompt)
            T = E.pos_host
            codes0 = E.step_frames(K0).clone()
            ev = torch.cuda.Event()
            ev.record(fs)
        c0 = codes0[:, 0].tolist()  # waits for the fast lane only (the batched step keeps running)
        k = c0.index(E.eos) if E.eos in c0 else K0
        k = min(k, max_frames)
        done = k < K0 or max_frames <= K0
        # first chunk straight from the fast lane's codes
        self.dec_stream.wait_event(ev)
        with torch.cuda.stream(self.dec_stream):
            R = self._prime(slot, prompt.ref_codes)
            g = self._dec_graph(K0)
            sl = torch.full((1,), slot, dtype=torch.int32, device=self.dev)
            pos = torch.full((1,), R, dtype=torch.int32, device=self.dev)
            wav = g(codes0[None].contiguous(), sl, pos)
            pcm = (wav[0, :k * self.sdec.spf].clamp(-1.0, 1.0) * 32767.0).to(torch.int16).cpu()
        b = pcm.numpy().tobytes() if k else b""
        if done:
            return b, True
        # hand the stream over to the batch: ordered after the batched step in flight on `main`
        main.wait_event(ev)
        self._kv_put(slot, E.kc[:, :, :T + K0], E.vc[:, :, :T + K0])
        self.code0[slot:slot + 1].copy_(E.code0)
        self.hidden[slot:slot + 1].copy_(E.hidden)
        self.pos[slot:slot + 1].copy_(E.pos)
        self.step[slot:slot + 1].copy_(E.step)
        self.fidx[slot:slot + 1].copy_(E.fidx)
        self.seen[slot].copy_(E.seen)
        self.trail[slot].copy_(E.trail[:self.L])
        self.codes[slot, :K0].copy_(codes0)
        self._e_ev = torch.cuda.Event()
        self._e_ev.record(main)
        self.info[slot] = dict(pos=T + K0, frames=K0, dec_pos=R + K0, max_frames=max_frames, done=False,
                               meta=meta, valid=k, dec=k, emitted=k, ci=1, t0=time.perf_counter())
        return b, False

    def release(self, slot: int):
        self.info[slot] = None

    def _prime(self, slot, ref):
        if ref is None or ref.shape[0] == 0:
            return 0  # position 0 reads zero history: no reset needed
        R = int(ref.shape[0])
        state = self.sdec._slot_state(slot)
        ent = self.voice_state.get(id(ref))
        if ent is not None and ent[0] is ref:
            for t, v in zip(state, ent[1]):
                t.copy_(v)
            return R
        sl = torch.full((1,), slot, dtype=torch.int32, device=self.dev)
        codes = ref.to(torch.int32)[None]
        first = min(R, self.sdec.RING)
        self.sdec(codes[:, :first].contiguous(), sl, torch.zeros(1, dtype=torch.int32, device=self.dev), initial=True)
        p = first
        while p < R:
            q = min(R, p + self.sdec.max_frames)
            self.sdec(codes[:, p:q].contiguous(), sl, torch.full((1,), p, dtype=torch.int32, device=self.dev))
            p = q
        self.voice_state[id(ref)] = (ref, [t.clone() for t in state])
        return R

    # ------------------------------------------------------------------ iteration

    def active(self):
        return [i for i, it in enumerate(self.info) if it is not None and not it["done"]]

    def generate(self, K: int):
        """Advance all active slots by K frames. Returns (slots, codes [n, K, 16] GPU, valid frames per row)."""
        return self.collect(self.launch(K))

    def launch(self, K: int):
        """Replay the batched step for all active slots without waiting; returns a handle for collect()."""
        act = self.active()
        if not act:
            return None
        B = _bucket_b(max(act) + 1, self.S)
        need = max(self.info[s]["pos"] for s in act) + K + 1
        Lb = next((b for b in self.LB if b >= need and b <= self.L), self.L)
        self._graph(K, B, Lb).replay()
        rows = kernels.h2d(act, self.dev)
        starts = kernels.h2d([self.info[s]["frames"] for s in act], self.dev)
        idx = (starts[:, None] + torch.arange(K, device=self.dev)[None, :]).clamp(max=self.L - 1)
        chunk = self.codes[rows[:, None], idx]
        if kernels.ASYNC_H2D[0]:  # a copy to pageable memory would block here
            c0 = torch.empty(chunk.shape[:2], dtype=chunk.dtype, pin_memory=True)
            c0.copy_(chunk[:, :, 0], non_blocking=True)
        else:
            c0 = chunk[:, :, 0].to("cpu", non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        for s in act:  # host bookkeeping for the frames now in flight
            self.info[s]["pos"] += K
        return (act, chunk, c0, ev, K)

    def collect(self, h):
        if h is None:
            return [], None, []
        act, chunk, c0, ev, K = h
        ev.synchronize()
        c0 = c0.tolist()
        valid = []
        for s, row in zip(act, c0):
            it = self.info[s]
            k = row.index(self.e.eos) if self.e.eos in row else K
            k = min(k, it["max_frames"] - it["frames"])
            it["frames"] += K
            if k < K or it["frames"] >= it["max_frames"]:
                it["done"] = True
            it["valid"] = it.get("valid", 0) + k
            valid.append(k)
        return act, chunk, valid

    # ---- growing decode chunks: first chunk after 2 frames (fast start), then 4, 8, 16 frames per
    #      decode call, so the codec runs far fewer, larger batches per second of audio.
    EARLY_CHUNKS = int(os.environ.get("UQT_EARLY_CHUNKS", "1"))  # 2 cut underruns at 64 clients (12% -> 4%) but not at 128+
    RAMP = (2, 2, 4, 8, 16)  # a second small chunk builds the playback buffer before chunks grow
    # Chunks that double run the client's buffer dry once generation needs more than half of real time
    # (RTF 0.5): a chunk is only sent when it is full, so a 16-frame chunk arrives 16 frames of generation
    # time after the one before. With DEADLINE a stream's pending frames go out early, undeferred if need
    # be, when its estimated buffer is about to run out. Costs more, smaller decode calls under load.
    DEADLINE = int(os.environ.get("UQT_DEADLINE", "1"))
    DEADLINE_MARGIN_MS = float(os.environ.get("UQT_DEADLINE_MARGIN_MS", "20"))
    # streams x frames per codec call: the late codec layers launch (rows / 32) blocks on one grid axis,
    # which must stay below 65535 (rows = streams * frames * 1920 samples)
    DEC_FRAME_BUDGET = 1024

    def _dec_max_b(self, F):
        return max(1, min(self.S, self.DEC_FRAME_BUDGET // F))

    def _dec_graph(self, F):
        from uqt.streaming_codec import StreamingDecodeGraphs

        g = self.dec_graphs.get(F)
        if g is None:
            sizes = [b for b in B_BUCKETS if b <= self._dec_max_b(F)]
            g = StreamingDecodeGraphs(self.sdec, sizes, frames=F, pool=self._dec_pool)
            self._dec_pool = g.pool
            self.dec_graphs[F] = g
        return g

    def decode_due(self):
        """Synchronous convenience wrapper: launch due decodes and wait for them."""
        return self.decode_finish(self.decode_launch())

    def decode_launch(self, split_first=False):
        """Launch decodes for every slot whose pending frames reached its next chunk size (or that
        finished) on the decode stream, without waiting. PCM is converted to int16 on the GPU and
        copied asynchronously into pinned host memory. Bookkeeping advances at launch time.
        Returns a handle for decode_finish()."""
        groups = {}
        now = time.perf_counter()
        dt = now - getattr(self, "_t_launch", now)
        self._t_launch = now
        if 0 < dt < 1.0:  # seconds per scheduler iteration
            self._iter_s = dt if not getattr(self, "_iter_s", 0) else 0.8 * self._iter_s + 0.2 * dt
        step = getattr(self, "_iter_s", 0.0)
        margin = self.DEADLINE_MARGIN_MS / 1000
        for s, it in enumerate(self.info):
            if it is None or it.get("flushed"):
                continue
            pend = it["valid"] - it["dec"]
            slack = None  # seconds until the client's buffer runs dry
            if self.DEADLINE and it.get("t0") is not None:
                slack = it["dec"] * self.sdec.spf / 24000 - it.get("trim_s", 0.0) - (now - it["t0"])
                if slack < -0.25:  # far behind real time: the gap is unavoidable, keep decode calls large
                    slack = None
            if pend <= 0:
                if it["done"]:
                    groups.setdefault((0, False), []).append((s, 0))
                continue
            target = self.RAMP[min(it["ci"], len(self.RAMP) - 1)]
            # the first EARLY_CHUNKS chunks of a stream are never held back by the caller: they build the
            # client's playback buffer, and a late second chunk is an audible gap
            first = split_first and (it["ci"] < self.EARLY_CHUNKS or (slack is not None and slack < step + margin))
            if slack is not None and 2 <= pend < target and slack < 2 * step + margin:
                target = max(f for f in self.RAMP if f <= pend)  # cannot wait for a full chunk
            if pend >= target:
                groups.setdefault((target, first), []).append((s, target))
            elif it["done"]:
                F = next(f for f in self.RAMP if f >= pend)
                groups.setdefault((F, first), []).append((s, pend))
        firsts, calls = [], []
        for (F, first), items in groups.items():
            dst = firsts if first else calls
            if F == 0:
                for s, _ in items:
                    self.info[s]["flushed"] = True
                dst.append((None, None, [(s, 0, True, None) for s, _ in items], 0))
                continue
            g = self._dec_graph(F)
            mb = self._dec_max_b(F)
            for j in range(0, len(items), mb):
                dst.append(self._decode_group(g, F, items[j:j + mb]))
        return (firsts, calls) if split_first else calls

    def decode_finish(self, calls):
        """Wait for launched decodes -> [(slot, int16 pcm bytes or None, finished)]."""
        out = []
        up = self.sdec.spf
        for host, ev, items, F in calls:
            if ev is not None:
                ev.synchronize()
                arr = host.numpy()
            for i, (s, n, fin, start) in enumerate(items):
                it = self.info[s]
                if it is None or (start is not None and it.get("emitted", 0) != start):
                    self.out_of_order = getattr(self, "out_of_order", 0) + 1  # must stay 0: audio would be scrambled
                elif start is not None:
                    it["emitted"] = start + n
                if it is not None and n and it["dec"] > 0 and it.get("t0") is None:
                    it["t0"] = time.perf_counter()  # the client's playback clock starts here
                b = arr[i, :n * up].tobytes() if (ev is not None and n) else None
                out.append((s, b, fin))
        return out

    def _decode_group(self, g, F, items):
        self.dec_calls = getattr(self, "dec_calls", 0) + 1
        slots = [s for s, _ in items]
        with torch.cuda.stream(self.dec_stream):
            self.dec_stream.wait_stream(torch.cuda.current_stream())
            rows = kernels.h2d(slots, self.dev)
            starts = kernels.h2d([self.info[s]["dec"] for s in slots], self.dev)
            idx = (starts[:, None] + torch.arange(F, device=self.dev)[None, :]).clamp(max=self.L - 1)
            codes = self.codes[rows[:, None], idx]
            pos = kernels.h2d([self.info[s]["dec_pos"] for s in slots], self.dev, torch.int32)
            wav = g(codes.contiguous(), rows.to(torch.int32), pos)
            pcm = (wav.clamp(-1.0, 1.0) * 32767.0).to(torch.int16)
            host = torch.empty(pcm.shape, dtype=torch.int16, pin_memory=True)
            host.copy_(pcm, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.dec_stream)
        res = []
        for s, n in items:
            it = self.info[s]
            start = it["dec"]
            it["dec"] += n
            it["dec_pos"] += F
            it["ci"] += 1
            fin = it["done"] and it["dec"] >= it["valid"]
            if fin:
                it["flushed"] = True
            res.append((s, n, fin, start))
        return (host, ev, res, F)

    def decode(self, slots, chunk, valid):
        """Batched streaming decode of the K new frames of each slot -> list of float32 numpy arrays."""
        from uqt.streaming_codec import StreamingDecodeGraphs

        n, K, _ = chunk.shape
        g = self.dec_graphs.get(K)
        if g is None:
            sizes = [b for b in B_BUCKETS if b <= self.S]
            g = StreamingDecodeGraphs(self.sdec, sizes, frames=K, pool=self._dec_pool)
            self._dec_pool = g.pool
            self.dec_graphs[K] = g
        with torch.cuda.stream(self.dec_stream):
            chunk.record_stream(self.dec_stream)
            sl = torch.tensor(slots, dtype=torch.int32, device=self.dev)
            pos = torch.tensor([self.info[s]["dec_pos"] for s in slots], dtype=torch.int32, device=self.dev)
            wav = g(chunk.contiguous(), sl, pos).cpu().numpy()
        out = []
        up = self.sdec.spf
        for i, s in enumerate(slots):
            self.info[s]["dec_pos"] += K
            out.append(wav[i, :valid[i] * up])
        return out

    def warmup(self, Ks=(1, 2), Bs=None, Lbs=None):
        Bs = Bs or [b for b in B_BUCKETS if b <= self.S]
        Lbs = Lbs or [b for b in self.LB if b <= self.L]
        t = time.perf_counter()
        for K in Ks:
            for B in Bs:
                for Lb in Lbs:
                    self._graph(K, B, Lb)
        self.warm_pack()
        for F in self.RAMP:
            self._dec_graph(F)
        for K in Ks:
            self.info[0] = dict(dec_pos=0)
            self.decode([0], torch.zeros(1, K, 16, dtype=torch.long, device=self.dev), [0])
            self.info[0] = None
        torch.cuda.synchronize()
        return time.perf_counter() - t
