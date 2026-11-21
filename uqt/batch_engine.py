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

import torch
import torch.nn.functional as F

from uqt import kernels
from uqt.engine import _rms, _rot_half


def _pow2_at_least(n, lo=1):
    b = lo
    while b < n:
        b *= 2
    return b


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
        self.kc = torch.zeros(nl, max_slots, T.nkv, max_len, T.hd, device=dev, dtype=dt)
        self.vc = torch.zeros_like(self.kc)
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
        self.fcp_u = torch.empty(max_slots, 15, cp.config.vocab_size, device=dev, dtype=torch.float32)
        self.t_u = torch.empty(max_slots, V, device=dev, dtype=torch.float32)
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
            q, k, v = F.linear(h, p["wqkv"]).split([Tk.nq, Tk.nk, Tk.nk], dim=-1)
            q = _rms(q.view(B, nh, hd), p["qn"], Tk.eps)
            k = _rms(k.view(B, nkv, hd), p["kn"], Tk.eps)
            v = v.view(B, nkv, hd)
            q = q * cos + _rot_half(q) * sin
            k = k * cos + _rot_half(k) * sin
            kernels.kv_write_rows(self.kc, self.vc, k, v, pos, i)
            Kc = self.kc[i, :B, :, :Lb]
            Vc = self.vc[i, :B, :, :Lb]
            sc = torch.matmul(q.view(B, nkv, g, hd), Kc.transpose(-1, -2)) * (hd ** -0.5) + mask
            pr = torch.softmax(sc, dim=-1, dtype=torch.float32).to(self.dt)
            o = torch.matmul(pr, Vc).reshape(B, nh * hd)
            x = x + F.linear(o, p["wo"])
            h = _rms(x, p["ln2"], Tk.eps)
            ga, up = F.linear(h, p["wgu"]).chunk(2, dim=-1)
            x = x + F.linear(F.silu(ga) * up, p["wd"])
        h = _rms(x, Tk.norm, Tk.eps)
        lg = F.linear(h, E.head).float()
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
        self.kc[:, slot, :, :T].copy_(E.kc[:, :, :T])
        self.vc[:, slot, :, :T].copy_(E.vc[:, :, :T])
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
        self.info[slot] = dict(pos=T, frames=0, dec_pos=R, max_frames=max_frames, done=False, meta=meta)

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
        B = _pow2_at_least(max(act) + 1)
        need = max(self.info[s]["pos"] for s in act) + K + 1
        Lb = next((b for b in self.LB if b >= need and b <= self.L), self.L)
        self._graph(K, B, Lb).replay()
        rows = torch.tensor(act, device=self.dev)
        starts = torch.tensor([self.info[s]["frames"] for s in act], device=self.dev)
        idx = (starts[:, None] + torch.arange(K, device=self.dev)[None, :]).clamp(max=self.L - 1)
        chunk = self.codes[rows[:, None], idx]
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
            valid.append(k)
        return act, chunk, valid

    def decode(self, slots, chunk, valid):
        """Batched streaming decode of the K new frames of each slot -> list of float32 numpy arrays."""
        from uqt.streaming_codec import StreamingDecodeGraphs

        n, K, _ = chunk.shape
        g = self.dec_graphs.get(K)
        if g is None:
            sizes = [b for b in (1, 2, 4, 8, 16, 32, 64, 128, 256) if b <= self.S]
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
        Bs = Bs or [b for b in (1, 2, 4, 8, 16, 32, 64, 128, 256) if b <= self.S]
        Lbs = Lbs or [b for b in self.LB if b <= self.L]
        t = time.perf_counter()
        for K in Ks:
            for B in Bs:
                for Lb in Lbs:
                    self._graph(K, B, Lb)
        for K in Ks:
            self.info[0] = dict(dec_pos=0)
            self.decode([0], torch.zeros(1, K, 16, dtype=torch.long, device=self.dev), [0])
            self.info[0] = None
        torch.cuda.synchronize()
        return time.perf_counter() - t
