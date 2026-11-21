"""Qwen3-TTS-12Hz decode engine.

Reuses the official `qwen_tts` package for weights, text tokenization, prompt construction,
speaker/codec encoders and the codec decoder. Replaces the per-frame generation loop with:

* a static, preallocated KV cache for the Talker (28 layers) and the code predictor (5 layers);
* all sampling on the GPU (top-k + exponential race, same distribution as HF multinomial);
* one CUDA graph per K frames (Talker step + 15 code-predictor steps + embedding sum + logits
  processing), with no host synchronisation inside a chunk;
* Talker attention bucketed by KV length so a graph only reads the live part of the cache.

Logits processing matches the reference `generate()` defaults: repetition penalty 1.05 on
previously generated code0 tokens, suppression of the 1024 special codec ids except EOS,
`min_new_tokens=2`, temperature 0.9, top-k 50 (Talker and code predictor).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


class _Captured(Exception):
    pass


def _rms(x, w, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def _rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _rope_tables(theta, head_dim, n, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64, device=device).float() / head_dim))
    pos = torch.arange(n, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _attn(q, k, v, mask, n_kv):
    """q [T, H, D]; k, v [n_kv, S, D]; mask additive [T, S] or None -> [T, H*D].

    Mirrors HF eager attention numerics: bf16 QK^T, fp32 softmax, bf16 PV.
    """
    T, H, D = q.shape
    g = H // n_kv
    qg = q.transpose(0, 1).reshape(n_kv, g * T, D)  # head h -> kv group h // g
    s = torch.matmul(qg, k.transpose(1, 2)) * (D ** -0.5)  # [n_kv, g*T, S]
    if mask is not None:
        s = s + mask.repeat(g, 1)  # rows ordered (g, T)
    p = torch.softmax(s, dim=-1, dtype=torch.float32).to(q.dtype)
    o = torch.matmul(p, v)  # [n_kv, g*T, D]
    return o.reshape(H, T, D).transpose(0, 1).reshape(T, H * D)


class _Stack:
    """A Qwen3-style decoder stack (q/k RMSNorm, GQA, SwiGLU) over an external KV cache."""

    def __init__(self, layers, norm, cfg):
        self.nh, self.nkv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.eps = cfg.rms_norm_eps
        self.nq, self.nk = self.nh * self.hd, self.nkv * self.hd
        self.L = []
        for l in layers:
            a, m = l.self_attn, l.mlp
            self.L.append(dict(
                ln1=l.input_layernorm.weight, ln2=l.post_attention_layernorm.weight,
                wqkv=torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], 0).contiguous(),
                wo=a.o_proj.weight, qn=a.q_norm.weight, kn=a.k_norm.weight,
                wgu=torch.cat([m.gate_proj.weight, m.up_proj.weight], 0).contiguous(), wd=m.down_proj.weight,
            ))
        self.norm = norm.weight

    def forward(self, x, cos, sin, kc, vc, write, S, mask):
        """x [T, D]. kc/vc [layers, n_kv, S_max, hd]. write(kc_l, vc_l, k, v) stores new K/V."""
        T = x.shape[0]
        for i, p in enumerate(self.L):
            h = _rms(x, p["ln1"], self.eps)
            q, k, v = F.linear(h, p["wqkv"]).split([self.nq, self.nk, self.nk], dim=-1)
            q = _rms(q.view(T, self.nh, self.hd), p["qn"], self.eps)
            k = _rms(k.view(T, self.nkv, self.hd), p["kn"], self.eps)
            v = v.view(T, self.nkv, self.hd)
            q = q * cos[:, None] + _rot_half(q) * sin[:, None]
            k = k * cos[:, None] + _rot_half(k) * sin[:, None]
            write(kc[i], vc[i], k, v)
            o = _attn(q, kc[i, :, :S], vc[i, :, :S], mask, self.nkv)
            x = x + F.linear(o, p["wo"])
            h = _rms(x, p["ln2"], self.eps)
            g, u = F.linear(h, p["wgu"]).chunk(2, dim=-1)
            x = x + F.linear(F.silu(g) * u, p["wd"])
        return _rms(x, self.norm, self.eps)


@dataclass
class SamplingCfg:
    greedy: bool = False
    temperature: float = 0.9
    top_k: int = 50
    cp_greedy: bool = False
    cp_temperature: float = 0.9
    cp_top_k: int = 50
    repetition_penalty: float = 1.05
    min_new_tokens: int = 2


@dataclass
class Prompt:
    embeds: torch.Tensor       # [T, D] Talker prefill embeddings
    trailing: torch.Tensor     # [Tt, D] text embeddings fed one per frame
    pad: torch.Tensor          # [D] tts_pad embedding (fed after trailing runs out)
    ref_codes: torch.Tensor | None = None  # [R, 16] voice-clone reference codes (decoder left context)


class Engine:
    KV_BUCKETS = (256, 512, 768, 1024, 1536, 2048, 3072, 4096)

    def __init__(self, tts, max_len: int = 2048, sampling: SamplingCfg | None = None):
        self.tts = tts
        m = tts.model
        self.m = m
        t = m.talker
        tc = m.config.talker_config
        cp = t.code_predictor
        cpc = tc.code_predictor_config
        p0 = next(t.parameters())
        self.dev, self.dtype = p0.device, p0.dtype
        self.s = sampling or SamplingCfg()
        self.max_len = max_len
        self.n_groups = tc.num_code_groups
        assert self.n_groups == 16

        self.talker = _Stack(t.model.layers, t.model.norm, tc)
        self.cp = _Stack(cp.model.layers, cp.model.norm, cpc)
        self.t_emb = t.get_input_embeddings().weight
        self.head = t.codec_head.weight
        self.cp_emb = [e.weight for e in cp.model.codec_embedding]
        self.cp_head = [h.weight for h in cp.lm_head]
        proj = cp.small_to_mtp_projection
        self.proj_w = getattr(proj, "weight", None)
        self.proj_b = getattr(proj, "bias", None)

        dev, dt = self.dev, self.dtype
        V = tc.vocab_size
        self.V = V
        self.eos = tc.codec_eos_token_id
        sup = torch.zeros(V, dtype=torch.bool, device=dev)
        sup[V - 1024:] = True
        sup[self.eos] = False
        self.suppress = sup
        self.eos_mask = torch.zeros(V, dtype=torch.bool, device=dev)
        self.eos_mask[self.eos] = True

        self.t_cos, self.t_sin = _rope_tables(tc.rope_theta, tc.head_dim, max_len, dev, dt)
        self.c_cos, self.c_sin = _rope_tables(cpc.rope_theta, cpc.head_dim, self.n_groups + 1, dev, dt)
        T, C = self.talker, self.cp
        self.kc = torch.zeros(len(T.L), T.nkv, max_len, T.hd, device=dev, dtype=dt)
        self.vc = torch.zeros_like(self.kc)
        self.ckc = torch.zeros(len(C.L), C.nkv, self.n_groups + 1, C.hd, device=dev, dtype=dt)
        self.cvc = torch.zeros_like(self.ckc)
        self.arange = torch.arange(max_len, device=dev)

        D = tc.hidden_size
        self.code0 = torch.zeros(1, dtype=torch.long, device=dev)
        self.hidden = torch.zeros(1, D, device=dev, dtype=dt)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.step = torch.zeros(1, dtype=torch.long, device=dev)
        self.fidx = torch.zeros(1, dtype=torch.long, device=dev)
        self.seen = torch.zeros(V, dtype=torch.bool, device=dev)
        self.trail = torch.zeros(max_len, D, device=dev, dtype=dt)
        self.codes = torch.zeros(max_len, self.n_groups, dtype=torch.long, device=dev)
        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.pos_host = 0
        self.frames_host = 0
        self._pool = None

    # ------------------------------------------------------------------ prompt

    @torch.inference_mode()
    def build_prompt(self, kind: str, **kw) -> Prompt:
        """Run the official prompt builder and capture what it would pass to talker.generate."""
        talker = self.m.talker
        captured = {}

        def fake_generate(**gkw):
            captured.update(gkw)
            raise _Captured()

        orig = talker.generate
        talker.generate = fake_generate
        try:
            if kind == "clone":
                self.tts.generate_voice_clone(**kw)
            elif kind == "custom":
                self.tts.generate_custom_voice(**kw)
            elif kind == "design":
                self.tts.generate_voice_design(**kw)
            else:
                raise ValueError(kind)
        except _Captured:
            pass
        finally:
            talker.generate = orig
        emb = captured["inputs_embeds"]
        assert emb.shape[0] == 1, "batch 1 only"
        ref = None
        vcp = kw.get("voice_clone_prompt")
        if kind == "clone" and vcp is not None:
            items = vcp if isinstance(vcp, list) else None
            if items and items[0].ref_code is not None:
                ref = items[0].ref_code.to(self.dev)
        return Prompt(
            embeds=emb[0].to(self.dtype),
            trailing=captured["trailing_text_hidden"][0].to(self.dtype),
            pad=captured["tts_pad_embed"].reshape(-1).to(self.dtype),
            ref_codes=ref,
        )

    # ------------------------------------------------------------------ sampling

    def _sample(self, logits, temperature, top_k, greedy):
        """logits fp32 [V] -> [1] long. Top-k + exponential race == softmax-multinomial over top-k."""
        if greedy:
            return logits.argmax(-1, keepdim=True)
        v, idx = torch.topk(logits / temperature, top_k)
        p = torch.softmax(v, dim=-1)
        e = torch.empty_like(p).exponential_()
        return idx.gather(-1, (p / e).argmax(-1, keepdim=True))

    def _talker_logits(self, h, ban_eos):
        s = F.linear(h, self.head)[0].float()
        rp = self.s.repetition_penalty
        if rp != 1.0:
            s = torch.where(self.seen, torch.where(s < 0, s * rp, s / rp), s)
        s = s.masked_fill(self.suppress, float("-inf"))
        s = s.masked_fill(self.eos_mask & ban_eos, float("-inf"))
        return self._sample(s, self.s.temperature, self.s.top_k, self.s.greedy)

    # ------------------------------------------------------------------ core

    def _write_static(self, start):
        def w(kc, vc, k, v):
            T = k.shape[0]
            kc[:, start:start + T].copy_(k.transpose(0, 1))
            vc[:, start:start + T].copy_(v.transpose(0, 1))
        return w

    def _write_dyn(self, kc, vc, k, v):
        kc.index_copy_(1, self.pos, k.transpose(0, 1))
        vc.index_copy_(1, self.pos, v.transpose(0, 1))

    def _causal(self, T):
        return torch.full((T, T), float("-inf"), device=self.dev, dtype=self.dtype).triu(1)

    def _proj(self, x):
        return x if self.proj_w is None else F.linear(x, self.proj_w, self.proj_b)

    def _frame(self, Lb):
        """One codec frame: code predictor for the current code0, then one Talker step."""
        s = self.s
        C = self.cp
        c0 = self.code0
        e0 = F.embedding(c0, self.t_emb)
        x = self._proj(torch.cat([self.hidden, e0], 0))
        h = C.forward(x, self.c_cos[:2], self.c_sin[:2], self.ckc, self.cvc,
                      self._write_static(0), 2, self._causal(2))
        codes, embs = [c0], [e0]
        c = self._sample(F.linear(h[-1:], self.cp_head[0])[0].float(), s.cp_temperature, s.cp_top_k, s.cp_greedy)
        for j in range(1, self.n_groups - 1):
            codes.append(c)
            e = F.embedding(c, self.cp_emb[j - 1])
            embs.append(e)
            h = C.forward(self._proj(e), self.c_cos[j + 1:j + 2], self.c_sin[j + 1:j + 2], self.ckc, self.cvc,
                          self._write_static(j + 1), j + 2, None)
            c = self._sample(F.linear(h, self.cp_head[j])[0].float(), s.cp_temperature, s.cp_top_k, s.cp_greedy)
        codes.append(c)
        embs.append(F.embedding(c, self.cp_emb[self.n_groups - 2]))
        self.codes.index_copy_(0, self.fidx, torch.cat(codes)[None])

        inp = torch.cat(embs, 0).sum(0, keepdim=True) + self.trail.index_select(0, self.step)
        cos = self.t_cos.index_select(0, self.pos)
        sin = self.t_sin.index_select(0, self.pos)
        mask = torch.zeros(1, Lb, device=self.dev, dtype=self.dtype).masked_fill(
            self.arange[:Lb] > self.pos, float("-inf"))
        h = self.talker.forward(inp, cos, sin, self.kc, self.vc, self._write_dyn, Lb, mask)
        nc = self._talker_logits(h, self.fidx < (s.min_new_tokens - 1))
        self.seen.index_fill_(0, nc, True)
        self.hidden.copy_(h)
        self.code0.copy_(nc)
        self.pos.add_(1)
        self.step.add_(1)
        self.fidx.add_(1)

    def _bucket(self, need):
        for b in self.KV_BUCKETS:
            if b >= need and b <= self.max_len:
                return b
        if need <= self.max_len:
            return self.max_len
        raise RuntimeError(f"sequence too long: {need} > {self.max_len}")

    @torch.inference_mode()
    def prefill(self, prompt: Prompt):
        T = prompt.embeds.shape[0]
        Tt = prompt.trailing.shape[0]
        assert T + 1 < self.max_len and Tt <= self.max_len
        self.trail.copy_(prompt.pad.expand(self.max_len, -1))
        self.trail[:Tt].copy_(prompt.trailing)
        h = self.talker.forward(prompt.embeds, self.t_cos[:T], self.t_sin[:T], self.kc, self.vc,
                                self._write_static(0), T, self._causal(T))
        self.seen.zero_()
        ban = torch.ones(1, dtype=torch.bool, device=self.dev) if self.s.min_new_tokens > 0 else \
            torch.zeros(1, dtype=torch.bool, device=self.dev)
        c = self._talker_logits(h[-1:], ban)
        self.seen.index_fill_(0, c, True)
        self.hidden.copy_(h[-1:])
        self.code0.copy_(c)
        self.pos.fill_(T)
        self.step.zero_()
        self.fidx.zero_()
        self.pos_host = T
        self.frames_host = 0

    def _run_frames(self, K, Lb):
        for _ in range(K):
            self._frame(Lb)

    def _graph(self, K, Lb):
        key = (K, Lb)
        g = self.graphs.get(key)
        if g is None:
            # Warm up on a side stream (required before capture), then restore state.
            snap = [t.clone() for t in (self.code0, self.hidden, self.pos, self.step, self.fidx, self.seen)]
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._run_frames(K, Lb)
            torch.cuda.current_stream().wait_stream(s)
            for dst, src in zip((self.code0, self.hidden, self.pos, self.step, self.fidx, self.seen), snap):
                dst.copy_(src)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._pool):
                self._run_frames(K, Lb)
            self._pool = g.pool()
            for dst, src in zip((self.code0, self.hidden, self.pos, self.step, self.fidx, self.seen), snap):
                dst.copy_(src)
            self.graphs[key] = g
        return g

    @torch.inference_mode()
    def step_frames(self, K: int, use_graph: bool = True):
        """Generate K frames; returns the [K, 16] code tensor view (still on GPU, not synced)."""
        Lb = self._bucket(self.pos_host + K)
        if use_graph:
            self._graph(K, Lb).replay()
        else:
            self._run_frames(K, Lb)
        out = self.codes[self.frames_host:self.frames_host + K]
        self.pos_host += K
        self.frames_host += K
        return out

    @torch.inference_mode()
    def warmup(self, prompt: Prompt, Ks=(1, 2, 4, 8), max_bucket: int = 1024):
        for Lb in [b for b in self.KV_BUCKETS if b <= min(max_bucket, self.max_len)]:
            for K in Ks:
                self.prefill(prompt)
                self.pos_host = max(self.pos_host, 0)
                self._graph(K, Lb)
        torch.cuda.synchronize()

    # ------------------------------------------------------------------ generation

    @torch.inference_mode()
    def generate_codes(self, prompt: Prompt, max_frames: int = 2048, chunk: int = 8, use_graph=True):
        """Non-streaming: returns [N, 16] codes on GPU (EOS frame excluded)."""
        self.prefill(prompt)
        out = []
        n = 0
        while n < max_frames:
            K = min(chunk, max_frames - n)
            if self.pos_host + K >= self.max_len:
                break
            c = self.step_frames(K, use_graph)
            c0 = c[:, 0].tolist()
            if self.eos in c0:
                k = c0.index(self.eos)
                out.append(c[:k].clone())
                break
            out.append(c.clone())
            n += K
        return torch.cat(out, 0) if out else torch.zeros(0, self.n_groups, dtype=torch.long, device=self.dev)

    # ------------------------------------------------------------------ decoding

    @property
    def decoder(self):
        return self.m.speech_tokenizer.model.decoder

    @torch.inference_mode()
    def decode_window(self, codes_all: torch.Tensor, s: int, e: int, left: int = 25) -> torch.Tensor:
        """Decode frames [s, e) of codes_all ([N, 16], context already included) with `left` frames of context."""
        dec = self.decoder
        a = max(0, s - left)
        wav = dec(codes_all[a:e].clamp(min=0).T[None])  # [1, 1, (e-a)*1920]
        up = int(dec.total_upsample)
        return wav.reshape(-1)[(s - a) * up:(e - a) * up]

    @torch.inference_mode()
    def stream(self, prompt: Prompt, first_chunk: int = 2, chunk: int = 8, left: int = 25,
               max_frames: int = 2048, use_graph: bool = True):
        """Yields float32 numpy audio chunks (24 kHz). Decoder sees ref codes as left context."""
        self.prefill(prompt)
        ctx = prompt.ref_codes if prompt.ref_codes is not None else torch.zeros(0, 16, dtype=torch.long, device=self.dev)
        R = ctx.shape[0]
        hist = [ctx]
        n = 0
        K = first_chunk
        while n < max_frames and self.pos_host + K < self.max_len:
            c = self.step_frames(K, use_graph)
            c0 = c[:, 0].tolist()  # host sync point (once per chunk)
            stop = self.eos in c0
            k = c0.index(self.eos) if stop else K
            if k > 0:
                hist.append(c[:k].clone())
                allc = torch.cat(hist, 0)
                hist = [allc]
                wav = self.decode_window(allc, R + n, R + n + k, left)
                n += k
                yield wav.float().cpu().numpy()
            if stop:
                break
            K = chunk
