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

import os

import torch
import torch.nn.functional as F

from uqt.wq import lin


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
        """x [T, D]. kc/vc [layers, n_kv, S_max, hd]. write(layer, k, v) stores new K/V."""
        T = x.shape[0]
        for i, p in enumerate(self.L):
            h = _rms(x, p["ln1"], self.eps)
            q, k, v = lin(h, p["wqkv"]).split([self.nq, self.nk, self.nk], dim=-1)
            q = _rms(q.view(T, self.nh, self.hd), p["qn"], self.eps)
            k = _rms(k.view(T, self.nkv, self.hd), p["kn"], self.eps)
            v = v.view(T, self.nkv, self.hd)
            q = q * cos[:, None] + _rot_half(q) * sin[:, None]
            k = k * cos[:, None] + _rot_half(k) * sin[:, None]
            write(i, k, v)
            o = _attn(q, kc[i, :, :S], vc[i, :, :S], mask, self.nkv)
            x = x + lin(o, p["wo"])
            h = _rms(x, p["ln2"], self.eps)
            g, u = lin(h, p["wgu"]).chunk(2, dim=-1)
            x = x + lin(F.silu(g) * u, p["wd"])
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

    def __init__(self, tts, max_len: int = 2048, sampling: SamplingCfg | None = None,
                 compile: bool = True, graph_decoder: bool = True, triton_sampler: bool = True,
                 fused_cp: bool = True, streaming_codec: bool = True, autotune: bool = True,
                 fused_talker: bool = False, weights: str = "bf16"):
        self.tts = tts
        m = tts.model
        m.requires_grad_(False)
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
        self.weights = weights
        if weights != "bf16":
            from uqt.wq import QW
            for st in (self.talker, self.cp):
                for p in st.L:
                    for k in ("wqkv", "wo", "wgu", "wd"):
                        p[k] = QW(p[k], weights)
            self.head = QW(self.head, weights)
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
        self.graphs: dict = {}
        self.triton_sampler = triton_sampler
        self._fast = None
        self.pos_host = 0
        self.frames_host = 0
        self._pool = None
        self.inp = torch.zeros(1, D, device=dev, dtype=dt)
        self.pf_emb = torch.zeros(max_len, D, device=dev, dtype=dt)
        self.pf_len = torch.ones(1, dtype=torch.long, device=dev)
        self.ban_first = torch.tensor([self.s.min_new_tokens > 0], device=dev)
        self.full = torch.zeros(4 * max_len, self.n_groups, dtype=torch.long, device=dev)
        self.graph_decoder = graph_decoder
        self.dec_graphs: dict = {}
        self._dec_pool = None
        self.dec_stream = torch.cuda.Stream()
        self.upsample = int(self.m.speech_tokenizer.model.decoder.total_upsample)
        if compile:
            import torch._dynamo as dynamo
            import torch._inductor.config as icfg
            icfg.fallback_random = True  # keep aten RNG so sampling stays graph-capturable
            dynamo.config.cache_size_limit = 128
            self._frame_c = torch.compile(self._frame, fullgraph=True, dynamic=False)
            self._prefill_c = torch.compile(self._prefill_core, fullgraph=True, dynamic=False)
            tmode = "max-autotune-no-cudagraphs" if autotune else None
            self._talker_fused_c = torch.compile(self._talker_fused, fullgraph=True, dynamic=False, mode=tmode)
        else:
            self._frame_c, self._prefill_c = self._frame, self._prefill_core
            self._talker_fused_c = self._talker_fused
        self.ftk = None
        if fused_talker and dt == torch.bfloat16 and compile:
            from uqt.fused_talker import FusedTalker
            self.ftk = FusedTalker(self)
            self._pre_fused_c = torch.compile(self._pre_fused, fullgraph=True, dynamic=False)
            self._post_fused_c = torch.compile(self._post_fused, fullgraph=True, dynamic=False)
        self.sdec = None
        if streaming_codec:
            from uqt.streaming_codec import StreamingCodecDecoder
            self.sdec = StreamingCodecDecoder(self.m.speech_tokenizer.model.decoder, num_slots=1)
            self.sdec_graphs: dict = {}
            self.sdec_pool = None
            self.voice_dec_state: dict = {}
            self.slot_t = torch.zeros(1, dtype=torch.int32, device=dev)
        self.fcp = None
        if fused_cp and dt == torch.bfloat16:
            from uqt.fused_cp import FusedCodePredictor
            self.fcp = FusedCodePredictor(cp, max_batch=1)
            # UQT_CP_BF16=1 keeps the code predictor in bf16 so it can use the PDL-chained path
            if weights != "bf16" and os.environ.get("UQT_CP_BF16", "0") != "1":
                self.fcp.quantize(weights)
            self.fcodes = torch.zeros(1, self.n_groups, dtype=torch.long, device=dev)
            self.fcp_u = torch.empty(1, self.n_groups - 1, cpc.vocab_size, device=dev, dtype=torch.float32)

    # ------------------------------------------------------------------ prompt

    @torch.inference_mode()
    def build_prompt(self, kind: str, fast: bool = True, **kw) -> Prompt:
        """Fast cached builder when the request is a supported case, else the official builder."""
        if fast:
            if self._fast is None:
                from uqt.prompt import FastPromptBuilder
                self._fast = FastPromptBuilder(self)
            extra = set(kw) - {"text", "language", "speaker", "instruct", "voice_clone_prompt"}
            p = None
            if not extra and isinstance(kw.get("text"), str) and isinstance(kw.get("language"), str):
                if kind == "custom":
                    p = self._fast.custom(kw["text"], kw["language"], kw.get("speaker"), kw.get("instruct"))
                elif kind == "clone" and kw.get("voice_clone_prompt") is not None:
                    p = self._fast.clone(kw["text"], kw["language"], kw["voice_clone_prompt"])
            if p is not None:
                return p
        return self.build_prompt_official(kind, **kw)

    @torch.inference_mode()
    def build_prompt_official(self, kind: str, **kw) -> Prompt:
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
        if self.triton_sampler:
            from uqt import kernels
            u = torch.rand(logits.shape[-1], device=logits.device, dtype=torch.float32)
            return kernels.topk_sample(logits, u, 1.0 / temperature, top_k)
        v, idx = torch.topk(logits / temperature, top_k)
        p = torch.softmax(v, dim=-1)
        e = torch.empty_like(p).exponential_()
        return idx.gather(-1, (p / e).argmax(-1, keepdim=True))

    def _talker_logits(self, h, ban_eos):
        return self._logits_sample(lin(h, self.head)[0], ban_eos)

    def _logits_sample(self, logits, ban_eos):
        s = logits.float()
        rp = self.s.repetition_penalty
        if rp != 1.0:
            s = torch.where(self.seen, torch.where(s < 0, s * rp, s / rp), s)
        s = s.masked_fill(self.suppress, float("-inf"))
        s = s.masked_fill(self.eos_mask & ban_eos, float("-inf"))
        return self._sample(s, self.s.temperature, self.s.top_k, self.s.greedy)

    # ------------------------------------------------------------------ core

    @staticmethod
    def _write_static(kc, vc, start):
        from uqt import kernels

        def w(i, k, v):
            kernels.kv_write_static(kc, vc, k, v, i, start)
        return w

    def _write_dyn(self, i, k, v):
        from uqt import kernels

        kernels.kv_write_dyn(self.kc, self.vc, k, v, self.pos, i)

    def _causal(self, T):
        return torch.full((T, T), float("-inf"), device=self.dev, dtype=self.dtype).triu(1)

    def _proj(self, x):
        return x if self.proj_w is None else F.linear(x, self.proj_w, self.proj_b)

    def _frame_cp(self):
        """Code predictor for the current code0 -> codes row + next Talker input embedding."""
        s = self.s
        C = self.cp
        c0 = self.code0
        e0 = F.embedding(c0, self.t_emb)
        x = self._proj(torch.cat([self.hidden, e0], 0))
        h = C.forward(x, self.c_cos[:2], self.c_sin[:2], self.ckc, self.cvc,
                      self._write_static(self.ckc, self.cvc, 0), 2, self._causal(2))
        codes, embs = [c0], [e0]
        c = self._sample(F.linear(h[-1:], self.cp_head[0])[0].float(), s.cp_temperature, s.cp_top_k, s.cp_greedy)
        for j in range(1, self.n_groups - 1):
            codes.append(c)
            e = F.embedding(c, self.cp_emb[j - 1])
            embs.append(e)
            h = C.forward(self._proj(e), self.c_cos[j + 1:j + 2], self.c_sin[j + 1:j + 2], self.ckc, self.cvc,
                          self._write_static(self.ckc, self.cvc, j + 1), j + 2, None)
            c = self._sample(F.linear(h, self.cp_head[j])[0].float(), s.cp_temperature, s.cp_top_k, s.cp_greedy)
        codes.append(c)
        embs.append(F.embedding(c, self.cp_emb[self.n_groups - 2]))
        self.codes.index_copy_(0, self.fidx, torch.cat(codes)[None])
        self.inp.copy_(torch.cat(embs, 0).sum(0, keepdim=True) + self.trail.index_select(0, self.step))

    def _frame(self, Lb):
        """One codec frame: code predictor for the current code0, then one Talker step."""
        self._frame_cp()
        self._frame_talker(Lb)

    def _frame_talker(self, Lb):
        s = self.s
        inp = self.inp
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

    def _bucket(self, need, buckets=None):
        for b in (buckets or self.KV_BUCKETS):
            if need <= b <= self.max_len:
                return b
        if need <= self.max_len:
            return self.max_len
        raise RuntimeError(f"sequence too long: {need} > {self.max_len}")

    # ------------------------------------------------------------------ graphs

    _STATE = ("code0", "hidden", "pos", "step", "fidx", "seen")

    def _capture(self, key, fn, pool_attr="_pool"):
        """Warm up `fn` on a side stream (compiles/autotunes), restore state, capture a CUDA graph."""
        g = self.graphs.get(key)
        if g is not None:
            return g
        snap = [getattr(self, n).clone() for n in self._STATE]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
            fn()
        torch.cuda.current_stream().wait_stream(s)
        for n, v in zip(self._STATE, snap):
            getattr(self, n).copy_(v)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=getattr(self, pool_attr)):
            fn()
        setattr(self, pool_attr, g.pool())
        for n, v in zip(self._STATE, snap):
            getattr(self, n).copy_(v)
        self.graphs[key] = g
        return g

    def _fcp_step(self):
        """Fused Triton code predictor (vLLM-Omni port): writes self.fcodes [1, 16]."""
        s = self.s
        if s.cp_greedy:
            self.fcp_u.fill_(0.5)  # constant Gumbel noise -> argmax
            inv_t, k = 1.0, 0
        else:
            self.fcp_u.uniform_()
            inv_t, k = 1.0 / s.cp_temperature, s.cp_top_k
        e0 = F.embedding(self.code0, self.t_emb)
        self.fcodes.copy_(self.fcp(self.code0, e0, self.hidden, inv_t, k, self.fcp_u))

    def _talker_fused(self, Lb):
        c = self.fcodes[0]
        embs = [F.embedding(c[:1], self.t_emb)] + [F.embedding(c[i:i + 1], self.cp_emb[i - 1])
                                                    for i in range(1, self.n_groups)]
        self.codes.index_copy_(0, self.fidx, self.fcodes)
        self.inp.copy_(torch.stack(embs, 0).sum(0) + self.trail.index_select(0, self.step))
        self._frame_talker(Lb)

    def _pre_fused(self):
        c = self.fcodes[0]
        embs = [F.embedding(c[:1], self.t_emb)] + [F.embedding(c[i:i + 1], self.cp_emb[i - 1])
                                                    for i in range(1, self.n_groups)]
        self.codes.index_copy_(0, self.fidx, self.fcodes)
        self.inp.copy_(torch.stack(embs, 0).sum(0) + self.trail.index_select(0, self.step))

    def _post_fused(self):
        h = self.ftk.final_hidden()
        nc = self._logits_sample(self.ftk.logits[0], self.fidx < (self.s.min_new_tokens - 1))
        self.seen.index_fill_(0, nc, True)
        self.hidden.copy_(h)
        self.code0.copy_(nc)
        self.pos.add_(1)
        self.step.add_(1)
        self.fidx.add_(1)

    def _run_frames(self, K, Lb, compiled=True):
        if self.ftk is not None and self.fcp is not None and compiled:
            for _ in range(K):
                self._fcp_step()
                self._pre_fused_c()
                self.ftk.step(self.inp, Lb)
                self._post_fused_c()
            return
        if self.fcp is not None and compiled:
            for _ in range(K):
                self._fcp_step()
                self._talker_fused_c(Lb)
            return
        f = self._frame_c if compiled else self._frame
        for _ in range(K):
            f(Lb)

    # ------------------------------------------------------------------ prefill

    PF_BUCKETS = (64, 128, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048)

    def _prefill_core(self, Tb):
        """Prefill a zero-padded prompt of bucket length Tb; real length is self.pf_len (device).

        Padded positions write junk K/V at [T, Tb); decode overwrites position p before it is
        ever attended (decode attends to positions <= p only).
        """
        h = self.talker.forward(self.pf_emb[:Tb], self.t_cos[:Tb], self.t_sin[:Tb], self.kc, self.vc,
                                self._write_static(self.kc, self.vc, 0), Tb, self._causal(Tb))
        hl = h.index_select(0, self.pf_len - 1)
        self.seen.zero_()
        c = self._talker_logits(hl, self.ban_first)
        self.seen.index_fill_(0, c, True)
        self.hidden.copy_(hl)
        self.code0.copy_(c)
        self.pos.copy_(self.pf_len)
        self.step.zero_()
        self.fidx.zero_()

    @torch.inference_mode()
    def prefill(self, prompt: Prompt, use_graph: bool = True):
        T = prompt.embeds.shape[0]
        Tt = prompt.trailing.shape[0]
        assert T + 1 < self.max_len and Tt <= self.max_len
        self.trail.copy_(prompt.pad.expand(self.max_len, -1))
        self.trail[:Tt].copy_(prompt.trailing)
        Tb = self._bucket(T, self.PF_BUCKETS) if use_graph else T
        self.pf_emb[:T].copy_(prompt.embeds)
        self.pf_emb[T:Tb].zero_()
        self.pf_len.fill_(T)
        if use_graph:
            self._capture(("prefill", Tb), lambda: self._prefill_c(Tb)).replay()
        else:
            self._prefill_core(Tb)
        self.pos_host = T
        self.frames_host = 0

    # ------------------------------------------------------------------ frames

    @torch.inference_mode()
    def step_frames(self, K: int, use_graph: bool = True):
        """Generate K frames; returns the [K, 16] code tensor view (still on GPU, not synced)."""
        Lb = self._bucket(self.pos_host + K)
        if use_graph:
            self._capture(("frames", K, Lb), lambda: self._run_frames(K, Lb)).replay()
        else:
            self._run_frames(K, Lb, compiled=False)
        out = self.codes[self.frames_host:self.frames_host + K]
        self.pos_host += K
        self.frames_host += K
        return out

    @torch.inference_mode()
    def warmup(self, prompt: Prompt, Ks=(1, 2, 4, 8), max_bucket: int = 1024, prefill_max: int = 512,
               dec_windows=()):
        for Tb in [b for b in self.PF_BUCKETS if b <= prefill_max]:
            self._capture(("prefill", Tb), lambda Tb=Tb: self._prefill_c(Tb))
        for Lb in [b for b in self.KV_BUCKETS if b <= min(max_bucket, self.max_len)]:
            for K in Ks:
                self.prefill(prompt)
                self._capture(("frames", K, Lb), lambda K=K, Lb=Lb: self._run_frames(K, Lb))
        for n in dec_windows:
            self._dec(torch.zeros(n, self.n_groups, dtype=torch.long, device=self.dev))
        if self.sdec is not None:
            with torch.cuda.stream(self.dec_stream):
                for k in range(1, 9):
                    self._sdec_call(torch.zeros(k, self.n_groups, dtype=torch.long, device=self.dev), 0)
            torch.cuda.synchronize()
        torch.cuda.synchronize()

    # ------------------------------------------------------------------ generation

    @torch.inference_mode()
    def generate_codes(self, prompt: Prompt, max_frames: int = 2048, chunk: int = 8, use_graph=True):
        """Non-streaming: returns [N, 16] codes on GPU (EOS frame excluded)."""
        self.prefill(prompt, use_graph)
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

    def _dec(self, win: torch.Tensor) -> torch.Tensor:
        """Run the codec decoder on a [n, 16] code window -> [n * 1920] waveform (graphed per n)."""
        dec = self.decoder
        n = win.shape[0]
        if not self.graph_decoder:
            return dec(win.clamp(min=0).T[None]).reshape(-1)
        ent = self.dec_graphs.get(n)
        if ent is None:
            inp = torch.zeros(1, self.n_groups, n, dtype=torch.long, device=self.dev)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                dec(inp)
                dec(inp)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(g, pool=self._dec_pool):
                    out = dec(inp)
            except Exception as e:  # decoder not capturable: fall back to eager for good
                print(f"[uqt] decoder graph capture failed ({e!r}); using eager decoder")
                self.graph_decoder = False
                torch.cuda.synchronize()
                return dec(win.clamp(min=0).T[None]).reshape(-1)
            self._dec_pool = g.pool()
            ent = (inp, out, g)
            self.dec_graphs[n] = ent
        inp, out, g = ent
        inp.copy_(win.clamp(min=0).T[None])
        g.replay()
        return out.reshape(-1)

    # ---- stateful streaming codec (vLLM-Omni port): frame-exact, no left-context recompute

    def _sdec_call(self, codes_k: torch.Tensor, pos: int) -> torch.Tensor:
        """Decode k new frames [k, 16] of slot 0 starting at stream position `pos` -> [k * 1920] fp32."""
        from uqt.streaming_codec import StreamingDecodeGraphs

        k = codes_k.shape[0]
        g = self.sdec_graphs.get(k)
        if g is None:
            g = StreamingDecodeGraphs(self.sdec, [1], frames=k, pool=self.sdec_pool)
            self.sdec_pool = g.pool
            self.sdec_graphs[k] = g
        pos_t = torch.full((1,), pos, dtype=torch.int32, device=self.dev)
        return g(codes_k[None].contiguous(), self.slot_t, pos_t)[0]

    def _sdec_prime(self, ref: torch.Tensor | None) -> int:
        """Run the reference codes through the decoder state (cached per reference tensor)."""
        if ref is None or ref.shape[0] == 0:
            return 0
        R = int(ref.shape[0])
        ent = self.voice_dec_state.get(id(ref))
        state = self.sdec._slot_state(0)
        if ent is not None and ent[0] is ref:
            for t, v in zip(state, ent[1]):
                t.copy_(v)
            return R
        codes = ref.to(torch.int32)[None]
        first = min(R, self.sdec.RING)
        self.sdec(codes[:, :first].contiguous(), self.slot_t, torch.zeros(1, dtype=torch.int32, device=self.dev),
                  initial=True)
        p = first
        while p < R:
            q = min(R, p + self.sdec.max_frames)
            self.sdec(codes[:, p:q].contiguous(), self.slot_t,
                      torch.full((1,), p, dtype=torch.int32, device=self.dev))
            p = q
        self.voice_dec_state[id(ref)] = (ref, [t.clone() for t in state])
        return R

    @torch.inference_mode()
    def decode_window(self, codes_all: torch.Tensor, s: int, e: int, left: int = 25) -> torch.Tensor:
        """Decode frames [s, e) of codes_all ([N, 16], context included) with `left` frames of context."""
        a = max(0, s - left)
        up = self.upsample
        return self._dec(codes_all[a:e])[(s - a) * up:(e - a) * up]

    @torch.inference_mode()
    def stream(self, prompt: Prompt, schedule=(1, 2, 4, 8), left: int = 25, max_frames: int = 2048,
               use_graph: bool = True, overlap: bool = True, first_chunk=None, chunk=None):
        """Yields float32 numpy audio chunks (24 kHz).

        Chunk sizes follow `schedule` (last entry repeats). With `overlap`, frame generation for
        chunk i+1 is launched before chunk i is decoded, and decoding runs on a separate stream.
        """
        if first_chunk is not None or chunk is not None:  # legacy args
            schedule = (first_chunk or 2, chunk or 8)
        self.prefill(prompt, use_graph)
        ctx = prompt.ref_codes
        R = 0 if ctx is None else int(ctx.shape[0])
        if R:
            self.full[:R].copy_(ctx)
        gen_s = torch.cuda.current_stream()
        dec_s = self.dec_stream if overlap else gen_s
        base = 0
        if self.sdec is not None:
            dec_s.wait_stream(gen_s)
            with torch.cuda.stream(dec_s):
                base = self._sdec_prime(ctx)  # overlaps with the first frames on the generation stream
        n, i = 0, 0
        c = self.step_frames(schedule[0], use_graph)
        while True:
            c0 = c[:, 0].tolist()  # waits for this chunk's frames only
            K = len(c0)
            stop = self.eos in c0
            k = c0.index(self.eos) if stop else K
            nxt = None
            if not stop and n + k < max_frames:
                K2 = schedule[min(i + 1, len(schedule) - 1)]
                K2 = min(K2, max_frames - n - k)
                if self.pos_host + K2 < self.max_len:
                    nxt = self.step_frames(K2, use_graph)  # async; overlaps with decode below
            if k > 0:
                # chunk i is complete (host synced above), so the decode stream needs no wait.
                with torch.cuda.stream(dec_s):
                    if self.sdec is not None:
                        wav = self._sdec_call(c[:k], base + n).cpu().numpy()
                    else:
                        self.full[R + n:R + n + k].copy_(c[:k])
                        wav = self.decode_window(self.full, R + n, R + n + k, left).float().cpu().numpy()
                n += k
                yield wav
            if nxt is None:
                break
            c = nxt
            i += 1
