"""Fast Talker prompt construction (batch 1).

Reproduces `Qwen3TTSForConditionalGeneration.generate()` prompt building for the two common cases
and caches everything that does not depend on the request text:

* CustomVoice (`non_streaming_mode=True`, the wrapper default): role tokens + codec prefix
  (think/language/speaker/pad/bos) are cached per (language, speaker); per request only the text
  token embeddings go through `text_projection`.
* Voice clone, ICL mode (`non_streaming_mode=False`, the wrapper default): the codec prefix with the
  reference x-vector, the reference-code embedding sum and the projected reference text are cached
  per voice-clone prompt item.

Anything else (instructions, language "auto", dialect speakers, x-vector-only clones) returns None
so the caller falls back to the official builder.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class FastPromptBuilder:
    def __init__(self, engine):
        self.e = engine
        tts = engine.tts
        m = engine.m
        self.t = m.talker
        self.cfg = m.config
        self.tc = m.config.talker_config
        self.tok = getattr(tts.processor, "tokenizer", tts.processor)
        self.dev, self.dt = engine.dev, engine.dtype
        self.cache: dict = {}
        tc = self.tc
        with torch.inference_mode():
            bos_eos_pad = self._proj_ids([self.cfg.tts_bos_token_id, self.cfg.tts_eos_token_id, self.cfg.tts_pad_token_id])
            self.tts_bos, self.tts_eos, self.tts_pad = bos_eos_pad[0:1], bos_eos_pad[1:2], bos_eos_pad[2:3]
            self.codec_pad = self._codec([tc.codec_pad_id])
            self.codec_bos = self._codec([tc.codec_bos_id])
            role_ids = self._ids(self._assistant("x"))[:3]
            self.role = self._proj_ids(role_ids)
            self.last_custom = self.tts_pad + self.codec_bos

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _assistant(text):
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    @staticmethod
    def _ref(text):
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    def _ids(self, s):
        return self.tok(s)["input_ids"]

    def _tensor(self, ids):
        return torch.tensor(ids, dtype=torch.long, device=self.dev)

    def _proj_ids(self, ids):
        return self.t.text_projection(self.t.get_text_embeddings()(self._tensor(ids)))

    def _codec(self, ids):
        return self.t.get_input_embeddings()(self._tensor(ids))

    def _lang_id(self, language):
        lang = language.lower()
        if lang == "auto" or lang not in self.tc.codec_language_id:
            return None
        return self.tc.codec_language_id[lang]

    def _prefix(self, lang_id, spk_embed):
        """role (3) + tts_pad*(n-2) + tts_bos added to codec prefix minus its last row."""
        tc = self.tc
        c0 = self._codec([tc.codec_think_id, tc.codec_think_bos_id, lang_id, tc.codec_think_eos_id])
        c1 = self._codec([tc.codec_pad_id, tc.codec_bos_id])
        codec = torch.cat([c0, spk_embed.reshape(1, -1).to(self.dt), c1], 0)
        n = codec.shape[0]
        tt = torch.cat([self.tts_pad.expand(n - 2, -1), self.tts_bos], 0) + codec[:-1]
        return torch.cat([self.role, tt], 0), codec

    # ---------------------------------------------------------------- builders

    @torch.inference_mode()
    def custom(self, text, language, speaker, instruct=None):
        from uqt.engine import Prompt

        if instruct or speaker is None:
            return None
        lang_id = self._lang_id(language)
        spk = speaker.lower()
        if lang_id is None or spk not in self.tc.spk_id:
            return None
        if language.lower() == "chinese" and self.tc.spk_is_dialect.get(spk):
            return None
        key = ("custom", lang_id, spk)
        pre = self.cache.get(key)
        if pre is None:
            pre, _ = self._prefix(lang_id, self._codec([self.tc.spk_id[spk]]))
            self.cache[key] = pre
        text_id = self._ids(self._assistant(text))[3:-5]
        body = torch.cat([self._proj_ids(text_id), self.tts_eos], 0) + self.codec_pad
        emb = torch.cat([pre, body, self.last_custom], 0)
        return Prompt(embeds=emb, trailing=self.tts_pad, pad=self.tts_pad.reshape(-1))

    @torch.inference_mode()
    def clone(self, text, language, voice_clone_prompt):
        from uqt.engine import Prompt

        items = voice_clone_prompt if isinstance(voice_clone_prompt, list) else None
        if not items or len(items) != 1:
            return None
        it = items[0]
        if it.x_vector_only_mode or not it.icl_mode or it.ref_code is None or not it.ref_text:
            return None
        lang_id = self._lang_id(language)
        if lang_id is None:
            return None
        key = ("clone", id(it), lang_id)
        ent = self.cache.get(key)
        if ent is None or ent[0] is not it:
            pre, _ = self._prefix(lang_id, it.ref_spk_embedding.to(self.dev))
            rc = it.ref_code.to(self.dev)
            groups = [self.t.get_input_embeddings()(rc[:, 0])] + [
                self.t.code_predictor.get_input_embeddings()[i - 1](rc[:, i]) for i in range(1, rc.shape[1])]
            codec = torch.stack(groups, 1).sum(1)
            codec = torch.cat([self.codec_bos, codec], 0)
            ref_proj = self._proj_ids(self._ids(self._ref(it.ref_text))[3:-2])
            ent = (it, pre, codec, ref_proj, rc)
            self.cache[key] = ent
        _, pre, codec, ref_proj, rc = ent
        text_id = self._ids(self._assistant(text))[3:-5]
        tx = torch.cat([ref_proj, self._proj_ids(text_id), self.tts_eos], 0)
        R1 = codec.shape[0]
        if tx.shape[0] > R1:
            icl, trailing = tx[:R1] + codec, tx[R1:]
        else:
            icl = torch.cat([tx, self.tts_pad.expand(R1 - tx.shape[0], -1)], 0) + codec
            trailing = self.tts_pad
        return Prompt(embeds=torch.cat([pre, icl], 0), trailing=trailing, pad=self.tts_pad.reshape(-1),
                      ref_codes=rc)
