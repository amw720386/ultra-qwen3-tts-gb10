# ultra-qwen3-tts

Goal: the fastest Qwen3-TTS-12Hz-1.7B inference on a single NVIDIA H100, with accuracy at parity
with the reference implementation.

## Metrics

| Metric | Definition |
|---|---|
| TTFA | Wall time from request to the first audio chunk (streaming), median. |
| TTFA (voiced) | Wall time until the chunk containing the first sample above -40 dBFS arrives. |
| RTF | Wall time / audio duration (lower is faster; 0.25 = 4x real time). |
| WER | Whisper large-v3, greedy, English, Whisper `EnglishTextNormalizer`, corpus-level. |
| SIM (proxy) | Cosine of `microsoft/wavlm-base-plus-sv` x-vectors, generated vs prompt audio. |

Voice-clone accuracy set: LibriSpeech test-clean, target utterances 4-10 s, prompt = a different
3-8 s utterance of the same speaker (`bench/phase0.py::prep`).

## Phase 0: baselines

```powershell
modal run bench/phase0.py --n 60
```

Results land in `results/phase0/`. H100 80GB HBM3 (SXM), bf16, batch 1, 60-utterance accuracy set.

| System | Variant | TTFA (median) | RTF (streaming, chunk 8) | RTF (accuracy set) | WER | SIM (proxy) |
|---|---|---|---|---|---|---|
| Ground truth (LibriSpeech audio) | - | - | - | - | 1.80% | 0.954 |
| qwen-tts 0.1.1 (official, sdpa) | Base | 6913 ms (non-streaming) | - | 0.959 | 1.44% | 0.955 |
| qwen-tts 0.1.1 (official, sdpa) | CustomVoice | 7793 ms (non-streaming) | - | 1.193 | 1.35% | - |
| faster-qwen3-tts 0.5.4 | Base | 97 ms (chunk 2) / 221 ms (chunk 8) | 0.293 | 0.303 | 1.26% | 0.945 |
| faster-qwen3-tts 0.5.4 | CustomVoice | 115 ms (chunk 2) / 214 ms (chunk 8) | 0.258 | 0.239 | 2.71% | - |

Notes:
- faster-qwen3-tts costs about 21 ms of GPU wall time per 80 ms codec frame, plus about 55 ms fixed
  overhead before the first chunk. A bandwidth estimate for the same work is about 2 ms per frame.
- The 2.71% CustomVoice WER is driven mostly by one hallucinated utterance; 60 utterances is too small
  to separate a real accuracy regression from sampling noise. Later phases use a larger set and
  multiple seeds before any parity claim.
- Several residual errors are reference-text quirks in LibriSpeech ("shewd", "emerg would") that
  every system, including ground truth, gets "wrong".

## Phase 1: uqt engine

`uqt/engine.py` reuses the official package for weights, tokenization, prompt construction and the
codec, and replaces the generation loop:

- static KV caches, one CUDA graph per chunk of K frames, no host sync inside a chunk;
- `torch.compile`d Talker step, with KV writes as opaque in-place custom ops (otherwise inductor
  copies a whole layer cache per write);
- fused Triton code predictor with programmatic dependent launch (`uqt/fused_cp.py`, ported from
  vLLM-Omni, Apache-2.0) and a single-kernel top-k sampler (`uqt/kernels.py`);
- prompt prefill graphed over padded length buckets; codec decoder graphed per window size and run
  on its own stream, overlapped with generation of the next chunk; chunk sizes ramp 1, 2, 4, 8.

Correctness: with greedy decoding in fp32 the engine reproduces the official codes exactly
(6 prompts, up to 119 frames each, `modal run bench/phase1.py::parity_all`). In bf16, kernel
rounding differences make greedy outputs diverge after a few frames (the same happens between
eager and compiled PyTorch), so bf16 accuracy is judged by WER/SIM.

```powershell
modal run bench/phase1.py::bench_all --n 60
```

H100 80GB HBM3, bf16, batch 1. TTFA includes prompt construction (cached voice prompt) and is
measured on the host when the first PCM chunk is available.

| System | Variant | Gen ms / frame | TTFA (first chunk) | TTFA (voiced) | RTF (streaming) | WER | SIM (proxy) |
|---|---|---|---|---|---|---|---|
| faster-qwen3-tts 0.5.4 | Base | ~21 | 97 ms | 97 ms | 0.293 | 1.26% | 0.945 |
| faster-qwen3-tts 0.5.4 | CustomVoice | ~21 | 115 ms | 115 ms | 0.258 | 2.71% | - |
| uqt (phase 1) | Base | 3.66 | 15.5 ms | 37.6 ms | 0.057 | 1.53% | 0.953 |
| uqt (phase 1) | CustomVoice | 3.85 | 13.5 ms | 17.4 ms | 0.057 | 1.98% | - |

Per-frame breakdown (CustomVoice, graphed): fused code predictor 1.83 ms (15 dependent steps),
Talker step 2.06 ms. "Voiced" TTFA waits for the first chunk containing audio above -40 dBFS; the
Base model emits ~80-300 ms of leading silence on this voice, as the official implementation does.

## Phase 2: latency and accuracy parity

Changes on top of Phase 1:

- `uqt/prompt.py`: native prompt construction with per-voice / per-speaker caching; bit-identical
  to the official builder in bf16 (12/12 prompts), 0.24 ms instead of 2.0-2.4 ms.
- `uqt/streaming_codec.py`: stateful, frame-exact streaming codec decoder (ported from vLLM-Omni,
  Apache-2.0). Matches the official full-utterance decode at 30-36 dB SNR (bf16 rounding) where the
  Phase 1 25-frame-window decode reached only 11-22 dB; 0.54-0.65 ms per chunk instead of
  4.4-5.7 ms. The decoder state after a voice's reference codes is cached per voice.
- Talker step compiled with inductor `max-autotune` (1.71 ms instead of 2.06 ms per step).
- `uqt/fused_talker.py` (opt-in, `fused_talker=True`): the Talker as a PDL kernel chain with a
  split-key attention kernel. Numerically sound (logit cosine >= 0.99994, greedy argmax agreement
  59/60 frames) but 1.94 ms per step, slower than the autotuned path; at batch 1 each of the ~170
  kernels is latency-bound. Kept for future work (megakernel).

### Accuracy: 300 LibriSpeech pairs x 2 seeds

```powershell
modal run bench/phase0.py::prep --n-pairs 300
modal run bench/eval_large.py --n 300 --seeds 1,2
```

| Variant | System | WER seed 1 | WER seed 2 | SIM seed 1 | SIM seed 2 |
|---|---|---|---|---|---|
| Base | official qwen-tts | 1.26% | 1.04% | 0.956 | 0.955 |
| Base | uqt (Phase 2) | 1.06% | 1.06% | 0.954 | 0.956 |
| CustomVoice | official qwen-tts | 5.85% | 1.86% | - | - |
| CustomVoice | uqt (Phase 2) | 2.53% | 1.80% | - | - |

Paired bootstrap over utterances (both seeds pooled), uqt minus official corpus WER:
Base -0.09 points (95% CI -0.23 to +0.04); CustomVoice -1.69 points (95% CI -6.0 to +0.7).
CustomVoice WER is dominated by rare runaway generations in both systems (one official utterance
produced 210 extra words in seed 1).

### Latency (H100, bf16, batch 1, `modal run bench/phase1.py::bench_all --n 60`)

| Variant | Gen ms / frame | TTFA (first chunk) | TTFA (voiced) | RTF (streaming) |
|---|---|---|---|---|
| Base | 3.47 | 7.3 ms | 24.9 ms | 0.046 |
| CustomVoice | 3.48 | 7.6 ms | 10.9 ms | 0.046 |

TTFA uses chunk schedule 1, 2, 4, 8 (first chunk) or 2, 4, 8 (voiced numbers, which are lowest with
that schedule). Prompt build 0.6 ms, prefill 2.1 ms, first frame 3.5 ms, first decode 0.6 ms.
Startup (compile, autotune, graph capture) currently takes about 9 minutes per process.
