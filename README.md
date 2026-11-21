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
