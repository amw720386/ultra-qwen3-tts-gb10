# ultra-qwen3-tts-multistream

Goal: the most concurrent real-time Qwen3-TTS-12Hz-1.7B streams on a single NVIDIA H100, with
accuracy at parity with the reference implementation. The single-stream (one GPU per voice) engine
lives in the sibling repository `ultra-qwen3-tts`; this repository shares its kernels and adds a
continuous-batching engine.

## Engine (`uqt/batch_engine.py`)

- Every active stream owns a slot (KV cache, sampling state, trailing text, codec state).
- One CUDA-graphed step advances all slots `[0, B)` by K frames (K = 2, i.e. 160 ms of audio):
  - fused Triton code predictor over B rows (`uqt/fused_cp.py`, ported from vLLM-Omni);
  - compiled batched Talker step: one GEMM per projection for all rows, per-row RoPE, KV write and
    masked attention, per-row repetition penalty / suppression / top-k sampling;
  - graphs are captured per (K, B bucket, KV-length bucket); B buckets step by 32 above 64.
- New requests are admitted between steps with a packed prefill: all prompts arriving in a step run
  as one sequence (positions restart per prompt, block-diagonal causal mask), then each prompt's
  K/V is copied into its slot. Checked against single-prompt prefill: first sampled code 8/8 equal,
  hidden-state cosine >= 0.9996 (greedy).
- Audio comes from the stateful, frame-exact streaming codec (`uqt/streaming_codec.py`), batched over
  slots and run on its own stream, overlapped with the next generation step.

## Load test (`bench/load_test.py`)

```powershell
modal run bench/load_test.py --variant CustomVoice --concs 512,768,896,1024 --max-slots 256
modal run bench/load_test.py --variant CustomVoice --concs 128,256,384,512 --max-slots 512 --max-len 512 --think-s -1
```

Two client models, both closed loop, texts drawn from LibriSpeech test-clean (4-10 s of audio):

- **Conversational**: each client sends a request, listens to the whole reply in real time, pauses
  0.5 s, then sends the next (a voice agent). "C" is the number of simultaneous conversations.
- **Back-to-back**: each client sends its next request as soon as the previous one finishes
  generating; "C" is close to the number of streams generating at once.

Metrics: TTFA = request arrival to the first PCM chunk on the host (includes queueing and prefill);
RTF = generation time after the first chunk / audio duration; an underrun is any chunk that reaches
the client after playback (started at the first chunk) needed it.

### Results (H100 80GB HBM3, bf16, CustomVoice, K = 2)

Conversational clients, 256 slots:

| Conversations | Streams generating (avg) | Audio s / s | TTFA p50 | TTFA p90 | TTFA p99 | RTF p90 | Underruns |
|---|---|---|---|---|---|---|---|
| 256 | 13 | 144 | 23 ms | 32 ms | 41 ms | 0.11 | 0 |
| 512 | 36 | 261 | 56 ms | 69 ms | 77 ms | 0.15 | 0 |
| 768 | 56 | 364 | 60 ms | 74 ms | 87 ms | 0.17 | 0 |
| 896 | 73 | 400 | 71 ms | 92 ms | 108 ms | 0.21 | 0 |
| 1024 | 126 | 621 | 81 ms | 196 ms | 552 ms | 0.23 | 0 |

(The 256-conversation row is from the earlier 128-slot run without packed prefill.)

Back-to-back clients, 512 slots:

| Clients | Streams generating (avg) | Audio s / s | TTFA p50 | TTFA p90 | RTF p90 | Underruns |
|---|---|---|---|---|---|---|
| 128 | 92 | 596 | 51 ms | 60 ms | 0.16 | 0 |
| 256 | 188 | 892 | 69 ms | 87 ms | 0.22 | 0 |
| 384 | 296 | 1046 | 93 ms | 207 ms | 0.29 | 0 |
| 512 | 384 | 1066 | 119 ms | 442 ms | 0.38 | 0 |

At 512 back-to-back clients every stream is still generated 2.6x faster than real time. TTFA tails
at that level come from all clients starting within 0.5 s.

Step-time scan (one step = 2 frames for every slot, generation + codec): B = 1: 8.6 ms;
64: 14.9 ms; 128: 18.2 ms; 256: 26.5 ms; 512: 43.9 ms (24.9 ms generation + 19.0 ms codec).

### Accuracy under load

Whisper large-v3 WER of audio produced during the back-to-back runs: 2.11% at 128 clients
(256 utterances) and 1.90% at 512 clients (281 utterances), no utterance above 50% WER. The
single-stream engine scores 1.80-2.53% and the official implementation 1.86-5.85% on the same
300-utterance set (see `ultra-qwen3-tts`).

### Known issues

- Every packed-prefill bucket must be captured at startup (`BatchEngine.warm_pack`): a bucket
  first reached under load compiled lazily and stalled admission for ~34 s.
- Startup (compile and graph capture for every batch / length bucket) takes several minutes.
