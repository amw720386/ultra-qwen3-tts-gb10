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

## Head-to-head on one H100 (same client, same texts)

All systems run as HTTP servers in their own process on the same Modal H100 80GB HBM3, streaming raw
16-bit 24 kHz PCM, measured with `bench/h2h/client.py` (time from request send; 60 LibriSpeech
texts for latency, 300 for load; CustomVoice, speaker Ryan). Competitors use their fastest documented
configuration: vLLM-Omni 0.32.0 `qwen3_tts_fused_single_gpu.yaml` + `talker_stream_first_audio`
(best of 4 documented profiles), sglang-omni @777c17f with the CuTeDSL sm90 backend and stateful
codec (CI CustomVoice preset). Scripts: `bench/h2h/{vllm_omni,sglang_omni,uqt_h2h}.py`.

Single stream:

| System | TTFA p50 / p90 | TTFA voiced p50 | RTF p50 | Playback gap after first chunk | WER (60 utts) |
|---|---|---|---|---|---|
| uqt single-stream engine | **9.0 / 10.2 ms** | **16.2 ms** | **0.046** | none | 2.80% |
| vLLM-Omni 0.32 (fused) | 13.9 / 14.4 ms | 118.5 ms | 0.053 | ~25 ms on every stream | 1.89% |
| sglang-omni (CuTeDSL) | 22.2 / 23.3 ms | 22.2 ms | 0.048 | none | 1.62% |
| uqt batched server, lone request | 21.0 / 22.3 ms | 21.1 ms | 0.053 | none | 2.07% |

WER differences on 60 utterances are within sampling noise (the 300 x 2-seed comparison against the
official implementation in `ultra-qwen3-tts` shows parity).

Back-to-back clients (throughput in audio seconds per second / TTFA p50):

| Clients | uqt batched server | vLLM-Omni 0.32 (fused) | sglang-omni (CuTeDSL) |
|---|---|---|---|
| 1 | 18 / 21 ms | 18 / 14 ms | 19 / 22 ms |
| 8 | 84 / 33 ms | 103 / 17 ms | 90 / 30 ms |
| 32 | 208 / 49 ms | 316 / 21 ms | 193 / 32 ms |
| 64 | 281 / 64 ms | 542 / 24 ms | 238 / 35 ms (18% underruns) |
| 128 | 477 / 87 ms | **815 / 33 ms** | 255 / 1.7 s (queueing) |
| 256 | 600 / **150 ms** | 923 / 878 ms (queueing) | 257 / 5.2 s (queueing) |

uqt has no underruns at any level; vLLM-Omni's streams all have a ~25 ms playback gap after the first
chunk. Behind HTTP, vLLM-Omni currently delivers more throughput than the uqt batched server between
8 and 256 clients; the uqt server loses throughput relative to the same engine measured in-process
(892 audio-s/s at 256 back-to-back clients) and decodes every 2-frame chunk, where vLLM-Omni decodes
25-frame chunks.
