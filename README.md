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

All systems run as HTTP servers on the same Modal H100 80GB HBM3 container type (16 CPUs), streaming
raw 16-bit 24 kHz PCM, and are measured with `bench/h2h/client.py` from request send (60
LibriSpeech texts for latency, 300 for load; CustomVoice, speaker Ryan). Competitors use their
fastest documented configuration: vLLM-Omni 0.32.0 `qwen3_tts_fused_single_gpu.yaml` +
`talker_stream_first_audio` (best of 4 documented profiles, 128 slots), sglang-omni @777c17f with the
CuTeDSL sm90 backend and stateful codec (CI CustomVoice preset, 64 running requests). uqt runs as a
GPU engine process (`uqt/engine_proc.py`) behind a Go gateway (`gateway/`), 512 slots, growing decode
chunks (2, 4, 8, 16 frames). Scripts: `bench/h2h/{vllm_omni,sglang_omni,uqt_go_h2h}.py`.

Methodology note: a single asyncio/httpx client process saturates at about 200 audio-seconds per
second of streamed PCM (verified against a synthetic Go server with no engine: header delays of
5-9 s at 256 connections). Earlier high-concurrency numbers in this repository were partly
client-limited for every system. The load generator now spreads clients over up to 16 processes
(about 32 connections each); against the synthetic server it sustains 1,729 audio-s/s at 512
connections with the scripted per-stream timing.

Single stream (sequential requests):

| System | TTFA p50 | TTFA voiced p50 | RTF p50 | Playback gap after first chunk |
|---|---|---|---|---|
| uqt single-stream engine (Python server) | **9.0 ms** | **16.2 ms** | **0.046** | none |
| uqt batched engine + Go gateway | 22.5 ms | 22.5 ms | 0.051 | none |
| vLLM-Omni 0.32 (fused) | 16.3 ms | 120 ms | 0.053 | ~20-25 ms on every stream |
| sglang-omni (CuTeDSL) | 20.0 ms | 20.0 ms | 0.047 | none |

Back-to-back clients (audio seconds per second / TTFA p50 / voiced TTFA p50):

| Clients | uqt (engine process + Go gateway) | vLLM-Omni 0.32 (fused) | sglang-omni (CuTeDSL) |
|---|---|---|---|
| 1 | 18 / 23 ms / 23 ms | 18 / 16 ms / 120 ms | 20 / 20 ms / 20 ms |
| 8 | 88 / 33 ms / 33 ms | 90 / 20 ms / 145 ms | 89 / 26 ms / 26 ms |
| 32 | 215 / 61 ms / 61 ms | **245** / 25 ms / 207 ms | 201 / 29 ms / 43 ms |
| 64 | 366 / 80 ms / 80 ms | 369 / 35 ms / 297 ms | 270 / 32 ms / 62 ms |
| 128 | **546** / 104 ms / 104 ms | 513 / 55 ms / 438 ms | 277 / 1.5 s (queue) |
| 256 | **734** / 160 ms / 160 ms | 608 / 1.4 s / 1.8 s (queue) | 278 / 4.7 s (queue) |
| 384 | **894** / 209 ms / 209 ms | 577 / 3.0 s / 3.4 s (queue) | - |
| 512 | **965** / 257 ms / 258 ms | 649 / 3.9 s / 4.3 s (queue) | - |

uqt has no underruns at any level; vLLM-Omni's streams almost all have a ~20 ms playback gap after
a near-silent first chunk, so its first-byte TTFA is not first audible audio. Below 128 clients
vLLM-Omni delivers its first bytes sooner and matches or slightly exceeds uqt throughput; from 128
clients up uqt delivers more audio (up to +49% at 512 clients) without queueing.

WER on the 60 latency utterances (Whisper large-v3): uqt 2.07-2.80%, vLLM-Omni 1.89%, sglang-omni
1.62%; differences of this size on 60 utterances are within sampling noise (see the 300 x 2-seed
parity study against the official implementation in `ultra-qwen3-tts`).
