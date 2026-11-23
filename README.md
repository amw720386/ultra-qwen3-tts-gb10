# ultra-qwen3-tts-gb10

Fast streaming inference for Qwen3-TTS-12Hz-1.7B on the NVIDIA GB10, with accuracy at parity with
the reference implementation. It targets both single-stream latency (time to first audio,
real-time factor) and batched throughput (many concurrent real-time streams on one box).

## Hardware notes

- GB10: Blackwell GPU, compute capability **sm_121**, with **unified LPDDR5x memory (~273 GB/s)**
  shared with the CPU.
- Decode is bandwidth-bound on this part: every Talker step streams the weights from LPDDR5x. Most
  of the speedup comes from moving fewer bytes per step (FP8 weights, attention that reads only live
  KV) and from fusing codec stages so intermediates never round-trip through memory.

## Running on a GB10 box

Everything runs directly on the box. Setup quirks:

```bash
# container: NGC PyTorch 25.10 (nvcr.io/nvidia/pytorch:25.10-py3)
pip uninstall -y torchao
pip install --no-deps torchaudio==2.9.0     # --no-deps keeps the NGC torch build
export UQT_NO_PDL=1                         # required with the container's Triton 3.4
```

Benchmarks (from the repository root):

```bash
# single-stream timings + batched step-time scan
python -m bench.gb10.bench_local --variant CustomVoice --weights fp8 --batch 1,8,32,64,128,256

# accuracy: 300 texts per weight format, Whisper large-v3 WER
python -m bench.gb10.acc_local --weights bf16,fp8 --n 300 --manifest /root/manifest.json

# codec decoder: fused vs unfused path (waveform SNR + time per stream-frame)
python -m bench.gb10.codec_fuse_test --cases 64x2,256x2

# streaming HTTP head-to-head on the box (engine process + Go gateway, or a running competitor server)
python -m bench.gb10.h2h_local --system uqt --weights fp8
```

The accuracy manifest is the LibriSpeech test-clean set built by `bench/baseline.py::prep`. Further
profilers and microbenchmarks are in `bench/gb10/`; measured outputs are in `results/gb10/`.

Switches: `--weights bf16|fp8` selects the weight format; `UQT_BATTN=0` turns off batched decode
attention; `UQT_RES_UNIT=0`, `UQT_FUSED_TRANS=0` and `UQT_MM_RES=0` disable individual fused codec
kernels for A/B comparisons; `UQT_FP8_BIG=0` keeps bf16 weights above 128 rows; `UQT_KV_TRITON=0`
uses indexed assignment for cache writes; `UQT_KV=bf16|int8|fp8` selects the batched KV cache format
(default int8).

Server experiments kept as opt-in switches (measured on GB10, no throughput gain at 1-256 clients):
`UQT_K_HIGH=4` (4-frame steps under load), `UQT_ADMIT_MIN=4` (batch admissions under load),
`UQT_EARLY_CHUNKS=2` (never defer the second audio chunk; fewer underruns at 64 clients only).
FP4 weights (`--weights fp4`, e2m1 with a bf16 scale per 32 weights) are kept opt-in: the rewritten
FP4 kernel beats FP8 on single-row GEMMs (talker MLP up-projection 0.073 vs 0.112 ms) and the
300-clip WER is 2.45% (one catastrophic failure included), but end to end it only improves
single-stream generation from 13.7 to 13.1 ms/frame, while prefill and batches above 16 rows get
slower (the code predictor's small GEMMs are latency-bound, not bandwidth-bound). GB10 has native
FP4 tensor-core instructions, but Triton (3.4 and 3.7) lowers `tl.dot_scaled` on sm_121 to a
software upcast plus bf16 MMA, so they are not used here.
Measured but not adopted: a W8A8 Triton GEMM (FP8 activations per row per K block) ran no faster
than the FP8 weight-only kernel on GB10 and had higher error (`results/gb10/w8a8_triton_test.json`).

## Techniques

- **FP8 weight-only GEMMs** (`uqt/wq.py`): linear weights stored as float8 e4m3 with one fp32
  scale per output channel, dequantized in registers inside Triton kernels; tensor-core math stays
  bf16. Half the weight bytes of bf16 per decode step. Above 128 rows a second kernel with a
  row-tile-fastest grid (each weight tile read from DRAM about once) is used where it measured
  faster: every shape up to 192 rows, the widest projections up to 320 (+19-21% engine throughput at
  160-192 streams). Prefill-sized calls use bf16 weights.
- **Hybrid FP8 + FP4 weights with 1-2 row GEMVs** (`--weights hyb`): every linear keeps an FP8 copy
  (prefill, batches) and an FP4 copy (e2m1, bf16 scale per 32). With one or two rows, as in
  single-stream decode, a multiply-reduce GEMV on CUDA cores reads the FP4 copy at ~190-260 GB/s
  instead of padding the row to 16 for tensor cores: single-stream generation 13.7 -> 9.25 ms/frame.
  A persistent megakernel for the code predictor was measured first and gave nothing: CUDA-graph
  replayed GEMVs already stream weights at the bandwidth limit (`results/gb10/mega_micro.json`).
- **Batched live-length decode attention** (`kernels.bdec_attn`): one kernel attends every active
  stream over only its own live KV length instead of padding to the longest stream in the batch.
- **Fused streaming codec kernels** (`uqt/streaming_codec.py`), on the stateful, frame-exact
  streaming decoder:
  - fused residual unit: conv1 + activation + conv2 + residual add in one kernel;
  - fused block entry: SnakeBeta + transposed convolution as one implicit GEMM;
  - fused output convolution with the final SnakeBeta folded in.
- **Tensor-core decode attention** (`UQT_BATTN_KIND=dot`, default; ported from the H100 backport):
  the two query heads of each kv head padded to 16 rows and attended with `tl.dot`, +1-3% engine
  throughput on GB10 (`UQT_BATTN_KIND=fma` restores the CUDA-core kernel).
- **INT8 KV cache for the batched engine** (`UQT_KV=int8`, default): every (token, kv head) row of
  K and V is stored as int8 with its own fp32 absmax scale, quantized inside the cache-write kernel;
  the attention kernel converts int8 to bf16 exactly and applies the scales to the logits (K) and to
  the probabilities (V). It halves the cache bytes read, and the attention kernel alone is 1.6-1.8x
  faster than BF16 at 128-256 streams (2.7x at 64 streams x 200 tokens, where the smaller cache
  fits in L2; `results/gb10/kv8_micro.json`, synthetic Gaussian keys). Plain e4m3 is
  available as `UQT_KV=fp8`. On a second GB10, three servers per format run alternately
  (`results/gb10/kv_server_runs.json`, mean of server means, audio-s/s): 64 clients 101.8 (BF16) vs
  114.3 (INT8), 128 clients 122.4 vs 140.0, with no overlap between the runs; 256 clients 156.0 vs
  165.9, where the runs overlap (BF16 128-165, INT8 151-182). Median per-stream RTF was 6-9% lower
  with INT8 in each of the three pairs at 128 and 256 clients. The earlier single runs on the first
  GB10 showed no gain at 64 (103.0 vs 101.9) or 128 clients (129.3 vs 130.6).
  Batched WER, 300 texts x 5 seeds (1,500 clips per format, 256 slots, hybrid weights,
  `bench/gb10/kv_wer_stats.py`): BF16 2.09%, INT8 1.81%, FP8 2.21%; clips over 50% errors 2, 1 and 4.
  INT8 minus BF16 is -0.28 points (95% bootstrap CI over texts -0.62 to +0.01), so INT8 shows no
  loss and no proven gain; FP8 minus BF16 is +0.13 (-0.31 to +0.60). On the real cache (24 prompts,
  one step, BF16 weights, `results/gb10/kv_real_error.json`) the mean row error is 1.4% (K) and 0.8%
  (V) for INT8 against 2.6% and 2.7% for FP8.
- **Single-launch K/V cache writes** for the batched step (`kernels.kv_write_rows`).
- **Tuned implicit-GEMM tiles**: tile shapes per decoder width (including the 768-wide units),
  swept on GB10 at the shapes the batched engine runs.
- Carried over from the H100 engines: CUDA-graphed Talker and code-predictor steps, the fused
  Triton code predictor, continuous batching with packed prefill, and per-voice cached codec state.

## Results (GB10)

Single stream:

| Configuration | ms / frame | RTF | TTFA |
|---|---|---|---|
| Stock reference implementation | - | 0.72 | - |
| uqt, bf16 | 28.3 | 0.38 | 47 ms |
| uqt, FP8 weights | 13.6 | 0.18 | 25 ms |
| uqt, hybrid FP8 + FP4 weights (`--weights hyb`) | **9.25** | **0.127** | **20.6 ms** |

Accuracy (300 clips, Whisper large-v3 WER, excluding catastrophic failures): bf16 2.82%, FP8 1.70%,
hybrid FP8 + FP4 2.05% (one catastrophic failure in each of FP8 and hybrid).
One seed each: no FP8 loss is visible here, but the run-to-run spread was not measured, so this
does not rank the three.

Batched engine throughput (audio seconds generated per second, engine only):

| Streams | 1 | 8 | 32 | 64 | 128 | 160 | 192 | 256 |
|---|---|---|---|---|---|---|---|---|
| bf16 | 2.7 | 21.9 | 73.4 | 114.5 | 157.9 | - | - | 193.0 |
| FP8 + batched attention + fused codec + large-batch FP8 GEMMs | **5.4** | **40.8** | **126.3** | **197.0** | **233.4** | **273.7** | **289.9** | **289.9** |

N streams in real time need N audio-s/s, so every configuration above runs all its streams faster
than real time, including 256 (289.9 audio-s/s).

Accuracy of the batched serving path: all 300 texts through the batched engine at 256 slots
(FP8, large-batch FP8 GEMMs, fused codec) give 1.89% WER excluding one catastrophic failure, at
parity with the single-stream runs above.

Codec decoder time per stream-frame at 256 streams x 2 frames: **157 us -> 99 us** with the fused
kernels. Against an fp32 decode, the waveform is about 28 dB SNR for both the old and the fused
paths (the fused path equal or slightly closer).

### Head-to-head on GB10

Streaming HTTP (`/v1/audio/speech`, raw PCM), CustomVoice, voice Ryan, measured on the box with the
same multi-process client (`bench/gb10/h2h_local.py`), one server at a time. Each client sends
requests back to back. Throughput in audio-s/s; TTFA = time to first audio byte, voiced = time to
the first non-silent audio (p50, ms).

| Clients | uqt (engine process + Go gateway; hybrid weights, INT8 KV) | vLLM-Omni 0.32 (2-stage default) | sglang-omni 0.2 |
|---|---|---|---|
| 1 | **8.1** / 31 / 31 | 2.4 / 60 / 858 | 2.6 / 78 / 78 |
| 8 | **25.8** / 91 / 91 | 13.1 / 109 / 1035 | 17.1 / 106 / 106 |
| 32 | **78.9** / 153 / 153 | 27.2 / 162 / 1857 | 39.3 / 93 / 195 |
| 64 | **101.9** / 201 / 201 | 36.0 / 261 / 3154 | 61.5 / 103 / 248 |
| 128 | **130.6** / 332 / 332 | 40.8 / 10090 / 13124 | 69.3 / 5280 / 5410 (15 of 384 failed) |
| 256 | **174.0** / 521 / 522 | 41.7 / 31965 / 34979 | 72.2 / 5030 / 5158 (410 of 768 failed) |

Cells: throughput / TTFA p50 / voiced TTFA p50. Voiced TTFA is when the chunk holding the first
voiced sample arrives, not when a listener hears it: our clips open with about 90 ms of near-silence
(median of 300), which plays in real time after the first byte. uqt's and
vLLM-Omni's clips do this (90 ms median of 40 each); sglang-omni's start almost at once (10 ms). uqt's
voice is therefore heard at roughly 240 ms at 32 clients and 290 ms at 64, against about 195 and
250 ms for sglang-omni, so the voiced column does not show a uqt lead over it at 32-64 clients.
`bench/h2h/client.py` now records `heard_ms`; the tables have not
been rerun with it. uqt completed every request at every load.
Each row is one run, all on the same first GB10; the three-run spread at 256 clients on the second
GB10 is above, and single runs there differ by up to 20%, so treat the 256 row as approximate.

Playback gaps: share of requests with at least one chunk later than real-time playback started at
the first byte (no client buffer, any size of delay counts), and median RTF, from the same runs:

| Clients | uqt (INT8 KV) | vLLM-Omni | sglang-omni |
|---|---|---|---|
| 1 | 0% / 0.12 | 100% / 0.40 | 0% / 0.37 |
| 8 | 0% / 0.22 | 100% / 0.46 | 29% / 0.38 |
| 32 | 1% / 0.30 | 100% / 0.81 | 70% / 0.54 |
| 64 | 4% / 0.40 | 100% / 1.38 | 99% / 0.75 |
| 128 | 95% / 0.64 | 100% / 1.49 | 100% / 0.84 |
| 256 | 100% / 1.05 | 100% / 1.50 | 100% / 0.84 |

Up to 64 clients about 1 request in 20 or fewer has a late chunk with uqt (5-7% at 64 on the
second GB10). At 128 nearly every request has a late chunk even
though generation is faster than real time on average, and at 256 it is slower than real time. The
size of those delays was not recorded.
Earlier uqt configurations on the same harness: FP8 weights with a BF16 cache measured 5.4 / 41 / 41
at 1 client and 166.9 / 556 / 558 at 256 (`results/gb10/h2h-uqt.json`); hybrid weights with a BF16
cache 8.1 / 31 / 31 and 159.9 / 560 / 560 (`results/gb10/h2h-uqt-hyb-kvwrite.json`).
Setup notes for both competitor stacks on sm_121 (aarch64 wheels, a one-line `num_stages` patch
for an sglang-omni Triton kernel that exceeds GB10's 99 KiB shared memory) are in
`bench/gb10/competitors/`. vLLM-Omni's fused single-GPU profile does not start on GB10 (a Triton
kernel in its MTP graph capture needs 136 KiB of shared memory), so the default 2-stage profile is
the one measured.

## Layout

```
uqt/            single-stream and batched engines, kernels, FP8 weight-only GEMMs,
                streaming codec, engine process and HTTP server
gateway/        Go streaming gateway in front of the engine process
bench/gb10/     on-box GB10 benchmarks, accuracy check, profilers, microbenchmarks
bench/          H100 Modal benchmarks (baseline, engine_bench, eval_large, load_test, h2h/)
results/gb10/   GB10 measurements
results/h100/   H100 measurements carried over from the origin repositories
```

## Origin

This repository started from the H100 work: the single-stream engine in
[`amw720386/ultra-qwen3-tts`](https://github.com/amw720386/ultra-qwen3-tts) and the batched
engine in
[`amw720386/ultra-qwen3-tts-multistream`](https://github.com/amw720386/ultra-qwen3-tts-multistream),
where H100 results and head-to-head comparisons are documented. The H100 Modal scripts under
`bench/` and their results under `results/h100/` are kept for reference.
