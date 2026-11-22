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
kernels for A/B comparisons.

## Techniques

- **FP8 weight-only GEMMs** (`uqt/wq.py`): linear weights stored as float8 e4m3 with one fp32
  scale per output channel, dequantized in registers inside Triton kernels; tensor-core math stays
  bf16. Half the weight bytes of bf16 per decode step. Large-M calls (prefill) use bf16 weights.
- **Batched live-length decode attention** (`kernels.bdec_attn`): one kernel attends every active
  stream over only its own live KV length instead of padding to the longest stream in the batch.
- **Fused streaming codec kernels** (`uqt/streaming_codec.py`), on the stateful, frame-exact
  streaming decoder:
  - fused residual unit: conv1 + activation + conv2 + residual add in one kernel;
  - fused block entry: SnakeBeta + transposed convolution as one implicit GEMM;
  - fused output convolution with the final SnakeBeta folded in.
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
| uqt, FP8 weights | **13.6** | **0.18** | **25 ms** |

Accuracy (300 clips, Whisper large-v3 WER, excluding catastrophic failures): bf16 2.82%, FP8 1.70%.
The difference is within seed noise: FP8 loses no accuracy.

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

| Clients | uqt (engine process + Go gateway, FP8) | vLLM-Omni 0.32 (2-stage default) | sglang-omni 0.2 |
|---|---|---|---|
| 1 | **5.4** / 41 / 41 | 2.4 / 60 / 858 | 2.6 / 78 / 78 |
| 8 | **26.7** / 106 / 106 | 13.1 / 109 / 1035 | 17.1 / 106 / 106 |
| 32 | **69.8** / 161 / 161 | 27.2 / 162 / 1857 | 39.3 / 93 / 195 |
| 64 | **99.3** / 216 / 216 | 36.0 / 261 / 3154 | 61.5 / 103 / 248 |
| 128 | **124.9** / 351 / 352 | 40.8 / 10090 / 13124 | 69.3 / 5280 / 5410 (15 of 384 failed) |
| 256 | **166.9** / 556 / 558 | 41.7 / 31965 / 34979 | 72.2 / 5030 / 5158 (410 of 768 failed) |

Cells: throughput / TTFA p50 / voiced TTFA p50. uqt completed every request at every load.
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
