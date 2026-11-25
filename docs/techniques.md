# Techniques

## The hardware

- GB10: Blackwell GPU, compute capability **sm_121**, with **unified LPDDR5x memory (~273 GB/s)**
  shared with the CPU.
- Decode is bandwidth-bound on this part: every Talker step streams the weights from LPDDR5x. Most
  of the speedup comes from moving fewer bytes per step (FP8 weights, attention that reads only live
  KV) and from fusing codec stages so intermediates never round-trip through memory.

## What the engine does

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

## Serving loop

In `uqt/engine_proc.py`:
- Host-to-GPU copies in the step launch and decode are non-blocking (`UQT_ASYNC_H2D`). A blocking copy
  waited for the step that was just launched, so decode and admission did not overlap generation.
- A request that arrives while a step runs joins the next step (`UQT_LATE_ADMIT`).
- A stream's pending audio goes out early when its client buffer is about to run dry, before the chunk
  is full (`UQT_DEADLINE`, `UQT_DEADLINE_MARGIN_MS`, 120). Requests with a late chunk at 128 clients went from
  about 99% of requests to a few percent.
- One-frame steps (`UQT_K_LOW_FROM`, `UQT_K_LOW_AT`: above 8 and up to 64 streams, active or queued) let a
  request wait for a shorter step. From 41 to 64 streams (`UQT_FRESH1_FROM`, `UQT_FRESH1_AT`) the step that
  admits a stream is one frame and that frame is sent at once, with at least 60 ms of it kept
  (`UQT_F1_KEEP_MS`), so the model's opening silence plays while the next frame is generated. Both are off
  during a burst (more than `UQT_BURST_N`=24 admissions in half a second). A/B in one server, 4 rounds
  (`results/gb10/ab-firstbyte5.json`), first byte p50: 103 -> 89 ms at 32 clients, 134 -> 90 at 64; voice heard 115 -> 104 and
  220 -> 161; throughput at 48 and 64 clients 4-5% lower; gaps over 20 ms no more frequent at 48-256, about
  a point more at 16-32; any gap at 128 6% -> 12% (almost all under 20 ms). Server WER at 32 and 64 clients, 2,400 clips per
  setting (`results/gb10/swer-firstbyte`, before the burst rule and with a 72-80 stream cap): 2.29% before, 2.15% with
  it (6 and 7 clips over 50% errors, 1.91% for both without them); difference -0.13 (95% CI -0.57 to
  +0.28). Not helpful: pinned copies in prefill admission, other admission lead times, decoding first chunks before the step.
  A one-frame and a two-frame decode of the same codes agree to 33 dB SNR.
- Chunks of one stream are always delivered in order; `engine_proc` counts any that are not (`ooo`).

## Switches, and what was measured but not adopted

Switches: `--weights bf16|fp8` selects the weight format; `UQT_BATTN=0` turns off batched decode
attention; `UQT_RES_UNIT=0`, `UQT_FUSED_TRANS=0` and `UQT_MM_RES=0` disable individual fused codec
kernels for A/B comparisons; `UQT_FP8_BIG=0` keeps bf16 weights above 128 rows; `UQT_KV_TRITON=0`
uses indexed assignment for cache writes; `UQT_KV=bf16|int8|fp8` selects the batched KV cache format
(default int8).

`UQT_PACK_MIN=0` (prefill sizes of 32-96 tokens) cuts first byte by 4-5 ms at 32-64 clients, but WER was
higher in the same test (excluding garbled clips, +0.19 points against the earlier settings, CI 0.03 to
0.35), so it is off. Prompt riding (prompt tokens as extra rows of the batched step) was bounded, not ported:
it could save at most about 10% of a step at 32-128 streams and 1-2% at 256, and would add a step to the
first byte (`bench/gb10/ride_bound.py`).

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
