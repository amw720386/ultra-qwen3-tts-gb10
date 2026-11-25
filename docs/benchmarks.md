# Benchmarks

Every number is from one NVIDIA GB10 unless it says otherwise. `tools/release_numbers.py` turns the raw runs into
`results/gb10/release.json`, which the README, its charts and the video read.

## Single stream

| Configuration | ms / frame | RTF | TTFA |
|---|---|---|---|
| uqt, bf16 | 28.3 | 0.38 | 47 ms |
| uqt, FP8 weights | 13.6 | 0.18 | 25 ms |
| uqt, hybrid FP8 + FP4 weights (`--weights hyb`) | **9.25** | **0.127** | **20.6 ms** |

## Accuracy

Single stream (300 clips, Whisper large-v3 WER, excluding catastrophic failures): bf16 2.82%, FP8 1.70%,
hybrid FP8 + FP4 2.05% (one catastrophic failure in each of FP8 and hybrid).
One seed each: no FP8 loss is visible here, but the run-to-run spread was not measured, so this
does not rank the three.

## Batched engine

Throughput (audio seconds generated per second, engine only):

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

## Head-to-head

Streaming HTTP (`/v1/audio/speech`, raw PCM), CustomVoice, voice Ryan, measured on one GB10 with the
same multi-process client (`bench/gb10/h2h_local.py`), one server at a time, default settings for uqt.
Each client sends requests back to back. One run at 1 client, three at 8-64, two at 128-256; ranges
are the spread of those runs. uqt and sglang-omni at 8-64 were measured on the same day
(`results/gb10/final2-uqt.jsonl`, `comp2-sglang.jsonl`); vLLM-Omni, and sglang-omni at 1, 128 and 256,
two days earlier on the same machine (`comp-*.jsonl`).

| Clients | uqt (engine process + Go gateway; hybrid weights, INT8 KV) | vLLM-Omni 0.32 (2-stage default) | sglang-omni 0.2 |
|---|---|---|---|
| 1 | 9 / 32 / 33 / 42 | 2 / 56 / 60 / 873 | 3 / 79 / 79 / 89 |
| 8 | 29-31 / 77-78 / 81-84 / 88 | 12-14 / 96-111 / 126-132 / 1023-1042 | 13-16 / 101-106 / 112-114 / 112-115 |
| 16 | 44-50 / 76-80 / 110-116 / 86-92 | not measured | 25-26 / 114-117 / 328-392 / 125-127 |
| 32 | 73-80 / 87-90 / 128-136 / 100-105 | 29-30 / 155-168 / 223-240 / 1853-1896 | 40-44 / 93-97 / 458-464 / 200-207 |
| 64 | 88-122 / 91-93 / 162-170 / 161-164 | 37 / 266-267 / 356-377 / 3104-3172 | 63-65 / 100-104 / 462-486 / 260-264 |
| 128 | 127-138 / 229-231 / 392-393 / 312-321 | 40 / 10275-10539 / 11286-11760 / 13299-13542 | 70-71 / 4558-4710 / 6416-6542 / 4698-4842 (33-34 of 384 failed) |
| 256 | 183-192 / 324 / 1076-1188 / 415 | 42 / 31039-31511 / 32571-32719 / 34078-34506 | 72 / 4834-5007 / 6120-6214 / 5008-5219 (422-434 of 768 failed) |

Cells: throughput (audio-s/s) / first byte p50 / first byte p90 / voice heard p50 (ms). "Voice heard"
is when a listener hears the voice: first byte, plus any stall, plus the silence a clip opens with.
uqt drops up to 100 ms of that silence (about 90 ms, `UQT_TRIM_MAX_MS=0` keeps it) while a generation
step takes at most 56 ms, so that it shows up at up to 32 clients, and at 41-64 streams it keeps at least 60 ms
of the first frame (see the serving loop below); sglang-omni's clips start almost at once on their own.

- uqt leads on throughput, voice heard, and first byte (p50 and p90) at every load measured. It
  completed every request; sglang-omni rejected about 9% at 128 and 55% at 256 ("request queue is
  full", its default limits).
- **Not a win:** per-stream RTF at 256 (0.98 against 0.84, on the streams sglang-omni kept). The margin on
  median first byte at 32 clients is small (3-10 ms).
- One of the three 64-client uqt runs was slower in throughput than the others (88 audio-s/s; the other two
  were above 100).

## Playback gaps

Median RTF / share of requests with any chunk later than real-time playback started
at the first byte (no client buffer, any size of delay counts) / p90 of the worst delay per request in ms:

| Clients | uqt | vLLM-Omni | sglang-omni |
|---|---|---|---|
| 1 | 0.12 / 0% / 0 | 0.40 / 100% / 722 | 0.37 / 0% / 0 |
| 8 | 0.21 / 17-25% / 13-22 | 0.47 / 100% / 884 | 0.38 / 29-33% / 75-87 |
| 16 | 0.23 / 8% / 0 | not measured | 0.42 / 21-40% / 12-38 |
| 32 | 0.29 / 3-4% / 0 | 0.82 / 100% / 1702 | 0.54 / 65-72% / 131-165 |
| 64 | 0.41 / 5-9% / 0 | 1.38 / 100% / 5576 | 0.75 / 97-99% / 421-456 |
| 128 | 0.64 / 4-10% / 0 | 1.50 / 100% / 6577 | 0.84 / 100% / 435 |
| 256 | 0.98 / 60-63% / 75-97 | 1.50 / 100% / 6582 | 0.84 / 98% / 430 |

At 1-128 clients a client with a small buffer hears no gaps (p90 delay at most about 22 ms, at 8 clients,
where the silence trim removes part of the buffer). At 256 generation runs at real time with no
headroom, so about two thirds of requests have a late chunk (p90 about 75-140 ms).

## Accuracy through the server

Word error rate through the running server under load (INT8 KV, hybrid weights, 300 texts x 4 interleaved
runs at each of 64 and 256 clients = 2,400 clips per setting; `results/gb10/swer-reps`): settings before
the changes below 2.23% (9 clips over 50% errors, 1.85% without them), current settings 2.12% (3 clips,
1.93%). Current minus earlier is -0.11 points (95% bootstrap CI over texts -0.52 to +0.27), so no loss
was measured. Clips over 25 s, at 256 clients over 3,000 clips each: 1 earlier, 2 now.

All three systems through their own servers on one GB10 (a second unit of the same model; `bench/gb10/werq.sh`,
`bench/gb10/wer3_stats.py`, `results/gb10/wer3`): the same 300 texts, 4 runs at 32 and 4 at 64 clients, 2,400 clips
per system, every clip different audio, Whisper large-v3, uqt with final default settings.

| System | WER | Excluding garbled clips (>50% errors) | Garbled | Failed requests |
|---|---|---|---|---|
| uqt | 2.46% | 1.91% | 9 | 0 |
| sglang-omni 0.2 | 1.96% | 1.87% | 3 | 0 |
| vLLM-Omni 0.32 | 2.93% | 1.97% | 3 | 1 (ran to its 640-token limit without ending) |

- Differences (95% bootstrap CI over texts): uqt minus sglang-omni +0.50 (+0.07 to +1.04), excluding garbled clips
  +0.04 (-0.12 to +0.20); uqt minus vLLM-Omni -0.47 (-2.53 to +0.79), excluding garbled -0.06 (-0.24 to +0.12).
- uqt made 9 garbled clips against 3 (Fisher exact p = 0.15). Three of uqt's nine are one text with unusual spelling, scored in different
  runs; four are runaways of 25-32 s. This gap did not repeat in the larger test below.
- sglang-omni also caps a clip's length from its text. Cutting uqt's audio at that cap, offline, lowers uqt to 2.32%
  (+0.35 against sglang-omni, CI +0.02 to +0.74), so the cap explains only part of the difference.

A larger test of the same texts and servers, at 64 clients on another GB10 (`bench/gb10/garbleq.sh`, `bench/gb10/garble_stats.py`,
`results/gb10/garble`, `results/gb10/garble_stats.json`): 40 runs of 300 clips per setting = 12,000 clips, no failed requests. `uqt`
is the engine before the hang fix described below, `uqtfix` the same settings with it, and `uqtbf16` unquantized weights and KV with the fix.

| Setting | WER (95% bootstrap CI over texts) | Excluding garbled clips | Garbled clips (95% Wilson CI) | Over 25 s |
|---|---|---|---|---|
| uqt, default, fixed | 2.48% (1.85 to 3.20) | 1.89% | 20, 0.17% (0.11 to 0.26) | 3 |
| uqt, default, before the fix | 2.17% (1.76 to 2.61) | 1.94% | 24, 0.20% (0.13 to 0.30) | 4 |
| uqt, unquantized, fixed | 2.36% (1.82 to 2.99) | 1.86% | 30, 0.25% (0.18 to 0.36) | 4 |
| sglang-omni 0.2 | 2.40% (1.86 to 3.01) | 1.95% | 26, 0.22% (0.15 to 0.32) | 1 |

- No pair of settings differs in garbled clips (Fisher exact p 0.20 to 0.89), in clips over 25 s (p 0.38 to 1.0) or in overall WER
  (every interval includes zero). One text (`2094-142345-0024`) sits near the 50% line in every system; leaving it out changes nothing.
- Excluding garbled clips, unquantized uqt is 0.09 points lower than sglang-omni (0.02 to 0.16) and 0.08 lower than quantized uqt before
  the fix (0.02 to 0.14): small, and the fixed default against sglang-omni is 0.07 lower (-0.00 to 0.13), so only a hint.
- All clips of uqt's default settings and both tests: 53 garbled of 26,400, WER 2.34%, against 29 of 14,400 and 2.33% for sglang-omni
  (WER difference +0.01, CI -0.37 to +0.39; excluding garbled -0.02, CI -0.08 to +0.03).

Hang fixed on the way (`uqt/graphs.py`, `bench/gb10/cublas_ws_race.py`, `results/gb10/cublas_ws_race.log`): captured CUDA graphs shared
one cuBLAS workspace, and two graphs replayed at once on different streams could deadlock the GPU. Unquantized uqt hung that way in its
first 64-client run, and again when reproduced. With a workspace per graph pool, unquantized and default uqt both finished 40 runs of
300 clips without a hang. The micro-test hung in three of three runs without the fix and in none with it. No micro-test output was wrong in either
case. The default settings before the fix never hung in 40 runs.

Speed with and without the fix (`bench/gb10/fixspeed_stats.py`, `results/gb10/fixspeed`, four server starts each, no failed requests):
mean throughput -2.2% at 32 clients (18 passes each, permutation p = 0.10), -1.0% at 64 (4 each, p = 0.60), +0.7% at 128 (4 each,
p = 0.86), -0.4% at 256 (10 each, p = 0.90); median first byte -1.4% to -0.4% at 32-256. No cost was measured. That
box is slower than the one behind the head-to-head tables (voice heard about 345 ms at 128 clients against 316 ms, with and without the
fix), so it was used only to compare the two versions and the head-to-head numbers are unchanged.

## Competitor setups

Setup notes for both competitor stacks on sm_121 (aarch64 wheels, a one-line `num_stages` patch
for an sglang-omni Triton kernel that exceeds GB10's 99 KiB shared memory) are in
`bench/gb10/competitors/`. vLLM-Omni's fused single-GPU profile does not start on GB10 (a Triton
kernel in its MTP graph capture needs 136 KiB of shared memory), so the default 2-stage profile is
the one measured.
