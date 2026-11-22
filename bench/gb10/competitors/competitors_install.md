# Competitor stacks on GB10 (sm_121, aarch64, CUDA 13 driver)

Both stacks are installed in their own venvs. System python was not touched. Scripts, logs and smoke wavs are in /root/competitors/.
Smoke client: python3 /root/competitors/smoke.py PORT OUT.wav [voice] [extra-json] (stdlib only). It sends one streaming
/v1/audio/speech request (stream=true, stream_format=audio, response_format=pcm, text "The quick brown fox jumps over the lazy dog.",
voice Ryan, language English) and prints the first-byte latency (TTFA), total time, audio seconds and RMS/peak.

## vLLM-Omni: WORKS
- Versions: vllm 0.32.0 (PyPI aarch64 manylinux_2_28 wheel), vllm-omni 0.32.0, torch 2.13.0+cu130 (arch list sm_80..sm_120; the sm_120 SASS runs on sm_121),
  triton 3.7.1, flashinfer-python 0.7.0.post1, transformers 5.19.0.
- Install:
  uv venv -p 3.12 /root/venv-vllm
  VIRTUAL_ENV=/root/venv-vllm uv pip install vllm==0.32.0 vllm-omni==0.32.0
  # fix: opencv-python (non-headless) is pulled in and needs libxcb.so.1, which is missing. Engine-core spawn then dies with
  # "ImportError: libxcb.so.1: cannot open shared object file" (raised from cv2/typing). Replace it with the headless build:
  VIRTUAL_ENV=/root/venv-vllm uv pip uninstall opencv-python opencv-python-headless
  VIRTUAL_ENV=/root/venv-vllm uv pip install opencv-python-headless==5.1.0.95
- Launch: /root/competitors/run_vllm.sh  (127.0.0.1:8091). Env: DEPLOY=qwen3_tts (default, 2-stage talker->code2wav, MRv2) or
  DEPLOY=qwen3_tts_fused_single_gpu. GMU sets gpu_memory_utilization per stage (default 0.15; the yaml defaults are 0.3 per stage for 2-stage and 0.5 for fused).
  It expands to:
  HF_HUB_OFFLINE=1 /root/venv-vllm/bin/vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni     --deploy-config /root/venv-vllm/lib/python3.12/site-packages/vllm_omni/deploy/qwen3_tts.yaml     --stage-overrides '{"0": {"gpu_memory_utilization": 0.15}, "1": {"gpu_memory_utilization": 0.15}}'     --host 127.0.0.1 --port 8091
- Startup takes about 3 min (CUDA graph capture). Ready when GET /health returns 200.
- Smoke results (warm server, concurrency 1; the parent benchmark was running at the same time, so treat these as indicative only):
  - qwen3_tts (2-stage):       TTFA 0.093 s / 0.086 s, audio 3.04 s / 2.96 s, total 0.73 / 0.91 s, RMS ~2000 (non-silent)
  - qwen3_tts_fused_single_gpu (GMU=0.2): TTFA 0.097 s / 0.091 s, audio 3.36 s / 3.28 s, total 0.75 / 0.71 s, RMS ~3000 (non-silent)

## sglang-omni: WORKS with a one-line local patch
- Versions: sglang-omni 0.2.0 (PyPI), sglang 0.5.22, sglang-kernel 0.4.9, torch 2.14.1+cu130, flashinfer-python 0.7.1,
  flash-attn-4 4.0.0b40, nvidia-cutlass-dsl 4.8.0, transformers 5.19.0, qwen-tts 0.1.1. Config file from the sglang-omni main clone at
  /root/competitors/sglang-omni/examples/configs/qwen3_tts_1_7b_customvoice.yaml (it only sets config_cls + model_path).
- Install:
  uv venv -p 3.12 /root/venv-sgl
  echo "pynini; python_version < '3'" > sgl_overrides.txt   # drop pynini: it has no aarch64 wheel, and the sdist build fails (needs OpenFST fst/util.h).
                                                           # It comes in through nemo-text-processing and is not used by Qwen3-TTS.
  VIRTUAL_ENV=/root/venv-sgl uv pip install --prerelease=allow --override sgl_overrides.txt sglang-omni==0.2.0
  VIRTUAL_ENV=/root/venv-sgl uv pip install --no-deps sox einops qwen-tts==0.1.1   # per the docs; /usr/bin/sox already exists
- Qwen3-TTS backend: SGLANG_OMNI_QTTS_BACKEND=default. The CuTeDSL kernels are used only on sm_90a/sm_100a/sm_103a (cutedsl/runtime.py),
  so sm_121 falls back to the default path automatically.
- Blocker without the patch: startup crashes while capturing the predictor graph (sglang_model.capture_predictor_graph ->
  utils/predictor_layers.py norm_qkv_rope_store_kernel, launched with num_stages=3):
    triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 114688, Hardware limit: 101376.
  sm_12x has 99 KiB of shared memory per SM, versus 227 KiB on H100/B200.
- Patch (in the venv only; backup at /root/competitors/predictor_layers.py.orig, diff in sgl_predictor_numstages.patch):
    predictor_layers.py:89  NUM_STAGES = 3  ->  NUM_STAGES = int(os.environ.get("SGLANG_OMNI_PREDICTOR_NUM_STAGES", "3"))
  run_sgl.sh sets SGLANG_OMNI_PREDICTOR_NUM_STAGES=2. This changes only software pipelining depth, not the math.
- Launch: /root/competitors/run_sgl.sh  (127.0.0.1:8092). MEM sets tts_engine mem_fraction_static (default 0.2; the package default is 0.85,
  which would claim about 100 GB of unified memory). It expands to:
  HF_HUB_OFFLINE=1 SGLANG_OMNI_PREDICTOR_NUM_STAGES=2 /root/venv-sgl/bin/sgl-omni serve     --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice     --config /root/competitors/sglang-omni/examples/configs/qwen3_tts_1_7b_customvoice.yaml     --tts_engine.engine.mem_fraction_static 0.2 --host 127.0.0.1 --port 8092
- Startup takes about 3-4 min (inductor compile workers plus graph capture). Ready when GET /health returns 200. Voices are lowercase (ryan, aiden, ...);
  "Ryan" works.
- Caveat: on unified memory, the incremental-codec CUDA graph capture checks cudaMemGetInfo free memory against a 3 GiB headroom. While the parent
  benchmark was using memory, it logged "CaptureFailure: free VRAM 2.67GiB is below 3.00GiB headroom" and disabled the warm0 vocoder graph
  ("cold 8 captured, window 28 captured, warm0 off"). On a quiet box this should capture fully. Check the log line
  "Qwen3-TTS vocoder CUDA graphs:" before benchmarking.
- Smoke results (warm server, concurrency 1; noisy box): TTFA 0.075 / 0.121 / 0.080 s, audio 4.48 / 3.44 / 2.80 s, total 1.72 / 1.37 / 1.10 s,
  RMS 1750-2600 (non-silent).

## Notes
- Plain sglang was not needed (sglang-omni pulls sglang 0.5.22).
- No builds from source were needed. Neither stack compiled CUDA code; only Triton/inductor JIT ran.
- All servers I started are stopped. Disk: venvs 8.5 GB (vllm) + 6.7 GB (sgl).
