#!/bin/bash
# sglang-omni 0.2.0 (sglang 0.5.22) Qwen3-TTS server on localhost:8092
# MEM = tts_engine mem_fraction_static (package default 0.85; lowered to share the box)
source /root/venv-sgl/bin/activate
MEM=${MEM:-0.2}
export HF_HUB_OFFLINE=1
# sm_121 (GB10) has 99 KiB smem/SM; the fused predictor Triton kernel needs 112 KiB at 3 stages.
# Requires the local one-line patch in sglang_omni/utils/predictor_layers.py (see sgl_predictor_numstages.patch).
export SGLANG_OMNI_PREDICTOR_NUM_STAGES=${SGLANG_OMNI_PREDICTOR_NUM_STAGES:-2}
exec sgl-omni serve --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice   --config /root/competitors/sglang-omni/examples/configs/qwen3_tts_1_7b_customvoice.yaml   --tts_engine.engine.mem_fraction_static $MEM --host 127.0.0.1 --port 8092 "$@"
