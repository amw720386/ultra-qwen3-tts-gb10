#!/bin/bash
# vLLM-Omni 0.32.0 Qwen3-TTS server on localhost:8091
# DEPLOY=qwen3_tts (default, 2-stage talker->code2wav) or qwen3_tts_fused_single_gpu
# GMU = gpu_memory_utilization per stage (default 0.15 to share the box)
source /root/venv-vllm/bin/activate
DEPLOY=${DEPLOY:-qwen3_tts}
GMU=${GMU:-0.15}
CFG=/root/venv-vllm/lib/python3.12/site-packages/vllm_omni/deploy/${DEPLOY}.yaml
if [ "$DEPLOY" = qwen3_tts_fused_single_gpu ]; then OV='{"0": {"gpu_memory_utilization": '$GMU'}}'; else OV='{"0": {"gpu_memory_utilization": '$GMU'}, "1": {"gpu_memory_utilization": '$GMU'}}'; fi
export HF_HUB_OFFLINE=1
exec vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni --deploy-config "$CFG" --stage-overrides "$OV" --host 127.0.0.1 --port 8091 "$@"
