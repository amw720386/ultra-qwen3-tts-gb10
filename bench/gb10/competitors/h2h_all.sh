#!/bin/bash
# Sequential GB10 head-to-head: one server at a time, client co-located.
cd /root/uqt-src
X='{"stream_format": "audio"}'
run_srv() { # name port health cmd...
  name=$1; port=$2; shift 2
  setsid "$@" > /root/competitors/srv-$name.log 2>&1 &
  pid=$!
  python -m bench.gb10.h2h_local --system $name --url http://127.0.0.1:$port/v1/audio/speech      --health http://127.0.0.1:$port/health --extra "$X" --out /root/results/h2h-$name.json > /root/competitors/h2h-$name.log 2>&1
  kill -TERM -$pid 2>/dev/null; sleep 15; kill -KILL -$pid 2>/dev/null; sleep 10
}
UQT_NO_PDL=1 python -m bench.gb10.h2h_local --system uqt --weights fp8 > /root/competitors/h2h-uqt.log 2>&1
sleep 10
DEPLOY=qwen3_tts GMU=0.3 run_srv vllm 8091 /root/competitors/run_vllm.sh
DEPLOY=qwen3_tts_fused_single_gpu GMU=0.5 run_srv vllm_fused 8091 /root/competitors/run_vllm.sh
MEM=0.5 run_srv sglang 8092 /root/competitors/run_sgl.sh
echo ALLDONE > /root/competitors/h2h_all.done
