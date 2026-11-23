#!/bin/bash
# Competitor reruns with the listener metrics (heard_ms, late_ms), three repeats per load, after any A/B run
cd /root/uqt-src
while pgrep -f "[b]ench.gb10.ab_live" > /dev/null; do sleep 20; done
X='{"stream_format": "audio"}'
run_srv() { # name port cmd...
  name=$1; port=$2; shift 2
  setsid "$@" > /root/competitors/srv-$name.log 2>&1 &
  pid=$!
  python -m bench.gb10.h2h_local --system $name --url http://127.0.0.1:$port/v1/audio/speech --health http://127.0.0.1:$port/health --extra "$X" --concs 1,8,32,64,128,256,8,32,64,128,256,8,32,64 --out /root/results/comp-$name.json > /root/comp-$name.log 2>&1
  kill -TERM -$pid 2>/dev/null; sleep 15; kill -KILL -$pid 2>/dev/null; sleep 10
}
MEM=0.5 run_srv sglang 8092 /root/competitors/run_sgl.sh
DEPLOY=qwen3_tts GMU=0.3 run_srv vllm 8091 /root/competitors/run_vllm.sh
echo ALLDONE > /root/compq.done
