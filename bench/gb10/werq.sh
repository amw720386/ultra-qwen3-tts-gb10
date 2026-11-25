#!/bin/bash
# WER through each server on the same 300 texts: uqt (default settings), vLLM-Omni, sglang-omni, one at a time,
# 4 runs at 32 and 4 at 64 clients each. Generates everything first, then scores it all with Whisper large-v3.
cd /root/uqt-wer; export UQT_NO_PDL=1
O=/root/results/wer3; mkdir -p $O
X='{"stream_format": "audio"}'
gen() { # name port extra
  for i in $(seq 1 360); do curl -sf http://127.0.0.1:$2/health > /dev/null && break; sleep 5; done
  python -m bench.gb10.server_wer --url http://127.0.0.1:$2/v1/audio/speech --extra "$3" --conc 16 --n 64 --gen-only --out /root/tmp-warm-$1
  for r in 0 1 2 3; do for c in 32 64; do
    python -m bench.gb10.server_wer --url http://127.0.0.1:$2/v1/audio/speech --extra "$3" --conc $c --gen-only --out $O/$1-c$c-r$r
  done; done
}
run_srv() { # name port extra cmd...
  name=$1; port=$2; x=$3; shift 3
  setsid "$@" > /root/srv-wer-$name.log 2>&1 &
  pid=$!
  gen $name $port "$x"
  kill -TERM -- -$pid 2>/dev/null; sleep 15; kill -KILL -- -$pid 2>/dev/null; sleep 10
}
uqt_srv() { /root/uqt-gw -sock /tmp/uqt.sock -addr 127.0.0.1:8091 & exec python -m uqt.engine_proc --sock /tmp/uqt.sock --max-slots 256 --max-len 512 --weights hyb; }
export -f uqt_srv
run_srv uqt 8091 '{}' bash -c uqt_srv
MEM=0.5 run_srv sglang 8092 "$X" /root/competitors/run_sgl.sh
DEPLOY=qwen3_tts GMU=0.3 run_srv vllm 8091 "$X" /root/competitors/run_vllm.sh
for d in $O/*; do
  python -m bench.gb10.server_wer --score-only --conc $(basename $d | sed 's/.*-c\([0-9]*\)-.*/\1/') --out $d
done
echo ALLDONE > /root/werq.done
