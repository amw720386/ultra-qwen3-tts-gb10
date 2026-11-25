#!/bin/bash
# Garbled-clip rate, 12,000 clips per setting (40 runs of the same 300 texts at 64 clients), one server at a time:
# uqt defaults, uqt without any quantization (bf16 weights and KV), sglang-omni. Generation first, scoring last.
cd /root/uqt-wer; export UQT_NO_PDL=1
O=/root/results/garble; mkdir -p $O; N=${N:-40}
gen() { # name port extra
  for i in $(seq 1 720); do curl -sf http://127.0.0.1:$2/health > /dev/null && break; sleep 5; done
  python -m bench.gb10.server_wer --url http://127.0.0.1:$2/v1/audio/speech --extra "$3" --conc 16 --n 64 --gen-only --out /root/tmp-warm-$1
  for r in $(seq 0 $((N-1))); do
    python -m bench.gb10.server_wer --url http://127.0.0.1:$2/v1/audio/speech --extra "$3" --conc 64 --gen-only --out $O/$1-c64-r$r
  done
}
run_srv() { # name port extra cmd...
  name=$1; port=$2; x=$3; shift 3
  setsid "$@" > /root/srv-garble-$name.log 2>&1 &
  pid=$!
  gen $name $port "$x"
  kill -TERM -- -$pid 2>/dev/null; sleep 15; kill -KILL -- -$pid 2>/dev/null; sleep 10
}
uqt_srv() { /root/uqt-gw -sock /tmp/uqt.sock -addr 127.0.0.1:8091 & exec python -m uqt.engine_proc --sock /tmp/uqt.sock --max-slots 256 --max-len 512 --weights $W; }
export -f uqt_srv
W=hyb run_srv uqt 8091 '{}' env W=hyb bash -c uqt_srv
W=bf16 UQT_KV=bf16 run_srv uqtbf16 8091 '{}' env W=bf16 UQT_KV=bf16 bash -c uqt_srv
MEM=0.5 run_srv sglang 8092 '{"stream_format": "audio"}' /root/competitors/run_sgl.sh
python -m bench.gb10.score_dirs $O
echo ALLDONE > /root/garbleq.done
