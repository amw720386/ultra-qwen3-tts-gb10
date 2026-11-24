#!/bin/bash
# final head-to-head after the first-byte changes: uqt with default settings, then sglang-omni again at the close loads
cd /root/uqt-fb; export UQT_NO_PDL=1
UQT_STATS=1 python -m bench.gb10.h2h_local --system uqt --concs 1,8,16,32,64,128,256,8,16,32,64,128,256,8,16,32,64 --out /root/results/final2-uqt.json > /root/final2-uqt.log 2>&1
sleep 10
MEM=0.5 setsid /root/competitors/run_sgl.sh > /root/competitors/srv-sglang2.log 2>&1 &
pid=$!
python -m bench.gb10.h2h_local --system sglang --url http://127.0.0.1:8092/v1/audio/speech --health http://127.0.0.1:8092/health --extra '{"stream_format": "audio"}' --concs 8,16,32,64,8,16,32,64,8,16,32,64 --out /root/results/comp2-sglang.json > /root/comp2-sglang.log 2>&1
kill -TERM -- -$pid 2>/dev/null; sleep 10; kill -KILL -- -$pid 2>/dev/null
echo ALLDONE > /root/fbq.done
