#!/bin/sh
# 5 seeds x 3 KV formats of batched WER (formats in parallel: startup is CPU compile, and WER does not
# depend on speed), then interleaved BF16/INT8 server runs with the GPU to themselves
cd /root/uqt-src; export UQT_NO_PDL=1
for kv in bf16 int8 fp8; do
  UQT_KV=$kv python -m bench.gb10.acc_batch --weights hyb --seeds 1,2,3,4,5 --out /root/results/kvseed-$kv > /root/kvseed-$kv.log 2>&1 &
done
wait
echo WERDONE > /root/kvwer.done
for rep in 1 2 3; do for kv in bf16 int8; do
  UQT_KV=$kv python -m bench.gb10.h2h_local --system uqt --concs 64,128,256,256,256 --out /root/results/kvh2h-$kv-r$rep.json > /root/kvh2h-$kv-r$rep.log 2>&1
done; done
echo ALLDONE > /root/kvq.done
