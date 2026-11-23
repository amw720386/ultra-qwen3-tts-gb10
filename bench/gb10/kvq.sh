#!/bin/sh
# real-cache error, then 5 seeds x 3 formats of batched WER (interleaved), then interleaved server runs
cd /root/uqt-src; export UQT_NO_PDL=1
python -m bench.gb10.kv_real_error > /root/kv_real.log 2>&1
for seed in 1 2 3 4 5; do for kv in bf16 int8 fp8; do
  UQT_KV=$kv python -m bench.gb10.acc_batch --weights hyb --seed $seed --out /root/results/kvseed-$kv-s$seed > /root/kvseed-$kv-s$seed.log 2>&1
done; done
for rep in 1 2 3; do for kv in bf16 int8; do
  UQT_KV=$kv python -m bench.gb10.h2h_local --system uqt --concs 64,128,256,256,256 --out /root/results/kvh2h-$kv-r$rep.json > /root/kvh2h-$kv-r$rep.log 2>&1
done; done
echo ALLDONE > /root/kvq.done
