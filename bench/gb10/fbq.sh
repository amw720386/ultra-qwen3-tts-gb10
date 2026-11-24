#!/bin/bash
# chosen first-byte settings against the current defaults, then interleaved server WER
cd /root/uqt-fb; export UQT_NO_PDL=1 UQT_STATS=1 UQT_WARM_KS=1,2
python -m bench.gb10.ab_live --variants bench/gb10/variants-firstbyte4.json --concs 1,8,16,32,48,64,80,96,128,256 --rounds 4 --swer base,new --swer-concs 32,64 --swer-reps 4 --out /root/results/ab-firstbyte4.json > /root/fb-ab4.log 2>&1
echo ALLDONE > /root/fbq.done
