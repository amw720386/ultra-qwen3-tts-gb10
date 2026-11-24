#!/bin/bash
# one-frame steps and one-frame first chunks, alone and combined
cd /root/uqt-fb; export UQT_NO_PDL=1 UQT_STATS=1 UQT_WARM_KS=1,2
python -m bench.gb10.ab_live --variants bench/gb10/variants-firstbyte2.json --concs 16,32,64,96,128 --rounds 3 --out /root/results/ab-firstbyte2.json > /root/fb-ab2.log 2>&1
echo ALLDONE > /root/fbq.done
