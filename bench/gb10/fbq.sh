#!/bin/bash
# first-byte switches: codec equality, then live A/B
cd /root/uqt-fb; export UQT_NO_PDL=1 UQT_STATS=1 UQT_WARM_KS=1,2
python -m bench.gb10.f1_equal > /root/fb-equal.log 2>&1
python -m bench.gb10.ab_live --variants bench/gb10/variants-firstbyte.json --concs 32,64 --rounds 3 --out /root/results/ab-firstbyte.json > /root/fb-ab.log 2>&1
echo ALLDONE > /root/fbq.done
