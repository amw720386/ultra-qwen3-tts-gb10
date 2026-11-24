#!/bin/bash
# after the competitor reruns: uqt with default settings on the same protocol, then WER through the server
cd /root/uqt-src
while [ ! -f /root/compq.done ]; do sleep 20; done
UQT_STATS=1 UQT_NO_PDL=1 python -m bench.gb10.h2h_local --system uqt --concs 1,8,32,64,128,256,8,32,64,128,256,8,32,64 --out /root/results/final-uqt.json > /root/final-uqt.log 2>&1
echo '{"new": {}}' > /tmp/v.json
UQT_STATS=1 python -m bench.gb10.ab_live --variants /tmp/v.json --concs 64 --rounds 1 --swer new --swer-concs 64,128,256 --out /root/results/final-swer.json > /root/final-swer.log 2>&1
echo ALLDONE > /root/finalq.done
