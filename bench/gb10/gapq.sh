#!/bin/sh
# playback gaps with and without deadline-aware chunk delivery: alternating servers
cd /root/uqt-src; export UQT_NO_PDL=1 UQT_STATS=1
while pgrep -f "[b]ench.gb10.h2h_local" > /dev/null; do sleep 10; done
for rep in 1 2 3; do for dl in 1 0; do
  UQT_DEADLINE=$dl python -m bench.gb10.h2h_local --system uqt --concs 32,64,96,128,160,192,256 --out /root/results/gap-dl$dl-r$rep.json > /root/gap-dl$dl-r$rep.log 2>&1
done; done
echo ALLDONE > /root/gapq.done
