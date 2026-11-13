#!/bin/bash
# Which DMA queues carry a same-core exchange between chip c and c^CM. Usage on a work node: p2p/xchip_routes.sh CM...
# Re-captures the 9-step HBM all_gather point of p2p/xchip.sh with every rank traced (p2p/dissect.py) and prints
# each rank's queue families (p2p/dissect_ranks.py) to logs/p2p/routes-c<CM>.txt.
W=/opt/dlami/nvme/trn3ub; D=$W/src/cluster/dk.sh; O=/w/runs/p2p/xchip
while pgrep -f "xchip.sh (core|chips)" > /dev/null; do sleep 20; done
for cm in "$@"; do
  r=16; while (( r <= 8 * cm )); do r=$((2 * r)); done
  [ -f $W/runs/p2p/xchip/ag_hbm-c${cm}k0/f4/n9/kernel.neff ] || $W/src/p2p/xchip.sh one ag_hbm $cm 0 0
  $D python3 p2p/dissect.py $O/ag_hbm-c${cm}k0/f4/n9 /w/runs/p2p/xroutes/c$cm 0-$((r - 1)) $r > $W/logs/p2p/routes-c$cm.log 2>&1
  $D python3 p2p/dissect_ranks.py /w/runs/p2p/xroutes/c$cm 9 > $W/logs/p2p/routes-c$cm.txt 2>&1
done
echo routes-done > $W/logs/p2p/routes.out
