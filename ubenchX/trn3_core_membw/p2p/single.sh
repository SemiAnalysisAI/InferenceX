#!/bin/bash
# Usage on a work node: p2p/single.sh   (data in runs/p2p/single, logs in logs/p2p/single-*.log)
# One pair alone (2 ranks; NEURON_RT_VISIBLE_CORES must be a contiguous range, so only adjacent cores):
#   lnc   cores 32-33 (chip 4, cores 0-1: LNC pair)
#   die   cores 41-42 (chip 5, cores 1-2: same die, other pair)
#   xchip cores 55-56 (chip 6 core 7 -> chip 7 core 0)
# plus a repeat of the 8-pair SBUF all_gather chip-to-chip run on chips 2-3 (cores 16-31).
W=/opt/dlami/nvme/trn3ub; D=$W/src/cluster/dk.sh; O=/w/runs/p2p/single; mkdir -p $W/logs/p2p
one() {  # route first_core
  for k in ag_hbm cp_hbm cpi_sbuf ag_sbuf; do
    fb=4,8192,65536; [ $k = ag_sbuf ] && fb=4,8192
    $D python3 p2p/probe.py --kernel $k --ranks 2 --cores $2-$(($2 + 1)) --mask 1 --fbytes $fb --steps 1,9,33 \
      --num-exec 5 --out $O/$k-$1 >> $W/logs/p2p/single-$1.log 2>&1
  done
}
one lnc 32 & one die 41 & one xchip 55 &
$D python3 p2p/probe.py --kernel ag_sbuf --ranks 16 --cores 16-31 --mask 8 --fbytes 4 --steps 1,9,33 --num-exec 5 \
  --out /w/runs/p2p/sweep-repeat/ag_sbuf-x8 > $W/logs/p2p/repeat-ag_sbuf-x8.log 2>&1 &
wait; echo done
