#!/bin/bash
# sendrecv 512 B on each LNC pair of chip 0 (logical cores 0-3 at LNC=2), one after another.
W=/opt/dlami/nvme/trn3ub; D=$W/src/cluster/dk.sh
for lc in 0 1 2 3; do
  $D python3 sendrecv/run_sendrecv.py --kernels xchg_dma,xchg_gpsimd --fbytes 4 --core $lc --out /w/runs/sendrecv-pairs/lc$lc > $W/logs/sendrecv-pairs-lc$lc.log 2>&1
done
echo done > $W/logs/sendrecv-pairs.out
