#!/bin/bash
# Usage: c2c/sweep_c2c.sh FIRST_CHIP LAST_CHIP   (runs 4 chips at a time, logs in /opt/dlami/nvme/trn3ub/logs)
W=/opt/dlami/nvme/trn3ub
run() { $W/src/cluster/dk.sh python3 c2c/run_c2c.py --chip $1 --masks 1,4 --fbytes 4,64,1024,8192 \
          --steps 2,8,32 --num-exec 7 --jobs 4 --out /w/runs/c2c > $W/logs/c2c-chip$1.log 2>&1; }
for c in $(seq $1 $2); do
  run $c &
  if (( (c - $1) % 4 == 3 )); then wait; fi
done
wait
