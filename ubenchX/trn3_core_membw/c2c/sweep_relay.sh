#!/bin/bash
# Usage: c2c/sweep_relay.sh FIRST_CHIP LAST_CHIP   (all chips in the range at once)
W=/opt/dlami/nvme/trn3ub
for c in $(seq $1 $2); do
  $W/src/cluster/dk.sh python3 c2c/run_relay.py --chip $c --jobs 3 --out /w/runs/relay > $W/logs/relay-chip$c.log 2>&1 &
done
wait
