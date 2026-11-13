#!/bin/bash
# Usage on a work node: membw/pinned.sh CHIP     (logs in /opt/dlami/nvme/trn3ub/logs/pinned-c<CHIP>-*.log)
# Every mem-bw series on ONE chip. The first sweeps sharded sizes across chips whose plateaus differ by
# ~8% (membw/chip_survey.py), which drew a sawtooth. Here the shards run on the chip's four HBM stacks at
# once (one core or logical core per stack), so every point of a series comes from the same chip.
#   core:  LNC=1, cores 8c + 2s (s = stack 0..3), sizes 8 B .. 8 GiB
#   stack: LNC=2, logical cores 4c + s,           sizes 8 B .. 8 GiB
#   chip:  LNC=2, all four logical cores at once (device_sweep.py), sizes 32 B .. 16 GiB
W=/opt/dlami/nvme/trn3ub; D=$W/src/cluster/dk.sh; c=$1; L=$W/logs/pinned-c$c
for s in 0 1 2 3; do
  $D python3 membw/sweep.py --tag core-c$c --kernels copy_h2h_static --sizes fine --max-bytes 8G \
    --core $((8 * c + 2 * s)) --shard $s/4 --jobs 12 > $L-core-s$s.log 2>&1 &
done; wait
for s in 0 1 2 3; do
  $D python3 membw/sweep.py --tag lnc2-c$c --kernels copy_h2h_static_spmd --lnc 2 --sizes fine --max-bytes 8G \
    --core $((4 * c + s)) --shard $s/4 --jobs 12 > $L-stack-s$s.log 2>&1 &
done; wait
$D python3 membw/device_sweep.py --tag device-c$c --device $c --sizes fine --max-bytes 16G --min-run-s 4 \
  --jobs 24 > $L-chip.log 2>&1
echo done
