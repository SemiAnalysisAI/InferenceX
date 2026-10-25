#!/usr/bin/env bash
set -euo pipefail
node=$(hostname -s)
{
    date -u --iso-8601=seconds
    printf 'MEMLOCK_SOFT '; ulimit -Sl
    printf 'MEMLOCK_HARD '; ulimit -Hl
    grep -E 'MemAvailable|HugePages|Hugepagesize' /proc/meminfo
    cat /proc/self/limits
    for device in /sys/class/infiniband/*; do
        [[ -e $device ]] || continue
        ibv_devinfo -v -d "${device##*/}" 2>&1 || true
    done
} > "/logs/pool-resources-${node}.txt"
cat > "/logs/perf-analysis-${node}.md" <<'EOF'
Diagnostic only. Original 1799GB host pool; two-node byte-checked write/read gate before model loading.
SMG manual/min_load, session key X-Dynamo-Session-ID, idle affinity 900s.
P64/D128, graph512, allocator0.92, pinned image and engine preserved.
One warmup request per lane, 300s profile; not a formal Pareto point.
TTFT/ITL/throughput/SLO are unavailable if the pre-model gate fails; inspect pool gate JSON and first native error.
Source: pool-resources, umbp daemon log and pool-gate outputs in this directory.
DSight was not used: native registration errors and byte equality are the acceptance evidence.
EOF
cp -n "/logs/perf-analysis-${node}.md" /logs/perf-analysis.md
bash /configs/k3-umbp-pool.sh
timeout --signal=TERM --kill-after=10s 360s python3 /configs/umbp_crossnode_gate.py > "/logs/pool-gate-${node}.log" 2>&1
