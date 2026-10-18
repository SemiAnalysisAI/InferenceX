#!/usr/bin/env bash
# X1-LP startup diagnostic (experiment branch only, not a benchmark): record what the Mooncake registration ENOMEM
# depends on, run the stock setup, then stop before the engine starts. Writes /logs/x1lp_mooncake_diag.txt.
set -eo pipefail
here="$(dirname "${BASH_SOURCE[0]}")"
out=/logs/x1lp_mooncake_diag.txt
config="${MOONCAKE_CONFIG_PATH:-/logs/mooncake_store_config.json}"
{
    echo "== host"; hostname; date -u +%FT%TZ; uname -r; nproc
    echo "== limits of this process (inherited by the engine)"; cat /proc/self/limits
    echo "== meminfo"
    grep -E '^(MemTotal|MemFree|MemAvailable|Cached|Mlocked|Unevictable|HugePages_Total|HugePages_Free|Hugepagesize|CommitLimit|Committed_AS):' /proc/meminfo
    echo "== numa"; ls -d /sys/devices/system/node/node* 2>/dev/null
    for node in /sys/devices/system/node/node*; do echo "$node: $(grep -E 'MemTotal|MemFree' "$node/meminfo" | tr -s ' ' | tr '\n' ' ')"; done
    echo "== cgroup"; cat /proc/self/cgroup
    for name in memory.max memory.high memory.current memory.peak memory.events memory.stat memory.limit_in_bytes memory.usage_in_bytes; do
        for path in $(find /sys/fs/cgroup -maxdepth 4 -name "$name" 2>/dev/null | head -3); do
            if [ "$name" = memory.stat ]; then echo "$path: $(grep -E '^(anon|file|unevictable|shmem) ' "$path" | tr '\n' ' ')"; else echo "$path: $(tr '\n' ' ' < "$path")"; fi
        done
    done
    echo "== rdma devices"
    for device in /sys/class/infiniband/*; do
        echo "$(basename "$device"): state=$(cat "$device/ports/1/state" 2>/dev/null) rate=$(cat "$device/ports/1/rate" 2>/dev/null) link=$(cat "$device/ports/1/link_layer" 2>/dev/null) numa=$(cat "$device/device/numa_node" 2>/dev/null)"
    done
    echo "== ibv_devinfo (memory registration capabilities)"
    if command -v ibv_devinfo >/dev/null; then
        ibv_devinfo -v 2>&1 | grep -E 'hca_id|max_mr_size|max_mr:|page_size_cap|max_pd:|max_qp:|max_mw:|fw_ver' | head -120
    else
        echo "ibv_devinfo not in the image"
    fi
    echo "== vm"; for f in /proc/sys/vm/max_map_count /proc/sys/vm/overcommit_memory /proc/sys/vm/nr_hugepages; do echo "$f: $(cat "$f")"; done
    echo "== selected environment (prefix-filtered; no tokens)"
    env | grep -E '^(MC_|MOONCAKE_|TOTAL_CPU_DRAM|SLURM_JOB_ID=|SLURM_MEM|SLURM_CPUS_ON_NODE=|CUDA_VISIBLE_DEVICES=)' | grep -viE 'token|secret|key' || true
    echo "== mooncake store config before rail selection"; cat "$config" 2>&1 || true
} > "$out" 2>&1 || true
bash "$here/kimik3-b300-mooncake.sh"
{ echo; echo "== mooncake store config after rail selection"; cat "$config" 2>&1 || true; } >> "$out"
cat "$out"
echo "x1lp diagnostic: stopping before the engine starts (startup diagnostic, not a benchmark result)" >&2
exit 97
