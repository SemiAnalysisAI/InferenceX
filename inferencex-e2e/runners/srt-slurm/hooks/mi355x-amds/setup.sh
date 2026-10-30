#!/usr/bin/env bash
set -eo pipefail

bash "$(dirname "${BASH_SOURCE[0]}")/check-rdma.sh"

# Preserve the legacy bare-process GPU drain gate. Slurm owns the node, but a
# process left outside the prior job's container can still retain VRAM and make
# the next model load fail much later with a misleading OOM.
# shellcheck source=runners/srt-slurm/hooks/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
wait_for_amd_gpu_clean

# Some MI355X experiments reserve large 2 MiB HugeTLB pools and leave the
# reservation behind after their Slurm allocation exits. Those free hugepages
# remain unavailable to ordinary host allocations, which can make a later
# unchanged SGLang HiCache recipe fail even on a 3 TiB node. Reclaim only free
# pages; pages currently used or reserved by host services are preserved.
meminfo=/proc/meminfo
nr_hugepages=/proc/sys/vm/nr_hugepages

read_hugepage_value() {
    local key="$1"
    awk -v key="${key}:" '$1 == key {print $2}' "$meminfo"
}

total=$(read_hugepage_value HugePages_Total)
free=$(read_hugepage_value HugePages_Free)
reserved=$(read_hugepage_value HugePages_Rsvd)
used=$((total - free))
target=$((used + reserved))

echo "MI355X host memory before preparation:"
grep -E '^(MemFree|MemAvailable|HugePages_Total|HugePages_Free|HugePages_Rsvd|HugePages_Surp|Hugetlb):' "$meminfo"

if (( target < total )); then
    printf '%s\n' "$target" | sudo -n tee "$nr_hugepages" >/dev/null
fi

# hipHostMalloc pins host memory on the calling GPU's NUMA node, and srtctl gives a
# partial-node worker GPUs 0..TP-1, all on node 0, so a DRAM KV tier and the weight
# staging buffers must fit in node 0's free pages. Pinning while reclaiming page
# cache stalls ranks minutes apart and past the engine's startup barrier.
if [[ "${KV_OFFLOADING:-}" == dram ]]; then
    sync
    echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
fi

after_total=$(read_hugepage_value HugePages_Total)
after_free=$(read_hugepage_value HugePages_Free)
echo "MI355X host memory after preparation (KV_OFFLOADING=${KV_OFFLOADING:-unset}):"
grep -E '^(MemFree|MemAvailable|HugePages_Total|HugePages_Free|HugePages_Rsvd|HugePages_Surp|Hugetlb):' "$meminfo"
grep -h ' MemFree:' /sys/devices/system/node/node*/meminfo

if (( after_total - after_free < used )); then
    echo "Host preparation released hugepages that were in use" >&2
    exit 1
fi
if (( after_free > reserved )); then
    echo "Host preparation could not reclaim all unused hugepages" >&2
    exit 1
fi

# A recipe whose head-node service registers NUMA-bound 2 MiB HugeTLB memory with
# the NIC (a Mooncake Store owner) sets HEAD_NODE_HUGEPAGES_PER_NUMA; unset, no
# pages are reserved. The container cannot grow the pool, so it is grown here, 2%
# above the pages host services hold, and each NUMA node must end with its share
# free. The recipe's teardown releases them (HUGEPAGES_TARGET=0 hugepage_sanity.sh).
pages_per_numa="${HEAD_NODE_HUGEPAGES_PER_NUMA:-}"
if [[ -n "$pages_per_numa" ]]; then
    launch_ip="${SLURM_LAUNCH_NODE_IPADDR:?srun did not set SLURM_LAUNCH_NODE_IPADDR}"
    if [[ " $(hostname -I) " == *" $launch_ip "* ]]; then
        numa_nodes=(/sys/devices/system/node/node[0-9]*)
        echo "[hugepage] $(hostname -s) is the head node; reserving $pages_per_numa 2 MiB pages on each of ${#numa_nodes[@]} NUMA nodes"
        held=$(awk '/^HugePages_Total:/ {total = $2} /^HugePages_Free:/ {free = $2} /^HugePages_Rsvd:/ {rsvd = $2} END {print total - free + rsvd}' "$meminfo")
        HUGEPAGES_TARGET=$(( held + pages_per_numa * ${#numa_nodes[@]} * 102 / 100 )) HUGEPAGE_GROW_TOLERANCE_PCT=1 \
            bash "$(dirname "${BASH_SOURCE[0]}")/hugepage_sanity.sh"
        for numa_node in "${numa_nodes[@]}"; do
            numa_free=$(cat "$numa_node/hugepages/hugepages-2048kB/free_hugepages")
            echo "[hugepage] ${numa_node##*/}: $numa_free free 2 MiB pages, $pages_per_numa needed"
            if (( numa_free < pages_per_numa )); then
                exit 1
            fi
        done
    fi
fi
