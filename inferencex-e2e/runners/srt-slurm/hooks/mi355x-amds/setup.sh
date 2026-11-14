#!/usr/bin/env bash
set -eo pipefail

bash "$(dirname "${BASH_SOURCE[0]}")/check-rdma.sh"

# Preserve the legacy bare-process GPU drain gate. Slurm owns the node, but a
# process left outside the prior job's container can still retain VRAM and make
# the next model load fail much later with a misleading OOM.
# shellcheck source=runners/srt-slurm/hooks/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
# After a vLLM job ends, a node can hold 3-7% VRAM per GPU with no process attached
# for more than 30 minutes before it drains. vLLM then fails its startup free-memory
# check, and SGLang silently caps max_running_requests below the benchmark's
# concurrency. Clean nodes read 0%, so wait for that.
wait_for_amd_gpu_clean 2 60

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

# Test only (branch thshan/glm52-pd-test-c48): a recipe whose name contains "_hugetlb"
# backs its HiCache host pools with 2 MiB hugetlb pages (SGLANG_HUGEPAGE_MODE=required),
# which memory compaction does not migrate. 710,000 pages (1,387 GiB) cover four PP ranks'
# KV + DSA indexer pools (~1,370 GiB). Reserved after the reclaim checks above;
# teardown.sh and every later job's setup release whatever is left free.
if [[ "${SRT_RECIPE:-}" == *_hugetlb* ]]; then
    want=$((used + reserved + 710000))
    sync
    echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
    for attempt in 1 2 3; do
        printf '%s\n' "$want" | sudo -n tee "$nr_hugepages" >/dev/null
        got=$(read_hugepage_value HugePages_Total)
        echo "hugetlb reservation attempt ${attempt}: ${got} of ${want} 2 MiB pages"
        (( got >= want )) && break
        sleep 20
    done
    grep -h -E 'HugePages_(Total|Free)' /sys/devices/system/node/node*/meminfo
    if (( got < want )); then
        printf '%s\n' "$((used + reserved))" | sudo -n tee "$nr_hugepages" >/dev/null
        echo "Could not reserve ${want} hugetlb pages (got ${got})" >&2
        exit 1
    fi
fi
