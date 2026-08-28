#!/usr/bin/env bash
set -euo pipefail

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
grep -E '^(MemAvailable|HugePages_Total|HugePages_Free|HugePages_Rsvd|HugePages_Surp|Hugetlb):' "$meminfo"

if (( target < total )); then
    printf '%s\n' "$target" | sudo -n tee "$nr_hugepages" >/dev/null
fi

after_total=$(read_hugepage_value HugePages_Total)
after_free=$(read_hugepage_value HugePages_Free)
echo "MI355X host memory after preparation:"
grep -E '^(MemAvailable|HugePages_Total|HugePages_Free|HugePages_Rsvd|HugePages_Surp|Hugetlb):' "$meminfo"

if (( after_total - after_free < used )); then
    echo "Host preparation released hugepages that were in use" >&2
    exit 1
fi
if (( after_free > reserved )); then
    echo "Host preparation could not reclaim all unused hugepages" >&2
    exit 1
fi
