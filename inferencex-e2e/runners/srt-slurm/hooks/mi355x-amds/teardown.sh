#!/usr/bin/env bash
# Test only (branch thshan/glm52-pd-test-c48): release the free hugetlb pages a "_hugetlb"
# recipe reserved in setup.sh. Pages still in use or reserved are kept.
set -eo pipefail
[[ "${SRT_RECIPE:-}" == *_hugetlb* ]] || exit 0
read_value() { awk -v key="$1:" '$1 == key {print $2}' /proc/meminfo; }
total=$(read_value HugePages_Total)
free=$(read_value HugePages_Free)
reserved=$(read_value HugePages_Rsvd)
target=$((total - free + reserved))
if (( target < total )); then
    printf '%s\n' "$target" | sudo -n tee /proc/sys/vm/nr_hugepages >/dev/null
fi
grep -E '^(HugePages_Total|HugePages_Free|HugePages_Rsvd):' /proc/meminfo
