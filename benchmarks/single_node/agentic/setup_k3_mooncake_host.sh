#!/usr/bin/env bash
# Reserve Mooncake's 2 MiB hugepage pool without starving ROCm/model loading.
set -euo pipefail

budget_gb=${1:?usage: setup_k3_mooncake_host.sh TOTAL_CPU_DRAM_GB}
headroom_gb=${MOONCAKE_HOST_HEADROOM_GB:-768}

mem_total_gb=$(awk '/MemTotal:/ {print int($2 / 1024 / 1024)}' /proc/meminfo)
total_pages=$(awk '/HugePages_Total:/ {print $2}' /proc/meminfo)
free_pages=$(awk '/HugePages_Free:/ {print $2}' /proc/meminfo)
used_pages=$((total_pages - free_pages))

# Mooncake gets the requested aggregate capacity plus 10% registration slack.
# Preserve already-used hugepages when calculating the host-wide target.
need_free_pages=$((budget_gb * 1024 / 2 * 10 / 9 + 2048))
want_pages=$((used_pages + need_free_pages))
want_gb=$((want_pages * 2 / 1024))

echo "Mooncake host memory plan: total=${mem_total_gb}GB budget=${budget_gb}GB hugepages=${want_gb}GB headroom=${headroom_gb}GB"
if [ $((want_gb + headroom_gb)) -gt "$mem_total_gb" ]; then
    echo "Error: hugepage plan would starve normal host memory: ${want_gb}+${headroom_gb}>${mem_total_gb} GB" >&2
    exit 1
fi
if [ "${MOONCAKE_HUGEPAGE_DRY_RUN:-0}" = "1" ]; then
    echo "Dry run: would reserve $want_pages hugepages"
    exit 0
fi

set_nr_hugepages() {
    local target=$1
    if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        echo "$target" | sudo -n tee /proc/sys/vm/nr_hugepages >/dev/null || true
    else
        echo "$target" > /proc/sys/vm/nr_hugepages 2>/dev/null || true
    fi
}

have_pages=$(cat /proc/sys/vm/nr_hugepages)
# Grow or shrink to the planned pool. Only growing left leftover exclusive-node
# reservations (~2.6 TiB on g11) and starved MemAvailable below 768 GB
# (CI 32770509531, 32771343534).
if [ "$want_pages" -ne "$have_pages" ]; then
    echo "hugepage pool have=$have_pages want=$want_pages used=$used_pages; setting exact target"
    set_nr_hugepages "$want_pages"
fi

now_pages=$(cat /proc/sys/vm/nr_hugepages)
if [ "$now_pages" -ne "$want_pages" ] && command -v docker >/dev/null 2>&1; then
    echo "Direct hugepage set failed (now=$now_pages want=$want_pages); trying privileged Docker"
    docker run --rm --privileged -v /proc:/hostproc alpine:latest \
        sh -c "echo $want_pages > /hostproc/sys/vm/nr_hugepages" || true
    now_pages=$(cat /proc/sys/vm/nr_hugepages)
fi

awk '/MemAvailable:|HugePages_/ {print}' /proc/meminfo
if [ "$now_pages" -lt "$want_pages" ]; then
    echo "Error: cannot reserve required hugepage pool: want=$want_pages now=$now_pages" >&2
    exit 1
fi
mem_available_gb=$(awk '/MemAvailable:/ {print int($2 / 1024 / 1024)}' /proc/meminfo)
if [ "$mem_available_gb" -lt "$headroom_gb" ]; then
    echo "Error: Mooncake hugepage reservation left ${mem_available_gb} GB normal memory; require ${headroom_gb} GB (now_pages=$now_pages want=$want_pages)" >&2
    exit 1
fi
