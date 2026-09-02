#!/usr/bin/env bash
set -euo pipefail

action=${1:?usage: hugepage_setup.sh reserve|restore JOB_ID [TARGET_PAGES]}
job_id=${2:?usage: hugepage_setup.sh reserve|restore JOB_ID [TARGET_PAGES]}
state_file="/tmp/inferencex-hugepages-${job_id}"

read_pages() {
    cat /proc/sys/vm/nr_hugepages
}

case "$action" in
    reserve)
        target_pages=${3:?reserve requires TARGET_PAGES}
        if [[ ! "$target_pages" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: invalid HugeTLB target: $target_pages" >&2
            exit 2
        fi
        page_kb=$(awk '/^Hugepagesize:/ {print $2}' /proc/meminfo)
        if [[ "$page_kb" != "2048" ]]; then
            echo "ERROR: expected 2 MiB HugeTLB pages, found ${page_kb:-unknown} KiB" >&2
            exit 1
        fi
        old_pages=$(read_pages)
        printf '%s\n' "$old_pages" > "$state_file"
        sudo -n sysctl -w "vm.nr_hugepages=$target_pages"
        actual_pages=$(read_pages)
        if (( actual_pages < target_pages )); then
            echo "ERROR: HugeTLB reservation incomplete on $(hostname): target=$target_pages actual=$actual_pages" >&2
            sudo -n sysctl -w "vm.nr_hugepages=$old_pages" || true
            rm -f "$state_file"
            exit 1
        fi
        echo "[hugetlb] reserved target=$target_pages actual=$actual_pages page_kb=$page_kb old=$old_pages host=$(hostname)"
        ;;
    restore)
        if [[ ! -f "$state_file" ]]; then
            echo "[hugetlb] no state file for job=$job_id host=$(hostname); nothing to restore"
            exit 0
        fi
        old_pages=$(<"$state_file")
        sudo -n sysctl -w "vm.nr_hugepages=$old_pages"
        rm -f "$state_file"
        echo "[hugetlb] restored nr_hugepages=$old_pages host=$(hostname)"
        ;;
    *)
        echo "ERROR: unknown action: $action" >&2
        exit 2
        ;;
esac
