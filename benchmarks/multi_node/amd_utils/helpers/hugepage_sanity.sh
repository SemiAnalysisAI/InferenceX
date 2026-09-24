#!/bin/bash
# Hugepage pre-flight, run on each allocated node before any container starts.
# A stale hugepage reservation is carved out of normal RAM and can make
# HiCache host-pool sizing fail at startup; this reclaims it, or grows the
# pool when a caller (the UMBP DRAM tier) needs hugepages.
#
# Safety: a shrink only releases idle pages. Pages in use or reserved (Rsvd)
# are kept: the kernel never shrinks the pool below in-use + reserved, and
# turns held pages above the target into surplus pages that are freed once
# their owner releases them.
#
# Knobs:
#   SKIP_HUGEPAGE_CHECK=1  skip this check entirely
#   HUGEPAGES_TARGET=<N>   desired nr_hugepages (default 0 = release all idle)
#   HUGEPAGE_GROW_TOLERANCE_PCT  allowed shortfall on grow (default 2)
#
# Exit code: 0 = OK or skipped; 1 = a needed change could not be made (large
# leftover reservation, or a grow short by more than the tolerance).
set -uo pipefail

TARGET="${HUGEPAGES_TARGET:-0}"
HOST="$(hostname -s)"

# Leftover reservation (pages) that fails the node if it can't be reclaimed.
FAIL_THRESHOLD_PAGES="${HUGEPAGE_FAIL_THRESHOLD_PAGES:-4096}"

log()      { echo "[hugepage] $HOST: $*"; }
log_warn() { echo "[hugepage] $HOST: WARNING: $*" >&2; }
log_fail() { echo "[hugepage] $HOST: FATAL: $*" >&2; }

if [[ "${SKIP_HUGEPAGE_CHECK:-0}" == "1" ]]; then
    log "SKIP_HUGEPAGE_CHECK=1 set; skipping hugepage pre-flight"
    exit 0
fi

meminfo_field() {
    awk -v k="$1:" '$1 == k { print $2; exit }' /proc/meminfo
}

TOTAL=$(meminfo_field HugePages_Total)
FREE=$(meminfo_field HugePages_Free)
RSVD=$(meminfo_field HugePages_Rsvd)
SIZE_KB=$(meminfo_field Hugepagesize)
: "${TOTAL:=0}" "${FREE:=0}" "${RSVD:=0}" "${SIZE_KB:=2048}"

gb() { echo $(( $1 * SIZE_KB / 1024 / 1024 )); }

if [[ "$TOTAL" -eq "$TARGET" ]]; then
    [[ "$TOTAL" -eq 0 ]] || log "nr_hugepages already at target ($TOTAL pages, $(gb "$TOTAL") GB)"
    exit 0
fi

IN_USE=$(( TOTAL - FREE ))
# Pages the kernel will not release: mapped (in use) plus reserved for an
# existing mapping but not yet faulted in (Rsvd, counted inside Free).
HELD=$(( IN_USE + RSVD ))
# Pool size expected after the change; the checks below compare against it.
EXPECTED=$TARGET
IS_GROW=0
if [[ "$TOTAL" -gt "$TARGET" ]]; then
    # Shrinking: release the idle pages, keep the held ones.
    if [[ "$HELD" -ge "$TOTAL" ]]; then
        log_warn "all $TOTAL hugepages ($(gb "$TOTAL") GB) are held ($IN_USE in use / $RSVD rsvd) -- nothing idle to release."
        log_warn "HiCache host-pool sizing may fail on this node. Re-run once the owning job finishes, or exclude this node."
        exit 0
    fi
    if [[ "$HELD" -gt "$TARGET" ]]; then
        EXPECTED=$HELD
        log_warn "$TOTAL hugepages ($(gb "$TOTAL") GB) reserved, $IN_USE in use / $RSVD rsvd -- releasing $(( TOTAL - HELD )) idle ($(gb $(( TOTAL - HELD ))) GB), keeping $HELD held ($(gb "$HELD") GB) until their owner frees them"
    else
        log "$TOTAL hugepages ($(gb "$TOTAL") GB) reserved, $HELD held -- reclaiming to $TARGET"
    fi
else
    IS_GROW=1
    log "growing nr_hugepages $TOTAL -> $TARGET ($(gb "$TARGET") GB)"
fi

apply() {
    if [[ $EUID -eq 0 ]]; then
        sysctl -w vm.nr_hugepages="$TARGET" >/dev/null 2>&1
    elif sudo -n true 2>/dev/null; then
        sudo -n sysctl -w vm.nr_hugepages="$TARGET" >/dev/null 2>&1
    else
        return 127
    fi
}

if ! apply; then
    RC=$?
    if [[ $RC -eq 127 ]]; then
        log_warn "cannot adjust nr_hugepages (no root and no passwordless sudo)"
    else
        log_warn "sysctl vm.nr_hugepages=$TARGET failed (rc=$RC)"
    fi
fi

NEW_TOTAL=$(meminfo_field HugePages_Total)
: "${NEW_TOTAL:=$TOTAL}"

if [[ "$NEW_TOTAL" -eq "$EXPECTED" ]]; then
    log "nr_hugepages now $NEW_TOTAL ($(gb "$NEW_TOTAL") GB); freed $(gb $(( TOTAL - NEW_TOTAL ))) GB back to normal allocation"
    exit 0
fi

# Shrink partially satisfied: fail only if the idle leftover is large (held
# pages are already accounted for in EXPECTED).
if [[ "$NEW_TOTAL" -gt "$EXPECTED" && $(( NEW_TOTAL - EXPECTED )) -ge "$FAIL_THRESHOLD_PAGES" ]]; then
    log_fail "still $NEW_TOTAL hugepages ($(gb "$NEW_TOTAL") GB) reserved after trying to reach $EXPECTED."
    log_fail "That much RAM carved out will make HiCache host-pool sizing fail later in model load."
    log_fail "Fix the node (sysctl vm.nr_hugepages=$TARGET), exclude it, or set SKIP_HUGEPAGE_CHECK=1 to proceed anyway."
    exit 1
fi

# Grow short: sysctl can silently grant fewer pages (fragmentation), and the
# caller would then fall back to 4 KiB pages. Fail if the shortfall exceeds
# a percentage of TARGET.
GROW_FAIL_TOLERANCE_PCT="${HUGEPAGE_GROW_TOLERANCE_PCT:-2}"
if [[ "$IS_GROW" -eq 1 && "$NEW_TOTAL" -lt "$TARGET" ]]; then
    DEFICIT=$(( TARGET - NEW_TOTAL ))
    TOLERANCE_PAGES=$(( (TARGET * GROW_FAIL_TOLERANCE_PCT + 99) / 100 ))
    if [[ "$DEFICIT" -gt "$TOLERANCE_PAGES" ]]; then
        log_fail "requested $TARGET hugepages ($(gb "$TARGET") GB) but kernel only granted $NEW_TOTAL ($(gb "$NEW_TOTAL") GB)."
        log_fail "Short by $DEFICIT pages ($(gb "$DEFICIT") GB), exceeding the ${GROW_FAIL_TOLERANCE_PCT}% tolerance ($TOLERANCE_PAGES pages)."
        log_fail "Proceeding would silently demote the workload to 4 KiB pages while config says hugepages are on -- refusing."
        log_fail "Fix the node's free/contiguous memory, exclude it, or set SKIP_HUGEPAGE_CHECK=1 to proceed anyway."
        exit 1
    fi
    log_warn "requested $TARGET hugepages, got $NEW_TOTAL (short by $DEFICIT pages, $(gb "$DEFICIT") GB) -- within ${GROW_FAIL_TOLERANCE_PCT}% tolerance; proceeding"
    exit 0
fi

log_warn "nr_hugepages is $NEW_TOTAL, wanted $EXPECTED (difference under the $FAIL_THRESHOLD_PAGES-page fail threshold); proceeding"
exit 0
