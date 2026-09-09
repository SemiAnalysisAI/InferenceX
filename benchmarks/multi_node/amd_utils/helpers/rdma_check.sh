#!/bin/bash
# Pre-flight RDMA QoS/DCQCN validation for a single node, run by job.slurm via
# `srun` across every allocated node BEFORE any container/GPU time is spent.
#
# A misconfigured NIC (PFC not covering the RoCE priority, DCQCN disabled, ...)
# doesn't make MoRI's cross-node EP/RDMA transfers error out -- it just quietly
# degrades, and the only symptom is unexplained tail latency or throughput
# variance in the benchmark numbers hours later. Failing fast here, before the
# job burns node-hours, is much cheaper than debugging that after the fact.
#
# check_qos()/check_dcqcn() below are trimmed/adapted from ROCm/mori's
# tools/env_check.sh:
#   https://github.com/ROCm/mori/blob/main/tools/env_check.sh
# The upstream script's expensive ib_write_bw/ib_write_lat full-mesh bandwidth
# and latency tests are intentionally NOT ported here -- this runs before
# every single job, so it must be fast (seconds), not a multi-minute
# fabric-wide benchmark of its own.
#
# Scoped to AMD Pollara (ionic) NICs via nicctl, matching this repo's current
# fleet (see env.sh's nicctl-based MORI_RDMA_TC/SL detection). If bnxt_re or
# mlx5 NICs are added to the fleet, port the bnxt_*/mlx5_* check_* functions
# from the upstream script following the same pattern.
#
# Exit code: 0 = OK (or gracefully skipped, e.g. no ionic NICs on this host),
#            1 = hard QoS/DCQCN misconfiguration -- do not proceed with the run.
set -uo pipefail

AINIC_MIN_VER="1.117.5-a-45"   # minimum recommended AINIC firmware for IBGDA

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

log_ok()   { echo -e "[$(hostname -s)] ${GREEN}[OK]${NC}   $*"; }
log_fail() { echo -e "[$(hostname -s)] ${RED}[FAIL]${NC} $*"; }
log_warn() { echo -e "[$(hostname -s)] ${YELLOW}[WARN]${NC} $*"; }
die()      { log_fail "$@"; exit 1; }

# version_ge <candidate> <min> -> true if <candidate> >= <min> (dotted/hyphenated, via sort -V)
version_ge() {
    local cand="$1" min="$2"
    [[ "$cand" == "$min" ]] && return 0
    [[ "$(printf '%s\n%s\n' "$cand" "$min" | sort -V | head -1)" == "$min" ]]
}

# check_ainic_version_recommendation <fw_version>
#   warns if firmware is on the IBGDA-incapable 1.117.1 branch, or below the
#   recommended minimum for cross-node MORI (EP over RDMA / IBGDA).
check_ainic_version_recommendation() {
    local ver="$1"
    [[ -n "$ver" ]] || { log_warn "cannot verify AINIC firmware version against recommendation (empty)"; return; }
    if [[ "$ver" =~ ^1\.117\.1([.-]|$) ]]; then
        log_warn "AINIC firmware $ver is on the 1.117.1 branch, which does NOT support IBGDA -- upgrade to >= $AINIC_MIN_VER"
    elif version_ge "$ver" "$AINIC_MIN_VER"; then
        log_ok "AINIC firmware $ver meets the recommended minimum (>= $AINIC_MIN_VER) for cross-node IBGDA"
    else
        log_warn "AINIC firmware $ver is below the recommended minimum (>= $AINIC_MIN_VER) for cross-node IBGDA"
    fi
}

# check_versions() -- informational only (never hard-fails the job).
check_versions() {
    local fw_output sw_output
    fw_output=$(sudo nicctl show version firmware 2>/dev/null)
    sw_output=$(sudo nicctl show version host-software 2>/dev/null)

    local fw_versions fw_count
    fw_versions=$(echo "$fw_output" | grep -i "firmware" | awk '{print $NF}' | sort -u)
    fw_count=$(echo "$fw_versions" | grep -c . || true)
    if [[ "$fw_count" -ne 1 ]]; then
        log_warn "firmware versions not consistent across NICs:"
        echo "$fw_versions"
        local v
        while read -r v; do [[ -n "$v" ]] && check_ainic_version_recommendation "$v"; done <<< "$fw_versions"
    else
        log_ok "firmware : $fw_versions"
        check_ainic_version_recommendation "$fw_versions"
    fi

    local nicctl_ver
    nicctl_ver=$(echo "$sw_output" | grep "nicctl" | awk '{print $NF}')
    [[ -n "$nicctl_ver" ]] && log_ok "nicctl : $nicctl_ver" || log_warn "cannot determine nicctl version"
}

# check_qos() -- HARD gate: classification type must be DSCP, and PFC no-drop
# must be enabled and cover every no-drop priority. Dies (exit 1) otherwise.
check_qos() {
    local qos_output
    qos_output=$(sudo nicctl show qos 2>/dev/null)
    [[ -n "$qos_output" ]] || die "sudo nicctl show qos returned nothing"

    local class_type
    class_type=$(echo "$qos_output" | grep "Classification type" | head -1 | awk '{print $NF}')
    [[ "$class_type" == "DSCP" ]] || die "classification type is '$class_type', expected 'DSCP'"
    log_ok "classification type : DSCP"

    local nd_prio_raw
    nd_prio_raw=$(echo "$qos_output" | grep "PFC no-drop priorities" | head -1 | awk '{print $NF}')
    [[ -n "$nd_prio_raw" ]] || die "cannot find PFC no-drop priority"
    local nd_prios=()
    IFS=',' read -ra nd_prios <<< "$nd_prio_raw"
    log_ok "no-drop priorities : ${nd_prios[*]}"

    local pfc_bitmap
    pfc_bitmap=$(echo "$qos_output" | grep "PFC priority bitmap" | head -1 | awk '{print $NF}')
    [[ -n "$pfc_bitmap" && "$pfc_bitmap" != "0x0" ]] || die "PFC is not enabled (bitmap=$pfc_bitmap)"
    local p
    for p in "${nd_prios[@]}"; do
        (( pfc_bitmap & (1 << p) )) || die "PFC bitmap $pfc_bitmap does not cover priority $p"
    done
    log_ok "PFC enabled for priorities ${nd_prios[*]} (bitmap=$pfc_bitmap)"
}

# check_dcqcn() -- HARD gate: DCQCN must be enabled on every ROCE device, and
# the CNP DSCP must be consistent across NICs. Exits 1 otherwise.
check_dcqcn() {
    local dcqcn_output
    dcqcn_output=$(sudo nicctl show dcqcn 2>/dev/null)
    [[ -n "$dcqcn_output" ]] || die "sudo nicctl show dcqcn returned nothing"

    local total
    total=$(echo "$dcqcn_output" | grep -c "ROCE device")
    [[ "$total" -gt 0 ]] || die "no ROCE devices found in dcqcn output"

    local disabled
    disabled=$(echo "$dcqcn_output" | grep "Status" | grep -v "Enabled" || true)
    if [[ -n "$disabled" ]]; then
        log_fail "some ROCE devices have DCQCN disabled:"
        echo "$disabled"
        exit 1
    fi
    log_ok "DCQCN enabled on all $total ROCE devices"

    local cnp_values cnp_count
    cnp_values=$(echo "$dcqcn_output" | grep "DSCP value used for CNP" | awk '{print $NF}' | sort -u)
    cnp_count=$(echo "$cnp_values" | grep -c . || true)
    [[ "$cnp_count" -eq 1 ]] || die "CNP DSCP not consistent across NICs: $cnp_values"
    log_ok "CNP DSCP = $cnp_values (consistent across all NICs)"
}

# ============================= main =============================

if ! command -v nicctl &>/dev/null; then
    log_warn "nicctl not found on $(hostname -s) -- skipping RDMA QoS/DCQCN pre-flight check (not an ionic NIC host, or nicctl not on PATH)"
    exit 0
fi

# nicctl exits 0 even with no NIC present, so check its output rather than its exit code.
_nicctl_probe=$(sudo nicctl show version firmware 2>&1 || true)
if echo "$_nicctl_probe" | grep -qiE 'No AMD NICs|Invalid card handle|Failed to get NIC'; then
    log_warn "nicctl present but no ionic NIC detected/accessible on $(hostname -s) -- skipping RDMA QoS/DCQCN pre-flight check"
    exit 0
fi

check_versions
check_qos
check_dcqcn

log_ok "RDMA QoS/DCQCN pre-flight check passed on $(hostname -s)"
