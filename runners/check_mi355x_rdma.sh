#!/usr/bin/env bash
set -euo pipefail

# Fast, per-node fabric preflight for MI355X srt-slurm allocations. This keeps
# the meaningful QoS/DCQCN gate from the retired amd_utils launcher without its
# Docker or job-control plumbing.

log() { printf '[%s] %s\n' "$(hostname -s)" "$*"; }
fail() { log "RDMA preflight failed: $*" >&2; exit 1; }

expected_devices="${IBDEVICES:-rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7}"
IFS=',' read -r -a devices <<< "$expected_devices"
for device in "${devices[@]}"; do
    [[ -d "/sys/class/infiniband/${device}" ]] || fail "missing device ${device}"
done
log "found all ${#devices[@]} expected RDMA devices: ${expected_devices}"

if ! command -v nicctl >/dev/null 2>&1; then
    log "nicctl is unavailable; device presence passed, QoS/DCQCN checks skipped"
    exit 0
fi

probe=$(sudo -n nicctl show version firmware 2>&1 || true)
if grep -qiE 'No AMD NICs|Invalid card handle|Failed to get NIC' <<< "$probe"; then
    fail "nicctl cannot access the AMD NICs"
fi

qos=$(sudo -n nicctl show qos 2>/dev/null) || fail "nicctl show qos failed"
classification=$(awk '/Classification type/ {print $NF; exit}' <<< "$qos")
[[ "$classification" == "DSCP" ]] || fail "classification is ${classification:-unset}, expected DSCP"

priorities=$(awk '/PFC no-drop priorities/ {print $NF; exit}' <<< "$qos")
bitmap=$(awk '/PFC priority bitmap/ {print $NF; exit}' <<< "$qos")
[[ -n "$priorities" ]] || fail "PFC no-drop priorities are missing"
[[ -n "$bitmap" && "$bitmap" != "0x0" ]] || fail "PFC is disabled"
IFS=',' read -r -a priority_values <<< "$priorities"
for priority in "${priority_values[@]}"; do
    priority="${priority//[^0-9]/}"
    [[ -n "$priority" ]] || fail "invalid PFC priority list: ${priorities}"
    (( bitmap & (1 << priority) )) || fail "PFC bitmap ${bitmap} does not cover priority ${priority}"
done

dcqcn=$(sudo -n nicctl show dcqcn 2>/dev/null) || fail "nicctl show dcqcn failed"
device_count=$(grep -c 'ROCE device' <<< "$dcqcn" || true)
(( device_count > 0 )) || fail "no RoCE devices reported by nicctl"
if grep 'Status' <<< "$dcqcn" | grep -qv 'Enabled'; then
    fail "DCQCN is disabled on at least one RoCE device"
fi
cnp_count=$(awk '/DSCP value used for CNP/ {print $NF}' <<< "$dcqcn" | sort -u | grep -c . || true)
(( cnp_count == 1 )) || fail "CNP DSCP is inconsistent across NICs"

log "RDMA QoS/DCQCN preflight passed"
