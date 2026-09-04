#!/usr/bin/env bash
# Report this compute node's EFA inventory and GPU affinity before the engine starts.
#
# b300-dsxe's compute fabric is EFA, not RoCE: 16 adapters (driver efa, 400 Gb
# each, two per GPU) that present no netdev and no IPv4. There are no per-rail
# subnets, so there is nothing to pin and nothing to compare subnets against --
# the recipes leave device selection to the fabric.
#
# What is worth checking is that the node has its full complement. A node
# missing adapters still runs, just slower and asymmetrically against its peers,
# which is the kind of thing that shows up as an unexplained outlier rather than
# an error. Everything here is diagnostic; nothing fails the job.

set -uo pipefail

EXPECTED_EFA=16

echo "=== [b300-fabric] node: $(hostname) ==="

mapfile -t EFA_DEVS < <(
    for d in /sys/class/infiniband/*; do
        [ -e "$d" ] || continue
        [ "$(basename "$(readlink -f "$d/device/driver")" 2>/dev/null)" = efa ] || continue
        basename "$d"
    done | sort
)

echo "--- EFA adapters: ${#EFA_DEVS[@]} (expected $EXPECTED_EFA) ---"
for dev in "${EFA_DEVS[@]}"; do
    state=$(awk '{print $2}' "/sys/class/infiniband/$dev/ports/1/state" 2>/dev/null)
    rate=$(awk '{print $1}' "/sys/class/infiniband/$dev/ports/1/rate" 2>/dev/null)
    pci=$(basename "$(readlink -f "/sys/class/infiniband/$dev/device")" 2>/dev/null)
    printf '  %-12s %-10s %sGb  %s\n' "$dev" "${state:-?}" "${rate:-?}" "${pci:-?}"
done

if [ "${#EFA_DEVS[@]}" -ne "$EXPECTED_EFA" ]; then
    echo "[b300-fabric] WARNING: ${#EFA_DEVS[@]} EFA adapters, expected $EXPECTED_EFA."
    echo "[b300-fabric] This node will move KV more slowly than its peers."
fi

down=0
for dev in "${EFA_DEVS[@]}"; do
    state=$(awk '{print $2}' "/sys/class/infiniband/$dev/ports/1/state" 2>/dev/null)
    [ "$state" = ACTIVE ] || { echo "[b300-fabric] WARNING: $dev is $state, not ACTIVE."; down=$((down + 1)); }
done
[ "$down" -eq 0 ] && echo "[b300-fabric] all ${#EFA_DEVS[@]} adapters ACTIVE."

echo "--- GPU / EFA affinity (by PCI bus) ---"
nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader 2>/dev/null | while IFS=, read -r idx bus; do
    gbus=$(echo "$bus" | tr -d ' ' | cut -d: -f2 | tr 'A-Z' 'a-z')
    near=""
    for dev in "${EFA_DEVS[@]}"; do
        pci=$(basename "$(readlink -f "/sys/class/infiniband/$dev/device")" 2>/dev/null)
        nbus=$(echo "$pci" | cut -d: -f2)
        d=$(( 16#$gbus - 16#$nbus )); [ "$d" -lt 0 ] && d=$(( -d ))
        [ "$d" -le 2 ] && near="$near $dev"
    done
    printf '  GPU%s (bus %s):%s\n' "$idx" "$gbus" "${near:- none}"
done

echo "=== [b300-fabric] done ==="
