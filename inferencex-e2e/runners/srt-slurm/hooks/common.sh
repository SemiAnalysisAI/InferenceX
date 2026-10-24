#!/usr/bin/env bash

# Shared host checks. Sourcing this file only defines functions.

# Poll VRAM usage every 10s for up to 15 minutes. A stricter threshold is useful
# when the engine sizes its KV cache from device-wide free memory.
wait_for_amd_gpu_clean() {
    local threshold="${1:-10}"
    local gpu_clean=false vram_max i
    for i in $(seq 1 90); do
        vram_max=$(rocm-smi --showmemuse 2>/dev/null \
            | grep -oE "GPU Memory Allocated \(VRAM%\): [0-9]+" \
            | awk '{if ($NF > m) m = $NF} END {print m+0}')
        if [ "${vram_max:-0}" -le "$threshold" ]; then
            echo "GPUs clean (vram%max=$vram_max <= $threshold after $((i * 10))s)"
            gpu_clean=true
            break
        fi
        echo "waiting for prior-job GPU memory reclaim: vram%max=$vram_max (target <= $threshold)"
        sleep 10
    done
    if [ "$gpu_clean" != "true" ]; then
        echo "Error: GPUs still draining prior job's memory after 15min" >&2
        return 1
    fi
}

# The RAPL energy counters are root-only by default (CVE-2020-8694). The CPU power
# exporter runs on the bare host as the job user, so the counters become that user's
# at mode 0400: no other account can read them. The owner persists until reboot.
open_rapl_counters() {
    local zones zone user
    zones=(/sys/class/powercap/intel-rapl/intel-rapl:[0-9]*/energy_uj)
    user=$(id -un)
    if [[ ! -e "${zones[0]}" ]]; then
        echo "[$(hostname -s)] no RAPL package zones under /sys/class/powercap" >&2
        return 1
    fi
    if ! sudo -n chown "$user" "${zones[@]}" || ! sudo -n chmod 0400 "${zones[@]}"; then
        echo "[$(hostname -s)] could not give $user the RAPL energy counters" >&2
        return 1
    fi
    for zone in "${zones[@]}"; do
        if ! cat "$zone" >/dev/null; then
            echo "[$(hostname -s)] $zone is not readable as $user" >&2
            return 1
        fi
        echo "[$(hostname -s)] $(cat "$(dirname "$zone")/name") energy_uj $(stat -c '%U %a' "$zone")"
    done
}
