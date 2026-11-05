#!/usr/bin/env bash

# Shared host checks. Sourcing this file only defines functions.

# Poll VRAM usage every 10s for up to $2 minutes (default 15). A stricter threshold
# is useful when the engine sizes its KV cache from device-wide free memory.
wait_for_amd_gpu_clean() {
    local threshold="${1:-10}" minutes="${2:-15}"
    local gpu_clean=false vram_max i
    for i in $(seq 1 $((minutes * 6))); do
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
        echo "Error: GPUs still draining prior job's memory after ${minutes}min" >&2
        return 1
    fi
}
