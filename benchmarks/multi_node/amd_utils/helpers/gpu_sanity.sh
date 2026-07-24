#!/bin/bash
# GPU sanity gate, run on each node AFTER all containers are stopped.
#
# At that point VRAM should fall back to driver baseline. Any large remaining
# usage means a bare (non-containerized) process is hogging the GPU -- which
# would otherwise surface as an opaque "memory capacity is unbalanced" OOM in
# the middle of model load ~30 min into the job. We fail fast and name the
# offender instead. We only report; we never kill another user's process.
#
# Ported from https://github.com/ROCm/InferenceY/blob/main/benchmarks/multi_node/amd_utils/gpu_sanity.sh
set -uo pipefail

# Fail the node if any single GPU still has >= this many GB used.
GPU_BUSY_GB="${GPU_BUSY_GB:-8}"
# Poll a few times to give docker stop time to release VRAM before judging.
GPU_DRAIN_TRIES="${GPU_DRAIN_TRIES:-6}"

if ! command -v rocm-smi >/dev/null 2>&1; then
    echo "[gpu-sanity] rocm-smi not found on $(hostname); skipping GPU check" >&2
    exit 0
fi

busy_gb=0
for _ in $(seq "$GPU_DRAIN_TRIES"); do
    used=$(rocm-smi --showmeminfo vram --json 2>/dev/null \
        | grep -oE '"VRAM Total Used Memory \(B\)": "[0-9]+"' \
        | grep -oE '[0-9]+' | sort -n | tail -1)
    busy_gb=$(( ${used:-0} / 1024 / 1024 / 1024 ))
    if [ "$busy_gb" -lt "$GPU_BUSY_GB" ]; then
        exit 0
    fi
    sleep 30
done

echo "FATAL: $(hostname) GPU still has ${busy_gb}GB used after stopping all containers" >&2
echo "       (threshold ${GPU_BUSY_GB}GB) -- a bare (non-containerized) process is hogging the GPU:" >&2
rocm-smi --showpids >&2 || true
exit 1
