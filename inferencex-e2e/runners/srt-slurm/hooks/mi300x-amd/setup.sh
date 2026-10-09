#!/usr/bin/env bash
set -eo pipefail

# RCCL cannot reclaim scratch memory on MEC firmware older than 177 and crashes.
# See https://rocm.docs.amd.com/en/docs-6.4.3/about/release-notes.html#amdgpu-driver-updates
minimum=177
firmware=$(rocm-smi --showfw | awk '/MEC firmware version/ {print $NF}' | sort -n | head -n 1)
if [[ -z "$firmware" || "$firmware" -lt "$minimum" ]]; then
    echo "[$(hostname -s)] MEC firmware ${firmware:-unknown} is older than $minimum" >&2
    exit 1
fi
echo "[$(hostname -s)] MEC firmware $firmware"
