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

# The RAPL energy counters are root-only by default (CVE-2020-8694). The CPU
# power exporter runs on the bare host as the job user, so open them for reading.
# The mode resets on reboot; nothing else about the host changes.
zones=(/sys/class/powercap/intel-rapl/intel-rapl:[0-9]*/energy_uj)
if [[ ! -e "${zones[0]}" ]]; then
    echo "[$(hostname -s)] no RAPL package zones under /sys/class/powercap" >&2
    exit 1
fi
sudo -n chmod o+r "${zones[@]}"
for zone in "${zones[@]}"; do
    echo "[$(hostname -s)] $(cat "$(dirname "$zone")/name") energy_uj mode $(stat -c %a "$zone")"
done
