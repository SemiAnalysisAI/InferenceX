#!/bin/bash
# Reproduce the H100 system-fence cost and show its PCIe traffic. Needs nvcc and nvidia-smi.
# Usage: ./run_repro.sh [sm_90|sm_100]   Run on an otherwise idle GPU 0. If CUDA_VISIBLE_DEVICES
# remaps GPUs, point `nvidia-smi -i` at the same physical GPU the program uses.
set -euo pipefail
ARCH=${1:-sm_90}
cd "$(dirname "$0")"
if [ ! -x fence_repro ] || [ fence_repro.cu -nt fence_repro ]; then
  nvcc -O2 -arch="$ARCH" -o fence_repro fence_repro.cu
fi

echo "== system"
nvidia-smi --query-gpu=name,driver_version,vbios_version,pci.bus_id --format=csv,noheader -i 0
cat /sys/class/dmi/id/sys_vendor /sys/class/dmi/id/product_name 2>/dev/null | tr '\n' ' '; echo
uname -r

echo "== per-op cost on GPU 0 (device-timed)"
./fence_repro

echo "== GPU 0 PCIe traffic, MB/s: ~6 s idle, then ~8 s of back-to-back fence.acq_rel.sys"
nvidia-smi dmon -i 0 -s t -o T -d 1 -c 17 > dmon.txt &
DM=$!
sleep 6
echo "spin starts $(date +%T)"
./fence_repro spin 8
echo "spin ends   $(date +%T)"
wait $DM
cat dmon.txt
