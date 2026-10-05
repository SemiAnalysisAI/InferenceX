#!/usr/bin/env bash
# Debug: hand the engram host table to kernels at its hipHostGetDevicePointer
# address instead of the host address (dsv41flash-sglang-engram-device-ptr.patch).
set -euo pipefail
patch_file=/configs/dsv41flash-sglang-engram-device-ptr.patch
target=/sgl-workspace/sglang/python/sglang/srt/layers/engram.py
cd /sgl-workspace/sglang
if patch -p1 --dry-run --reverse --force --silent < "$patch_file" > /dev/null 2>&1; then
    echo "engram device-ptr patch: already applied"
else
    patch -p1 --forward < "$patch_file"
    echo "engram device-ptr patch: applied"
fi
grep -n "_registered_device_ptr\|nbytes, _HOST_REGISTER_MAPPED\|def device_address" "$target"
