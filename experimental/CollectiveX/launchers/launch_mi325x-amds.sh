#!/usr/bin/env bash
# CollectiveX — MI325X (AMD CDNA3 gfx942, 8 GPU/node) wrapper.
# Scheduling, exclusions, and storage are supplied by the runner-local config.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Same-host MoRI traffic uses the SDMA/XGMI path by default.
export MORI_DISABLE_AUTO_XGMI="${MORI_DISABLE_AUTO_XGMI:-0}"
# The CDNA3 EP adapter uses the newer MoRI image and AsyncLL kernel with SDMA.
case "${CX_BENCH:-}" in
  mori)
    export CX_IMAGE="${CX_IMAGE:-rocm/sgl-dev:sglang-0.5.14-rocm720-mi35x-mori-0701}"
    export CX_MORI_KERNEL_TYPE="${CX_MORI_KERNEL_TYPE:-asyncll}"
    export MORI_ENABLE_SDMA="${MORI_ENABLE_SDMA:-1}"
    ;;
  *)
    export MORI_ENABLE_SDMA="${MORI_ENABLE_SDMA:-1}"
    ;;
esac
# MoRI initialization diagnostics record the selected transport path.
export MORI_APP_LOG_LEVEL="${MORI_APP_LOG_LEVEL:-info}"
export MORI_SHMEM_LOG_LEVEL="${MORI_SHMEM_LOG_LEVEL:-info}"
export MORI_IO_LOG_LEVEL="${MORI_IO_LOG_LEVEL:-info}"
exec bash "$HERE/launch_mi355x-amds.sh"
