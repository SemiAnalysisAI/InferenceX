#!/bin/bash
# ATOM/mooncake-specific environment setup for multi-node disaggregated serving.
#
# Sourced by server_atom.sh in place of env.sh (which is SGLang/MoRI-specific).
#
# REQUIRED ENVIRONMENT VARIABLES:
#   IBDEVICES - RDMA/InfiniBand device names (e.g., ionic_0,ionic_1,...)
#               Set by runner or auto-detected from hostname.

set -x

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# =============================================================================
# IBDEVICES detection (same as env.sh)
# =============================================================================

if [[ -z "$IBDEVICES" ]]; then
    DETECTED=$(ibv_devinfo 2>/dev/null | grep "hca_id:" | awk '{print $2}' | paste -sd',')
    if [[ -n "$DETECTED" ]]; then
        export IBDEVICES="$DETECTED"
        echo "[INFO] Auto-detected IBDEVICES=$IBDEVICES via ibv_devinfo on $(hostname -s)"
    else
        # ATOM uses mooncake proxy_ip/handshake_port for KV transfer — IBDEVICES is
        # not passed as a server argument (unlike SGLang --disaggregation-ib-device).
        # Log a warning but do not fail; mooncake will use its own RDMA device selection.
        echo "[WARN] Unable to detect RDMA devices via ibv_devinfo; IBDEVICES unset (non-fatal for ATOM/mooncake)" >&2
    fi
else
    echo "[INFO] Using IBDEVICES=$IBDEVICES (set by runner or environment)"
fi
export IBDEVICES

export SAFETENSORS_FAST_GPU=1
export VLLM_LOG_LEVEL=WARNING
export ATOM_LOG_LEVEL=WARNING
export AITER_LOG_LEVEL=WARNING
export LOG_LEVEL=WARNING
export LOGLEVEL=WARNING

# =============================================================================
# ATOM/mooncake-specific environment
# =============================================================================

# mooncake RDMA KV transfer library path
export LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}

# ATOM MoE gather/scatter interleave optimization
export ATOM_MOE_GU_ITLV=1

# ATOM_HOST_IP is set per-node in server_atom.sh (= host_ip, used as handshake IP)

# aiter logging (WARNING to reduce noise; use DEBUG for troubleshooting)
export AITER_LOG_LEVEL=WARNING

# Disable bf16->fp8 MoE bound (matches reference script)
export AITER_BF16_FP8_MOE_BOUND=0

# Clear stale ATOM cache on startup (server_atom.sh handles this via rm -rf)
# No env var needed; documented here for reference.

set +x

echo "[INFO] ATOM env: IBDEVICES=$IBDEVICES  LD_LIBRARY_PATH includes mooncake"
