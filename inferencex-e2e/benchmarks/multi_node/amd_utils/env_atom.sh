#!/bin/bash
# ATOM/mooncake environment, sourced by server_atom.sh in place of env.sh.
# IBDEVICES: RDMA device names (e.g. ionic_0,ionic_1,...), set by the runner or
# auto-detected.

set -x

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1


if [[ -z "$IBDEVICES" ]]; then
    DETECTED=$(ibv_devinfo 2>/dev/null | grep "hca_id:" | awk '{print $2}' | paste -sd',')
    if [[ -n "$DETECTED" ]]; then
        export IBDEVICES="$DETECTED"
        echo "[INFO] Auto-detected IBDEVICES=$IBDEVICES via ibv_devinfo on $(hostname -s)"
    else
        # ATOM passes no IB device to the server (mooncake picks its own RDMA device via
        # proxy_ip/handshake_port), so a missing IBDEVICES is non-fatal here.
        echo "[WARN] Unable to detect RDMA devices via ibv_devinfo; IBDEVICES unset (non-fatal for ATOM/mooncake)" >&2
    fi
else
    echo "[INFO] Using IBDEVICES=$IBDEVICES (set by runner or environment)"
fi
export IBDEVICES


export LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}

export SAFETENSORS_FAST_GPU=1

export VLLM_LOG_LEVEL=WARNING
export ATOM_LOG_LEVEL=WARNING
export AITER_LOG_LEVEL=WARNING
export LOG_LEVEL=WARNING
export LOGLEVEL=WARNING

set +x

echo "[INFO] ATOM env: IBDEVICES=$IBDEVICES  LD_LIBRARY_PATH includes mooncake"