#!/bin/bash
# vLLM/Nixl environment setup for multi-node disaggregated serving.
#
# REQUIRED ENVIRONMENT VARIABLES:
#   IBDEVICES - RDMA/InfiniBand device names (e.g., ionic_0,ionic_1,... or mlx5_0,mlx5_1,...)
#               Set by runner or auto-detected from hostname.
#
# The Docker image (built from vllm_disagg_inference.ubuntu.amd.Dockerfile) already
# sets LD_LIBRARY_PATH for UCX (/usr/local/ucx/lib) and RIXL (/usr/local/RIXL/install/lib).

set -x

# IBDEVICES configuration
# Prefer IBDEVICES set by runner (runners/launch_mi355x-amds.sh)
# Fall back to hostname detection if not set (for direct script execution)
if [[ -z "$IBDEVICES" ]]; then
    NODENAME=$(hostname -s)
    if [[ $NODENAME == GPU* ]] || [[ $NODENAME == smci355-ccs-aus* ]]; then
        export IBDEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
    elif [[ $NODENAME == mia1* ]]; then
        export IBDEVICES=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
    else
        DETECTED=$(ibv_devinfo 2>/dev/null | grep "hca_id:" | awk '{print $2}' | paste -sd',')
        if [[ -n "$DETECTED" ]]; then
            export IBDEVICES="$DETECTED"
        else
            echo "WARNING: Unable to detect RDMA devices. Set IBDEVICES explicitly." >&2
        fi
    fi
    echo "[INFO] Auto-detected IBDEVICES=$IBDEVICES from hostname $(hostname -s)"
else
    echo "[INFO] Using IBDEVICES=$IBDEVICES (set by runner or environment)"
fi

if [[ -z "$UCX_NET_DEVICES" ]]; then
    FIRST_IB=$(echo "$IBDEVICES" | cut -d',' -f1)
    if [[ -n "$FIRST_IB" ]]; then
        export UCX_NET_DEVICES="${FIRST_IB}:1"
    fi
    echo "[INFO] Auto-set UCX_NET_DEVICES=$UCX_NET_DEVICES"
else
    echo "[INFO] Using UCX_NET_DEVICES=$UCX_NET_DEVICES (set by environment)"
fi

export NCCL_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}' | head -n 1)
export NCCL_IB_HCA=${NCCL_IB_HCA:-$IBDEVICES}

# RoCEv2: use IPv4-mapped GID (index 1) for inter-node RDMA routing
export UCX_IB_GID_INDEX=${UCX_IB_GID_INDEX:-1}

set +x
echo "[INFO] IBDEVICES=$IBDEVICES  UCX_NET_DEVICES=$UCX_NET_DEVICES  NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME  UCX_IB_GID_INDEX=$UCX_IB_GID_INDEX"
