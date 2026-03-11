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
    # Use the first benic interface for UCX TCP transport (maps to ionic RDMA NIC).
    # We use TCP device names (benicXp1) instead of IB device names (ionic_X:1)
    # because ud_verbs/ionic crashes in ucp_request_memory_dereg (UCX bug with ionic provider).
    UCX_NET_DEV=$(ip -o link show 2>/dev/null | awk -F': ' '/benic1p1/{print $2}' | head -1)
    if [[ -n "$UCX_NET_DEV" ]]; then
        export UCX_NET_DEVICES="$UCX_NET_DEV"
    else
        FIRST_IB=$(echo "$IBDEVICES" | cut -d',' -f1)
        if [[ -n "$FIRST_IB" ]]; then
            export UCX_NET_DEVICES="${FIRST_IB}:1"
        fi
    fi
    echo "[INFO] Auto-set UCX_NET_DEVICES=$UCX_NET_DEVICES"
else
    echo "[INFO] Using UCX_NET_DEVICES=$UCX_NET_DEVICES (set by environment)"
fi

export NCCL_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}' | head -n 1)
export NCCL_IB_HCA=${NCCL_IB_HCA:-$IBDEVICES}

# RoCEv2: use IPv4-mapped GID (index 1) for inter-node RDMA routing
export UCX_IB_GID_INDEX=${UCX_IB_GID_INDEX:-1}

# QoS/DSCP configuration for lossless RoCEv2 fabric.
# Priority order: 1) Set by runner, 2) Detect via nicctl, 3) Detect from hostname
if [[ -n "$UCX_IB_TRAFFIC_CLASS" ]]; then
    echo "[INFO] Using UCX_IB_TRAFFIC_CLASS=$UCX_IB_TRAFFIC_CLASS (set by environment)"
elif command -v nicctl &> /dev/null; then
    ND_PRIO=$(nicctl show qos 2>/dev/null | awk '/PFC no-drop priorities/ {print $NF; exit}')
    ND_DSCP=$(nicctl show qos 2>/dev/null | awk -v p="$ND_PRIO" '
$1 == "DSCP" && $2 == ":" && $NF == p {
    print $3; exit
}')
    if [[ -n "$ND_DSCP" ]] && [[ -n "$ND_PRIO" ]]; then
        export UCX_IB_TRAFFIC_CLASS=$(( 4 * ND_DSCP ))
        export UCX_IB_SL=$ND_PRIO
        echo "[INFO] Detected QoS from nicctl: UCX_IB_TRAFFIC_CLASS=$UCX_IB_TRAFFIC_CLASS, UCX_IB_SL=$UCX_IB_SL"
    else
        echo "[WARN] nicctl available but QoS data unavailable; trying hostname detection."
        NODENAME=$(hostname -s)
        if [[ $NODENAME == GPU* ]] || [[ $NODENAME == smci355-ccs-aus* ]]; then
            export UCX_IB_TRAFFIC_CLASS=96
            echo "[INFO] Auto-detected UCX_IB_TRAFFIC_CLASS=$UCX_IB_TRAFFIC_CLASS from hostname $NODENAME"
        elif [[ $NODENAME == mia1* ]]; then
            export UCX_IB_TRAFFIC_CLASS=104
            echo "[INFO] Auto-detected UCX_IB_TRAFFIC_CLASS=$UCX_IB_TRAFFIC_CLASS from hostname $NODENAME"
        fi
    fi
else
    NODENAME=$(hostname -s)
    if [[ $NODENAME == GPU* ]] || [[ $NODENAME == smci355-ccs-aus* ]]; then
        export UCX_IB_TRAFFIC_CLASS=96
        echo "[INFO] Auto-detected UCX_IB_TRAFFIC_CLASS=$UCX_IB_TRAFFIC_CLASS from hostname $NODENAME"
    elif [[ $NODENAME == mia1* ]]; then
        export UCX_IB_TRAFFIC_CLASS=104
        echo "[INFO] Auto-detected UCX_IB_TRAFFIC_CLASS=$UCX_IB_TRAFFIC_CLASS from hostname $NODENAME"
    else
        echo "[INFO] No nicctl and unable to detect from hostname. Skipping QoS configuration."
    fi
fi

set +x
echo "[INFO] IBDEVICES=$IBDEVICES  UCX_NET_DEVICES=$UCX_NET_DEVICES  NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME  UCX_IB_GID_INDEX=$UCX_IB_GID_INDEX  UCX_IB_TRAFFIC_CLASS=${UCX_IB_TRAFFIC_CLASS:-unset}"
