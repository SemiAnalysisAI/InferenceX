#!/bin/bash
# Dual-engine environment setup for multi-node disaggregated serving.
#
# ENGINE=sglang (default): SGLang/MoRI environment
# ENGINE=vllm:             vLLM/Nixl environment
#
# REQUIRED ENVIRONMENT VARIABLES:
#   IBDEVICES - RDMA/InfiniBand device names (e.g., ionic_0,ionic_1,... or mlx5_0,mlx5_1,...)
#               Set by runner or auto-detected from hostname.
set -x

ENGINE="${ENGINE:-sglang}"
export PYTHONDONTWRITEBYTECODE=1

# =============================================================================
# Shared: IBDEVICES detection
# =============================================================================

# Prefer IBDEVICES set by runner (runners/launch_mi355x-amds.sh)
# Fall back to hostname detection if not set (for direct script execution)
if [[ -z "$IBDEVICES" ]]; then
    DETECTED=$(ibv_devinfo 2>/dev/null | grep "hca_id:" | awk '{print $2}' | paste -sd',')
    if [[ -n "$DETECTED" ]]; then
        export IBDEVICES="$DETECTED"
    else
        echo "WARNING: Unable to detect RDMA devices. Set IBDEVICES explicitly." >&2
    fi
    echo "[INFO] Auto-detected IBDEVICES=$IBDEVICES from hostname $(hostname -s)"
else
    echo "[INFO] Using IBDEVICES=$IBDEVICES (set by runner or environment)"
fi
export IBDEVICES

# Shared: Auto-detect default network interface (portable across clusters)
export GLOO_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}' | head -n 1)
export NCCL_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}' | head -n 1)

set +x

export NCCL_IB_HCA=${NCCL_IB_HCA:-$IBDEVICES}

# =============================================================================
# Engine-specific environment
# =============================================================================

if [[ "$ENGINE" == "vllm" ]]; then
    # =========================================================================
    # vLLM/Nixl-specific environment
    # =========================================================================
    set -x

    # UCX_NET_DEVICES: Use the first benic interface for UCX TCP transport
    if [[ -z "$UCX_NET_DEVICES" ]]; then
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

    # RoCEv2: use IPv4-mapped GID (index 1) for inter-node RDMA routing
    export UCX_IB_GID_INDEX=${UCX_IB_GID_INDEX:-1}

    # QoS/DSCP configuration for lossless RoCEv2 fabric.
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

else
    # =========================================================================
    # SGLang/MoRI-specific environment
    # =========================================================================

    export SGLANG_USE_AITER=1
    export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
    export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200

    # Disable allocating memory in one pass
    export MORI_SHMEM_MODE=ISOLATION
    export SGLANG_MORI_FP8_DISP=True

    if [[ "$MODEL_NAME" == *mxfp4* ]]; then
    export SGLANG_MORI_FP8_DISP=False
    fi

    export SGLANG_MORI_FP4_DISP=False
    export SGLANG_MORI_FP8_COMB=False

    # Per-role dispatch token limits (prefill uses higher throughput, decode uses lower)
    export MORI_MAX_DISPATCH_TOKENS_PREFILL=16384
    if [[ "$MODEL_NAME" == *mxfp4* ]]; then
        export MORI_MAX_DISPATCH_TOKENS_PREFILL=12288
    fi
    export MORI_MAX_DISPATCH_TOKENS_DECODE=160

    # set MTP size=1 when EP16
    export SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD=$((MORI_MAX_DISPATCH_TOKENS_DECODE * 2))

    export MORI_EP_LAUNCH_CONFIG_MODE=AUTO
    export MORI_IO_QP_MAX_SEND_WR=16384
    export MORI_IO_QP_MAX_CQE=32768
    export MORI_IO_QP_MAX_SGE=4

    export MORI_APP_LOG_LEVEL=INFO

    # Router logging control
    export SGLANG_ROUTER_STDOUT_LOGS="${SGLANG_ROUTER_STDOUT_LOGS:-0}"

    # QoS/DSCP configuration
    if [[ -n "$MORI_RDMA_TC" ]]; then
        echo "[INFO] Using MORI_RDMA_TC=$MORI_RDMA_TC (set by runner or environment)"
    elif command -v nicctl &> /dev/null; then
        ND_PRIO=$(nicctl show qos  2>/dev/null | awk '/PFC no-drop priorities/ {print $NF; exit}')
        ND_DSCP=$(nicctl show qos 2>/dev/null| awk -v p="$ND_PRIO" '
$1 == "DSCP" && $2 == ":" && $NF == p {
    print $3; exit
}')

        if [[ -n "$ND_DSCP" ]] && [[ -n "$ND_PRIO" ]]; then
            TC=$(( 4 * ND_DSCP ))
            export MORI_RDMA_SL=$ND_PRIO
            export MORI_RDMA_TC=$TC
            echo "[INFO] Detected QoS config from nicctl: MORI_RDMA_TC=$MORI_RDMA_TC, MORI_RDMA_SL=$MORI_RDMA_SL"
        else
            echo "[WARN] nicctl available but QoS data unavailable; trying hostname detection."
            # Fall back to hostname-based detection
            NODENAME=$(hostname -s)
            if [[ $NODENAME == GPU* ]] || [[ $NODENAME == smci355-ccs-aus* ]]; then
                export MORI_RDMA_TC=96
                echo "[INFO] Auto-detected MORI_RDMA_TC=$MORI_RDMA_TC from hostname $NODENAME"
            elif [[ $NODENAME == mia1* ]]; then
                export MORI_RDMA_TC=104
                echo "[INFO] Auto-detected MORI_RDMA_TC=$MORI_RDMA_TC from hostname $NODENAME"
            else
                echo "[INFO] Unable to detect MORI_RDMA_TC from hostname. Skipping RDMA QoS configuration."
            fi
        fi
    else
        # nicctl not available, try hostname-based detection
        NODENAME=$(hostname -s)
        if [[ $NODENAME == GPU* ]] || [[ $NODENAME == smci355-ccs-aus* ]]; then
            export MORI_RDMA_TC=96
            echo "[INFO] Auto-detected MORI_RDMA_TC=$MORI_RDMA_TC from hostname $NODENAME"
        elif [[ $NODENAME == mia1* ]]; then
            export MORI_RDMA_TC=104
            echo "[INFO] Auto-detected MORI_RDMA_TC=$MORI_RDMA_TC from hostname $NODENAME"
        else
            echo "[INFO] nicctl not found and unable to detect from hostname. Skipping RDMA QoS configuration."
            echo "       This is normal for clusters without QoS or outside Docker containers."
        fi
    fi

    # FIXME: WA for latest upstream 0305 image
    export PYTHONPATH=/sgl-workspace/aiter:${PYTHONPATH}

fi
