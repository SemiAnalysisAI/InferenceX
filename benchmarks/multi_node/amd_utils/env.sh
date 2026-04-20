#!/bin/bash
# SGLang/MoRI environment setup for multi-node disaggregated serving.
#
# REQUIRED ENVIRONMENT VARIABLES:
#   IBDEVICES - RDMA/InfiniBand device names (e.g., ionic_0,ionic_1,... or mlx5_0,mlx5_1,...)
#               This must be set by the runner script (runners/launch_mi355x-amds.sh)
#
# OPTIONAL ENVIRONMENT VARIABLES:
#   MORI_RDMA_TC - RDMA traffic class (e.g., 96, 104). Set by runner if cluster uses QoS.

set -x
export PYTHONDONTWRITEBYTECODE=1

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
        echo "ERROR: Unable to detect cluster from hostname $NODENAME and IBDEVICES not set" >&2
        exit 1
    fi
    echo "[INFO] Auto-detected IBDEVICES=$IBDEVICES from hostname $NODENAME"
else
    echo "[INFO] Using IBDEVICES=$IBDEVICES (set by runner or environment)"
fi
export IBDEVICES

# Auto-detect default network interface (portable across clusters)
export GLOO_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}' | head -n 1)
export NCCL_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}' | head -n 1)


export NCCL_IB_HCA=$IBDEVICES

export SGLANG_USE_AITER=1

export SGLANG_MORI_DISPATCH_DTYPE=auto
export SGLANG_MORI_FP8_COMB=true
export SGLANG_MORI_QP_PER_TRANSFER=2
export SGLANG_MORI_NUM_WORKERS=2

export MORI_IO_QP_MAX_SEND_WR=16384
export MORI_IO_QP_MAX_CQE=32768 
export MORI_IO_QP_MAX_SGE=4

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600

# Disable allocating memory in one pass
export MORI_SHMEM_MODE=ISOLATION

# Enable spec v2 
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

export SGLANG_LOG_MS=true
export SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS=32


# Per-role dispatch token limits (prefill uses higher throughput, decode uses lower)
# export MORI_MAX_DISPATCH_TOKENS_PREFILL=16384
# if [[ "$MODEL_NAME" == *mxfp4* || "$MODEL_NAME" == *MXFP4* ]]; then
#     export MORI_MAX_DISPATCH_TOKENS_PREFILL=12288
# fi

export MORI_MAX_DISPATCH_TOKENS_PREFILL=5120
export MORI_MAX_DISPATCH_TOKENS_DECODE=256
export MORI_MOE_MAX_INPUT_TOKENS_DECODE=2048

# set MTP size=1 when EP16
export SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD=$((MORI_MAX_DISPATCH_TOKENS_DECODE * 2))

export MORI_EP_LAUNCH_CONFIG_MODE=AUTO


export MORI_APP_LOG_LEVEL=INFO

# Router logging control:
# 0 (default) keeps noisy per-request access logs out of stdout while still logging to file.
# 1 mirrors router logs to stdout via tee (useful for live debugging).
export SGLANG_ROUTER_STDOUT_LOGS="${SGLANG_ROUTER_STDOUT_LOGS:-0}"

# QoS/DSCP configuration
# Priority order: 1) Set by runner, 2) Detect via nicctl, 3) Detect from hostname
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
        export MORI_IO_SL=$ND_PRIO
        export MORI_RDMA_TC=$TC
        export MORI_IO_TC=$TC
        echo "[INFO] Detected QoS config from nicctl: MORI_RDMA_TC=$MORI_RDMA_TC, MORI_RDMA_SL=$MORI_RDMA_SL, MORI_IO_TC=$MORI_IO_TC, MORI_IO_SL=$MORI_IO_SL"
    else
        echo "[WARN] nicctl available but QoS data unavailable; trying hostname detection."
        # Fall back to hostname-based detection
        NODENAME=$(hostname -s)
        if [[ $NODENAME == GPU* ]] || [[ $NODENAME == smci355-ccs-aus* ]]; then
            export MORI_RDMA_TC=96
            export MORI_IO_TC=96
            echo "[INFO] Auto-detected MORI_RDMA_TC=$MORI_RDMA_TC from hostname $NODENAME"
        elif [[ $NODENAME == mia1* ]]; then
            export MORI_RDMA_TC=104
            export MORI_IO_TC=104
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
        export MORI_IO_TC=96
        echo "[INFO] Auto-detected MORI_RDMA_TC=$MORI_RDMA_TC from hostname $NODENAME"
    elif [[ $NODENAME == mia1* ]]; then
        export MORI_RDMA_TC=104
        export MORI_IO_TC=104
        echo "[INFO] Auto-detected MORI_RDMA_TC=$MORI_RDMA_TC from hostname $NODENAME"
    else
        echo "[INFO] nicctl not found and unable to detect from hostname. Skipping RDMA QoS configuration."
        echo "       This is normal for clusters without QoS or outside Docker containers."
    fi
fi

# FIXME: WA for latest upstream 0305 image
export PYTHONPATH=/sgl-workspace/aiter:${PYTHONPATH}


set +x
