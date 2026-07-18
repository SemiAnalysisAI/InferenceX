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

ENGINE="${ENGINE:-sglang-disagg}"
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
        echo "[INFO] Auto-detected IBDEVICES=$IBDEVICES via ibv_devinfo on $(hostname -s)"
    else
        echo "ERROR: Unable to detect RDMA devices. Set IBDEVICES explicitly." >&2
        exit 1
    fi
else
    echo "[INFO] Using IBDEVICES=$IBDEVICES (set by runner or environment)"
fi
export IBDEVICES

# Shared: Auto-detect default network interface (portable across clusters)
# Only auto-detect if not already set by the runner/environment
if [[ -z "$GLOO_SOCKET_IFNAME" ]]; then
    export GLOO_SOCKET_IFNAME=$(ip route 2>/dev/null | grep '^default' | awk '{print $5}' | head -n 1)
fi
if [[ -z "$NCCL_SOCKET_IFNAME" ]]; then
    export NCCL_SOCKET_IFNAME=$(ip route 2>/dev/null | grep '^default' | awk '{print $5}' | head -n 1)
fi

set +x

export NCCL_IB_HCA=${NCCL_IB_HCA:-$IBDEVICES}

# =============================================================================
# MoRI-specific environment
# =============================================================================
# Shared by the vLLM MoRIIOConnector and the SGLang/MoRI KV-transfer path.

export MORI_IO_SQ_BACKOFF_TIMEOUT_US=50000
export MORI_IO_QP_MAX_SEND_WR=16384
export MORI_IO_QP_MAX_CQE=32768
export MORI_IO_QP_MAX_SGE=2
export MORI_IO_TC_DISABLE=0

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
    # nicctl may emit trailing commas (e.g. "24,"); keep the leading integer so the
    # arithmetic can't choke and unparseable output falls back to hostname detection.
    ND_PRIO="${ND_PRIO%%,*}"; ND_PRIO="${ND_PRIO//[!0-9]/}"
    ND_DSCP="${ND_DSCP%%,*}"; ND_DSCP="${ND_DSCP//[!0-9]/}"

    if [[ "$ND_DSCP" =~ ^[0-9]+$ ]] && [[ "$ND_PRIO" =~ ^[0-9]+$ ]]; then
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

# =============================================================================
# Engine-specific environment
# =============================================================================

if [[ "$ENGINE" == "vllm-disagg" ]]; then
    # =========================================================================
    # vLLM/Nixl-specific environment
    # =========================================================================
    export VLLM_USE_V1=1
    export VLLM_SERVER_DEV_MODE=0
    export VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1

    set -x

    # UCX_NET_DEVICES: Use the first tw-eth interface for UCX TCP transport
    if [[ -z "$UCX_NET_DEVICES" ]]; then
        UCX_NET_DEV=$(ip -o link show 2>/dev/null | awk -F': ' '/tw-eth/{print $2}' | head -1)
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
        # nicctl may emit trailing commas (e.g. "24,"); keep the leading integer so the
        # arithmetic can't choke and unparseable output falls back to hostname detection.
        ND_PRIO="${ND_PRIO%%,*}"; ND_PRIO="${ND_PRIO//[!0-9]/}"
        ND_DSCP="${ND_DSCP%%,*}"; ND_DSCP="${ND_DSCP//[!0-9]/}"
        if [[ "$ND_DSCP" =~ ^[0-9]+$ ]] && [[ "$ND_PRIO" =~ ^[0-9]+$ ]]; then
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
    # SGLang-specific environment
    # =========================================================================

    export SGLANG_USE_AITER=1
    export AITER_LOG_LEVEL=ERROR

    export SGLANG_MORI_DISPATCH_DTYPE=auto
    export MORI_COMBINE_DTYPE_PREFILL=fp8_direct_cast
    export MORI_COMBINE_DTYPE_DECODE=fp8
    export SGLANG_MORI_QP_PER_TRANSFER=4
    export SGLANG_MORI_NUM_WORKERS=4

    export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600
    export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600

    # GLM-5: uses NSA (not MLA), needs fused-decode-MLA disabled + fast loading
    MODEL_CONFIG_KEY="${MODEL_YAML_KEY:-$MODEL_NAME}"
    if [[ "$MODEL_CONFIG_KEY" == "GLM-5-FP8" || "$MODEL_CONFIG_KEY" == "GLM-5.2-FP8" ]]; then
        export SGLANG_ROCM_FUSED_DECODE_MLA=0
        export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
        export SAFETENSORS_FAST_GPU=1
    fi

    if [[ "$MODEL_CONFIG_KEY" == "GLM-5.2-FP8" ]]; then
        export SGLANG_DSA_FUSE_TOPK=false
        export SGLANG_OPT_USE_TOPK_V2=false
    fi

    # Disable allocating memory in one pass
    export MORI_SHMEM_MODE=ISOLATION

    # Enable spec v2
    export SGLANG_ENABLE_SPEC_V2=1
    export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0

    export SGLANG_LOG_MS=true
    export SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS=32

    export MORI_MAX_DISPATCH_TOKENS_PREFILL=8192
    export MORI_MAX_DISPATCH_TOKENS_DECODE=512

    export MORI_MOE_MAX_INPUT_TOKENS_PREFILL=32768
    export MORI_MOE_MAX_INPUT_TOKENS_DECODE=2703

    # set MTP size=1 when EP16
    export SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD=$((MORI_MAX_DISPATCH_TOKENS_DECODE * 2))

    export MORI_EP_LAUNCH_CONFIG_MODE=AUTO

    # Default to WARNING to cut per-op MoRI log spam on long multinode/eval
    # runs; override with MORI_APP_LOG_LEVEL=INFO when debugging.
    export MORI_APP_LOG_LEVEL="${MORI_APP_LOG_LEVEL:-WARNING}"

    # Router logging control:
    # 0 (default) keeps noisy per-request access logs out of stdout while still logging to file.
    # 1 mirrors router logs to stdout via tee (useful for live debugging).
    export SGLANG_ROUTER_STDOUT_LOGS="${SGLANG_ROUTER_STDOUT_LOGS:-0}"

    # =========================================================================
    # DeepSeek-V4-Pro PD recipe overrides
    # Placed at the end of the SGLang env block so it wins over the global
    # MoRI/SGLang defaults set above. Mirrors the validated DSv4 manual PD
    # commands (see dsv4_mi355x_sglang_disagg_plan.md §2). Only the SGLang/MoRI
    # env knobs are pinned here; CLI flags live in models.yaml and the cluster
    # NIC/socket vars (NCCL_IB_HCA, *_SOCKET_IFNAME, IBDEVICES) stay runner-derived.
    # =========================================================================
    if [[ "$MODEL_NAME" == "DeepSeek-V4-Pro" ]]; then
        # MoRI dispatch/combine dtypes: auto for both roles (not the fp8 split default)
        export SGLANG_MORI_DISPATCH_DTYPE=auto
        export MORI_COMBINE_DTYPE_PREFILL=auto
        export MORI_COMBINE_DTYPE_DECODE=auto

        # Per-role MoRI dispatch sizing (used by the harness chunked/MoE math)
        export MORI_MAX_DISPATCH_TOKENS_PREFILL=8192
        export MORI_MAX_DISPATCH_TOKENS_DECODE=64
        unset MORI_MOE_MAX_INPUT_TOKENS_PREFILL
        unset MORI_MOE_MAX_INPUT_TOKENS_DECODE

        # PER_RANK dispatch tokens are pinned independently of the sizing above
        # (16384 prefill / 128 decode in the reference recipe). server_sglang.sh
        # prefers these over the MORI_MAX_DISPATCH_TOKENS_* coupling when set.
        export MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK_PREFILL=16384
        export MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK_DECODE=128

        # Fixed inter-kernel switch threshold (not derived). NOTE: the DP+EP path in
        # server_sglang.sh recomputes this dynamically for the DEP topology.
        export SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD=4096

        # Overlap plan stream on for DSv4 (global default is 0)
        export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0

        # DSv4 model kernel routing (mirrors the single-node / manual PD recipe)
        export SGLANG_DEFAULT_THINKING=1
        export SGLANG_DSV4_REASONING_EFFORT=max
        export SGLANG_USE_ROCM700A=0
        export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
        export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
        export SGLANG_OPT_USE_FUSED_COMPRESS=true
        export SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true
        export SGLANG_OPT_FP8_WO_A_GEMM=false
        export SGLANG_OPT_USE_JIT_INDEXER_METADATA=false
        export SGLANG_OPT_USE_TOPK_V2=false
        export SGLANG_OPT_USE_AITER_INDEXER=true
        export SGLANG_OPT_USE_TILELANG_INDEXER=false
        export SGLANG_OPT_USE_TILELANG_MHC_PRE=false
        export SGLANG_OPT_USE_TILELANG_MHC_POST=false
        export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
        export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false
        export SGLANG_ROCM_USE_MULTI_STREAM=false
        export AITER_BF16_FP8_MOE_BOUND=0
        export SGLANG_EAGER_INPUT_NO_COPY=true
    fi

    # FIXME: WA for latest upstream 0305 image
    export PYTHONPATH=/sgl-workspace/aiter:${PYTHONPATH}

fi
