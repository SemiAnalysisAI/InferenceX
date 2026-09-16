#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/../../benchmark_lib.sh" --validation-only
check_env_vars \
    MORI_IO_SQ_BACKOFF_TIMEOUT_US MORI_IO_QP_MAX_SEND_WR MORI_IO_QP_MAX_CQE MORI_IO_QP_MAX_SGE MORI_IO_TC_DISABLE \
    UCX_IB_GID_INDEX MORI_APP_LOG_LEVEL SGLANG_ROUTER_STDOUT_LOGS TORCH_NCCL_BLOCKING_WAIT NCCL_BLOCKING_WAIT \
    SGLANG_OPT_USE_AITER_INDEXER
# Dual-engine environment setup for multi-node disaggregated serving.
#
# ENGINE=sglang-disagg or vllm-disagg selects the engine-specific block.
#
# REQUIRED ENVIRONMENT VARIABLES:
#   IBDEVICES - RDMA/InfiniBand device names (e.g., ionic_0,ionic_1,... or mlx5_0,mlx5_1,...)
#               Set by runner or auto-detected from hostname.
set -x

check_env_vars ENGINE
export PYTHONDONTWRITEBYTECODE=1

# job.slurm writes the recipe's HiCache/Mooncake tunables to hicache_mc_<JID>.env and
# mounts it at /config/hicache_mc.env. Source it (auto-export) so values like
# HICACHE_PAGE_SIZE=256 reach the container before server_sglang.sh validates them.
if [[ -f /config/hicache_mc.env ]]; then
    set -a
    source /config/hicache_mc.env
    set +a
    echo "[env.sh] sourced HiCache config from /config/hicache_mc.env (HICACHE_PAGE_SIZE=${HICACHE_PAGE_SIZE:-unset})"
fi

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

if [[ -z "$GLOO_SOCKET_IFNAME" ]]; then
    export GLOO_SOCKET_IFNAME=$(ip route 2>/dev/null | grep '^default' | awk '{print $5}' | head -n 1)
fi
if [[ -z "$NCCL_SOCKET_IFNAME" ]]; then
    export NCCL_SOCKET_IFNAME=$(ip route 2>/dev/null | grep '^default' | awk '{print $5}' | head -n 1)
fi

set +x

export NCCL_IB_HCA=${NCCL_IB_HCA:-$IBDEVICES}

# MoRI settings shared by the vLLM MoRIIOConnector and the SGLang/MoRI KV-transfer path.

export MORI_IO_SQ_BACKOFF_TIMEOUT_US
export MORI_IO_QP_MAX_SEND_WR
export MORI_IO_QP_MAX_CQE
export MORI_IO_QP_MAX_SGE
export MORI_IO_TC_DISABLE

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

if [[ "$ENGINE" == "vllm-disagg" ]]; then
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
    export UCX_IB_GID_INDEX

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

    export SGLANG_USE_AITER=1
    export AITER_LOG_LEVEL=ERROR

    export SGLANG_MORI_DISPATCH_DTYPE=auto
    export MORI_COMBINE_DTYPE_PREFILL=""
    export MORI_COMBINE_DTYPE_DECODE=""
    export SGLANG_MORI_QP_PER_TRANSFER=4
    export SGLANG_MORI_NUM_WORKERS=4

    export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600
    export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600

    export SGLANG_HEALTH_CHECK_TIMEOUT=600

    # GLM-5: uses NSA (not MLA), needs fused-decode-MLA disabled + fast loading
    if [[ "$MODEL_NAME" == "GLM-5-FP8" ]]; then
        export SGLANG_ROCM_FUSED_DECODE_MLA=0
        export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
        export SAFETENSORS_FAST_GPU=1
    fi

    # Disable allocating memory in one pass
    export MORI_SHMEM_MODE=ISOLATION

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
    export MORI_APP_LOG_LEVEL

    # SGLANG_ROUTER_STDOUT_LOGS=1 mirrors router logs to stdout via tee; 0 keeps the
    # noisy per-request access logs in the file only.
    export SGLANG_ROUTER_STDOUT_LOGS

    # Workaround for the 0305 upstream image.
    export PYTHONPATH=/sgl-workspace/aiter:${PYTHONPATH}

    # ROCm <= 7.2.0's HIP runtime does not honor cudaStreamCaptureModeThreadLocal, so
    # torch's ProcessGroupNCCL watchdog thread polling hipEventQuery() invalidates the
    # main thread's decode cuda-graph capture (hipErrorCapturedEvent -> "Rank 0
    # scheduler died during initialization"). Refs: sgl-project/sglang#29235, #24011;
    # ROCm/hip#3876; pytorch/pytorch#176251. Fixed in ROCm 7.2.2+; until the base image
    # is bumped, blocking NCCL waits avoid querying events during capture.
    export TORCH_NCCL_BLOCKING_WAIT
    export NCCL_BLOCKING_WAIT

    # DeepSeek-V4-Pro overrides; last in the block so they win over the defaults above.
    # The SGLANG_OPT_*/AITER_* knobs steer DSv4 off the default aiter CK fused-MoE path,
    # which raises "Unsupported kernel config for moe heuristic dispatch" at decode on
    # this fp4 model. CLI flags live in models.yaml; NIC/socket vars stay runner-derived.
    if [[ "$MODEL_NAME" == DeepSeek-V4-Pro* ]]; then
        export SGLANG_AITER_MLA_PERSIST=0
        ## resolve the OOR issue
        export HSA_NO_SCRATCH_RECLAIM=0
        export MORI_IO_QP_MAX_SEND_WR=32767
        # Unified radix tree: per-component (full-attn / SWA) cache management for
        # hybrid-attention models; set unconditionally, not gated on hicache.
        export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
        # Free out-of-window SWA KV slots during chunked prefill. Otherwise in-flight
        # requests pin SWA KV for their whole context, LRU flushes the trailing window of
        # cached sessions, and the prefix-cache hit rate collapses on multi-turn traces.
        export SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=1

        export SGLANG_MORI_DISPATCH_DTYPE=auto
        export MORI_COMBINE_DTYPE_PREFILL=auto
        export MORI_COMBINE_DTYPE_DECODE=auto

        export MORI_MAX_DISPATCH_TOKENS_PREFILL=8192
        export MORI_MAX_DISPATCH_TOKENS_DECODE=64
        unset MORI_MOE_MAX_INPUT_TOKENS_PREFILL
        unset MORI_MOE_MAX_INPUT_TOKENS_DECODE

        export SGLANG_MORI_RECV_BOUND=1

        # PER_RANK dispatch tokens pinned independently (16384 prefill / 128
        # decode); server_sglang.sh prefers these over the MORI_MAX_DISPATCH_*
        # coupling when set.
        export MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK_PREFILL=16384
        export MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK_DECODE=128

        export SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD=4096

        export SGLANG_DEFAULT_THINKING=1
        export SGLANG_DSV4_REASONING_EFFORT=high
        export SGLANG_USE_ROCM700A=0
        export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
        export SGLANG_OPT_FP8_WO_A_FUSED_INVROPE=1
        export AITER_BF16_FP8_MOE_BOUND=0
        export TORCH_BLAS_PREFER_HIPBLASLT=1
        # aiter batched GEMM for the absorbed MLA projections, carried by the v0.5.18
        # image and off by default in environ.py.
        export SGLANG_OPT_USE_AITER_BATCHED_GEMM=1
        # DP-attention-only SGLang internal knobs (shared-expert TP1 placement,
        # gatherv/reduce-scatter collectives) plus the wider HW-queue count DP
        # ranks need to overlap MoRI dispatch with compute.
        if [[ "$PREFILL_ENABLE_DP" == "true" || "$DECODE_ENABLE_DP" == "true" ]]; then
            export SGLANG_SHARED_EXPERT_TP1=1
            export SGLANG_DP_SHARED_EXPERT_LOCAL=1
            export SGLANG_DP_USE_GATHERV=1
            export SGLANG_DP_USE_REDUCE_SCATTER=1
            export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES_DP:-5}"
        else
            export GPU_MAX_HW_QUEUES=2
        fi
    fi

fi
