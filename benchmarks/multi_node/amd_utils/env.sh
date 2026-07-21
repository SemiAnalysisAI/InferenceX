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
# HiCache / Mooncake settings from job.slurm
# =============================================================================
# job.slurm writes the recipe-provided HiCache/Mooncake tunables to
# hicache_mc_<JID>.env and mounts it read-only at /config/hicache_mc.env. Source
# it here (auto-export) so values like HICACHE_PAGE_SIZE=256 reach the container
# before server_sglang.sh applies its "${VAR:-default}" fallbacks. Without this
# the vars arrive unset and server_sglang.sh defaults HICACHE_PAGE_SIZE to 1,
# overriding the recipe's --page-size. Empty values in the file are harmless:
# the "${VAR:-default}" fallbacks still treat "" as unset.
if [[ -f /config/hicache_mc.env ]]; then
    set -a
    source /config/hicache_mc.env
    set +a
    echo "[env.sh] sourced HiCache config from /config/hicache_mc.env (HICACHE_PAGE_SIZE=${HICACHE_PAGE_SIZE:-unset})"
fi

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

export MORI_IO_SQ_BACKOFF_TIMEOUT_US="${MORI_IO_SQ_BACKOFF_TIMEOUT_US:-50000}"
export MORI_IO_QP_MAX_SEND_WR="${MORI_IO_QP_MAX_SEND_WR:-16384}"
export MORI_IO_QP_MAX_CQE="${MORI_IO_QP_MAX_CQE:-32768}"
export MORI_IO_QP_MAX_SGE="${MORI_IO_QP_MAX_SGE:-2}"
export MORI_IO_TC_DISABLE="${MORI_IO_TC_DISABLE:-0}"

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
    # export MORI_COMBINE_DTYPE_PREFILL=fp8_direct_cast
    # export MORI_COMBINE_DTYPE_DECODE=fp8
    export MORI_COMBINE_DTYPE_PREFILL=""
    export MORI_COMBINE_DTYPE_DECODE=""
    export SGLANG_MORI_QP_PER_TRANSFER=4
    export SGLANG_MORI_NUM_WORKERS=4
    # Keep these as overridable defaults (not hard assignments), otherwise
    # later tuning blocks cannot raise them for high-concurrency runs.
    # export MORI_IO_SQ_BACKOFF_TIMEOUT_US="${MORI_IO_SQ_BACKOFF_TIMEOUT_US:-500000}"

    # export MORI_IO_QP_MAX_SEND_WR="${MORI_IO_QP_MAX_SEND_WR:-16384}"
    # export MORI_IO_QP_MAX_CQE=32768
    # export MORI_IO_QP_MAX_SGE=1

    # export MORI_IO_TC_DISABLE=0

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

    # FIXME: WA for latest upstream 0305 image
    export PYTHONPATH=/sgl-workspace/aiter:${PYTHONPATH}

    # Decode CUDA-graph capture crash on ROCm 7.2.0 (TP8+EP8, mori a2a).
    # Symptom: during decode cuda-graph capture, the torch ProcessGroupNCCL
    # *watchdog* thread calls hipEventQuery() to poll in-flight NCCL work.
    # ROCm <= 7.2.0's HIP runtime does NOT honor cudaStreamCaptureModeThreadLocal,
    # so the watchdog's cross-thread query touches the main thread's active
    # capture and invalidates it -> "HIP error: operation not permitted on an
    # event last recorded in a capturing stream (hipErrorCapturedEvent)" ->
    # watchdog aborts -> "Rank 0 scheduler died during initialization" (-6).
    # This is a HIP runtime bug, not OOM and not a mori/deepep-mode bug (it
    # fires for --deepep-mode normal and auto alike; EP8+mori just adds NCCL
    # PGs that make the watchdog race fire). Refs: sgl-project/sglang#29235,
    # #24011; ROCm/hip#3876; pytorch/pytorch#176251.
    # Real fix = ROCm 7.2.2+ (honors THREAD_LOCAL). Until the base image is
    # bumped, TORCH_NCCL_BLOCKING_WAIT=true makes NCCL work completion use a
    # blocking wait instead of the async watchdog hipEventQuery poll, so no
    # event is queried during capture. CUDA graph stays fully enabled.
    export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
    export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
    export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

    # =========================================================================
    # DeepSeek-V4-Pro PD recipe overrides
    # Placed at the end of the SGLang env block so it wins over the global
    # MoRI/SGLang defaults set above. Mirrors the validated DSv4 manual PD
    # commands (ported from InferenceX amd/dsv4_sgl_di). These SGLANG_OPT_* /
    # AITER_* kernel-routing knobs steer DSv4 away from the default aiter CK
    # fused-MoE path, which raises "Unsupported kernel config for moe heuristic
    # dispatch" at decode time on this fp4 model (job 19034 crash). Only the
    # SGLang/MoRI env knobs are pinned here; CLI flags live in models.yaml and
    # the cluster NIC/socket vars stay runner-derived.
    # =========================================================================
    if [[ "$MODEL_NAME" == "DeepSeek-V4-Pro" ]]; then
        # MoRI RDMA send-queue depth for DSv4 (overrides the global default above).
        export MORI_IO_QP_MAX_SEND_WR=32767
        # Unified radix tree: cache impl with per-component (full-attn / SWA)
        # management for hybrid-attention models. Set unconditionally (not gated on
        # hicache) so all SGLang runs use it.
        export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
        # Proactively free out-of-window SWA KV slots during chunked prefill.
        # Without it, in-flight requests pin SWA KV for their whole context, keeping
        # the SWA pool under constant eviction pressure; under LRU the trailing
        # window of cached sessions gets flushed, making prefix-cache hits bimodal
        # and collapsing the effective hit rate on multi-turn agentic workloads.
        export SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=1

        # MoRI dispatch/combine dtypes: auto for both roles (not the fp8 split default)
        export SGLANG_MORI_DISPATCH_DTYPE=auto
        export MORI_COMBINE_DTYPE_PREFILL=auto
        export MORI_COMBINE_DTYPE_DECODE=auto

        # Per-role MoRI dispatch sizing (used by the harness chunked/MoE math)
        export MORI_MAX_DISPATCH_TOKENS_PREFILL=8192
        export MORI_MAX_DISPATCH_TOKENS_DECODE=64
        unset MORI_MOE_MAX_INPUT_TOKENS_PREFILL
        unset MORI_MOE_MAX_INPUT_TOKENS_DECODE

        # PER_RANK dispatch tokens pinned independently (16384 prefill / 128
        # decode); server_sglang.sh prefers these over the MORI_MAX_DISPATCH_*
        # coupling when set.
        export MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK_PREFILL=16384
        export MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK_DECODE=128

        # Fixed inter-kernel switch threshold (not derived).
        export SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD=4096

        # Overlap plan stream on for DSv4 (global default is 0)
        # export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0

        # DSv4 model kernel routing (mirrors the single-node / manual PD recipe)
        export SGLANG_DEFAULT_THINKING=1
        export SGLANG_DSV4_REASONING_EFFORT=high
        export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
        export SGLANG_USE_AITER=1
        export SGLANG_USE_ROCM700A=0
        export SGLANG_OPT_USE_FUSED_COMPRESS=true
        export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
        export SGLANG_OPT_FP8_WO_A_GEMM=false
        export SGLANG_OPT_USE_JIT_INDEXER_METADATA=false
        export SGLANG_OPT_USE_TOPK_V2=false
        export SGLANG_OPT_USE_AITER_INDEXER=${SGLANG_OPT_USE_AITER_INDEXER:-true}
        export SGLANG_OPT_USE_TILELANG_INDEXER=false
        export SGLANG_OPT_USE_TILELANG_MHC_PRE=false
        export SGLANG_OPT_USE_TILELANG_MHC_POST=false
        export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
        export SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true
        export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false
        export SGLANG_ROCM_USE_MULTI_STREAM=false
        export AITER_BF16_FP8_MOE_BOUND=0
        export SGLANG_EAGER_INPUT_NO_COPY=true
        export SGLANG_SHARED_EXPERT_TP1=1
        export SGLANG_DP_SHARED_EXPERT_LOCAL=1
        export SGLANG_DP_USE_GATHERV=1
        export SGLANG_DP_USE_REDUCE_SCATTER=1
        export GPU_MAX_HW_QUEUES=5
    fi

fi