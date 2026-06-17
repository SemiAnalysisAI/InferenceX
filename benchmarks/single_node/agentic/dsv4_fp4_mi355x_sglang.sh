#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for DeepSeek-V4-Pro FP4 on MI355X using SGLang.
# Adapted from benchmarks/single_node/dsv4_fp4_mi355x_sglang.sh (fixed-seq-len
# sibling) with the agentic harness (build_replay_cmd / write_agentic_result_json
# / analyze_benchmark_distributions) swapped in for run_benchmark_serving.
#
# This launcher does NOT support CPU offload. SGLang's KV offload paths are
# different from vLLM's SimpleCPUOffloadConnector, and the matching agentic
# config (dsv4-fp4-mi355x-sglang-agentic) only sweeps offloading=none.
#
# Required env vars:
#   MODEL, TP, CONC, OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR

PORT=8787

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

if [ -z "${MAX_MODEL_LEN:-}" ] || [ "$MAX_MODEL_LEN" = "0" ]; then
    MAX_MODEL_LEN=1000000
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# ROCR/HIP visibility under slurm cgroups.
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

# `hf download` creates the target dir if missing and is itself idempotent.
# When MODEL_PATH is unset (stand-alone runs), fall back to the HF_HUB_CACHE
# Either way, MODEL_PATH is what the server is launched with.
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi
rocm-smi || true
amd-smi || true

# ---- Resolve traces and install deps ----------------------------------------
# https://huggingface.co/datasets/semianalysisai/cc-traces-weka-with-subagents-060826
 export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_with_subagents_060826

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

# ---- Hicache config ----------------------------------------------------------
# Reject anything other than none: this launcher has no SGLang CPU-offload
# wiring (different surface than vLLM's SimpleCPUOffloadConnector).

case "$OFFLOADING" in
    none)
        # Leave SGLang's default RadixAttention prefix cache on — agentic
        # replay needs it; --disable-radix-cache would zero the hit rate.
        ;;
    hicache)
        # DeepSeek V4 HiCache uses ratio-based capacity control, not GB-based.
        # DSv4 allocates several physical host sub-pools for each logical host
        # token. MI355X nodes have ~3 TB of host DRAM (similar to B200's 3.8
        # TiB), so ratio=8 at TP≥8 provides a large useful CPU tier within the
        # node budget. Lower TP configs use higher ratios to maintain adequate
        # host token capacity without exceeding DRAM limits.
        if [ "$TP" -ge 8 ]; then
            DEFAULT_HICACHE_RATIO=2
        else
            DEFAULT_HICACHE_RATIO=16
        fi
        HICACHE_RATIO="${HICACHE_RATIO:-$DEFAULT_HICACHE_RATIO}"
        HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
        HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
        HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first_direct}"
        export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
        CACHE_ARGS=(
            --enable-hierarchical-cache
            --hicache-ratio "$HICACHE_RATIO"
            --hicache-write-policy "$HICACHE_WRITE_POLICY"
            --hicache-io-backend "$HICACHE_IO_BACKEND"
            --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
        )
        echo "HiCache DSv4 CPU tier: ratio=$HICACHE_RATIO, write_policy=$HICACHE_WRITE_POLICY, io_backend=$HICACHE_IO_BACKEND, mem_layout=$HICACHE_MEM_LAYOUT"
        ;;
    *)
        echo "Error: unsupported OFFLOADING value '$OFFLOADING' (expected one of: none, hicache)" >&2
        exit 1
        ;;
esac

# ---- LLM server config ----------------------------------------------------------

WARMUP_ARGS=()
CUDA_GRAPH_MAX_BS="$CONC"
[ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64

export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=0
export SGLANG_OPT_USE_FUSED_COMPRESS=true
#export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export SGLANG_OPT_FP8_WO_A_GEMM=false
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=false
export SGLANG_OPT_USE_TOPK_V2=false
export SGLANG_OPT_USE_AITER_INDEXER=true
export SGLANG_OPT_USE_TILELANG_INDEXER=false
export SGLANG_OPT_USE_TILELANG_MHC_PRE=false
export SGLANG_OPT_USE_TILELANG_MHC_POST=false
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true
export AITER_BF16_FP8_MOE_BOUND=0
export SGLANG_EAGER_INPUT_NO_COPY=true

# multi-stream
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false
export SGLANG_ROCM_USE_MULTI_STREAM=false

# relax timeout 
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

# tree modification
export SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1

# Parallelism: pure TP, TP+EP, or DEP (DP-attn + EP). Matches the dsv4 b200
# vllm agentic launcher so the agentic sweep can probe both interactivity and
# throughput regimes.
PARALLEL_ARGS=(--tensor-parallel-size "$TP")
if [ "$DP_ATTENTION" = "true" ]; then
    PARALLEL_ARGS+=(
        --dp "$TP"
        --enable-dp-attention
        --enable-prefill-delayer
    )
fi
if [ "${EP_SIZE:-1}" -gt 1 ]; then
    PARALLEL_ARGS+=(--ep-size "$EP_SIZE")
fi

# --max-running-requests is per-engine. With DP-attn each DP engine handles
# only CONC/$TP sequences in steady state (the agentic harness load-balances
# users across DP ranks), so size the per-engine cap to that.
# Pure TP is a single engine and sees all CONC sequences itself.
if [ "$DP_ATTENTION" = "true" ]; then
    PER_ENGINE_MAX_RUNNING=$(( CONC / TP ))
    [ "$PER_ENGINE_MAX_RUNNING" -lt 1 ] && PER_ENGINE_MAX_RUNNING=1
else
    PER_ENGINE_MAX_RUNNING=$CONC
fi

set -x
echo "Starting sglang server..."

#image: lmsysorg/sglang-rocm:v0.5.12.post1-rocm720-mi35x-20260610
#    --page-size 256 \
#image: lmsysorg/sglang-rocm:v0.5.13.post1-rocm700-mi35x-20260616 
#    --page-size 1 \
sglang serve \
    --model-path $MODEL \
    --host=0.0.0.0 \
    --port $PORT \
    "${PARALLEL_ARGS[@]}" \
    --trust-remote-code \
    --attention-backend dsv4 \
    --max-running-requests ${CONC} \
    --mem-fraction-static 0.90 \
    --swa-full-tokens-ratio 0.15 \
    --page-size 1 \
    --context-length $MAX_MODEL_LEN \
    --chunked-prefill-size 8192 \
    --disable-shared-experts-fusion \
    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4 \
    --chat-template "$(dirname "$0")/../chat_templates/deepseek_v4_thinking.jinja" \
    --watchdog-timeout 1800 \
    --enable-metrics \
    "${CACHE_ARGS[@]}" \
    "${WARMUP_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"

run_agentic_replay_and_write_outputs "$RESULT_DIR"
