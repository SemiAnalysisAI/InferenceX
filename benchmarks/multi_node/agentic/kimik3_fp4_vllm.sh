#!/usr/bin/env bash
set -euo pipefail
set -x

# Scheduler- and hardware-independent aggregated vLLM server for Kimi K3 MXFP4.
#
# The benchmark configuration selects one of the official aggregate strategies:
#   https://recipes.vllm.ai/moonshotai/Kimi-K3?hardware=h200
#
# The runner supplies node ranks, rendezvous, and local GPU count.
# This script only translates the requested TP/EP/DP topology into vLLM flags.

WORLD_SIZE=$((MULTINODE_NODE_COUNT * MULTINODE_GPUS_PER_NODE))
read -r -a CONCURRENCIES <<< "$CONC_LIST"
MAX_CONCURRENCY=0
for concurrency in "${CONCURRENCIES[@]}"; do
    if [ "$concurrency" -gt "$MAX_CONCURRENCY" ]; then
        MAX_CONCURRENCY="$concurrency"
    fi
done
MAX_NUM_SEQS="$MAX_CONCURRENCY"
if [[ "$PREFILL_DP_ATTN" == "true" ]]; then
    MAX_NUM_SEQS=$(((MAX_CONCURRENCY + PREFILL_NUM_WORKERS - 1) / PREFILL_NUM_WORKERS))
fi

export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_USE_V2_MODEL_RUNNER=1
if [ "$PREFILL_PP_SIZE" -gt 1 ]; then
    export VLLM_USE_V2_MODEL_RUNNER=0
fi
export PYTHONNOUSERSITE=1

LOAD_FORMAT=auto
if [ "$PREFILL_PP_SIZE" -gt 1 ]; then
    LOAD_FORMAT=fastsafetensors
fi

VLLM_CMD=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --load-format "$LOAD_FORMAT"
    --moe-backend "$VLLM_MOE_BACKEND"
    --gpu-memory-utilization 0.95
    --enable-prefix-caching
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --language-model-only
    --max-num-seqs "$MAX_NUM_SEQS"
)

if [ "$PREFILL_PP_SIZE" -gt 1 ]; then
    VLLM_CMD+=(--attention-backend FLASHMLA)
fi
if [[ "$VLLM_DISABLE_FLASHINFER_AUTOTUNE" == "1" ]]; then
    VLLM_CMD+=(--no-enable-flashinfer-autotune)
fi
if [[ "$VLLM_DISABLE_CUSTOM_ALL_REDUCE" == "1" ]]; then
    VLLM_CMD+=(--disable-custom-all-reduce)
fi
if [[ "$PREFILL_DP_ATTN" == "true" ]]; then
    LOCAL_DATA_PARALLEL_SIZE=$((MULTINODE_GPUS_PER_NODE / (PREFILL_TP * PREFILL_PP_SIZE)))
    VLLM_CMD+=(
        --tensor-parallel-size "$PREFILL_TP"
        --pipeline-parallel-size "$PREFILL_PP_SIZE"
        --data-parallel-size "$PREFILL_NUM_WORKERS"
        --data-parallel-size-local "$LOCAL_DATA_PARALLEL_SIZE"
        --data-parallel-address "$MULTINODE_MASTER_ADDR"
        --data-parallel-start-rank "$((MULTINODE_NODE_RANK * LOCAL_DATA_PARALLEL_SIZE))"
        --data-parallel-hybrid-lb
    )
    if [ "$PREFILL_EP" -gt 1 ]; then
        VLLM_CMD+=(--enable-expert-parallel)
    fi
elif [[ "$PREFILL_DP_ATTN" == "false" ]]; then
    if [ "$((PREFILL_TP * PREFILL_PP_SIZE))" -ne "$WORLD_SIZE" ]; then
        echo "Cross-node TPxPP must match the allocated world size ($WORLD_SIZE)" >&2
        exit 1
    fi

    VLLM_CMD+=(
        --tensor-parallel-size "$PREFILL_TP"
        --pipeline-parallel-size "$PREFILL_PP_SIZE"
        --nnodes "$MULTINODE_NODE_COUNT"
        --node-rank "$MULTINODE_NODE_RANK"
        --master-addr "$MULTINODE_MASTER_ADDR"
    )
    if [ "$PREFILL_EP" -gt 1 ]; then
        VLLM_CMD+=(--enable-expert-parallel)
    fi
    if [[ "$VLLM_DISABLE_FUSED_ALLREDUCE_RMS" == "1" ]]; then
        VLLM_CMD+=(-cc.pass_config.fuse_allreduce_rms=False)
    fi
    if [ "$MULTINODE_NODE_RANK" -gt 0 ]; then
        VLLM_CMD+=(--headless)
    fi
else
    echo "PREFILL_DP_ATTN must be true or false, got: $PREFILL_DP_ATTN" >&2
    exit 1
fi

printf 'Kimi K3 rank %s command: ' "$MULTINODE_NODE_RANK"
printf '%q ' "${VLLM_CMD[@]}"
printf '\n'

exec "${VLLM_CMD[@]}"
