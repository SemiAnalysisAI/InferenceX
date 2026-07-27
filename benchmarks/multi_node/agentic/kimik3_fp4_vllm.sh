#!/usr/bin/env bash
set -euo pipefail
set -x

# Scheduler- and hardware-independent aggregated vLLM server for Kimi K3 MXFP4.
#
# The benchmark configuration selects one of the official aggregate strategies:
#   https://recipes.vllm.ai/moonshotai/Kimi-K3?hardware=h200
#
# The runner connector supplies node ranks, rendezvous, and local GPU count.
# This script only translates the requested TP/EP/DP topology into vLLM flags.

# shellcheck disable=SC1091 # Resolved relative to this entrypoint at runtime.
source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    MODEL_PATH \
    PORT \
    CONC_LIST \
    PREFILL_TP \
    PREFILL_EP \
    PREFILL_DP_ATTN \
    DECODE_NUM_WORKERS \
    MULTINODE_NODE_COUNT \
    MULTINODE_GPUS_PER_NODE \
    MULTINODE_NODE_RANK \
    MULTINODE_MASTER_ADDR

NODE_COUNT="$MULTINODE_NODE_COUNT"
GPUS_PER_NODE="$MULTINODE_GPUS_PER_NODE"
NODE_RANK="$MULTINODE_NODE_RANK"
MASTER_ADDR="$MULTINODE_MASTER_ADDR"

for topology_value in \
    "$NODE_COUNT" \
    "$GPUS_PER_NODE" \
    "$NODE_RANK" \
    "$PREFILL_TP" \
    "$PREFILL_EP" \
    "$DECODE_NUM_WORKERS"; do
    if ! [[ "$topology_value" =~ ^[0-9]+$ ]]; then
        echo "Invalid multi-node topology value: $topology_value" >&2
        exit 1
    fi
done
if [ "$NODE_COUNT" -eq 0 ] || [ "$GPUS_PER_NODE" -eq 0 ] || \
   [ "$NODE_RANK" -ge "$NODE_COUNT" ]; then
    echo "Invalid multi-node rank layout: rank=$NODE_RANK nodes=$NODE_COUNT GPUs/node=$GPUS_PER_NODE" >&2
    exit 1
fi
if [ "$PREFILL_TP" -eq 0 ] || [ "$PREFILL_EP" -eq 0 ]; then
    echo "PREFILL_TP and PREFILL_EP must be positive" >&2
    exit 1
fi
WORLD_SIZE=$((NODE_COUNT * GPUS_PER_NODE))
if [ "$DECODE_NUM_WORKERS" -ne 0 ]; then
    echo "The aggregated Kimi K3 entrypoint does not accept decode workers" >&2
    exit 1
fi

read -r -a CONCURRENCIES <<< "$CONC_LIST"
MAX_CONCURRENCY=0
for concurrency in "${CONCURRENCIES[@]}"; do
    if ! [[ "$concurrency" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid AgentX concurrency: $concurrency" >&2
        exit 1
    fi
    if [ "$concurrency" -gt "$MAX_CONCURRENCY" ]; then
        MAX_CONCURRENCY="$concurrency"
    fi
done

export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
export PYTHONNOUSERSITE=1

VLLM_CMD=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --load-format fastsafetensors
    --moe-backend "${VLLM_MOE_BACKEND:-auto}"
    --gpu-memory-utilization 0.95
    --enable-prefix-caching
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --language-model-only
    --max-model-len 1048576
    --max-num-seqs "$MAX_CONCURRENCY"
)

if [[ "${VLLM_DISABLE_FLASHINFER_AUTOTUNE:-0}" == "1" ]]; then
    VLLM_CMD+=(--no-enable-flashinfer-autotune)
fi
if [[ "${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-0}" == "1" ]]; then
    VLLM_CMD+=(--disable-custom-all-reduce)
fi

if [[ "$PREFILL_DP_ATTN" == "true" ]]; then
    DEPLOYMENT_ORIENTATION="throughput"
    VLLM_CMD+=(
        --data-parallel-size "$WORLD_SIZE"
        --data-parallel-size-local "$GPUS_PER_NODE"
        --data-parallel-address "$MASTER_ADDR"
        --data-parallel-start-rank "$((NODE_RANK * GPUS_PER_NODE))"
        --data-parallel-hybrid-lb
    )
    if [ "$PREFILL_EP" -gt 1 ]; then
        VLLM_CMD+=(--enable-expert-parallel)
    fi
elif [[ "$PREFILL_DP_ATTN" == "false" ]]; then
    if [ "$PREFILL_TP" -ne "$WORLD_SIZE" ]; then
        echo "Cross-node TP ($PREFILL_TP) must match the allocated world size ($WORLD_SIZE)" >&2
        exit 1
    fi

    DEPLOYMENT_ORIENTATION="latency"
    VLLM_CMD+=(
        --tensor-parallel-size "$PREFILL_TP"
        --nnodes "$NODE_COUNT"
        --node-rank "$NODE_RANK"
        --master-addr "$MASTER_ADDR"
    )
    if [ "$PREFILL_EP" -gt 1 ]; then
        DEPLOYMENT_ORIENTATION="balanced"
        VLLM_CMD+=(--enable-expert-parallel)
    fi
    if [[ "${VLLM_DISABLE_FUSED_ALLREDUCE_RMS:-0}" == "1" ]]; then
        VLLM_CMD+=(-cc.pass_config.fuse_allreduce_rms=False)
    fi
    if [ "$NODE_RANK" -gt 0 ]; then
        VLLM_CMD+=(--headless)
    fi
else
    echo "PREFILL_DP_ATTN must be true or false, got: $PREFILL_DP_ATTN" >&2
    exit 1
fi

printf 'Kimi K3 %s-oriented rank %s command: ' "$DEPLOYMENT_ORIENTATION" "$NODE_RANK"
printf '%q ' "${VLLM_CMD[@]}"
printf '\n'

if [[ "${KIMIK3_DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

exec "${VLLM_CMD[@]}"
