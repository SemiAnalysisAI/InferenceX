#!/usr/bin/env bash
set -euo pipefail
set -x

# Two-node aggregated vLLM server for Kimi K3 MXFP4 on H200.
#
# Official recipe:
#   https://recipes.vllm.ai/moonshotai/Kimi-K3?hardware=h200
#
# Strategy mapping from vllm-project/recipes:
#   latency    -> multi_node_tp  (TP16 across two nodes)
#   balanced   -> multi_node_tep (TP16 + expert parallelism)
#   throughput -> multi_node_dep (DEP16, eight local DP ranks per node)

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL MODEL_PATH PORT KIMIK3_STRATEGY CONC_LIST

NODE_COUNT="${KIMIK3_NODE_COUNT:-2}"
GPUS_PER_NODE="${KIMIK3_GPUS_PER_NODE:-8}"
NODE_RANK="${SLURM_PROCID:?SLURM_PROCID must identify the vLLM node rank}"
MASTER_ADDR="${KIMIK3_MASTER_ADDR:?KIMIK3_MASTER_ADDR must identify rank zero}"

if [ "$NODE_COUNT" -ne 2 ] || [ "$GPUS_PER_NODE" -ne 8 ]; then
    echo "Kimi K3 H200 Day-0 recipes require two nodes with eight GPUs each" >&2
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

export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
export PYTHONNOUSERSITE=1

VLLM_CMD=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --load-format fastsafetensors
    --moe-backend marlin
    --gpu-memory-utilization 0.95
    --no-enable-flashinfer-autotune
    --disable-custom-all-reduce
    --enable-prefix-caching
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --language-model-only
    --max-model-len 1048576
    --max-num-seqs "$MAX_CONCURRENCY"
)

case "$KIMIK3_STRATEGY" in
    latency)
        VLLM_CMD+=(
            --tensor-parallel-size "$((NODE_COUNT * GPUS_PER_NODE))"
            --nnodes "$NODE_COUNT"
            --node-rank "$NODE_RANK"
            --master-addr "$MASTER_ADDR"
            -cc.pass_config.fuse_allreduce_rms=False
        )
        if [ "$NODE_RANK" -gt 0 ]; then
            VLLM_CMD+=(--headless)
        fi
        ;;
    balanced)
        VLLM_CMD+=(
            --tensor-parallel-size "$((NODE_COUNT * GPUS_PER_NODE))"
            --nnodes "$NODE_COUNT"
            --node-rank "$NODE_RANK"
            --master-addr "$MASTER_ADDR"
            --enable-expert-parallel
            -cc.pass_config.fuse_allreduce_rms=False
        )
        if [ "$NODE_RANK" -gt 0 ]; then
            VLLM_CMD+=(--headless)
        fi
        ;;
    throughput)
        VLLM_CMD+=(
            --data-parallel-size "$((NODE_COUNT * GPUS_PER_NODE))"
            --data-parallel-size-local "$GPUS_PER_NODE"
            --data-parallel-address "$MASTER_ADDR"
            --data-parallel-start-rank "$((NODE_RANK * GPUS_PER_NODE))"
            --data-parallel-hybrid-lb
            --enable-expert-parallel
        )
        ;;
    *)
        echo "Unsupported KIMIK3_STRATEGY: $KIMIK3_STRATEGY" >&2
        exit 1
        ;;
esac

printf 'Kimi K3 rank %s command: ' "$NODE_RANK"
printf '%q ' "${VLLM_CMD[@]}"
printf '\n'

if [[ "${KIMIK3_DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

exec "${VLLM_CMD[@]}"
