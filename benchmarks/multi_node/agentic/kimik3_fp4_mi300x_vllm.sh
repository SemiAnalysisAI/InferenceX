#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    MODEL_PATH \
    PORT \
    CONC_LIST \
    PREFILL_NUM_WORKERS \
    PREFILL_TP \
    PREFILL_PP_SIZE \
    PREFILL_EP \
    PREFILL_DP_ATTN \
    DECODE_NUM_WORKERS \
    MULTINODE_NODE_COUNT \
    MULTINODE_GPUS_PER_NODE \
    MULTINODE_NODE_RANK \
    MULTINODE_MASTER_ADDR

require_agentic_kv_offload_none

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

if [[ "$MULTINODE_NODE_COUNT" != "2" ]]; then
    fail "this entrypoint serves exactly 2 nodes, got MULTINODE_NODE_COUNT=$MULTINODE_NODE_COUNT"
fi
if [[ "$MULTINODE_GPUS_PER_NODE" != "8" ]]; then
    fail "this entrypoint requires 8 GPUs per node, got MULTINODE_GPUS_PER_NODE=$MULTINODE_GPUS_PER_NODE"
fi
if [[ "$MULTINODE_NODE_RANK" != "0" && "$MULTINODE_NODE_RANK" != "1" ]]; then
    fail "MULTINODE_NODE_RANK must be 0 or 1, got '$MULTINODE_NODE_RANK'"
fi
if [[ "$PREFILL_TP" != "8" || "$PREFILL_PP_SIZE" != "2" ]]; then
    fail "this entrypoint serves only TP8 x PP2, got TP$PREFILL_TP x PP$PREFILL_PP_SIZE"
fi
if [[ "$PREFILL_EP" != "1" ]]; then
    fail "this entrypoint serves only EP1, got PREFILL_EP=$PREFILL_EP"
fi
if [[ "$PREFILL_DP_ATTN" != "false" ]]; then
    fail "this entrypoint does not enable DP attention, got PREFILL_DP_ATTN=$PREFILL_DP_ATTN"
fi
if [[ "$PREFILL_NUM_WORKERS" != "1" || "$DECODE_NUM_WORKERS" != "0" ]]; then
    fail "this entrypoint is aggregated: it needs 1 prefill worker and 0 decode workers, got ${PREFILL_NUM_WORKERS}P/${DECODE_NUM_WORKERS}D"
fi

read -r -a CONCURRENCIES <<< "$CONC_LIST"
if [[ "${#CONCURRENCIES[@]}" -ne 1 ]]; then
    fail "one concurrency per allocation is required, got CONC_LIST='$CONC_LIST'"
fi
case "${CONCURRENCIES[0]}" in
    1|2|4|8) ;;
    *) fail "concurrency must be 1, 2, 4, or 8, got '${CONCURRENCIES[0]}'" ;;
esac

if [[ -n "${AITER_SITUV2_A8W4+set}" ]]; then
    if [[ "$AITER_SITUV2_A8W4" != "0" && "$AITER_SITUV2_A8W4" != "1" ]]; then
        fail "AITER_SITUV2_A8W4 must be 0 or 1 when set, got '$AITER_SITUV2_A8W4'"
    fi
fi

export VLLM_ROCM_USE_AITER=1
export SAFETENSORS_FAST_GPU=1
export AITER_BF16_FP8_MOE_BOUND=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
export PYTHONNOUSERSITE=1

VLLM_CMD=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size 8
    --pipeline-parallel-size 2
    --nnodes 2
    --node-rank "$MULTINODE_NODE_RANK"
    --master-addr "$MULTINODE_MASTER_ADDR"
    --trust-remote-code
    --load-format auto
    --moe-backend auto
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
    --max-model-len 1048576
    --max-num-seqs "$CONC_LIST"
    --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
    --mm-encoder-tp-mode data
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --language-model-only
)
if [[ "$MULTINODE_NODE_RANK" == "1" ]]; then
    VLLM_CMD+=(--headless)
fi

echo "AITER_SITUV2_A8W4=${AITER_SITUV2_A8W4-unset}"
printf 'vLLM command:'
printf ' %q' "${VLLM_CMD[@]}"
printf '\n'

if [[ "${KIMIK3_VLLM_DRY_RUN:-0}" == "1" ]]; then
    echo "KIMIK3_VLLM_DRY_RUN=1 set; not starting the server"
    exit 0
fi

exec "${VLLM_CMD[@]}"
