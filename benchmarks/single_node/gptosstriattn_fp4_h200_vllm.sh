#!/usr/bin/env bash
# TriAttention-enabled vLLM benchmark for GPT-OSS-120B FP4 on H200.
#
# Differences from baseline gptoss_fp4_h100.sh:
#   - Installs triattention vLLM plugin
#   - Sets TRIATTN_RUNTIME_KV_BUDGET (2048 for code, 12000 for chat workloads)
#   - Sets TRIATTN_RUNTIME_SPARSE_STATS_PATH when calibrated stats are available
#   - Lowers max-num-batched-tokens to 1024 (prevents OOM from large prefill chunks)
#   - Explicitly disables prefix caching (incompatible with KV compression)

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

hf download "$MODEL"

# --- TriAttention plugin setup ---
pip install -q triattention 2>/dev/null || echo "[TriAttention] Package not pre-installed; relying on container image."

# Auto-detect KV budget from export filename: chat workloads get larger budget.
TRIATTN_KV_BUDGET="${TRIATTN_RUNTIME_KV_BUDGET:-2048}"
if [[ "${EXPORT_FILE:-}" == *chat_* ]]; then
    TRIATTN_KV_BUDGET="${TRIATTN_RUNTIME_KV_BUDGET:-12000}"
fi
export TRIATTN_RUNTIME_KV_BUDGET="$TRIATTN_KV_BUDGET"

# Use pre-calibrated sparse stats if available on the runner.
TRIATTN_STATS="/workspace/triattn_stats/gpt_oss_120b_stats.pt"
if [[ -f "$TRIATTN_STATS" ]]; then
    export TRIATTN_RUNTIME_SPARSE_STATS_PATH="$TRIATTN_STATS"
    echo "[TriAttention] Using calibrated stats: $TRIATTN_STATS"
else
    echo "[TriAttention] No calibrated stats found at $TRIATTN_STATS; using budget-only compression."
fi

export ENABLE_TRIATTENTION=1
echo "[TriAttention] KV_BUDGET=$TRIATTN_KV_BUDGET  STATS=${TRIATTN_RUNTIME_SPARSE_STATS_PATH:-<none>}"
# --- End TriAttention setup ---

if is_isb1_replay_benchmark && [ -n "${MAX_MODEL_LEN:-}" ]; then
    MAX_MODEL_LEN="${MAX_MODEL_LEN}"
else
    MAX_MODEL_LEN=10240
fi

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

cat > config.yaml << EOF
enable-prefix-caching: false
max-cudagraph-capture-size: 2048
max-num-batched-tokens: 1024
max-model-len: $MAX_MODEL_LEN
EOF

if [[ -n "${VLLM_CPU_OFFLOAD_GB:-}" ]]; then
    echo "cpu-offload-gb: ${VLLM_CPU_OFFLOAD_GB}" >> config.yaml
fi
if [[ -n "${VLLM_SWAP_SPACE_GB:-}" ]]; then
    echo "swap-space: ${VLLM_SWAP_SPACE_GB}" >> config.yaml
fi
if [[ -n "${OFFLOAD_MODE:-}" ]]; then
    apply_vllm_offload_config
fi

export PYTHONNOUSERSITE=1
export VLLM_MXFP4_USE_MARLIN=1
SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

start_gpu_monitor
if [[ -n "${OFFLOAD_MODE:-}" ]]; then
    start_kv_metrics_collector "${PORT:-8888}" /workspace/kv_metrics.csv 2.0
fi

set -x
vllm serve $MODEL --host=0.0.0.0 --port=$PORT \
--config config.yaml \
--gpu-memory-utilization=0.9 \
--tensor-parallel-size=$TP \
--max-num-seqs=$CONC $VLLM_OFFLOAD_EXTRA_ARGS \
> $SERVER_LOG 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

run_single_node_benchmark \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts $(( $CONC * 10 )) \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --server-pid "$SERVER_PID"

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

if [[ -n "${OFFLOAD_MODE:-}" ]]; then
    stop_kv_metrics_collector
fi
stop_gpu_monitor
set +x
