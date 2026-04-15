#!/usr/bin/env bash

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

hf download "$MODEL"

if is_isb1_replay_benchmark && [ -n "${MAX_MODEL_LEN:-}" ]; then
    MAX_MODEL_LEN="${MAX_MODEL_LEN}"
else
    MAX_MODEL_LEN=10240
fi

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

PREFIX_CACHING_CONFIG="no-enable-prefix-caching: true"
if is_isb1_replay_benchmark; then
    PREFIX_CACHING_CONFIG=""
fi
if [[ -n "${OFFLOAD_MODE:-}" ]]; then
    apply_vllm_offload_config
fi

cat > config.yaml << EOF
$PREFIX_CACHING_CONFIG
max-cudagraph-capture-size: 2048
max-num-batched-tokens: 8192
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

# Start GPU monitoring (power, temperature, clocks every second)
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

# Wait for server to be ready
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

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
if [[ -n "${OFFLOAD_MODE:-}" ]]; then
    stop_kv_metrics_collector
fi
stop_gpu_monitor
set +x
