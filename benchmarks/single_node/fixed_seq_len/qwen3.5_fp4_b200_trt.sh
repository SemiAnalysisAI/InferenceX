#!/usr/bin/env bash

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    DP_ATTENTION \
    EP_SIZE

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL, EP_SIZE: $EP_SIZE, DP_ATTENTION: $DP_ATTENTION"

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

nvidia-smi

SERVER_LOG=/workspace/server.log
EXTRA_CONFIG_FILE="qwen3.5-fp4-trt.yml"

if [[ "$DP_ATTENTION" == "true" ]]; then
    case "$TP" in
        4) MAX_BATCH_SIZE=256 ;;   # tp4 / ep4 with attention DP
        8) MAX_BATCH_SIZE=128 ;;   # tp8 / ep8 with attention DP
        *) MAX_BATCH_SIZE=$(( CONC > 16 ? CONC : 16 )) ;;
    esac
elif [[ "$TP" == "2" ]]; then
    if [[ "$EP_SIZE" == "2" && "$CONC" -ge 32 ]]; then
        MAX_BATCH_SIZE=32          # tp2 / ep2 at high concurrency
    else
        MAX_BATCH_SIZE=256         # tp2 / ep1, or tp2 / ep2 at low concurrency
    fi
elif [[ "$TP" -ge 4 ]]; then
    MAX_BATCH_SIZE=512             # tp>=4 without attention DP
else
    MAX_BATCH_SIZE=$(( CONC > 16 ? CONC : 16 ))
fi

if [[ "$DP_ATTENTION" == "true" ]]; then
    MOE_BACKEND="CUTEDSL"
    MODE_CONFIG="attention_dp_config:
    enable_balance: true
    batching_wait_iters: 10
    timeout_iters: 500"
else
    MOE_BACKEND="TRTLLM"
    MODE_CONFIG="batch_wait_timeout_iters: 50
batch_wait_max_tokens_ratio: 0.45"
fi

cat > "$EXTRA_CONFIG_FILE" << EOF
backend: pytorch
print_iter_log: true
enable_layerwise_nvtx_marker: false
disable_overlap_scheduler: false
enable_iter_perf_stats: true
enable_chunked_prefill: false
stream_interval: 20
num_postprocess_workers: 4
enable_attention_dp: $DP_ATTENTION
scheduler_config:
    capacity_scheduler_policy: MAX_UTILIZATION
    context_chunking_policy: FIRST_COME_FIRST_SERVED
kv_cache_config:
    free_gpu_memory_fraction: 0.9
    enable_block_reuse: false
    dtype: fp8
cuda_graph_config:
    enable_padding: true
    max_batch_size: $MAX_BATCH_SIZE
moe_config:
    backend: $MOE_BACKEND
    use_low_precision_moe_combine: true
$MODE_CONFIG
EOF

echo "Generated config file contents:"
cat "$EXTRA_CONFIG_FILE"

MAX_MODEL_LEN=$(( MAX_MODEL_LEN > 8192 ? MAX_MODEL_LEN : 8192 ))

case "${ISL}_${OSL}" in
    8192_1024) MAX_NUM_TOKENS=32768 ;;
    1024_1024) MAX_NUM_TOKENS=16384 ;;
    *)
        MAX_NUM_TOKENS=$(( ISL + OSL + 256 ))
        MAX_NUM_TOKENS=$(( MAX_NUM_TOKENS > 8192 ? MAX_NUM_TOKENS : 8192 ))
        ;;
esac

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
    MAX_NUM_TOKENS="$EVAL_MAX_MODEL_LEN"
fi

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
mpirun -n 1 --oversubscribe --allow-run-as-root \
    trtllm-serve "$MODEL" --port="$PORT" \
    --trust_remote_code \
    --backend=pytorch \
    --max_batch_size "$MAX_BATCH_SIZE" \
    --max_seq_len="$MAX_MODEL_LEN" \
    --max_num_tokens="$MAX_NUM_TOKENS" \
    --tp_size="$TP" --ep_size="$EP_SIZE" \
    --extra_llm_api_options="$EXTRA_CONFIG_FILE" \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend openai \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$(( CONC * 10 ))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
