#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    EP_SIZE \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    DP_ATTENTION

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

echo "TP: $TP, EP: $EP_SIZE, CONC: $CONC, ISL: $ISL, OSL: $OSL, DP_ATTENTION: $DP_ATTENTION"

hf download "$MODEL"

# Derive max_batch_size from (TP, DP_ATTENTION). For dp-attn the DEP-8 case uses 128, DEP-4 uses 256.
# For non-dp-attn, TP=2 uses 256 (retained for compat), else 512.
if [[ "$DP_ATTENTION" == "true" ]]; then
    if [[ "$TP" == "8" ]]; then
        MAX_BATCH_SIZE=128
    else
        MAX_BATCH_SIZE=256
    fi
else
    if [[ "$TP" == "2" ]]; then
        MAX_BATCH_SIZE=256
    else
        MAX_BATCH_SIZE=512
    fi
fi

# cuda_graph batch_sizes: powers of 2 up to min(256, MAX_BATCH_SIZE), plus 384,512 when MAX_BATCH_SIZE=512.
CUDA_GRAPH_BATCH_SIZES="1, 2, 4, 8, 16, 32, 64, 128"
if (( MAX_BATCH_SIZE >= 256 )); then
    CUDA_GRAPH_BATCH_SIZES="$CUDA_GRAPH_BATCH_SIZES, 256"
fi
if (( MAX_BATCH_SIZE >= 512 )); then
    CUDA_GRAPH_BATCH_SIZES="$CUDA_GRAPH_BATCH_SIZES, 384, 512"
fi

# MoE backend: CUTEDSL for dp-attn configs, TRTLLM otherwise.
if [[ "$DP_ATTENTION" == "true" ]]; then
    MOE_BACKEND=CUTEDSL
else
    MOE_BACKEND=TRTLLM
fi

# max_num_tokens scales with input seq length.
case "$ISL" in
    8192) MAX_NUM_TOKENS=33792 ;;
    *)    MAX_NUM_TOKENS=16384 ;;
esac

# Hand-tuned hybrid: 8k/1k DEP-4 at conc 256 wants TRTLLM MoE and a tighter
# token budget instead of the CUTEDSL default that other dp-attn points use.
if [[ "$ISL" == "8192" && "$TP" == "4" && "$EP_SIZE" == "4" \
      && "$DP_ATTENTION" == "true" && "$CONC" == "256" ]]; then
    MOE_BACKEND=TRTLLM
    MAX_NUM_TOKENS=24576
fi

# batch_wait_max_tokens_ratio (non-dp-attn only) scales with concurrency.
case "$CONC" in
    4|8|16)  BATCH_WAIT_RATIO=0.0625 ;;
    32)      BATCH_WAIT_RATIO=0.125 ;;
    64)      BATCH_WAIT_RATIO=0.25 ;;
    128)     BATCH_WAIT_RATIO=0.5 ;;
    *)       BATCH_WAIT_RATIO=0.75 ;;
esac

EXTRA_CONFIG_FILE="$(pwd)/extra-llm-api-config.yml"

cat > "$EXTRA_CONFIG_FILE" << EOF
max_batch_size: $MAX_BATCH_SIZE
max_num_tokens: $MAX_NUM_TOKENS
num_postprocess_workers: 4
backend: pytorch
print_iter_log: true
enable_layerwise_nvtx_marker: false
disable_overlap_scheduler: false
enable_iter_perf_stats: true
enable_chunked_prefill: false
stream_interval: 20
scheduler_config:
  capacity_scheduler_policy: MAX_UTILIZATION
  context_chunking_policy: FIRST_COME_FIRST_SERVED
kv_cache_config:
  free_gpu_memory_fraction: 0.9
  enable_block_reuse: false
  dtype: fp8
cuda_graph_config:
  enable_padding: true
  batch_sizes: [$CUDA_GRAPH_BATCH_SIZES]
moe_config:
  backend: $MOE_BACKEND
EOF

if [[ "$DP_ATTENTION" == "true" ]]; then
    cat >> "$EXTRA_CONFIG_FILE" << EOF
enable_attention_dp: true
attention_dp_config:
  enable_balance: true
  batching_wait_iters: 10
  timeout_iters: 500
EOF
else
    cat >> "$EXTRA_CONFIG_FILE" << EOF
batch_wait_timeout_iters: 50
batch_wait_max_tokens_ratio: $BATCH_WAIT_RATIO
EOF
fi

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

# --- audit: dump env + generated config before launching the server ---
echo "=============== env (resolved) ==============="
printf '  %-22s = %s\n' \
    MODEL                 "$MODEL" \
    TP                    "$TP" \
    EP_SIZE               "$EP_SIZE" \
    DP_ATTENTION          "$DP_ATTENTION" \
    CONC                  "$CONC" \
    ISL                   "$ISL" \
    OSL                   "$OSL" \
    MAX_MODEL_LEN         "$MAX_MODEL_LEN" \
    RANDOM_RANGE_RATIO    "$RANDOM_RANGE_RATIO" \
    RESULT_FILENAME       "$RESULT_FILENAME" \
    MAX_BATCH_SIZE        "$MAX_BATCH_SIZE" \
    MAX_NUM_TOKENS        "$MAX_NUM_TOKENS" \
    MOE_BACKEND           "$MOE_BACKEND" \
    BATCH_WAIT_RATIO      "$BATCH_WAIT_RATIO" \
    CUDA_GRAPH_BATCH_SIZES "$CUDA_GRAPH_BATCH_SIZES" \
    SERVER_LOG            "$SERVER_LOG" \
    PORT                  "$PORT" \
    EVAL_ONLY             "${EVAL_ONLY:-}"
echo "=============== $EXTRA_CONFIG_FILE ==============="
ls -la "$EXTRA_CONFIG_FILE"
cat "$EXTRA_CONFIG_FILE"
echo "=============================================="

mpirun -n 1 --oversubscribe --allow-run-as-root \
    trtllm-serve "$MODEL" --port="$PORT" \
    --trust_remote_code \
    --backend=pytorch \
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
    --num-prompts $(( CONC * 10 )) \
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
