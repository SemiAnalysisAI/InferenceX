#!/usr/bin/env bash

# DeepSeek-V4-Pro single-node TRTLLM bring-up recipe. This intentionally starts
# with low-concurrency TP-only STP points; DSv4 has no Jinja chat template and
# TRTLLM does not currently have a DSv4-specific chat parser wired here.

source "$(dirname "$0")/../benchmark_lib.sh"

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

if [[ "$MODEL" != /* ]]; then
    hf download "$MODEL"
fi

nvidia-smi

SERVER_LOG="$PWD/server.log"
PORT=${PORT:-8888}
EXTRA_CONFIG_FILE="dsv4-fp4-trt.yml"

MOE_BACKEND="TRTLLM"
MAX_BATCH_SIZE=$(( CONC > 8 ? CONC : 8 ))
CUDA_GRAPH_MAX_BATCH_SIZE="$MAX_BATCH_SIZE"
KV_CACHE_FREE_MEM_FRACTION="${KV_CACHE_FREE_MEM_FRACTION:-0.80}"

if [[ "$DP_ATTENTION" == "true" ]]; then
    echo "DSv4 TRTLLM bring-up only supports TP-only search-space entries for now." >&2
    exit 1
fi

cat > "$EXTRA_CONFIG_FILE" << EOF
cuda_graph_config:
    enable_padding: true
    max_batch_size: $CUDA_GRAPH_MAX_BATCH_SIZE
enable_attention_dp: false
print_iter_log: true
kv_cache_config:
    dtype: fp8
    free_gpu_memory_fraction: $KV_CACHE_FREE_MEM_FRACTION
    enable_block_reuse: false
stream_interval: 10
num_postprocess_workers: 4
moe_config:
    backend: $MOE_BACKEND
EOF

echo "Generated config file contents:"
cat "$EXTRA_CONFIG_FILE"

MAX_MODEL_LEN=$(( MAX_MODEL_LEN > 8192 ? MAX_MODEL_LEN : 8192 ))
MAX_NUM_TOKENS=$(( ISL + OSL + 256 ))
MAX_NUM_TOKENS=$(( MAX_NUM_TOKENS > 8192 ? MAX_NUM_TOKENS : 8192 ))

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
    MAX_NUM_TOKENS="$EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor --output "$PWD/gpu_metrics.csv"

set -x
mpirun -n 1 --oversubscribe --allow-run-as-root \
    trtllm-serve "$MODEL" --port="$PORT" \
    --trust_remote_code \
    --backend=pytorch \
    --max_batch_size="$MAX_BATCH_SIZE" \
    --max_seq_len="$MAX_MODEL_LEN" \
    --max_num_tokens="$MAX_NUM_TOKENS" \
    --tp_size="$TP" \
    --ep_size="$EP_SIZE" \
    --extra_llm_api_options="$EXTRA_CONFIG_FILE" \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!

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
    --result-dir "$PWD/" \
    --dsv4 \
    --trust-remote-code \
    --server-pid "$SERVER_PID"

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
