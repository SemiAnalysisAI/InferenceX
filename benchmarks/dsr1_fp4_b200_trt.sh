#!/usr/bin/env bash

source "$(dirname "$0")/benchmark_lib.sh"

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

hf download "$MODEL"

# ========= Determine other parameters based on ISL, OSL, CONC =========
CUDA_GRAPH_MAX_BATCH_SIZE=$CONC
MOE_BACKEND="TRTLLM"
PIECEWISE_CUDA_GRAPHS="false"

if [[ "$ISL" == "1024" && "$OSL" == "1024" ]]; then
    if [[ "$TP" == "8" && "$EP_SIZE" == "8" ]]; then
        PIECEWISE_CUDA_GRAPHS="true"
    fi
fi

if [[ "$DP_ATTENTION" == "true" ]]; then
    MOE_BACKEND="CUTLASS"
    CUDA_GRAPH_MAX_BATCH_SIZE=$(( CONC < 4 ? CONC : CONC / 4 ))
fi

echo "MOE_BACKEND set to '$MOE_BACKEND'"

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}
EXTRA_CONFIG_FILE="dsr1-fp4.yml"

cat > $EXTRA_CONFIG_FILE << EOF
cuda_graph_config:
    enable_padding: true
    max_batch_size: $CUDA_GRAPH_MAX_BATCH_SIZE
enable_attention_dp: $DP_ATTENTION
print_iter_log: true
kv_cache_config:
    dtype: fp8
    free_gpu_memory_fraction: 0.8
    enable_block_reuse: false 
stream_interval: 10
moe_config:
    backend: $MOE_BACKEND
EOF

if [[ "$DP_ATTENTION" == "true" ]]; then
    cat << EOF >> $EXTRA_CONFIG_FILE
attention_dp_config:
    batching_wait_iters: 0
    enable_balance: true
    timeout_iters: 60
EOF
fi

set -x

MAX_NUM_TOKENS=$(( ($CONC+$ISL+64+63)/64*64 ))
MAX_MODEL_LEN=$(( MAX_MODEL_LEN > 8192 ? MAX_MODEL_LEN : 8192 ))
MAX_NUM_TOKENS=$(( MAX_NUM_TOKENS > 8192 ? MAX_NUM_TOKENS : 8192 ))

if [[ "$PIECEWISE_CUDA_GRAPHS" == "true" ]]; then
    # [2^i for i in range(8)] + [i for i in range(256, max_num_tokens, 256)] + [max_num_tokens]
    capture_tokens=(1 2 4 8 16 32 64 128)
    capture_tokens+=( $(seq 256 256 $MAX_NUM_TOKENS))
    CAPTURE_TOKENS_LIST=$(printf "%s, " "${capture_tokens[@]}")

    cat << EOF >> $EXTRA_CONFIG_FILE
torch_compile_config:
    capture_num_tokens: [${CAPTURE_TOKENS_LIST%, }]
    enable_piecewise_cuda_graph: true 
EOF
fi

# Launch TRT-LLM server
mpirun -n 1 --oversubscribe --allow-run-as-root \
    trtllm-serve $MODEL --port=$PORT \
    --trust_remote_code \
    --backend=pytorch \
    --max_seq_len=$MAX_MODEL_LEN \
    --max_num_tokens=$MAX_NUM_TOKENS \
    --tp_size=$TP --ep_size=$EP_SIZE \
    --extra_llm_api_options=$EXTRA_CONFIG_FILE \
    > $SERVER_LOG 2>&1 &

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
    --num-prompts $(( $CONC * 10 )) \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi
set +x