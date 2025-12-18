#!/usr/bin/env bash

# Source benchmark utilities early
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
    PORT_OFFSET \
    DP_ATTENTION \
    EP_SIZE

echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"

echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL, EP_SIZE: $EP_SIZE, DP_ATTENTION: $DP_ATTENTION"

hf download $MODEL

# ========= Determine DP_ATTENTION, EP_SIZE and MOE_BACKEND based on ISL, OSL, CONC =========
MOE_BACKEND="CUTLASS"

echo "MOE_BACKEND set to '$MOE_BACKEND'"

SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)
PORT=$(( 8888 + $PORT_OFFSET ))
EXTRA_CONFIG_FILE="dsr1-fp8.yml"

cat > $EXTRA_CONFIG_FILE << EOF
cuda_graph_config:
    enable_padding: true
    max_batch_size: 128
enable_attention_dp: $DP_ATTENTION
print_iter_log: true
kv_cache_config:
    dtype: fp8
    free_gpu_memory_fraction: 0.75
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

# Launch TRT-LLM server
PYTHONNOUSERSITE=1 mpirun -n 1 --oversubscribe --allow-run-as-root \
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

cat $SERVER_LOG

