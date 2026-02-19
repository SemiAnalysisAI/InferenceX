#!/usr/bin/env bash

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

hf download "$MODEL"

# ========= Determine other parameters based on ISL, OSL, CONC =========
MOE_BACKEND="TRTLLM"
PIECEWISE_CUDA_GRAPHS="true"
MAX_BATCH_SIZE=$CONC
KV_CACHE_FREE_MEM_FRACTION=0.8
MTP=3

# DP ATTENTION requires different optimizations
if [[ "$DP_ATTENTION" == "true" ]]; then
    MOE_BACKEND="DEEPGEMM"
    PIECEWISE_CUDA_GRAPHS="false"
    MAX_BATCH_SIZE=$(( CONC < 8 ? CONC : CONC / 8 ))
    KV_CACHE_FREE_MEM_FRACTION=0.7
    # use the new MOE backend from latest trtllm to get better comms
    export ENABLE_CONFIGURABLE_MOE=1
    MTP=1
fi

# currently narrow CONC cases don't benefit from PW CUDA
if [[ "$ISL" == "1024" && "$OSL" == "1024" ]]; then
    if [[ $CONC -le 4 ]]; then
        PIECEWISE_CUDA_GRAPHS="false"
    fi
elif [[ "$ISL" == "1024" && "$OSL" == "8192" ]]; then
    if [[ $CONC -le 8 ]]; then
        PIECEWISE_CUDA_GRAPHS="false"
    fi
elif [[ "$ISL" == "8192" && "$OSL" == "1024" ]]; then
    if [[ $CONC -le 16 ]]; then
        PIECEWISE_CUDA_GRAPHS="false"
    fi
fi


echo "MOE_BACKEND='$MOE_BACKEND', MTP='$MTP'"

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}
EXTRA_CONFIG_FILE="dsr1-fp8-mtp.yml"

cat > $EXTRA_CONFIG_FILE << EOF
cuda_graph_config:
    enable_padding: true
    max_batch_size: $MAX_BATCH_SIZE
enable_attention_dp: $DP_ATTENTION
print_iter_log: true
kv_cache_config:
    dtype: fp8
    free_gpu_memory_fraction: $KV_CACHE_FREE_MEM_FRACTION
    enable_block_reuse: false 
stream_interval: 10
moe_config:
    backend: $MOE_BACKEND
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: ${MTP}
EOF

if [[ "$DP_ATTENTION" == "true" ]]; then
    cat << EOF >> $EXTRA_CONFIG_FILE
attention_dp_config:
    batching_wait_iters: 0
    enable_balance: true
    timeout_iters: 60
EOF
fi

MAX_NUM_TOKENS=$(( ((MTP+1)*MAX_BATCH_SIZE+ISL+64+63)/64*64 ))

# prep PW CUDA config per the documentation
if [[ "$PIECEWISE_CUDA_GRAPHS" == "true" ]]; then
    # [2^i for i in range(8)] + [i for i in range(256, max_num_tokens, 256)] + [max_num_tokens]
    capture_tokens=(1 2 4 8 16 32 64 128)
    capture_tokens+=( $(seq 256 256 $MAX_NUM_TOKENS))
    if [ $((MAX_NUM_TOKENS%256)) -ne 0 ]; then
        capture_tokens+=($MAX_NUM_TOKENS)
    fi
    CAPTURE_TOKENS_LIST=$(printf "%s, " "${capture_tokens[@]}")

    cat << EOF >> $EXTRA_CONFIG_FILE
torch_compile_config:
    capture_num_tokens: [${CAPTURE_TOKENS_LIST%, }]
    enable_piecewise_cuda_graph: true 
EOF
fi

set -x
# Launch TRT-LLM server
mpirun -n 1 --oversubscribe --allow-run-as-root \
    trtllm-serve $MODEL --port=$PORT \
    --trust_remote_code \
    --backend=pytorch \
    --max_batch_size=$MAX_BATCH_SIZE \
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
    --result-dir /workspace/ \
    --use-chat-template

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi