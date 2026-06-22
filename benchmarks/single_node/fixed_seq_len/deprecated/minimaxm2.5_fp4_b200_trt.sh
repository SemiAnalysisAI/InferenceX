#!/usr/bin/env bash

# Source benchmark utilities early
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

MAX_NUM_TOKENS=16384
MAX_CAPTURE_TOKENS=$(( MAX_NUM_TOKENS < CONC * ISL ? MAX_NUM_TOKENS : CONC * ISL ))
CAPTURE_TOKENS_LIST=(1 2 4 8 12 16 24 32 48 64 96 128 192 256 384 512 768)
CAPTURE_TOKENS_LIST+=( $(seq 1024 128 2047))
CAPTURE_TOKENS_LIST+=( $(seq 2048 256 4095))
if [[ $MAX_CAPTURE_TOKENS -ge 4096 ]]; then
    CAPTURE_TOKENS_LIST+=( $(seq 4096 512 $MAX_CAPTURE_TOKENS))
fi
CAPTURE_TOKENS_LIST=$(printf "%s, " "${CAPTURE_TOKENS_LIST[@]}")

CAPTURE_BATCH_LIST=(1 2 4 8 12 )
if [[ $CONC -ge 16 ]]; then
    MAX_CAPTURE_BATCH=$(( CONC < 256 ? CONC : 255 ))
    CAPTURE_BATCH_LIST+=( $(seq 16 8 $MAX_CAPTURE_BATCH ))
fi
if [[ $CONC -ge 256 ]]; then
    MAX_CAPTURE_BATCH=$(( CONC < 512 ? CONC : 511 ))
    CAPTURE_BATCH_LIST+=( $(seq 256 16 $MAX_CAPTURE_BATCH))
fi
if [[ $CONC -ge 512 ]]; then
    MAX_CAPTURE_BATCH=$(( CONC < 768 ? CONC : 767 ))
    CAPTURE_BATCH_LIST+=( $(seq 512 32 $MAX_CAPTURE_BATCH))
fi
if [[ $CONC -ge 1024 ]]; then
    CAPTURE_BATCH_LIST+=( $(seq 768 64 $CONC))
fi
CAPTURE_BATCH_LIST=$(printf "%s, " "${CAPTURE_BATCH_LIST[@]}")
MAX_CAPTURE_TOKENS=$(( CONC < 16 ? 4096 : MAX_NUM_TOKENS ))

CONFIG_FILE="minimax-fp4.yaml"
cat << EOF > $CONFIG_FILE
cuda_graph_config:
    enable_padding: true
    batch_sizes: [${CAPTURE_BATCH_LIST%, }]
moe_config:
    backend: TRTLLM
    use_low_precision_moe_combine: true
enable_attention_dp: $DP_ATTENTION
torch_compile_config:
    capture_num_tokens: [${CAPTURE_TOKENS_LIST%, }]
    enable_piecewise_cuda_graph: true
stream_interval: 100
print_iter_log: true
max_num_tokens: $MAX_NUM_TOKENS
kv_cache_config:
    free_gpu_memory_fraction: 0.9
    enable_block_reuse: False
    dtype: fp8
scheduler_config:
    capacity_scheduler_policy: MAX_UTILIZATION
    context_chunking_policy: FIRST_COME_FIRST_SERVED
nvfp4_gemm_config:
    allowed_backends:
    - cutlass
    - cublaslt
    - cutedsl
    - cuda_core
max_seq_len: $MAX_MODEL_LEN
num_postprocess_workers: 4
EOF

if [[ $DP_ATTENTION == true ]]; then
cat << EOF >> $CONFIG_FILE
attention_dp_config:
    enable_balance: true
EOF
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

echo "Generated config file contents:"
cat $CONFIG_FILE

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x

# Launch TRT-LLM server
mpirun -n 1 --oversubscribe --allow-run-as-root \
    trtllm-serve $MODEL --port=$PORT \
    --trust_remote_code \
    --backend=pytorch \
    --max_batch_size $CONC \
    --tp_size=$TP --ep_size=$EP_SIZE \
    --config=$CONFIG_FILE \
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
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
