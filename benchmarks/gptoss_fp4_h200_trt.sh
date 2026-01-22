#!/usr/bin/env bash

source "$(dirname "$0")/benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    DP_ATTENTION \
    EP_SIZE

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

hf download "$MODEL"
SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)
PORT=${PORT:-8888}


set +x

export TRTLLM_ENABLE_PDL=1 

set -x
cat > gptoss-config.yml << EOF
cuda_graph_config:
  enable_padding: true
  max_batch_size: $CONC
enable_attention_dp: $DP_ATTENTION
kv_cache_config:
  dtype: auto
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.85
moe_config:
  backend: TRITON
num_postprocess_workers: 4
print_iter_log: true
stream_interval: 20 
EOF

PYTHONNOUSERSITE=1 mpirun -n 1 --oversubscribe --allow-run-as-root \
trtllm-serve $MODEL \
--max_batch_size $CONC \
--max_num_tokens 20000 \
--backend pytorch \
--extra_llm_api_options gptoss-config.yml \
--ep_size=$EP_SIZE \
--trust_remote_code \
--gpus_per_node 8 \
--host 0.0.0.0 \
--port $PORT \
--tp_size=$TP \
--pp_size=1 \
> $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Source benchmark utilities
source "$(dirname "$0")/benchmark_lib.sh"

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