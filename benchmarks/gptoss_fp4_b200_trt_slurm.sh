#!/usr/bin/env bash

# === Required Env Vars ===
# MODEL
# TP
# CONC
# ISL
# OSL
# MAX_MODEL_LEN
# RANDOM_RANGE_RATIO
# RESULT_FILENAME
# PORT_OFFSET
# DP_ATTENTION
# EP_SIZE

# GPTOSS TRTLLM Deployment Guide:
# https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/deployment-guide/quick-start-recipe-for-gpt-oss-on-trtllm.md

echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"

echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL, EP_SIZE: $EP_SIZE, DP_ATTENTION: $DP_ATTENTION"

hf download $MODEL
SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)
PORT=$(( 8888 + $PORT_OFFSET ))

# ========= Determine DP_ATTENTION, EP_SIZE and MOE_BACKEND based on ISL, OSL, CONC =========
MOE_BACKEND="TRTLLM"

echo "MOE_BACKEND set to '$MOE_BACKEND'"

EXTRA_CONFIG_FILE="gptoss-fp4.yml"
export TRTLLM_ENABLE_PDL=1
export NCCL_GRAPH_REGISTER=0

cat > $EXTRA_CONFIG_FILE << EOF
cuda_graph_config:
    enable_padding: true
    max_batch_size: $CONC
enable_attention_dp: $DP_ATTENTION
kv_cache_config:
    dtype: fp8
    enable_block_reuse: false
    free_gpu_memory_fraction: 0.85
print_iter_log: true
stream_interval: 20
num_postprocess_workers: 4
moe_config:
    backend: $MOE_BACKEND
EOF

if [[ "$DP_ATTENTION" == "true" ]]; then
    cat << EOF >> $EXTRA_CONFIG_FILE
attention_dp_config:
    enable_balance: true
EOF
fi

echo "Generated config file contents:"
cat $EXTRA_CONFIG_FILE

set -x

MAX_NUM_TOKENS=20000

# Launch TRT-LLM server
mpirun -n 1 --oversubscribe --allow-run-as-root \
    trtllm-serve $MODEL --port=$PORT \
    --trust_remote_code \
    --backend=pytorch \
    --max_batch_size 512 \
    --max_seq_len=$MAX_MODEL_LEN \
    --max_num_tokens=$MAX_NUM_TOKENS \
    --tp_size=$TP --ep_size=$EP_SIZE \
    --extra_llm_api_options=$EXTRA_CONFIG_FILE \
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
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $(( $CONC ))
    append_lm_eval_summary
fi
set +x
