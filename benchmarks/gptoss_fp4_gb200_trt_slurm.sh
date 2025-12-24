#!/usr/bin/env bash

# === Required Env Vars ===
# MODEL_PATH (set by launch script - direct path to model files)
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
# NUM_NODES (set by launch script for multi-node)
# GPUS_PER_NODE (set by launch script)

# GPTOSS TRTLLM Deployment Guide:
# https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/deployment-guide/quick-start-recipe-for-gpt-oss-on-trtllm.md

echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"

echo "TP: $TP, CONC: $CONC, ISL: $ISL, OSL: $OSL, EP_SIZE: $EP_SIZE, DP_ATTENTION: $DP_ATTENTION"
echo "MODEL_PATH: /models (mounted from $MODEL_PATH)"
echo "NUM_NODES: ${NUM_NODES:-1}, GPUS_PER_NODE: ${GPUS_PER_NODE:-$TP}"

# Use direct model path - mounted at /models inside container
# This avoids HuggingFace permission issues on shared filesystem
LOCAL_MODEL_PATH="/models"

SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)
PORT=$(( 8888 + $PORT_OFFSET ))

# ========= Determine DP_ATTENTION, EP_SIZE and MOE_BACKEND based on ISL, OSL, CONC =========
MOE_BACKEND="TRTLLM"

echo "MOE_BACKEND set to '$MOE_BACKEND'"

EXTRA_CONFIG_FILE="gptoss-fp4.yml"
export TRTLLM_ENABLE_PDL=1

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
    #export TRTLLM_MOE_ALLTOALL_BACKEND="mnnvlthroughput"
    export TRTLLM_FORCE_ALLTOALL_METHOD="NotEnabled"
    #export TRTLLM_MOE_A2A_WORKSPACE_MB="2048"
    cat << EOF >> $EXTRA_CONFIG_FILE
attention_dp_config:
    enable_balance: true
EOF
fi

echo "Generated config file contents:"
cat $EXTRA_CONFIG_FILE

set -x

MAX_NUM_TOKENS=20000

# Launch TRT-LLM server using direct model path
# For multi-node, mpirun uses SLURM's allocated nodes
if [[ "${NUM_NODES:-1}" -gt 1 ]]; then
    echo "Multi-node launch: $TP processes across $NUM_NODES nodes ($GPUS_PER_NODE GPUs/node)"
    mpirun -n $TP --map-by ppr:${GPUS_PER_NODE}:node --allow-run-as-root \
        trtllm-serve $LOCAL_MODEL_PATH --port=$PORT \
        --trust_remote_code \
        --backend=pytorch \
        --max_batch_size 512 \
        --max_seq_len=$MAX_MODEL_LEN \
        --max_num_tokens=$MAX_NUM_TOKENS \
        --tp_size=$TP --ep_size=$EP_SIZE \
        --extra_llm_api_options=$EXTRA_CONFIG_FILE \
        > $SERVER_LOG 2>&1 &
else
    echo "Single-node launch: $TP GPUs"
    mpirun -n 1 --oversubscribe --allow-run-as-root \
        trtllm-serve $LOCAL_MODEL_PATH --port=$PORT \
        --trust_remote_code \
        --backend=pytorch \
        --max_batch_size 512 \
        --max_seq_len=$MAX_MODEL_LEN \
        --max_num_tokens=$MAX_NUM_TOKENS \
        --tp_size=$TP --ep_size=$EP_SIZE \
        --extra_llm_api_options=$EXTRA_CONFIG_FILE \
        > $SERVER_LOG 2>&1 &
fi

SERVER_PID=$!

# Source benchmark utilities
source "$(dirname "$0")/benchmark_lib.sh"

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$LOCAL_MODEL_PATH" \
    --port "$PORT" \
    --backend openai \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts $(( $CONC * 10 )) \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

