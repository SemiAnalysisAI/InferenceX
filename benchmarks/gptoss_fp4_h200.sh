#!/usr/bin/env bash

# === Required Env Vars ===
# MODEL
# TP
# CONC
# ISL
# OSL
# RANDOM_RANGE_RATIO
# RESULT_FILENAME
# PORT_OFFSET

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

hf download "$MODEL"

set -x
pip install datasets pandas

# Calculate max-model-len based on ISL and OSL
if [ "$ISL" = "1024" ] && [ "$OSL" = "1024" ]; then
    CALCULATED_MAX_MODEL_LEN=$((ISL + OSL + 20))
elif [ "$ISL" = "8192" ] || [ "$OSL" = "8192" ]; then
    CALCULATED_MAX_MODEL_LEN=$((ISL + OSL + 200))
else
    CALCULATED_MAX_MODEL_LEN=${MAX_MODEL_LEN:-10240}  
fi

# Create config.yaml
cat > config.yaml << EOF
async-scheduling: true
no-enable-prefix-caching: true
max-cudagraph-capture-size: 2048
max-num-batched-tokens: 8192
max-model-len: $CALCULATED_MAX_MODEL_LEN
EOF

SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)
PORT=$(( 8888 + $PORT_OFFSET ))

export TORCH_CUDA_ARCH_LIST="9.0"
export VLLM_MXFP4_USE_MARLIN=1

PYTHONNOUSERSITE=1 vllm serve $MODEL --host 0.0.0.0 --port $PORT --config config.yaml \
 --gpu-memory-utilization 0.9 --tensor-parallel-size $TP --max-num-seqs $CONC  \
 > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Source benchmark utilities
source "$(dirname "$0")/benchmark_lib.sh"

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts $(( $CONC * 10 )) \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/
