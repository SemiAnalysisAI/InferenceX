#!/usr/bin/env bash

# === Required Env Vars ===
# MODEL
# PORT
# TP
# CONC
# ISL
# OSL
# RANDOM_RANGE_RATIO
# RESULT_FILENAME

echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"

cat > config.yaml << EOF
async-scheduling: true
no-enable-prefix-caching: true
max-cudagraph-capture-size: 2048
max-num-batched-tokens: 8192
max-model-len: 10240
EOF

SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)
export TORCH_CUDA_ARCH_LIST="9.0"
export VLLM_MXFP4_USE_MARLIN=1

set -x
PYTHONNOUSERSITE=1 vllm serve $MODEL --host=0.0.0.0 --port=$PORT \
--config config.yaml \
--gpu-memory-utilization=0.9 \
--tensor-parallel-size=$TP \
--max-num-seqs=$CONC  \
 > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Source benchmark utilities
source "$(dirname "$0")/benchmark_lib.sh"

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

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
