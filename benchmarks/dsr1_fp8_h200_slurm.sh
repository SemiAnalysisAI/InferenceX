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

echo "JOB \$SLURM_JOB_ID running on \$SLURMD_NODENAME"

pip3 install --user sentencepiece
hf download $MODEL
PORT=$(( 8888 + $PORT_OFFSET ))
SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)

export TORCH_CUDA_ARCH_LIST="9.0"

set -x
if [[ $ISL -eq 1024 && $OSL -eq 1024 ]]; then
    PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path $MODEL --tokenizer-path $MODEL \
    --host 0.0.0.0 --port $PORT --trust-remote-code \
    --tensor-parallel-size=$TP --data-parallel-size=1 \
    --disable-radix-cache --max-running-requests 512 --cuda-graph-max-bs 512 \
    --chunked-prefill-size 32768 --max-prefill-tokens 32768 --mem-fraction-static 0.82 \
    --attention-backend flashinfer --stream-interval 10 \
    --decode-log-interval 1 \
    > $SERVER_LOG 2>&1 &
else
    PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path $MODEL --tokenizer-path $MODEL \
    --host 0.0.0.0 --port $PORT --trust-remote-code \
    --tensor-parallel-size=$TP --data-parallel-size=1 \
    --disable-radix-cache --max-running-requests 256 --cuda-graph-max-bs 256 \
    --chunked-prefill-size 32768 --max-prefill-tokens 32768 --mem-fraction-static 0.82 \
    --attention-backend flashinfer --stream-interval 10 \
    --decode-log-interval 1 \
    > $SERVER_LOG 2>&1 &
fi

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
