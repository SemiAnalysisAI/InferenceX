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
# NUM_PROMPTS
# PORT_OFFSET

export SGLANG_USE_AITER=1
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
PORT=$(( 8888 + $PORT_OFFSET ))

PREFILL_SIZE=196608
if [[ "$ISL" == "8192" && "$OSL" == "1024" ]]; then
        if [[ "$CONC" -gt "32" ]]; then
		PREFILL_SIZE=32768
	fi
fi

SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)

set -x
python3 -m sglang.launch_server --model-path=$MODEL --trust-remote-code \
--host=0.0.0.0 --port=$PORT \
--tensor-parallel-size=$TP \
--chunked-prefill-size=$PREFILL_SIZE \
--mem-fraction-static=0.8 \
--disable-radix-cache \
--num-continuous-decode-steps=4 \
--max-prefill-tokens=$PREFILL_SIZE \
--cuda-graph-max-bs=128 \
--attention-backend aiter \
--kv-cache-dtype fp8_e4m3 > $SERVER_LOG 2>&1 &

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
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

