#!/usr/bin/env bash

# === Required Env Vars === 
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# MAX_MODEL_LEN
# RANDOM_RANGE_RATIO
# TP
# CONC
# ISL
# OSL


cat > config.yaml << EOF
compilation-config: '{"cudagraph_mode":"PIECEWISE"}'
async-scheduling: true
no-enable-prefix-caching: true
cuda-graph-sizes: 2048
max-num-batched-tokens: 8192
max-model-len: 10240
EOF

export PYTHONNOUSERSITE=1
SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)

set -x
vllm serve $MODEL --host=0.0.0.0 --port=$PORT \
--config config.yaml \
--gpu-memory-utilization=0.9 \
--tensor-parallel-size=$TP \
--max-num-seqs=$CONC  \
--disable-log-requests > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Show logs until server is ready
tail -f $SERVER_LOG &
TAIL_PID=$!
set +x
until curl --output /dev/null --silent --fail http://0.0.0.0:$PORT/health; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server died before becoming healthy. Exiting."
        exit 1
    fi
    sleep 5
done
kill $TAIL_PID

pip install -q datasets pandas

# Source benchmark utilities
source "$(dirname "$0")/benchmark_lib.sh"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts $(( $CONC * 10 )) \
    --max-concurrency 512 \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/