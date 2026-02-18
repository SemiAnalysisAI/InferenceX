#!/usr/bin/env bash

source "$(dirname "$0")/benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

hf download "$MODEL"

nvidia-smi

# Calculate max-model-len based on ISL and OSL
if [ "$ISL" = "1024" ] && [ "$OSL" = "1024" ]; then
    CALCULATED_MAX_MODEL_LEN=$((ISL + OSL + 20))
elif [ "$ISL" = "8192" ] || [ "$OSL" = "8192" ]; then
    CALCULATED_MAX_MODEL_LEN=$((ISL + OSL + 200))
else
    CALCULATED_MAX_MODEL_LEN=${MAX_MODEL_LEN:-10240}
fi

cat > config.yaml << EOF
no-enable-prefix-caching: true
max-model-len: $CALCULATED_MAX_MODEL_LEN
EOF

export PYTHONNOUSERSITE=1

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

set -x
vllm serve $MODEL --host 0.0.0.0 --port $PORT \
--config config.yaml \
--gpu-memory-utilization 0.9 \
--tensor-parallel-size $TP \
--mm-encoder-tp-mode data \
--trust-remote-code \
--max-num-seqs 512 \
--disable-log-requests > $SERVER_LOG 2>&1 &

SERVER_PID=$!

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
    --num-prompts $(( CONC * 10 )) \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi
set +x
