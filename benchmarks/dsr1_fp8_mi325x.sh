#!/usr/bin/bash

# Source benchmark utilities early
source "$(dirname "$0")/benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)
PORT=8888
hf download $MODEL

# Reference
# https://rocm.docs.amd.com/en/docs-7.0-rc1/preview/benchmark-docker/inference-sglang-deepseek-r1-fp8.html#run-the-inference-benchmark

export SGLANG_USE_AITER=1
export SGLANG_AITER_MLA_PERSIST=1

set -x
python3 -m sglang.launch_server \
--model-path=$MODEL --host=0.0.0.0 --port=$PORT --trust-remote-code \
--tensor-parallel-size=$TP \
--mem-fraction-static=0.8 \
--cuda-graph-max-bs=128 \
--chunked-prefill-size=131072 \
--num-continuous-decode-steps=4 \
--max-prefill-tokens=131072 \
--kv-cache-dtype fp8_e4m3 \
--attention-backend aiter \
--disable-radix-cache \
> $SERVER_LOG 2>&1 &

SERVER_PID=$!

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

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi
set +x
