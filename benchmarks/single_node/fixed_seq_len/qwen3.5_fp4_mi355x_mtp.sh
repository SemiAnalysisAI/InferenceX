#!/usr/bin/env bash

source "$(dirname "$0")/../../benchmark_lib.sh"

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

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

export SGLANG_USE_AITER=1
# AllReduce latency on TP=2: INT4 quick-reduce + single-stage AITER AllReduce.
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_AR_1STAGE=1
# Built-in MTP head (NEXTN) runs on the ROCm-hardened spec-v1 chain path.
export SGLANG_ENABLE_SPEC_V2=0

SERVER_LOG=/workspace/server.log
MEM_FRAC_STATIC=${MEM_FRAC_STATIC:-0.85}

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
fi

# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
python3 -m sglang.launch_server --model-path=$MODEL --trust-remote-code \
--host=0.0.0.0 --port=$PORT \
--tensor-parallel-size=$TP \
--attention-backend aiter \
--mem-fraction-static $MEM_FRAC_STATIC \
--model-loader-extra-config '{"enable_multithread_load": true}' \
--watchdog-timeout 1200  \
--disable-radix-cache \
--enable-dp-attention \
--mamba-ssm-dtype bfloat16 \
--disable-shared-experts-fusion \
--max-running-requests $CONC \
--speculative-algorithm NEXTN \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4 \
> $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID" --sleep-interval 60

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --use-chat-template

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
