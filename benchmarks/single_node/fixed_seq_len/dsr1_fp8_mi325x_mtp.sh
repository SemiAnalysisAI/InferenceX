#!/usr/bin/bash

# DeepSeek-R1-0528 FP8 on MI325X with EAGLE/MTP speculative decoding.
# Mirrors dsr1_fp8_mi325x.sh and adds the speculative-* flags.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    EP_SIZE

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

SERVER_LOG=/workspace/server.log
PORT=8888
if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

export SGLANG_USE_AITER=1
export SGLANG_AITER_MLA_PERSIST=1
export SGLANG_ENABLE_SPEC_V2=1

start_gpu_monitor

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi

set -x
python3 -m sglang.launch_server \
--model-path=$MODEL --host=0.0.0.0 --port=$PORT --trust-remote-code \
--tensor-parallel-size=$TP \
--ep-size $EP_SIZE \
--mem-fraction-static=0.8 \
--cuda-graph-max-bs=128 \
--chunked-prefill-size=131072 \
--num-continuous-decode-steps=4 \
--max-prefill-tokens=131072 \
--kv-cache-dtype fp8_e4m3 \
--attention-backend aiter \
--speculative-algorithm EAGLE \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4 \
--disable-radix-cache \
$EVAL_CONTEXT_ARGS > $SERVER_LOG 2>&1 &

SERVER_PID=$!

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
    --result-dir /workspace/ \
    --use-chat-template

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
