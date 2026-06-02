#!/usr/bin/env bash

# DeepSeek-R1-0528 FP8 on MI355X with EAGLE/MTP speculative decoding.
# Mirrors dsr1_fp8_mi355x.sh and adds the speculative-* flags.

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

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

# Reference
# https://rocm.docs.amd.com/en/docs-7.0-docker/benchmark-docker/inference-sglang-deepseek-r1-fp8.html

export SGLANG_USE_AITER=1
export SGLANG_AITER_MLA_PERSIST=1
export SGLANG_ENABLE_SPEC_V2=1
export RCCL_MSCCL_ENABLE=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4

SERVER_LOG=/workspace/server.log

# Keep server-side speculative decoding capacity aligned with the matrix row.
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-$CONC}"
CUDA_GRAPH_MAX_BATCH_SIZE="${CUDA_GRAPH_MAX_BATCH_SIZE:-$CONC}"

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi
start_gpu_monitor

python3 -m sglang.launch_server \
    --attention-backend aiter \
    --model-path $MODEL \
    --host=0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TP \
    --ep-size $EP_SIZE \
    --trust-remote-code \
    --chunked-prefill-size 196608 \
    --mem-fraction-static 0.8 --disable-radix-cache \
    --num-continuous-decode-steps 8 \
    --max-prefill-tokens 196608 \
    --kv-cache-dtype fp8_e4m3 \
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BATCH_SIZE" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
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
    --num-prompts "$((CONC * 10))" \
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
