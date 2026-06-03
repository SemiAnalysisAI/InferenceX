#!/usr/bin/env bash

# DeepSeek-R1-0528 FP8 on H200 with EAGLE/MTP speculative decoding.
# Mirrors dsr1_fp8_h200.sh and adds the speculative-* flags from
# dsr1_fp8_b200_mtp.sh (the production sglang MTP template).
# Keeps the H200's flashinfer attention backend (no trtllm_mla path on
# H200 for this image).

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

pip3 install --user --break-system-packages sentencepiece

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

# MTP only supports TP=8 for now (matching dsr1_fp8_b200_mtp.sh)
if [[ $TP -ne 8 ]]; then
  echo "MTP only supports TP=8, got TP=$TP!"
  exit 1
fi

SERVER_LOG=/workspace/server.log

# MTP (Multi-Token Prediction) Config - EAGLE speculative decoding
SPECULATIVE_NUM_STEPS=2
SPECULATIVE_DRAFT_TOKENS=3
SPECULATIVE_EAGLE_TOPK=1

export SGLANG_ENABLE_SPEC_V2=1
export TORCH_CUDA_ARCH_LIST="9.0"

start_gpu_monitor

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi

set -x
if [[ $ISL -eq 1024 && $OSL -eq 1024 ]]; then
    MAX_RUNNING_REQUESTS=512
    CUDA_GRAPH_MAX_BS=512
else
    MAX_RUNNING_REQUESTS=256
    CUDA_GRAPH_MAX_BS=256
fi

PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path $MODEL \
--host 0.0.0.0 --port $PORT --trust-remote-code \
--tensor-parallel-size=$TP --data-parallel-size=1 \
--ep-size $EP_SIZE \
--disable-radix-cache \
--max-running-requests $MAX_RUNNING_REQUESTS \
--cuda-graph-max-bs $CUDA_GRAPH_MAX_BS \
--chunked-prefill-size 32768 --max-prefill-tokens 32768 --mem-fraction-static 0.82 \
--attention-backend flashinfer --stream-interval 10 \
--decode-log-interval 1 \
--speculative-algorithm EAGLE \
--speculative-num-steps $SPECULATIVE_NUM_STEPS \
--speculative-num-draft-tokens $SPECULATIVE_DRAFT_TOKENS \
--speculative-eagle-topk $SPECULATIVE_EAGLE_TOPK \
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
