#!/usr/bin/env bash

# GLM-5 FP8 on MI325X with EAGLE / MTP speculative decoding.
# Mirrors glm5_fp8_mi325x.sh and adds the speculative-* flags.

source "$(dirname "$0")/../../../benchmark_lib.sh"

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

SERVER_LOG=/workspace/server.log
CONTEXT_LENGTH=$((ISL + OSL + 20))
MAX_PREFILL_TOKENS=32768

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
else EVAL_CONTEXT_ARGS="--context-length $CONTEXT_LENGTH"
fi

start_gpu_monitor

# Launch args follow sglang issue #25672 comment 4485916205:
# tilelang NSA backends + fp8_e4m3 KV cache + multithread model load,
# plus EAGLE/MTP speculative decoding.
python3 -m sglang.launch_server \
    --model-path $MODEL \
    --host=0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TP \
    --ep-size $EP_SIZE \
    --trust-remote-code \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --tokenizer-worker-num 6 \
    --cuda-graph-max-bs $CONC \
    --disable-radix-cache \
    --max-prefill-tokens $MAX_PREFILL_TOKENS \
    --scheduler-recv-interval 30 \
    --mem-fraction-static 0.80 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
    --nsa-prefill-backend tilelang \
    --nsa-decode-backend tilelang \
    --kv-cache-dtype fp8_e4m3 \
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
