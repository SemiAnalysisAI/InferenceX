#!/usr/bin/env bash

# DeepSeek-R1-0528 FP4 on B200 with EAGLE/MTP speculative decoding.
# Mirrors dsr1_fp4_b200.sh and adds the speculative-* flags from
# dsr1_fp8_b200_mtp.sh (the production B200 sglang MTP template).

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

nvidia-smi

# MTP only supports TP=8 for now (matching dsr1_fp8_b200_mtp.sh)
if [[ $TP -ne 8 ]]; then
  echo "MTP only supports TP=8, got TP=$TP!"
  exit 1
fi

SERVER_LOG=/workspace/server.log

if [[ $CONC -ge 16 ]]; then
  SCHEDULER_RECV_INTERVAL=30
else
  SCHEDULER_RECV_INTERVAL=10
fi
echo "SCHEDULER_RECV_INTERVAL: $SCHEDULER_RECV_INTERVAL, CONC: $CONC, ISL: $ISL, OSL: $OSL"

# MTP (Multi-Token Prediction) Config - EAGLE speculative decoding
SPECULATIVE_NUM_STEPS=2
SPECULATIVE_DRAFT_TOKENS=3
SPECULATIVE_EAGLE_TOPK=1

export SGLANG_ENABLE_SPEC_V2=1

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi
start_gpu_monitor

set -x
PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path $MODEL --host 0.0.0.0 --port $PORT --trust-remote-code \
--tensor-parallel-size=$TP --data-parallel-size=1 \
--cuda-graph-max-bs 512 --max-running-requests 512 --mem-fraction-static 0.82 --kv-cache-dtype fp8_e4m3 \
--chunked-prefill-size 16384 --max-prefill-tokens 16384 \
--ep-size $EP_SIZE --quantization modelopt_fp4 --enable-flashinfer-allreduce-fusion --scheduler-recv-interval $SCHEDULER_RECV_INTERVAL \
--enable-symm-mem --disable-radix-cache --attention-backend trtllm_mla --moe-runner-backend flashinfer_trtllm --stream-interval 30 \
--speculative-algorithm EAGLE \
--speculative-num-steps $SPECULATIVE_NUM_STEPS \
--speculative-num-draft-tokens $SPECULATIVE_DRAFT_TOKENS \
--speculative-eagle-topk $SPECULATIVE_EAGLE_TOPK \
$EVAL_CONTEXT_ARGS > $SERVER_LOG 2>&1 &

SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts $((CONC * 10)) \
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
