#!/usr/bin/env bash

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    DP_ATTENTION \
    EP_SIZE \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    MAX_MODEL_LEN

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

# sglang ships in the image at the SHA encoded in the image tag (built
# from the amd/deepseek_v4 branch in sgl-project/sglang). To bump sglang,
# bump the image tag in .github/configs/amd-master.yaml.

export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=0
export SGLANG_DP_USE_GATHERV=1
export SGLANG_OPT_USE_FUSED_COMPRESS=true
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export SGLANG_OPT_FP8_WO_A_GEMM=false
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=false
export SGLANG_OPT_USE_TOPK_V2=false
export SGLANG_OPT_USE_AITER_INDEXER=true
export SGLANG_OPT_USE_TILELANG_INDEXER=false
export SGLANG_OPT_USE_TILELANG_MHC_PRE=false
export SGLANG_OPT_USE_TILELANG_MHC_POST=false
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true
export AITER_BF16_FP8_MOE_BOUND=0
export SGLANG_EAGER_INPUT_NO_COPY=true

# multi-stream
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false
export SGLANG_ROCM_USE_MULTI_STREAM=false

SERVER_LOG=/workspace/server.log

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi
# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

PARALLEL_ARGS=(
    --tensor-parallel-size "$TP"
)
CHUNKED_PREFILL_SIZE=$ISL
if [ "${DP_ATTENTION}" = "true" ]; then
    CHUNKED_PREFILL_SIZE=$((ISL * TP))
    PARALLEL_ARGS+=(
        --dp "$TP"
        --enable-dp-attention
        --enable-prefill-delayer
	--prefill-delayer-max-delay-ms 5000
    )
fi
if [ "${EP_SIZE:-1}" -gt 1 ]; then
    PARALLEL_ARGS+=(--ep-size "$EP_SIZE")
fi

sglang serve \
    --model-path $MODEL \
    --host=0.0.0.0 \
    --port $PORT \
    "${PARALLEL_ARGS[@]}" \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend dsv4 \
    --max-running-requests ${CONC} \
    --mem-fraction-static 0.90 \
    --swa-full-tokens-ratio 0.15 \
    --page-size 256 \
    --context-length $MAX_MODEL_LEN \
    --chunked-prefill-size $CHUNKED_PREFILL_SIZE \
    --disable-shared-experts-fusion \
    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4 \
    --chat-template "$(dirname "$0")/../chat_templates/deepseek_v4_thinking.jinja" \
    --watchdog-timeout 1800 $EVAL_CONTEXT_ARGS > $SERVER_LOG 2>&1 &

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
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
