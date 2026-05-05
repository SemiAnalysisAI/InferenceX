#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4-Pro on MI355X via vLLM, FP4 MoE kernel path.
#
# Recipe source: recipes.vllm.ai DeepSeek-V4-Pro page (AMD ROCm), aligned
# with vllm-project/vllm#40871 (base DSv4 ROCm support; lm_eval gsm8k
# 0.9538 flexible-extract on MI355X TP=8) and #41217 (MLA Indexer
# optimization).
#
# DeepSeek-V4-Pro ships an FP4+FP8 mixed checkpoint: MoE expert weights
# are stored in FP4 while attention/norm/router stay in FP8. The aiter
# MoE backend handles MXFP4 expert weights natively on ROCm gfx950.
#
# Image: rocm/vllm-dev:deepseek-v4-latest already ships the validated
# vLLM build, so no PR clone or runtime patching is needed here.

source "$(dirname "$0")/../benchmark_lib.sh"

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

if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_LINEAR=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    SERVE_MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
else
    SERVE_MAX_MODEL_LEN="$MAX_MODEL_LEN"
fi

# Recipe defaults from the docs (MI355X TP=8 validated): max-num-seqs 128,
# max-num-batched-tokens 8192. Allow CONC sweeps above 128 to grow the
# scheduler queue without the user having to override env vars.
MAX_NUM_SEQS=${MAX_NUM_SEQS:-$(( CONC > 128 ? CONC : 128 ))}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}

start_gpu_monitor

set -x
vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --distributed-executor-backend mp \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.6 \
    --max-model-len "$SERVE_MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --moe-backend aiter \
    --no-enable-prefix-caching \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --async-scheduling \
    --enforce-eager \
    --trust-remote-code > "$SERVER_LOG" 2>&1 &

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
    --trust-remote-code

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
