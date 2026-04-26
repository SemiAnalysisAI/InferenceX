#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4-Pro FP8 on MI355X via vLLM with AITER MLA decode.
# Based on vllm-project/vllm#40889 (AITER-accelerated sparse MLA decode,
# stacked on #40871 which adds base DSv4 ROCm support).
#
# Requires an image that already has #40871 compiled (the base adds C++
# kernels in csrc/). PR #40889 is Python-only and is patched in at runtime.
# Once #40889 merges, update the image and remove the overlay block below.

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
export VLLM_ENGINE_READY_TIMEOUT_S=3600

# Overlay PR #40889 Python files on top of the image's installed vLLM.
# PR #40889 is Python-only (3 files); the base C++ from #40871 must already
# be compiled in the image. Bump VLLM_PR_SHA when the PR moves.
VLLM_PR_SHA="b3a4a44f01e565219dd353611712d0ea2e8d11ee"
VLLM_PR_DIR="/tmp/vllm-pr40889"

if [ ! -d "$VLLM_PR_DIR/.git" ]; then
    git clone --filter=blob:none https://github.com/ChuanLi1101/vllm.git "$VLLM_PR_DIR"
fi
(
    cd "$VLLM_PR_DIR"
    git fetch --depth=1 origin "$VLLM_PR_SHA" 2>/dev/null \
        || git fetch --depth=1 origin rocm/aiter-mla-dsv4-decode
    git checkout --force "$VLLM_PR_SHA"
    test "$(git rev-parse HEAD)" = "$VLLM_PR_SHA"
)

VLLM_SITE=$(python3 -c "import vllm; print(vllm.__path__[0])")
mkdir -p "$VLLM_SITE/v1/attention/ops"
cp "$VLLM_PR_DIR/vllm/v1/attention/ops/rocm_aiter_dsv4_decode.py" \
   "$VLLM_SITE/v1/attention/ops/"
cp "$VLLM_PR_DIR/vllm/model_executor/layers/deepseek_v4_attention.py" \
   "$VLLM_SITE/model_executor/layers/"
cp "$VLLM_PR_DIR/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py" \
   "$VLLM_SITE/model_executor/layers/fused_moe/oracle/"
echo "Patched 3 files from PR #40889 @ ${VLLM_PR_SHA:0:12}"

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    MAX_MODEL_LEN="$EVAL_MAX_MODEL_LEN"
fi

start_gpu_monitor

set -x
vllm serve $MODEL --port $PORT \
    --tensor-parallel-size $TP \
    --gpu-memory-utilization 0.95 \
    --max-model-len $MAX_MODEL_LEN \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --enforce-eager \
    --moe-backend "triton_unfused" \
    --no-enable-prefix-caching \
    --max-num-seqs 256 \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 > $SERVER_LOG 2>&1 &

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
