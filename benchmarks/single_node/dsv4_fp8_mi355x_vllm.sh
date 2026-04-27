#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4-Pro FP8 on MI355X via vLLM with AITER MLA decode.
# Based on vllm-project/vllm#40889 (AITER-accelerated sparse MLA decode,
# stacked on #40871 which adds base DSv4 ROCm support).
#
# Uses the ATOM MI355X image as the base (ROCm 7.2.2, PyTorch 2.10,
# aiter with MLA decode, MI355X GPU detection). vLLM is rebuilt from
# the PR branch on top. Once both PRs merge into a release, switch to
# a vLLM ROCm MI355X image and remove the build.

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

# Build vLLM from PR #40889 branch (includes #40871 base). The ATOM
# image provides ROCm 7.2.2 toolchain (hipcc, cmake, ninja, torch,
# aiter with MLA decode); we rebuild vLLM in-place. --no-deps avoids
# disturbing the ATOM image's pinned ROCm/torch packages.
# Bump VLLM_PR_SHA when the PR moves.
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

    pip install --no-build-isolation --no-deps --force-reinstall -e .
)

python3 -c "import vllm; print(f'vLLM {vllm.__version__} from {vllm.__path__[0]}')"

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
