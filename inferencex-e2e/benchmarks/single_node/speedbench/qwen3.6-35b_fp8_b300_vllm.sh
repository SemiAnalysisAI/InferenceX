#!/usr/bin/env bash
set -eo pipefail
repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$repo_root/benchmarks/check_env.sh"
check_env_vars CATEGORY CHAT_TEMPLATE_KWARGS_ON IMAGE MODEL MODEL_PATH MTP_LIST \
    OUT_YAML PORT PRECISION SPEEDBENCH_OUTPUT_LEN THINKING_MODES TP

if [[ "$MODEL" != Qwen/Qwen3.6-35B-A3B-FP8 || "$PRECISION" != fp8 ]]; then
    echo "This collector requires Qwen/Qwen3.6-35B-A3B-FP8 and PRECISION=fp8." >&2
    exit 1
fi
# The 512-wide FP8 expert dimension must remain divisible by its 128-wide blocks.
if [[ "$TP" != 1 && "$TP" != 2 && "$TP" != 4 ]]; then
    echo "Qwen3.6-35B-A3B-FP8 requires TP=1, 2, or 4." >&2
    exit 1
fi
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_RUST_FRONTEND=0
export MODEL_REVISION=95a723d08a9490559dae23d0cff1d9466213d989
exec python3 -m infx.bench speedbench-al \
    --concurrency 64 --seed 42 --spec-method mtp \
    --sampling '{"on":{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repetition_penalty":1.0},"off":{"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repetition_penalty":1.0}}' \
    -- \
    --tensor-parallel-size "$TP" --data-parallel-size 1 \
    --gpu-memory-utilization 0.80 \
    --max-model-len 32768 --max-num-batched-tokens 8192 --max-num-seqs 64 \
    --language-model-only --reasoning-parser qwen3 \
    --no-enable-prefix-caching
