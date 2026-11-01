#!/usr/bin/env bash
set -eo pipefail
repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$repo_root/benchmarks/check_env.sh"
check_env_vars \
    CATEGORY CHAT_TEMPLATE_KWARGS_ON IMAGE MODEL MODEL_PATH MTP_LIST \
    OUT_YAML PORT SPEEDBENCH_OUTPUT_LEN THINKING_MODES TP

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_RUST_FRONTEND=0
exec python3 -m infx.bench speedbench-al \
    --concurrency 1 \
    --temperature 1.0 \
    --top-p 0.95 \
    -- \
    --tensor-parallel-size "$TP" --data-parallel-size 1 \
    --trust-remote-code \
    --gpu-memory-utilization 0.80 \
    --max-model-len 32768 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 32 \
    --kv-cache-dtype fp8 \
    --tokenizer-mode auto \
    --reasoning-parser glm47 \
    --chat-template-content-format string \
    --no-enable-prefix-caching
