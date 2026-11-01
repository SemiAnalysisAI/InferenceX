#!/usr/bin/env bash
set -eo pipefail
repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$repo_root/benchmarks/check_env.sh"
check_env_vars \
    CATEGORY CHAT_TEMPLATE_KWARGS_ON EP_SIZE IMAGE MODEL MODEL_PATH MTP_LIST \
    OUT_YAML PORT SPEEDBENCH_OUTPUT_LEN THINKING_MODES TP

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m infx.bench speedbench-al \
    --concurrency 32 \
    --temperature 1.0 \
    --top-p 0.95 \
    -- \
    --tp-size "$TP" --ep-size "$EP_SIZE" \
    --trust-remote-code \
    --mem-fraction-static 0.80 \
    --context-length 32768 \
    --chunked-prefill-size 8192 \
    --max-prefill-tokens 8192 \
    --max-running-requests 32 \
    --cuda-graph-max-bs 32 \
    --kv-cache-dtype fp8_e4m3 \
    --dsa-prefill-backend trtllm \
    --dsa-decode-backend trtllm \
    --moe-runner-backend flashinfer_trtllm \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --disable-radix-cache
