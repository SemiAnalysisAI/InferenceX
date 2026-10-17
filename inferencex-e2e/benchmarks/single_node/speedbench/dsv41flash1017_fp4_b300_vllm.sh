#!/usr/bin/env bash
set -eo pipefail
repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$repo_root/benchmarks/check_env.sh"
check_env_vars MODEL MODEL_PATH IMAGE TP PORT MTP_LIST THINKING_MODES CATEGORY \
    SPEEDBENCH_OUTPUT_LEN CHAT_TEMPLATE_KWARGS_ON OUT_YAML
export PYTHONPATH="$repo_root"
export MODEL_REVISION=1bae0e1088993b9b82de08f59de5be5a34e42170
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_USE_RUST_FRONTEND=1 PYTHONUNBUFFERED=1
exec python3 -m infx.bench speedbench-dsv41
