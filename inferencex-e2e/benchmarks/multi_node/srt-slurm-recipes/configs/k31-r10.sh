#!/usr/bin/env bash
set -eo pipefail
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars K31_ROUTER_ONLY
if [[ "$K31_ROUTER_ONLY" == 1 ]]; then
    /opt/runtime-venv/bin/python -c 'import importlib.metadata as m; print("vllm-router", m.version("vllm-router"))'
    exit 0
fi
/opt/runtime-venv/bin/python /opt/k31-unified/verify-image.py
mkdir -p "$HOME"

check_env_vars KV_OFFLOAD_BACKEND
if [[ $KV_OFFLOAD_BACKEND == umbp ]]; then
    bash /configs/k31-r10-umbp-pool.sh
fi
