#!/usr/bin/env bash
set -eo pipefail
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars K31_ROUTER_ONLY
if [[ "$K31_ROUTER_ONLY" == 1 ]]; then
    /opt/runtime-venv/bin/python -c 'import importlib.metadata as m; print("vllm-router", m.version("vllm-router"))'
    exit 0
fi
/opt/runtime-venv/bin/python /configs/verify-shardq-image.py
check_env_vars HOME KV_OFFLOAD_BACKEND AITER_JIT_DIR TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR
mkdir -p "$HOME" "$AITER_JIT_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
if [[ $KV_OFFLOAD_BACKEND == umbp ]]; then
    bash /configs/k31-r10-umbp-pool.sh
fi
