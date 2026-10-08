#!/usr/bin/env bash
# Install the vLLM Router that fronts single-node DP-attention ranks (worker and router
# containers), at the master config's router version.
set -eo pipefail
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars ROUTER_VERSION
pip_install=(python3 -m pip install)
if python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages; then
    pip_install+=(--break-system-packages)
fi
"${pip_install[@]}" --quiet "vllm-router==${ROUTER_VERSION}"
