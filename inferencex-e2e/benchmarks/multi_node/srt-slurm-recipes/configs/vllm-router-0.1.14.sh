#!/usr/bin/env bash
# Install the vLLM Router that fronts single-node DP-attention ranks (worker and router containers).
set -euo pipefail
pip_install=(python3 -m pip install)
if python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages; then
    pip_install+=(--break-system-packages)
fi
"${pip_install[@]}" --quiet vllm-router==0.1.14
