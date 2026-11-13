#!/usr/bin/env bash
set -eo pipefail
if [[ "${K31_ROUTER_ONLY:-0}" == 1 ]]; then
    echo '73aed9d76abb2582082097283d159744471033320fe91d4d60e9099d5acd7e08  /usr/local/bin/vllm-router' | sha256sum -c -
    return 0 2>/dev/null || exit 0
fi
export PATH=/opt/runtime-venv/bin:$PATH
/opt/runtime-venv/bin/python /opt/k31-r10/image_gate.py
mkdir -p "$HOME"
/opt/runtime-venv/bin/python /opt/k31-r10/reuse_aiter_module.py
