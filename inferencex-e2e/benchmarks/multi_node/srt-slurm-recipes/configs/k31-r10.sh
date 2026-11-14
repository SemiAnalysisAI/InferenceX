#!/usr/bin/env bash
set -eo pipefail
if [[ "${K31_ROUTER_ONLY:-0}" == 1 ]]; then
    echo '51dbf39f0cd34fbc81939f5b9ba022fcc5b743af5d6bafdb6c4b110fc6c7af0d  /usr/local/bin/vllm-router' | sha256sum -c -
    return 0 2>/dev/null || exit 0
fi
export PATH=/opt/runtime-venv/bin:$PATH
/opt/runtime-venv/bin/python /opt/k31-r10/image_gate.py
mkdir -p "$HOME"
/opt/runtime-venv/bin/python /opt/k31-r10/reuse_aiter_module.py
