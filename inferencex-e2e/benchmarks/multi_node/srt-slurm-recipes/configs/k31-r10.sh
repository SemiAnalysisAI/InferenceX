#!/usr/bin/env bash
set -euo pipefail
if [[ "${K31_ROUTER_ONLY:-0}" == 1 ]]; then
    echo '969c62f851950477a2ca1b187013e5ed5aabd08748d2941b96c36e322a8daeec  /usr/local/bin/vllm-router' | sha256sum -c -
    return 0 2>/dev/null || exit 0
fi
export PATH=/opt/runtime-venv/bin:$PATH
/opt/runtime-venv/bin/python /opt/k31-r10/image_gate.py
mkdir -p "$HOME"
/opt/runtime-venv/bin/python /opt/k31-r10/reuse_aiter_module.py
