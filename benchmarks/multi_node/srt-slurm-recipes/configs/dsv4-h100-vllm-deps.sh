#!/bin/bash
# Container setup required by the upstream vLLM DeepSeek-V4-Pro recipe.

set -euo pipefail

apt-get -y update
apt-get install -y --no-install-recommends --allow-change-held-packages \
    curl \
    numactl
pip install msgpack

# vLLM's recipe calls out DeepGEMM as a required extra. Prefer the installer
# shipped in the image so it matches the installed vLLM revision; retain the
# upstream recipe command as a fallback for images without the source tree.
if ! python3 -c 'import deep_gemm' >/dev/null 2>&1; then
    if [[ -f /vllm-workspace/tools/install_deepgemm.sh ]]; then
        bash /vllm-workspace/tools/install_deepgemm.sh
    else
        bash <(curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm/v0.21.0/tools/install_deepgemm.sh)
    fi
fi

if [[ -f /configs/patches/vllm_numa_bind_hash_fix.py ]]; then
    python3 /configs/patches/vllm_numa_bind_hash_fix.py
fi
