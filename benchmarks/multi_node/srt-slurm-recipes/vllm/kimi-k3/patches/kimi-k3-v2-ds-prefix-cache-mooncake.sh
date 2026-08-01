#!/bin/bash

set -euo pipefail

# Keep the Kimi-specific vLLM image and install the matching CUDA 13 Mooncake
# runtime into it. The generic Mooncake package ships a CUDA 12-linked master,
# while vllm/vllm-openai:kimi-k3 is CUDA 13-only.
bash /configs/patches/vllm-container-deps.sh

apt-get -y update
apt-get install -y --no-install-recommends --allow-change-held-packages \
    ibverbs-providers \
    numactl

python3 -m pip install msgpack
python3 -m pip uninstall -y mooncake-transfer-engine mooncake-transfer-engine-cuda13 || true
python3 -m pip install --no-deps 'mooncake-transfer-engine-cuda13==0.3.12.post1'

# Apply the same Kimi V2 DS prefix-cache and NIXL SSM-state fixes as the
# no-offload arm after dependency setup, so both arms run identical vLLM code.
python3 /configs/patches/patch_kimi_k3_v2_ds_prefix_cache.py
