#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Keep the v1.0.36 vLLM setup behavior, then install the CUDA 13 Mooncake
# package required by the official distributed Mooncake profile.
apt-get -y update
apt-get install -y --no-install-recommends --allow-change-held-packages \
    ibverbs-providers \
    numactl

python3 -m pip install msgpack
python3 -m pip uninstall -y mooncake-transfer-engine mooncake-transfer-engine-cuda13 || true
python3 -m pip install --no-deps 'mooncake-transfer-engine-cuda13==0.3.12.post1'

if [[ -f /configs/patches/vllm_numa_bind_hash_fix.py ]]; then
    python3 /configs/patches/vllm_numa_bind_hash_fix.py
fi
