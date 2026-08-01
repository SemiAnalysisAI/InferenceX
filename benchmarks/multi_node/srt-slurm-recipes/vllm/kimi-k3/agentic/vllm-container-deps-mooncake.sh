#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Keep the v1.0.36 vLLM setup behavior, then add the dependency required by
# the official distributed Mooncake profile.
apt-get -y update
apt-get install -y --no-install-recommends --allow-change-held-packages numactl

python3 -m pip install msgpack 'mooncake-transfer-engine==0.3.11.post1'

if [[ -f /configs/patches/vllm_numa_bind_hash_fix.py ]]; then
    python3 /configs/patches/vllm_numa_bind_hash_fix.py
fi
