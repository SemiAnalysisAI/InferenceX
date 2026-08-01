#!/bin/bash

set -euo pipefail

bash /configs/patches/vllm-container-deps.sh
python3 /configs/patches/patch_kimi_k3_v2_ds_prefix_cache.py
