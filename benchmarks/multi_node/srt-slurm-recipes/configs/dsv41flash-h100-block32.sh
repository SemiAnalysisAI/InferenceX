#!/usr/bin/env bash
# Install the measured H100 DeepSeek-V4.1-Flash block-32 tilings into the worker's SGLang.
set -euo pipefail
assets=/infmax-workspace/benchmarks/multi_node/srt-slurm-recipes/configs/dsv41flash-block32
python3 "$assets/install_h100_block32_configs.py" "$assets/kernel_configs/h100_dsv41_block32" /logs
