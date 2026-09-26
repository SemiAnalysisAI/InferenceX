#!/usr/bin/env bash
# Install the measured H200 DeepSeek-V4.1-Flash block-32 tilings into the worker's SGLang.
set -euo pipefail
agentic=/infmax-workspace/benchmarks/single_node/agentic
python3 "$agentic/install_h200_block32_configs.py" "$agentic/kernel_configs/h200_dsv41_block32" /logs "$DSV41_BLOCK32_TP"
