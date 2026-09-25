#!/usr/bin/env bash
# Split heterogeneous KV backing storage into valid SimpleCPUOffload regions in the worker's vLLM.
set -euo pipefail
python3 /infmax-workspace/runners/patch_vllm_simple_kv_offload.py
