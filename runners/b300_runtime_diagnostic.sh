#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_CACHE=/workspace/diagnostics/pr3088-runtime-cpu/runtime-tests/hf-cache
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /tmp
exec timeout --signal=TERM --kill-after=10s 150s python3 /workspace/diagnostics/pr3088-runtime-cpu/runtime_check.py
