#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --no-cache-dir --upgrade "vllm-router==0.1.15"
command -v vllm-router >/dev/null
python3 - <<'PY'
from importlib.metadata import version

assert version("vllm-router") == "0.1.15"
print(f"vllm-router {version('vllm-router')}")
PY
