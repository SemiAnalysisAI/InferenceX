#!/usr/bin/env bash

set -euo pipefail

# The pinned backend image already carries vLLM and every pure-Python Router
# dependency. Install the released official Router wheel (and its compiled
# orjson dependency) without replacing the serving runtime.
python3 -m pip install --no-cache-dir "vllm-router==0.1.15"

command -v vllm-router >/dev/null
vllm-router --help >/dev/null
