#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --no-cache-dir --upgrade "vllm-router==0.1.15"
command -v vllm-router >/dev/null
vllm-router --version
