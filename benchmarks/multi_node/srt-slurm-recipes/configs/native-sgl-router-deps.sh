#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --no-cache-dir --upgrade "sglang-router==0.3.2"
python3 -c 'import sglang_router; print(sglang_router.__file__)'
