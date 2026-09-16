#!/usr/bin/env bash
set -eo pipefail

python3 -m pip uninstall --break-system-packages -y mooncake-transfer-engine-cuda13 mooncake-transfer-engine-efa-cuda13
python3 -m pip install --break-system-packages --no-deps mooncake-transfer-engine-efa-cuda13==0.3.13.post1
