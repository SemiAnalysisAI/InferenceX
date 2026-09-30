#!/usr/bin/env bash
# Pin the worker's Mooncake store client to the release the master runs.
set -euo pipefail
pip_install=(python3 -m pip install)
if python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages; then
    pip_install+=(--break-system-packages)
fi
"${pip_install[@]}" --quiet --no-cache-dir --no-deps --force-reinstall mooncake-transfer-engine-cuda13==0.3.11.post1
python3 -c "from mooncake.store import MooncakeDistributedStore" >/dev/null
