#!/usr/bin/env bash
# Install Mooncake at the master config's kv-offload-backend version, for the vLLM worker's
# store client (setup_script) and the mooncake-master service (preamble) alike.
set -eo pipefail
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars KV_OFFLOAD_BACKEND_VERSION
pip_install=(python3 -m pip install)
if python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages; then
    pip_install+=(--break-system-packages)
fi
"${pip_install[@]}" --quiet --no-cache-dir --no-deps --force-reinstall \
    "mooncake-transfer-engine-cuda13==${KV_OFFLOAD_BACKEND_VERSION}"
python3 -c "from mooncake.store import MooncakeDistributedStore" >/dev/null
