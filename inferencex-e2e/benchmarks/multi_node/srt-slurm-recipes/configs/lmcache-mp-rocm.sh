#!/usr/bin/env bash
# Install the ROCm LMCache wheel, at the master config's kv-offload-backend version, for
# the MI300X MiniMax-M3 AgentX LMCache point. It runs twice: as the vLLM worker's
# setup_script (the connector is imported there) and in the lmcache-server service's
# preamble (its own container).
set -eo pipefail
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars KV_OFFLOAD_BACKEND_VERSION
version=$KV_OFFLOAD_BACKEND_VERSION
pip_install=(python3 -m pip install)
if python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages; then
    pip_install+=(--break-system-packages)
fi
"${pip_install[@]}" --quiet --no-cache-dir --no-deps \
    "sortedcontainers==2.4.0" \
    "opentelemetry-exporter-prometheus==0.61b0" \
    "cupy-rocm-7-0==14.1.1" \
    "lmcache==${version}" \
    --find-links "https://github.com/LMCache/LMCache/releases/expanded_assets/v${version}-rocm"
python3 -c "import cupy; import lmcache.integration.vllm.lmcache_mp_connector; import opentelemetry.exporter.prometheus"
