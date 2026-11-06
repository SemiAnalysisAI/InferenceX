#!/usr/bin/env bash
# Install the ROCm LMCache wheel, at the master config's kv-offload-backend version, for
# the MI355X Kimi-K3.1 vLLM AgentX LMCache point. It runs twice: as the vLLM worker's
# setup_script (the connector is imported there) and in the lmcache-server service's
# preamble (its own container).
set -eo pipefail
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars KV_OFFLOAD_BACKEND_VERSION
pip_install=(python3 -m pip install)
if python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages; then
    pip_install+=(--break-system-packages)
fi
# Dev builds are published only on the rolling nightly-rocm release.
release="v${KV_OFFLOAD_BACKEND_VERSION}-rocm"
if [[ "$KV_OFFLOAD_BACKEND_VERSION" == *.dev* ]]; then
    release=nightly-rocm
fi
"${pip_install[@]}" --quiet --no-cache-dir --no-deps \
    "sortedcontainers==2.4.0" \
    "opentelemetry-exporter-prometheus==0.61b0" \
    "cupy-rocm-7-0==14.1.1" \
    "lmcache==${KV_OFFLOAD_BACKEND_VERSION}" \
    --find-links "https://github.com/LMCache/LMCache/releases/expanded_assets/${release}"
# dev276 cannot tile Kimi K3.1's KDA state row (411 bytes per token, no page padding) and
# fails register_kv_caches. Overlay the one file LMCache #6318 (3e47dfbe) changes in the
# package until a nightly-rocm wheel ships it; dev276 plus that diff equals this file.
if [[ "$KV_OFFLOAD_BACKEND_VERSION" == "0.5.7.dev276+rocm7.2" ]]; then
    python3 - <<'PY'
import hashlib, urllib.request
import lmcache.integration.vllm.kv_cache_group_edits as m
url = ("https://raw.githubusercontent.com/LMCache/LMCache/"
       "3e47dfbe/lmcache/integration/vllm/kv_cache_group_edits.py")
src = urllib.request.urlopen(url, timeout=60).read()
assert hashlib.sha256(src).hexdigest() == "2b2c809339bb685ef3f955be3467d33be9e549283f24de49ba0084fd655b7c03", "unexpected LMCache #6318 file"
open(m.__file__, "wb").write(src)
print(f"patched {m.__file__} with LMCache #6318")
PY
fi
python3 -c "import cupy; import lmcache.integration.vllm.lmcache_mp_connector; import opentelemetry.exporter.prometheus"
