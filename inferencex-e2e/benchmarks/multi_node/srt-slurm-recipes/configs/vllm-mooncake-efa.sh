#!/usr/bin/env bash
# Swap the image's Mooncake wheel for the EFA build, for clusters whose NICs are EFA: below 0.3.14 only
# the -efa wheel has the EFA transport. It links the libfabric the image ships. The wheel is pinned to the
# master config's kv-offload-backend version. The vLLM workers (setup_script) and Mooncake stores
# (preamble) share a container on each node, so one run does the work.
set -eo pipefail
source /infmax-workspace/benchmarks/check_env.sh
check_env_vars KV_OFFLOAD_BACKEND_VERSION
exec 9>/tmp/vllm-mooncake-efa.lock
flock 9
marker=/var/lib/vllm-mooncake-efa.done
[ -f "$marker" ] && exit 0
pip=(python3 -m pip)
break_system=()
if "${pip[@]}" install --help 2>/dev/null | grep -q -- --break-system-packages; then
    break_system=(--break-system-packages)
fi
"${pip[@]}" uninstall -y "${break_system[@]}" mooncake-transfer-engine mooncake-transfer-engine-cuda13
"${pip[@]}" install --quiet --no-cache-dir --no-deps "${break_system[@]}" \
    "mooncake-transfer-engine-efa-cuda13==${KV_OFFLOAD_BACKEND_VERSION}"
mooncake_engine=$(python3 -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("mooncake").origin).with_name("engine.so"))')
fabric_loaded=$(ldd "$mooncake_engine" | awk '$1 == "libfabric.so.1" { print $3 }')
printf 'Mooncake libfabric: %s\n' "$fabric_loaded"
if [ ! -e "$fabric_loaded" ]; then
    echo "Error: the image ships no libfabric for Mooncake's EFA transport" >&2
    exit 1
fi
/opt/amazon/efa/bin/fi_info -p efa | grep -q 'provider: efa'
touch "$marker"
