#!/usr/bin/env bash
# Static tests for the LMCache source pin, command defaults, sizing, and connector JSON.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
. "$HERE/lmcache_mp.sh"

TOTAL_CPU_DRAM_GB=1799
unset LMCACHE_L1_SIZE_GB
lmcache_mp_size_l1
[[ "$LMCACHE_L1_SIZE_GB" == 1799 ]]

mapfile -t args < <(lmcache_mp_server_args)
cmd=" ${args[*]} "
for expected in \
    " --chunk-size 12288 " \
    " --separate-object-groups " \
    " --max-cpu-workers 8 " \
    " --max-gpu-workers 1 " \
    " --eviction-policy LRU " \
    " --supported-transfer-mode lmcache_driven " \
    " --l1-init-size-gb 10 "
do
    [[ "$cmd" == *"$expected"* ]] || {
        echo "missing LMCache command fragment: $expected" >&2
        exit 1
    }
done

J=$(lmcache_mp_connector_json)
python3 - "$J" <<'PY'
import json
import sys

cfg = json.loads(sys.argv[1])
assert cfg["kv_connector"] == "LMCacheMPConnector"
assert cfg["kv_connector_module_path"] == "lmcache.integration.vllm.lmcache_mp_connector"
assert cfg["kv_role"] == "kv_both"
assert cfg["kv_connector_extra_config"]["lmcache.mp.port"] == 6555
assert cfg["kv_connector_extra_config"]["lmcache.mp.mq_timeout"] == 6000.0
PY

grep -q 'LMCACHE_GIT_REF:-140819c9d57a975dbc5678a6459a218e544cb58b' "$HERE/lmcache_mp.sh"
grep -q 'git clone --filter=blob:none https://github.com/LMCache/LMCache.git' "$HERE/lmcache_mp.sh"
grep -q -- 'pip install -e . --no-build-isolation' "$HERE/lmcache_mp.sh"
! grep -Eq 'nightly-rocm|LMCACHE_VERSION' "$HERE/lmcache_mp.sh"

echo "LMCache command and sizing tests passed"
