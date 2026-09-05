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

# lmcache_mp_server_args is normally invoked through process substitution.
# Verify lmcache_mp_start initializes the readiness endpoint in the parent
# shell instead of losing it with the command-producing subshell.
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
unset LMCACHE_HOST LMCACHE_PORT LMCACHE_HTTP_PORT
DRY_RUN=1 lmcache_mp_start "$tmpdir" test-host >/dev/null
[[ "$LMCACHE_HOST" == 127.0.0.1 ]]
[[ "$LMCACHE_PORT" == 6555 ]]
[[ "$LMCACHE_HTTP_PORT" == 8090 ]]

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

grep -q 'LMCACHE_VERSION:-0.5.5.dev101+rocm7.2' "$HERE/lmcache_mp.sh"
grep -q 'releases/expanded_assets/nightly-rocm' "$HERE/lmcache_mp.sh"
grep -q -- 'pip install --quiet --no-cache-dir --no-deps' "$HERE/lmcache_mp.sh"
! grep -Eq 'LMCACHE_GIT_REF|git clone|pip install -e' "$HERE/lmcache_mp.sh"

echo "LMCache command and sizing tests passed"
