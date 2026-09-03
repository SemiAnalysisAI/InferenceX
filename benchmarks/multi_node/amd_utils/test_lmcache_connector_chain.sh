#!/usr/bin/env bash
# Static connector-chain test; requires only bash and python3.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHIM=$(mktemp)
trap 'rm -f "$SHIM"' EXIT

python3 - "$HERE/server_vllm.sh" "$HERE" > "$SHIM" <<'PY'
import sys

path, here = sys.argv[1:]
lines = open(path, encoding="utf-8-sig").read().splitlines()
start = next(i for i, line in enumerate(lines) if line.startswith("lmcache_attached_to_role() {"))
end = next(i for i in range(start, len(lines)) if lines[i].startswith("# ===") and "Container Synchronization" in lines[i + 1])
body = "\n".join(lines[start:end]).replace('$(dirname "${BASH_SOURCE[0]}")', here)
print("set -euo pipefail")
print(body)
PY
bash -n "$SHIM"

json_get() {
    python3 -c "import json,sys; d=json.load(sys.stdin); $1"
}

run_chain() {
    local role="$1" offload="$2"
    KV_OFFLOADING="$offload" KV_OFFLOAD_BACKEND=lmcache-k3 \
        LMCACHE_ON_DECODE="${3:-false}" \
        NODE0_ADDR=10.0.0.1 PROXY_PING_PORT=36367 SERVER_PORT=2584 \
        LMCACHE_PORT=6555 LMCACHE_MQ_TIMEOUT=6000 \
        bash -c ". '$SHIM'; build_kv_transfer_config_json '$role'"
}

J=$(run_chain kv_producer none)
[[ "$(printf '%s' "$J" | json_get 'print(d["kv_connector"])')" == "MoRIIOConnector" ]]
[[ "$(printf '%s' "$J" | json_get 'print(d["kv_role"])')" == "kv_producer" ]]
[[ "$(printf '%s' "$J" | json_get 'print(d["kv_load_failure_policy"])')" == "recompute" ]]

for role in kv_producer; do
    J=$(run_chain "$role" dram)
    [[ "$(printf '%s' "$J" | json_get 'print(d["kv_connector"])')" == "MultiConnector" ]]
    [[ "$(printf '%s' "$J" | json_get 'print(d["kv_load_failure_policy"])')" == "recompute" ]]
    [[ "$(printf '%s' "$J" | json_get 'print(d["kv_connector_extra_config"]["connectors"][0]["kv_connector"])')" == "MoRIIOConnector" ]]
    [[ "$(printf '%s' "$J" | json_get 'print(d["kv_connector_extra_config"]["connectors"][0]["kv_role"])')" == "$role" ]]
    [[ "$(printf '%s' "$J" | json_get 'print(d["kv_connector_extra_config"]["connectors"][1]["kv_connector"])')" == "LMCacheMPConnector" ]]
    [[ "$(printf '%s' "$J" | json_get 'print(d["kv_connector_extra_config"]["connectors"][1]["kv_role"])')" == "kv_both" ]]
    [[ "$(printf '%s' "$J" | json_get 'print(d["kv_connector_extra_config"]["connectors"][1]["kv_connector_extra_config"]["lmcache.mp.mq_timeout"])')" == "6000.0" ]]
done

J=$(run_chain kv_consumer dram)
[[ "$(printf '%s' "$J" | json_get 'print(d["kv_connector"])')" == "MoRIIOConnector" ]]
[[ "$(printf '%s' "$J" | json_get 'print(d["kv_role"])')" == "kv_consumer" ]]

J=$(run_chain kv_consumer dram true)
[[ "$(printf '%s' "$J" | json_get 'print(d["kv_connector"])')" == "MultiConnector" ]]
[[ "$(printf '%s' "$J" | json_get 'print(d["kv_connector_extra_config"]["connectors"][1]["kv_connector"])')" == "LMCacheMPConnector" ]]

echo "LMCache connector-chain tests passed"
