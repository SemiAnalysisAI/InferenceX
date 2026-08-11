#!/usr/bin/env bash
# Static tests for the connector chain build_kv_transfer_config_json produces. No GPU, no cluster.
#
# Two invariants are worth locking down because both are easy to break in a refactor and neither
# fails loudly at runtime:
#
#   1. MoRIIO is FIRST, the tier is SECOND. MultiConnector loads from the first child that reports
#      matched tokens. On decode MoRIIO is the only thing that can produce the KDA recurrent state --
#      that state is not prefix-cacheable and cannot come out of a content-addressed DRAM tier -- so
#      a tier that got asked first could claim MLA blocks for a request whose KDA state MoRIIO is
#      still fetching. That is precisely the half-restored state the Mooncake fault points at.
#   2. The tier is absent on decode unless asked for. Measured on the Mooncake tier, decode's
#      load_get was 0 with saves only, so a decode-side tier is bandwidth spent on KV nobody rereads.
#
# The function is sourced out of server_vllm.sh rather than reimplemented, so these test the shipping
# code path.
set +e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PASS=0
FAIL=0
check() {
    local what="$1" got="$2" want="$3"
    if [[ "$got" == "$want" ]]; then
        PASS=$((PASS + 1)); printf '  ok   %-56s %s\n' "$what" "$got"
    else
        FAIL=$((FAIL + 1)); printf '  FAIL %-56s got=%s want=%s\n' "$what" "$got" "$want"
    fi
}

# server_vllm.sh does a lot at source time, so lift just the function under test rather than sourcing
# the whole file.
#
# Finding its end needs care: the body embeds a python3 -c '...' block whose own lines include `}` and
# `}))` at column 0, so "first column-0 closing brace" stops in the middle of the JSON builder and
# yields a shim that cannot parse. The reliable marker for this function is that it ends with the
# heredoc terminator EOF followed by the closing brace. If that shape ever changes the extraction
# fails loudly here, which is the right place for it to fail.
SHIM=$(mktemp)
python3 - "$HERE/server_vllm.sh" "$HERE" > "$SHIM" <<'PY'
import sys

path, here = sys.argv[1], sys.argv[2]
lines = open(path, encoding="utf-8").read().splitlines()

start = next(i for i, l in enumerate(lines) if l.startswith("build_kv_transfer_config_json() {"))
end = None
for i in range(start + 1, len(lines)):
    if lines[i] == "}" and lines[i - 1] == "EOF":
        end = i
        break
if end is None:
    sys.exit("could not find the end of build_kv_transfer_config_json (expected EOF then })")

body = "\n".join(lines[start : end + 1])
# The function sources lmcache_mp.sh relative to BASH_SOURCE; in a shim that has to be pinned.
body = body.replace('$(dirname "${BASH_SOURCE[0]}")', here)

print("set +e")
print("host_name=testhost")
print("SLURM_JOB_ID=1")
print("NODE0_ADDR=10.0.0.1")
print("PROXY_PING_PORT=36367")
print("SERVER_PORT=2584")
print(body)
PY
if [[ ! -s "$SHIM" ]]; then
    echo "FATAL: could not extract build_kv_transfer_config_json from server_vllm.sh" >&2
    exit 1
fi
bash -n "$SHIM" || { echo "FATAL: extracted shim is not valid bash" >&2; exit 1; }

run_chain() {
    # usage: run_chain <role> <backend> [extra env assignments...]
    local role="$1" backend="$2"; shift 2
    env KV_OFFLOADING=dram KV_OFFLOAD_BACKEND="$backend" "$@" \
        bash -c ". '$SHIM'; build_kv_transfer_config_json $role" 2>/dev/null
}

jq_py() { python3 -c "import json,sys; d=json.load(sys.stdin); $1"; }

echo "=== mooncake, prefill: MoRIIO first, store second ==="
J=$(run_chain kv_producer mooncake)
check "top-level connector" "$(printf '%s' "$J" | jq_py 'print(d["kv_connector"])')" "MultiConnector"
check "child 0 is MoRIIO" \
  "$(printf '%s' "$J" | jq_py 'print(d["kv_connector_extra_config"]["connectors"][0]["kv_connector"])')" \
  "MoRIIOConnector"
check "child 1 is the Mooncake store" \
  "$(printf '%s' "$J" | jq_py 'print(d["kv_connector_extra_config"]["connectors"][1]["kv_connector"])')" \
  "MooncakeStoreConnector"
check "MoRIIO carries the producer role" \
  "$(printf '%s' "$J" | jq_py 'print(d["kv_connector_extra_config"]["connectors"][0]["kv_role"])')" \
  "kv_producer"

echo
echo "=== mooncake, decode: store dropped when MOONCAKE_DECODE_STORE=0 ==="
J=$(run_chain kv_consumer mooncake MOONCAKE_DECODE_STORE=0)
check "only MoRIIO remains" \
  "$(printf '%s' "$J" | jq_py 'print(len(d["kv_connector_extra_config"]["connectors"]))')" "1"
J=$(run_chain kv_consumer mooncake MOONCAKE_DECODE_STORE=1)
check "store kept when explicitly asked for" \
  "$(printf '%s' "$J" | jq_py 'print(len(d["kv_connector_extra_config"]["connectors"]))')" "2"

echo
echo "=== lmcache-k3, prefill: MoRIIO first, LMCache second, kv_both on the tier ==="
J=$(run_chain kv_producer lmcache-k3 LMCACHE_PORT=6555 LMCACHE_MQ_TIMEOUT=300)
check "child 0 is still MoRIIO" \
  "$(printf '%s' "$J" | jq_py 'print(d["kv_connector_extra_config"]["connectors"][0]["kv_connector"])')" \
  "MoRIIOConnector"
check "child 1 is LMCacheMPConnector" \
  "$(printf '%s' "$J" | jq_py 'print(d["kv_connector_extra_config"]["connectors"][1]["kv_connector"])')" \
  "LMCacheMPConnector"
check "tier takes kv_both, not the node's PD role" \
  "$(printf '%s' "$J" | jq_py 'print(d["kv_connector_extra_config"]["connectors"][1]["kv_role"])')" \
  "kv_both"
check "tier port reaches the child config" \
  "$(printf '%s' "$J" | jq_py 'print(d["kv_connector_extra_config"]["connectors"][1]["kv_connector_extra_config"]["lmcache.mp.port"])')" \
  "6555"
check "no Mooncake child on the lmcache path" \
  "$(printf '%s' "$J" | grep -c Mooncake)" "0"

echo
echo "=== lmcache-k3, decode: tier absent by default, present when asked ==="
J=$(run_chain kv_consumer lmcache-k3 LMCACHE_PORT=6555)
check "decode default: MoRIIO only" \
  "$(printf '%s' "$J" | jq_py 'print(len(d["kv_connector_extra_config"]["connectors"]))')" "1"
J=$(run_chain kv_consumer lmcache-k3 LMCACHE_PORT=6555 LMCACHE_ON_DECODE=true)
check "LMCACHE_ON_DECODE=true attaches it" \
  "$(printf '%s' "$J" | jq_py 'print(d["kv_connector_extra_config"]["connectors"][1]["kv_connector"])')" \
  "LMCacheMPConnector"

echo
echo "=== the load-failure policy stays at the TOP level, where the scheduler reads it ==="
J=$(run_chain kv_producer lmcache-k3 LMCACHE_PORT=6555)
check "kv_load_failure_policy defaults to recompute" \
  "$(printf '%s' "$J" | jq_py 'print(d["kv_load_failure_policy"])')" "recompute"
J=$(run_chain kv_producer lmcache-k3 LMCACHE_PORT=6555 KV_LOAD_FAILURE_POLICY=recompute)
check "explicit value honoured" \
  "$(printf '%s' "$J" | jq_py 'print(d["kv_load_failure_policy"])')" "recompute"

echo
echo "=== no tier: the MoRIIO-only fallback is still valid JSON with the policy present ==="
J=$(env KV_OFFLOADING=none bash -c ". '$SHIM'; build_kv_transfer_config_json kv_producer" 2>/dev/null)
check "fallback connector" "$(printf '%s' "$J" | jq_py 'print(d["kv_connector"])')" "MoRIIOConnector"
check "fallback carries the policy" \
  "$(printf '%s' "$J" | jq_py 'print(d["kv_load_failure_policy"])')" "recompute"

rm -f "$SHIM"
echo
printf '=== %d passed, %d failed ===\n' "$PASS" "$FAIL"
[[ "$FAIL" -eq 0 ]]
