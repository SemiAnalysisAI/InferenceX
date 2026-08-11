#!/usr/bin/env bash
# Static tests for lmcache_mp.sh's numeric couplings. No GPU, no model, no cluster.
#
# Worth having because every constraint encoded there has a run that died on it, and all of those
# failures land at engine init -- after the weight load -- so rediscovering one costs a whole
# bring-up. These assertions take under a second.
set +e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
. "$HERE/lmcache_mp.sh"

PASS=0
FAIL=0
check() {
    local what="$1" got="$2" want="$3"
    if [[ "$got" == "$want" ]]; then
        PASS=$((PASS + 1)); printf '  ok   %-52s %s\n' "$what" "$got"
    else
        FAIL=$((FAIL + 1)); printf '  FAIL %-52s got=%s want=%s\n' "$what" "$got" "$want"
    fi
}
expect_fail() {
    local what="$1"; shift
    if "$@" >/dev/null 2>&1; then
        FAIL=$((FAIL + 1)); printf '  FAIL %-52s expected non-zero exit\n' "$what"
    else
        PASS=$((PASS + 1)); printf '  ok   %-52s rejected\n' "$what"
    fi
}
reset_env() {
    unset LMCACHE_UNIFIED_BLOCK LMCACHE_CHUNK_SIZE LMCACHE_MNBT \
          LMCACHE_CHUNK_SIZE_OVERRIDE LMCACHE_MAX_NUM_BATCHED_TOKENS KV_CACHE_DTYPE
}

echo "=== fp8: N=1536, chunk=2N (the KDA group's tokens_per_block, not N), mnbt just under 2N ==="
reset_env; KV_CACHE_DTYPE=fp8 lmcache_mp_derive_geometry 8 >/dev/null
check "fp8 N"                     "$LMCACHE_UNIFIED_BLOCK" 1536
check "fp8 chunk-size"            "$LMCACHE_CHUNK_SIZE"    3072
check "fp8 max-num-batched-tokens" "$LMCACHE_MNBT"          3000
check "fp8 chunk % N == 0"        "$(( LMCACHE_CHUNK_SIZE % LMCACHE_UNIFIED_BLOCK ))" 0
check "fp8 mnbt in [N,2N)"        "$(( LMCACHE_MNBT >= LMCACHE_UNIFIED_BLOCK && LMCACHE_MNBT < 2*LMCACHE_UNIFIED_BLOCK ))" 1

echo
echo "=== bf16: N=768, chunk=N (where the two coincide, which is why fp8 exposed the bug) ==="
reset_env; KV_CACHE_DTYPE=auto lmcache_mp_derive_geometry 8 >/dev/null
check "bf16 N"                     "$LMCACHE_UNIFIED_BLOCK" 768
check "bf16 chunk-size"            "$LMCACHE_CHUNK_SIZE"    768
check "bf16 max-num-batched-tokens" "$LMCACHE_MNBT"          1464
check "bf16 mnbt in [N,2N)"        "$(( LMCACHE_MNBT >= LMCACHE_UNIFIED_BLOCK && LMCACHE_MNBT < 2*LMCACHE_UNIFIED_BLOCK ))" 1

echo
echo "=== fp8 variants name the same page size ==="
reset_env; KV_CACHE_DTYPE=fp8_e4m3 lmcache_mp_derive_geometry 8 >/dev/null
check "fp8_e4m3 N" "$LMCACHE_UNIFIED_BLOCK" 1536

echo
echo "=== the guards actually reject, rather than warning and continuing ==="
reset_env
expect_fail "chunk not a multiple of N" \
    env KV_CACHE_DTYPE=fp8 LMCACHE_CHUNK_SIZE_OVERRIDE=2000 \
        bash -c ". '$HERE/lmcache_mp.sh'; lmcache_mp_derive_geometry 8"
expect_fail "mnbt below N" \
    env KV_CACHE_DTYPE=fp8 LMCACHE_MAX_NUM_BATCHED_TOKENS=1000 \
        bash -c ". '$HERE/lmcache_mp.sh'; lmcache_mp_derive_geometry 8"
expect_fail "mnbt at 2N (interval is half-open)" \
    env KV_CACHE_DTYPE=fp8 LMCACHE_MAX_NUM_BATCHED_TOKENS=3072 \
        bash -c ". '$HERE/lmcache_mp.sh'; lmcache_mp_derive_geometry 8"
expect_fail "vLLM's own default 8192 is outside [N,2N)" \
    env KV_CACHE_DTYPE=fp8 LMCACHE_MAX_NUM_BATCHED_TOKENS=8192 \
        bash -c ". '$HERE/lmcache_mp.sh'; lmcache_mp_derive_geometry 8"

echo
echo "=== an explicit N override still gets validated, not trusted ==="
reset_env; KV_CACHE_DTYPE=fp8 LMCACHE_UNIFIED_BLOCK=1024 lmcache_mp_derive_geometry 8 >/dev/null
check "override N=1024 -> chunk 2048" "$LMCACHE_CHUNK_SIZE" 2048
check "override N=1024 -> mnbt 1976"  "$LMCACHE_MNBT"        1976

echo
echo "=== connector JSON shape ==="
reset_env
LMCACHE_PORT=6555 LMCACHE_MQ_TIMEOUT=300 LMCACHE_CONNECTOR_MODULE_PATH= \
  J=$(lmcache_mp_connector_json)
check "no module path -> connector name" \
  "$(printf '%s' "$J" | python3 -c 'import json,sys; print(json.load(sys.stdin)["kv_connector"])')" \
  "LMCacheMPConnector"
check "no module path -> kv_role is kv_both (tier, not a transfer leg)" \
  "$(printf '%s' "$J" | python3 -c 'import json,sys; print(json.load(sys.stdin)["kv_role"])')" \
  "kv_both"
check "no module path -> field absent" \
  "$(printf '%s' "$J" | python3 -c 'import json,sys; print("kv_connector_module_path" in json.load(sys.stdin))')" \
  "False"

LMCACHE_PORT=6555 LMCACHE_HOST=127.0.0.1 LMCACHE_MQ_TIMEOUT=300 \
LMCACHE_CONNECTOR_MODULE_PATH=lmcache.integration.vllm.lmcache_mp_connector \
  J2=$(lmcache_mp_connector_json)
check "module path -> field present" \
  "$(printf '%s' "$J2" | python3 -c 'import json,sys; print(json.load(sys.stdin)["kv_connector_module_path"])')" \
  "lmcache.integration.vllm.lmcache_mp_connector"
check "module path -> ZMQ-style host added" \
  "$(printf '%s' "$J2" | python3 -c 'import json,sys; print(json.load(sys.stdin)["kv_connector_extra_config"]["lmcache.mp.host"])')" \
  "tcp://127.0.0.1"

echo
echo "=== serve args ==="
reset_env; KV_CACHE_DTYPE=fp8 lmcache_mp_derive_geometry 8 >/dev/null
A=$(LMCACHE_MAX_NUM_SEQS=32 LMCACHE_GPU_MEM_UTIL=0.85 lmcache_mp_serve_args | paste -sd' ')
check "serve args carry the derived budget" \
  "$(printf '%s' "$A" | grep -c -- '--max-num-batched-tokens 3000')" 1
check "serve args force align" \
  "$(printf '%s' "$A" | grep -c -- '--mamba-cache-mode align')" 1
check "serve args carry the GPU headroom levers" \
  "$(printf '%s' "$A" | grep -c -- '--max-num-seqs 32 --gpu-memory-utilization 0.85')" 1
A2=$(lmcache_mp_serve_args | paste -sd' ')
check "headroom levers omitted when unset" \
  "$(printf '%s' "$A2" | grep -c -- '--max-num-seqs')" 0

echo
printf '=== %d passed, %d failed ===\n' "$PASS" "$FAIL"
[[ "$FAIL" -eq 0 ]]
