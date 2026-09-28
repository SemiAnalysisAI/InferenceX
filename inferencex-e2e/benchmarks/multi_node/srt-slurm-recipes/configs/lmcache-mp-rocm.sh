#!/usr/bin/env bash
# Start one LMCache MP server per TP rank in the worker container before vLLM,
# as the legacy MI300X MiniMax-M3 AgentX script did. A variant opts in with
# LMCACHE_SHARDS and LMCACHE_L1_SHARD_GB; its kv-transfer-config lists
# tcp://127.0.0.1:5555 through 5555 + LMCACHE_SHARDS - 1.
set -euo pipefail
[[ -n "${LMCACHE_SHARDS:-}" ]] || exit 0
: "${LMCACHE_L1_SHARD_GB:?}"
version=0.5.3
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

pids=()
for ((shard = 0; shard < LMCACHE_SHARDS; shard++)); do
    # Detached so the servers outlive this preamble and serve the vLLM step.
    setsid lmcache server \
        --host 127.0.0.1 --port $((5555 + shard)) \
        --http-host 127.0.0.1 --http-port $((8080 + shard)) \
        --l1-size-gb "$LMCACHE_L1_SHARD_GB" --l1-init-size-gb 10 \
        --l1-read-ttl-seconds 7200 --chunk-size 256 --max-workers 2 \
        --eviction-policy LRU --supported-transfer-mode lmcache_driven \
        > "/logs/lmcache_server_${shard}.log" 2>&1 < /dev/null &
    pids+=($!)
done
for ((shard = 0; shard < LMCACHE_SHARDS; shard++)); do
    for ((attempt = 0; ; attempt++)); do
        python3 -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2)' \
            "http://127.0.0.1:$((8080 + shard))/healthcheck" 2> /dev/null && break
        if ! kill -0 "${pids[$shard]}" 2>/dev/null || (( attempt >= 600 )); then
            echo "ERROR: LMCache server $shard did not become ready" >&2
            tail -n 50 "/logs/lmcache_server_${shard}.log" >&2 || true
            exit 1
        fi
        sleep 1
    done
done
echo "LMCache: ${LMCACHE_SHARDS} servers ready, ${LMCACHE_L1_SHARD_GB} GB L1 each"
