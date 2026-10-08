#!/usr/bin/env bash
# Start the standalone LMCache server that ATOM's lmcache_mp KV tier
# (ATOM_KV_OFFLOAD=lmcache_mp) connects to, from the lmcache CLI the ATOM image
# ships; nothing is installed. srtctl runs this in the worker container before
# ATOM starts, so the server shares the GPUs whose KV it copies and lives until
# the worker exits. Its port and sizes come from the recipe's role env.
set -euo pipefail

source /infmax-workspace/benchmarks/check_env.sh
check_env_vars LMCACHE_MP_PORT LMCACHE_CHUNK_SIZE LMCACHE_MP_L1_SIZE_GB \
    LMCACHE_MP_L1_READ_TTL_SECONDS LMCACHE_MP_EVICTION_WATERMARK

log=/tmp/lmcache-mp-server.log
setsid nohup lmcache server \
    --host 127.0.0.1 --port "$LMCACHE_MP_PORT" \
    --chunk-size "$LMCACHE_CHUNK_SIZE" --null-block-id -1 --separate-object-groups \
    --supported-transfer-mode lmcache_driven \
    --l1-size-gb "$LMCACHE_MP_L1_SIZE_GB" --l1-use-lazy \
    --l1-read-ttl-seconds "$LMCACHE_MP_L1_READ_TTL_SECONDS" \
    --eviction-policy LRU --eviction-trigger-watermark "$LMCACHE_MP_EVICTION_WATERMARK" \
    >"$log" 2>&1 </dev/null &
pid=$!

for _ in $(seq 1 150); do
    if (echo >"/dev/tcp/127.0.0.1/$LMCACHE_MP_PORT") 2>/dev/null; then
        echo "LMCache MP server (pid $pid) listening on 127.0.0.1:$LMCACHE_MP_PORT; log $log"
        exit 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "LMCache MP server exited before listening:" >&2
        tail -n 50 "$log" >&2
        exit 1
    fi
    sleep 2
done
echo "LMCache MP server did not listen on 127.0.0.1:$LMCACHE_MP_PORT within 300 s:" >&2
tail -n 50 "$log" >&2
exit 1
