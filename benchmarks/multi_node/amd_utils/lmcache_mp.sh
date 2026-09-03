#!/usr/bin/env bash
# Node-local LMCache MP server used as the optional DRAM child of vLLM MultiConnector.
# This file is sourced by server_vllm.sh and has no source-time side effects.

lmcache_mp_install_native_deps() {
    local lib missing=0
    for lib in libglog.so.0 libjsoncpp.so.25 libibverbs.so.1 librdmacm.so.1 libnuma.so.1; do
        if ! ldconfig -p 2>/dev/null | grep -q "$lib"; then
            missing=1
            break
        fi
    done
    (( missing == 0 )) && return 0

    apt-get update -q
    DEBIAN_FRONTEND=noninteractive apt-get install -q -y --no-install-recommends \
        libgoogle-glog0v5 libjsoncpp25 libibverbs1 librdmacm1 libnuma1
}

lmcache_mp_install() {
    local ref="${LMCACHE_GIT_REF:-140819c9d57a975dbc5678a6459a218e544cb58b}"
    local src="${LMCACHE_SRC:-/opt/lmcache-src}"
    lmcache_mp_install_native_deps || return 1

    if [[ -d "$src/.git" ]] \
       && [[ "$(git -C "$src" rev-parse HEAD 2>/dev/null)" == "$ref" ]] \
       && python3 -c "from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector" 2>/dev/null; then
        return 0
    fi

    rm -rf "$src"
    git clone --filter=blob:none https://github.com/LMCache/LMCache.git "$src" || return 1
    git -C "$src" checkout --detach "$ref" || return 1
    (
        cd "$src"
        export BUILD_WITH_HIP=1
        if command -v uv >/dev/null 2>&1; then
            uv pip install --system -e . --no-build-isolation
        else
            python3 -m pip install -e . --no-build-isolation
        fi
    ) || return 1

    if [[ "$(git -C "$src" rev-parse HEAD 2>/dev/null)" != "$ref" ]]; then
        return 1
    fi
    python3 -c \
        "from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector" \
        || return 1
}

lmcache_mp_assert_hybrid_ok() {
    python3 - <<'PY'
from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector

assert issubclass(LMCacheMPConnector, SupportsHMA)
PY
}

lmcache_mp_size_l1() {
    if [[ -z "${LMCACHE_L1_SIZE_GB:-}" ]]; then
        LMCACHE_L1_SIZE_GB="${TOTAL_CPU_DRAM_GB:-}"
    fi
    if [[ ! "$LMCACHE_L1_SIZE_GB" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: LMCache requires positive LMCACHE_L1_SIZE_GB or TOTAL_CPU_DRAM_GB" >&2
        return 1
    fi
    if [[ "${TOTAL_CPU_DRAM_GB:-}" =~ ^[1-9][0-9]*$ ]] \
       && (( LMCACHE_L1_SIZE_GB > TOTAL_CPU_DRAM_GB )); then
        echo "ERROR: LMCACHE_L1_SIZE_GB=${LMCACHE_L1_SIZE_GB} exceeds TOTAL_CPU_DRAM_GB=${TOTAL_CPU_DRAM_GB}" >&2
        return 1
    fi
    export LMCACHE_L1_SIZE_GB
}

lmcache_mp_server_args() {
    LMCACHE_HOST="${LMCACHE_HOST:-127.0.0.1}"
    LMCACHE_PORT="${LMCACHE_PORT:-6555}"
    LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8090}"
    # SA DCP8 geometry: 1536-token attention block * DCP=8 = 12288, which is
    # also the scheduler UNIFIED page. Prefill is DCP=1 but still uses 12288
    # because it is a valid multiple of every 1536-token hybrid KV group.
    LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-12288}"
    LMCACHE_L1_INIT_SIZE_GB="${LMCACHE_L1_INIT_SIZE_GB:-10}"
    LMCACHE_MAX_CPU_WORKERS="${LMCACHE_MAX_CPU_WORKERS:-8}"
    LMCACHE_MAX_GPU_WORKERS="${LMCACHE_MAX_GPU_WORKERS:-1}"

    printf '%s\n' \
        lmcache server \
        --host "$LMCACHE_HOST" \
        --port "$LMCACHE_PORT" \
        --http-host "$LMCACHE_HOST" \
        --http-port "$LMCACHE_HTTP_PORT" \
        --l1-size-gb "$LMCACHE_L1_SIZE_GB" \
        --l1-init-size-gb "$LMCACHE_L1_INIT_SIZE_GB" \
        --chunk-size "$LMCACHE_CHUNK_SIZE" \
        --separate-object-groups \
        --max-cpu-workers "$LMCACHE_MAX_CPU_WORKERS" \
        --max-gpu-workers "$LMCACHE_MAX_GPU_WORKERS" \
        --eviction-policy "${LMCACHE_EVICTION_POLICY:-LRU}" \
        --supported-transfer-mode "${LMCACHE_TRANSFER_MODE:-lmcache_driven}" \
        --shm-name "${LMCACHE_SHM_NAME:-}"
}

lmcache_mp_start() {
    local logdir="${1:-/run_logs}"
    local host_name="${2:-$(hostname)}"
    local -a cmd

    lmcache_mp_size_l1 || return 1
    mapfile -t cmd < <(lmcache_mp_server_args)
    LMCACHE_LOG="${logdir}/lmcache_${host_name}.log"
    printf '%q ' "${cmd[@]}" > "${logdir}/lmcache_${host_name}_command.txt"
    printf '\n' >> "${logdir}/lmcache_${host_name}_command.txt"

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY RUN: ${cmd[*]}"
        return 0
    fi

    "${cmd[@]}" > "$LMCACHE_LOG" 2>&1 &
    LMCACHE_PID=$!
    export LMCACHE_PID LMCACHE_LOG LMCACHE_HOST LMCACHE_PORT LMCACHE_HTTP_PORT
    lmcache_mp_wait_ready
}

lmcache_mp_wait_ready() {
    local timeout="${LMCACHE_READY_TIMEOUT_S:-600}"
    local start=$SECONDS
    while (( SECONDS - start < timeout )); do
        if curl -sf -m 2 -o /dev/null "http://${LMCACHE_HOST}:${LMCACHE_HTTP_PORT}/healthcheck"; then
            echo "[lmcache] ready on ${LMCACHE_HOST}:${LMCACHE_PORT}"
            return 0
        fi
        if [[ -n "${LMCACHE_PID:-}" ]] && ! kill -0 "$LMCACHE_PID" 2>/dev/null; then
            echo "ERROR: LMCache exited during startup" >&2
            cat "$LMCACHE_LOG" >&2 || true
            return 1
        fi
        sleep 1
    done
    echo "ERROR: LMCache did not become ready within ${timeout}s" >&2
    cat "$LMCACHE_LOG" >&2 || true
    return 1
}

lmcache_mp_stop() {
    [[ -n "${LMCACHE_PID:-}" ]] || return 0
    kill "$LMCACHE_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
        kill -0 "$LMCACHE_PID" 2>/dev/null || return 0
        sleep 1
    done
    kill -9 "$LMCACHE_PID" 2>/dev/null || true
}

lmcache_mp_connector_json() {
    LMCACHE_PORT="${LMCACHE_PORT:-6555}" \
    LMCACHE_MQ_TIMEOUT="${LMCACHE_MQ_TIMEOUT:-6000}" \
    python3 -c '
import json, os
print(json.dumps({
    "kv_connector": "LMCacheMPConnector",
    "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "lmcache.mp.port": int(os.environ["LMCACHE_PORT"]),
        "lmcache.mp.mq_timeout": float(os.environ["LMCACHE_MQ_TIMEOUT"]),
    },
}))
'
}
