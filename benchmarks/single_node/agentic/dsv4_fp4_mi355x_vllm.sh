#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K2.5 FP4 on MI355X using vLLM.
#
# Required env vars:
#   MODEL, TP, CONC, OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR
#
# OFFLOADING values:
#   none    - vLLM GPU KV only.
#   cpu     - vLLM native CPU offload.
#   lmcache - LMCache MP server + vLLM LMCacheMPConnector.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR

PORT=${PORT:-8888}
DURATION=${DURATION:-1800}
MAX_DELAY=${MAX_DELAY:-60}
ADVANCE_MIN=${ADVANCE_MIN:-0.0}
ADVANCE_MAX=${ADVANCE_MAX:-0.7}
EP_SIZE=${EP_SIZE:-1}
if [[ -z "${MAX_MODEL_LEN:-}" || "$MAX_MODEL_LEN" == "0" ]]; then
    MAX_MODEL_LEN=169472
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# ROCR/HIP visibility for vLLM 0.14+
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
rocm-smi || true
amd-smi || true

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# Install amd-quark for MXFP4 (manual install due to ROCm vLLM bug)
pip install amd-quark

# Workaround for MEC FW <177 RCCL memory reclaim issue
version=$(rocm-smi --showfw 2>/dev/null | grep MEC | head -n 1 | awk '{print $NF}')
if [[ "$version" == "" || ${version:-0} -lt 177 ]]; then
    export HSA_NO_SCRATCH_RECLAIM=1
fi

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
LMCACHE_LOG="$RESULT_DIR/lmcache_server.log"
mkdir -p "$RESULT_DIR"

OFFLOAD_ARGS=()
HYBRID_KV_ARGS=(--no-disable-hybrid-kv-cache-manager)
PREFIX_CACHE_ARGS=()
LMCACHE_PID=""

cleanup_lmcache_server() {
    if [[ -n "$LMCACHE_PID" ]] && kill -0 "$LMCACHE_PID" 2>/dev/null; then
        kill "$LMCACHE_PID" 2>/dev/null || true
        wait "$LMCACHE_PID" 2>/dev/null || true
    fi
}

trap cleanup_lmcache_server EXIT

wait_for_lmcache_ready() {
    { set +x; } 2>/dev/null
    local attempts="${LMCACHE_READY_ATTEMPTS:-120}"
    local tail_pid=""

    while [ ! -f "$LMCACHE_LOG" ]; do
        if [[ -n "$LMCACHE_PID" ]] && ! kill -0 "$LMCACHE_PID" 2>/dev/null; then
            echo "LMCache server died before creating log file. Exiting." >&2
            exit 1
        fi
        sleep 1
    done

    tail -f -n +1 "$LMCACHE_LOG" &
    tail_pid=$!

    for ((i = 1; i <= attempts; i++)); do
        if curl --output /dev/null --silent --fail "http://127.0.0.1:${LMCACHE_HTTP_PORT}/healthcheck"; then
            kill "$tail_pid" 2>/dev/null || true
            wait "$tail_pid" 2>/dev/null || true
            return 0
        fi
        if [[ -n "$LMCACHE_PID" ]] && ! kill -0 "$LMCACHE_PID" 2>/dev/null; then
            echo "LMCache server died before becoming healthy. Log follows:" >&2
            kill "$tail_pid" 2>/dev/null || true
            wait "$tail_pid" 2>/dev/null || true
            cat "$LMCACHE_LOG" >&2 || true
            exit 1
        fi
        sleep 1
    done

    echo "Timed out waiting for LMCache server healthcheck. Log follows:" >&2
    kill "$tail_pid" 2>/dev/null || true
    wait "$tail_pid" 2>/dev/null || true
    cat "$LMCACHE_LOG" >&2 || true
    exit 1
}

case "$OFFLOADING" in
    none) ;;
    cpu)
        unset VLLM_USE_SIMPLE_KV_OFFLOAD
        # MI355X nodes have ~2.7 TiB of host DRAM available for offload;
        # reserve 2.5 TB for the offload pool (leaves ~200 GB headroom for
        # worker RSS / page cache / slurm cgroup).
        TOTAL_CPU_DRAM_GB=$((2500 / (8 / TP)))
        # Use vLLM's regular native KV-offload path (OffloadingConnector),
        # NOT the SimpleCPUOffloadConnector. The "native" backend resolves to
        # OffloadingConnector by default; setting VLLM_USE_SIMPLE_KV_OFFLOAD=1
        # would switch it to SimpleCPUOffloadConnector. We intentionally leave
        # that env var UNSET here so the regular OffloadingConnector path is
        # used. The shortcut --kv_offloading_backend native + --kv_offloading_size
        # form constructs the KVTransferConfig at engine startup
        # (vllm/config/vllm.py:662).
        OFFLOAD_ARGS=(
            --kv_offloading_backend native
            --kv_offloading_size "$TOTAL_CPU_DRAM_GB"
            --disable-hybrid-kv-cache-manager
        )
        ;;
    lmcache)
        { set +x; } 2>/dev/null
        unset VLLM_USE_SIMPLE_KV_OFFLOAD

        git clone https://github.com/LMCache/LMCache.git
        cd LMCache
        pip install -r requirements/build.txt 
        CXX=hipcc BUILD_WITH_HIP=1 pip install -e .   --no-build-isolation
        cd ..

        python3 -c "import lmcache.integration.vllm.lmcache_mp_connector" >/dev/null

        # Match the B200 Kimi LMCache setup: keep a 2.5 TB semantic CPU KV
        # pool, but let the external MP server own that pool so vLLM does not
        # split --kv-offloading-size across TP ranks through the integrated
        # LMCache backend.
        TOTAL_CPU_DRAM_GB=$((2500 / (8 / TP)))
        LMCACHE_HOST="${LMCACHE_HOST:-127.0.0.1}"
        LMCACHE_PORT="${LMCACHE_PORT:-5555}"
        LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
        # LMCacheMPConnector concatenates lmcache.mp.host and port into the
        # ZMQ endpoint. Bind the server to a raw host, but pass the connector a
        # ZMQ-style host string.
        LMCACHE_CONNECT_HOST="${LMCACHE_CONNECT_HOST:-tcp://$LMCACHE_HOST}"
        LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-$TOTAL_CPU_DRAM_GB}"
        LMCACHE_L1_INIT_SIZE_GB="${LMCACHE_L1_INIT_SIZE_GB:-20}"
        # LMCache read locks are leases on chunks that lookup has promised
        # vLLM can retrieve. The default 300s TTL is too short for this
        # long-context agentic queue: TP8/conc32 can spend >300s between
        # lookup and retrieve while GPU KV is saturated, which leaves the
        # object present in L1 but no longer readable. Keep the 2.5 TB pool
        # size unchanged and only extend the lookup-to-retrieve lease.
        LMCACHE_L1_READ_TTL_SECONDS="${LMCACHE_L1_READ_TTL_SECONDS:-7200}"
        LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
        LMCACHE_MAX_WORKERS="${LMCACHE_MAX_WORKERS:-$TP}"
        export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
        export LMCACHE_BLOCKING_TIMEOUT_SECS=120

        echo "Starting LMCache MP server..."
        LMCACHE_CMD=(
            lmcache server
            --host "$LMCACHE_HOST"
            --port "$LMCACHE_PORT"
            --http-host "$LMCACHE_HOST"
            --http-port "$LMCACHE_HTTP_PORT"
            --l1-size-gb "$LMCACHE_L1_SIZE_GB"
            --l1-init-size-gb "$LMCACHE_L1_INIT_SIZE_GB"
            --l1-read-ttl-seconds "$LMCACHE_L1_READ_TTL_SECONDS"
            --chunk-size "$LMCACHE_CHUNK_SIZE"
            --max-workers "$LMCACHE_MAX_WORKERS"
            --eviction-policy LRU
        )
        printf '%q ' "${LMCACHE_CMD[@]}" > "$RESULT_DIR/lmcache_command.txt"
        printf '\n' >> "$RESULT_DIR/lmcache_command.txt"
        "${LMCACHE_CMD[@]}" > "$LMCACHE_LOG" 2>&1 &
        LMCACHE_PID=$!
        echo "LMCache server PID: $LMCACHE_PID"
        wait_for_lmcache_ready

        HYBRID_KV_ARGS=(--disable-hybrid-kv-cache-manager)
        PREFIX_CACHE_ARGS=(--enable-prefix-caching)
        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"$LMCACHE_CONNECT_HOST\",\"lmcache.mp.port\":$LMCACHE_PORT}}"
            --disable-hybrid-kv-cache-manager
        )
        ;;
    *) echo "Error: unsupported OFFLOADING value '$OFFLOADING'" >&2; exit 1 ;;
esac

EP_ARGS=()
if [ "$EP_SIZE" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

echo "Starting vllm server..."
export PYTHONNOUSERSITE=1

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size="$TP"
    "${EP_ARGS[@]}"
    --gpu-memory-utilization 0.80
    --kv-cache-dtype fp8 \
    --distributed-executor-backend mp \
    --async-scheduling \
    --trust-remote-code
    --max-model-len "$MAX_MODEL_LEN"
    --moe-backend triton_unfused \
    --tokenizer-mode deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --max-num-seqs "$CONC"
    "${HYBRID_KV_ARGS[@]}"
    "${PREFIX_CACHE_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"

run_agentic_replay_and_write_outputs "$RESULT_DIR"
