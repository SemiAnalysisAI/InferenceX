#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K2.7 FP4 on MI355X using vLLM.
#   KV_OFFLOADING=none                              -> GPU KV only
#   KV_OFFLOADING=dram KV_OFFLOAD_BACKEND=lmcache   -> LMCache MP server + connector
#   KV_OFFLOADING=dram KV_OFFLOAD_BACKEND=mooncake  -> Mooncake embedded store + connector
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION, EP_SIZE


source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# ROCR/HIP visibility for vLLM 0.14+
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi
rocm-smi || true
amd-smi || true

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# Install amd-quark for MXFP4 (manual install due to ROCm vLLM bug)
pip install amd-quark

# Disable AITER RMSNorm for TP < 8 due to accuracy issues
if [ "${TP}" -lt 8 ]; then
  export VLLM_ROCM_USE_AITER_RMSNORM=0
fi
# Workaround for MEC FW <177 RCCL memory reclaim issue
version=$(rocm-smi --showfw 2>/dev/null | grep MEC | head -n 1 | awk '{print $NF}')
if [[ "$version" == "" || ${version:-0} -lt 177 ]]; then
    export HSA_NO_SCRATCH_RECLAIM=1
fi

export VLLM_ROCM_USE_AITER=1
# This model has 64 KV heads. AMD flags AITER-MLA as officially supporting only
# 16/128 heads, but empirically it runs this model and (forced via --block-size=1
# below) is the ONLY ROCm MLA backend that accepts block_size=1 and is ~3x faster
# than TRITON_MLA on the agentic trace (best-perf config, run 29337493492).
# Deliberately keeping AITER-MLA ON for throughput; revisit if accuracy validation
# shows the 64-head path is numerically wrong (then AITER_MLA=0 + drop block-size).
export VLLM_ROCM_USE_AITER_MLA=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
# Avoid intermittent symm_mem all-reduce rendezvous hang at engine init on
# MI35x nodes (see KIMIK27_CONC64_LMCACHE_RUNBOOK error #4).
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
LMCACHE_LOG="$RESULT_DIR/lmcache_server.log"
MOONCAKE_MASTER_LOG="$RESULT_DIR/mooncake_master.log"
mkdir -p "$RESULT_DIR"

OFFLOAD_ARGS=()
PREFIX_CACHE_ARGS=()

# ---- Offload service cleanup ------------------------------------------------
LMCACHE_PID=""
MOONCAKE_MASTER_PID=""
ROUTER_PID=""

cleanup_offload_services() {
    if [[ -n "$ROUTER_PID" ]] && kill -0 "$ROUTER_PID" 2>/dev/null; then
        kill "$ROUTER_PID" 2>/dev/null || true
        wait "$ROUTER_PID" 2>/dev/null || true
    fi
    if [[ -n "$LMCACHE_PID" ]] && kill -0 "$LMCACHE_PID" 2>/dev/null; then
        kill "$LMCACHE_PID" 2>/dev/null || true
        wait "$LMCACHE_PID" 2>/dev/null || true
    fi
    if [[ -n "$MOONCAKE_MASTER_PID" ]] && kill -0 "$MOONCAKE_MASTER_PID" 2>/dev/null; then
        kill "$MOONCAKE_MASTER_PID" 2>/dev/null || true
        wait "$MOONCAKE_MASTER_PID" 2>/dev/null || true
    fi
}
trap cleanup_offload_services EXIT

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

if [[ "$KV_OFFLOADING" == "none" ]]; then
    OFFLOAD_MODE="none"
else
    OFFLOAD_MODE="${KV_OFFLOAD_BACKEND:?KV_OFFLOAD_BACKEND required when KV_OFFLOADING=dram}"
fi

case "$OFFLOAD_MODE" in
    none)
        # GPU-only KV baseline: keep on-GPU prefix caching ON (no DRAM offload)
        # so it's apples-to-apples vs the lmcache cell.
        PREFIX_CACHE_ARGS=(--enable-prefix-caching)
        ;;
    lmcache)
        unset VLLM_USE_SIMPLE_KV_OFFLOAD

        # Build LMCache against ROCm if the connector isn't importable. Clone to
        # a container-local dir (NOT bind-mounted /workspace) so the next job's
        # checkout `clean: true` won't trip over root-owned build artifacts.
        if ! python3 -c "import lmcache.integration.vllm.lmcache_mp_connector" >/dev/null 2>&1; then
            LMCACHE_SRC_DIR="${LMCACHE_SRC_DIR:-/opt/lmcache-src}"
            LMCACHE_GIT_REF="${LMCACHE_GIT_REF:-aaf7c0d3}"
            rm -rf "$LMCACHE_SRC_DIR"
            git clone https://github.com/LMCache/LMCache.git "$LMCACHE_SRC_DIR"
            ( cd "$LMCACHE_SRC_DIR"
              git checkout "$LMCACHE_GIT_REF"
              pip install -r requirements/build.txt
              CXX=hipcc BUILD_WITH_HIP=1 pip install -e . --no-build-isolation )
            python3 -c "import lmcache.integration.vllm.lmcache_mp_connector" >/dev/null
        fi

        LMCACHE_HOST="${LMCACHE_HOST:-127.0.0.1}"
        LMCACHE_PORT="${LMCACHE_PORT:-5555}"
        LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
        LMCACHE_CONNECT_HOST="${LMCACHE_CONNECT_HOST:-tcp://$LMCACHE_HOST}"
        # LMCache L1 is SHM-backed: if L1 > /dev/shm free it silently disables
        # SHM and falls back to the pickle path (crashes at load). Cap L1 to 90%
        # of /dev/shm free so SHM stays enabled.
        SHM_FREE_GB=$(df -BG --output=avail /dev/shm 2>/dev/null | tail -1 | tr -dc '0-9')
        SHM_CAP_GB=$(( SHM_FREE_GB * 90 / 100 ))
        LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-$TOTAL_CPU_DRAM_GB}"
        if [ -n "$SHM_CAP_GB" ] && [ "$SHM_CAP_GB" -gt 0 ] && [ "$LMCACHE_L1_SIZE_GB" -gt "$SHM_CAP_GB" ]; then
            echo "Capping LMCACHE_L1_SIZE_GB $LMCACHE_L1_SIZE_GB -> $SHM_CAP_GB to fit /dev/shm (${SHM_FREE_GB}G free)"
            LMCACHE_L1_SIZE_GB="$SHM_CAP_GB"
        fi
        LMCACHE_L1_INIT_SIZE_GB="${LMCACHE_L1_INIT_SIZE_GB:-20}"
        LMCACHE_L1_READ_TTL_SECONDS="${LMCACHE_L1_READ_TTL_SECONDS:-7200}"
        LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
        LMCACHE_MAX_WORKERS="${LMCACHE_MAX_WORKERS:-$((TP * 2))}"
        export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
        export LMCACHE_BLOCKING_TIMEOUT_SECS=60

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

        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"$LMCACHE_CONNECT_HOST\",\"lmcache.mp.port\":$LMCACHE_PORT}}"
        )
        ;;
    mooncake)
        unset VLLM_USE_SIMPLE_KV_OFFLOAD

        # Mooncake embedded mode contributes one global segment per GPU rank to
        # a shared distributed store, so pre-divide the aggregate host-memory
        # budget across TP ranks. Mirrors the MI355X DSv4 vLLM recipe.
        MOONCAKE_PER_RANK_GB=$((TOTAL_CPU_DRAM_GB / TP))

        # No prebuilt ROCm wheel: build the Mooncake transfer engine from source
        # if the store module isn't importable. Clone to a container-local dir
        # (NOT bind-mounted /workspace) so the next job's checkout `clean: true`
        # won't trip over root-owned build artifacts.
        #
        # A bare `make install` only lays down the C++ libs, so the launcher can
        # import `mooncake` but the vLLM worker SUBPROCESSES cannot (fresh
        # site-packages) -> "Please install mooncake ..." at KV-cache init. Build
        # the wheel via Mooncake's own scripts/build_wheel.sh (auditwheel bundles
        # every .so into a self-contained package) and pip install it so the
        # `mooncake` package lands in site-packages and is importable everywhere.
        if ! python3 -c "from mooncake.store import MooncakeDistributedStore" >/dev/null 2>&1; then
            MOONCAKE_SRC_DIR="${MOONCAKE_SRC_DIR:-/opt/mooncake-src}"
            MOONCAKE_GIT_REF="${MOONCAKE_GIT_REF:-main}"
            MOONCAKE_PYVER="$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
            pip install --quiet build auditwheel patchelf
            rm -rf "$MOONCAKE_SRC_DIR"
            git clone https://github.com/kvcache-ai/Mooncake.git "$MOONCAKE_SRC_DIR"
            ( cd "$MOONCAKE_SRC_DIR"
              git checkout "$MOONCAKE_GIT_REF"
              bash dependencies.sh
              mkdir -p build && cd build
              cmake .. -DWITH_STORE=ON
              make -j
              make install
              cd ..
              # Produces a self-contained wheel under mooncake-wheel/dist/.
              bash scripts/build_wheel.sh "$MOONCAKE_PYVER"
              pip install --force-reinstall mooncake-wheel/dist/*.whl

              # CMake installs the Python package into the user site, which
              # shadows the wheel installed by pip. Ensure the CLI wrapper can
              # find the native master binary in the package it imports.
              MOONCAKE_PACKAGE_DIR=$(python3 -c 'import pathlib, mooncake; print(pathlib.Path(mooncake.__file__).parent)')
              install -m 0755 build/mooncake-store/src/mooncake_master \
                  "$MOONCAKE_PACKAGE_DIR/mooncake_master" )
            python3 -c "from mooncake.store import MooncakeDistributedStore" >/dev/null
            test -x "$(python3 -c 'import pathlib, mooncake; print(pathlib.Path(mooncake.__file__).parent / "mooncake_master")')"
        fi

        MOONCAKE_MASTER_PORT=$((PORT + 12000))
        MOONCAKE_CONFIG_PATH="$RESULT_DIR/mooncake_config.json"
        cat > "$MOONCAKE_CONFIG_PATH" <<EOF
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:$MOONCAKE_MASTER_PORT",
  "global_segment_size": "${MOONCAKE_PER_RANK_GB}GB",
  "local_buffer_size": "2GB",
  "protocol": "tcp",
  "device_name": "",
  "enable_offload": false
}
EOF
        export MOONCAKE_CONFIG_PATH
        export MC_ENABLE_DEST_DEVICE_AFFINITY=1
        export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
        export MC_SLICE_SIZE=1048576
        export MC_WORKERS_PER_CTX=8

        echo "Starting Mooncake master on port $MOONCAKE_MASTER_PORT..."
        mooncake_master --port "$MOONCAKE_MASTER_PORT" \
            --eviction_high_watermark_ratio=0.80 \
            --eviction_ratio=0.10 \
            --default_kv_lease_ttl=60s \
            > "$MOONCAKE_MASTER_LOG" 2>&1 &
        MOONCAKE_MASTER_PID=$!
        echo "Mooncake master PID: $MOONCAKE_MASTER_PID"
        sleep 10
        if ! kill -0 "$MOONCAKE_MASTER_PID" 2>/dev/null; then
            echo "Mooncake master died during startup. Log follows:" >&2
            cat "$MOONCAKE_MASTER_LOG" >&2 || true
            exit 1
        fi

        PREFIX_CACHE_ARGS=(--enable-prefix-caching)
        OFFLOAD_ARGS=(
            --kv-transfer-config
            '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true}}'
        )
        ;;
    *)
        echo "Error: unsupported KV_OFFLOAD_BACKEND '$OFFLOAD_MODE' (expected: lmcache, mooncake)" >&2
        exit 1
        ;;
esac

EP_ARGS=()
if [ "$EP_SIZE" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

# Parallelism layout:
#   DP_ATTENTION=false -> pure TP: attention TP-sharded across all $TP GPUs in a
#       single engine (low TPOT, capacity-capped at high CONC).
#   DP_ATTENTION=true  -> DEP: per-DP-rank attention (TP1 x DP=$TP) with experts
#       EP-sharded across the DP ranks (requires EP_SIZE>1). Grows KV capacity +
#       decode width with concurrency. Mirrors dsv4_fp4_b300_vllm.sh.
PARALLEL_ARGS=(--tensor-parallel-size="$TP")
USE_VLLM_ROUTER=false
VLLM_BACKEND_PORT="$PORT"
if [ "$DP_ATTENTION" = "true" ]; then
    PARALLEL_ARGS=(--tensor-parallel-size=1 --data-parallel-size="$TP")
    USE_VLLM_ROUTER=true
    VLLM_BACKEND_PORT=$((PORT + 1))
    VLLM_ROUTER_METRICS_PORT=$((PORT + 10000))
    # vllm-router expands the one HTTP backend into one logical worker per DP
    # rank and sends X-data-parallel-rank per request; consistent_hash pins each
    # conversation to a rank so its prefix cache stays warm. aiperf's stable
    # X-Correlation-ID is aliased to the router's X-Session-ID for that affinity.
    export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=1
    pip install --quiet "vllm-router==0.1.14"
fi

echo "Starting vllm server..."
export PYTHONNOUSERSITE=1
DEFAULT_VLLM_SPEC_CONFIG=$(cat <<'EOF'
{
  "method": "eagle3",
  "model": "lightseekorg/kimi-k2.6-eagle3.1-mla",
  "num_speculative_tokens": 4,
  "rejection_sample_method": "synthetic",
  "synthetic_acceptance_length": 3.24
}
EOF
)
export VLLM_SPEC_CONFIG="${VLLM_SPEC_CONFIG:-$DEFAULT_VLLM_SPEC_CONFIG}"

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$VLLM_BACKEND_PORT"
    "${PARALLEL_ARGS[@]}"
    "${EP_ARGS[@]}"
    --gpu-memory-utilization 0.90
    --trust-remote-code
    --mm-encoder-tp-mode data
    --speculative-config "$VLLM_SPEC_CONFIG"
    "${PREFIX_CACHE_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$VLLM_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# In DEP mode, front the DP engine with vllm-router on $PORT so the agentic
# client (which targets $PORT) load-balances across DP ranks with prefix
# affinity. Pure-TP serves the client directly on $PORT.
if [ "$USE_VLLM_ROUTER" = "true" ]; then
    ROUTER_LOG="$RESULT_DIR/router.log"
    echo "Starting vLLM router on port $PORT for $TP DP ranks..."
    vllm-router \
        --worker-urls "http://localhost:$VLLM_BACKEND_PORT" \
        --policy consistent_hash \
        --intra-node-data-parallel-size "$TP" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --prometheus-host 127.0.0.1 \
        --prometheus-port "$VLLM_ROUTER_METRICS_PORT" \
        --request-timeout-secs 14400 \
        --disable-retries > "$ROUTER_LOG" 2>&1 &
    ROUTER_PID=$!
    echo "Router PID: $ROUTER_PID"
    wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"
fi

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"

run_agentic_replay_and_write_outputs "$RESULT_DIR"
