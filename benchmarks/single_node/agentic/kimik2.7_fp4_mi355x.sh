#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K2.7 FP4 on MI355X using vLLM.
# Offload handling mirrors the validated DSv4 MI355X agentic recipe.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION,
#   EP_SIZE, DP_ATTENTION
#
# KV_OFFLOADING=none            - vLLM GPU KV only.
# KV_OFFLOADING=dram requires KV_OFFLOAD_BACKEND:
#   native  - vLLM OffloadingConnector to CPU DRAM.
#   lmcache - LMCache MP server + vLLM LMCacheMPConnector.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

PORT=${PORT:-8888}
DURATION=${DURATION:-1800}
EP_SIZE=${EP_SIZE:-1}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# ROCR/HIP visibility for vLLM 0.14+
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

# `hf download` creates the target dir if missing and is itself idempotent.
# When MODEL_PATH is unset (stand-alone runs), fall back to the HF_HUB_CACHE
# Either way, MODEL_PATH is what the server is launched with.
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
# Cap the replay corpus at 256k (470 traces, max in+out <= 256k) instead of the
# unfiltered 052726 corpus whose ~1M-token traces get rejected and add no perf
# signal at high concurrency.
#export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_with_subagents_256k
#060226
#export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_with_subagents_060226_256k

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# Agentic cache warmup duration (seconds); matches the validated DSv4 MI355X
# recipe. Overridable via env for fast offload-mode diagnostics.
export AIPERF_AGENTIC_CACHE_WARMUP_DURATION="${AIPERF_AGENTIC_CACHE_WARMUP_DURATION:-600}"

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
export VLLM_ROCM_USE_AITER_MLA=0
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
export VLLM_ROCM_USE_AITER_FP4BMM=0
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
LMCACHE_LOG="$RESULT_DIR/lmcache_server.log"
mkdir -p "$RESULT_DIR"

OFFLOAD_ARGS=()

# ---- Lmcache config ----------------------------------------------------------
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

if agentic_kv_offload_enabled; then
case "${KV_OFFLOAD_BACKEND:-}" in
  native)
    require_agentic_kv_offload_backend native
    # ---- vLLM native config ----------------------------------------------------------
    unset VLLM_USE_SIMPLE_KV_OFFLOAD
    # Partition the aggregate host-DRAM budget across the ranks that share the
    # offload pool (TOTAL_CPU_DRAM_GB comes from the sweep's dram-utilization).
    TOTAL_CPU_DRAM_PARTITION_GB="$((TOTAL_CPU_DRAM_GB / (8 / TP)))"
    # Use vLLM's regular native KV-offload path (OffloadingConnector), NOT the
    # SimpleCPUOffloadConnector. The "native" backend resolves to
    # OffloadingConnector by default; leaving VLLM_USE_SIMPLE_KV_OFFLOAD unset
    # keeps that path. --kv_offloading_backend native + --kv_offloading_size
    # constructs the KVTransferConfig at engine startup.
    # Keep the hybrid kv-cache manager enabled (default) — disabling it lowers
    # the cache hit rate, per the validated DSv4 MI355X recipe.
    OFFLOAD_ARGS=(
        --kv_offloading_backend native
        --kv_offloading_size "$TOTAL_CPU_DRAM_PARTITION_GB"
    )
    ;;
  lmcache)
    require_agentic_kv_offload_backend lmcache
    # ---- LMCache config ----------------------------------------------------------
    { set +x; } 2>/dev/null
    unset VLLM_USE_SIMPLE_KV_OFFLOAD

    # LMCache on ROCm must be built from source with BUILD_WITH_HIP=1 — the
    # PyPI wheel is CUDA-only and silently falls back to a slow copy path
    # (no warm-pass win). Pin a release tag and verify the HIP c_ops import
    # plus the ROCm CuPy build, per the AMD LMCache workshop recipe.
    LMCACHE_VERSION="${LMCACHE_VERSION:-v0.5.0}"
    if ! python3 -c "from lmcache import c_ops" 2>/dev/null; then
        command -v hipcc >/dev/null || { echo "ERROR: hipcc not found — need the ROCm toolchain to build LMCache" >&2; exit 1; }
        pip uninstall -y lmcache >/dev/null 2>&1 || true
        rm -rf LMCache
        git clone --depth 1 --branch "$LMCACHE_VERSION" https://github.com/LMCache/LMCache.git
        cd LMCache
        pip install -r requirements/build.txt
        CXX=hipcc BUILD_WITH_HIP=1 pip install -e . --no-build-isolation
        cd ..
    fi

    # ROCm CuPy (force-reinstall so a half-removed CUDA build can't linger)
    # and re-pin deps the source build / cupy pull can disturb.
    pip uninstall -y cupy cupy-cuda12x cupy-cuda13x nixl nixl-cu12 nixl-cu13 nixl_ep >/dev/null 2>&1 || true
    pip install --force-reinstall --no-cache-dir cupy-rocm-7-0
    pip install -q "numpy==2.1.3" "grpcio==1.78.0" || true

    # Fail loudly if we're on the slow CUDA-wheel fallback (c_ops missing or
    # CuPy not the ROCm build) — otherwise the run silently loses the offload.
    python3 -c "
import lmcache
from lmcache import c_ops
import cupy
from cupy_backends.cuda.api import runtime as r
assert getattr(r, 'is_hip', False), 'CuPy is not the ROCm build (is_hip=False)'
print('lmcache', lmcache.__version__, '| cupy', cupy.__version__, '| is_hip =', r.is_hip)
"
    python3 -c "import lmcache.integration.vllm.lmcache_mp_connector" >/dev/null

    TOTAL_CPU_DRAM_PARTITION_GB="$((TOTAL_CPU_DRAM_GB / (8 / TP)))"
    LMCACHE_HOST="${LMCACHE_HOST:-127.0.0.1}"
    LMCACHE_PORT="${LMCACHE_PORT:-5555}"
    LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
    # LMCacheMPConnector concatenates lmcache.mp.host and port into the ZMQ
    # endpoint. Bind the server to a raw host, pass the connector a ZMQ-style host.
    LMCACHE_CONNECT_HOST="${LMCACHE_CONNECT_HOST:-tcp://$LMCACHE_HOST}"
    LMCACHE_L1_SIZE_GB="${TOTAL_CPU_DRAM_PARTITION_GB}"
    if [ "$LMCACHE_L1_SIZE_GB" -gt "$TOTAL_CPU_DRAM_GB" ]; then
        echo "Error: LMCACHE_L1_SIZE_GB=$LMCACHE_L1_SIZE_GB exceeds configured capacity $TOTAL_CPU_DRAM_GB" >&2
        exit 1
    fi
    LMCACHE_L1_INIT_SIZE_GB="${LMCACHE_L1_INIT_SIZE_GB:-20}"
    # Extend the lookup-to-retrieve lease well past the default 300s: a
    # long-context agentic queue can spend minutes between lookup and retrieve
    # while GPU KV is saturated, which otherwise expires the lease.
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

    OFFLOAD_ARGS=(
        --kv-transfer-config
        "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"$LMCACHE_CONNECT_HOST\",\"lmcache.mp.port\":$LMCACHE_PORT}}"
    )
    ;;
  *)
    echo "Error: unsupported KV_OFFLOAD_BACKEND '${KV_OFFLOAD_BACKEND:-}' (expected: native, lmcache)" >&2
    exit 1
    ;;
esac
fi

# ---- LLM server config ----------------------------------------------------------
EP_ARGS=()
if [ "$EP_SIZE" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

echo "Starting vllm server..."
export PYTHONNOUSERSITE=1

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
export VLLM_ROCM_USE_AITER_MLA=0
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
export VLLM_ROCM_USE_AITER_FP4BMM=0
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4

# AgentX concurrency counts live session trees, not individual requests.
# Subagent fan-out can push instantaneous request concurrency above CONC, so
# leave 2x headroom rather than clipping those bursts at the scheduler.
MAX_NUM_SEQS=$((2 * CONC))

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size="$TP"
    "${EP_ARGS[@]}"
    --gpu-memory-utilization 0.90
    --trust-remote-code
    --async-scheduling
    --distributed-executor-backend mp
    --moe-backend aiter
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'
    --enable-prefix-caching
    --no-disable-hybrid-kv-cache-manager
    --max-num-seqs "$MAX_NUM_SEQS"
    --mm-encoder-tp-mode data
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