#!/usr/bin/env bash
set -euo pipefail
set -x

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL IMAGE TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
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

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
resolve_trace_source
install_agentic_deps

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export PYTHONNOUSERSITE=1

SERVER_LOG="$RESULT_DIR/server.log"
ROUTER_LOG="$RESULT_DIR/router.log"
MOONCAKE_MASTER_LOG="$RESULT_DIR/mooncake_master.log"
mkdir -p "$RESULT_DIR"

install_mooncake_rocm() {
    local mooncake_tag="v0.3.11.post1"
    local mooncake_src="/tmp/Mooncake-$mooncake_tag"
    local mooncake_stage="/tmp/mooncake-stage-$mooncake_tag"
    local build_jobs
    local cache_root
    local cache_key
    local cache_archive
    local cache_tmp
    local engine_path
    local os_version
    local python_abi
    local rocm_version

    build_jobs=$(nproc)
    if ((build_jobs > 32)); then
        build_jobs=32
    fi

    os_version=$(. /etc/os-release && printf '%s-%s' "$ID" "$VERSION_ID")
    python_abi=$(python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
    rocm_version=$(sed -n '1p' /opt/rocm/.info/version 2>/dev/null || true)
    if [[ -z "$rocm_version" ]]; then
        rocm_version=$(hipconfig --version)
    fi
    rocm_version=${rocm_version//[^[:alnum:]._-]/_}
    cache_root="${HF_HUB_CACHE:?HF_HUB_CACHE must be set}/inferencex/mooncake"
    cache_key="${mooncake_tag}-${os_version}-${python_abi}-${rocm_version}-$(uname -m)-hip"
    cache_archive="$cache_root/$cache_key.tar.gz"
    mkdir -p "$cache_root"

    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake git libasio-dev libboost-dev libcurl4-openssl-dev \
        libgflags-dev libgoogle-glog-dev libibverbs-dev libjsoncpp-dev \
        libnuma-dev libpython3-dev libssl-dev libunwind-dev liburing-dev \
        libxxhash-dev libyaml-cpp-dev libzstd-dev ninja-build pybind11-dev

    exec 9>"$cache_archive.lock"
    flock -w 1800 9
    if [[ -f "$cache_archive" ]] && ! tar -tzf "$cache_archive" >/dev/null 2>&1; then
        rm -f "$cache_archive"
    fi
    if [[ ! -f "$cache_archive" ]]; then
        echo "Building HIP Mooncake cache artifact: $cache_archive"
        rm -rf "$mooncake_src" "$mooncake_stage"
        git clone --depth 1 --branch "$mooncake_tag" --recurse-submodules \
            --shallow-submodules https://github.com/kvcache-ai/Mooncake.git "$mooncake_src"
        cmake -S "$mooncake_src/extern/yalantinglibs" \
            -B "$mooncake_src/extern/yalantinglibs/build" \
            -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF
        cmake --build "$mooncake_src/extern/yalantinglibs/build" -j "$build_jobs"
        cmake --install "$mooncake_src/extern/yalantinglibs/build"
        cmake -S "$mooncake_src" -B "$mooncake_src/build" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF -DUSE_HIP=ON \
            -DWITH_EP=OFF -DWITH_STORE=ON -DWITH_STORE_RUST=OFF \
            -DWITH_RUST_EXAMPLE=OFF -DBUILD_EXAMPLES=OFF -DBUILD_UNIT_TESTS=OFF
        cmake --build "$mooncake_src/build" -j "$build_jobs"
        mkdir -p "$mooncake_stage"
        DESTDIR="$mooncake_stage" cmake --install "$mooncake_src/build"
        cache_tmp=$(mktemp "$cache_root/$cache_key.tmp.XXXXXX")
        tar -C "$mooncake_stage" -czf "$cache_tmp" .
        mv -f "$cache_tmp" "$cache_archive"
    else
        echo "Using HIP Mooncake cache artifact: $cache_archive"
    fi
    tar -C / -xzf "$cache_archive"
    engine_path=$(python3 -c 'import mooncake.engine; print(mooncake.engine.__file__)')
    ldd "$engine_path" | grep -q 'libamdhip64.so'
    exec 9>&-
}

OFFLOAD_ARGS=()
if require_agentic_kv_offload_backend mooncake; then
        PER_RANK_GB=$((TOTAL_CPU_DRAM_GB / TP))
        if ! python3 -c "from mooncake.store import MooncakeDistributedStore" >/dev/null 2>&1; then
            install_mooncake_rocm
        fi
        python3 -c "from mooncake.store import MooncakeDistributedStore" >/dev/null
        MOONCAKE_MASTER_PORT=$((PORT + 12000))
        MOONCAKE_CONFIG_PATH="$RESULT_DIR/mooncake_config.json"
        cat > "$MOONCAKE_CONFIG_PATH" <<EOF
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:$MOONCAKE_MASTER_PORT",
  "global_segment_size": "${PER_RANK_GB}GB",
  "local_buffer_size": "4GB",
  "protocol": "tcp",
  "device_name": "",
  "enable_offload": false
}
EOF
        export MOONCAKE_CONFIG_PATH PYTHONHASHSEED=0 MC_SLICE_SIZE=1048576 MC_WORKERS_PER_CTX=4
        export MC_TCP_ENABLE_CONNECTION_POOL=1
        mooncake_master --port "$MOONCAKE_MASTER_PORT" \
            --default_kv_lease_ttl=120s \
            --eviction_high_watermark_ratio=0.80 \
            --eviction_ratio=0.10 > "$MOONCAKE_MASTER_LOG" 2>&1 &
        MOONCAKE_MASTER_PID=$!
        sleep 2
        kill -0 "$MOONCAKE_MASTER_PID"
        OFFLOAD_ARGS=(
            --kv-transfer-config
            '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true}}'
        )
fi

PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
if [[ "$DP_ATTENTION" == "true" ]]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP")
fi

EP_ARGS=()
if (( EP_SIZE > 1 )); then
    EP_ARGS=(--enable-expert-parallel)
fi

KV_CACHE_ARGS=()
if [[ "$IMAGE" == vllm/vllm-openai-rocm:nightly-* ]]; then
    KV_CACHE_ARGS=(--kv-cache-dtype fp8)
fi

VLLM_BACKEND_PORT="$PORT"
if [[ "$DP_ATTENTION" == "true" ]]; then
    VLLM_BACKEND_PORT=$((PORT + 1))
    export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=1
    agentic_pip_install --quiet 'vllm-router==0.1.14'
fi

MAX_NUM_SEQS=$((2 * CONC))
GPU_MEMORY_UTILIZATION=0.95
if (( TP == 4 )); then
    GPU_MEMORY_UTILIZATION=0.98
fi

vllm serve "$MODEL_PATH" --served-model-name "$MODEL" \
    --host 0.0.0.0 \
    --port "$VLLM_BACKEND_PORT" \
    "${PARALLEL_ARGS[@]}" \
    "${EP_ARGS[@]}" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --block-size 128 \
    --language-model-only \
    --attention-backend TRITON_ATTN \
    --enable-prefix-caching \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --tool-call-parser minimax_m3 \
    --reasoning-parser minimax_m3 \
    --enable-auto-tool-choice \
    --trust-remote-code \
    "${KV_CACHE_ARGS[@]}" \
    "${OFFLOAD_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server_ready --port "$VLLM_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "$DP_ATTENTION" == "true" ]]; then
    vllm-router \
        --worker-urls "http://localhost:$VLLM_BACKEND_PORT" \
        --policy consistent_hash \
        --intra-node-data-parallel-size "$TP" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --prometheus-host 127.0.0.1 \
        --prometheus-port "$((PORT + 10000))" \
        --request-timeout-secs 14400 \
        --disable-retries > "$ROUTER_LOG" 2>&1 &
    ROUTER_PID=$!
    wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"
fi

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
