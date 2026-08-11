#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-V4-Pro FP8 AgentX replay on one 8xMI325X node. The checkpoint is
# dequantized to FP8 because gfx942 has no native MXFP4 support. The published
# path is pure TP8. TEP8 remains in the requested collection matrix and uses a
# pinned upstream DeepSeek V4 graph fix while its correctness is validated.

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

resolve_trace_source
install_agentic_deps
agentic_pip_install --quiet Pillow fastapi uvicorn

install_vllm_dsv4_graph_fix() {
    # vLLM PR #51430 narrows DeepSeek V4's eager CUDA-graph region and reports
    # clean real-model MTP + expert-parallel evaluation. Keep v0.27.0's ROCm
    # binaries and install only that exact merged Python source revision.
    local graph_commit="1795b1ba7505aca16121c3eb3b319503863e1f9a"
    local graph_src="/tmp/vllm-dsv4-graph-fix"
    rm -rf "$graph_src"
    git clone --depth 1 --branch test/v027-dsv4-graph-fix-20260811 \
        https://github.com/cquil11/vllm.git "$graph_src"
    if [[ "$(git -C "$graph_src" rev-parse HEAD)" != "$graph_commit" ]]; then
        echo "Unexpected vLLM DeepSeek V4 graph-fix revision" >&2
        exit 1
    fi
    VLLM_USE_PRECOMPILED=1 \
    VLLM_ROCM_WHEEL_INDEX=https://wheels.vllm.ai/rocm/0.27.0/rocm723 \
        uv pip install --system --no-deps --no-build-isolation --editable \
        "$graph_src"
    python3 -c 'import vllm; print("vLLM source:", vllm.__file__)'
}

if (( EP_SIZE > 1 )) && [[ "$DP_ATTENTION" != "true" ]]; then
    install_vllm_dsv4_graph_fix
fi

export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"
export VLLM_ENGINE_READY_TIMEOUT_S=10800
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export PYTHONNOUSERSITE=1

SERVER_LOG="$RESULT_DIR/server.log"
ROUTER_LOG="$RESULT_DIR/router.log"
MOONCAKE_MASTER_LOG="$RESULT_DIR/mooncake_master.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
ROUTER_PID=""
MOONCAKE_MASTER_PID=""

# Mooncake does not publish a ROCm wheel. Reuse the established MI300X/MI325X
# recipe convention: build the pinned release once per immutable ROCm/Python
# combination, cache the staged install on shared storage, and verify that the
# loaded transfer engine is actually linked against HIP.
install_mooncake_rocm() {
    local mooncake_tag="v0.3.11.post1"
    local mooncake_src="/tmp/Mooncake-$mooncake_tag"
    local mooncake_stage="/tmp/mooncake-stage-$mooncake_tag"
    local build_jobs cache_root cache_key cache_archive cache_tmp
    local engine_path os_version python_abi rocm_version

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
if agentic_kv_offload_enabled; then
    require_agentic_kv_offload_backend mooncake
    # TOTAL_CPU_DRAM_GB is the generator-capped aggregate node budget.
    # Embedded Mooncake contributes one segment per rank, so divide it here.
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
    # Mooncake v0.3.11.post1 emits its transfer polling loop at VLOG(1).
    # Keep normal INFO diagnostics while suppressing that unbounded hot-loop output.
    export GLOG_v=0
    export MOONCAKE_CONFIG_PATH PYTHONHASHSEED=0 MC_SLICE_SIZE=1048576
    # Match the proven MI355X TCP-store convention. Mooncake 0.3.11's pooled
    # TCP path retained thousands of simultaneous sockets on this workload and
    # timed out 60-second transfers; the unpooled path plus eight workers keeps
    # transfer concurrency bounded by the store workers instead.
    export MC_ENABLE_DEST_DEVICE_AFFINITY=1
    export MC_WORKERS_PER_CTX=8
    unset MC_TCP_ENABLE_CONNECTION_POOL
    MOONCAKE_KV_LEASE_TTL=120s
    # A full-context transfer can starve a rank's client heartbeat beyond the
    # master's 10-second default even while that rank remains healthy. Keep the
    # client registered long enough for the bounded transfer timeout to resolve.
    MOONCAKE_CLIENT_TTL=120
    mooncake_master --port "$MOONCAKE_MASTER_PORT" \
        --default_kv_lease_ttl="$MOONCAKE_KV_LEASE_TTL" \
        --client_ttl="$MOONCAKE_CLIENT_TTL" \
        --eviction_high_watermark_ratio=0.80 \
        --eviction_ratio=0.10 > "$MOONCAKE_MASTER_LOG" 2>&1 &
    MOONCAKE_MASTER_PID=$!
    sleep 2
    kill -0 "$MOONCAKE_MASTER_PID"
    OFFLOAD_ARGS=(
        --kv-transfer-config
        '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true}}'
    )
else
    require_agentic_kv_offload_none
fi

PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
if [[ "$DP_ATTENTION" == "true" ]]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP")
fi

EP_ARGS=()
if (( EP_SIZE > 1 )); then
    EP_ARGS=(--enable-expert-parallel)
fi

SCHEDULING_ARGS=(--async-scheduling)

USE_VLLM_ROUTER=false
VLLM_BACKEND_PORT="$PORT"
if [[ "$DP_ATTENTION" == "true" ]]; then
    if (( EP_SIZE != TP )); then
        echo "ERROR: MI325X DP-attention requires EP_SIZE == TP so FP8 experts remain sharded" >&2
        exit 1
    fi
    USE_VLLM_ROUTER=true
    VLLM_BACKEND_PORT=$((PORT + 1))
    export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=1
    export AIPERF_SERVER_METRICS_URLS="http://localhost:${VLLM_BACKEND_PORT}/metrics"
    agentic_pip_install --quiet 'vllm-router==0.1.14'
fi

if (( EP_SIZE > 1 )) && [[ "$DP_ATTENTION" != "true" ]]; then
    # TEP's correlated low-concurrency trajectories contain a few deterministic
    # metadata-only turns. Do not let three early turns abort a one-hour run
    # before the sample is representative; the unchanged strict 10% post-run
    # validator remains authoritative for the completed result.
    AIPERF_LIVE_FAILED_REQUEST_THRESHOLD=0.50
fi

# The 16K prefill budget and 4*CONC sequence-cap probes were neutral, while
# piecewise graphs regressed. Restore the official 8K/2*CONC/FULL_DECODE_ONLY
# baseline before isolating INT4 Quick Reduce.
MAX_NUM_SEQS=$((2 * CONC))
# The existing MI300X/MI325X fixed-sequence recipes use K=2. Keep that MTP
# depth and its measured golden acceptance length for every AgentX point.
NUM_SPEC_TOKENS=2
SYNTHETIC_ACCEPT_LEN=2.27
if [[ "${EVAL_ONLY:-false}" == "true" ]]; then
    SPEC_CONFIG="{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}}"
else
    SPEC_CONFIG="{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS},\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":${SYNTHETIC_ACCEPT_LEN}}"
fi

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$ROUTER_PID" "vLLM router"
    stop_background_process_tree "${SERVER_PID:-}" "vLLM server" 60
    stop_background_process_tree "$MOONCAKE_MASTER_PID" "Mooncake master"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$VLLM_BACKEND_PORT"
    --trust-remote-code
    "${SCHEDULING_ARGS[@]}"
    --distributed-executor-backend mp
    --quantization deepseek_v4_fp8
    --kv-cache-dtype fp8
    "${PARALLEL_ARGS[@]}"
    "${EP_ARGS[@]}"
    --gpu-memory-utilization 0.9
    --block-size 256
    --max-num-batched-tokens 8192
    --max-num-seqs "$MAX_NUM_SEQS"
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --speculative-config "$SPEC_CONFIG"
    --tokenizer-mode deepseek_v4
    --tool-call-parser deepseek_v4
    --reasoning-parser deepseek_v4
    --enable-auto-tool-choice
    --enable-prefix-caching
    --enable-prompt-tokens-details
    --no-disable-hybrid-kv-cache-manager
    "${OFFLOAD_ARGS[@]}"
)

printf '%q ' "${VLLM_CMD[@]}" > "$RESULT_DIR/vllm_command.txt"
printf '\n' >> "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server_ready --port "$VLLM_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "$USE_VLLM_ROUTER" == "true" ]]; then
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

if [[ "${EVAL_ONLY:-false}" == "true" ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    # Full-context AgentX responses can remain healthy for several minutes
    # after the admission window closes. Let already-admitted requests drain
    # so their observed TTFT/ITL enters the strict coverage calculation. This
    # does not extend admissions, change the workload, or lower the 98% gate.
    REPLAY_CMD+=" --benchmark-grace-period 1800"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
