#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4.1-Flash Engram on local NVMe. Eager retrieval callbacks
# refresh fixed staging rows on every piecewise CUDA graph replay.
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION \
    EVAL_ONLY ENGRAM_SSD_DIR VLLM_ENGINE_READY_TIMEOUT_S DSV41_MIN_CUDAGRAPH_CAPTURE_SIZE
export GPU_COUNT="$TP"

if [[ -n "${MODEL_PATH:-}" && "$MODEL_PATH" != "$MODEL" ]]; then
    hf download "$MODEL" --local-dir "$MODEL_PATH"
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

nvidia-smi
resolve_trace_source
install_agentic_deps
mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
MOONCAKE_MASTER_LOG="$RESULT_DIR/mooncake_master.log"
MOONCAKE_MASTER_PID=""
export VLLM_ENGINE_READY_TIMEOUT_S
export VLLM_USE_RUST_FRONTEND=1
export VLLM_USE_V2_MODEL_RUNNER=1
# Set before importing vLLM: the eager-break decorator is resolved at import.
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
# SimpleCPUOffloadConnector resolves prefix-cache block hashes when it selects
# blocks to store, and asserts if they have already been retired. The other
# B200 vLLM agentic recipes carry the same retention interval for this reason;
# without it the vllm-simple rows fail in build_connector_meta at warmup.
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=32768
export VLLM_RPC_TIMEOUT=600000
export PYTHONUNBUFFERED=1

# ---- Patch the installed vLLM ------------------------------------------------
# --dry-run first so a drifted image fails here with a clear message rather
# than mid-benchmark, and -N makes a re-run on the same container a no-op.
VLLM_DIR="$(python3 -c 'import os, vllm; print(os.path.dirname(vllm.__file__))')"
ENGRAM_PATCH="$INFERENCEX_REPO_ROOT/benchmarks/patches/vllm-dsv41flash-engram-ssd.patch"
if patch -p1 -N --dry-run -d "$VLLM_DIR" < "$ENGRAM_PATCH" > /dev/null 2>&1; then
    patch -p1 -N -d "$VLLM_DIR" < "$ENGRAM_PATCH"
elif ! patch -p1 -R --dry-run -d "$VLLM_DIR" < "$ENGRAM_PATCH" > /dev/null 2>&1; then
    echo "Engram SSD patch does not apply to $VLLM_DIR: the image has drifted." >&2
    echo "Re-generate benchmarks/patches/vllm-dsv41flash-engram-ssd.patch." >&2
    exit 1
fi
# Exercise the installed patch with changing IDs across actual CUDA replays.
python3 "$INFERENCEX_REPO_ROOT/benchmarks/patches/check_dsv41flash_ssd_replay.py" \
    --result-dir "$RESULT_DIR" 2>&1 | tee "$RESULT_DIR/engram_replay_check.log"

# Require the caller's local-disk mount. Warm page-cache hits can hide a
# network mount, so inspect the filesystem rather than assuming the path.
mkdir -p "$ENGRAM_SSD_DIR"
ENGRAM_FSTYPE="$(df -PT "$ENGRAM_SSD_DIR" | awk 'NR==2{print $2}')"
case "$ENGRAM_FSTYPE" in
    xfs|ext4) ;;
    *)
        echo "ENGRAM_SSD_DIR=$ENGRAM_SSD_DIR is $ENGRAM_FSTYPE, not local disk." >&2
        exit 1
        ;;
esac
df -h "$ENGRAM_SSD_DIR" | tail -1

# Pyxis shares the host network; port 8888 can already belong to a host service.
select_available_server_port
export AIPERF_SERVER_URL="http://localhost:${PORT}"
export AIPERF_SERVER_METRICS_URLS="${AIPERF_SERVER_URL}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"
echo "Using vLLM endpoint ${AIPERF_SERVER_URL}"

# ---- KV offload ---------------------------------------------------------------
# The point of moving Engram to disk is to leave DRAM for KV, so this recipe
# supports the same KV offload backends as the other B200 vLLM agentic recipes
# rather than requiring kv-offloading=none. The two are independent: Engram
# rows are read on the host and copied to the device per step, while a KV
# connector owns its own pinned pool.
#
# The page cache and the KV pool compete for the same DRAM, and that resolves
# itself in the right direction: a connector's pool is pinned and therefore
# unreclaimable, while the Engram mapping is clean page cache, so the kernel
# evicts Engram pages under pressure instead of failing the KV allocation. The
# cost of that eviction is a page fault on the next lookup of an evicted row,
# which must be measured separately from logical gather traffic.
OFFLOAD_ARGS=()
case "$KV_OFFLOAD_BACKEND" in
    "")
        require_agentic_kv_offload_none
        ;;
    vllm-simple)
        require_agentic_kv_offload_backend vllm-simple
        CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / GPU_COUNT ))
        # Identical prefixes must hash to identical block keys across DP ranks.
        export PYTHONHASHSEED=42
        OFFLOAD_CONFIG=$(cat <<EOF
{
  "kv_connector": "SimpleCPUOffloadConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "cpu_bytes_to_use_per_rank": ${CPU_BYTES_PER_RANK},
    "enable_cross_layers_blocks": "true",
    "lazy_offload": false
  }
}
EOF
)
        OFFLOAD_ARGS=(
            --kv-transfer-config
            "$OFFLOAD_CONFIG"
        )
        ;;
    mooncake)
        require_agentic_kv_offload_backend mooncake
        # Embedded mode contributes one segment per GPU rank to a shared
        # distributed store, so pre-divide the aggregate host-memory budget.
        PER_RANK_GB=$((TOTAL_CPU_DRAM_GB / GPU_COUNT))

        MOONCAKE_VERSION=0.3.11.post1
        agentic_pip_install --quiet --no-cache-dir --no-deps \
            --force-reinstall "mooncake-transfer-engine-cuda13==$MOONCAKE_VERSION"
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
  "protocol": "rdma",
  "device_name": "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_10,mlx5_11",
  "enable_offload": false
}
EOF
        export MOONCAKE_CONFIG_PATH
        export MC_ENABLE_DEST_DEVICE_AFFINITY=1
        # Identical prefixes must hash to identical store keys across DP ranks.
        export PYTHONHASHSEED=0
        export WITH_NVIDIA_PEERMEM=0
        export MC_SLICE_SIZE=1048576
        export MC_WORKERS_PER_CTX=4

        # Each rank contributes a separate segment. Evict early enough to
        # avoid an imbalanced rank exhausting its segment.
        MOONCAKE_EVICTION_HIGH_WATERMARK_RATIO=0.80
        MOONCAKE_EVICTION_RATIO=0.10
        # Mooncake's default 5s read lease is shorter than the observed
        # transfer latency for large DSv4 hybrid-KV loads on B200 TCP.
        MOONCAKE_KV_LEASE_TTL=60s

        echo "Starting Mooncake master on port $MOONCAKE_MASTER_PORT..."
        mooncake_master --port "$MOONCAKE_MASTER_PORT" \
            --eviction_high_watermark_ratio="$MOONCAKE_EVICTION_HIGH_WATERMARK_RATIO" \
            --eviction_ratio="$MOONCAKE_EVICTION_RATIO" \
            --default_kv_lease_ttl="$MOONCAKE_KV_LEASE_TTL" \
            > "$MOONCAKE_MASTER_LOG" 2>&1 &
        MOONCAKE_MASTER_PID=$!
        sleep 2
        if ! kill -0 "$MOONCAKE_MASTER_PID" 2>/dev/null; then
            echo "Mooncake master died during startup." >&2
            cat "$MOONCAKE_MASTER_LOG" >&2
            exit 1
        fi
        unset VLLM_USE_SIMPLE_KV_OFFLOAD
        OFFLOAD_ARGS=(
            --kv-transfer-config
            '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true}}'
        )
        ;;
    *)
        echo "Error: unsupported B200 KV_OFFLOAD_BACKEND='$KV_OFFLOAD_BACKEND'" >&2
        exit 1
        ;;
esac

NUM_SPEC_TOKENS=5
check_env_vars DSV41_MIN_CUDAGRAPH_CAPTURE_SIZE
CAPTURE_SIZE="$DSV41_MIN_CUDAGRAPH_CAPTURE_SIZE"
while (( CAPTURE_SIZE < CONC * (1 + NUM_SPEC_TOKENS) && CAPTURE_SIZE < 2048 )); do
    CAPTURE_SIZE=$((CAPTURE_SIZE * 2))
done

# Adaptive verification forces FULL graphs in this image; real block
# rejection remains enabled for eval, using the supported PIECEWISE path.
if [[ "$EVAL_ONLY" == true ]]; then
    SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}'
else
    SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","synthetic_acceptance_length":3.51,"enable_adaptive_verification":false}'
fi

# Host row retrieval executes between graph pieces on each live step.
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0 --port "$PORT" --tensor-parallel-size "$TP"
    --language-model-only
    --tokenizer-mode deepseek_v41
    --tool-call-parser deepseek_v41 --enable-auto-tool-choice
    --reasoning-parser deepseek_v41
    --engram-config "{\"cpu_offload\":true,\"disk_offload_dir\":\"$ENGRAM_SSD_DIR\"}"
    "${OFFLOAD_ARGS[@]}"
    --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
    --speculative-config "$SPEC_CONFIG"
    --max-model-len 1048576
    --max-cudagraph-capture-size "$CAPTURE_SIZE"
    --disable-uvicorn-access-log
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
cleanup_offload_services() {
    local rc=$?
    trap - EXIT
    if [[ -n "$MOONCAKE_MASTER_PID" ]] && kill -0 "$MOONCAKE_MASTER_PID" 2>/dev/null; then
        kill "$MOONCAKE_MASTER_PID" 2>/dev/null || true
    fi
    exit "$rc"
}
trap cleanup_offload_services EXIT

"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# One shard per rank per Engram layer. Four files where there should be eight
# means two layers collided on one path and are being served from one table,
# which no throughput number would reveal.
EXPECTED_SHARDS=$((TP * 2))
FOUND_SHARDS="$(find "$ENGRAM_SSD_DIR" -name 'engram_*_v*_r*.weight.bin' | wc -l)"
du -sh "$ENGRAM_SSD_DIR"
if [[ "$FOUND_SHARDS" -ne "$EXPECTED_SHARDS" ]]; then
    echo "Expected $EXPECTED_SHARDS Engram shards, found $FOUND_SHARDS." >&2
    exit 1
fi

if [[ "$EVAL_ONLY" == true ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
