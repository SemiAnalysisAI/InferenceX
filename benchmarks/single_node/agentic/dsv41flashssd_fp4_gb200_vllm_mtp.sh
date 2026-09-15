#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4.1-Flash with the Engram n-gram tables served from disk instead
# of pinned host RAM, on GB200 (aarch64).
#
# This variant exists to separate two questions the B200 recipe conflates:
# whether the table can live on disk at all, and whether one copy of it can be
# shared by many servers. It therefore takes the table directory as an input
# and accepts either
#
#   ENGRAM_SSD_SHARED=0 (default)  node-local storage, one table per node
#   ENGRAM_SSD_SHARED=1            a shared filesystem, one table for all nodes
#
# The guard below still refuses a shared mount unless that flag is set, so the
# default behaviour matches the B200 recipe and a network filesystem cannot be
# benchmarked by accident.
#
# GB200 nodes on the watchtower cluster have no /raid. Node-local storage is
# /mnt/numa0 and /mnt/numa1 (14 T XFS on md0/md1, one per NUMA node); the only
# storage every node sees is Lustre. Both array roots are root-owned, and
# /mnt/numa1/models is the one path a benchmark user can create under, hence
# the default below -- change it if the cluster grows a proper scratch dir.
#
# The table is 23.60 GiB per rank per Engram layer and the model has two, so
# TP4 holds ~189 GiB in host memory. It is a pure gather -- one row per head
# per layer per token -- so the working set is tiny next to the table, which
# makes it a candidate for file-backed paging. This maps each shard from
# $ENGRAM_SSD_DIR and gathers rows on the host, leaving ~257 GiB of host RAM
# free at no cost to decode throughput.
#
# The row gather runs on a worker while the decoder layers execute. The forward
# thread issues the id copy on the stream that produced the ids and records an
# event; the worker waits on it and then does numpy and filesystem work only,
# touching no CUDA, which is what keeps it clear of cudagraph capture.
#
# Against an otherwise identical inline build at 8k1k concurrency 16: mean TTFT
# 697 ms against 985 ms, P99 TTFT 4.1s against 6.5s, with throughput and TPOT
# unchanged. The gain is the prefill tail, where a step gathers thousands of
# rows and has decoder compute to hide them behind.
#
# Measured against dsv41flash-fp4-b200-vllm-agentic-dspark on B200 TP4 at
# 8k1k, concurrency 16, CUDA graphs, three runs per arm:
#   disk     14,168 tok/s mean (1.4% spread), 95 GB host RAM
#   baseline 13,632 tok/s mean (12.4% spread), 352 GB host RAM
# Decode is slightly better (median TPOT 8.45-8.60 ms vs 8.57-9.78 ms): UVA
# issues ~16k scattered 264-byte PCIe reads per layer, while this gathers in
# DRAM and page cache and then does one contiguous H2D. Prefill's tail is
# worse (P99 TTFT ~6.0s vs ~3.9s) from cold-page first touches.
#
# Requires the vLLM patch in benchmarks/patches (vllm-project/vllm#56512 plus
# the disk tier); drop this recipe once that lands upstream.
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION
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
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
export VLLM_USE_RUST_FRONTEND=1
export VLLM_USE_V2_MODEL_RUNNER=1
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
python3 -c "from vllm.config.engram import EngramConfig; assert 'disk_offload_dir' in EngramConfig.__dataclass_fields__"

# Where the table lives. Default is node-local; a shared filesystem has to be
# asked for explicitly, because every row gather on one is a network round trip
# and benchmarking that by accident is the failure this guard exists to prevent.
ENGRAM_SSD_DIR="${ENGRAM_SSD_DIR:-/mnt/numa1/models/engram}"
ENGRAM_SSD_SHARED="${ENGRAM_SSD_SHARED:-0}"
mkdir -p "$ENGRAM_SSD_DIR"
require_engram_table_placement "$ENGRAM_SSD_DIR" "$ENGRAM_SSD_SHARED"
ENGRAM_FSTYPE="$(df -PT "$ENGRAM_SSD_DIR" | awk 'NR==2{print $2}')"
df -h "$ENGRAM_SSD_DIR" | tail -1
# Record the placement next to the results so a row can never be misattributed.
printf 'engram_ssd_dir=%s\nengram_fstype=%s\nengram_ssd_shared=%s\n' \
    "$ENGRAM_SSD_DIR" "$ENGRAM_FSTYPE" "$ENGRAM_SSD_SHARED" \
    | tee "$RESULT_DIR/engram_placement.txt"

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
# measured at about 70 microseconds per 4 KiB read on this array.
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
CAPTURE_SIZE=1
while (( CAPTURE_SIZE < CONC * (1 + NUM_SPEC_TOKENS) && CAPTURE_SIZE < 2048 )); do
    CAPTURE_SIZE=$((CAPTURE_SIZE * 2))
done

if [[ "${EVAL_ONLY:-false}" == true ]]; then
    SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}'
else
    SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","synthetic_acceptance_length":3.51,"enable_adaptive_verification":false}'
fi

# Piecewise capture, not the default. The row gather is host work -- a
# device-to-host copy of the ids and a read from a mapped file -- which is
# illegal while a stream is capturing, so the lookup skips it during capture
# and relies on this forward still executing in Python on every live step.
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
FOUND_SHARDS="$(find "$ENGRAM_SSD_DIR" -name 'engram_v*_r*.weight.bin' | wc -l)"
du -sh "$ENGRAM_SSD_DIR"
# A shared table is expected to be reused, not rebuilt. Count the reuse lines so
# a run that silently rebuilt 189 GiB is distinguishable from one that mapped it.
ENGRAM_REUSED="$(grep -ac 'reusing finished' "$SERVER_LOG" || true)"
echo "engram_shards_reused=$ENGRAM_REUSED of $EXPECTED_SHARDS" \
    | tee -a "$RESULT_DIR/engram_placement.txt"
if [[ "$FOUND_SHARDS" -ne "$EXPECTED_SHARDS" ]]; then
    echo "Expected $EXPECTED_SHARDS Engram shards, found $FOUND_SHARDS." >&2
    exit 1
fi

if [[ "${EVAL_ONLY:-false}" == true ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
