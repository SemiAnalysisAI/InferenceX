#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4.1-Flash with the Engram n-gram tables served from local NVMe
# instead of pinned host RAM.
#
# The table is 23.60 GiB per rank per Engram layer and the model has two, so
# TP4 holds ~189 GiB in host memory. It is a pure gather -- one row per head
# per layer per token -- so the working set is tiny next to the table, which
# makes it a candidate for file-backed paging. This maps each shard from
# $ENGRAM_SSD_DIR and gathers rows on the host, leaving ~257 GiB of host RAM
# free at no cost to decode throughput.
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
require_agentic_kv_offload_none
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
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
export VLLM_USE_RUST_FRONTEND=1
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

# Node-local NVMe. A network mount would make every row gather a round trip,
# so fail loudly rather than silently benchmarking the filesystem.
ENGRAM_SSD_DIR="${ENGRAM_SSD_DIR:-/raid/engram}"
mkdir -p "$ENGRAM_SSD_DIR"
ENGRAM_FSTYPE="$(df -PT "$ENGRAM_SSD_DIR" | awk 'NR==2{print $2}')"
case "$ENGRAM_FSTYPE" in
    nfs|nfs4|cifs|tmpfs|ramfs)
        echo "ENGRAM_SSD_DIR=$ENGRAM_SSD_DIR is $ENGRAM_FSTYPE, not local disk." >&2
        exit 1
        ;;
esac
df -h "$ENGRAM_SSD_DIR" | tail -1

NUM_SPEC_TOKENS=5
CAPTURE_SIZE=1
while (( CAPTURE_SIZE < CONC * (1 + NUM_SPEC_TOKENS) && CAPTURE_SIZE < 2048 )); do
    CAPTURE_SIZE=$((CAPTURE_SIZE * 2))
done

select_available_server_port
export AIPERF_SERVER_URL="http://localhost:${PORT}"
export AIPERF_SERVER_METRICS_URLS="${AIPERF_SERVER_URL}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"
echo "Using vLLM endpoint ${AIPERF_SERVER_URL}"

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
    --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
    --speculative-config "$SPEC_CONFIG"
    --max-model-len 1048576
    --max-cudagraph-capture-size "$CAPTURE_SIZE"
    --disable-uvicorn-access-log
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# One shard per rank per Engram layer. Four files where there should be eight
# means two layers collided on one path and are being served from one table,
# which no throughput number would reveal.
EXPECTED_SHARDS=$((TP * 2))
FOUND_SHARDS="$(find "$ENGRAM_SSD_DIR" -name 'engram_v*_r*.weight.bin' | wc -l)"
du -sh "$ENGRAM_SSD_DIR"
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
