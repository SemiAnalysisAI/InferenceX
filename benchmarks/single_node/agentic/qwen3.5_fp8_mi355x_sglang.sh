#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Qwen3.5 FP8 on MI355X using SGLang.
#
# Required env vars:
#   MODEL, TP, CONC, OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR
#
# OFFLOADING values:
#   none    - SGLang GPU KV only with radix cache disabled.
#   hicache - SGLang HiCache with local CPU hierarchical cache.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR

PORT=${PORT:-8888}
DURATION=${DURATION:-1800}
MAX_DELAY=${MAX_DELAY:-60}
ADVANCE_MIN=${ADVANCE_MIN:-0.0}
ADVANCE_MAX=${ADVANCE_MAX:-0.7}
EP_SIZE=${EP_SIZE:-1}
SCHEDULER_RECV_INTERVAL=${SCHEDULER_RECV_INTERVAL:-30}
if [ -z "${MAX_MODEL_LEN:-}" ] || [ "$MAX_MODEL_LEN" = "0" ]; then
    MAX_MODEL_LEN=131072
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
rocm-smi || true
amd-smi || true

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

CACHE_ARGS=()
case "$OFFLOADING" in
    none)
        CACHE_ARGS=(--disable-radix-cache)
        ;;
    hicache)
        # HiCache extends RadixAttention, so do not pass --disable-radix-cache.
        # MI355X nodes have about 3 TB of host DRAM, but HiCache allocates a
        # large host pool in every TP rank after the model/runtime have already
        # consumed memory. A 2 TB node-total target becomes 250 GB/rank at TP=8
        # and has failed with only ~120-190 GB free by the time later ranks
        # attach the host pool. Keep the node-total knob for one-off tuning, but
        # cap the default per-rank pool below the failed 250 GB/rank request.
        TOTAL_CPU_DRAM_GB="${HICACHE_TOTAL_CPU_DRAM_GB:-2000}"
        HICACHE_MAX_SIZE_GB_PER_RANK="${HICACHE_MAX_SIZE_GB_PER_RANK:-180}"
        HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through_selective}"
        # SGLang --hicache-size is per rank, while the workflow input is a
        # node-total DRAM budget. Divide by TP unless HICACHE_SIZE_GB is set
        # directly for one-off tuning.
        HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-$((TOTAL_CPU_DRAM_GB / TP))}"
        if [ "$HICACHE_SIZE_GB" -gt "$HICACHE_MAX_SIZE_GB_PER_RANK" ]; then
            HICACHE_SIZE_GB="$HICACHE_MAX_SIZE_GB_PER_RANK"
        fi
        if [ "$HICACHE_SIZE_GB" -lt 1 ]; then
            echo "Error: computed HICACHE_SIZE_GB=$HICACHE_SIZE_GB from TOTAL_CPU_DRAM_GB=$TOTAL_CPU_DRAM_GB and TP=$TP" >&2
            exit 1
        fi
        echo "HiCache CPU pool: ${HICACHE_SIZE_GB} GB per rank across TP=${TP}"
        CACHE_ARGS=(
            --page-size 64
            --enable-hierarchical-cache
            --hicache-size "$HICACHE_SIZE_GB"
            --hicache-io-backend kernel
            --hicache-mem-layout page_first
            --hicache-write-policy "$HICACHE_WRITE_POLICY"
        )
        ;;
    *)
        echo "Error: unsupported OFFLOADING value '$OFFLOADING' (expected one of: none, hicache)" >&2
        exit 1
        ;;
esac

echo "Starting SGLang server..."
export PYTHONNOUSERSITE=1

{ set +x; } 2>/dev/null
SGLANG_CMD=(
    python3 -m sglang.launch_server
    --attention-backend triton
    --model-path "$MODEL"
    --host=0.0.0.0
    --port "$PORT"
    --tensor-parallel-size "$TP"
    --ep-size "$EP_SIZE"
    --trust-remote-code
    --tokenizer-worker-num 6
    --enable-aiter-allreduce-fusion
    --cuda-graph-max-bs "$CONC"
    --max-running-requests "$CONC"
    --max-prefill-tokens 32768
    --scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL"
    --mem-fraction-static 0.8
    --context-length "$MAX_MODEL_LEN"
    --enable-metrics
    "${CACHE_ARGS[@]}"
)
printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# ---- Run benchmark ----------------------------------------------------------
build_replay_cmd "$RESULT_DIR"

echo "$REPLAY_CMD" > "$RESULT_DIR/benchmark_command.txt"

set -x
$REPLAY_CMD 2>&1 | tee "$RESULT_DIR/benchmark.log" || true
set +x

write_agentic_result_json "$RESULT_DIR"

# ---- Post-processing --------------------------------------------------------
python3 "$AGENTIC_DIR/scripts/analyze_benchmark_distributions.py" \
    "$RESULT_DIR/aiperf_artifacts" -o "$RESULT_DIR" 2>&1 || true
