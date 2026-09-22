#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4.1-Flash AgentX on B300 with SGLang, supporting STP and DSpark.
# The KV cache is GPU-resident; caller SPEC_DECODING selects the serving mode.
# https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL TP EP_SIZE CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION
check_env_vars EVAL_ONLY SPEC_DECODING
require_agentic_kv_offload_none
export GPU_COUNT="$TP"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# Complete/resume partial downloads instead of trusting nonempty directories.
if [[ -n "${MODEL_PATH:-}" && "$MODEL_PATH" != "$MODEL" ]]; then
    hf download "$MODEL" --local-dir "$MODEL_PATH"
else
    hf download "$MODEL"
    MODEL_PATH=$(python3 -c 'from huggingface_hub import snapshot_download; import sys; print(snapshot_download(repo_id=sys.argv[1], local_files_only=True))' "$MODEL")
    export MODEL_PATH
fi

nvidia-smi
resolve_trace_source
install_agentic_deps
mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# Use the default DSpark precision shipped by the pinned SGLang nightly.

# Agentic warmup dispatches hundreds of large prompts at once and SGLang's
# tokenizer can leave bytes unacknowledged past AIPerf's default 30 s
# TCP_USER_TIMEOUT, so Linux aborts live localhost connections.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
# Outlast AIPerf's pooled connections so an inter-turn idle gap cannot race
# Uvicorn's five-second keep-alive closure.
export SGLANG_TIMEOUT_KEEP_ALIVE=900

# AgentX measures the thinking-on regime, which is also the committed golden-AL
# curve. SGLang ships thinking off by default for this model.
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV41_REASONING_EFFORT=high

# TP2 keeps Engram tables in host RAM for long-context AgentX KV. Its per-rank
# anonymous mappings can use THP without host sysctl changes. TP4 tests native
# FP8 tables on GPU, where its smaller weight shard leaves room for them.
if (( TP == 4 )); then
    # TP4 can fit the native FP8 tables on GPU; qualify this placement on B300.
    export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=0
    unset SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT
else
    export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
    export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
fi

# AgentX concurrency counts live session trees, not individual requests.
# Allow subagent fan-out to exceed CONC without clipping request bursts, but
# never let the pool exceed the decode graph batch: a DSpark verify step for a
# batch above the captured 64 runs eagerly and allocates its attention
# workspace on the fly, which OOMed the H200 eval at 128 running requests
# (6.4 GiB allocation with 2 GiB free, run 35306704553). Batches within the
# graph tier reuse the capture-time workspace instead.
CUDA_GRAPH_MAX_BS=64
MAX_RUNNING_REQUESTS=$((2 * CONC))
if (( MAX_RUNNING_REQUESTS > CUDA_GRAPH_MAX_BS )); then
    MAX_RUNNING_REQUESTS=$CUDA_GRAPH_MAX_BS
fi

# AgentX reuses long prefixes across turns even at low session concurrency.
# At memory fraction 0.70, TP4 has 110.55 GiB of KV budget but TP2 has only
# 38.83 GiB. Give TP2's larger working sets more KV space before retaining
# additional SWA tails; keep the conservative low-concurrency allocation.
MEM_FRACTION_STATIC=0.70
if (( TP == 4 )); then
    # GPU-resident tables add about 47 GiB/rank; retain room for long prefixes.
    MEM_FRACTION_STATIC=0.80
elif (( TP == 2 && CONC >= 32 )); then
    # At C64, 0.80 left 43.59 GiB after graphs but only 18.29M full tokens.
    # Retain more long prefixes while leaving room for transient prefills.
    MEM_FRACTION_STATIC=0.85
elif (( TP == 2 && CONC >= 16 )); then
    MEM_FRACTION_STATIC=0.80
fi
SWA_PREFIX_TAILS=$((64 * CONC))
if (( SWA_PREFIX_TAILS > 4096 )); then
    SWA_PREFIX_TAILS=4096
fi
if (( SWA_PREFIX_TAILS < 128 )); then
    SWA_PREFIX_TAILS=128
fi

# Saturation arms carry a larger in-flight working set than the 30-minute
# default warmup drain allows.
if (( CONC >= 32 )); then
    export AGENTIC_WARMUP_GRACE_PERIOD=3600
fi

# Pyxis shares the host network; port 8888 can already belong to a host service.
select_available_server_port
export AIPERF_SERVER_URL="http://localhost:${PORT}"
export AIPERF_SERVER_METRICS_URLS="${AIPERF_SERVER_URL}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"
echo "Using SGLang endpoint ${AIPERF_SERVER_URL}"

# STP and accuracy evaluations must never inherit synthetic acceptance.
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATE_ACC_TOKEN_MODE
SPECULATIVE_ARGS=()
case "$SPEC_DECODING" in
    none)
        echo "Non-speculative decoding; synthetic acceptance disabled"
        ;;
    mtp)
        # Existing measured curve: dsv41flash_dspark.yaml, thinking_on, K5.
        SPECULATIVE_ARGS=(--speculative-algorithm DSPARK --speculative-dspark-block-size 5)
        if [[ "$EVAL_ONLY" != true ]]; then
            export SGLANG_SIMULATE_ACC_LEN=3.51
            export SGLANG_SIMULATE_ACC_METHOD=match-expected
            export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
        fi
        ;;
    *)
        echo "Unsupported SPEC_DECODING: $SPEC_DECODING" >&2
        exit 1
        ;;
esac

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0 --port "$PORT"
    --trust-remote-code
    # Feed mmap weight copies sequentially from shared Lustre storage.
    --weight-loader-prefetch-checkpoints
    --tp "$TP" --ep-size "$EP_SIZE"
    # Backends resolve automatically (dsv4 / flashinfer_mxfp4 / flashinfer_cutedsl
    # on Blackwell); the cookbook warns that overriding them costs decode speed.
    # Bound transient prefill allocations: the sparse-attention indexer and
    # DSpark buffers scale with the chunk times the 1M context. Static KV
    # memory is selected above from the measured TP2/TP4 weight footprints.
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --chunked-prefill-size 4096
    # Long AgentX prefills otherwise starve ready decode requests.
    --prefill-decode-interval 16
    "${SPECULATIVE_ARGS[@]}"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --swa-prefix-tails "$SWA_PREFIX_TAILS"
    --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS"
    --reasoning-parser auto
    --tool-call-parser auto
    # Draft-token forward passes under long-context agentic load block the
    # scheduler long enough to trip the 1800 s default watchdog mid-warmup.
    --watchdog-timeout 3600
    --enable-metrics
)
write_command "$RESULT_DIR/sglang_command.txt" "${SGLANG_CMD[@]}"
{
    echo "=== SGLANG_* env vars at launch ==="
    env | grep -E '^SGLANG_' | sort
    echo "==================================="
} | tee "$SERVER_LOG"
"${SGLANG_CMD[@]}" >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "${EVAL_ONLY}" == true ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics ${AIPERF_SERVER_METRICS_URLS}"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
