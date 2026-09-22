#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4.1-Flash AgentX on B200 with shipped-default DSpark.
# TP4 covers the full concurrency curve; TP2 covers C1-C8.
# https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL TP EP_SIZE CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION
check_env_vars EVAL_ONLY SPEC_DECODING
require_agentic_kv_offload_none
export GPU_COUNT="$TP"

if (( TP == 2 )); then
    # Test whether expandable segments alone avoid load-time fragmentation.
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    echo "TP2 CUDA allocator: $PYTORCH_CUDA_ALLOC_CONF"
    python3 - <<'PY_STOCK_LOADER'
import hashlib
from pathlib import Path
import sglang

loader = Path(sglang.__file__).parent / "srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py"
actual = hashlib.sha256(loader.read_bytes()).hexdigest()
expected = "0ff4ca142e71ec0baabc00844b4854e554c607c1a356ffe13419c59b1f51813c"
if actual != expected:
    raise SystemExit(f"Expected pinned stock loader {expected}, found {actual}")
print(f"Verified unmodified stock MXFP4 loader: {actual}")
PY_STOCK_LOADER
fi

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

# Keep Engram in per-rank host shards to reserve HBM for reusable KV.
case "$TP" in
    2|4) export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 ;;
    *) echo "Unsupported DSpark TP=$TP; expected 2 or 4" >&2; exit 1 ;;
esac
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank

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
# Chunked requests leave reusable SWA tails in the radix tree. Size retained
# tails by session concurrency, rather than the capped running-request count.
# The tails and full KV share a fixed pool; the cap preserves full-prefix space.
SWA_PREFIX_TAILS=$((64 * CONC))
MEM_FRACTION_STATIC=0.80
CHUNKED_PREFILL_SIZE=4096
if (( TP == 2 )); then
    if (( CONC > 8 )); then
        echo "TP2 supports CONC <= 8 within its smaller KV budget" >&2
        exit 1
    fi
    # TP2 has twice as many chunk boundaries and approximately 147.76 GiB of
    # target plus draft weights. Smaller chunks bound indexer workspace.
    SWA_PREFIX_TAILS=$((128 * CONC))
    MEM_FRACTION_STATIC=0.92
    CHUNKED_PREFILL_SIZE=2048
fi
if (( SWA_PREFIX_TAILS < 128 )); then
    SWA_PREFIX_TAILS=128
elif (( SWA_PREFIX_TAILS > 4096 )); then
    SWA_PREFIX_TAILS=4096
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

# Throughput uses the committed golden acceptance curve; evals verify real
# draft tokens. Leave the pinned nightly's draft computation/precision defaults.
if [[ "$SPEC_DECODING" != mtp ]]; then
    echo "Unsupported SPEC_DECODING=$SPEC_DECODING; expected mtp" >&2
    exit 1
fi
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATE_ACC_TOKEN_MODE
DSPARK_BLOCK_SIZE=5
DSV41_GOLDEN_AL=3.51
if [[ "$EVAL_ONLY" != true ]]; then
    export SGLANG_SIMULATE_ACC_LEN="$DSV41_GOLDEN_AL"
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi
echo "DSpark block size: $DSPARK_BLOCK_SIZE, golden AL=$DSV41_GOLDEN_AL"

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0 --port "$PORT"
    --trust-remote-code
    --tp "$TP" --ep-size "$EP_SIZE"
    # Backends resolve automatically (dsv4 / flashinfer_mxfp4 / flashinfer_cutedsl
    # on Blackwell); the cookbook warns that overriding them costs decode speed.
    # The sparse-attention indexer and DSpark prefill buffers scale with the
    # chunk times the 1M context; bound chunks to retain transient workspace.
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    # Long AgentX prefills otherwise starve active draft/verify decode rounds.
    --prefill-decode-interval 16
    --swa-prefix-tails "$SWA_PREFIX_TAILS"
    --speculative-algorithm DSPARK
    --speculative-dspark-block-size "$DSPARK_BLOCK_SIZE"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
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
