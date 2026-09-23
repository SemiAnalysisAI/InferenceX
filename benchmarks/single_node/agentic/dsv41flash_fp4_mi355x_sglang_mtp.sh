#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4.1-Flash AgentX on MI355X with SGLang DSpark, following the
# cookbook's MI350X cell, with radix caching enabled for AgentX prefix reuse.
# The KV cache and TP4 Engram tables are GPU-resident.
# https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL TP EP_SIZE CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION
check_env_vars EVAL_ONLY PORT
require_agentic_kv_offload_none
export GPU_COUNT="$TP"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# ROCR/HIP visibility under slurm cgroups.
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

# Complete/resume partial downloads instead of trusting nonempty directories.
if [[ -n "${MODEL_PATH:-}" && "$MODEL_PATH" != "$MODEL" ]]; then
    hf download "$MODEL" --local-dir "$MODEL_PATH"
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi
rocm-smi || true
amd-smi || true

# A server killed minutes earlier can still be draining HBM (KFD reclaim takes
# minutes), and booting into a half-drained node fails RCCL init with HIP
# 'unhandled cuda error'. Idle GPUs sit at up to ~4% VRAM, draining ones at
# 50-90%, so require every GPU <= 10%.
GPU_CLEAN=false
for i in $(seq 1 90); do
    VRAM_MAX=$(rocm-smi --showmemuse 2>/dev/null | grep -oE "GPU Memory Allocated \(VRAM%\): [0-9]+" | awk '{if ($NF > m) m = $NF} END {print m+0}')
    if [[ "${VRAM_MAX:-0}" -le 10 ]]; then echo "GPUs clean (vram%max=$VRAM_MAX after $((i*10))s)"; GPU_CLEAN=true; break; fi
    echo "waiting for prior-job GPU memory reclaim: vram%max=$VRAM_MAX"; sleep 10
done
[[ "$GPU_CLEAN" == true ]] || { echo "Error: GPUs still draining prior job's memory after 15min" >&2; exit 1; }

# Pin the full-context corpus for this 1M-context recipe.
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
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

# Use the official model-preview image's native kernels, allocator and tuning
# CSV. TP4 Engram remains GPU-resident, as in the official MI350X recipe.
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=0
export SGLANG_USE_AITER=1
export SGLANG_MOE_PADDING=1
export AITER_FLYDSL_FORCE_REDUCE=1
export ROCM_QUICK_REDUCE_QUANTIZATION=NONE

# Long-prefill scratch can fill the native allocator cache and starve HIP/RCCL
# allocations outside PyTorch. The preview's allocator only activates GC when
# per_process_memory_fraction is below 1; reclaim unused blocks at 80% of 99%.
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,per_process_memory_fraction:0.99

CUDA_GRAPH_MAX_BS=64

# Saturation arms carry a larger in-flight working set than the 30-minute
# default warmup drain allows.
if (( CONC >= 32 )); then
    export AGENTIC_WARMUP_GRACE_PERIOD=3600
fi

# Use the runner-specific port assigned by launch_mi355x-amds.sh.
export AIPERF_SERVER_URL="http://localhost:${PORT}"
export AIPERF_SERVER_METRICS_URLS="${AIPERF_SERVER_URL}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"
echo "Using SGLang endpoint ${AIPERF_SERVER_URL}"

# DSpark is the checkpoint's own bundled draft: no EAGLE/MTP path and no
# --speculative-num-steps knob; the block size is the only tunable. Golden AL:
# golden_al_distribution/dsv41flash_dspark.yaml, thinking_on, five draft tokens.
# Throughput fixes acceptance to AL 3.51; accuracy evals keep real verification.
DSPARK_BLOCK_SIZE=5
DSV41_GOLDEN_AL=3.51
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATE_ACC_TOKEN_MODE
if [[ "${EVAL_ONLY}" != true ]]; then
    export SGLANG_SIMULATE_ACC_LEN="$DSV41_GOLDEN_AL"
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi
echo "DSpark block size: $DSPARK_BLOCK_SIZE, golden AL=$DSV41_GOLDEN_AL"

# Official MI350X TP4/EP4 recipe adapted for AgentX radix prefix reuse.
SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0 --port "$PORT"
    --trust-remote-code
    --tp "$TP" --ep-size "$EP_SIZE"
    # C32 still exhausted HBM at 0.80 with 4096-token chunks; reserve
    # another 10% of physical HBM for native long-prefill scratch.
    --mem-fraction-static 0.70
    # Native FP4 prefill scratch scales with query tokens times context.
    # The 16384-token default OOMed five canonical cells; match the existing
    # 4096-token breakable graph ceiling without reducing model context.
    --chunked-prefill-size 4096
    # AgentX C32 fanout reached 63 running requests before a native HIP illegal
    # access. Bound admission; additional client requests remain queued.
    --max-running-requests 32
    --speculative-algorithm DSPARK
    --speculative-dspark-block-size "$DSPARK_BLOCK_SIZE"
    --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS"
    --cuda-graph-backend-prefill breakable
    --cuda-graph-max-bs-prefill 4096
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
    env | grep -E '^SGLANG_|^PYTORCH_(HIP|CUDA|ALLOC)' | sort
    echo "==================================="
} | tee "$SERVER_LOG"
SERVER_PID=""
cleanup_server() {
    local rc=$?
    trap - EXIT INT TERM
    stop_background_process_tree "$SERVER_PID" "SGLang server" 60
    exit "$rc"
}
trap cleanup_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
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
