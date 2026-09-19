#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4.1-Flash AgentX on MI355X with SGLang DSpark, following the
# cookbook's verified MI350X low-latency cell: AITER kernels, radix cache off,
# breakable prefill graphs. The KV cache is GPU-resident.
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

# Engram tables stay on the GPU here. The shared host copy
# (SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1) that the CUDA arms use made decode
# graph capture fail on gfx950 with hipErrorIllegalAddress at the first batch
# (bs=16, 0/12 captured, run 35311289442), while the first sweep captured
# graphs up to bs=32 with the tables resident (run 35304555945). The 288 GB
# card holds the ~46 GiB of tables next to the weights; the 1M-context
# prefill working set is bounded by --mem-fraction-static 0.60 and the
# 2048-token chunk below instead.

# Cookbook MI350X environment.
# Cap the HIP hardware queues per rank, as the DeepSeek-V4 MI355X SGLang arm
# does. In runs 35304555945 and 35362380897 the server died a few requests
# into AgentX warmup when RCCL queues aborted with HSA_STATUS_ERROR_OUT_OF_
# RESOURCES ("the runtime failed to allocate the necessary resources") with
# 67 GB of HBM still free after the KV pool: the eager 1M-context prefill
# path plus RCCL exhausted the device's hardware queues, not its memory.
export GPU_MAX_HW_QUEUES=2
# MEC firmware below 177 has an RCCL memory-reclaim issue (see the Kimi-K3
# MI355X arm). With the queue cap alone, run 35372886390 still lost c8 to the
# same RCCL HSA_STATUS_ERROR_OUT_OF_RESOURCES abort while c1-c32 served, so
# keep scratch from being reclaimed on affected firmware as well.
mec_version=$(rocm-smi --showfw 2>/dev/null | grep MEC | head -n 1 | awk '{print $NF}')
if [[ "$mec_version" == "" || ${mec_version:-0} -lt 177 ]]; then
    export HSA_NO_SCRATCH_RECLAIM=1
fi
# The ROCm image runs the torch caching allocator with fixed segments (aiter
# logs expandable_segments=False). Eager chunked prefill of one 126k-token
# AgentX prompt then hoards every non-static byte in differently sized cached
# blocks: at --mem-fraction-static 0.60 a single running request exhausted the
# 112 GB outside the static pool 14 chunks in, and the next RCCL kernel launch
# aborted with HSA_STATUS_ERROR_OUT_OF_RESOURCES at 0 MB free (run
# 35460273555, c2; the same abort with 72 GB at c2 and 107 GB at c16 earlier).
# Expandable segments let the allocator grow and reuse one arena instead, as
# SGLang already does on CUDA; aiter's custom all-reduce supports the mode.
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export SGLANG_USE_AITER=1
export SGLANG_MOE_PADDING=1
export AITER_FLYDSL_FORCE_REDUCE=1
export ROCM_QUICK_REDUCE_QUANTIZATION=NONE

# AgentX concurrency counts live session trees, not individual requests.
# Allow subagent fan-out to exceed CONC without clipping request bursts, but
# cap the pool at 128: DSpark verify buffers scale with it, and 256 at c128
# did not fit next to the graphs on H200. Decode graphs stay at the cookbook's
# 64; larger batches decode eagerly, as the cookbook's high-throughput cell does.
MAX_RUNNING_REQUESTS=$((2 * CONC))
if (( MAX_RUNNING_REQUESTS > 128 )); then
    MAX_RUNNING_REQUESTS=128
fi
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
if [[ "${EVAL_ONLY}" != true ]]; then
    export SGLANG_SIMULATE_ACC_LEN="$DSV41_GOLDEN_AL"
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi
echo "DSpark block size: $DSPARK_BLOCK_SIZE, golden AL=$DSV41_GOLDEN_AL"

# --disable-radix-cache is the cookbook's ROCm setting; the server rejects
# hierarchical caching alongside it, which is consistent with kv-offloading none.
SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0 --port "$PORT"
    --trust-remote-code
    --tp "$TP" --ep-size "$EP_SIZE"
    --disable-radix-cache
    # 0.60 rather than the cookbook's 0.8, and a 2048-token prefill chunk (half
    # the CUDA arms' 4096, halving each chunk's transient indexer buffers): the sparse-attention indexer and DSpark prefill buffers
    # scale with the chunk times the 1M context (the default 16384 exhausted
    # HBM on the first 66k-99k-token prompts), and the per-chunk RCCL
    # collectives shrink with it. With 8192 and the queue cap, c1-c16 served
    # but c32 still lost a rank 27 warmup requests in (run 35374653446).
    # With the Engram tables resident on the GPU the weights take 128.8 GB of
    # each 288 GB MI355X, so 0.75 left only 72 GB outside the static pool and
    # the c2 prefill of a 126k-token prompt still aborted its RCCL queue with
    # HSA_STATUS_ERROR_OUT_OF_RESOURCES at 0 MB free (run 35376928227). The
    # full-attention KV costs 1.67 KB per token, so 0.60 still reserves a
    # ~25M-token pool (0.75 reserved 51M) while eager prefill gets 115 GB.
    --mem-fraction-static 0.60
    --chunked-prefill-size 2048
    --speculative-algorithm DSPARK
    --speculative-dspark-block-size "$DSPARK_BLOCK_SIZE"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    # The cookbook's breakable prefill graph is disabled here: with the Engram
    # tables in host memory, capturing the 2048-token prefill graph raised
    # hipErrorIllegalAddress on every rank (run 35306715045, c1). Prefill of
    # 60k-600k-token AgentX prompts runs eagerly; decode graphs are unchanged.
    --cuda-graph-backend-prefill disabled
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
