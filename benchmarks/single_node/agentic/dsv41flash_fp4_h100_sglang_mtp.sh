#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4.1-Flash AgentX on H100 with SGLang DSpark. A copy of
# dsv41flash_fp4_sglang_mtp.sh rather than a symlink: H100 is not in the
# cookbook's hardware table, and 80 GB cards cannot hold the resident weights
# plus the row-sharded Engram tables (~23.6 GiB per rank at TP8, measured on
# the vLLM arm) and still leave a KV pool. The Engram tables move to one shared
# host copy, and the prefill chunk is halved so the sparse-attention indexer's
# [chunk, context] scoring buffer fits next to the weights at 1M context.
# https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL TP EP_SIZE CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION
check_env_vars EVAL_ONLY
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
    export MODEL_PATH="$MODEL"
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

# One shared host copy of the two fp8 Engram tables instead of a row-sharded
# copy per rank: both engram all-reduces disappear, the freed HBM goes to the
# KV pool, and output is bitwise unchanged. This is the SGLang counterpart of
# the vLLM arm's engram cpu_offload; kv-offloading stays none because the KV
# cache itself is GPU-resident.
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1

# AgentX concurrency counts live session trees, not individual requests.
# Allow subagent fan-out to exceed CONC without clipping request bursts, and
# keep decode graphs covering that fan-out down to the cookbook's 64.
MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS
if (( CUDA_GRAPH_MAX_BS < 64 )); then
    CUDA_GRAPH_MAX_BS=64
elif (( CUDA_GRAPH_MAX_BS > 128 )); then
    CUDA_GRAPH_MAX_BS=128
fi

# The indexer's scoring buffer and the hyper-connection activations scale with
# the prefill chunk times the 1M context. 4096 at mem-fraction 0.8 left 16 GB
# of headroom on the 80 GB card and c8 OOMed in eager extend once five
# requests were live with a 570k-token prompt pending (run 35304509605); 2048
# with 0.7 leaves 24 GB and halves the per-chunk working set.
# Back to 4096 with the static fraction kept at 0.7: at 2048 prefill ran near
# 1,000 tok/s and c4 failed AIPerf's 95% latency-coverage check (TTFT 87.6%,
# ITL 88.4% over the 3600 s window, run 35307250127) while c1/c2 passed.
# 0.7 leaves 24 GB for the doubled per-chunk working set instead of the
# 16 GB that OOMed c8 at 0.8/4096.
CHUNKED_PREFILL_SIZE=4096

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

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0 --port "$PORT"
    --trust-remote-code
    --tp "$TP" --ep-size "$EP_SIZE"
    # The cookbook's verified Hopper (H200) cell pins these two backends.
    --attention-backend dsv4 --moe-runner-backend flashinfer_mxfp4
    --mem-fraction-static 0.7
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
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
