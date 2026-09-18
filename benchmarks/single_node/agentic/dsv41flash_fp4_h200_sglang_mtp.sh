#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4.1-Flash AgentX on H200 with SGLang native DSpark, following the
# cookbook's verified H200 TP8/EP8 low-latency cell. The KV cache is GPU-resident.
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
# copy per rank: the SGLang analogue of the vLLM arm's Engram CPU offload. It
# frees ~23 GiB of HBM per GPU for the 1M-context prefill working set and the
# KV pool, and output is bitwise unchanged (cookbook). The first sweep ran
# with the tables on GPU and the server died on the first long AgentX prompts
# (run 35304458924: c128 died at startup with torch.OutOfMemoryError (12 GiB allocation, 5 GiB free of 139.8 GiB, 127.6 GiB already held by PyTorch)).
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1

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
    # The cookbook resolves backends automatically and warns against overriding
    # them; its verified H200 cell is the one exception and pins these two.
    --attention-backend dsv4 --moe-runner-backend flashinfer_mxfp4
    # 0.70 rather than the cookbook's 0.8, and a bounded prefill chunk: the
    # sparse-attention indexer and DSpark prefill buffers scale with the chunk
    # times the 1M context, and the default 16384 chunk exhausted HBM on the
    # first 66k-99k-token AgentX prompts.
    --mem-fraction-static 0.70
    --chunked-prefill-size 8192
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
