#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4.1-Flash AgentX on H200 with SGLang native DSpark, following the
# vLLM baseline topologies TP4/EP1 and TP8/EP1 in this isolated experiment.
# The KV cache is GPU-resident.
# https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL TP EP_SIZE CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION
check_env_vars EVAL_ONLY DP_ATTENTION
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

# Install measured H200 launch configurations: the replicated projection on
# TP4 and TP8, plus the qualified TP8 sharded shapes. Kernel code, precision
# and large-prefill tilings remain unchanged.
python3 "$(dirname "$0")/install_h200_block32_configs.py" \
    "$(dirname "$0")/kernel_configs/h200_dsv41_block32" "$RESULT_DIR" "$TP"

# Matched C1 screens favored CUTLASS on TP8 p90 interactivity, while Marlin
# retained a small throughput/interactivity advantage on TP4. Both consume
# native MXFP4 weights with BF16 activations; dense GEMMs are unchanged.
# EP1 tensor-shards the 2304-wide routed experts: TP4=576, TP8=288.
# The shipped SM90 CUTLASS method requires multiples of 128 and rejects
# those widths; native Marlin supports padding without changing precision.
MOE_RUNNER_BACKEND=marlin
if (( TP == 8 && EP_SIZE != 1 )); then
    MOE_RUNNER_BACKEND=flashinfer_mxfp4
fi

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

# Move the two fp8 Engram tables to host memory, freeing ~23 GiB of HBM per
# GPU for the 1M-context prefill working set and KV pool. Use row-sharded
# anonymous mappings: H200 compute nodes allow anonymous THP with madvise,
# but shmem_enabled=never prevents the shared memfd layout from using huge
# pages. This changes table placement, preserving checkpoint weights/scales.
# The first sweep ran
# with the tables on GPU and the server died on the first long AgentX prompts
# (run 35304458924: c128 died at startup with torch.OutOfMemoryError (12 GiB allocation, 5 GiB free of 139.8 GiB, 127.6 GiB already held by PyTorch)).
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
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
if [[ "$DP_ATTENTION" == true ]] && (( MAX_RUNNING_REQUESTS < TP )); then
    # SGLang divides this global cap by attention DP size when sizing pools.
    # Low-concurrency DP still needs at least one request slot per rank.
    MAX_RUNNING_REQUESTS=$TP
fi

# Saturation arms carry a larger in-flight working set than the 30-minute
# default warmup drain allows.
if (( CONC >= 32 )); then
    export AGENTIC_WARMUP_GRACE_PERIOD=3600
fi

# Pyxis shares the host network; port 8888 can already belong to a host service.
select_available_server_port
export AIPERF_SERVER_URL="http://localhost:${PORT}"
SGLANG_BACKEND_PORT="$PORT"
PARALLEL_ARGS=(--tp "$TP" --ep-size "$EP_SIZE")
if [[ "$DP_ATTENTION" == true ]]; then
    # The shipped MoE DSpark worker requires attn_tp=1 under DP attention.
    # Keep the engine-wide 4096-token chunk budget across the supported DP8 arm;
    # SGLang divides it by DP, yielding 512 tokens/rank at TP8/DP8.
    PARALLEL_ARGS+=(--enable-dp-attention --dp-size "$TP" --enable-dp-lm-head)
    SGLANG_BACKEND_PORT=$((PORT + 1))
    SGLANG_ROUTER_METRICS_PORT=$((PORT + 10000))
    export AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID=true
elif [[ "$DP_ATTENTION" != false ]]; then
    echo "Error: DP_ATTENTION must be true or false, got '$DP_ATTENTION'" >&2
    exit 1
fi
export AIPERF_SERVER_METRICS_URLS="http://localhost:${SGLANG_BACKEND_PORT}/metrics"
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
    --host 0.0.0.0 --port "$SGLANG_BACKEND_PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    --attention-backend dsv4 --moe-runner-backend "$MOE_RUNNER_BACKEND"
    # 0.70 rather than the cookbook's 0.8, and a bounded prefill chunk: the
    # sparse-attention indexer and DSpark prefill buffers scale with the chunk
    # times the 1M context, and the default 16384 chunk exhausted HBM on the
    # first 66k-99k-token AgentX prompts.
    --mem-fraction-static 0.70
    # 4096, as on B200/GB200: at 8192 the indexer's prefill top-k allocated 5 GiB
    # with 29 requests in flight and OOMed c32 (run 35308550355).
    --chunked-prefill-size 4096
    # Keep active decode requests progressing while long prefixes are queued.
    --prefill-decode-interval 16
    # The default 4*max-running-requests retains too few SWA prefix tails:
    # C16 exhausted its 94,976-slot SWA pool while millions of full-pool
    # slots remained free. Rebalance the existing KV budget toward reusable
    # tails; weights, KV precision and the total static budget stay unchanged.
    --swa-prefix-tails 1024
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
wait_for_server_ready --port "$SGLANG_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "$DP_ATTENTION" == true ]]; then
    # Stable session keys preserve prefix reuse across turns. The DP-aware
    # router selects a rank through SGLang's supported routed_dp_rank path.
    ROUTER_LOG="$RESULT_DIR/router.log"
    ROUTER_CMD=(
        python3 -m sglang_router.launch_router
        --worker-urls "http://localhost:$SGLANG_BACKEND_PORT"
        --policy consistent_hashing
        --request-id-headers x-correlation-id
        --dp-aware
        --host 0.0.0.0 --port "$PORT"
        --prometheus-host 127.0.0.1
        --prometheus-port "$SGLANG_ROUTER_METRICS_PORT"
        --connect-timeout-secs 900 --request-timeout-secs 14400
        --disable-health-check --disable-retries
    )
    write_command "$RESULT_DIR/router_command.txt" "${ROUTER_CMD[@]}"
    "${ROUTER_CMD[@]}" > "$ROUTER_LOG" 2>&1 &
    ROUTER_PID=$!
    wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"
fi

if [[ "${EVAL_ONLY}" == true ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics ${AIPERF_SERVER_METRICS_URLS}"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
