#!/usr/bin/env bash
set -euo pipefail
set -x

# Full-context AgentX qualification for GLM-5.2 FP8 on one 8xMI325X node.
# The SGLang GLM-5.2 cookbook supports TP8 FP8 on MI325X with the DSA
# TileLang backends. This recipe explicitly requests the native 1M context and
# GPU-resident BF16 KV pool; SGLang startup fails if that allocation cannot be
# satisfied. MTP is intentionally disabled because the AMD path is unvalidated.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi
rocm-smi || true
amd-smi || true

# GLM-5.2 has a native 1M context, so replay the complete AgentX corpus rather
# than the repository's default 256K-capped corpus for unrecognized families.
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
resolve_trace_source
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"
SERVER_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

CACHE_ARGS=()
if require_agentic_kv_offload_backend hicache; then
    # GLM-5.2's DSA KV pool is replicated across the TP ranks. A 0.75 host
    # tier adds roughly 0.75M cold-prefix tokens per rank while staying well
    # inside the MI325X runner's measured 3 TB CPU-DRAM budget.
    HICACHE_RATIO="${HICACHE_RATIO:-0.75}"
    CACHE_ARGS=(
        --enable-hierarchical-cache
        --hicache-ratio "$HICACHE_RATIO"
        --hicache-write-policy "${HICACHE_WRITE_POLICY:-write_back}"
        --hicache-io-backend "${HICACHE_IO_BACKEND:-direct}"
        --hicache-mem-layout "${HICACHE_MEM_LAYOUT:-page_first_direct}"
    )
fi

export PYTHONNOUSERSITE=1
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export SGLANG_TIMEOUT_KEEP_ALIVE=900
# The mi30x image's sgl-kernel DSA top-k JIT includes CUDA's
# cooperative_groups.h while compiling for gfx942. Use SGLang's portable
# Torch fallback and disable both the fused top-k path and its independently
# gated CUDA-graph planning kernel.
export SGLANG_DSA_FUSE_TOPK=false
export SGLANG_OPT_USE_TOPK_V2=false

PARALLEL_ARGS=(--tp "$TP" --ep-size "$EP_SIZE")

# Keep the cookbook profiles as separate topology/cache series so the AgentX
# dashboard can compare their complete curves instead of overlaying identical
# labels. All profiles retain the same 1M request limit and 1M HBM KV pool.
PROFILE=low-latency
CHUNKED_PREFILL_ARGS=(--chunked-prefill-size 131072)
MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_ARGS=(--cuda-graph-max-bs "$MAX_RUNNING_REQUESTS")
if [ "$EP_SIZE" -gt 1 ]; then
    PROFILE=balanced
    CHUNKED_PREFILL_ARGS=(--chunked-prefill-size 32768)
    MAX_RUNNING_REQUESTS=80
    CUDA_GRAPH_ARGS=(--cuda-graph-max-bs 128)
elif [ "$KV_OFFLOADING" != "none" ]; then
    PROFILE=hicache
    CHUNKED_PREFILL_ARGS=(--chunked-prefill-size 32768)
    MAX_RUNNING_REQUESTS=80
    CUDA_GRAPH_ARGS=(--cuda-graph-max-bs 128)
fi
echo "GLM-5.2 MI325X AgentX profile: $PROFILE"

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    --dsa-prefill-backend tilelang
    --dsa-decode-backend tilelang
    --dsa-topk-backend torch
    --kv-cache-dtype bfloat16
    --tool-call-parser glm47
    --reasoning-parser glm45
    --context-length 1048576
    --max-total-tokens 1048576
    "${CHUNKED_PREFILL_ARGS[@]}"
    --mem-fraction-static 0.85
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    "${CUDA_GRAPH_ARGS[@]}"
    "${CACHE_ARGS[@]}"
    --watchdog-timeout 1800
    --enable-metrics
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

echo "Starting SGLang server for MI325X..."
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "${EVAL_ONLY}" == "true" ]]; then
    export SWEBENCH_AGENT_STEP_LIMIT=150
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics http://localhost:$PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
