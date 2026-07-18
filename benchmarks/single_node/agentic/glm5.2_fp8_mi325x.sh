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

if [[ "$TP" != "8" || "$EP_SIZE" != "1" || "$DP_ATTENTION" != "false" ]]; then
    echo "Error: GLM-5.2 MI325X full-context qualification requires TP8/EP1 without DP attention" >&2
    exit 1
fi
if [[ "$KV_OFFLOADING" != "none" ]]; then
    echo "Error: KV_OFFLOADING=$KV_OFFLOADING is not supported by this recipe" >&2
    exit 1
fi

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

export PYTHONNOUSERSITE=1
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export SGLANG_TIMEOUT_KEEP_ALIVE=900
# The mi30x image's sgl-kernel DSA top-k JIT includes CUDA's
# cooperative_groups.h while compiling for gfx942. Use SGLang's portable
# Torch fallback and disable both the fused top-k path and its independently
# gated CUDA-graph planning kernel.
export SGLANG_DSA_FUSE_TOPK=false
export SGLANG_OPT_USE_TOPK_V2=false

MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS
[ "$CUDA_GRAPH_MAX_BS" -gt 256 ] && CUDA_GRAPH_MAX_BS=256

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tp "$TP"
    --ep-size "$EP_SIZE"
    --dsa-prefill-backend tilelang
    --dsa-decode-backend tilelang
    --dsa-topk-backend torch
    --kv-cache-dtype bfloat16
    --tool-call-parser glm47
    --reasoning-parser glm45
    --context-length 1048576
    --max-total-tokens 1048576
    --chunked-prefill-size 131072
    --mem-fraction-static 0.85
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
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
