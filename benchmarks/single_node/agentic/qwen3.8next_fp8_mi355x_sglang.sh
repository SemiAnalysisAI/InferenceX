#!/usr/bin/env bash
set -euo pipefail
set -x

# AgentX trace replay for Qwen3.8-Flash-Next FP8 on MI355X with SGLang.
# Day-zero recipe; SGLang is the plan-of-record engine for this model
# (MODELS.md).
#
# Two deliberate differences from the NVIDIA arms:
#
#   * No speculative decoding. The MI355X SGLang path for this model does not
#     drive MTP yet, so this arm runs the target model alone and carries no
#     synthetic-acceptance pin. Add MTP and the golden AL once ROCm supports
#     it. Until then this arm is not directly comparable to the spec-decode
#     NVIDIA arms on the published frontier.
#
#   * FP8, not FP4. NVFP4 is greyed out for MI355X in the SGLang cookbook and
#     there is no AMD FP4 checkpoint for this model.
#
# KNOWN BLOCKER, upstream, not in this recipe: on the current
# lmsysorg/sglang-rocm:qwen38flashnext image the server aborts while loading
# weights with
#   AssertionError: Expected 1.0, got 0.00019931793212890625 in skipped
#   model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale
#   qwen4_exp.py:2039 in load_weights
# The model registers no parameter for the PLE ngram embedding, so load_weights
# takes the "skipped" branch, which assumes any orphaned _scale must be a no-op
# and asserts it is 1.0. The checkpoint really does quantize that embedding:
# its 128 shard_N.weight tensors are F8_E4M3 with a single BF16 weight_scale of
# ~1.99e-4, and modules_to_not_convert lists ple.conv1d, ple.key_proj and
# ple.value_proj but not the ngram embedding.
#
# Suppressing the assert is NOT a fix: the shards would load as raw FP8 with
# the scale never applied, leaving that embedding wrong by ~5000x with no error
# reported. The CUDA image carries a different SGLang build and loads the same
# checkpoint, so this is specific to the ROCm build. It clears when that image
# implements the quantized PLE ngram embedding; the recipe below is otherwise
# the cookbook's verified balanced command and needs no further change.

source "$(dirname "$0")/../../benchmark_lib.sh"

export EVAL_FRAMEWORK="lm-eval"

check_env_vars \
    MODEL TP CONC EP_SIZE KV_OFFLOADING \
    TOTAL_CPU_DRAM_GB RESULT_DIR DURATION

SCHEDULER_RECV_INTERVAL=${SCHEDULER_RECV_INTERVAL:-30}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
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

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
resolve_trace_source
install_agentic_deps

export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "SGLang server" 60
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

CACHE_ARGS=()
if require_agentic_kv_offload_backend hicache; then
    HICACHE_RATIO="${HICACHE_RATIO:-1.5}"
    HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
    HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
    HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first_direct}"
    echo "HiCache CPU tier: ratio=$HICACHE_RATIO, write_policy=$HICACHE_WRITE_POLICY, io_backend=$HICACHE_IO_BACKEND, mem_layout=$HICACHE_MEM_LAYOUT, dram_budget=${TOTAL_CPU_DRAM_GB} GB, tp=$TP"
    CACHE_ARGS=(
        --enable-hierarchical-cache
        --hicache-ratio "$HICACHE_RATIO"
        --hicache-write-policy "$HICACHE_WRITE_POLICY"
        --hicache-io-backend "$HICACHE_IO_BACKEND"
        --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
    )
fi

PARALLEL_ARGS=(
    --tp "$TP"
    --dp 1
    --ep-size "$EP_SIZE"
)

TOKENIZER_ARGS=()
if [ "$TP" -ge 4 ]; then
    TOKENIZER_ARGS=(--tokenizer-worker-num 6)
fi

MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS="$CONC"
[ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64

export PYTHONNOUSERSITE=1
export SGLANG_USE_AITER=1
export SGLANG_USE_AITER_UNIFIED_ATTN=1
export AITER_FLYDSL_FORCE=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
export SGLANG_TIMEOUT_KEEP_ALIVE=1800


SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    # Verified flags from the SGLang cookbook playground for this model on
    # MI355X / FP8 / balanced / single node. Low-latency and high-throughput
    # are not offered for this part, and NVFP4 is greyed out, so balanced FP8
    # at TP8 is the whole of the verified AMD surface today.
    --tp-size "$TP"
    --attention-backend aiter
    --page-size 32
    --kv-cache-dtype auto
    --chunked-prefill-size 16384
    --watchdog-timeout 1200
    --mem-fraction-static 0.9
    --model-loader-extra-config '{"enable_multithread_load": true}'
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --stream-interval 50
    --scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL"
    "${TOKENIZER_ARGS[@]}"
    --tokenizer-path "$MODEL"
    --enable-metrics
    --enable-cache-report
    "${CACHE_ARGS[@]}"
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"
"${SGLANG_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --apply-chat-template"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
