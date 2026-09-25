#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4.1-Flash on MI355X: native DSpark, GPU-resident KV.
# https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION
check_env_vars EVAL_ONLY EP_SIZE DP_ATTENTION
export GPU_COUNT="$TP"

# Complete/resume partial downloads instead of trusting nonempty directories.
if [[ -n "${MODEL_PATH:-}" && "$MODEL_PATH" != "$MODEL" ]]; then
    hf download "$MODEL" --local-dir "$MODEL_PATH"
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
# AITER's Triton MoE GEMM warns on every call that Gluon (gfx1250-only) is
# unavailable; on gfx950 that was 98% of the server log. Set back to WARNING
# when diagnosing new AITER startup or runtime failures.
export AITER_TRITON_LOG_LEVEL=ERROR
# DeepseekV41ForCausalLM is not torch-compiled upstream, so the default
# cudagraph_mode=FULL_AND_PIECEWISE aborts at engine init ("piecewise CUDA
# graphs unavailable"); amd/attention.py uses eager_break_during_capture.
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export OMP_NUM_THREADS=1
# Pin the full-context corpus for this 1M-context recipe.
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
resolve_trace_source
install_agentic_deps
mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_RUST_FRONTEND=1
export PYTHONUNBUFFERED=1

# vllm-project/vllm#57491 widened the two is_cuda() gates to is_cuda_alike(), so
# on gfx950 this image resolves an Engram config and an explicit value is needed
# rather than the VLLM_PLE_CPU_OFFLOAD default.
#
# The tables cost 47.2 GiB per rank at TP=4, so 94.4 GiB at TP=2, which does not
# fit beside half of the 511 GB checkpoint on a 288 GiB card: TP=2 always
# offloads. TP=4 keeps them resident, as the validated concurrency 1-32 run
# measured, because offloaded lookups go to pinned host memory over UVA and
# nothing below c128 is short of KV. Resident leaves 14.83M KV tokens, which is
# 232K per request at c64 and healthy, but only 116K at c128, under the 122K at
# which TP=2 c64 collapsed. Offloading lifts it to 33.06M, so 258K at c128.
if (( TP == 2 || CONC >= 128 )); then
    ENGRAM_CONFIG='{"cpu_offload":true}'
else
    ENGRAM_CONFIG='{"cpu_offload":false}'
fi

# Graph capture covers twice the outer concurrency, floored at the #3058 size of
# 128 sequences, across the 1+5 DSpark token shape. Twice leaves headroom for
# AgentX subagent fan-out above the outer concurrency.
NUM_SPEC_TOKENS=5
GRAPH_NUM_SEQS=$((2 * CONC))
if (( GRAPH_NUM_SEQS < 128 )); then
    GRAPH_NUM_SEQS=128
fi
CAPTURE_SIZE=1
while (( CAPTURE_SIZE < GRAPH_NUM_SEQS * (1 + NUM_SPEC_TOKENS) && CAPTURE_SIZE < 2048 )); do
    CAPTURE_SIZE=$((CAPTURE_SIZE * 2))
done

# The sparse-attention indexer and its companion per-rank buffers scale with
# --max-num-batched-tokens at roughly 4.4 MiB per token, measured on gfx950, so
# a smaller prefill chunk buys KV room. TP=2 starts from half the per-rank space
# and is the arm that runs short: at the upstream 16384 it holds 7.84M KV
# tokens, 122K per request at c64, where run 35574132719 fell to a 17.6% prefix
# cache hit rate, 187 s TTFT and 150 tok/s against 94.8%, 1.3 s and 957 tok/s at
# c32. Every point that held had 232K per request or more, so keep the upstream
# chunk through c32 (245K at TP=2) and trade it away only above that. B300 runs
# 8192 at TP=4 and the Blackwell TP=2 arms run 4096 (#3320, #3321).
if (( CONC >= 128 )); then
    (( TP == 2 )) && BATCHED_TOKENS=4096 || BATCHED_TOKENS=8192
elif (( TP == 2 && CONC >= 64 )); then
    BATCHED_TOKENS=8192
else
    BATCHED_TOKENS=16384
fi

# DSpark verifies 1+5 tokens per sequence, so a decode batch of max_num_seqs
# needs six times that many token slots. The MI355X API-server default of 1024
# sequences therefore wants 6144, and below that the engram projection faults
# during profiling: TP=2 c128 at 4096 leaves four slots per sequence and dies
# with HSA_STATUS_ERROR_EXCEPTION at M=1024, N=129280, K=256, reproduced on two
# separate GPU pairs. Where the chunk is that small, cap in-flight sequences at
# the shape graph capture already covers, which also keeps the largest decode
# batch on a captured graph. Leave the default alone everywhere else.
DEFAULT_MAX_NUM_SEQS=1024
MAX_NUM_SEQS=""
if (( BATCHED_TOKENS < DEFAULT_MAX_NUM_SEQS * (1 + NUM_SPEC_TOKENS) )); then
    MAX_NUM_SEQS="$GRAPH_NUM_SEQS"
fi

# Use the runner-specific port assigned by launch_mi355x-amds.sh.
# DP-attention serves one API server with DP engine ranks behind a native
# vllm-router: AIPerf sends traffic to the router on $PORT and scrapes the
# engine's /metrics on the backend port. Pure TP keeps the backend on $PORT and
# leaves the router disabled, so the metrics URL is unchanged there.
USE_VLLM_ROUTER=false
VLLM_BACKEND_PORT="$PORT"
if [ "$DP_ATTENTION" = "true" ]; then
    USE_VLLM_ROUTER=true
    VLLM_BACKEND_PORT=$((PORT + 1))
    VLLM_ROUTER_VERSION=0.1.15
    VLLM_ROUTER_POLICY=consistent_hash
    VLLM_ROUTER_METRICS_PORT=$((PORT + 10000))
    # AIPerf's X-Correlation-ID is stable across a conversation's turns; alias it
    # to the router's X-Session-ID so every turn lands on the same DP rank.
    export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=1
    agentic_pip_install --quiet "vllm-router==$VLLM_ROUTER_VERSION"
fi
export AIPERF_SERVER_URL="http://localhost:${PORT}"
export AIPERF_SERVER_METRICS_URLS="http://localhost:${VLLM_BACKEND_PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"
echo "Using vLLM endpoint ${AIPERF_SERVER_URL}"

# Golden AL: golden_al_distribution/dsv41flash_dspark.yaml, thinking_on, five draft tokens.
# Accuracy evals keep real block rejection; throughput fixes acceptance to AL 3.51.
# Adaptive verification stays off in both modes on ROCm: it trims verification
# requests on device, which DeepseekV4IndexerBackend does not support, so the
# engine refuses to start with it enabled.
if [[ "${EVAL_ONLY}" == true ]]; then
    SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}'
else
    SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","synthetic_acceptance_length":3.51,"enable_adaptive_verification":false}'
fi

# KV offload to host DRAM. vllm-simple uses vLLM's SimpleCPUOffloadConnector; the
# per-rank byte budget divides the aggregate host-DRAM capacity across the TP/DP
# ranks. Identical prefixes must hash to identical block keys across ranks, so pin
# PYTHONHASHSEED. KV_OFFLOADING=none leaves OFFLOAD_ARGS empty (GPU-resident KV).
OFFLOAD_ARGS=()
if agentic_kv_offload_enabled; then
    check_env_vars KV_OFFLOAD_BACKEND
    case "$KV_OFFLOAD_BACKEND" in
      vllm-simple)
        require_agentic_kv_offload_backend vllm-simple
        CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / TP ))
        export PYTHONHASHSEED=42
        SIMPLE_LAZY_OFFLOAD="false"
        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":$CPU_BYTES_PER_RANK,\"lazy_offload\":$SIMPLE_LAZY_OFFLOAD}}"
        )
        echo "SimpleCPUOffloadConnector: ${CPU_BYTES_PER_RANK} B/rank x ${TP} ranks, lazy_offload=$SIMPLE_LAZY_OFFLOAD"
        ;;
      *)
        echo "Error: unsupported KV_OFFLOAD_BACKEND='$KV_OFFLOAD_BACKEND' (expected vllm-simple)" >&2
        exit 1
        ;;
    esac
fi

# DP-attention replaces pure TP with attention data parallelism: one DP rank per
# GPU (tensor-parallel-size 1, data-parallel-size TP) fronted by the router. EP
# stays off unless the arm requests it. The DP scheduler knobs and the per-rank
# max-num-seqs cap match the dsv4 MI355X arm.
PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
if [ "$DP_ATTENTION" = "true" ]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP")
fi

EP_ARGS=()
if [ "${EP_SIZE:-1}" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

DP_SCHED_ARGS=()
if [ "$DP_ATTENTION" = "true" ]; then
    DP_SCHED_ARGS=(
        --prefill-schedule-interval 8
        --long-prefill-token-threshold 16384
    )
    # The router balances CONC sessions across the DP ranks, so cap in-flight
    # sequences per rank at the outer concurrency.
    MAX_NUM_SEQS="$CONC"
fi

VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0 --port "$VLLM_BACKEND_PORT"
    "${PARALLEL_ARGS[@]}"
    "${EP_ARGS[@]}"
    "${DP_SCHED_ARGS[@]}"
    --language-model-only
    --tokenizer-mode deepseek_v41
    --tool-call-parser deepseek_v41 --enable-auto-tool-choice
    --reasoning-parser deepseek_v41
    --engram-config "$ENGRAM_CONFIG"
    # aiter, not aiter_triton_mxfp4_bf16: the plain name opens vLLM's full
    # priority list and the CK kernel at its head wins. CK quantizes
    # activations to FP8 internally and dispatches the a8w4 experts
    # (mfma_moe1_silu_mul_afp8_wfp4_bf16 / mfma_moe2_afp8_wfp4_bf16); the
    # Triton name forces the W4A16 _moe_gemm_a16w4 kernel instead.
    --moe-backend aiter
    --gpu-memory-utilization 0.9
    --speculative-config "$SPEC_CONFIG"
    --max-model-len 1048576
    --max-cudagraph-capture-size "$CAPTURE_SIZE"
    --max-num-batched-tokens "$BATCHED_TOKENS"
    # vllm-project/vllm#56227 added SWA bounded replay (default on) after the
    # eed1f3d0 pin and before this one. It pads the replayed tokens' slots in the
    # prefix-cacheable groups, but the window clamp it relies on landed in the
    # FlashInfer and FlashMLA kernels; the ROCm sparse SWA path only gained the
    # replay_start kwarg. On gfx950 every TP=2 and TP=4 point of run 35567570539
    # died with HSA_STATUS_ERROR_MEMORY_FAULT at the first prefix hit carrying a
    # replay start. Drop this once ROCm clamps too; prefix caching stays on.
    --no-swa-bounded-replay
    --disable-uvicorn-access-log
    "${OFFLOAD_ARGS[@]}"
)
if [[ -n "$MAX_NUM_SEQS" ]]; then
    VLLM_CMD+=(--max-num-seqs "$MAX_NUM_SEQS")
fi
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
SERVER_PID=""
ROUTER_PID=""
ROUTER_LOG="$RESULT_DIR/router.log"
cleanup_server() {
    local rc=$?
    trap - EXIT INT TERM
    stop_background_process_tree "$ROUTER_PID" "vLLM router"
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    exit "$rc"
}
trap cleanup_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$VLLM_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "$USE_VLLM_ROUTER" = "true" ]; then
    echo "Starting native vLLM router on port $PORT for $TP DP ranks..."
    vllm-router \
        --worker-urls "http://localhost:$VLLM_BACKEND_PORT" \
        --policy "$VLLM_ROUTER_POLICY" \
        --intra-node-data-parallel-size "$TP" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --prometheus-host 127.0.0.1 \
        --prometheus-port "$VLLM_ROUTER_METRICS_PORT" \
        --request-timeout-secs 14400 \
        --disable-retries > "$ROUTER_LOG" 2>&1 &
    ROUTER_PID=$!
    echo "Router PID: $ROUTER_PID"
    wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"
fi

if [[ "${EVAL_ONLY}" == true ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
