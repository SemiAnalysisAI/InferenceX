#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4.1-Flash on MI355X: native DSpark, GPU-resident KV.
# https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml
source "$(dirname "$0")/../../benchmark_lib.sh"
check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION
check_env_vars EVAL_ONLY
require_agentic_kv_offload_none
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
# rather than the VLLM_PLE_CPU_OFFLOAD default. Offload on every arm, as every
# NVIDIA DSv4.1-Flash arm has since #2963. Measured on gfx950 at TP=4, batched
# 16384: resident leaves 37.96 GiB of KV (14.15x max concurrency at 1M context),
# offloaded leaves 84.54 GiB (31.52x).
ENGRAM_CONFIG='{"cpu_offload":true}'

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
# Cap in-flight sequences at the shape graph capture already covers, so the
# largest decode batch stays on a captured graph. This also drops max_num_seqs
# from the MI355X API-server default of 1024, which sized scheduler state for
# four times the sequences this sweep can actually run.
MAX_NUM_SEQS="$GRAPH_NUM_SEQS"

# The sparse-attention indexer and its companion per-rank buffers scale with
# --max-num-batched-tokens at roughly 4.4 MiB per token, measured on gfx950.
# The upstream 16384 is what separates a KV pool that survives concurrency 64
# from one that collapses: at TP=2 it leaves 20.06 GiB (7.48x), and TP=2 c64 of
# run 35574132719 fell to a 17.6% prefix cache hit rate, 187 s TTFT and 150
# tok/s. At 4096 the same arm keeps 79.34 GiB (39.44x). TP=4 has twice the
# per-rank room, so 8192 is enough there: 121.03 GiB (54.15x), against the B300
# arm's 132.48 GiB (60.17x). B300 runs 8192 at TP=4 and the Blackwell TP=2 arms
# run 4096 (#3320, #3321).
if (( TP == 2 )); then
    BATCHED_TOKENS=4096
else
    BATCHED_TOKENS=8192
fi

# Use the runner-specific port assigned by launch_mi355x-amds.sh.
export AIPERF_SERVER_URL="http://localhost:${PORT}"
export AIPERF_SERVER_METRICS_URLS="${AIPERF_SERVER_URL}/metrics"
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
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0 --port "$PORT" --tensor-parallel-size "$TP"
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
    --max-num-seqs "$MAX_NUM_SEQS"
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
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
SERVER_PID=""
cleanup_server() {
    local rc=$?
    trap - EXIT INT TERM
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    exit "$rc"
}
trap cleanup_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "${EVAL_ONLY}" == true ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
