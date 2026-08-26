#!/usr/bin/env bash
set -euo pipefail
set -x
source "$(dirname "$0")/../../benchmark_lib.sh"
wait_for_amd_gpu_clean

export AIPERF_EXPERIMENTAL_FAST=0
export AIPERF_WARMUP_REQUESTS_PER_LANE=1
check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE

DP_SIZE=1
export DP_SIZE
TOTAL_RANKS=$(( TP * DP_SIZE ))

if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
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
resolve_trace_source
install_agentic_deps

DCP_SIZE=8
export DCP_SIZE

export SKIP_PATCH_OPUS_ROWS=1
bash "$(dirname "$0")/apply_kimi_k3_patches.sh" || true

export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
export VLLM_ROCM_USE_AITER=1
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
export AITER_BF16_FP8_MOE_BOUND=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export GPU_ARCHS=gfx950
export VLLM_ROCM_USE_AITER_MOE=1
export AITER_SITUV2_A8W4=1
export HSA_NO_SCRATCH_RECLAIM=1
export VLLM_K3_KDA_SAFE_STAGES=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

export VLLM_ENGINE_READY_TIMEOUT_S=7200
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export PYTHONNOUSERSITE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"
SERVER_PID=""

cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export PYTHONHASHSEED=42

# KV offload. TOTAL_CPU_DRAM_GB is the aggregate host-DRAM budget the matrix
# generator derives from dram-utilization and the runner's available CPU DRAM;
# per the agentic README it must be consumed as given, never replaced with a
# model-specific constant. Worth 3.3x on the non-DCP path (T92 vs T64), so the
# absence of this block is not a neutral simplification.
OFFLOAD_ARGS=()
if agentic_kv_offload_enabled; then
    case "${KV_OFFLOAD_BACKEND:-}" in
      vllm-simple)
        require_agentic_kv_offload_backend "$KV_OFFLOAD_BACKEND"
        CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / TOTAL_RANKS ))
        # Identical prefixes must hash to identical block keys across ranks.
        export PYTHONHASHSEED=42
        SIMPLE_LAZY_OFFLOAD="${SIMPLE_LAZY_OFFLOAD:-false}"
        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":$CPU_BYTES_PER_RANK,\"lazy_offload\":$SIMPLE_LAZY_OFFLOAD}}"
        )
        echo "SimpleCPUOffloadConnector: ${CPU_BYTES_PER_RANK} B/rank x ${TOTAL_RANKS} ranks, lazy_offload=$SIMPLE_LAZY_OFFLOAD"
        ;;
      *)
        echo "KV offload requested (KV_OFFLOADING=$KV_OFFLOADING) but backend '${KV_OFFLOAD_BACKEND:-unset}' is not handled here" >&2
        ;;
    esac
fi

KV_CACHE_DTYPE=fp8
CP_ARGS=(
    --decode-context-parallel-size "$DCP_SIZE"
    --dcp-comm-backend a2a
    --attention-backend ROCM_AITER_MLA
    --cp-kv-cache-interleave-size 1
)
export VLLM_USE_DIRECT_DCP_A2A=0
export VLLM_USE_DIRECT_DCP_Q_GATHER=0
export VLLM_USE_DIRECT_DCP_KV_GATHER=0
export VLLM_ALLOW_DCP_FULL_CUDAGRAPH=1
export VLLM_DCP_Q_REPLICATE=1
export VLLM_ROCM_USE_AITER_MLA=1
export AITER_DISABLE_FMHA_OPUS=1
export VLLM_ROCM_USE_AITER_MLA=1
export AITER_DISABLE_FMHA_OPUS=1

SPEC_ARGS=()

CHUNKED_PREFILL_ARGS=(--max-num-batched-tokens 8192)
ASYNC_SCHED_ARGS=(--no-async-scheduling)
MLA_PREFILL_ARGS=(--attention-config "{\"mla_prefill_backend\":\"ROCM_AITER_FA\"}")

MAX_NUM_SEQS=80
CUDAGRAPH_CAPTURE_SIZES="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80"
MAX_CUDAGRAPH_CAPTURE_SIZE=80
CUDAGRAPH_MODE=FULL_AND_PIECEWISE
COMPILATION_CONFIG_ARGS=(--compilation-config "{\"mode\":3,\"cudagraph_mode\":\"$CUDAGRAPH_MODE\",\"max_cudagraph_capture_size\":$MAX_CUDAGRAPH_CAPTURE_SIZE,\"custom_ops\":[\"+fused_rms_norm_gated\"],\"cudagraph_capture_sizes\":[$CUDAGRAPH_CAPTURE_SIZES]}")

GPU_MEM_UTIL=0.9

VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --moe-backend auto
    --tensor-parallel-size "$TP"
    --load-format fastsafetensors
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --language-model-only
    --max-num-seqs "$MAX_NUM_SEQS"
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --max-model-len 1048576
    --enable-prefix-caching
    --enable-prompt-tokens-details
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    "${CHUNKED_PREFILL_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
    "${CP_ARGS[@]}"
    "${SPEC_ARGS[@]}"
    "${ASYNC_SCHED_ARGS[@]}"
    "${MLA_PREFILL_ARGS[@]}"
    "${COMPILATION_CONFIG_ARGS[@]}"
)

printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"

"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
