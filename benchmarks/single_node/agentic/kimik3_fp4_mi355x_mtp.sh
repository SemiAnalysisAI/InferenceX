#!/usr/bin/env bash
set -euo pipefail
set -x
source "$(dirname "$0")/../../benchmark_lib.sh"
wait_for_amd_gpu_clean

export EVAL_ONLY="${EVAL_ONLY:-false}"
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

if [ "$CONC" -le 4 ]; then LOW_CONC_INTERACTIVE=1; else LOW_CONC_INTERACTIVE=0; fi
if [ -n "${DCP_SIZE:-}" ]; then
    DCP_SOURCE=matrix
else
    if [ "$CONC" -le 4 ]; then DCP_SIZE=1; else DCP_SIZE=8; fi
    DCP_SOURCE=conc-fallback
fi
export DCP_SIZE
echo "[dcp] size=$DCP_SIZE source=$DCP_SOURCE conc=$CONC"

export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
export VLLM_ROCM_USE_AITER=1
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
export AITER_BF16_FP8_MOE_BOUND=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export GPU_ARCHS=gfx950
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION="${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION:-NONE}"
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

OFFLOAD_ARGS=()
if agentic_kv_offload_enabled; then
    case "${KV_OFFLOAD_BACKEND:-}" in
      vllm-simple)
        require_agentic_kv_offload_backend "$KV_OFFLOAD_BACKEND"
        CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / TOTAL_RANKS ))
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
EP_ARGS=()
if [ "${EP_SIZE:-1}" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
    echo "EP: expert parallelism ON (EP_SIZE=$EP_SIZE)"
fi

CP_ARGS=(--attention-backend ROCM_AITER_MLA)
if [ "$DCP_SIZE" -gt 1 ]; then
    CP_ARGS+=(
        --decode-context-parallel-size "$DCP_SIZE"
        --dcp-comm-backend a2a
        --cp-kv-cache-interleave-size 1
    )
    export VLLM_USE_DIRECT_DCP_A2A=0
    export VLLM_USE_DIRECT_DCP_Q_GATHER=0
    export VLLM_USE_DIRECT_DCP_KV_GATHER=0
    export VLLM_ALLOW_DCP_FULL_CUDAGRAPH=1
    export VLLM_DCP_Q_REPLICATE=1
    echo "[dcp] ENABLED size=$DCP_SIZE backend=a2a interleave=1"
else
    echo "[dcp] DISABLED -- no DCP args, no DCP env"
fi
export VLLM_USE_DIRECT_DCP_A2A=0
export VLLM_USE_DIRECT_DCP_Q_GATHER=0
export VLLM_USE_DIRECT_DCP_KV_GATHER=0
export VLLM_ALLOW_DCP_FULL_CUDAGRAPH=1
export VLLM_DCP_Q_REPLICATE=1
export VLLM_ROCM_USE_AITER_MLA=1
export AITER_DISABLE_FMHA_OPUS=1
export VLLM_ROCM_USE_AITER_MLA=1
export AITER_DISABLE_FMHA_OPUS=1

SPEC_ENABLE="${SPEC_DECODING:-}"
case "${RESULT_FILENAME:-}" in *_spec-mtp_*) SPEC_ENABLE=mtp;; esac
if [ "$LOW_CONC_INTERACTIVE" != "1" ]; then SPEC_ENABLE=""; fi
SPEC_ARGS=()
if [ "$SPEC_ENABLE" = "mtp" ]; then
    SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-8}"
    case "$SPEC_NUM_TOKENS" in
        8) SYNTHETIC_ACCEPT_LEN=4.00 ;;
        *) echo "[spec] no golden AL wired for num_speculative_tokens=$SPEC_NUM_TOKENS; take it from golden_al_distribution/kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method.yaml and add the case" >&2; exit 1 ;;
    esac
    SPEC_ARGS=(
        --speculative-config
        "{\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
    )
    echo "MTP: speculative decoding ON (k=$SPEC_NUM_TOKENS, synthetic accept=$SYNTHETIC_ACCEPT_LEN)"
fi

CHUNKED_PREFILL_ARGS=(--max-num-batched-tokens 8192)
if [ "${ASYNC_SCHED:-0}" = "1" ]; then
    ASYNC_SCHED_ARGS=(--async-scheduling)
else
    ASYNC_SCHED_ARGS=(--no-async-scheduling)
fi
MLA_PREFILL_ARGS=(--attention-config "{\"mla_prefill_backend\":\"ROCM_AITER_FA\"}")

MAX_NUM_SEQS=$(( CONC * 2 ))
if [ "$MAX_NUM_SEQS" -lt 8 ]; then MAX_NUM_SEQS=8; fi
if [ "$MAX_NUM_SEQS" -gt 24 ]; then MAX_NUM_SEQS=80; fi

SPEC_ROWS=1
if [ "${#SPEC_ARGS[@]}" -gt 0 ]; then SPEC_ROWS=$(( SPEC_NUM_TOKENS + 1 )); fi
MAX_CUDAGRAPH_CAPTURE_SIZE=$(( MAX_NUM_SEQS * SPEC_ROWS ))
CUDAGRAPH_CAPTURE_SIZES=$(seq -s, 1 "$MAX_CUDAGRAPH_CAPTURE_SIZE")
echo "graphs: dense ladder 1..$MAX_CUDAGRAPH_CAPTURE_SIZE (mns=$MAX_NUM_SEQS x $SPEC_ROWS rows), DCP=$DCP_SIZE"
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
    "${EP_ARGS[@]}"
    "${SPEC_ARGS[@]}"
    "${ASYNC_SCHED_ARGS[@]}"
    "${MLA_PREFILL_ARGS[@]}"
    "${COMPILATION_CONFIG_ARGS[@]}"
)

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

for _a in CP_ARGS SPEC_ARGS CHUNKED_PREFILL_ARGS ASYNC_SCHED_ARGS MLA_PREFILL_ARGS OFFLOAD_ARGS COMPILATION_CONFIG_ARGS; do
    grep -q "\${$_a\[@\]}" "$0" || echo "[orphan-check] WARNING: $_a is built but never passed to VLLM_CMD" >&2
done

printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi