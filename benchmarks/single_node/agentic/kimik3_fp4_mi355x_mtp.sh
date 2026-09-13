#!/usr/bin/env bash
set -euo pipefail
set -x
source "$(dirname "$0")/../../benchmark_lib.sh"
wait_for_amd_gpu_clean

export EVAL_ONLY="${EVAL_ONLY:-false}"
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
resolve_trace_source
install_agentic_deps

export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MLA=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION="${VLLM_ROCM_QUICK_REDUCE_QUANTIZATION:-NONE}"

export AITER_SITUV2_A8W4=1
export AITER_FLYDSL_STAGE2_FP8="${AITER_FLYDSL_STAGE2_FP8:-1}"
export AITER_BF16_FP8_MOE_BOUND=0
export AITER_DISABLE_FMHA_OPUS=1
export SAFETENSORS_FAST_GPU=1
export GPU_ARCHS=gfx950
export HSA_NO_SCRATCH_RECLAIM=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_K3_KDA_SAFE_STAGES=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=42

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

SPEC_ARGS=()
SPEC_ROWS=1
KDA_ARGS=()
case "$CONC" in
    1|2|4|8|10|12|14)
        DCP_SIZE="${DCP_SIZE:-1}"
        if [ "$CONC" -eq 1 ]; then SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-4}"
        else SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-3}"; fi
        case "$SPEC_NUM_TOKENS" in
            1) SYNTHETIC_ACCEPT_LEN=1.85 ;;
            2) SYNTHETIC_ACCEPT_LEN=2.51 ;;
            3) SYNTHETIC_ACCEPT_LEN=3.00 ;;
            4) SYNTHETIC_ACCEPT_LEN=3.36 ;;
            5) SYNTHETIC_ACCEPT_LEN=3.62 ;;
            6) SYNTHETIC_ACCEPT_LEN=3.75 ;;
            7) SYNTHETIC_ACCEPT_LEN=3.84 ;;
            8) SYNTHETIC_ACCEPT_LEN=4.00 ;;
            *) echo "[spec] no golden AL for k=$SPEC_NUM_TOKENS" >&2; exit 1 ;;
        esac
        DRAFT_KV_DTYPE="${DRAFT_KV_DTYPE:-fp8}"
        SPEC_BASE="\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"kv_cache_dtype\":\"$DRAFT_KV_DTYPE\",\"draft_sample_method\":\"probabilistic\""
        if [ "${EVAL_ONLY:-false}" = "true" ]; then
            SPEC_ARGS=(--speculative-config "{$SPEC_BASE,\"rejection_sample_method\": \"block\"}")
            echo "MTP: k=$SPEC_NUM_TOKENS LIVE block rejection (accuracy gate) draft_kv=$DRAFT_KV_DTYPE"
        else
            SPEC_ARGS=(--speculative-config "{$SPEC_BASE,\"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}")
            echo "MTP: k=$SPEC_NUM_TOKENS synthetic_accept=$SYNTHETIC_ACCEPT_LEN draft_kv=$DRAFT_KV_DTYPE"
        fi
        SPEC_ROWS=$(( SPEC_NUM_TOKENS + 1 ))
        KDA_ARGS=(--additional-config '{"kda_prefill_backend":"triton"}')
        MAX_NUM_SEQS="${MAX_NUM_SEQS:-$(( CONC > 2 ? CONC : 2 ))}"
        if [ "$CONC" -eq 1 ]; then MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"
        else MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"; fi
        ;;
    *)
        DCP_SIZE="${DCP_SIZE:-8}"
        if [ "$CONC" -gt 64 ]; then MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-24576}"
        else MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"; fi
        if [ "$CONC" -lt 72 ]; then MAX_NUM_SEQS="${MAX_NUM_SEQS:-$(( CONC * 14 / 10 ))}"
        elif [ "$CONC" -eq 72 ]; then MAX_NUM_SEQS="${MAX_NUM_SEQS:-96}"
        else MAX_NUM_SEQS="${MAX_NUM_SEQS:-112}"; fi
        ;;
esac
export DCP_SIZE

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.89}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"

LADDER=$(( MAX_NUM_SEQS * SPEC_ROWS ))
CUDAGRAPH_CAPTURE_SIZES=$(seq -s, 1 "$LADDER")
COMPILATION_CONFIG_ARGS=(--compilation-config "{\"mode\":3,\"cudagraph_mode\":\"$CUDAGRAPH_MODE\",\"max_cudagraph_capture_size\":$LADDER,\"custom_ops\":[\"+fused_rms_norm_gated\"],\"cudagraph_capture_sizes\":[$CUDAGRAPH_CAPTURE_SIZES]}")

CP_ARGS=(--attention-backend ROCM_AITER_MLA)
if [ "$DCP_SIZE" -gt 1 ]; then
    CP_ARGS+=(--decode-context-parallel-size "$DCP_SIZE" --dcp-comm-backend a2a --cp-kv-cache-interleave-size 1)
fi

OFFLOAD_ARGS=()
if agentic_kv_offload_enabled; then
    CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / TOTAL_RANKS ))
    OFFLOAD_ARGS=(--kv-transfer-config "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":$CPU_BYTES_PER_RANK,\"lazy_offload\":false}}")
fi

EP_ARGS=()
if [ "${EP_SIZE:-1}" -gt 1 ]; then EP_ARGS=(--enable-expert-parallel); fi

echo "[cfg] conc=$CONC dcp=$DCP_SIZE gmu=$GPU_MEM_UTIL mns=$MAX_NUM_SEQS ladder=1..$LADDER spec_rows=$SPEC_ROWS chunk=$MAX_BATCHED_TOKENS cudagraph=$CUDAGRAPH_MODE offload=${KV_OFFLOADING:-none}"

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
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
    --max-model-len 1048576
    --kv-cache-dtype fp8
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --enable-prefix-caching
    --enable-prompt-tokens-details
    --no-async-scheduling
    --attention-config '{"mla_prefill_backend":"ROCM_AITER_FA"}'
    "${OFFLOAD_ARGS[@]}"
    "${CP_ARGS[@]}"
    "${EP_ARGS[@]}"
    "${SPEC_ARGS[@]}"
    "${KDA_ARGS[@]}"
    "${COMPILATION_CONFIG_ARGS[@]}"
)

printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"

"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"



wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
