#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Qwen3.5 FP8 on H200 with SGLang MTP.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../benchmark_lib.sh"

check_env_vars \
    MODEL MODEL_PREFIX TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR \
    DURATION EP_SIZE EVAL_ONLY SPEC_DECODING

if ! require_agentic_kv_offload_backend hicache; then
    echo "Error: the H200 MTP recipe requires KV_OFFLOADING=dram with HiCache" >&2
    exit 1
fi

if [ "$TP" -ne 8 ] || [ "$EP_SIZE" -ne 1 ] || [ "$SPEC_DECODING" != "mtp" ]; then
    echo "Error: the H200 MTP recipe requires TP=8, EP_SIZE=1, and SPEC_DECODING=mtp" >&2
    exit 1
fi

if [ "$TOTAL_CPU_DRAM_GB" -lt 1200 ]; then
    echo "Error: TP8 requires at least 1200 GB host DRAM; generated budget is ${TOTAL_CPU_DRAM_GB} GB" >&2
    exit 1
fi

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
nvidia-smi

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_with_subagents_256k
resolve_trace_source
install_agentic_deps

agentic_pip_install --no-deps --force-reinstall flashinfer_python==0.6.17
agentic_pip_install \
    --no-deps --force-reinstall flashinfer-cubin==0.6.17 \
    --index-url https://flashinfer.ai/whl
agentic_pip_install \
    --no-deps --force-reinstall flashinfer-jit-cache==0.6.17+cu130 \
    --index-url https://flashinfer.ai/whl/cu130

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

export TORCH_CUDA_ARCH_LIST=9.0
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer-cache
export SGL_ENABLE_JIT_DEEPGEMM=false
export SGLANG_ENABLE_FLASHINFER_GEMM=true
export SGLANG_ENABLE_SPEC_V2=1

SPEC_ARGS=(
    --speculative-algorithm NEXTN
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
)

if [ "$EVAL_ONLY" != "true" ]; then
    export SGLANG_SIMULATE_ACC_LEN=3.39
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi

CACHE_ARGS=(
    --page-size 64
    --enable-hierarchical-cache
    --hicache-ratio 0.9
    --hicache-io-backend kernel
    --hicache-mem-layout page_first_direct
    --hicache-write-policy write_back
)

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tensor-parallel-size "$TP"
    --data-parallel-size 1
    --expert-parallel-size "$EP_SIZE"
    --quantization fp8
    --kv-cache-dtype fp8_e4m3
    --mamba-ssm-dtype bfloat16
    --mamba-scheduler-strategy extra_buffer
    --mamba-track-interval 8192
    --attention-backend flashinfer
    --cuda-graph-max-bs 32
    --max-running-requests 128
    --max-prefill-tokens 16384
    --chunked-prefill-size 16384
    --mem-fraction-static 0.78
    --max-mamba-cache-size 360
    --allow-auto-truncate
    --stream-interval 50
    --scheduler-recv-interval 10
    --tokenizer-worker-num 6
    --enable-cache-report
    --enable-symm-mem
    --enable-metrics
    "${CACHE_ARGS[@]}"
    "${SPEC_ARGS[@]}"
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

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

{
    echo "=== SGLANG_* env vars at launch ==="
    env | grep -E '^SGLANG_' | sort
    echo "==================================="
} > "$SERVER_LOG"

"${SGLANG_CMD[@]}" >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "$EVAL_ONLY" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --apply-chat-template"
    REPLAY_CMD+=" --server-metrics http://localhost:$PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
