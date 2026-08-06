#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K3 MXFP4 on MI355X using vLLM,
# DSpark speculative decoding, and LMCache MP host-DRAM KV offload.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, KV_OFFLOAD_BACKEND,
#   TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION, EP_SIZE, PORT

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL TP CONC KV_OFFLOADING KV_OFFLOAD_BACKEND TOTAL_CPU_DRAM_GB \
    RESULT_DIR DURATION EP_SIZE PORT

if [ "$TP" -ne 8 ]; then
    echo "Error: Kimi-K3 MXFP4 requires TP=8 on MI355X; got TP=$TP." >&2
    exit 1
fi

if [ "$EP_SIZE" -gt 1 ]; then
    echo "Error: this Kimi-K3 DSpark recipe supports pure TP8 only; got EP_SIZE=$EP_SIZE." >&2
    exit 1
fi

require_agentic_kv_offload_backend lmcache

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

# Do not start the large Kimi-K3 load while a previous Slurm job is still
# releasing VRAM on the allocated MI355X GPUs.
wait_for_amd_gpu_clean

# The cluster launcher mounts its persistent Hugging Face cache here. MODEL_PATH
# may instead point at a pre-staged snapshot.
export HF_HUB_CACHE="${HF_HUB_CACHE:-/models/huggingface_hub}"
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

# Environment used by the validated Kimi-K3 + DSpark + LMCache image.
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=42
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MLA=1
export VLLM_ROCM_USE_SKINNY_GEMM=0
export AITER_SITUV2_A8W4=1
export AITER_BF16_FP8_MOE_BOUND=0
export VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1200}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
export HSA_NO_SCRATCH_RECLAIM=1
export SAFETENSORS_FAST_GPU=1

SERVER_LOG="$RESULT_DIR/server.log"
LMCACHE_LOG="$RESULT_DIR/lmcache_server.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
LMCACHE_PID=""

cleanup_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    stop_background_process_tree "$LMCACHE_PID" "LMCache server" 30
    exit "$exit_code"
}
trap cleanup_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_lmcache_ready() {
    { set +x; } 2>/dev/null
    local attempts="${LMCACHE_READY_ATTEMPTS:-1800}"
    local paths=(/healthcheck /health /v1/health /status /)
    local i path

    for ((i = 1; i <= attempts; i++)); do
        for path in "${paths[@]}"; do
            if curl --output /dev/null --silent --fail \
                    "http://127.0.0.1:${LMCACHE_HTTP_PORT}${path}"; then
                echo "LMCache server healthy after ${i}s (endpoint ${path})"
                set -x
                return 0
            fi
        done

        if ! kill -0 "$LMCACHE_PID" 2>/dev/null; then
            echo "LMCache server exited before becoming healthy. Log follows:" >&2
            tail -200 "$LMCACHE_LOG" >&2 || true
            exit 1
        fi

        if ((i % 60 == 0)); then
            echo "Still waiting for LMCache (${i}s/${attempts}s, L1=${LMCACHE_L1_SIZE_GB}GB)"
        fi
        sleep 1
    done

    echo "Timed out waiting for LMCache on port $LMCACHE_HTTP_PORT. Log follows:" >&2
    tail -200 "$LMCACHE_LOG" >&2 || true
    exit 1
}

# This is the pool size validated on the MI355X host. Keep it overridable for
# hosts with a different /dev/shm allocation, but never exceed the matrix's DRAM
# budget.
LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-906}"
if [ "$LMCACHE_L1_SIZE_GB" -gt "$TOTAL_CPU_DRAM_GB" ]; then
    echo "Error: LMCache L1=${LMCACHE_L1_SIZE_GB}GB exceeds the generated DRAM budget ${TOTAL_CPU_DRAM_GB}GB." >&2
    exit 1
fi

LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
LMCACHE_MP_PORT="${LMCACHE_MP_PORT:-6000}"
LMCACHE_CMD=(
    lmcache server
    --host 127.0.0.1
    --port "$LMCACHE_MP_PORT"
    --http-host 127.0.0.1
    --http-port "$LMCACHE_HTTP_PORT"
    --l1-size-gb "$LMCACHE_L1_SIZE_GB"
    --l1-init-size-gb 20
    --l1-read-ttl-seconds 7200
    --chunk-size 1536
    --max-workers 8
    --eviction-trigger-watermark 0.85
    --eviction-ratio 0.10
    --eviction-policy LRU
)
printf '%q ' "${LMCACHE_CMD[@]}" > "$RESULT_DIR/lmcache_command.txt"
printf '\n' >> "$RESULT_DIR/lmcache_command.txt"
"${LMCACHE_CMD[@]}" > "$LMCACHE_LOG" 2>&1 &
LMCACHE_PID=$!
echo "LMCache server PID: $LMCACHE_PID"
wait_for_lmcache_ready

# AgentX concurrency counts live session trees, not individual requests.
# Subagent fan-out can push instantaneous request concurrency above CONC, so
# leave 2x headroom rather than clipping those bursts at the scheduler.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$((2 * CONC))}"
SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-3}"
SPEC_DRAFT_MODEL="${SPEC_DRAFT_MODEL:-Inferact/Kimi-K3-DSpark}"
SPEC_DRAFT_MODEL_PATH="${SPEC_DRAFT_MODEL_PATH:-$SPEC_DRAFT_MODEL}"
if [[ "$SPEC_DRAFT_MODEL_PATH" == "$SPEC_DRAFT_MODEL" ]]; then
    hf download "$SPEC_DRAFT_MODEL"
fi
SPEC_CONFIG="{\"model\":\"${SPEC_DRAFT_MODEL_PATH}\",\"num_speculative_tokens\":${SPEC_NUM_TOKENS},\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\"}"
KV_TRANSFER_CONFIG="{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://127.0.0.1\",\"lmcache.mp.port\":${LMCACHE_MP_PORT}}}"

VLLM_CMD=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --moe-backend auto
    --tensor-parallel-size "$TP"
    --load-format auto
    --gpu-memory-utilization 0.85
    --mm-encoder-tp-mode data
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens 3000
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --enable-prefix-caching
    --mamba-cache-mode align
    --kv-cache-dtype fp8
    # Keep eager execution until the patched graph path is validated in this image.
    --enforce-eager
    --speculative-config "$SPEC_CONFIG"
    --kv-transfer-config "$KV_TRANSFER_CONFIG"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "vLLM server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
