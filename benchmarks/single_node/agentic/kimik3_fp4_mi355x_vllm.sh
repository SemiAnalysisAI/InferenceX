#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace-replay benchmark for Kimi-K3 on AMD MI355X (gfx950) via vLLM.
# ROCm adaptation of benchmarks/single_node/agentic/kimik3_fp4_b300_vllm.sh:
# same AIPerf agentic harness + trace dataset, but the serve flags swap the
# NVIDIA-only bits (FlashInfer allreduce, fastsafetensors, Rust frontend,
# FLASHINFER MLA prefill) for the MI355X day-0 recipe (AITER).
#
# Attention: ROCM_AITER_MLA (asm persistent) — the fast path for K3's 12 heads/
# rank (TP8). REQUIRES a vLLM image with the asm-MLA non-divisor head support:
#   * vLLM #50578  (asm decode pad-to-16 for non-divisor small head counts)
#   * vLLM PR-A    (fp8 asm MLA *prefill* pad-to-16 + 16-head PS metadata)
# On stock vLLM these are absent and the asm path breaks for 12 heads. The asm
# prefill also needs ROCm/aiter #4452 (64-bit paged-KV offsets) for >4GB KV.
#
# Validated base image (carries the vLLM patches + AITER build above):
#   vllm/vllm-openai-rocm:nightly-cb8104839c141609d99f1254459ef3a4f1bd4263
# Pinned in configs/amd-master.yaml (kimik3-fp4-mi355x-vllm-agentic).
#
# Required env vars: MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION
# Optional: MAX_MODEL_LEN (replay --max-context-length), MODEL_PATH (pre-staged weights)

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION

if [ "$TP" -ne 8 ]; then
    echo "Error: Kimi-K3 on MI355X requires TP=8 (~1.5 TB checkpoint), got TP='$TP'" >&2
    exit 1
fi

# ---- Bootstrap the container from the pinned base image ----------------------
# The image pinned in configs/amd-master.yaml is the STOCK ROCm vLLM nightly
# (cb8104839c...); this idempotently applies the aiter rebuild + tuned GEMM CSV
# + triton 3.7.0 + vLLM ASM patches it needs (see apply_k3_fp4_fp8asm_dspark_patches.sh).
# No-op once markers are present. Set SKIP_K3_BOOTSTRAP=1 for a pre-baked image.
RECIPE_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ "${SKIP_K3_BOOTSTRAP:-0}" != "1" ]; then
    bash "$RECIPE_DIR/apply_k3_fp4_fp8asm_dspark_patches.sh"
fi

# Resolve weights: MODEL_PATH (pre-staged) else HF cache.
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi
    export MODEL_PATH="$MODEL"
fi
if [ -n "$ROCR_VISIBLE_DEVICES" ]; then export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"; fi

# ---- MI355X day-0 serving environment (AITER) -------------------------------
export VLLM_ROCM_USE_AITER=1
export GPU_ARCHS=gfx950
export AITER_SITUV2_A8W4=1
export AITER_BF16_FP8_MOE_BOUND=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export SAFETENSORS_FAST_GPU=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
# Outlast the AIPerf keep-alive pool so an inter-turn idle gap can't race a
# socket close into a warmup-abort (see the B300 recipe's note).
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

# ---- Resolve traces + install AIPerf (isolated venv) ------------------------
resolve_trace_source
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

# ---- KV offloading ----------------------------------------------------------
OFFLOAD_ARGS=()
case "${KV_OFFLOAD_BACKEND:-}" in
    "")
        require_agentic_kv_offload_none
        ;;
    native)
        require_agentic_kv_offload_backend native
        # The native tier backs the whole node-wide region with ONE /dev/shm
        # file, so the request must fit tmpfs, not just RAM. At dram-utilization
        # 0.60 TOTAL_CPU_DRAM_GB resolves above the default /dev/shm cap (50% of
        # RAM); madvise(MADV_POPULATE_WRITE) then takes a SIGBUS on the full
        # tmpfs and engine init dies with EFAULT ("Bad address") ~20 min in.
        KV_OFFLOADING_SIZE="${KV_OFFLOADING_SIZE:-$TOTAL_CPU_DRAM_GB}"
        SHM_FREE_GB=$(df -B1G --output=avail /dev/shm 2>/dev/null | tail -1 | tr -dc '0-9')
        SHM_BUDGET_GB=$(( ${SHM_FREE_GB:-0} * 9 / 10 ))
        if [ "${SHM_FREE_GB:-0}" -gt 0 ] && [ "$KV_OFFLOADING_SIZE" -gt "$SHM_BUDGET_GB" ]; then
            echo "KV offload: /dev/shm has ${SHM_FREE_GB} GiB free; clamping --kv-offloading-size ${KV_OFFLOADING_SIZE} -> ${SHM_BUDGET_GB} GiB"
            KV_OFFLOADING_SIZE="$SHM_BUDGET_GB"
        fi
        if [ "$KV_OFFLOADING_SIZE" -lt "${KV_OFFLOAD_MIN_GB:-64}" ]; then
            echo "Error: /dev/shm (${SHM_FREE_GB:-0} GiB free) cannot back a useful native KV tier; raise the container shm-size or switch this arm to kv-offload-backend vllm-simple" >&2
            exit 1
        fi
        OFFLOAD_ARGS=(
            --kv-offloading-size "$KV_OFFLOADING_SIZE"
            --kv-offloading-backend native
        )
        ;;
    vllm-simple)
        require_agentic_kv_offload_backend vllm-simple
        CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / TP ))
        export PYTHONHASHSEED=42
        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":${CPU_BYTES_PER_RANK},\"lazy_offload\":false}}"
        )
        ;;
    *)
        echo "Error: unsupported KV_OFFLOAD_BACKEND='$KV_OFFLOAD_BACKEND'" >&2
        exit 1
        ;;
esac

# Fixed max-num-seqs (not 2*CONC): 64 is the validated headroom for the ASM MLA
# fp8 prefill at uncapped (~1M) context — higher values re-introduce the
# activation-arena OOM that capped long context. Override via MAX_NUM_SEQS.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
# MNBT 16384 is mandated and shared with the -mtp sibling, so the base and
# DSpark curves are measured on the same prefill chunk size.
MNBT="${MNBT:-16384}"

# --- fp8 KV on K3 (dense MLA) via the ASM persistent MLA path ------------------
# K3's attention is DENSE MLA with 12 heads/rank at TP8 (a non-divisor of 16).
#   * The AITER *asm persistent* MLA does batched fp8 decode once the query
#     heads are padded 12->16 (vLLM #50578) and, for prefill, the fp8 PS asm
#     kernel + 16-head PS metadata (vLLM PR-A). This is faster than TRITON_MLA
#     and, with ROCm/aiter #4452 (64-bit paged-KV offsets), serves the FULL
#     native (~1M) context uncapped with no OOM (validated 470k/590k prefills).
#   * Do NOT set VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT (asm reads the native layout).
# KV_CACHE_DTYPE=auto falls back to bf16 KV on the same asm path.
KVDTYPE_ARGS=(
    --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}"
    --attention-backend "${ATTENTION_BACKEND:-ROCM_AITER_MLA}"
)

export VLLM_ROCM_USE_AITER_MOE=1

# KV-cache memory pin, same rationale as the -mtp sibling. At gpu-mem 0.95 with
# TP8 K3-fp4 the auto-sizer leaves KV at ~73 GiB against ~198 GiB of weights and
# ~10.5 GiB of cudagraphs, i.e. under 4 GiB spare of the 288 GiB device. The
# first long-context prefill plus a runtime Triton JIT then exhausts the device
# and HSA aborts with "Available Free mem : 0 MB". Prefix caching only needs the
# ~64k prefix stored once (~6 GiB), so pin KV to 32 GiB (~2M tokens) and leave
# the rest as activation headroom. Do NOT touch gpu-mem.
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-34359738368}"
KVMEM_ARG=(); [ -n "$KV_CACHE_MEMORY" ] && KVMEM_ARG=(--kv-cache-memory "$KV_CACHE_MEMORY")

echo "Starting vllm server (MI355X/AITER, DSV4-agentic-derived config)..."
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0 --port "$PORT"
    --tensor-parallel-size "$TP"
    --async-scheduling
    --distributed-executor-backend mp
    --gpu-memory-utilization 0.95
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MNBT"
    --trust-remote-code
    --load-format auto
    --moe-backend auto
    --mm-encoder-tp-mode data
    "${KVDTYPE_ARGS[@]}"
    "${KVMEM_ARG[@]}"
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'
    --enable-prefix-caching
    # native hybrid KV (MLA + KDA) — no padding; the fix for the capture fault.
    --no-disable-hybrid-kv-cache-manager
    --reasoning-parser kimi_k3
    --tool-call-parser kimi_k3
    --enable-auto-tool-choice
    --disable-uvicorn-access-log
    "${OFFLOAD_ARGS[@]}"
)
# Optional per-seq context cap (default: model native, like the DSV4 recipe).
if [ -n "${MAX_MODEL_LEN:-}" ]; then VLLM_CMD+=(--max-model-len "$MAX_MODEL_LEN"); fi
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"; printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# The agentic single-node eval selector groups by (model, runner, framework,
# precision) and eval-marks exactly one entry in that group -- which can be THIS
# base arm, not the -dspark arm. So the accuracy gate may drive this script with
# EVAL_ONLY=true; run the GSM8K eval then, or the gate replays throughput and gets
# no score. Mirrors kimik3_fp4_b300_vllm.sh.
if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi

# cleanup: free the GPU (orphaned TP workers otherwise hold VRAM)
[[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true
pkill -9 -f "/usr/local/bin/vll[m]" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
pkill -9 -f "multiprocessing.spawn" 2>/dev/null || true
for _ in $(seq 1 30); do pgrep -f "EngineCore|multiprocessing.spawn" >/dev/null 2>&1 || break; sleep 2; done
set +x
