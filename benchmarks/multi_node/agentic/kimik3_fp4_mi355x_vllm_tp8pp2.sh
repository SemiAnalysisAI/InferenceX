#!/usr/bin/env bash
# Kimi-K3 MXFP4 MI355X aggregated multinode PP2 agentic benchmark (2 nodes).
# Supports TP8×PP2 (16 GPUs) or TP4×PP2 (8 GPUs, 4 per node).
#
# CI e2e entry: configs/amd-master.yaml kimik3-fp4-mi355x-vllm-agentic-tp8pp2
#   → launch_mi355x-amds.sh → kimik3_fp4_mi355x_vllm.sh → kimik3_agg_pp_job.slurm
#   → this script (per-rank container). Local smoke still uses
#   experimental/kimik3-v4/run_kimik3_tp8pp2_smoke_g06_g17.sh.
#
# MI355X adaptation of the B200 srt-slurm recipe
# benchmarks/multi_node/srt-slurm-recipes/vllm/kimi-k3/agentic/
# agg-b200-tp8pp2-retention0-agentic.yaml
#
# Serving profile aligned with single-node kimik3_fp4_mi355x_mtp.sh /
# configs/amd-master.yaml kimik3-fp4-mi355x-vllm-agentic-mtp:
#   - ROCm image + apply_k3_container_patches.sh
#   - fp8 KV, dram + vllm-simple CPU offload (default), FULL_AND_PIECEWISE cudagraph
#   - max-num-seqs 20, max-model-len 1M, gpu-mem 0.84 with offload
#   - AITER MoE SITUV2 A8W4 + asm MLA (no offline aiter GEMM tune in CI)
#   - optional DSpark via SPEC_DECODE=true (CI tp8pp2 default remains none)
#
# Topology:
#   - TP8 x PP2, plain TP (EP off), aggregated (no P/D disagg)
#
# Required env (agentic):
#   MODEL, MODEL_PREFIX, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR,
#   DURATION, RESULT_FILENAME
# Multinode required:
#   NODE_RANK (0 or 1), MASTER_ADDR, MASTER_PORT, NNODES (default 2)
# Optional:
#   TP (default 8), PP (default 2), PORT, KV_OFFLOAD_BACKEND (vllm-simple when dram),
#   GPU_MEM_UTIL, ENFORCE_EAGER (default false), MAX_NUM_SEQS (default 20),
#   MAX_NUM_BATCHED_TOKENS (default 8192; caps chunked-prefill M to avoid OOM)
#   RUN_EVAL=true → after agentic profiling, run lm-eval (gsm8k) against the warm engine
#   SPEC_DECODE=true → DSpark (Inferact/Kimi-K3-DSpark, TRITON_MLA), same as
#                      kimik3_fp4_mi355x_mtp.sh; SPEC_NUM_TOKENS default 2
#   DRAFT_MODEL_PATH → draft weights (default Inferact/Kimi-K3-DSpark or /hf_cache/…)
#   ASYNC_SCHEDULING: true|1 → --async-scheduling;
#                     false|0 → --no-async-scheduling;
#                     unset/auto → omit (engine default; matches prior smoke)
#
set -euo pipefail
set -x

source "$(dirname "$0")/../../benchmark_lib.sh"

NODE_RANK="${NODE_RANK:-${SLURM_PROCID:-}}"
NNODES="${NNODES:-2}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29500}"
TP="${TP:-8}"
PP="${PP:-2}"
PORT="${PORT:-8000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-20}"
# B200 agg-tp8pp2 recipe + prior MI355X OOM (M≈336k): keep prefill chunks bounded.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-44}"
# Lower bound of the capture list. The aiter MXFP4 fused_moe picks M-specific
# kernel variants below M≈8 (M=2 selects _w4_gui_kw2_fp8 +
# t32x256x256_reduce_persist) and those fault with an illegal device access
# during capture on gfx950; the tuned GEMM CSVs carry no M<=8 rows either.
# Raise this to skip those sizes — batches below the smallest captured size are
# padded up to it, so only the capture list shrinks.
MIN_CUDAGRAPH_CAPTURE_SIZE="${MIN_CUDAGRAPH_CAPTURE_SIZE:-1}"

check_env_vars MODEL MODEL_PREFIX CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION RESULT_FILENAME

if [[ "$PP" -ne 2 ]]; then
    echo "Error: this recipe requires PP=2 (got PP=$PP)" >&2
    exit 1
fi
if [[ "$TP" -ne 4 && "$TP" -ne 8 ]]; then
    echo "Error: supported TP is 4 or 8 (got TP=$TP)" >&2
    exit 1
fi
if [[ -z "$NODE_RANK" ]]; then
    echo "Error: NODE_RANK (or SLURM_PROCID) must be set for multinode PP" >&2
    exit 1
fi
if [[ -z "$MASTER_ADDR" ]]; then
    echo "Error: MASTER_ADDR must be set to the rank-0 node management IP" >&2
    exit 1
fi

wait_for_amd_gpu_clean

if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

# ---- MI355X / ROCm env (from kimik3_fp4_mi355x_mtp.sh) ----------------------
export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
export VLLM_ROCM_USE_AITER=1
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
export AITER_BF16_FP8_MOE_BOUND=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export PYTHONNOUSERSITE=1

mec_version=$(rocm-smi --showfw 2>/dev/null | grep MEC | head -n 1 | awk '{print $NF}')
if [[ "$mec_version" == "" || ${mec_version:-0} -lt 177 ]]; then
    export HSA_NO_SCRATCH_RECLAIM=1
fi

export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}"
export DISTRIBUTED_TIMEOUT_S="${DISTRIBUTED_TIMEOUT_S:-1800}"
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export EVAL_ONLY="${EVAL_ONLY:-false}"
export EVAL_FRAMEWORK="${EVAL_FRAMEWORK:-lm-eval}"

if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

# ---- KV offload (kimik3-fp4-mi355x-vllm-agentic-mtp default: dram/simple) ---
OFFLOAD_ARGS=()
if agentic_kv_offload_enabled; then
    case "${KV_OFFLOAD_BACKEND:-}" in
    vllm-simple)
        require_agentic_kv_offload_backend vllm-simple
        CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / TP ))
        export PYTHONHASHSEED=42
        SIMPLE_LAZY_OFFLOAD="${SIMPLE_LAZY_OFFLOAD:-false}"
        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":$CPU_BYTES_PER_RANK,\"lazy_offload\":$SIMPLE_LAZY_OFFLOAD}}"
        )
        echo "SimpleCPUOffloadConnector: ${CPU_BYTES_PER_RANK} B/rank x ${TP} ranks, lazy_offload=$SIMPLE_LAZY_OFFLOAD"
        ;;
    *)
        echo "Error: unsupported KV_OFFLOAD_BACKEND='${KV_OFFLOAD_BACKEND:-}' (expected vllm-simple when KV_OFFLOADING=dram)" >&2
        exit 1
        ;;
    esac
else
    require_agentic_kv_offload_none
fi

if agentic_kv_offload_enabled; then
    DEFAULT_GPU_MEM_UTIL=0.84
else
    DEFAULT_GPU_MEM_UTIL=0.90
fi
GPU_MEM_UTIL="${GPU_MEM_UTIL:-$DEFAULT_GPU_MEM_UTIL}"

# Map host REPO paths to container /workspace before merge (ln passes $HOME/InferenceX/...).
_normalize_aiter_extra_csv() {
    local p="${1:-}"
    [[ -n "$p" ]] || return 0
    if [[ -f "$p" ]]; then
        echo "$p"
        return
    fi
    if [[ "$p" == *"/InferenceX/"* ]]; then
        local ws="/workspace/${p#*InferenceX/}"
        if [[ -f "$ws" ]]; then
            echo "$ws"
            return
        fi
    fi
    echo "$p"
}

# Merge offline aiter GEMM tune rows BEFORE patches/workers import aiter.
AITER_GEMM_EXTRA_CSV="$(_normalize_aiter_extra_csv "${AITER_GEMM_EXTRA_CSV:-}")"
AITER_GEMM_EXTRA_CSV="${AITER_GEMM_EXTRA_CSV:-${GITHUB_WORKSPACE:-/workspace}/experimental/kimik3-v4/aiter/kimik3_bf16_tuned_gemm.extra.csv}"
export AITER_CONFIG_GEMM_BF16="${AITER_CONFIG_GEMM_BF16:-${AITER_CONFIG_DIR:-/tmp/aiter_configs}/bf16_tuned_gemm.csv}"
if [[ "${AITER_GEMM_MERGE:-auto}" != "0" ]]; then
    if [[ -f "$AITER_GEMM_EXTRA_CSV" ]] && awk -F, 'NR>1 && $1 ~ /^gfx/ {found=1; exit} END{exit !found}' "$AITER_GEMM_EXTRA_CSV"; then
        export AITER_GEMM_EXTRA_CSV
        bash "${GITHUB_WORKSPACE:-/workspace}/experimental/kimik3-v4/aiter/merge_aiter_gemm_configs.sh"
        if ! grep -q ",614,1536,128," "$AITER_CONFIG_GEMM_BF16" 2>/dev/null; then
            echo "WARN: M614 tuned rows not found in ${AITER_CONFIG_GEMM_BF16} (extra=${AITER_GEMM_EXTRA_CSV})" >&2
        else
            echo "aiter GEMM merge OK: M614 rows present in ${AITER_CONFIG_GEMM_BF16}"
        fi
    else
        echo "WARN: skip aiter GEMM merge — extra csv missing or empty: ${AITER_GEMM_EXTRA_CSV}" >&2
    fi
fi
export AITER_CONFIG_GEMM_BF16

# Split N=6288,K=7168 GEMMs into flydsl-friendly N chunks before workers load aiter.
# Prefer sitecustomize (aiter_site on PYTHONPATH) for per-worker install.
# Optional eager install in the parent process: set AITER_N6288_CHUNK_PATCH=1.
# Skip entirely with AITER_N6288_CHUNK_PATCH=0.
if [[ "${AITER_N6288_CHUNK_PATCH:-1}" != "0" ]]; then
    # Eager install WITHOUT aiter_site on PYTHONPATH (avoids helper fork-storm).
    PYTHONPATH="${GITHUB_WORKSPACE:-/workspace}/experimental/kimik3-v4/aiter" \
      AITER_N6288_CHUNK_PATCH=1 \
      python3 "${GITHUB_WORKSPACE:-/workspace}/experimental/kimik3-v4/aiter/patch_gemm_n6288_chunk.py" \
      || echo "WARN: eager patch_gemm_n6288_chunk failed (workers may still install via sitecustomize)" >&2
else
    echo "Skipping patch_gemm_n6288_chunk.py (AITER_N6288_CHUNK_PATCH=0)"
fi

# aiter custom-all-reduce skips the IPC-meta gather when a rank has no new graph
# buffers, desyncing its TCP-store sequence and deadlocking the TP group at the end
# of cudagraph capture. Skip with AITER_CA_FLUSH_SYNC_PATCH=0.
if [[ "${AITER_CA_FLUSH_SYNC_PATCH:-1}" != "0" ]]; then
    PYTHONPATH="${GITHUB_WORKSPACE:-/workspace}/experimental/kimik3-v4/aiter" \
      AITER_CA_FLUSH_SYNC_PATCH=1 \
      python3 "${GITHUB_WORKSPACE:-/workspace}/experimental/kimik3-v4/aiter/patch_ca_graph_flush_sync.py" \
      || echo "WARN: eager patch_ca_graph_flush_sync failed (workers may still install via sitecustomize)" >&2
else
    echo "Skipping patch_ca_graph_flush_sync.py (AITER_CA_FLUSH_SYNC_PATCH=0)"
fi

# K3 container patches (triton 3.7.0 + vLLM/aiter hotfixes) — idempotent.
bash "$(dirname "$0")/../../single_node/agentic/apply_k3_container_patches.sh"

# vLLM #50514 (open): DSpark/EAGLE3 under PP — draft on last PP stage only.
# Required when SPEC_DECODE is on with PP>1; no-op if already applied.
case "${SPEC_DECODE:-false}" in
true|TRUE|1|yes|YES|on|ON|mtp|dspark)
    bash "${GITHUB_WORKSPACE:-/workspace}/experimental/kimik3-v4/apply_vllm_50514_pp_spec.sh" \
      || { echo "ERROR: apply_vllm_50514_pp_spec.sh failed (needed for DSpark+PP)" >&2; exit 1; }
    ;;
esac

# ---- Optional DSpark (matches kimik3_fp4_mi355x_mtp.sh) ---------------------
SPEC_ARGS=()
case "${SPEC_DECODE:-false}" in
true|TRUE|1|yes|YES|on|ON|mtp|dspark)
    SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-2}"
    DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-Inferact/Kimi-K3-DSpark}"
    # Prefer a local HF-cache checkout when the hub id is not a directory.
    if [[ ! -d "$DRAFT_MODEL_PATH" ]]; then
        for cand in \
            "/hf_cache/Kimi-K3-DSpark" \
            "/hf_cache/models--Inferact--Kimi-K3-DSpark" \
            "${HF_HOME:-}/Kimi-K3-DSpark" \
            "${HUGGINGFACE_HUB_CACHE:-}/Kimi-K3-DSpark"; do
            if [[ -n "$cand" && -d "$cand" && -n "$(ls -A "$cand" 2>/dev/null)" ]]; then
                DRAFT_MODEL_PATH="$cand"
                break
            fi
        done
    fi
    SYNTHETIC_ACCEPT_LEN="${SYNTHETIC_ACCEPT_LEN:-2.51}"
    # Use real block rejection when accuracy-checking (EVAL_ONLY / RUN_EVAL), or when
    # explicitly requested. Synthetic AL is only for throughput bring-up.
    REJECTION_SAMPLE_METHOD="${REJECTION_SAMPLE_METHOD:-}"
    if [[ -z "$REJECTION_SAMPLE_METHOD" ]]; then
        if [[ "${EVAL_ONLY:-false}" == "true" || "${RUN_EVAL:-false}" == "true" ]]; then
            REJECTION_SAMPLE_METHOD=block
        else
            REJECTION_SAMPLE_METHOD=synthetic
        fi
    fi
    case "$REJECTION_SAMPLE_METHOD" in
    block|BLOCK)
        SPEC_ARGS=(
            --speculative-config
            "{\"model\":\"$DRAFT_MODEL_PATH\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\": \"block\"}"
        )
        ;;
    synthetic|SYNTHETIC)
        SPEC_ARGS=(
            --speculative-config
            "{\"model\":\"$DRAFT_MODEL_PATH\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
        )
        ;;
    *)
        echo "Error: REJECTION_SAMPLE_METHOD='$REJECTION_SAMPLE_METHOD' (expected block|synthetic)" >&2
        exit 1
        ;;
    esac
    echo "DSpark enabled: draft=${DRAFT_MODEL_PATH} tokens=${SPEC_NUM_TOKENS} rejection=${REJECTION_SAMPLE_METHOD} eval_only=${EVAL_ONLY:-false} run_eval=${RUN_EVAL:-false}"
    ;;
false|FALSE|0|no|NO|off|OFF|none|"")
    echo "SPEC_DECODE=off (STP / no speculative decoding)"
    ;;
*)
    echo "Error: SPEC_DECODE='${SPEC_DECODE}' (expected true|false|mtp|dspark)" >&2
    exit 1
    ;;
esac

if (( MIN_CUDAGRAPH_CAPTURE_SIZE > MAX_CUDAGRAPH_CAPTURE_SIZE )); then
    echo "Error: MIN_CUDAGRAPH_CAPTURE_SIZE=$MIN_CUDAGRAPH_CAPTURE_SIZE exceeds MAX_CUDAGRAPH_CAPTURE_SIZE=$MAX_CUDAGRAPH_CAPTURE_SIZE" >&2
    exit 1
fi
CUDAGRAPH_CAPTURE_SIZES="$(seq -s, "$MIN_CUDAGRAPH_CAPTURE_SIZE" "$MAX_CUDAGRAPH_CAPTURE_SIZE")"
COMPILATION_CONFIG_ARGS=(
    --compilation-config
    "{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"max_cudagraph_capture_size\":$MAX_CUDAGRAPH_CAPTURE_SIZE,\"custom_ops\":[\"+fused_rms_norm_gated\"],\"cudagraph_capture_sizes\":[$CUDAGRAPH_CAPTURE_SIZES]}"
)

SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

COMMON_VLLM_ARGS=(
    --trust-remote-code
    --moe-backend auto
    --tensor-parallel-size "$TP"
    --pipeline-parallel-size "$PP"
    --distributed-timeout-seconds "$DISTRIBUTED_TIMEOUT_S"
    --load-format fastsafetensors
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --language-model-only
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --max-model-len 1048576
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --kv-cache-dtype fp8
    --enable-prefix-caching
    "${COMPILATION_CONFIG_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
    "${SPEC_ARGS[@]}"
    --master-addr "$MASTER_ADDR"
    --master-port "$MASTER_PORT"
    --nnodes "$NNODES"
    --node-rank "$NODE_RANK"
)
if [[ "${ENFORCE_EAGER:-false}" == "true" ]]; then
    COMMON_VLLM_ARGS+=(--enforce-eager)
fi
# Fallback when the aiter custom-all-reduce capture path misbehaves: drop to PYNCCL.
if [[ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" == "1" ]]; then
    COMMON_VLLM_ARGS+=(--disable-custom-all-reduce)
fi
# Async scheduling toggle for PP (vLLM --async-scheduling / --no-async-scheduling).
case "${ASYNC_SCHEDULING:-auto}" in
true|TRUE|1|yes|YES|on|ON)
    COMMON_VLLM_ARGS+=(--async-scheduling)
    echo "ASYNC_SCHEDULING=on -> --async-scheduling"
    ;;
false|FALSE|0|no|NO|off|OFF)
    COMMON_VLLM_ARGS+=(--no-async-scheduling)
    echo "ASYNC_SCHEDULING=off -> --no-async-scheduling"
    ;;
auto|AUTO|"")
    echo "ASYNC_SCHEDULING=auto -> omit (vLLM default)"
    ;;
*)
    echo "Error: ASYNC_SCHEDULING='${ASYNC_SCHEDULING}' (expected true|false|auto)" >&2
    exit 1
    ;;
esac

if [[ "$NODE_RANK" -eq 1 ]]; then
    echo "Starting rank-1 headless PP worker on $(hostname)"
    HEADLESS_ARGS=(--headless)
    VLLM_CMD=(vllm serve "$MODEL_PATH" --served-model-name "$MODEL" "${COMMON_VLLM_ARGS[@]}" "${HEADLESS_ARGS[@]}")
    printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
    printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
    exec "${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1
fi

# Rank 0: OpenAI API + agentic trace replay (CONC_LIST sequential, like agentic_srt.sh).
resolve_trace_source
install_agentic_deps

BASE_RESULT_DIR="${RESULT_DIR}"
BASE_RESULT_FILENAME="${RESULT_FILENAME}"
read -r -a CONCURRENCIES <<< "${CONC_LIST:-$CONC}"
if [[ "${#CONCURRENCIES[@]}" -eq 0 ]]; then
    echo "ERROR: CONC_LIST/CONC must contain at least one concurrency" >&2
    exit 1
fi

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

VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    "${COMMON_VLLM_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$BASE_RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$BASE_RESULT_DIR/vllm_command.txt"
SERVER_LOG="$BASE_RESULT_DIR/server.log"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "${EVAL_ONLY}" == "true" ]]; then
    run_eval --port "$PORT"
else
    for concurrency in "${CONCURRENCIES[@]}"; do
        if ! [[ "$concurrency" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: invalid agentic concurrency: $concurrency" >&2
            exit 1
        fi
        export CONC="$concurrency"
        # Workflow expects ${RESULT_FILENAME}_concN.json (see agentic_srt.sh).
        export RESULT_FILENAME="${BASE_RESULT_FILENAME}_conc${concurrency}"
        RESULT_DIR="${BASE_RESULT_DIR}/conc_${concurrency}"
        mkdir -p "$RESULT_DIR"
        # Also mirror under /logs/agentic for launch_mi355x-amds.sh staging.
        if [[ -d /logs/agentic ]]; then
            mkdir -p "/logs/agentic/conc_${concurrency}"
            # Keep aiperf under RESULT_DIR; staging copies from rank0 or /logs.
        fi
        echo "Running agentic concurrency $concurrency of: ${CONCURRENCIES[*]}"
        build_replay_cmd "$RESULT_DIR"
        run_agentic_replay_and_write_outputs "$RESULT_DIR"
        if [[ -d /logs/agentic/conc_${concurrency} && -d "$RESULT_DIR" ]]; then
            cp -a "$RESULT_DIR"/. "/logs/agentic/conc_${concurrency}/" 2>/dev/null || true
        fi
    done
    # Optional accuracy check against the still-warm engine (gsm8k via lm-eval).
    if [[ "${RUN_EVAL:-false}" == "true" ]]; then
        echo "RUN_EVAL=true: running lm-eval after agentic profiling"
        run_eval --framework lm-eval --port "$PORT"
        append_lm_eval_summary || true
    fi
fi
