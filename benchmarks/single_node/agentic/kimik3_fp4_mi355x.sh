#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K3 MXFP4 on MI355X (gfx950) using vLLM.
#
# Upstream recipe: https://recipes.vllm.ai/moonshotai/Kimi-K3?hardware=mi355x
# (vllm-project/recipes#684, models/moonshotai/Kimi-K3.yaml ->
# hardware_overrides.amd), which lists mi355x as verified. The env/flags below
# are that AMD block; the AMD block replaces the base --load-format
# fastsafetensors with auto.
#
# K3 is a 2.8T-param natively-multimodal MoE (896 experts, 16 routed + 2 shared)
# built on Kimi Delta Attention (KDA), gated MLA and Attention Residuals. vLLM
# serves it through a dedicated ROCm path (vllm/models/kimi_k3/amd/*) selected by
# current_platform.is_rocm(); its KDA and attn_res triton kernels are validated
# on gfx950. The CUDA-only FlashKDA extension is gated behind
# VLLM_GPU_LANG=CUDA upstream and is neither built nor used here.
#
# TP=8 only: the MXFP4 checkpoint is ~1.56 TB, i.e. ~195 GB/GPU across 8 GPUs of
# MI355X's 288 GB HBM. TP=4 would need ~390 GB/GPU and cannot load.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION, EP_SIZE
#
# KV_OFFLOADING=dram requires KV_OFFLOAD_BACKEND=vllm-native.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# ROCR/HIP visibility for vLLM 0.14+
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

# The ~1.56 TB checkpoint is pre-staged on the NFS HF hub cache, which
# launch_mi355x-amds.sh mounts as HF_HUB_CACHE for this model (the node-local
# /var/lib NVMe cache cannot hold it). These calls are no-ops there.
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

# ---- Resolve traces and install deps ----------------------------------------
# No WEKA_LOADER_OVERRIDE needed: kimik3* is on resolve_trace_source's unfiltered
# allowlist (benchmark_lib.sh), so this replays the full unfiltered 062126 v7
# corpus rather than the 256k-capped variant. Note that allowlist is a hardcoded
# MODEL_PREFIX match, NOT a native-context-length rule as its comment suggests --
# kimik3 had to be added explicitly. --max-model-len below is K3's full 1M native
# context so the unfiltered corpus can be replayed without truncation.
resolve_trace_source
install_agentic_deps

# Workaround for MEC FW <177 RCCL memory reclaim issue
version=$(rocm-smi --showfw 2>/dev/null | grep MEC | head -n 1 | awk '{print $NF}')
if [[ "$version" == "" || ${version:-0} -lt 177 ]]; then
    export HSA_NO_SCRATCH_RECLAIM=1
fi

# ---- upstream recipe: hardware_overrides.amd extra_env -----------------------
export VLLM_ROCM_USE_AITER=1
export SAFETENSORS_FAST_GPU=1
# AITER a8w4 MoE path for the MXFP4-weight/MXFP8-activation QAT checkpoint.
# Set to 0 to fall back to the AITER a16w4 MoE path.
export AITER_SITUV2_A8W4=1
export AITER_BF16_FP8_MOE_BOUND=0
# REQUIRED on ROCm per the upstream recipe: the build auto-enables this to 1.
export VLLM_USE_BREAKABLE_CUDAGRAPH=0

# 2.8T of weights loaded off NFS takes far longer than the default to be ready.
export VLLM_ENGINE_READY_TIMEOUT_S=7200

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

OFFLOAD_ARGS=()
PREFIX_CACHE_ARGS=()

# TOTAL_CPU_DRAM_GB is the aggregate host-DRAM budget the matrix generator
# derives from dram-utilization and the runner's available-cpu-dram-mib, capped
# at the 2,861,022 MiB (3 TB decimal) agentic limit. Per
# benchmarks/single_node/agentic/README.md it must be consumed as given, never
# replaced with a model-specific constant; backends with per-rank pools divide it.
case "${KV_OFFLOAD_BACKEND:-}" in
    vllm-native)
        require_agentic_kv_offload_backend vllm-native
        unset VLLM_USE_SIMPLE_KV_OFFLOAD
        # vLLM's regular native KV-offload path (OffloadingConnector), NOT
        # SimpleCPUOffloadConnector: the "vllm-native" backend resolves to
        # OffloadingConnector by default, and VLLM_USE_SIMPLE_KV_OFFLOAD=1 would
        # switch it. Left UNSET deliberately. --kv_offloading_size takes the
        # aggregate budget undivided.
        OFFLOAD_ARGS=(
            --kv_offloading_backend native
            --kv_offloading_size "$TOTAL_CPU_DRAM_GB"
            --disable-hybrid-kv-cache-manager
        )
        ;;
    vllm-simple)
        require_agentic_kv_offload_backend vllm-simple
        # SimpleCPUOffloadConnector's cpu_bytes_to_use is PER RANK, so divide the
        # aggregate budget by the rank count (single-node TP => GPU_COUNT, which
        # the launcher exports; fall back to TP for stand-alone runs).
        SIMPLE_RANKS="${GPU_COUNT:-$TP}"
        CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / SIMPLE_RANKS ))
        # Identical prefixes must hash to identical block keys across ranks.
        export PYTHONHASHSEED=42
        # Plain TP (no DP-attention here) uses lazy offload, matching the dsv4
        # plain-TP ladder; eager offload only buys cross-rank block-hash
        # stability under DEP, which this recipe does not run.
        OFFLOAD_CONFIG=$(cat <<EOF
{
  "kv_connector": "SimpleCPUOffloadConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "cpu_bytes_to_use": ${CPU_BYTES_PER_RANK},
    "enable_cross_layers_blocks": "true",
    "lazy_offload": true
  }
}
EOF
)
        OFFLOAD_ARGS=(--kv-transfer-config "$OFFLOAD_CONFIG")
        ;;
    "")
        # KV_OFFLOADING=none: assert the pairing and run fully GPU-resident.
        require_agentic_kv_offload_backend none || true
        ;;
    *)
        echo "Error: unsupported KV_OFFLOAD_BACKEND='$KV_OFFLOAD_BACKEND' for this recipe" >&2
        exit 1
        ;;
esac

EP_ARGS=()
if [ "$EP_SIZE" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

echo "Starting vllm server..."
export PYTHONNOUSERSITE=1

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size="$TP"
    "${EP_ARGS[@]}"
    --trust-remote-code
    --load-format auto
    --moe-backend auto
    # 0.90, NOT the upstream recipe's 0.95: MI355X reports 287.98 GiB total but
    # only ~271 GiB free at startup (~17 GiB driver/framework overhead), so 0.95
    # (273.59 GiB) hard-fails before KV sizing with "Free memory on device cuda:N
    # ... is less than desired GPU memory utilization". Measured on g17; 0.90 is
    # also what every other MI355X recipe here uses.
    --gpu-memory-utilization 0.90
    # K3's full 1M native context, matching the unfiltered corpus that
    # resolve_trace_source now picks for kimik3*.
    --max-model-len 1048576
    --max-num-seqs "$CONC"
    --max-num-batched-tokens 4096
    --mm-encoder-tp-mode data
    --reasoning-parser kimi_k3
    "${PREFIX_CACHE_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
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
