#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K3 MXFP4 on MI355X (gfx950) using vLLM
# WITH DSpark speculative decoding (draft: Inferact/Kimi-K3-DSpark).
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
# ACCEPTANCE: per docs/PR_REVIEW_CHECKLIST.md rule 10, agentic spec-decode
# throughput points must simulate acceptance at the committed golden AL from
# golden_al_distribution/, not measure it. This recipe pins vLLM synthetic
# rejection sampling to the kimi-k3 thinking_on AL for num_speculative_tokens=7
# (golden_al_distribution/kimik3_dspark.yaml). The EVAL_ONLY accuracy run uses
# real target verification instead: synthetic acceptance bypasses verification
# and would zero the eval score.
#
# The upstream recipe additionally pins "attention_backend": "FLASHINFER_MLA",
# which CANNOT be used on ROCm -- flashinfer is absent from the image and
# platforms/rocm.py rejects the backend ("Selected backend
# AttentionBackendEnum.FLASHINFER_MLA is not valid ... Reason: ['ImportError']").
# Verified by A/B on 8x MI355X: with the pin the draft fails at construction,
# without it the server reaches ready and generates. The pin is therefore
# dropped and the server runs TRITON_ATTN, as the MiniMax-M3 ROCm MTP recipe
# does. Note there is no vllm/models/kimi_k3/amd/dspark*.py, so on ROCm the
# draft runs the NVIDIA implementation while the base model uses kimi_k3.amd.*.
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

# ---- DSpark draft + golden acceptance length --------------------------------
DRAFT_MODEL="${DRAFT_MODEL:-Inferact/Kimi-K3-DSpark}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-7}"
# Committed golden AL for kimi-k3 / thinking_on / this draft length. Kept in sync
# with golden_al_distribution/kimik3_dspark.yaml; the guard below hard-fails if
# that file is ever recollected and this constant goes stale.
GOLDEN_AL=3.78
GOLDEN_AL_FILE="golden_al_distribution/kimik3_dspark.yaml"
# Drift guard: hard-fail if the committed golden AL no longer matches what this
# recipe pins, or if it cannot be parsed. Fail closed on purpose -- silently
# falling back to a stale constant is exactly the failure this is here to stop.
# (Note for maintainers: "int" is a reserved word in awk, hence in_th.)
if [ -f "$GOLDEN_AL_FILE" ]; then
    FILE_AL=$(awk -v k="$NUM_SPEC_TOKENS" '
        /^kimi-k3:/            { in_model = 1; next }
        /^[^[:space:]#]/       { in_model = 0 }
        in_model && /thinking_on:/ { in_th = 1; next }
        in_th && $1 == k":"    { print $2; exit }
    ' "$GOLDEN_AL_FILE")
    if [ -z "$FILE_AL" ]; then
        echo "Golden AL guard: could not read an AL for num_speculative_tokens=$NUM_SPEC_TOKENS from $GOLDEN_AL_FILE" >&2
        exit 1
    fi
    if [ "$FILE_AL" != "$GOLDEN_AL" ]; then
        echo "Golden AL drift: $GOLDEN_AL_FILE says $FILE_AL for num_speculative_tokens=$NUM_SPEC_TOKENS, recipe pins $GOLDEN_AL" >&2
        exit 1
    fi
    echo "Golden AL check OK: num_speculative_tokens=$NUM_SPEC_TOKENS -> AL $FILE_AL"
else
    echo "Golden AL guard: $GOLDEN_AL_FILE is missing" >&2
    exit 1
fi
if [[ "$DRAFT_MODEL" != /* ]]; then hf download "$DRAFT_MODEL"; fi

# Upstream caps --max-num-seqs at 32 under spec decoding (drafting needs extra
# VRAM). Agentic convention tracks CONC, so take the lower of the two.
MAX_SEQS="$CONC"
if [ "$MAX_SEQS" -gt 32 ]; then MAX_SEQS=32; fi

# Throughput pins synthetic acceptance to the golden AL. EVAL_ONLY uses real
# target verification (rejection_sample_method block, as the official recipe).
if [ "${EVAL_ONLY:-false}" = "true" ]; then
    SPEC_CONFIG="{\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS,\"method\":\"dspark\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\"}"
else
    SPEC_CONFIG="{\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS,\"method\":\"dspark\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":$GOLDEN_AL}"
fi

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
        # No --disable-hybrid-kv-cache-manager: that came from an MLA-uniform
        # recipe. K3's KV specs are heterogeneous (KDA state + MLA latent) and
        # cannot be promoted to one unified type, so the hybrid manager stays on.
        #
        # UNIT CONVERSION: --kv_offloading_size is GiB (vllm/config/vllm.py does
        # cpu_bytes_to_use = kv_offloading_size * (1 << 30)), but
        # TOTAL_CPU_DRAM_GB is DECIMAL GB (the agentic README divides bytes by
        # 1e9). Passing it raw would over-request by ~7.4% and breach the
        # 2,861,022 MiB agentic cap, so convert decimal GB -> GiB.
        KV_OFFLOAD_GIB=$(( TOTAL_CPU_DRAM_GB * 1000000000 / 1073741824 ))
        OFFLOAD_ARGS=(
            --kv_offloading_backend native
            --kv_offloading_size "$KV_OFFLOAD_GIB"
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
        # Keys from the official K3 recipe command: cpu_bytes_to_use_per_rank
        # (which the connector honours as an explicit per-rank override of
        # cpu_bytes_to_use/world_size). The official example hardcodes
        # 236223201280 B (220 GiB/rank); we substitute the agentic budget
        # instead, because benchmarks/single_node/agentic/README.md requires
        # scripts to consume TOTAL_CPU_DRAM_GB, dividing it for per-rank backends.
        #
        # lazy_offload is a JSON BOOLEAN, not the official command's string
        # "false": the connector does bool(extra_config.get("lazy_offload")),
        # and bool("false") is True in Python, so the string silently selects
        # LAZY. We pass false to actually get eager offload.
        OFFLOAD_CONFIG=$(cat <<EOF
{
  "kv_connector": "SimpleCPUOffloadConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "cpu_bytes_to_use_per_rank": ${CPU_BYTES_PER_RANK},
    "lazy_offload": false
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
    --max-num-seqs "$MAX_SEQS"
    --max-num-batched-tokens 4096
    --speculative-config "$SPEC_CONFIG"
    --mm-encoder-tp-mode data
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    # Prefix caching is MANDATORY here and passed explicitly, not left to the
    # default: measured on gfx950, omitting it still trips
    #   AssertionError: tokens_per_block=1048576 not divisible by
    #   tokens_per_hash=3145728. Hybrid models (e.g. Mamba+Attention) need
    #   --enable-prefix-caching to align block sizes.
    # K3 is hybrid (69 KDA + 24 gated MLA), so the block/hash alignment only
    # holds with prefix caching on. It is also the right measurement: agentic
    # trace replay exists to exercise large shared prefixes.
    --enable-prefix-caching
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
