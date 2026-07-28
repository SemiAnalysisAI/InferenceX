#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay for Kimi-K3 MXFP4 on MI355X (gfx950), vLLM, with DSpark
# speculative decoding (draft: Inferact/Kimi-K3-DSpark).
#
# Recipe: https://recipes.vllm.ai/moonshotai/Kimi-K3?hardware=mi355x
# (vllm-project/recipes#684 -> hardware_overrides.amd). Deviations, all measured:
#   - gpu-memory-utilization 0.88, not 0.95: only ~271 GiB is free at the
#     worker's startup check, below the 273.59 GiB that 0.95 demands.
#   - --enable-prefix-caching is required; K3 is hybrid (69 KDA + 24 gated MLA)
#     and vLLM asserts tokens_per_block % tokens_per_hash without it.
#   - speculative-config drops "attention_backend": "FLASHINFER_MLA" -- absent on
#     ROCm, rejected by platforms/rocm.py. Server runs TRITON_ATTN.
#   - lazy_offload is a JSON boolean; bool("false") is True in Python.
#
# Acceptance follows docs/PR_REVIEW_CHECKLIST.md rule 10: throughput points
# simulate acceptance at the committed golden AL from
# golden_al_distribution/kimik3_dspark.yaml, EVAL_ONLY verifies for real.
#
# TP=8 only: the ~1.56 TB checkpoint needs ~195 GB/GPU of 288 GB HBM.
# On ROCm the draft runs the NVIDIA dspark_mla implementation (no amd/dspark*.py
# exists); the base model still resolves to kimi_k3.amd.model.
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

# The ~1.5 TB checkpoint stages to the node-local /var/lib NVMe hub cache that
# launch_mi355x-amds.sh mounts as HF_HUB_CACHE (~6.4 TB free there). The first
# job on a cold node pays the download; every later job on that node reads from
# local NVMe, which is faster than NFS and avoids share contention.
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

# 2.8T of weights takes far longer than the default to become ready, especially
# on a cold node that must download the checkpoint first.
export VLLM_ENGINE_READY_TIMEOUT_S=7200

# ---- DSpark draft + golden acceptance length --------------------------------
DRAFT_MODEL="${DRAFT_MODEL:-Inferact/Kimi-K3-DSpark}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-7}"
# Committed golden AL for kimi-k3 / thinking_on / this draft length, taken from
# the curve whose collector used the SAME sampling this recipe does
# (draft_sample_method=probabilistic, rejection_sample_method=block). The plain
# kimik3_dspark.yaml curve does not state its sampling and runs lower at every
# draft length (3.78 vs 3.84 at k=7), so it is the wrong target here. The guard
# below hard-fails if the committed file is recollected and this goes stale.
GOLDEN_AL=3.84
GOLDEN_AL_FILE="golden_al_distribution/kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method.yaml"
# Fail closed if the pinned AL does not match the committed curve: a missing,
# unparseable or drifted value all collapse to one mismatch check, so a
# recollected curve can never leave a stale constant in place.
# ("int" is reserved in awk, hence in_th.)
FILE_AL=$(awk -v k="$NUM_SPEC_TOKENS" '
    /^kimi-k3:/                { in_model = 1; next }
    /^[^[:space:]#]/           { in_model = 0 }
    in_model && /thinking_on:/ { in_th = 1; next }
    in_th && $1 == k":"        { print $2; exit }
' "$GOLDEN_AL_FILE" 2>/dev/null)
if [ "$FILE_AL" != "$GOLDEN_AL" ]; then
    echo "Golden AL mismatch: $GOLDEN_AL_FILE gives '${FILE_AL:-<unreadable>}' for num_speculative_tokens=$NUM_SPEC_TOKENS, recipe pins $GOLDEN_AL" >&2
    exit 1
fi
echo "Golden AL OK: num_speculative_tokens=$NUM_SPEC_TOKENS -> $FILE_AL"
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
        # lazy_offload MUST be a JSON boolean: the connector does
        # bool(extra_config.get("lazy_offload", False)), so the official command's
        # string "false" is truthy and silently selects LAZY. We pass true
        # deliberately. Eager was tried and reverted: each of the 8 workers pinned
        # its full per-rank pool up front (324.67 GB each, 2597 GB total), which
        # starved the shm_broadcast ring and killed EngineCore during warmup.
        OFFLOAD_CONFIG=$(cat <<EOF
{
  "kv_connector": "SimpleCPUOffloadConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "cpu_bytes_to_use_per_rank": ${CPU_BYTES_PER_RANK},
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
    # 0.88, NOT the upstream recipe's 0.95: MI355X reports 287.98 GiB total but
    # only ~271 GiB free at the worker's startup check (~17 GiB driver/framework
    # overhead), so 0.95 (273.59 GiB) hard-fails before KV sizing with "Free
    # memory on device cuda:N ... is less than desired GPU memory utilization".
    # Measured ceiling is ~0.94. Held at 0.88 rather than the 0.90 the non-MTP
    # recipe uses: the resident DSpark draft plus its own state consume HBM on top
    # of the ~195 GB/GPU of target weights (GPU KV drops 2,204,913 -> 1,723,308
    # tokens with the draft loaded), and leaked-VRAM incidents on this fleet have
    # repeatedly left GPUs a few GiB short of the threshold.
    --gpu-memory-utilization 0.88
    # K3's full 1M native context, matching the unfiltered corpus that
    # resolve_trace_source now picks for kimik3*.
    --max-model-len 1048576
    --max-num-seqs "$MAX_SEQS"
    # 16384, not the 4096 this recipe started with. The agentic corpus is
    # prefill-dominated (measured 5,548 input tok/s against 16 output tok/s, a
    # ~350:1 ratio), and at 4096 a single 167k-token trace prefill becomes ~41
    # chunks. Each chunk drives its own DRAM-offload evict/restore round, and the
    # resulting churn made one sample_tokens RPC exceed the executor timeout:
    # measured on the non-MTP twin at conc 8, warmup froze at 25 completed
    # requests and EngineCore died with "TimeoutError: RPC call to sample_tokens
    # timed out" after ~1500 blocks were restored from CPU in a single step.
    # At 16384 the same row completed warmup with 409 requests and 0 errors, then
    # ran the full profiling phase. The tradeoff is real but favourable: peak
    # activation rises 2.17 -> 5.41 GiB, which at a fixed --gpu-memory-utilization
    # costs ~14% of the GPU KV pool (2,204,913 -> 1,893,092 tokens), yet the ~4x
    # drop in chunk count more than pays for it.
    --max-num-batched-tokens 16384
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
