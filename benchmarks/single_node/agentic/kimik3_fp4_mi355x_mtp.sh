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
#   - --enable-prefix-caching IS passed, because SimpleCPUOffloadConnector
#     requires it and silently disables offload without it. KNOWN FAILING: with
#     it, DSpark + KV offload takes an 8-rank GPU memory access fault after CUDA
#     graph capture. Both offload backends fail the same way, so this recipe
#     cannot currently produce DSpark rows with KV offload; see the serve block.
#   - speculative-config drops "attention_backend": "FLASHINFER_MLA" -- absent on
#     ROCm, rejected by platforms/rocm.py; replaced with TRITON_MLA, not dropped.
#   - lazy_offload is a JSON boolean; bool("false") is True in Python.
#
# Acceptance DEVIATES from docs/PR_REVIEW_CHECKLIST.md rule 10 and needs codeowner
# sign-off: rule 10 wants throughput points to simulate acceptance at the
# committed golden AL, but rejection_sample_method synthetic page-faults during
# CUDA graph capture on gfx950, so every row runs block (real verification)
# instead. See the SPEC_CONFIG block below for the measurements and consequences.
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
# kimik3 had to be added explicitly. The served context is K3's full 1M native
# window so the unfiltered corpus replays without truncation; vLLM derives it
# rather than it being pinned with --max-model-len (see the serve block).
resolve_trace_source
install_agentic_deps

# Workaround for MEC FW <177 RCCL memory reclaim issue
version=$(rocm-smi --showfw 2>/dev/null | grep MEC | head -n 1 | awk '{print $NF}')
if [[ "$version" == "" || ${version:-0} -lt 177 ]]; then
    export HSA_NO_SCRATCH_RECLAIM=1
fi

# ---- upstream recipe: hardware_overrides.amd extra_env -----------------------
# Upstream extra_env, unmodified. VLLM_ROCM_USE_AITER cannot be turned off: the
# parent switch also disables VLLM_ROCM_USE_AITER_MOE and AITER is the only MXFP4
# MoE backend on gfx950, so 0 gives "NotImplementedError: No MXFP4 MoE backend
# supports the deployment configuration".
#
# An earlier revision of this recipe additionally set VLLM_ROCM_USE_AITER_LINEAR=0,
# VLLM_ROCM_USE_AITER_TRITON_GEMM=0 and VLLM_ROCM_USE_SKINNY_GEMM=0, on the theory
# that the page fault came from AITER's untuned bf16 GEMM paths. That was measured
# and REFUTED (runs/30367040157): the toggles took effect -- the "not found tuned
# config in bf16_tuned_gemm.csv" messages went to zero -- and the fault was
# unchanged. They only cost throughput by pushing linear GEMMs onto untuned
# torch/Triton paths, so they are removed. The configuration measured working runs
# AITER entirely at its defaults.
export VLLM_ROCM_USE_AITER=1
export SAFETENSORS_FAST_GPU=1
# AITER a8w4 MoE path for the MXFP4-weight/MXFP8-activation QAT checkpoint.
# Set to 0 to fall back to the AITER a16w4 MoE path. Unaffected by the LINEAR
# and GEMM toggles above; VLLM_ROCM_USE_AITER_MOE is left at its default (on).
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

# --max-num-seqs is the upstream recipe's fixed 128, NOT the agentic convention of
# tracking CONC. This is deliberate and is the one remaining difference between
# this recipe and the only configuration measured working on gfx950.
#
# Tracking CONC (previously min(CONC, 32)) makes conc 1 emit
# cudagraph_capture_sizes [1, 2, 4, 8, 16], and capture page-faults at index 2
# with hipErrorIllegalAddress. The verified-working diagnostic ran 128, which
# emits 51 capture sizes [1, 2, 4 ... 496, 512] and captured all of them cleanly
# -- including the same small sizes that fault under the short list, so the
# trigger is not a single poisonous shape. GPU KV also differs (1,525,407 tokens
# at 128 vs 1,601,340 at 1), so allocator layout shifts too, which fits an
# out-of-bounds access that only faults when the neighbouring page is unmapped.
#
# Cost: --max-num-seqs no longer tracks CONC, so the server's batching ceiling is
# constant across the ladder instead of matching the offered concurrency. At these
# concurrencies the load generator is the binding constraint, so this should not
# change what is measured, but it is a real deviation from agentic convention and
# from the sibling non-MTP recipe.
MAX_SEQS=128

# The draft's attention backend MUST be pinned. The upstream recipe pins
# FLASHINFER_MLA, which cannot work here: flashinfer is absent from the ROCm
# image and platforms/rocm.py rejects it outright ("Selected backend
# AttentionBackendEnum.FLASHINFER_MLA is not valid for this configuration.
# Reason: [ImportError]"), failing the draft at construction (dspark_mla.py:59
# -> nvidia/mla.py:267 get_attn_backend). Leaving the key off entirely is NOT a
# safe substitute: with no pin the draft resolves to a backend that page-faults
# on gfx950 during CUDA graph capture -- measured twice, once as
# "hipErrorIllegalAddress" during capture and once as a kernel-level
# "amdgpu [gfxhub0] retry page fault ... VM_L2_PROTECTION_FAULT_STATUS
# 0x00301031, Faulty UTCL2 client ID: TCP" that killed the worker before Python
# could log anything. TRITON_MLA is ROCm-native and fixes it: verified on
# 8x MI355X, server reaches ready and returns coherent completions with no
# page faults in dmesg.
SPEC_ATTN_BACKEND="TRITON_MLA"

# rejection_sample_method is "block" on EVERY row, including throughput.
#
# THIS DEVIATES FROM docs/PR_REVIEW_CHECKLIST.md RULE 10, which requires agentic
# spec-decode throughput points to simulate acceptance via synthetic rejection
# sampling pinned to the committed golden AL. It is a deliberate, temporary
# deviation and needs codeowner sign-off; it is not an oversight.
#
# Reason: "block" is what the only configuration measured working on gfx950 used
# (verified on 8x MI355X with TRITON_MLA pinned and --max-num-seqs 128: server
# reaches ready, returns coherent completions, dmesg shows no page faults), so
# this recipe matches that configuration rather than diverging from it.
#
# Do NOT read this as "synthetic is the bug". That hypothesis was tested and
# refuted: with block, TRITON_MLA and 4096, conc 1 still page-faulted at capture
# step 2/5 (runs/30349509927). The graph-capture fault reproduces across
# 16384 (runs/30343017203), 4096 (runs/30344670346) and both rejection methods,
# so neither batch size nor the rejection method is the trigger. The remaining
# difference from the working configuration is --max-num-seqs, addressed above.
# No wvSplitK frames in any of them, so this is distinct from the skinny-GEMM
# fault the non-MTP recipe hits at conc <= 5.
#
# If --max-num-seqs 128 proves sufficient, synthetic should be restored to comply
# with rule 10 -- it is a one-line change and has not been retested since.
#
# Consequence for the data: throughput rows now run REAL verification, so their
# acceptance length is whatever the draft actually achieves rather than the
# committed golden AL. Those numbers are therefore NOT comparable to other
# agentic spec-decode entries in the changelog until synthetic works and this is
# reverted. The golden AL guard above is deliberately retained so the committed
# target still cannot drift silently and restoring synthetic is a one-line change.
SPEC_CONFIG="{\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"$SPEC_ATTN_BACKEND\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\"}"

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
    # --max-model-len is deliberately NOT passed. K3's config.json has no
    # max_position_embeddings, but vLLM still derives the full 1M native context:
    # the verified-working diagnostic omitted the flag and resolved to
    # max_seq_len=1048576, byte-identical to passing it explicitly. Omitting it
    # keeps this recipe aligned with that diagnostic and with the upstream recipe,
    # which also does not pass it. This is presentational only - it does not change
    # the served context length, and it is NOT a fix for anything.
    --max-num-seqs "$MAX_SEQS"
    # 4096, NOT 16384. 16384 was tried here and reverted: it page-faults during
    # CUDA graph capture on gfx950. Measured at conc 1 on 8x MI355X, twice - once
    # with the draft attention_backend unpinned, and again after pinning
    # TRITON_MLA, which rules the pin out as the cause: "Compile and warming up
    # model for size 16384" -> "Capturing CUDA graphs (PIECEWISE) 2/5" ->
    # c10::AcceleratorError hipErrorIllegalAddress. No wvSplitK frames, so this is
    # distinct from the skinny-GEMM fault the non-MTP twin hits at conc <= 5.
    # 4096 is the value verified working end to end with TRITON_MLA (server ready,
    # coherent completions, no page faults in dmesg).
    #
    # This does leave a known cost: vLLM warns "max_num_scheduled_tokens is set to
    # 4090 based on the speculative decoding settings ... consider increasing
    # max_num_batched_tokens", because drafting reserves slots out of the budget.
    # Raising it is the right direction once the graph-capture fault is fixed
    # upstream, but it cannot be raised today.
    --max-num-batched-tokens 4096
    --speculative-config "$SPEC_CONFIG"
    --mm-encoder-tp-mode data
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --language-model-only
    # --enable-prefix-caching IS required here, and it is required BY the offload
    # connector, not by the model.
    #
    # SimpleCPUOffloadConnector refuses to run without it and silently turns
    # itself off:
    #   simple_cpu_offload_connector.py:83] Detected prefix caching disabled,
    #   disabling CPU offload since it requires prefix caching.
    # So dropping this flag does NOT give "DSpark with DRAM offload" -- it gives
    # DSpark with no offload at all, while the sweep config still declares
    # kv-offloading: dram. That is a silent mis-measurement and was briefly
    # committed here (2a3c0d134) before being caught; do not reintroduce it.
    #
    # With the flag, the connector activates
    #   (simple_cpu_offload_connector.py:89] role=WORKER, per_rank=220.00 GB,
    #    world_size=8, mode=lazy)
    # and the run dies with 8x "Memory access fault by GPU node-N on address
    # 0x7..., Reason: Unknown" just after CUDA graph capture completes. So prefix
    # caching is not the defect -- it is the switch that activates the offload
    # path, and the defect is KV offload x DSpark.
    #
    # That is measured against BOTH offload implementations, so it is not a bug in
    # one connector: vllm-simple (SimpleCPUOffloadConnector) and vllm-native
    # (OffloadingConnector / CPUOffloadingSpec, 2234 GiB) both reach capture 100%
    # and then take the same fault. The non-MTP twin runs either connector with
    # prefix caching and completes warmup and profiling, and the upstream config
    # runs prefix caching with no connector, so DSpark is the differentiator.
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
