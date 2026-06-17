#!/usr/bin/env bash

# MiniMax-M3 MXFP8 MI300X (gfx942) single-node vLLM recipe.
# Reuses the dedicated ROCm image and applies the checked-in hybrid gfx94x
# MXFP8 MoE and single-chunk index top-k patches before starting vLLM. Block
# size 128 is mandatory for MSA sparse attention. Keep the default BF16 KV
# cache on gfx942: the checkpoint has no calibrated q/prob scales for ROCm FP8
# attention, and vLLM's fallback scale of 1.0 corrupts model accuracy.
# Target image vLLM revision: 4a560dd8db67c270f5e2afb614558271b76f2294.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    EP_SIZE \
    DP_ATTENTION \
    CONC \
    ISL \
    OSL \
    MAX_MODEL_LEN \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

VLLM_PACKAGE_ROOT="$(
    python - <<'PY'
from pathlib import Path

import vllm

print(Path(vllm.__file__).resolve().parent.parent)
PY
)"
M3_AMD_MODEL="$VLLM_PACKAGE_ROOT/vllm/models/minimax_m3/amd/model.py"
MXFP8_PATCH="$(dirname "$0")/minimaxm3_mi300x_mxfp8.patch"
MXFP8_EP_PATCH="$(dirname "$0")/minimaxm3_mi300x_ep_mxfp8.patch"
MXFP8_ORACLE="$VLLM_PACKAGE_ROOT/vllm/model_executor/layers/fused_moe/oracle/mxfp8.py"
if ! grep -q "Using fused CDNA3 (gfx94x)" "$MXFP8_ORACLE"; then
    if ! patch --batch --forward -d "$VLLM_PACKAGE_ROOT" -p1 < "$MXFP8_PATCH"; then
        echo "Failed to apply the MI300X MXFP8 patch" >&2
        exit 1
    fi
fi
if ! grep -q "Using fused CDNA3 (gfx94x)" "$MXFP8_ORACLE"; then
    echo "MI300X MXFP8 backend marker is missing after patching" >&2
    exit 1
fi
if ! grep -q "profiled gfx94x MiniMax-M3 EP8" "$MXFP8_ORACLE"; then
    if ! patch --batch --forward -d "$VLLM_PACKAGE_ROOT" -p1 < "$MXFP8_EP_PATCH"; then
        echo "Failed to apply the MI300X EP8 MXFP8 tuning patch" >&2
        exit 1
    fi
fi
if ! grep -q "profiled gfx94x MiniMax-M3 EP8" "$MXFP8_ORACLE"; then
    echo "MI300X EP8 MXFP8 tuning marker is missing after patching" >&2
    exit 1
fi

INDEX_TOPK_PATCH="$(dirname "$0")/minimaxm3_mi300x_index_topk.patch"
INDEX_TOPK_SOURCE="$VLLM_PACKAGE_ROOT/vllm/models/minimax_m3/common/ops/index_topk.py"
INDEX_TOPK_SOURCE_SHA256="20351dd410d409c2c779d1d05d3d715633323f6b0e022e3ae6fae1c487ab5888"
INDEX_TOPK_PATCHED_SHA256="36f5132ef789c74b7f88be8bd34a6ca1ea6c3ee6561213305f1db9f1b9cbd6fe"
index_topk_sha256="$(sha256sum "$INDEX_TOPK_SOURCE" | awk '{print $1}')"
if [ "$index_topk_sha256" = "$INDEX_TOPK_SOURCE_SHA256" ]; then
    if ! patch --batch --dry-run -d "$VLLM_PACKAGE_ROOT" -p1 < "$INDEX_TOPK_PATCH"; then
        echo "Failed to validate the MI300X index top-k patch" >&2
        exit 1
    fi
    if ! patch --batch -d "$VLLM_PACKAGE_ROOT" -p1 < "$INDEX_TOPK_PATCH"; then
        echo "Failed to apply the MI300X index top-k patch" >&2
        exit 1
    fi
elif [ "$index_topk_sha256" != "$INDEX_TOPK_PATCHED_SHA256" ]; then
    echo "MI300X index top-k source fingerprint mismatch: $index_topk_sha256" >&2
    exit 1
fi
index_topk_sha256="$(sha256sum "$INDEX_TOPK_SOURCE" | awk '{print $1}')"
if [ "$index_topk_sha256" != "$INDEX_TOPK_PATCHED_SHA256" ]; then
    echo "MI300X index top-k patched fingerprint mismatch: $index_topk_sha256" >&2
    exit 1
fi
echo "MI300X index top-k patch ready: $index_topk_sha256"

M3_KV_CACHE_MODE="${M3_KV_CACHE_MODE:-bf16}"
KV_CACHE_ARGS=()
case "$M3_KV_CACHE_MODE" in
    bf16)
        ;;
    fp8)
        # Port the validated ROCm FNUZ dtype handling from vLLM PR #45563.
        FP8_KV_PATCH="$(dirname "$0")/minimaxm3_mi300x_fp8_kv.patch"
        SPARSE_ATTN_SOURCE="$VLLM_PACKAGE_ROOT/vllm/models/minimax_m3/common/ops/sparse_attn.py"
        SPARSE_ATTN_IMPL="$VLLM_PACKAGE_ROOT/vllm/models/minimax_m3/common/sparse_attention.py"
        if ! grep -q "_FP8_DTYPES" "$SPARSE_ATTN_SOURCE"; then
            if ! patch --batch --dry-run -d "$VLLM_PACKAGE_ROOT" -p1 \
                < "$FP8_KV_PATCH"; then
                echo "Failed to validate the MiniMax M3 FP8 KV-cache patch" >&2
                exit 1
            fi
            if ! patch --batch -d "$VLLM_PACKAGE_ROOT" -p1 < "$FP8_KV_PATCH"; then
                echo "Failed to apply the MiniMax M3 FP8 KV-cache patch" >&2
                exit 1
            fi
        fi
        if ! grep -q "current_platform.fp8_dtype()" "$SPARSE_ATTN_IMPL" \
            || ! grep -q "torch.float8_e4m3fnuz" "$SPARSE_ATTN_SOURCE"; then
            echo "MiniMax M3 FP8 KV-cache dtype fix is missing after patching" >&2
            exit 1
        fi
        python3 -m py_compile "$SPARSE_ATTN_SOURCE" "$SPARSE_ATTN_IMPL"
        KV_CACHE_ARGS=(--kv-cache-dtype fp8)
        echo "M3 main KV-cache experiment mode: fp8"
        ;;
    *)
        echo "Invalid M3_KV_CACHE_MODE: $M3_KV_CACHE_MODE" >&2
        exit 2
        ;;
esac

M3_INDEX_KV_CACHE_MODE="${M3_INDEX_KV_CACHE_MODE:-bf16}"
INDEX_KV_CACHE_ARGS=()
case "$M3_INDEX_KV_CACHE_MODE" in
    bf16)
        ;;
    fp8)
        if [ "$M3_KV_CACHE_MODE" != "fp8" ]; then
            echo "M3 FP8 index cache requires M3_KV_CACHE_MODE=fp8" >&2
            exit 2
        fi
        FP8_INDEX_PATCH="$(dirname "$0")/minimaxm3_mi300x_fp8_index_cache_after_tuning.patch"
        M3_INDEXER="$VLLM_PACKAGE_ROOT/vllm/models/minimax_m3/common/indexer.py"
        if ! grep -q "minimax_m3_index_k_quant_and_cache" "$INDEX_TOPK_SOURCE"; then
            if ! patch --batch --dry-run -d "$VLLM_PACKAGE_ROOT" -p1 \
                < "$FP8_INDEX_PATCH"; then
                echo "Failed to validate the tuned MiniMax M3 FP8 index-cache patch" >&2
                exit 1
            fi
            if ! patch --batch -d "$VLLM_PACKAGE_ROOT" -p1 \
                < "$FP8_INDEX_PATCH"; then
                echo "Failed to apply the tuned MiniMax M3 FP8 index-cache patch" >&2
                exit 1
            fi
        fi
        FP8_INDEX_TYPED_LOAD_PATCH="$(dirname "$0")/minimaxm3_mi300x_fp8_index_typed_load.patch"
        if ! grep -q "kernel_index_cache" "$INDEX_TOPK_SOURCE"; then
            if ! patch --batch --dry-run -d "$VLLM_PACKAGE_ROOT" -p1 \
                < "$FP8_INDEX_TYPED_LOAD_PATCH"; then
                echo "Failed to validate the MiniMax M3 FP8 index typed-load patch" >&2
                exit 1
            fi
            if ! patch --batch -d "$VLLM_PACKAGE_ROOT" -p1 \
                < "$FP8_INDEX_TYPED_LOAD_PATCH"; then
                echo "Failed to apply the MiniMax M3 FP8 index typed-load patch" >&2
                exit 1
            fi
        fi
        if ! grep -q "cache_head_dim" "$M3_INDEXER" \
            || ! grep -q "self._fp8_index" "$M3_AMD_MODEL" \
            || ! grep -q "kernel_index_cache" "$INDEX_TOPK_SOURCE" \
            || grep -q "float8e4b15" "$INDEX_TOPK_SOURCE"; then
            echo "MiniMax M3 FP8 index-cache markers are missing after patching" >&2
            exit 1
        fi
        python3 - "$INDEX_TOPK_SOURCE" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text()
score_launch = """        USE_PDL=use_pdl,
        USE_FP8=use_fp8,
        **score_launch,
"""
if score_launch not in source:
    raise SystemExit("FP8 index flags are not attached to the tuned score launch")
PY
        python3 -m py_compile "$M3_AMD_MODEL" "$M3_INDEXER" "$INDEX_TOPK_SOURCE"
        INDEX_KV_CACHE_ARGS=(--attention_config.indexer_kv_dtype=fp8)
        echo "M3 index KV-cache experiment mode: fp8"
        ;;
    *)
        echo "Invalid M3_INDEX_KV_CACHE_MODE: $M3_INDEX_KV_CACHE_MODE" >&2
        exit 2
        ;;
esac

M3_SHARED_EXPERT_STREAM_MODE="${M3_SHARED_EXPERT_STREAM_MODE:-off}"
export M3_SHARED_EXPERT_STREAM_MODE
case "$M3_SHARED_EXPERT_STREAM_MODE" in
    off)
        ;;
    on)
        # vLLM already implements shared/routed expert overlap with an
        # auxiliary stream. Enable the existing path on ROCm using the
        # one-line platform guard change from vLLM PR #38665.
        export VLLM_DISABLE_SHARED_EXPERTS_STREAM=0
        python3 /workspace/utils/patch_vllm_rocm_shared_experts_stream.py
        echo "M3 shared-expert stream experiment mode: on"
        ;;
    *)
        echo "Invalid M3_SHARED_EXPERT_STREAM_MODE: $M3_SHARED_EXPERT_STREAM_MODE" >&2
        exit 2
        ;;
esac

M3_DBO_MODE="${M3_DBO_MODE:-off}"
DBO_ARGS=()
case "$M3_DBO_MODE" in
    off)
        ;;
    decode|prefill|all)
        DBO_PATCH="$(dirname "$0")/minimaxm3_mi300x_dbo.patch"
        DBO_LOCAL_MOE_PATCH="$(dirname "$0")/minimaxm3_mi300x_dbo_local_moe.patch"
        DBO_COMM_SOURCE="$VLLM_PACKAGE_ROOT/vllm/distributed/communication_op.py"
        DBO_CONFIG_SOURCE="$VLLM_PACKAGE_ROOT/vllm/config/vllm.py"
        DBO_MOE_SOURCE="$VLLM_PACKAGE_ROOT/vllm/model_executor/layers/fused_moe/modular_kernel.py"
        DBO_LOCAL_MOE_SOURCE="$VLLM_PACKAGE_ROOT/vllm/model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py"
        if ! grep -q "_all_reduce_with_dbo_yields" "$DBO_COMM_SOURCE"; then
            if ! patch --batch --dry-run -d "$VLLM_PACKAGE_ROOT" -p1 \
                < "$DBO_PATCH"; then
                echo "Failed to validate the MiniMax M3 DBO patch" >&2
                exit 1
            fi
            if ! patch --batch -d "$VLLM_PACKAGE_ROOT" -p1 < "$DBO_PATCH"; then
                echo "Failed to apply the MiniMax M3 DBO patch" >&2
                exit 1
            fi
        fi
        if ! grep -q "def supports_dbo" "$DBO_MOE_SOURCE"; then
            if ! patch --batch --dry-run -d "$VLLM_PACKAGE_ROOT" -p1 \
                < "$DBO_LOCAL_MOE_PATCH"; then
                echo "Failed to validate the MiniMax M3 local-MoE DBO patch" >&2
                exit 1
            fi
            if ! patch --batch -d "$VLLM_PACKAGE_ROOT" -p1 \
                < "$DBO_LOCAL_MOE_PATCH"; then
                echo "Failed to apply the MiniMax M3 local-MoE DBO patch" >&2
                exit 1
            fi
        fi
        if ! grep -q "_all_reduce_with_dbo_yields" "$DBO_COMM_SOURCE" \
            || ! grep -q "Microbatching with DP+EP" "$DBO_CONFIG_SOURCE" \
            || ! grep -q "self.prepare_finalize.supports_dbo" "$DBO_MOE_SOURCE" \
            || ! grep -q "def supports_dbo" "$DBO_LOCAL_MOE_SOURCE" \
            || ! grep -q "dbo_prefill_min_seq_len" \
                "$VLLM_PACKAGE_ROOT/vllm/config/parallel.py" \
            || ! grep -q "num_ubatches > 1" \
                "$VLLM_PACKAGE_ROOT/vllm/v1/worker/gpu_model_runner.py"; then
            echo "MiniMax M3 DBO markers are missing after patching" >&2
            exit 1
        fi
        python3 -m py_compile \
            "$VLLM_PACKAGE_ROOT/vllm/config/parallel.py" \
            "$DBO_CONFIG_SOURCE" \
            "$DBO_COMM_SOURCE" \
            "$VLLM_PACKAGE_ROOT/vllm/engine/arg_utils.py" \
            "$DBO_MOE_SOURCE" \
            "$DBO_LOCAL_MOE_SOURCE" \
            "$VLLM_PACKAGE_ROOT/vllm/v1/worker/dp_utils.py" \
            "$VLLM_PACKAGE_ROOT/vllm/v1/worker/gpu_model_runner.py" \
            "$VLLM_PACKAGE_ROOT/vllm/v1/worker/gpu_ubatch_wrapper.py" \
            "$VLLM_PACKAGE_ROOT/vllm/v1/worker/ubatch_utils.py" \
            "$VLLM_PACKAGE_ROOT/vllm/v1/worker/ubatching.py"
        case "$M3_DBO_MODE" in
            decode)
                DBO_ARGS=(
                    --enable-dbo
                    --disable-custom-all-reduce
                    --dbo-decode-token-threshold 32
                    --dbo-prefill-token-threshold 2147483647
                )
                ;;
            prefill)
                DBO_ARGS=(
                    --enable-dbo
                    --disable-custom-all-reduce
                    --dbo-decode-token-threshold 2147483647
                    --dbo-prefill-token-threshold 512
                    --dbo-prefill-min-seq-len 2048
                )
                ;;
            all)
                DBO_ARGS=(
                    --enable-dbo
                    --disable-custom-all-reduce
                    --dbo-decode-token-threshold 32
                    --dbo-prefill-token-threshold 512
                    --dbo-prefill-min-seq-len 2048
                )
                ;;
        esac
        echo "M3 DBO experiment mode: $M3_DBO_MODE"
        ;;
    *)
        echo "Invalid M3_DBO_MODE: $M3_DBO_MODE" >&2
        exit 2
        ;;
esac

if [[ "$MODEL" != /* ]]; then hf download "$MODEL"; fi

if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

SERVER_LOG=/workspace/server.log
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_USE_BREAKABLE_CUDAGRAPH=0

M3_AITER_AR_RMS_MODE="${M3_AITER_AR_RMS_MODE:-off}"
M3_AITER_AR_RMS_REQUESTED_MODE="$M3_AITER_AR_RMS_MODE"
if [ "$M3_AITER_AR_RMS_MODE" = "fused" ] \
    && [ "${PROFILE:-0}" != "1" ] \
    && [ "$CONC" -eq 1 ]; then
    # Same-node graph benchmarks regress 2.2% at c1 with the two-stage AITER
    # primitive. Keep eager profiling available, but use the proven graph path
    # for production c1 runs.
    M3_AITER_AR_RMS_MODE=off
    echo "M3 AITER AR+RMS graph policy: using off for concurrency 1"
fi
export M3_AITER_AR_RMS_REQUESTED_MODE
export M3_AITER_AR_RMS_MODE
case "$M3_AITER_AR_RMS_MODE" in
    off)
        ;;
    control|fused)
        # Enable AITER only to make the fused allreduce+RMSNorm op available.
        # M3 does not support torch.compile, so the runtime patch calls this
        # one op directly from its existing helper. The model
        # patch also defers FFN/MoE output reductions into the following Gemma
        # norm so the same helper covers both transformer residual boundaries.
        # Keep every independently selectable AITER path disabled so the
        # control and fused runs differ only at that helper call.
        export VLLM_ROCM_USE_AITER=1
        export VLLM_ROCM_USE_AITER_PAGED_ATTN=0
        export VLLM_ROCM_USE_AITER_LINEAR=0
        export VLLM_ROCM_USE_AITER_LINEAR_HIPBMM=0
        export VLLM_ROCM_USE_AITER_MOE=0
        export VLLM_ROCM_USE_AITER_RMSNORM=0
        export VLLM_ROCM_USE_AITER_MLA=0
        export VLLM_ROCM_USE_AITER_MHA=0
        export VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=0
        export VLLM_ROCM_USE_AITER_TRITON_ROPE=0
        export VLLM_ROCM_USE_AITER_FP8BMM=0
        export VLLM_ROCM_USE_AITER_FP4BMM=0
        export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0
        export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
        export VLLM_ROCM_USE_AITER_TRITON_GEMM=0

        if [ "$M3_AITER_AR_RMS_MODE" = "fused" ]; then
            # The image's amd-aiter 0.1.13.post1 has incorrect memory ordering
            # in the two-stage fused allreduce+RMSNorm kernel. Install the
            # checksummed ROCm 7.2 wheel containing ROCm/aiter#2890.
            python3 /workspace/utils/install_minimaxm3_aiter.py
        fi

        python3 /workspace/utils/patch_minimaxm3_aiter_ar_rms.py

        DEFERRED_FFN_AR_PATCH="$(dirname "$0")/minimaxm3_mi300x_deferred_ffn_ar.patch"
        M3_MODEL_SOURCE="$VLLM_PACKAGE_ROOT/vllm/models/minimax_m3/amd/model.py"
        M3_MODEL_SOURCE_SHA256="91d81f8613e32f7afbd65c289f7885c5371263f70503bd053f97880989bf7536"
        M3_MODEL_PATCHED_SHA256="d26aa77cfce7c6162b0d1ebe2b403b854f5abe8f656b3a8deda2db1d89318ea8"
        m3_model_sha256="$(sha256sum "$M3_MODEL_SOURCE" | awk '{print $1}')"
        if [ "$m3_model_sha256" = "$M3_MODEL_SOURCE_SHA256" ]; then
            if ! patch --batch --dry-run -d "$VLLM_PACKAGE_ROOT" -p1 \
                < "$DEFERRED_FFN_AR_PATCH"; then
                echo "Failed to validate the M3 deferred FFN allreduce patch" >&2
                exit 1
            fi
            if ! patch --batch -d "$VLLM_PACKAGE_ROOT" -p1 \
                < "$DEFERRED_FFN_AR_PATCH"; then
                echo "Failed to apply the M3 deferred FFN allreduce patch" >&2
                exit 1
            fi
        elif [ "$m3_model_sha256" != "$M3_MODEL_PATCHED_SHA256" ]; then
            echo "M3 model source fingerprint mismatch: $m3_model_sha256" >&2
            exit 1
        fi
        python3 -m py_compile "$M3_MODEL_SOURCE"
        m3_model_sha256="$(sha256sum "$M3_MODEL_SOURCE" | awk '{print $1}')"
        if [ "$m3_model_sha256" != "$M3_MODEL_PATCHED_SHA256" ]; then
            echo "M3 model patched fingerprint mismatch: $m3_model_sha256" >&2
            exit 1
        fi
        echo "M3 deferred FFN allreduce patch ready: $m3_model_sha256"

        echo "M3 AITER AR+RMS experiment mode: $M3_AITER_AR_RMS_MODE"
        ;;
    *)
        echo "Invalid M3_AITER_AR_RMS_MODE: $M3_AITER_AR_RMS_MODE" >&2
        exit 2
        ;;
esac

if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
fi

PARALLEL_ARGS=(--tensor-parallel-size "$TP")
M3_EXPERT_PLACEMENT_STRATEGY="${M3_EXPERT_PLACEMENT_STRATEGY:-linear}"
case "$M3_EXPERT_PLACEMENT_STRATEGY" in
    linear)
        ;;
    round_robin)
        EXPERT_PLACEMENT_PATCH="$(dirname "$0")/minimaxm3_mi300x_ungrouped_round_robin.patch"
        EXPERT_MAP_MANAGER="$VLLM_PACKAGE_ROOT/vllm/model_executor/layers/fused_moe/expert_map_manager.py"
        FUSED_MOE_LAYER="$VLLM_PACKAGE_ROOT/vllm/model_executor/layers/fused_moe/layer.py"
        if ! grep -q "allow_ungrouped_round_robin" "$EXPERT_MAP_MANAGER"; then
            if ! patch --batch --dry-run -d "$VLLM_PACKAGE_ROOT" -p1 \
                < "$EXPERT_PLACEMENT_PATCH"; then
                echo "Failed to validate the M3 round-robin placement patch" >&2
                exit 1
            fi
            if ! patch --batch -d "$VLLM_PACKAGE_ROOT" -p1 \
                < "$EXPERT_PLACEMENT_PATCH"; then
                echo "Failed to apply the M3 round-robin placement patch" >&2
                exit 1
            fi
        fi
        if ! grep -q "allow_ungrouped_round_robin=True" "$M3_AMD_MODEL" \
            || ! grep -q "allow_ungrouped_round_robin=allow_ungrouped_round_robin" \
                "$FUSED_MOE_LAYER"; then
            echo "M3 round-robin placement opt-in markers are missing" >&2
            exit 1
        fi
        python3 -m py_compile \
            "$EXPERT_MAP_MANAGER" \
            "$FUSED_MOE_LAYER" \
            "$M3_AMD_MODEL"
        echo "M3 expert placement experiment mode: round_robin"
        ;;
    *)
        echo "Invalid M3_EXPERT_PLACEMENT_STRATEGY: $M3_EXPERT_PLACEMENT_STRATEGY" >&2
        exit 2
        ;;
esac
if [ "${DP_ATTENTION}" = "true" ]; then
    PARALLEL_ARGS=(
        --tensor-parallel-size 1
        --data-parallel-size "$TP"
        --enable-expert-parallel
        --expert-placement-strategy "$M3_EXPERT_PLACEMENT_STRATEGY"
    )
elif [ "$EP_SIZE" -gt 1 ]; then
    PARALLEL_ARGS+=(
        --enable-expert-parallel
        --expert-placement-strategy "$M3_EXPERT_PLACEMENT_STRATEGY"
    )
fi

PROFILE_ARGS=()
if [ "${PROFILE:-0}" = "1" ]; then
    profile_token_budget="${M3_PROFILE_TOKEN_BUDGET:-8192}"
    case "$profile_token_budget" in
        8192|16384|32768)
            ;;
        *)
            echo "Invalid M3_PROFILE_TOKEN_BUDGET: $profile_token_budget" >&2
            exit 2
            ;;
    esac
    M3_PROFILE_PHASE="${M3_PROFILE_PHASE:-decode}"
    case "$M3_PROFILE_PHASE" in
        decode)
            profile_prefill_iterations=$(( (ISL * CONC + profile_token_budget - 1) / profile_token_budget ))
            profile_delay=$((profile_prefill_iterations + 16))
            profile_description="one steady-state decode iteration after $profile_delay engine iterations"
            ;;
        prefill)
            profile_delay=0
            profile_description="the first ${profile_token_budget}-token chunked-prefill iteration"
            ;;
        *)
            echo "Invalid M3_PROFILE_PHASE: $M3_PROFILE_PHASE" >&2
            exit 2
            ;;
    esac
    benchmark_num_prompts="$CONC"
    export VLLM_TORCH_PROFILER_DIR="${VLLM_TORCH_PROFILER_DIR:-/tmp/inferencex-profile/${RESULT_FILENAME}}"
    rm -rf "$VLLM_TORCH_PROFILER_DIR"
    mkdir -p "$VLLM_TORCH_PROFILER_DIR"

    profiler_config="$(
        printf '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":false,"torch_profiler_with_flops":false,"torch_profiler_use_gzip":true,"torch_profiler_dump_cuda_time_total":false,"torch_profiler_record_shapes":false,"torch_profiler_with_memory":false,"ignore_frontend":true,"delay_iterations":%d,"max_iterations":1}' \
            "$VLLM_TORCH_PROFILER_DIR" "$profile_delay"
    )"
    PROFILE_ARGS=(
        --max-num-batched-tokens "$profile_token_budget"
        --profiler-config "$profiler_config"
        --compilation-config '{"cudagraph_mode":"NONE"}'
    )
    # ROCTracer does not expose every kernel launched inside a HIP graph.
    echo "Profiling $profile_description."
else
    benchmark_num_prompts="$((CONC * 10))"
fi

start_gpu_monitor

set -x
vllm serve "$MODEL" --port "$PORT" \
    "${PARALLEL_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
    "${KV_CACHE_ARGS[@]}" \
    "${INDEX_KV_CACHE_ARGS[@]}" \
    "${DBO_ARGS[@]}" \
    --block-size 128 \
    --no-enable-prefix-caching \
    --language-model-only \
    --max-model-len "$MAX_MODEL_LEN" \
    --attention-backend TRITON_ATTN \
    --tool-call-parser minimax_m3 \
    --reasoning-parser minimax_m3 \
    --enable-auto-tool-choice > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$benchmark_num_prompts" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --trust-remote-code

if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

stop_gpu_monitor
set +x
