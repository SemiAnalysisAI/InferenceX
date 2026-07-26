#!/usr/bin/env bash
set -euo pipefail
set -x

# AgentX trace replay for the full NVIDIA GLM-5.2 NVFP4 checkpoint on the
# 8x RTX PRO 6000 node. The digest-pinned Gilded Gnosis v20 image carries the
# SM120 B12X sparse-MLA, DCP, and ModelOpt NVFP4 stack used on this GPU family:
# https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm5.2_v20.md

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    DCP_SIZE \
    CONC \
    KV_OFFLOADING \
    TOTAL_CPU_DRAM_GB \
    RESULT_DIR \
    DURATION \
    EP_SIZE \
    DP_ATTENTION

if [[ "$TP" != "8" || "$DCP_SIZE" != "4" ]]; then
    echo "GLM-5.2 RTX AgentX requires TP8 with DCP4" >&2
    exit 1
fi
if [[ "$KV_OFFLOADING" != "none" ]]; then
    echo "GLM-5.2 RTX AgentX initially supports GPU-resident KV cache only" >&2
    exit 1
fi
if [[ "$DP_ATTENTION" == "true" ]]; then
    echo "GLM-5.2 RTX AgentX does not support DP attention" >&2
    exit 1
fi
if [[ "$EP_SIZE" != "1" && "$EP_SIZE" != "$TP" ]]; then
    echo "GLM-5.2 RTX AgentX supports EP1 or TP8+EP8 only" >&2
    exit 1
fi

# Gilded Gnosis v20 releases the source B12X MoE parameters after preparing
# its packed expert owner. The generic modular wrapper consequently reports
# zero local experts during memory profiling, while the prepared owner retains
# the authoritative count. Apply the narrowly scoped EP fix directly to the
# installed vLLM package until the pinned image includes it.
apply_glm52_b12x_ep_patch() {
    local patch_file
    local site_packages
    local target
    local current_sha256
    local patch_sha256
    local expected_patch_sha256=dd0fec7546830a403951e2a1780683f0157b3590192417eb799a9a2e291c95a3
    local base_sha256=97619dfabe6e1a34017a328cf353a1a54988dce50c4b6040e4e6eb32d25599ae
    local result_sha256=2df39f0be834abce3096ed1a09e8e1dd13addcf0e52ff4fdb099fd11ab610310

    patch_file="$(
        cd "$(dirname "$0")/../../patches/vllm"
        pwd
    )/glm52_b12x_ep_released_source.patch"
    patch_sha256="$(sha256sum "$patch_file" | awk '{print $1}')"
    if [[ "$patch_sha256" != "$expected_patch_sha256" ]]; then
        echo "GLM-5.2 B12X EP patch file checksum mismatch: $patch_sha256" >&2
        exit 1
    fi

    site_packages="$(
        python3 -c \
            'import importlib.util, pathlib; spec = importlib.util.find_spec("vllm"); assert spec and spec.origin; print(pathlib.Path(spec.origin).parent.parent)'
    )"
    target="$site_packages/vllm/model_executor/layers/fused_moe/b12x_ep_moe.py"
    current_sha256="$(sha256sum "$target" | awk '{print $1}')"

    case "$current_sha256" in
        "$base_sha256")
            patch --batch --forward --fuzz=0 -p1 -d "$site_packages" < "$patch_file"
            ;;
        "$result_sha256")
            echo "GLM-5.2 B12X EP vLLM patch already applied"
            ;;
        *)
            echo "Installed vLLM does not match the expected B12X EP source" >&2
            echo "Expected: $base_sha256 or $result_sha256" >&2
            echo "Actual:   $current_sha256" >&2
            exit 1
            ;;
    esac

    current_sha256="$(sha256sum "$target" | awk '{print $1}')"
    if [[ "$current_sha256" != "$result_sha256" ]]; then
        echo "GLM-5.2 B12X EP vLLM patch verification failed" >&2
        exit 1
    fi
    python3 -m py_compile "$target"
}

if [[ "$EP_SIZE" -gt 1 ]]; then
    apply_glm52_b12x_ep_patch
fi

SERVED_MODEL_NAME="$MODEL"
MODEL_REVISION="${GLM52_MODEL_REVISION:-aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa}"
TARGET_MODEL_PATH="${MODEL_PATH:-$MODEL}"
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --revision "$MODEL_REVISION" --local-dir "$MODEL_PATH"
    fi
elif [[ "$TARGET_MODEL_PATH" != /* ]]; then
    hf download "$TARGET_MODEL_PATH" --revision "$MODEL_REVISION"
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi
nvidia-smi
nvidia-smi topo -m || true

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
resolve_trace_source
install_agentic_deps

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export PYTHONNOUSERSITE=1
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

# Gilded Gnosis v20 / SparkInfer runtime contract. Its custom PCIe all-reduce
# keeps GPU P2P available; the runner-level NCCL_IB_DISABLE=1 only avoids the
# node's broken bnxt_re probe.
export CUDA_DEVICE_MAX_CONNECTIONS=32
export CUTE_DSL_ARCH=sm_120a
export TORCH_CUDA_ARCH_LIST=12.0a
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SAFETENSORS_FAST_GPU=1
export INSTANTTENSOR_BACKEND=BUFFERED
export VLLM_USE_AOT_COMPILE=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_MEGA_AOT_ARTIFACT=1
export VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_USE_B12X_WO_PROJECTION=1
export VLLM_USE_B12X_MHC=1
export VLLM_USE_B12X_FP8_GEMM=1
export VLLM_USE_B12X_MOE=1
export VLLM_USE_B12X_SPARSE_INDEXER=1
export VLLM_USE_B12X_DCP_A2A=1
export VLLM_DCP_A2A_MAX_TOKENS=64
export VLLM_DCP_A2A_LARGE_BACKEND=ag_rs
export VLLM_DCP_PROJECT_BEFORE_MERGE=1
export VLLM_DCP_PROJECT_BEFORE_MERGE_MIN_PREFILL_TOKENS=1024
export VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE=1
export VLLM_DCP_QUERY_SPLIT=1
export VLLM_B12X_MLA_CKV_GATHER=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_PCIE_ALLREDUCE=1
export VLLM_PCIE_ALLREDUCE_BACKEND=b12x
export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB
export VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE=84KB
export VLLM_USE_B12X_PCIE_DMA=1
export VLLM_PCIE_DMA_FP8=0
export B12X_PCIE_DMA_FP8=0
export VLLM_DCP_GLOBAL_TOPK=1
export VLLM_NF3_GRID188_DECODE=1
export B12X_MLA_SM120_UNIFIED=1
export B12X_DENSE_SPLITK_TURBO=1
export B12X_W4A16_TC_DECODE=1
export B12X_W4A8_TINY_DECODE=1
export B12X_MOE_FORCE_A8=0
export B12X_MOE_FORCE_A16=1
export NCCL_PROTO=LL,LL128,Simple
export NCCL_P2P_LEVEL=SYS
export LD_PRELOAD=/opt/libnccl-local-inference.so.2.30.4
export VLLM_NCCL_SO_PATH=/opt/libnccl-local-inference.so.2.30.4

unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS

export TMPDIR="${TMPDIR:-/container-tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$XDG_CACHE_HOME/vllm}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-$VLLM_CACHE_ROOT}"
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$XDG_CACHE_HOME/tilelang}"
export TILELANG_TMP_DIR="${TILELANG_TMP_DIR:-$TILELANG_CACHE_DIR/tmp}"
export TVM_CACHE_DIR="${TVM_CACHE_DIR:-$XDG_CACHE_HOME/tvm}"
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-$XDG_CACHE_HOME/tvm-ffi}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$XDG_CACHE_HOME/torch_extensions}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$XDG_CACHE_HOME/flashinfer}"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="${VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR:-$XDG_CACHE_HOME/flashinfer-autotune}"
export CUTE_DSL_CACHE_DIR="${CUTE_DSL_CACHE_DIR:-$XDG_CACHE_HOME/cute-dsl}"
export B12X_CUTE_COMPILE_CACHE_DIR="${B12X_CUTE_COMPILE_CACHE_DIR:-$XDG_CACHE_HOME/b12x-cute}"
export DG_JIT_CACHE_DIR="${DG_JIT_CACHE_DIR:-$XDG_CACHE_HOME/deep-gemm}"
export MM_SPARSE_ATTN_AOT_CACHE="${MM_SPARSE_ATTN_AOT_CACHE:-$XDG_CACHE_HOME/minfer/mm_sparse_attn}"
export MINFER_FMHA_CACHE_DIR="${MINFER_FMHA_CACHE_DIR:-$XDG_CACHE_HOME/minfer/fmha_sm100}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$XDG_CACHE_HOME/cuda}"
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-$XDG_CACHE_HOME/cupy}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$XDG_CACHE_HOME/numba}"

mkdir -p \
    "$RESULT_DIR" \
    "$TMPDIR" \
    "$VLLM_CACHE_DIR" \
    "$TILELANG_CACHE_DIR" \
    "$TILELANG_TMP_DIR" \
    "$TVM_CACHE_DIR" \
    "$TVM_FFI_CACHE_DIR" \
    "$TRITON_CACHE_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR" \
    "$TORCH_EXTENSIONS_DIR" \
    "$FLASHINFER_WORKSPACE_BASE" \
    "$VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR" \
    "$CUTE_DSL_CACHE_DIR" \
    "$B12X_CUTE_COMPILE_CACHE_DIR" \
    "$DG_JIT_CACHE_DIR" \
    "$MM_SPARSE_ATTN_AOT_CACHE" \
    "$MINFER_FMHA_CACHE_DIR" \
    "$CUDA_CACHE_PATH" \
    "$CUPY_CACHE_DIR" \
    "$NUMBA_CACHE_DIR"

# AgentX concurrency counts live session trees. Subagent fan-out can create
# more simultaneous requests, so retain the standard 2x scheduler headroom.
MAX_NUM_SEQS=$((2 * CONC))
MAX_CUDAGRAPH_CAPTURE_SIZE=$((4 * MAX_NUM_SEQS))
MAX_MODEL_LEN=1048576
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.96}"

EP_ARGS=()
if [[ "$EP_SIZE" -gt 1 ]]; then
    # Gilded Gnosis v20's B12X EP path shards experts across the TP group,
    # computes from replicated inputs, and reduces the local expert outputs.
    EP_ARGS=(--enable-expert-parallel)
fi

REVISION_ARGS=()
if [[ "$TARGET_MODEL_PATH" != /* ]]; then
    REVISION_ARGS=(--revision "$MODEL_REVISION")
fi

# Exact full/shared schedule from NVIDIA's pinned checkpoint indexer_types.
GLM52_INDEX_TOPK_PATTERN=FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS
HF_OVERRIDES="$(
    printf '{"use_index_cache":true,"index_topk_pattern":"%s"}' \
        "$GLM52_INDEX_TOPK_PATTERN"
)"

SERVER_LOG="$RESULT_DIR/server.log"
VLLM_CMD=(
    vllm serve "$TARGET_MODEL_PATH"
    "${REVISION_ARGS[@]}"
    --served-model-name "$SERVED_MODEL_NAME"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tensor-parallel-size "$TP"
    "${EP_ARGS[@]}"
    --decode-context-parallel-size "$DCP_SIZE"
    --dcp-comm-backend a2a
    --dcp-kv-cache-interleave-size 1
    --kv-cache-dtype fp8
    --attention-backend B12X_MLA_SPARSE
    --moe-backend b12x
    --quantization modelopt_fp4
    --load-format instanttensor
    -cc.pass_config.fuse_allreduce_rms=True
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens 8192
    --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE"
    --async-scheduling
    --enable-chunked-prefill
    --enable-prefix-caching
    --enable-flashinfer-autotune
    --enable-auto-tool-choice
    --tool-call-parser glm47
    --reasoning-parser glm45
    --default-chat-template-kwargs '{"reasoning_effort":"high"}'
    --enable-prompt-tokens-details
    --enable-force-include-usage
    --enable-request-id-headers
    --hf-overrides "$HF_OVERRIDES"
)

printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup_agentic_server() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    exit "$exit_code"
}
trap cleanup_agentic_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "${EVAL_ONLY:-false}" == "true" ]]; then
    export SWEBENCH_AGENT_STEP_LIMIT=150
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
