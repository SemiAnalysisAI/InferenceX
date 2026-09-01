#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K3 MXFP4 on MI355X / MI350X (gfx950)
# using vLLM.
#
# The server command is the AMD reference `vllm serve` for this model, i.e. the
# upstream vLLM recipe's amd block (vllm-project/recipes,
# https://recipes.vllm.ai/moonshotai/Kimi-K3) as run in practice:
#
#   --trust-remote-code --moe-backend auto --tensor-parallel-size 8
#   --load-format auto --gpu-memory-utilization 0.95 --mm-encoder-tp-mode data
#   --max-num-seqs 128 --max-num-batched-tokens 4096 --enable-auto-tool-choice
#   --tool-call-parser kimi_k3 --reasoning-parser kimi_k3
#
# with env VLLM_ROCM_USE_AITER=1 SAFETENSORS_FAST_GPU=1 AITER_SITUV2_A8W4=1
# AITER_BF16_FP8_MOE_BOUND=0 VLLM_USE_BREAKABLE_CUDAGRAPH=0.
#
# K3 is a 2.8T-parameter natively-multimodal MoE (896 routed experts, 16/token
# plus shared) on Kimi Delta Attention, gated MLA and Attention Residuals, with
# a 1M-token native context.
#
# TP=8 ONLY. The MXFP4 checkpoint is 1.561 TB decimal (1.420 TiB, 96
# safetensors), ~195 GB/GPU across 8 GPUs of the 288 GB part; TP=4 would need
# ~390 GB/GPU and cannot load. Upstream strategy_min_gpus agrees (single_node_tp
# and multi_node_tep both 8, DEP 16+), which is why there is no DP-attention arm.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION,
#   EP_SIZE
#
# Perf-search knobs. Each defaults to the reference command's value, so an
# otherwise-unset run reproduces the reference exactly:
#   GPU_MEM_UTIL             0.95   (reference)
#   MAX_NUM_BATCHED_TOKENS   8192   (default)
#   AITER_A8W4               1      (reference; 0 = aiter a16w4 MoE path)
#   LANGUAGE_MODEL_ONLY      true   
#   KV_CACHE_DTYPE           fp8    (default for every arm; =auto for a bf16 A/B)
#   KV_BLOCK_SIZE            unset  (unset -> vLLM sizes the page; 128 under fp8)
#   MAX_MODEL_LEN            1M     
#   SPEC_DECODE              true   (this is the _mtp DSpark recipe; =false for a no-spec A/B)
#   SPEC_NUM_TOKENS          2      (DSpark draft length; validated by the _mtp config)

source "$(dirname "$0")/../../benchmark_lib.sh"

wait_for_amd_gpu_clean

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [ "$TP" -ne 8 ]; then
    echo "Error: Kimi-K3 MXFP4 is a 1.56 TB checkpoint and only fits at TP=8 on" >&2
    echo "       288 GB gfx950 parts (~195 GB/GPU). Got TP=$TP." >&2
    exit 1
fi

# ROCR/HIP visibility for vLLM 0.14+
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

# `hf download` creates the target dir if missing and is itself idempotent. The
# 1.56 TB checkpoint is normally pre-staged, so these calls are a no-op there.
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
resolve_trace_source
install_agentic_deps

# ---- Reference env block ----------------------------------------------------
export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
export VLLM_ROCM_USE_AITER=1
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
export AITER_SITUV2_A8W4=1
export AITER_BF16_FP8_MOE_BOUND=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export AITER_QUICK_REDUCE_QUANTIZATION=INT4

# Workaround for MEC FW <177 RCCL memory reclaim issue (shared with the other
# gfx950 recipes in this tree).
mec_version=$(rocm-smi --showfw 2>/dev/null | grep MEC | head -n 1 | awk '{print $NF}')
if [[ "$mec_version" == "" || ${mec_version:-0} -lt 177 ]]; then
    export HSA_NO_SCRATCH_RECLAIM=1
fi

# 2.8T of weights off a shared/NFS mount takes far longer than the default.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"

# Long agentic turns against a 1M context: keep the client from timing out
# mid-request while the server is prefill-bound.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

# ---- Server config ----------------------------------------------------------
SERVER_LOG="$RESULT_DIR/server.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
LMCACHE_PID=""

cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    stop_background_process_tree "$LMCACHE_PID" "LMCache server"
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---- KV offload -------------------------------------------------------------
# TOTAL_CPU_DRAM_GB is the aggregate host-DRAM budget the matrix generator
# derives from dram-utilization and the runner's available-cpu-dram-mib, capped
# at the 3,095,781 MiB (3 TB decimal) agentic limit. Per
# benchmarks/single_node/agentic/README.md it must be consumed as given and
# never replaced with a model-specific constant.
OFFLOAD_ARGS=()

if agentic_kv_offload_enabled; then
case "${KV_OFFLOAD_BACKEND:-}" in
  vllm-simple)
    require_agentic_kv_offload_backend "$KV_OFFLOAD_BACKEND"
    CPU_BYTES_PER_RANK=$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / TP ))
    # Identical prefixes must hash to identical block keys across ranks.
    export PYTHONHASHSEED=42
    SIMPLE_LAZY_OFFLOAD="${SIMPLE_LAZY_OFFLOAD:-false}"
    OFFLOAD_ARGS=(
        --kv-transfer-config
        "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":$CPU_BYTES_PER_RANK,\"lazy_offload\":$SIMPLE_LAZY_OFFLOAD}}"
    )
    echo "SimpleCPUOffloadConnector: ${CPU_BYTES_PER_RANK} B/rank x ${TP} ranks, lazy_offload=$SIMPLE_LAZY_OFFLOAD"
    ;;
      lmcache)
    require_agentic_kv_offload_backend "$KV_OFFLOAD_BACKEND"

    # Keep the image's tested torch/ROCm stack and install only LMCache's
    # missing runtime dependencies, same as the MiniMax-M3 lmcache arm.
    LMCACHE_VERSION="0.5.5.dev63+rocm7.2"
    export KV_OFFLOAD_BACKEND_METADATA="{\"name\":\"lmcache\",\"version\":\"${LMCACHE_VERSION}\"}"
    LMCACHE_ROCM_INDEX="https://github.com/LMCache/LMCache/releases/expanded_assets/nightly-rocm"
    agentic_pip_install --quiet --no-cache-dir --no-deps \
        "sortedcontainers==2.4.0" \
        "opentelemetry-exporter-prometheus==0.61b0" \
        "cupy-rocm-7-0==14.1.1" \
        "lmcache==${LMCACHE_VERSION}" --find-links "$LMCACHE_ROCM_INDEX"

    SITE_PACKAGES=$(python3 -c \
        'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent.parent)')

    # LMCache #4834 supports the non-trivial DCP interleave selected by vLLM,
    # resolves hybrid KV-group geometry from the engine configuration, and
    # namespaces cache keys by DCP layout. Pin the exact reviewed revision.
    LMCACHE_PR4834_BASE="1af7803551ab05905cb2c46fba403e1e5c1de575"
    LMCACHE_PR4834_HEAD="347b0c8a780d550de24fdd508174597e603b3af2"
    LMCACHE_PR4834_SHA256="25820903e7208fb1a7f263ca41e448e4f041f4af8e88f942dd766f549776a651"
    LMCACHE_PR4834_SOURCE_SHA256="095f39d10bdc3dfa0f917140ee77c6c007c6c829e58675ffc32f72f796774201"
    LMCACHE_PR4834_PATCH=$(mktemp)
    LMCACHE_PR4834_SOURCE_PATCH=$(mktemp)
    curl --fail --location --silent --show-error \
        "https://github.com/LMCache/LMCache/compare/${LMCACHE_PR4834_BASE}...${LMCACHE_PR4834_HEAD}.diff" \
        --output "$LMCACHE_PR4834_PATCH"
    echo "$LMCACHE_PR4834_SHA256  $LMCACHE_PR4834_PATCH" | sha256sum --check
    awk '/^diff --git a\/lmcache\// {keep=1} \
        /^diff --git / && $0 !~ /^diff --git a\/lmcache\// {keep=0} keep' \
        "$LMCACHE_PR4834_PATCH" > "$LMCACHE_PR4834_SOURCE_PATCH"
    echo "$LMCACHE_PR4834_SOURCE_SHA256  $LMCACHE_PR4834_SOURCE_PATCH" | \
        sha256sum --check
    patch -d "$SITE_PACKAGES" -p1 --forward --batch < \
        "$LMCACHE_PR4834_SOURCE_PATCH"

    # vLLM #51705 adds ROCm AITER MLA decode LSE output for DCP and keeps full
    # CUDA graphs enabled. Pin both compare endpoints and the downloaded diff
    # checksum so this experiment cannot silently follow later PR revisions.
    VLLM_PR51705_BASE="e0d27040ddcc5ac31cf01c5b04a7d764ccba656d"
    VLLM_PR51705_HEAD="e1843114b7c233a9c71ad44b28bf63426ad64836"
    VLLM_PR51705_SHA256="88a31df8cd0ffd308aa4a5c550142734939a8d83c7132cfa924f8f80b2387b55"
    VLLM_PR51705_SOURCE_SHA256="063ee0140d559b54da665b6ef95a3bea64b33d6eaff6cb9a827616a6cc4d8890"
    VLLM_PR51705_PATCH=$(mktemp)
    VLLM_PR51705_SOURCE_PATCH=$(mktemp)
    curl --fail --location --silent --show-error \
        "https://github.com/vllm-project/vllm/compare/${VLLM_PR51705_BASE}...${VLLM_PR51705_HEAD}.diff" \
        --output "$VLLM_PR51705_PATCH"
    echo "$VLLM_PR51705_SHA256  $VLLM_PR51705_PATCH" | sha256sum --check
    sed -n '/^diff --git a\/vllm\//,$p' "$VLLM_PR51705_PATCH" > \
        "$VLLM_PR51705_SOURCE_PATCH"
    echo "$VLLM_PR51705_SOURCE_SHA256  $VLLM_PR51705_SOURCE_PATCH" | \
        sha256sum --check
    patch -d "$SITE_PACKAGES" -p1 --forward --batch < \
        "$VLLM_PR51705_SOURCE_PATCH"

    # LMCache 0.5.5's transfer-channel layer eagerly imports the Mooncake
    # backend (mooncake_te_impl.py -> `from mooncake.engine import
    # TransferEngine`), whose native .so resolves all of its DT_NEEDED libs at
    # import. The vLLM ROCm image ships none of them, so the import sanity
    # check below (and the LMCache server) would otherwise fail with
    # "ImportError: lib*.so: cannot open shared object file" (first libglog,
    # then libjsoncpp, ...). Provision Mooncake's full runtime lib set from the
    # distro before importing. apt-get install is idempotent, so run it
    # whenever any of the libs is still missing rather than gating on one.
    LMCACHE_NATIVE_LIBS=(libglog.so.0 libjsoncpp.so.25 libibverbs.so.1 librdmacm.so.1 libnuma.so.1)
    for lib in "${LMCACHE_NATIVE_LIBS[@]}"; do
        if ! ldconfig -p | grep -q "$lib"; then
            apt-get update
            apt-get install -y \
                libgoogle-glog0v5 libjsoncpp25 libibverbs1 librdmacm1 libnuma1
            break
        fi
    done
    python3 -c \
        "import cupy; from lmcache.integration.vllm.lmcache_mp_connector import get_lmcache_model_name, get_lmcache_scheduler_block_size; assert callable(get_lmcache_model_name); assert callable(get_lmcache_scheduler_block_size); from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAImpl; assert AiterMLAImpl.can_return_lse_for_decode; assert AiterMLAImpl.lse_base_on_e; import opentelemetry.exporter.prometheus" \
        >/dev/null

    # One MP server for the node, per the Kimi-K3 recipe
    # (docs.lmcache.ai/recipes/kimi_k3.html), with --chunk-size sized for
    # THIS stack rather than the recipe's CUDA-path 768: the connector
    # requires the chunk to be a multiple of every engine KV group's
    # tokens_per_block. The hybrid KDA/MLA layout registers attention groups
    # at 1536 tokens and a KDA state group at 3072. Under DCP, LMCache scales
    # the attention group by DCP_SIZE, so DCP8 requires 1536 * 8 = 12288,
    # which is also divisible by the KDA group size. The multi-group layout
    # also requires one object group per sliding-window size:
    # --separate-object-groups.
    LMCACHE_PORT=6555
    LMCACHE_HTTP_PORT=8090
    LMCACHE_LOG="$RESULT_DIR/lmcache_server.log"

    LMCACHE_L1_SIZE_GB="$TOTAL_CPU_DRAM_GB"
    LMCACHE_CHUNK_SIZE=$((1536 * ${DCP_SIZE:-8}))

    LMCACHE_CMD=(
        lmcache server
        --host 127.0.0.1
        --port "$LMCACHE_PORT"
        --http-host 127.0.0.1
        --http-port "$LMCACHE_HTTP_PORT"
        --l1-size-gb "$LMCACHE_L1_SIZE_GB"
        --l1-init-size-gb 10
        --chunk-size "$LMCACHE_CHUNK_SIZE"
        --separate-object-groups
        --enable-extra-logging
        --extra-logging-interval 30
        --max-cpu-workers 8
        --max-gpu-workers 1
        --eviction-policy LRU
        --supported-transfer-mode lmcache_driven
        --shm-name ""
    )
    append_command "$RESULT_DIR/lmcache_command.txt" "${LMCACHE_CMD[@]}"
    "${LMCACHE_CMD[@]}" > "$LMCACHE_LOG" 2>&1 &
    LMCACHE_PID=$!
    wait_for_ready \
        --endpoint "http://127.0.0.1:${LMCACHE_HTTP_PORT}/healthcheck" \
        --log "$LMCACHE_LOG" \
        --pid "$LMCACHE_PID" \
        --sleep-interval 1 \
        --timeout 600

    # 100k-330k-token agentic prefixes make single retrieves large; use the
    # same MQ timeout headroom as the MiniMax-M3 arm.
    OFFLOAD_ARGS=(
        --kv-transfer-config
        "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.port\":$LMCACHE_PORT,\"lmcache.mp.mq_timeout\":6000.0}}"
    )
    ;;
    *)
    echo "Error: unsupported KV_OFFLOAD_BACKEND='$KV_OFFLOAD_BACKEND' (expected vllm-simple or lmcache)" >&2
    exit 1
    ;;
esac
fi

# ---- LLM server  ------------------------------------------------------------

# ---- Parallelism ------------------------------------------------------------
EP_ARGS=()
if [ "$EP_SIZE" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

# ---- Speculative / Util------------------------------------------------------
case "${SPEC_DECODING:-mtp}:$CONC" in
    # No KV offload; the working set fits in HBM.
    mtp:1)
        SYNTHETIC_ACCEPT_LEN=3.75
        SPEC_NUM_TOKENS=6
        GPU_MEM_UTIL=0.9
        MAX_NUM_BATCHED_TOKENS=16384
        ;;
    mtp:2|mtp:4|mtp:8|mtp:10|mtp:12|mtp:14|mtp:40)
        SYNTHETIC_ACCEPT_LEN=3.00
        SPEC_NUM_TOKENS=3
        GPU_MEM_UTIL=0.9
        MAX_NUM_BATCHED_TOKENS=8192
        ;;
    none:40)
        SPEC_NUM_TOKENS=0
        GPU_MEM_UTIL=0.9
        MAX_NUM_BATCHED_TOKENS=16384
        ;;
    *)
        SPEC_NUM_TOKENS=0
        GPU_MEM_UTIL=0.85
        MAX_NUM_BATCHED_TOKENS=4096
        ;;
esac

SPEC_ARGS=()
if [ "$SPEC_NUM_TOKENS" -gt 0 ]; then
if [ "${EVAL_ONLY:-false}" = "true" ]; then
    SPEC_ARGS=(
        --speculative-config
        "{\"model\":\"${DRAFT_MODEL:-Inferact/Kimi-K3-DSpark}\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"kv_cache_dtype\":\"fp8\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\": \"block\"}"
    )
else
    SPEC_ARGS=(
        --speculative-config
        "{\"model\":\"${DRAFT_MODEL:-Inferact/Kimi-K3-DSpark}\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"kv_cache_dtype\":\"fp8\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
    )
    fi
fi

# ---- HIP graph ------------------------------------------------------------
MAX_NUM_SEQS=$((2 * CONC))
if [ "${SPEC_DECODING:-mtp}:$CONC" = "none:40" ]; then
    MAX_CUDAGRAPH_CAPTURE_SIZE=4096
    CUDAGRAPH_CAPTURE_SIZES="$(seq -s, 1 "$MAX_NUM_SEQS"),128,256,512,1024,2048,4096"
else
    MAX_CUDAGRAPH_CAPTURE_SIZE=$((MAX_NUM_SEQS * (1 + SPEC_NUM_TOKENS)))
    CUDAGRAPH_CAPTURE_SIZES="$(seq -s, 2 "$MAX_CUDAGRAPH_CAPTURE_SIZE")"
fi
COMPILATION_CONFIG_ARGS=(--compilation-config "{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"max_cudagraph_capture_size\":$MAX_CUDAGRAPH_CAPTURE_SIZE,\"custom_ops\":[\"+fused_rms_norm_gated\",\"+quant_fp8\",\"+grouped_topk\",\"+sparse_attn_indexer\",\"none\"],\"cudagraph_capture_sizes\":[$CUDAGRAPH_CAPTURE_SIZES]}")

echo "Starting vllm server..."
export PYTHONNOUSERSITE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1200}"


# ---- DCP       ------------------------------------------------------------
# DCP shards decode KV across the TP ranks, so it must divide TP.
DCP_SIZE="${DCP_SIZE:-8}"
if [ $((TP % DCP_SIZE)) -ne 0 ]; then
    echo "Error: TP='$TP' must be divisible by DCP_SIZE='$DCP_SIZE'" >&2
    exit 1
fi
CP_ARGS=()
ATTN_BE_ARGS=()
if [ "$DCP_SIZE" -gt 1 ]; then
    CP_KV_CACHE_INTERLEAVE_SIZE=1
    if [ "${KV_OFFLOAD_BACKEND:-}" = "lmcache" ]; then
        # vLLM otherwise adjusts this to the resolved 1536-token attention
        # block only in worker-local config copies. Pass it explicitly so the
        # LMCache scheduler and workers use the same DCP cache namespace.
        CP_KV_CACHE_INTERLEAVE_SIZE=1536
    fi
    CP_ARGS+=(
        --decode-context-parallel-size "$DCP_SIZE"
        --dcp-comm-backend a2a
        --cp-kv-cache-interleave-size "$CP_KV_CACHE_INTERLEAVE_SIZE"
    )
    ATTN_BE_ARGS+=(--attention-backend ROCM_AITER_MLA)
fi
export VLLM_USE_DIRECT_DCP_A2A=0
export VLLM_USE_DIRECT_DCP_Q_GATHER=0
export VLLM_USE_DIRECT_DCP_KV_GATHER=0
export VLLM_ALLOW_DCP_FULL_CUDAGRAPH=1
export PREFIX_CACHING_HASH_ALGO=sha256

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --moe-backend auto
    --tensor-parallel-size "$TP"
    "${EP_ARGS[@]}"
    --load-format fastsafetensors
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --language-model-only
    --max-num-seqs "$MAX_NUM_SEQS"
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --max-model-len 1048576
    --stream-interval 10
    --enable-prefix-caching
    --prefix-match-unit 128
    --kv-cache-dtype "fp8"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --attention-config '{"mla_prefill_backend":"ROCM_AITER_FA","use_prefill_query_quantization":true}'
    "${ATTN_BE_ARGS[@]}"
    "${COMPILATION_CONFIG_ARGS[@]}"
    "${SPEC_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
    "${CP_ARGS[@]}"
)
printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY:-false}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
