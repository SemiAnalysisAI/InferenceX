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
# REQUIRED on ROCm per the upstream recipe: the build auto-enables this to 1.
export VLLM_USE_BREAKABLE_CUDAGRAPH=0

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
    LMCACHE_VERSION="0.5.5.dev11+rocm7.2"
    LMCACHE_ROCM_INDEX="https://github.com/LMCache/LMCache/releases/expanded_assets/nightly-rocm"
    agentic_pip_install --quiet --no-cache-dir --no-deps \
        "sortedcontainers==2.4.0" \
        "opentelemetry-exporter-prometheus==0.61b0" \
        "cupy-rocm-7-0==14.1.1" \
        "lmcache==${LMCACHE_VERSION}" --find-links "$LMCACHE_ROCM_INDEX"
    python3 -c \
        "import cupy; import lmcache.integration.vllm.lmcache_mp_connector; import opentelemetry.exporter.prometheus" \
        >/dev/null

    # One MP server for the node, per the Kimi-K3 recipe
    # (docs.lmcache.ai/recipes/kimi_k3.html), with --chunk-size sized for
    # THIS stack rather than the recipe's CUDA-path 768: the connector
    # requires the chunk to be a multiple of every engine KV group's
    # tokens_per_block, and the hybrid KDA/MLA layout here registers
    # attention groups at 1536 ("Setting attention block size to 1536",
    # run 31644990546) plus a KDA state group at 3072 (run 31645828378),
    # so 3072 is the minimum valid chunk. The multi-group layout also
    # requires one object group per sliding-window size:
    # --separate-object-groups.
    LMCACHE_PORT=6555
    LMCACHE_HTTP_PORT=8090
    LMCACHE_LOG="$RESULT_DIR/lmcache_server.log"

    LMCACHE_L1_SIZE_GB="$TOTAL_CPU_DRAM_GB"

    LMCACHE_CMD=(
        lmcache server
        --host 127.0.0.1
        --port "$LMCACHE_PORT"
        --http-host 127.0.0.1
        --http-port "$LMCACHE_HTTP_PORT"
        --l1-size-gb "$LMCACHE_L1_SIZE_GB"
        --l1-init-size-gb 10
        --chunk-size 3072
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
# TODO: merge 51171 to main
# apply_pr51171.sh
# -----------------------------------------------------------------------------
# Apply vLLM PR #51171 ("Reach FULL cudagraphs for AITER MLA speculative
# decoding") to the vLLM installed inside the container image
#   vllm/vllm-openai-rocm:nightly-a3561ef8e49d3545c4078df43444beb4c98ae124
#   (vllm 0.26.1rc1.dev1120+ga3561ef8e)
#
# This is the patch that gives the Kimi-K3 DSpark spec-decode path FULL
# cudagraphs (TritonMLA._cudagraph_support -> UNIFORM_BATCH) instead of the
# PIECEWISE downgrade -> ~3.5x (conc8) / ~4.7x (conc1) TPOT, acceptance
# unchanged.
#
# Safe to `source` OR `bash`. Idempotent. Backs up each patched file to
# <file>.orig51171. Only touches the container's site-packages vLLM.
#
# On this base the diff applies with fuzz EXCEPT gpu_worker.py Hunk #1 (an
# import) whose context differs; that single import line is added here instead.
#
# Usage:
#   # inside the container, from the dir holding pr51171.diff:
#   source apply_pr51171.sh
#   # or point it at the diff explicitly:
#   PR51171_DIFF=/path/to/pr51171.diff bash apply_pr51171.sh
#   # if no local diff is found it is fetched from GitHub.
# -----------------------------------------------------------------------------

apply_pr51171() {
    local src diff sp f n

    # --- 1. locate the PR diff -------------------------------------------------
    src="${PR51171_DIFF:-}"
    if [ -z "$src" ]; then
        for c in ./pr51171.diff "$HOME/pr51171.diff" "$(dirname "${BASH_SOURCE:-$0}" 2>/dev/null)/pr51171.diff"; do
            [ -f "$c" ] && { src="$c"; break; }
        done
    fi
    if [ -z "$src" ] || [ ! -f "$src" ]; then
        echo "[pr51171] no local pr51171.diff; fetching from GitHub ..."
        src="$(mktemp /tmp/pr51171.XXXXXX.diff)"
        curl -fsSL "https://github.com/vllm-project/vllm/pull/51171.diff" -o "$src" \
            || { echo "[pr51171] ERROR: fetch failed; set PR51171_DIFF=/path/to/pr51171.diff" >&2; return 1; }
    fi

    # keep only sections that patch files under vllm/ (drop tests/ etc.)
    diff="$(mktemp /tmp/pr51171_vllm.XXXXXX.diff)"
    awk '/^diff --git /{keep=($3 ~ /^a\/vllm\//)} keep' "$src" > "$diff"
    [ -s "$diff" ] || { echo "[pr51171] ERROR: diff has no vllm/ hunks: $src" >&2; return 1; }

    # --- 2. locate installed vLLM ---------------------------------------------
    sp="$(python -c 'import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))' 2>/dev/null)"
    if [ -z "$sp" ] || [ ! -d "$sp/vllm" ]; then
        echo "[pr51171] ERROR: cannot locate installed vllm" >&2; return 1
    fi
    echo "[pr51171] target: $sp/vllm  (vllm $(python -c 'import vllm;print(vllm.__version__)' 3>/dev/null))"

    # --- 3. apply the diff (idempotent) ---------------------------------------
    local tm="$sp/vllm/v1/attention/backends/mla/triton_mla.py"
    if grep -q "AttentionCGSupport.UNIFORM_BATCH" "$tm" 2>/dev/null; then
        echo "[pr51171] already applied (triton_mla UNIFORM_BATCH present); skipping patch step."
    else
        echo "[pr51171] applying diff ..."
        # --forward skips already-applied hunks; the gpu_worker import Hunk#1
        # is expected to fail on this base (handled in step 4), so tolerate rc!=0.
        patch -p1 --forward --fuzz=3 -b -z .orig51171 -d "$sp" < "$diff" || true
        find "$sp/vllm/v1/worker" "$sp/vllm/v1/attention/backends/mla" \
            -name '*.rej' -delete 2>/dev/null
    fi

    # --- 4. ensure the get_kv_cache_capacity import (gpu_worker Hunk #1) -------
    f="$sp/vllm/v1/worker/gpu_worker.py"
    if ! grep -q "from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity" "$f"; then
        if grep -q "from vllm.utils.torch_utils import set_random_seed" "$f"; then
            sed -i "/from vllm.utils.torch_utils import set_random_seed/a from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity" "$f"
            echo "[pr51171] added get_kv_cache_capacity import to gpu_worker.py"
        else
            echo "[pr51171] WARN: import anchor not found in gpu_worker.py; add manually:" >&2
            echo "         from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity" >&2
        fi
    fi

    # --- 5. verify -------------------------------------------------------------
    if python -m py_compile \
        "$sp/vllm/v1/attention/backends/mla/rocm_aiter_mla.py" \
        "$tm" "$f" 2>/dev/null; then
        n="$(grep -c UNIFORM_BATCH "$tm")"
        echo "[pr51171] py_compile OK; triton_mla UNIFORM_BATCH markers=$n (expect >=1)."
        echo "[pr51171] done — restart the vLLM server for it to take effect."
        rm -f "$diff"; [ "${src#/tmp/}" != "$src" ] && rm -f "$src"
        return 0
    else
        echo "[pr51171] ERROR: py_compile FAILED — inspect the patched files." >&2
        return 1
    fi
}

apply_pr51171


# ---- Parallelism ------------------------------------------------------------
EP_ARGS=()
if [ "$EP_SIZE" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

# ---- Speculative ------------------------------------------------------------
if [ "$CONC" = 1 ]; then
    SYNTHETIC_ACCEPT_LEN=3.75
    SPEC_NUM_TOKENS=6
else
    SYNTHETIC_ACCEPT_LEN=2.51
    SPEC_NUM_TOKENS=2
fi


if [ "${EVAL_ONLY:-false}" = "true" ]; then
    SPEC_ARGS=(
        --speculative-config
        "{\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\": \"block\"}"
    )
else
    SPEC_ARGS=(
        --speculative-config
        "{\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
    )
fi

# ---- HIP graph ------------------------------------------------------------
MAX_NUM_SEQS=20
MAX_CUDAGRAPH_CAPTURE_SIZE=60
CUDAGRAPH_CAPTURE_SIZES="$(seq -s, 2 "$MAX_CUDAGRAPH_CAPTURE_SIZE")"
COMPILATION_CONFIG_ARGS=(--compilation-config "{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"max_cudagraph_capture_size\":$MAX_CUDAGRAPH_CAPTURE_SIZE,\"custom_ops\":[\"+fused_rms_norm_gated\"],\"cudagraph_capture_sizes\":[$CUDAGRAPH_CAPTURE_SIZES]}")

echo "Starting vllm server..."
export PYTHONNOUSERSITE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1200}"

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
    --gpu-memory-utilization 0.95
    --language-model-only
    --max-num-seqs "$MAX_NUM_SEQS"
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --max-model-len 1048576
    --enable-prefix-caching
    --kv-cache-dtype "fp8"
    --max-num-batched-tokens 16384
    "${COMPILATION_CONFIG_ARGS[@]}"
    "${SPEC_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
)
    #--attention-config '{"mla_prefill_backend":"ROCM_AITER_FA"}'
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
