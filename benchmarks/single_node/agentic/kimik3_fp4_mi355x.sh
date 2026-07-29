#!/usr/bin/env bash
set -euo pipefail
set -x

# Agentic trace replay benchmark for Kimi-K3 MXFP4 on MI355X / MI350X (gfx950)
# using vLLM.
#
# The server command is the AMD reference `vllm serve` for this model, i.e. the
# upstream vLLM recipe's amd block (vllm-project/recipes,
# models/moonshotai/Kimi-K3.yaml, date_updated 2026-07-25) as run in practice:
#
#   --trust-remote-code --moe-backend auto --tensor-parallel-size 8
#   --load-format auto --gpu-memory-utilization 0.95 --mm-encoder-tp-mode data
#   --max-num-seqs 128 --max-num-batched-tokens 4096 --enable-auto-tool-choice
#   --tool-call-parser kimi_k3 --reasoning-parser kimi_k3
#
# with env VLLM_ROCM_USE_AITER=1 SAFETENSORS_FAST_GPU=1 AITER_SITUV2_A8W4=1
# AITER_BF16_FP8_MOE_BOUND=0 VLLM_USE_BREAKABLE_CUDAGRAPH=0.
#
# The DRAM-offload arm adds LMCache's LMCacheMPConnector against a local
# `lmcache server`, again matching the reference command.
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
# KV_OFFLOADING=dram requires KV_OFFLOAD_BACKEND=lmcache. Mooncake is
# deliberately unsupported: the upstream recipe marks both
# kv_store_distributed_mooncake and kv_store_centralized_mooncake as
# `unsupported` on every hardware target for this model.
#
# Perf-search knobs. Each defaults to the reference command's value, so an
# otherwise-unset run reproduces the reference exactly:
#   GPU_MEM_UTIL             0.95   (reference)
#   MAX_NUM_SEQS             128    (reference)
#   MAX_NUM_BATCHED_TOKENS   4096   (reference)
#   AITER_A8W4               1      (reference; 0 = aiter a16w4 MoE path)
#   LANGUAGE_MODEL_ONLY      false  (reference loads the vision tower)
#   KV_CACHE_DTYPE           auto   (unset -> flag not passed at all)
#   MAX_MODEL_LEN            unset  (unset -> vLLM derives K3's 1M context)
#   SPEC_DECODE              false  (DSpark; UNVALIDATED on ROCm)

source "$(dirname "$0")/../../benchmark_lib.sh"

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
# kimik3* is on resolve_trace_source's unfiltered allowlist (benchmark_lib.sh),
# so this replays the full 062126 v7 corpus rather than the 256k-capped variant.
resolve_trace_source
install_agentic_deps

# ---- Reference env block ----------------------------------------------------
export VLLM_ROCM_USE_AITER=1
export SAFETENSORS_FAST_GPU=1
# AITER a8w4 MoE path for the MXFP4-weight / MXFP8-activation QAT checkpoint.
# Upstream: "set AITER_SITUV2_A8W4 to 0 along with AITER master flag to use
# aiter a16w4 MoE path. Set it to 1 to use aiter a8w4 MoE path." Swept.
export AITER_SITUV2_A8W4="${AITER_A8W4:-1}"
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
LMCACHE_LOG="$RESULT_DIR/lmcache_server.log"
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

wait_for_lmcache_ready() {
    { set +x; } 2>/dev/null
    # Generous by default. With memory pinning unavailable on ROCm, LMCache
    # disables its LazyMemoryAllocator and allocates the whole L1 pool UP FRONT,
    # so "ready" can be minutes away and scales with --l1-size-gb. A 180s budget
    # timed out at exactly 178s on a 2249 GB pool.
    local attempts="${LMCACHE_READY_ATTEMPTS:-1800}"
    # The health route has moved across LMCache versions, and a wrong path is
    # indistinguishable from a slow start: both just never return 200. Probe the
    # known spellings and report which one answered.
    local paths=(/healthcheck /health /v1/health /status /)
    local i p
    for ((i = 1; i <= attempts; i++)); do
        for p in "${paths[@]}"; do
            if curl --output /dev/null --silent --fail \
                    "http://127.0.0.1:${LMCACHE_HTTP_PORT}${p}"; then
                echo "LMCache server healthy after ${i}s (endpoint ${p})"
                set -x
                return 0
            fi
        done
        if [[ -n "$LMCACHE_PID" ]] && ! kill -0 "$LMCACHE_PID" 2>/dev/null; then
            echo "LMCache server died before becoming healthy. Log follows:" >&2
            cat "$LMCACHE_LOG" >&2 || true
            exit 1
        fi
        # Heartbeat so a slow eager allocation is visibly distinct from a hang.
        if (( i % 60 == 0 )); then
            echo "  ... still waiting for LMCache (${i}s / ${attempts}s), L1=${LMCACHE_L1_SIZE_GB:-?}GB"
        fi
        sleep 1
    done
    echo "Timed out after ${attempts}s waiting for LMCache healthcheck." >&2
    echo "Tried endpoints: ${paths[*]} on port ${LMCACHE_HTTP_PORT}. Log follows:" >&2
    cat "$LMCACHE_LOG" >&2 || true
    exit 1
}

# ---- KV offload -------------------------------------------------------------
# TOTAL_CPU_DRAM_GB is the aggregate host-DRAM budget the matrix generator
# derives from dram-utilization and the runner's available-cpu-dram-mib, capped
# at the 2,861,022 MiB (3 TB decimal) agentic limit. Per
# benchmarks/single_node/agentic/README.md it must be consumed as given and
# never replaced with a model-specific constant.
OFFLOAD_ARGS=()

if agentic_kv_offload_enabled; then
case "${KV_OFFLOAD_BACKEND:-}" in
  lmcache|lmcache-k27|lmcache-budget)
    require_agentic_kv_offload_backend "$KV_OFFLOAD_BACKEND"
    # The server profile has to be selectable from config: the agentic matrix
    # has no per-cell env channel, so the backend NAME carries it.
    #   lmcache      -> AMD K3 reference server command
    #   lmcache-k27  -> kimik2.7_fp4_mi355x.sh's server command
    if [ "$KV_OFFLOAD_BACKEND" = "lmcache-k27" ]; then
        LMCACHE_PROFILE=k2.7
    fi
    # lmcache-budget: same reference server, but L1 sized from the agentic DRAM
    # budget (shm-capped) instead of the reference's flat 512 GB -- i.e. what
    # kimik2.7 does, which ran a 1199 GB pool successfully on this fleet. The
    # agentic README also asks recipes to consume TOTAL_CPU_DRAM_GB rather than
    # a constant. Kept as a separate backend name because the matrix has no
    # per-cell env channel to vary it from config.
    if [ "$KV_OFFLOAD_BACKEND" = "lmcache-budget" ]; then
        LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-$TOTAL_CPU_DRAM_GB}"
    fi

    # LMCache on K3 REQUIRES prefix caching. K3's Kimi Delta Attention layers
    # are Mamba KV-cache groups, and vLLM only selects mamba_cache_mode='align'
    # -- the mode that keeps reusable state snapshots -- when prefix caching is
    # enabled. Without it the connector refuses to initialise at KV-transfer
    # setup with:
    #   ValueError: LMCache cannot serve this model's KV cache groups:
    #   group N: MambaSpec with mamba_cache_mode='none' (only 'align' keeps
    #   reusable state snapshots)
    # (lmcache/integration/vllm/kv_cache_group_edits.py:137). Measured on
    # gfx950 with LMCache v0.5.3.dev47. This is a hard dependency, not a tuning
    # choice, so force it here rather than leaving it to the caller -- the
    # agentic matrix has no per-cell env channel to set it from config.
    if [ "${PREFIX_CACHING:-}" = "false" ]; then
        echo "Error: PREFIX_CACHING=false is incompatible with KV_OFFLOAD_BACKEND=lmcache." >&2
        echo "       LMCache needs mamba_cache_mode='align', which vLLM only selects when" >&2
        echo "       prefix caching is enabled. Use KV_OFFLOADING=none to measure without it." >&2
        exit 1
    fi
    PREFIX_CACHING=true

    # --max-num-batched-tokens and --chunk-size stay at the AMD reference
    # values (4096 / 1024). They were briefly pinned to 768 because LMCache
    # dev HEAD (v0.5.3.dev47) rejects the reference values on this hybrid model:
    #   ValueError: Mamba-hybrid models with LMCache require
    #     block_size <= max_num_batched_tokens < 2 * block_size ... block_size=768
    #   AssertionError: LMCache chunk size should be a multiple of vLLM block size
    # CONFIRMED on 0.5.1 as well (run 30348060242, g09): the pinned release
    # raises the same ValueError, so these are properties of the LMCache/vLLM
    # Mamba-hybrid integration, not of dev HEAD. Only the LazyMemoryAllocator
    # stall was dev-specific. Both stay pinned to 768 for the LMCache arms.
    #
    # Worth keeping in view either way: at ~106k-token average ISL, 768 would
    # mean ~138 chunked-prefill steps per turn versus ~26 at 4096, so the two
    # settings are not performance-equivalent.
    MAX_NUM_BATCHED_TOKENS="${LMCACHE_MAX_NUM_BATCHED_TOKENS:-768}"
    LMCACHE_K3_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE_OVERRIDE:-768}"
    LMCACHE_CHUNK_SIZE="$LMCACHE_K3_CHUNK_SIZE"
    LMCACHE_CHUNK_SIZE_K27="$LMCACHE_K3_CHUNK_SIZE"
    echo "LMCache: --max-num-batched-tokens=$MAX_NUM_BATCHED_TOKENS --chunk-size=$LMCACHE_K3_CHUNK_SIZE (reference values)"

    # LMCache is NOT in the kimi-k3 image (verified: no `lmcache` module and no
    # CLI), so build it against ROCm. Clone to a container-local dir, NOT the
    # bind-mounted /workspace, so a later job's `clean: true` checkout does not
    # trip over root-owned build artifacts.
    #
    # The matrix passes kv-offload-backend as JSON in KV_OFFLOAD_BACKEND_METADATA
    # (e.g. {"name":"lmcache","version":"<sha>"}). Honour its `version` as the
    # build ref so a version A/B is a config change rather than a recipe edit;
    # an explicit LMCACHE_GIT_REF still wins, and the pin below is the fallback.
    LMCACHE_CFG_VERSION=""
    if [ -n "${KV_OFFLOAD_BACKEND_METADATA:-}" ]; then
        LMCACHE_CFG_VERSION=$(KV_META="$KV_OFFLOAD_BACKEND_METADATA" python3 -c '
import json, os
try:
    d = json.loads(os.environ["KV_META"])
    print(d.get("version", "") if isinstance(d, dict) else "")
except Exception:
    print("")
' 2>/dev/null || true)
    fi

    # The AMD reference recipe pins a PyPI release: `uv pip install
    # "lmcache==0.5.1"`. Do not substitute a git build of dev HEAD -- dev
    # (v0.5.3.dev47) carries a LazyMemoryAllocator that expands the pinned L1
    # pool ~10 GB per ~17 s DURING serving, which starves the vLLM worker until
    # the executor RPC deadline fires ("RPC call to sample_tokens timed out",
    # observed on g09 and g11). It also adds Mamba-hybrid constraints
    # (block_size <= max_num_batched_tokens < 2*block_size, and
    # chunk_size % block_size == 0) that the reference command does not satisfy.
    LMCACHE_VERSION="${LMCACHE_VERSION:-${LMCACHE_CFG_VERSION:-0.5.1}}"
    if ! python3 -c "import lmcache.integration.vllm.lmcache_mp_connector" >/dev/null 2>&1; then
        echo "Installing lmcache==$LMCACHE_VERSION"
        if command -v uv >/dev/null 2>&1; then
            uv pip install --system "lmcache==$LMCACHE_VERSION" \
                || agentic_pip_install --quiet "lmcache==$LMCACHE_VERSION"
        else
            agentic_pip_install --quiet "lmcache==$LMCACHE_VERSION"
        fi
        python3 -c "import lmcache.integration.vllm.lmcache_mp_connector" >/dev/null
    fi
    python3 -c "import lmcache; print('lmcache', getattr(lmcache,'__version__','?'))" || true

    LMCACHE_HOST="${LMCACHE_HOST:-127.0.0.1}"
    LMCACHE_PORT="${LMCACHE_PORT:-5555}"
    LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"

    # L1 is SHM-backed: if it exceeds free /dev/shm, LMCache silently disables
    # SHM and falls back to a pickle path that crashes at load. Cap at 90% of
    # free /dev/shm so SHM stays enabled, and say so loudly -- the capped value
    # is the number that actually backs the run.
    LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-512}"   # reference value

    # Optional ceiling on the eager allocation. Memory pinning is unavailable in
    # the ROCm container ("CudaPinMemoryBackend: neither torch cudart nor
    # libcudart is available"), so LMCache disables its LazyMemoryAllocator and
    # allocates the ENTIRE L1 pool before serving a request.
    #
    # Default is NO ceiling, matching kimik2.7_fp4_mi355x.sh, which sizes L1 to
    # TOTAL_CPU_DRAM_GB capped only by /dev/shm and successfully ran a 1199 GB
    # pool on this same fleet. The agentic README also requires consuming
    # TOTAL_CPU_DRAM_GB rather than substituting a model-specific constant. Set
    # LMCACHE_L1_MAX_GB to impose one (the AMD reference command uses 512).
    LMCACHE_L1_MAX_GB="${LMCACHE_L1_MAX_GB:-0}"
    if [ "$LMCACHE_L1_MAX_GB" -gt 0 ] && [ "$LMCACHE_L1_SIZE_GB" -gt "$LMCACHE_L1_MAX_GB" ]; then
        echo "WARNING: capping LMCACHE_L1_SIZE_GB ${LMCACHE_L1_SIZE_GB} -> ${LMCACHE_L1_MAX_GB}" \
             "(eager allocation ceiling). The offload pool for this cell is" \
             "${LMCACHE_L1_MAX_GB}G, NOT the ${TOTAL_CPU_DRAM_GB}G the matrix budgeted."
        LMCACHE_L1_SIZE_GB="$LMCACHE_L1_MAX_GB"
    fi

    SHM_FREE_GB=$(df -BG --output=avail /dev/shm 2>/dev/null | tail -1 | tr -dc '0-9')
    if [ -n "$SHM_FREE_GB" ] && [ "$SHM_FREE_GB" -gt 0 ]; then
        SHM_CAP_GB=$(( SHM_FREE_GB * 90 / 100 ))
        if [ "$LMCACHE_L1_SIZE_GB" -gt "$SHM_CAP_GB" ]; then
            echo "WARNING: capping LMCACHE_L1_SIZE_GB ${LMCACHE_L1_SIZE_GB} -> ${SHM_CAP_GB}" \
                 "to fit /dev/shm (${SHM_FREE_GB}G free). The offload pool for this" \
                 "cell is ${SHM_CAP_GB}G, not the ${TOTAL_CPU_DRAM_GB}G the matrix budgeted."
            LMCACHE_L1_SIZE_GB="$SHM_CAP_GB"
        fi
    fi

    LMCACHE_L1_INIT_SIZE_GB="${LMCACHE_L1_INIT_SIZE_GB:-20}"
    # --max-gpu-workers 1 avoids concurrent-GPU-transfer stalls under heavy
    # async-load pressure; CPU-side workers stay at 8.
    LMCACHE_MAX_GPU_WORKERS="${LMCACHE_MAX_GPU_WORKERS:-1}"
    LMCACHE_MAX_CPU_WORKERS="${LMCACHE_MAX_CPU_WORKERS:-8}"
    LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-1024}"
    LMCACHE_L1_ALIGN_BYTES="${LMCACHE_L1_ALIGN_BYTES:-16384}"
    LMCACHE_EVICTION_WATERMARK="${LMCACHE_EVICTION_WATERMARK:-0.85}"
    LMCACHE_EVICTION_RATIO="${LMCACHE_EVICTION_RATIO:-0.10}"
    LMCACHE_MQ_TIMEOUT="${LMCACHE_MQ_TIMEOUT:-300}"
    # Identical prefixes must hash to identical block keys across ranks.
    export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

    # Two server profiles. `reference` is the AMD K3 reference command.
    # `k2.7` reproduces kimik2.7_fp4_mi355x.sh's server exactly -- the one
    # configuration known to have served this agentic trace on this fleet
    # (1199 GB L1, TP4) -- so the two can be A/B'd without a recipe edit.
    # The K2.7 flags all still exist on LMCache dev: --max-workers lives in
    # lmcache/v1/multiprocess/config.py alongside the newer split
    # --max-gpu-workers/--max-cpu-workers, and --l1-read-ttl-seconds in
    # lmcache/v1/distributed/config.py.
    LMCACHE_PROFILE="${LMCACHE_PROFILE:-reference}"
    echo "Starting LMCache MP server (profile=$LMCACHE_PROFILE, L1=${LMCACHE_L1_SIZE_GB}GB)..."
    case "$LMCACHE_PROFILE" in
      k2.7)
        export LMCACHE_BLOCKING_TIMEOUT_SECS="${LMCACHE_BLOCKING_TIMEOUT_SECS:-60}"
        LMCACHE_CMD=(
            lmcache server
            --host "$LMCACHE_HOST"
            --port "$LMCACHE_PORT"
            --http-host "$LMCACHE_HOST"
            --http-port "$LMCACHE_HTTP_PORT"
            --l1-size-gb "$LMCACHE_L1_SIZE_GB"
            --l1-init-size-gb "$LMCACHE_L1_INIT_SIZE_GB"
            --l1-read-ttl-seconds "${LMCACHE_L1_READ_TTL_SECONDS:-7200}"
            --chunk-size "${LMCACHE_CHUNK_SIZE_K27:-256}"
            --max-workers "${LMCACHE_MAX_WORKERS:-$((TP * 2))}"
            --eviction-policy LRU
        )
        ;;
      reference)
        LMCACHE_CMD=(
            lmcache server
            --host "$LMCACHE_HOST"
            --port "$LMCACHE_PORT"
            --http-host "$LMCACHE_HOST"
            --http-port "$LMCACHE_HTTP_PORT"
            --l1-size-gb "$LMCACHE_L1_SIZE_GB"
            --l1-init-size-gb "$LMCACHE_L1_INIT_SIZE_GB"
            --max-gpu-workers "$LMCACHE_MAX_GPU_WORKERS"
            --max-cpu-workers "$LMCACHE_MAX_CPU_WORKERS"
            --chunk-size "$LMCACHE_CHUNK_SIZE"
            --l1-align-bytes "$LMCACHE_L1_ALIGN_BYTES"
            --eviction-trigger-watermark "$LMCACHE_EVICTION_WATERMARK"
            --eviction-ratio "$LMCACHE_EVICTION_RATIO"
            --eviction-policy LRU
            --supported-transfer-mode lmcache_driven
        )
        ;;
      *)
        echo "Error: unsupported LMCACHE_PROFILE '$LMCACHE_PROFILE' (expected: reference, k2.7)" >&2
        exit 1
        ;;
    esac
    printf '%q ' "${LMCACHE_CMD[@]}" > "$RESULT_DIR/lmcache_command.txt"
    printf '\n' >> "$RESULT_DIR/lmcache_command.txt"
    "${LMCACHE_CMD[@]}" > "$LMCACHE_LOG" 2>&1 &
    LMCACHE_PID=$!
    echo "LMCache server PID: $LMCACHE_PID"
    wait_for_lmcache_ready

    # LMCacheMPConnector is registered in this image's vLLM (verified against
    # KVConnectorFactory), so the reference profile needs no
    # kv_connector_module_path. The k2.7 profile passes it (and the ZMQ-style
    # lmcache.mp.host) exactly as kimik2.7_fp4_mi355x.sh does.
    if [ "$LMCACHE_PROFILE" = "k2.7" ]; then
        LMCACHE_CONNECT_HOST="${LMCACHE_CONNECT_HOST:-tcp://$LMCACHE_HOST}"
        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"$LMCACHE_CONNECT_HOST\",\"lmcache.mp.port\":$LMCACHE_PORT}}"
        )
    else
        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.port\":$LMCACHE_PORT,\"lmcache.mp.mq_timeout\":$LMCACHE_MQ_TIMEOUT}}"
        )
    fi
    ;;
  vllm-simple|vllm-simple-fp8)
    require_agentic_kv_offload_backend "$KV_OFFLOAD_BACKEND"
    # vllm-simple-fp8 is the same connector with an fp8 KV cache. fp8 halves
    # bytes/token in the GPU pool, which on this KV-bound corpus moves the
    # eviction wall itself rather than just adding headroom (measured: the pool
    # peaks at 98.6-99.8% usage even at c8). ROCm maps fp8 -> fp8_e4m3.
    # UNVALIDATED on K3's hybrid geometry: the pool spans Kimi Delta Attention
    # state and gated-MLA latent, and fp8 support across both spec types is
    # unconfirmed on this build.
    if [ "$KV_OFFLOAD_BACKEND" = "vllm-simple-fp8" ]; then
        KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
    fi
    # vLLM's own SimpleCPUOffloadConnector -- the AMD reference's native
    # offload path. Unlike LMCache it does not go through the Mamba
    # [conv_state, ssm_state] adapter that K3's Kimi Delta Attention breaks
    # ("expected a Mamba [conv_state, ssm_state] tensor list, got Tensor",
    # runs 30350911388), so it is the only offload tier that can currently
    # initialise on this model.
    #
    # cpu_bytes_to_use is server-wide; cpu_bytes_to_use_per_rank overrides it
    # per rank (simple_cpu_offload_connector.py:66). The agentic README requires
    # consuming TOTAL_CPU_DRAM_GB, so derive it rather than hardcode the
    # reference's 236223201280 (= 220 GiB/rank, 1760 GiB across 8 ranks).
    SIMPLE_RANKS="${GPU_COUNT:-$TP}"
    CPU_BYTES_PER_RANK="${SIMPLE_CPU_BYTES_PER_RANK:-$(( TOTAL_CPU_DRAM_GB * 1000 * 1000 * 1000 / SIMPLE_RANKS ))}"
    # Identical prefixes must hash to identical block keys across ranks.
    export PYTHONHASHSEED=42
    # lazy_offload MUST be a JSON boolean. The reference command passes the
    # STRING "false", and the connector does
    #   lazy_offload = bool(extra_config.get("lazy_offload", False))
    # (simple_cpu_offload_connector.py:77) -- bool("false") is True in Python,
    # so the string silently selects LAZY, the opposite of what it reads as.
    # Default to real eager offload; SIMPLE_LAZY_OFFLOAD=true swaps it.
    # The connector logs "lazy"/"eager" at line 95, so server.log confirms which
    # mode actually engaged.
    SIMPLE_LAZY_OFFLOAD="${SIMPLE_LAZY_OFFLOAD:-false}"
    OFFLOAD_ARGS=(
        --kv-transfer-config
        "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":$CPU_BYTES_PER_RANK,\"lazy_offload\":$SIMPLE_LAZY_OFFLOAD}}"
    )
    echo "SimpleCPUOffloadConnector: ${CPU_BYTES_PER_RANK} B/rank x ${SIMPLE_RANKS} ranks, lazy_offload=$SIMPLE_LAZY_OFFLOAD"
    ;;
  vllm-native)
    require_agentic_kv_offload_backend vllm-native
    # OffloadingConnector, vLLM's other native tier. --kv_offloading_size is
    # GiB (vllm/config/vllm.py multiplies by 1<<30) while TOTAL_CPU_DRAM_GB is
    # decimal GB, so convert or we over-request by ~7.4%.
    unset VLLM_USE_SIMPLE_KV_OFFLOAD
    KV_OFFLOAD_GIB=$(( TOTAL_CPU_DRAM_GB * 1000000000 / 1073741824 ))
    OFFLOAD_ARGS=(
        --kv_offloading_backend native
        --kv_offloading_size "$KV_OFFLOAD_GIB"
    )
    echo "OffloadingConnector: ${KV_OFFLOAD_GIB} GiB"
    ;;
  mooncake)
    echo "Error: Mooncake is unsupported for Kimi-K3. The upstream recipe marks" >&2
    echo "       kv_store_{distributed,centralized}_mooncake as 'unsupported' on" >&2
    echo "       every hardware target for this model." >&2
    exit 1
    ;;
  *)
    echo "Error: unsupported KV_OFFLOAD_BACKEND '${KV_OFFLOAD_BACKEND:-}' (expected: lmcache, lmcache-k27)" >&2
    exit 1
    ;;
esac
fi

# ---- Parallelism ------------------------------------------------------------
# TP8 or TEP8. No DP-attention arm: upstream strategy_min_gpus.multi_node_dep is
# 16, so DEP is not a single-node strategy for this model.
EP_ARGS=()
if [ "$EP_SIZE" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

# ---- Multimodal vs text-only ------------------------------------------------
# The reference command loads the vision tower and passes --mm-encoder-tp-mode
# data, so that is the default here. --language-model-only is an upstream
# opt_in_feature ("skip the vision encoder for text-only workloads") and the
# agentic corpus never sends an image, so it is a swept axis.
#
# Note this build's help describes the flag more narrowly than upstream's
# phrasing: "disables all multimodal inputs by setting all modality limits to 0.
# Equivalent to setting --limit-mm-per-prompt to 0 for every modality" -- input
# gating, which does not by itself guarantee the vision tower goes unloaded.
# Whether it returns HBM to the KV pool is measured by comparing
# "model weights take N GiB" in server.log across both settings. Upstream marks
# it mutually exclusive with encoder parallelism, so --mm-encoder-tp-mode is
# only passed when multimodal inputs are enabled.
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-false}"
MM_ARGS=(--mm-encoder-tp-mode data)
if [ "$LANGUAGE_MODEL_ONLY" = "true" ]; then
    MM_ARGS=(--language-model-only)
fi

# ---- Optional axes ----------------------------------------------------------
# Only emitted when set away from the reference, so the default command line is
# byte-for-byte the reference one.
#
# fp8 KV halves bytes/token in the pool, which moves the KV-capacity wall itself
# rather than just adding headroom -- the dominant effect on a prefill-heavy 1M
# context corpus. Not on by default because K3's KV geometry is HYBRID (Kimi
# Delta Attention state + gated-MLA latent) and fp8 across both spec types is
# unconfirmed on this build.
KV_CACHE_DTYPE_ARGS=()
if [ -n "${KV_CACHE_DTYPE:-}" ] && [ "${KV_CACHE_DTYPE}" != "auto" ]; then
    KV_CACHE_DTYPE_ARGS=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi

# Left unset by default so vLLM derives K3's native 1M context, which is what
# the unfiltered corpus needs. Set explicitly only to test truncation effects.
MAX_MODEL_LEN_ARGS=()
if [ -n "${MAX_MODEL_LEN:-}" ] && [ "${MAX_MODEL_LEN}" != "0" ]; then
    MAX_MODEL_LEN_ARGS=(--max-model-len "$MAX_MODEL_LEN")
fi

# The reference command passes neither --enable-prefix-caching nor
# --no-enable-prefix-caching, and this build's default is None (vLLM decides
# internally), so by default we pass nothing and stay aligned. Two reasons this
# is a knob rather than a hardcode: agentic trace replay exists to exercise
# large shared prefixes, so the resolved value must be confirmed from
# server.log; and K3 is hybrid (Kimi Delta Attention + gated MLA), where block
# and hash sizes only align with prefix caching on -- an omission has been
# reported to trip "tokens_per_block not divisible by tokens_per_hash" at load.
# Set PREFIX_CACHING=true/false to force it either way.
# ON by default for EVERY arm. This trace is built around large shared
# prefixes -- theoretical prefix-cache hit is 98.1%, and a live kvnone cell
# measured 92.8% server-side -- so a run with reuse disabled is not measuring
# the workload. Reuse also costs essentially no KV (1,414,660 vs 1,420,824
# tokens) and improves ITL (484 vs 577 ms). The offload arms additionally
# require it: LMCache needs mamba_cache_mode='align', which vLLM only selects
# when prefix caching is on. Turning it off for kvnone alone would also make
# the kvnone-vs-offload comparison at matched concurrency meaningless.
#
# It was briefly defaulted off for kvnone after two cells (c2/g19, c4/g17,
# run 30412966635) died in warmup with it on, one dump naming the Mamba
# block-zeroing path (new_block_ids_to_zero=[1615] at 327,936 computed
# tokens). That was the wrong call on the evidence: c1 in the SAME run, same
# flag, cleared warmup with 0 errors and profiled 47 minutes clean before
# dying only to `srun: error: Node failure on mia1-p01-g16`. The arm plainly
# runs with the flag on, and the cluster was throwing three different
# failures across three nodes that week. Reverted.
#
# Note vLLM resolves the flag's default to False for this model, so ON must be
# passed explicitly. PREFIX_CACHING=false forces it off for a deliberate A/B.
PREFIX_CACHE_ARGS=(--enable-prefix-caching)
if [ "${PREFIX_CACHING:-}" = "false" ]; then
    PREFIX_CACHE_ARGS=(--no-enable-prefix-caching)
fi

# The upstream DSpark config pins "attention_backend": "FLASHINFER_MLA", which
# is CUDA-only and cannot be used verbatim on gfx950; SPEC_ATTN_BACKEND
# overrides it. Golden AL on B300 is 3.78 at 7 draft tokens
# (golden_al_distribution/kimik3_dspark.yaml), so this is the largest decode-side
# lever if it can be made to run here.
SPEC_ARGS=()
if [ "${SPEC_DECODE:-false}" = "true" ]; then
    SPEC_DRAFT_MODEL="${SPEC_DRAFT_MODEL:-Inferact/Kimi-K3-DSpark}"
    SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-7}"
    SPEC_ATTN_BACKEND="${SPEC_ATTN_BACKEND:-TRITON_MLA}"
    SPEC_ARGS=(
        --speculative-config
        "{\"model\":\"$SPEC_DRAFT_MODEL\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"$SPEC_ATTN_BACKEND\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\"}"
    )
fi

# ---- Eval-only path -----------------------------------------------------------
# Mirrors the kimik2.7 agentic EVAL v2 configuration that scored on SWE-bench
# (run 30258968315). Two things are needed beyond the throughput config, and
# both are gated on EVAL_ONLY so the measured serving config is untouched.
EVAL_SERVE_ARGS=()
if [ "${EVAL_ONLY:-false}" = "true" ]; then
    # The kimi_k3 tool-call and reasoning parsers are already passed
    # unconditionally by this recipe (they are part of the AMD reference
    # command), unlike kimik2.7 where they had to be added for eval.
    #
    # With tool calls flowing, vLLM builds a grammar for them and the default
    # backend `auto` resolves to xgrammar, which rejects Kimi's tool-call tokens
    # ("Failed to advance FSM" -> HTTP 500 -> empty patches -> a near-zero
    # score). Move off xgrammar. llguidance is the guidance backend's runtime
    # and is not in the ROCm image, so install it on demand.
    SO_BACKEND="${STRUCTURED_OUTPUTS_BACKEND:-guidance}"
    if [ "$SO_BACKEND" != "auto" ]; then
        python3 -c 'import llguidance' 2>/dev/null || pip install --quiet llguidance || true
        if python3 -c 'import llguidance' 2>/dev/null; then
            # `vllm serve --help` lists config SECTIONS, not flags; the flag
            # names only appear under --help=all.
            VLLM_SERVE_HELP="$(vllm serve --help=all 2>/dev/null || vllm serve --help 2>/dev/null || true)"
            if grep -q -- '--structured-outputs-config' <<<"$VLLM_SERVE_HELP"; then
                EVAL_SERVE_ARGS+=(--structured-outputs-config "{\"backend\":\"$SO_BACKEND\"}")
            elif grep -q -- '--guided-decoding-backend' <<<"$VLLM_SERVE_HELP"; then
                EVAL_SERVE_ARGS+=(--guided-decoding-backend "$SO_BACKEND")
            else
                echo "WARN: no structured-outputs backend flag in this image; leaving the default in place" >&2
            fi
        else
            echo "WARN: llguidance unavailable; leaving the default structured-outputs backend (xgrammar) in place" >&2
        fi
    fi

    # 300 SWE-bench Lite instances at the sweep's conc would not finish inside
    # SWEBENCH_AGENT_TIMEOUT (6h). Accuracy does not depend on the conc point,
    # only wall-clock does, so widen serving concurrency for eval and let the
    # harness match it.
    EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-64}"
    export SWEBENCH_AGENT_WORKERS="${SWEBENCH_AGENT_WORKERS:-$EVAL_MAX_NUM_SEQS}"
    MAX_NUM_SEQS="$EVAL_MAX_NUM_SEQS"
fi

# 0.88, not the reference's 0.95. Measured on cluster:mi355x-amds: 0.95 asks for
# 273.59 of 287.98 GiB and cleared only 2 of 9 cells. Observed free memory at
# startup across seven nodes -- g09 281, g11 275/212/208, g14 256, g16 271/262,
# g15 21, g18 22 -- so even nominally clean nodes sit below 273.59 once driver
# overhead and transient co-tenancy are counted. This is not a denylist problem:
# g11 and g16 each measured both above and below the line hours apart. 0.88
# (253.4 GiB) clears every observation except the two genuinely occupied nodes.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"

echo "Starting vllm server..."
export PYTHONNOUSERSITE=1

## Long-context forward passes (~370K tokens with fp8 KV + DRAM offload) can exceed
## vLLM's default 300s worker RPC timeout, killing the engine with
## "RPC call to sample_tokens timed out". Widen it.
#export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1200}"
#
## Patch aiter's Gluon MLA kernel with a fixed version.
#MLA_GLUON_DST="/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/gluon/mla_gluon.py"
#MLA_GLUON_SRC="https://gist.githubusercontent.com/seungrokj/f64cb547829360bfb304f5e794d284ac/raw/mla_gluon.py"
#if [ -f "$MLA_GLUON_DST" ]; then
#    echo "Patching $MLA_GLUON_DST from gist..."
#    curl --silent --fail --location "$MLA_GLUON_SRC" -o "$MLA_GLUON_DST" \
#        && echo "Patched mla_gluon.py" \
#        || echo "WARN: failed to patch mla_gluon.py; leaving the image version in place" >&2
#else
#    echo "WARN: $MLA_GLUON_DST not found; skipping mla_gluon.py patch" >&2
#fi

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --moe-backend auto
    --tensor-parallel-size "$TP"
    "${EP_ARGS[@]}"
    --load-format auto
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    "${MM_ARGS[@]}"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    "${MAX_MODEL_LEN_ARGS[@]}"
    "${PREFIX_CACHE_ARGS[@]}"
    "${KV_CACHE_DTYPE_ARGS[@]}"
    "${SPEC_ARGS[@]}"
    "${EVAL_SERVE_ARGS[@]}"
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
