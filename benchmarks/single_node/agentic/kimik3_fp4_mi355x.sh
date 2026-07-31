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
#   KV_CACHE_DTYPE           fp8    (default for every arm; =auto for a bf16 A/B)
#   KV_BLOCK_SIZE            unset  (unset -> vLLM sizes the page; 128 under fp8)
#   MAX_MODEL_LEN            unset  (unset -> vLLM derives K3's 1M context)
#   SPEC_DECODE              false  (enabled by the _mtp DSpark wrapper)
#   ENFORCE_EAGER            false  (reference; DSpark wrapper keeps it false)

source "$(dirname "$0")/../../benchmark_lib.sh"

# Force the eval framework to lm-eval, matching minimaxm3_fp4_mi355x.sh.
# run_eval derives its default as swebench for agentic scenarios
# (scenario_default=swebench when IS_AGENTIC/SCENARIO_TYPE=agentic-coding), but
# EVAL_FRAMEWORK takes precedence over that default
# (benchmark_lib.sh:1471 framework=${EVAL_FRAMEWORK:-...}). Left overridable so
# the SWE-bench path below can still be selected with EVAL_FRAMEWORK=swebench.
export EVAL_FRAMEWORK="${EVAL_FRAMEWORK:-lm-eval}"

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
# Only the LMCache arms set this (to `--mamba-cache-mode align`); every other
# arm leaves vLLM to pick the mode.
LMCACHE_MAMBA_CACHE_MODE_ARGS=()

# fp8 KV is the DEFAULT for every arm on this model.
#
# It is the first thing that made the trace run well: run 30442578333 (c8,
# kvnone, 3600s) scored 32.59 output tok/s / 696 total tok/s/GPU at TTFT 2.6s,
# against 19.17 / 535 / 69.1s for the bf16 offload cell in the same window.
# K3 costs ~217 KiB KV/token -- 3x K2.7 -- so halving the KV element size is
# the single largest lever available, exactly as in
# [[fp8-kv-cache-mandatory]] for kimik2.7.
#
# This is set here, before the KV_OFFLOAD_BACKEND case below, for two reasons:
# the agentic matrix has no per-cell env channel (so a kvnone cell has no other
# way to ask for fp8), and the mla_gluon patch further down gates on
# KV_CACHE_DTYPE=fp8 and must see it already set.
#
# Set KV_CACHE_DTYPE=auto for a bf16 A/B.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"

if agentic_kv_offload_enabled; then
case "${KV_OFFLOAD_BACKEND:-}" in
  lmcache|lmcache-k27|lmcache-budget|lmcache-k3)
    require_agentic_kv_offload_backend "$KV_OFFLOAD_BACKEND"
    # lmcache-k3: LMCache's OWN validated K3 server command
    # (docs/source/recipes/kimi_k3.rst, PR #4277). Deliberately minimal --
    # the `reference` profile's flags (--max-gpu-workers, --l1-align-bytes,
    # --eviction-trigger-watermark, --supported-transfer-mode) are not part of
    # what upstream validated on K3, and an unrecognised flag is a silent
    # server-start failure. Start from the known-good command; add tuning
    # flags only once this arm produces a number.
    if [ "$KV_OFFLOAD_BACKEND" = "lmcache-k3" ]; then
        LMCACHE_PROFILE=k3upstream
    fi
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
    # Both values derive from vLLM's UNIFIED block size N, which is NOT a
    # constant: it depends on TP and on the KV dtype. Upstream's recipe quotes
    # N=768 for K3 at TP8, but that is the bf16 figure, and it explicitly says
    # to "re-read N from vLLM's `Setting attention block size to N tokens`
    # startup log line" after changing anything. Measured on this fleet with
    # --kv-cache-dtype fp8 at TP8, three independent runs (30453589555,
    # 30521327099, 30525206671) all log:
    #     Setting attention block size to 1536 tokens
    # fp8 halves the element size, so twice as many tokens fit one page. Since
    # fp8 is this recipe's default, hardcoding 768 would violate BOTH LMCache
    # constraints on every default run.
    #
    # Constraints (lmcache/integration/vllm, and upstream's recipe):
    #   chunk_size % N == 0                  -> chunk_size = N
    #   N <= max_num_batched_tokens < 2 * N  -> sit just under 2N
    # `align` snapshots the KDA state only on a block boundary at the end of a
    # scheduler step, so the per-step budget must cover at least one full block
    # but stay below two. Upstream validated 1500 against N=768 (just under
    # 2N); 3000 against N=1536 is the same ratio.
    if [ "${KV_CACHE_DTYPE:-}" = "fp8" ]; then
        LMCACHE_UNIFIED_BLOCK="${LMCACHE_UNIFIED_BLOCK:-1536}"
    else
        LMCACHE_UNIFIED_BLOCK="${LMCACHE_UNIFIED_BLOCK:-768}"
    fi
    LMCACHE_K3_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE_OVERRIDE:-$LMCACHE_UNIFIED_BLOCK}"
    MAX_NUM_BATCHED_TOKENS="${LMCACHE_MAX_NUM_BATCHED_TOKENS:-$(( LMCACHE_UNIFIED_BLOCK * 2 - 72 ))}"
    LMCACHE_CHUNK_SIZE="$LMCACHE_K3_CHUNK_SIZE"
    LMCACHE_CHUNK_SIZE_K27="$LMCACHE_K3_CHUNK_SIZE"
    # Guard the derivation rather than trusting it: a wrong N is a 20-minute
    # engine-init failure after a 1.56 TB weight load.
    if [ $(( LMCACHE_K3_CHUNK_SIZE % LMCACHE_UNIFIED_BLOCK )) -ne 0 ]; then
        echo "Error: LMCache chunk-size $LMCACHE_K3_CHUNK_SIZE is not a multiple of N=$LMCACHE_UNIFIED_BLOCK." >&2
        exit 1
    fi
    if [ "$MAX_NUM_BATCHED_TOKENS" -lt "$LMCACHE_UNIFIED_BLOCK" ] \
       || [ "$MAX_NUM_BATCHED_TOKENS" -ge $(( LMCACHE_UNIFIED_BLOCK * 2 )) ]; then
        echo "Error: --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS outside [N, 2N) for N=$LMCACHE_UNIFIED_BLOCK." >&2
        exit 1
    fi
    echo "LMCache: N=$LMCACHE_UNIFIED_BLOCK (kv-cache-dtype=${KV_CACHE_DTYPE:-auto})" \
         "--max-num-batched-tokens=$MAX_NUM_BATCHED_TOKENS --chunk-size=$LMCACHE_K3_CHUNK_SIZE"

    # Required by upstream's K3 recipe, and by the KDA backend generally:
    # `align` is the only Mamba cache mode the KDA backend supports, and it is
    # the mode that keeps reusable state snapshots. vLLM picks it implicitly
    # when prefix caching is on, but upstream passes it explicitly and so do
    # we -- an implicit default that silently flips to 'none' is exactly how
    # this arm failed before.
    LMCACHE_MAMBA_CACHE_MODE_ARGS=(--mamba-cache-mode align)

    # Upstream's K3 recipe requires this on the LMCache path. It defaults to 0
    # and gates the fused latent-MoE tail in the K3 model definition.
    export VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION="${VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION:-1}"

    # GPU-memory headroom for the LMCache arm specifically.
    #
    # Attempt 1 (run 30559676068) cleared every LMCache-specific hurdle -- the
    # source build, both K3 edit classes, `align`, N=1536, a healthy server --
    # and then died on the SAME allocation wall the no-offload arm hits:
    # HSA_STATUS_ERROR_OUT_OF_RESOURCES / "Available Free mem : 0 MB" on the
    # workers. But it hit it at num_computed_tokens=184,320 with kv_cache_usage
    # at 5.2%, against 360,960-373,248 at 10-21% without the connector. The
    # threshold roughly HALVED, because LMCacheMPConnector holds GPU-side
    # staging buffers on top of the transient chunked-prefill workspace.
    #
    # Two levers, both scoped to this arm so the no-offload curve is untouched:
    #
    # 1. max_num_seqs 128 -> 32. This is the structural one. The chunked MLA
    #    workspace is sized from max_num_seqs, and under fp8 it had already
    #    doubled (98,304 -> 196,608 rows) because the page went 768 -> 1536.
    #    128 slots was never meaningful at concurrency 8 -- the widest tree
    #    fan-out observed in the trace is 13 live streams in one lane, so 32
    #    leaves real headroom while cutting the workspace 4x.
    # 2. gpu_memory_utilization 0.88 -> 0.85 (244.8 GiB, ~36 GiB free vs ~28).
    #    An arm with an extra GPU-side consumer needs more slack than one
    #    without; 0.88 is calibrated for the no-offload case.
    #
    # Both are overridable, and both should be walked back toward the
    # no-offload values once this arm produces a number -- a like-for-like
    # comparison against 682.63 tok/s/GPU eventually needs matching
    # max_num_seqs.
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

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

    # LMCache K3 support is NOT in any PyPI release yet (latest is 0.5.2,
    # 2026-07-22). Upstream's own recipe -- docs/source/recipes/kimi_k3.rst,
    # added by LMCache PR #4277 -- says: "use a nightly build from 2026-07-27
    # or newer; stable LMCache releases include K3 support starting from
    # 0.5.3". Two commits on dev are what make K3 work at all:
    #   #4206 (2026-07-23) _MambaUnifiedViewEdit -- re-views a Mamba/KDA
    #         UNIFIED state tensor as a single attention tensor. This is the
    #         fix for our long-standing blocker
    #         "ValueError: expected a Mamba [conv_state, ssm_state] tensor
    #          list, got Tensor" (run 30350911388): K3's Kimi Delta Attention
    #         supplies one fused state tensor, and before #4206 the adapter
    #         only understood the [conv_state, ssm_state] pair.
    #   #4278 (2026-07-28) _SubpagedMLAAttentionViewEdit -- re-views K3's
    #         kernel-paged MLA tensor [N*12, 64, 576] as logical pages
    #         [N, 768, 576].
    # 0.5.1 predates BOTH, which is why every LMCache attempt on this model
    # has died at engine init. The historical "pin 0.5.1, never build dev"
    # note in this file was correct for its date and is now obsolete.
    #
    # Nightly WHEELS are CUDA-only, so ROCm must build from source. Flags are
    # upstream's (docs/source/getting_started/installation.rst): gfx950 is
    # MI355X/MI350X.
    #
    # LMCACHE_GIT_REF pins the build. Default is the #4278 merge commit rather
    # than dev HEAD so the build is reproducible and cannot drift under us.
    # A config-supplied `version` that looks like a release (0.5.3+) still
    # installs from PyPI once such a release exists.
    LMCACHE_GIT_REF="${LMCACHE_GIT_REF:-1dfa07772b8b2ae79653c830d63e297da39c970f}"
    LMCACHE_VERSION="${LMCACHE_VERSION:-${LMCACHE_CFG_VERSION:-git}}"
    if ! python3 -c "import lmcache.integration.vllm.lmcache_mp_connector" >/dev/null 2>&1; then
        if [ "$LMCACHE_VERSION" = "git" ]; then
            # Clone to a container-local dir, NOT the bind-mounted /workspace:
            # a later job's `clean: true` checkout trips over root-owned build
            # artifacts left behind there.
            LMCACHE_SRC="${LMCACHE_SRC:-/opt/lmcache-src}"
            echo "Building lmcache from source at ref $LMCACHE_GIT_REF (ROCm/gfx950)"
            rm -rf "$LMCACHE_SRC"
            git clone --filter=blob:none https://github.com/LMCache/LMCache.git "$LMCACHE_SRC"
            git -C "$LMCACHE_SRC" checkout --detach "$LMCACHE_GIT_REF"
            git -C "$LMCACHE_SRC" --no-pager log -1 --oneline
            (
                cd "$LMCACHE_SRC"
                export PYTORCH_ROCM_ARCH="gfx942,gfx950"
                export TORCH_DONT_CHECK_COMPILER_ABI=1
                export CXX=hipcc
                export BUILD_WITH_HIP=1
                if command -v uv >/dev/null 2>&1; then
                    uv pip install --system -e . --no-build-isolation \
                        || python3 -m pip install -e . --no-build-isolation
                else
                    python3 -m pip install -e . --no-build-isolation
                fi
            )
        else
            echo "Installing lmcache==$LMCACHE_VERSION"
            if command -v uv >/dev/null 2>&1; then
                uv pip install --system "lmcache==$LMCACHE_VERSION" \
                    || agentic_pip_install --quiet "lmcache==$LMCACHE_VERSION"
            else
                agentic_pip_install --quiet "lmcache==$LMCACHE_VERSION"
            fi
        fi
        # Fail here, loudly, rather than 20 minutes later inside vLLM: this
        # runs before the engine loads 1.56 TB of weights.
        python3 -c "import lmcache.integration.vllm.lmcache_mp_connector" >/dev/null
    fi
    python3 -c "import lmcache; print('lmcache', getattr(lmcache,'__version__','?'))" || true
    # Assert the two K3 fixes are actually present in whatever got installed.
    # A silently-old build would otherwise fail much later with the original
    # Mamba ValueError and look like a fresh bug.
    python3 - <<'PYEOF'
import sys
try:
    from lmcache.integration.vllm import kv_cache_group_edits as e
except Exception as exc:
    sys.exit(f"FATAL: cannot import kv_cache_group_edits: {exc}")
names = {type(x).__name__ for x in getattr(e, "_EDITS", ())}
missing = {"_MambaUnifiedViewEdit", "_SubpagedMLAAttentionViewEdit"} - names
if missing:
    sys.exit(
        f"FATAL: installed lmcache lacks K3 support: missing {sorted(missing)}. "
        f"Registered edits: {sorted(names)}. Needs dev >= #4278 (2026-07-28)."
    )
print(f"lmcache K3 edits present: {sorted(names)}")
PYEOF

    LMCACHE_HOST="${LMCACHE_HOST:-127.0.0.1}"
    LMCACHE_PORT="${LMCACHE_PORT:-5555}"
    LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"

    # L1 is SHM-backed: if it exceeds free /dev/shm, LMCache silently disables
    # SHM and falls back to a pickle path that crashes at load. Cap at 90% of
    # free /dev/shm so SHM stays enabled, and say so loudly -- the capped value
    # is the number that actually backs the run.
    if [ "${LMCACHE_PROFILE:-}" = "k3upstream" ]; then
        # Upstream's recipe says --l1-size-gb 100, but that is a
        # does-it-work value, not a capacity value. Attempt 2 (run
        # 30563325963) ran the whole 3600 s green at 100 GB and the host tier
        # did NOTHING: 122,770 chunks stored, 1,879 evictions, and an external
        # prefix cache hit rate of 0.0%. A write-only cache.
        #
        # The arithmetic says it could not have worked. K3 costs ~217 KiB of
        # KV per token at bf16, ~108 KiB at fp8, so 100 GB holds well under
        # 1M tokens -- SMALLER than the 3,322,266-token GPU pool it is meant
        # to back. An offload tier below the size of the GPU pool can only
        # evict before reuse. Worse, it dragged the GPU tier down with it:
        # GPU prefix hit fell 88.4% -> 33.6% and GPU KV usage 87.5% -> 52.7%,
        # consistent with vLLM releasing blocks to an external tier that then
        # threw them away.
        #
        # Attempt 3 (run 30575475359) then sized L1 from TOTAL_CPU_DRAM_GB,
        # which the /dev/shm cap landed at 1360 GB, and that BROKE GPU IPC:
        #   Error handling request RequestType.REGISTER_KV_CACHE
        #     torch.UntypedStorage._new_shared_cuda(...)
        #     torch.AcceleratorError: CUDA error: invalid device pointer
        #                             (hipErrorInvalidDevicePointer)
        # once per TP rank, so all 8 vLLM workers then died on
        # "LMCache server did not respond to register_kv_caches within 300.0s".
        # The timeout was the symptom; the registration itself failed.
        #
        # Attempt 2 at 100 GB logged ZERO IPC errors and registered cleanly.
        # L1 size was the only difference between the two runs, so pinning
        # ~1.36 TB of host memory is what exhausts the driver's mapping
        # resources and makes the subsequent GPU IPC import invalid. Note the
        # lazy allocator was NOT the problem here: 134 expansions completed in
        # 3m37s and the pool was fully allocated long before vLLM finished
        # loading weights.
        #
        # So L1 has a working range: above the GPU tier (GPU KV is ~52 GiB/rank
        # at GMU 0.85, ~416 GB across TP8) and below the IPC ceiling somewhere
        # under 1360 GB. 512 GB registered cleanly (run 30579846678) and lifted
        # external hits 0.00% -> 0.82%, but throughput did not move.
        #
        # LMCACHE_L1_SHM_FRACTION sizes L1 as a fraction of FREE /dev/shm,
        # which is the resource that actually binds (shm was 1512 GB free on
        # this fleet, so 0.60 -> ~907 GB). Sizing off shm rather than
        # TOTAL_CPU_DRAM_GB is deliberate: the DRAM budget is 2399 GB, so any
        # fraction of it lands past the shm cap and straight back on the 1360
        # GB value that broke IPC.
        #
        # 0.60 bisects the known range -- 1.8x the 512 GB that works, 0.67x
        # the 1360 GB that fails -- so this run locates the IPC ceiling as
        # well as testing whether more cache changes the throughput picture.
        # If it fails to register, the ceiling is between 512 and 907 and the
        # next step is 0.45 (~680 GB), not a return to 512.
        LMCACHE_L1_SHM_FRACTION="${LMCACHE_L1_SHM_FRACTION:-0.60}"
        _shm_free_gb=$(df -BG --output=avail /dev/shm 2>/dev/null | tail -1 | tr -dc '0-9')
        if [ -n "$_shm_free_gb" ] && [ "${_shm_free_gb:-0}" -gt 0 ]; then
            _l1_from_shm=$(awk -v f="$LMCACHE_L1_SHM_FRACTION" -v s="$_shm_free_gb" \
                'BEGIN { printf "%d", f * s }')
            LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-$_l1_from_shm}"
            echo "LMCache L1 sized from /dev/shm: ${LMCACHE_L1_SHM_FRACTION} x ${_shm_free_gb}G free = ${LMCACHE_L1_SIZE_GB}G"
        else
            # df failed -- fall back to the value already proven to register.
            LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-512}"
            echo "WARNING: could not read free /dev/shm; falling back to L1=${LMCACHE_L1_SIZE_GB}G" >&2
        fi
    else
        LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-512}"   # reference value
    fi

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
      k3upstream)
        # Upstream's validated K3 command, verbatim apart from --host and the
        # --http-* pair, which this recipe's healthcheck needs and which every
        # other profile already passes:
        #   lmcache server --port 6555 --chunk-size 768 --max-workers 4 \
        #                  --l1-size-gb 100 --eviction-policy LRU
        # --chunk-size comes from the N derivation above (1536 under fp8, not
        # upstream's bf16 768). L1 defaults to upstream's 100 GB here rather
        # than the reference 512 GB: memory pinning is unavailable in the ROCm
        # container, so LMCache allocates the WHOLE L1 pool before serving,
        # and a smaller pool means a much shorter and more predictable
        # bring-up. Raise it via LMCACHE_L1_SIZE_GB once the arm is green.
        LMCACHE_CMD=(
            lmcache server
            --host "$LMCACHE_HOST"
            --port "$LMCACHE_PORT"
            --http-host "$LMCACHE_HOST"
            --http-port "$LMCACHE_HTTP_PORT"
            --chunk-size "$LMCACHE_CHUNK_SIZE"
            --max-workers "${LMCACHE_MAX_WORKERS:-4}"
            --l1-size-gb "$LMCACHE_L1_SIZE_GB"
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
        echo "Error: unsupported LMCACHE_PROFILE '$LMCACHE_PROFILE' (expected: reference, k2.7, k3upstream)" >&2
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
  vllm-simple|vllm-simple-fp8|vllm-simple-fp8-lazy)
    require_agentic_kv_offload_backend "$KV_OFFLOAD_BACKEND"
    # vllm-simple-fp8-lazy is vllm-simple-fp8 with lazy offload. In eager mode
    # the connector stores every completed block to CPU immediately; in lazy
    # mode it only stores under GPU-block pressure (manager.py _lazy_mode /
    # _estimate_lazy_target_blocks walk the GPU free queue). Every fp8+connector
    # cell so far has died while the GPU pool was nearly EMPTY -- 11.6% usage
    # with a 0.0% external hit rate in run 30505656455, i.e. mid store-storm
    # with nothing yet to gain from the host tier -- so the store path is where
    # to look, and lazy removes almost all of it in that regime.
    #
    # Also the mode Wenyao's kimik3_fp4_mi355x_mtp.sh pins (SIMPLE_LAZY_OFFLOAD
    # =true) as part of the gfx950 bundle validated in PR #2367 against an
    # eight-rank GPU memory access fault.
    if [ "$KV_OFFLOAD_BACKEND" = "vllm-simple-fp8-lazy" ]; then
        SIMPLE_LAZY_OFFLOAD="${SIMPLE_LAZY_OFFLOAD:-true}"
    fi
    # vllm-simple-fp8 is the same connector with an fp8 KV cache. fp8 halves
    # bytes/token in the GPU pool, which on this KV-bound corpus moves the
    # eviction wall itself rather than just adding headroom (measured: the pool
    # peaks at 98.6-99.8% usage even at c8). ROCm maps fp8 -> fp8_e4m3.
    # UNVALIDATED on K3's hybrid geometry: the pool spans Kimi Delta Attention
    # state and gated-MLA latent, and fp8 support across both spec types is
    # unconfirmed on this build.
    case "$KV_OFFLOAD_BACKEND" in
      vllm-simple-fp8|vllm-simple-fp8-lazy)
        KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
        ;;
    esac
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

# INERT ON K3 -- kept only so the override stays discoverable. K3 is hybrid, and
# vLLM raises the attention block size until the attention page is at least the
# mamba/KDA page (interface.py:911, "Setting attention block size to N tokens to
# ensure that attention page size is >= mamba page size"). Measured: bf16 -> 768
# tokens, fp8 -> 1536. --block-size 64 was silently overridden to 1536 in run
# 30505656455, so this knob cannot move the page on this model; only
# kv_cache_dtype can.
#
# That also explains the connector's identical 14122 CPU blocks across dtypes:
# fp8 halves bytes/token but the page doubles in tokens, so page BYTES are
# unchanged and manager.py:199 derives the same pool.
KV_BLOCK_SIZE_ARGS=()
if [ -n "${KV_BLOCK_SIZE:-}" ] && [ "${KV_BLOCK_SIZE}" != "0" ]; then
    KV_BLOCK_SIZE_ARGS=(--block-size "$KV_BLOCK_SIZE")
fi

# Left unset by default so vLLM derives K3's native 1M context, which is what
# the unfiltered corpus needs. Set explicitly only to test truncation effects.
MAX_MODEL_LEN_ARGS=()
if [ -n "${MAX_MODEL_LEN:-}" ] && [ "${MAX_MODEL_LEN}" != "0" ]; then
    MAX_MODEL_LEN_ARGS=(--max-model-len "$MAX_MODEL_LEN")
fi

EAGER_ARGS=()
if [ "${ENFORCE_EAGER:-false}" = "true" ]; then
    EAGER_ARGS=(--enforce-eager)
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
# ON by default for non-DSpark arms. This trace is built around large shared
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
case "${PREFIX_CACHING:-true}" in
    true)
        PREFIX_CACHE_ARGS=(--enable-prefix-caching)
        ;;
    false)
        PREFIX_CACHE_ARGS=(--no-enable-prefix-caching)
        ;;
    auto)
        # Match the upstream AMD command by letting vLLM resolve its default.
        PREFIX_CACHE_ARGS=()
        ;;
    *)
        echo "Error: PREFIX_CACHING must be true, false, or auto." >&2
        exit 1
        ;;
esac

# The upstream DSpark config pins "attention_backend": "FLASHINFER_MLA", which
# is CUDA-only and cannot be used verbatim on gfx950; SPEC_ATTN_BACKEND
# overrides it. Use real block rejection for both throughput and evaluation so
# this path is the exact AMD reproducer and exercises target-model verification.
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
# 2026-07-30: 0.90 (259.19 GiB) was tried and REVERTED. It does not fail at
# engine init -- it comes up clean with a 4,302,357-token pool -- and then dies
# mid-prefill. Run 30521327099 lost c4, c8 and c16 on three separate nodes with
# an identical signature: HSA_STATUS_ERROR_OUT_OF_RESOURCES / "Available Free
# mem : 0 MB" on 4 of 8 ranks, surfacing as hipErrorUnknown at
# gpu_model_runner.py:314. Every death was at num_computed_tokens 360,960 to
# 364,032 with num_running_reqs=1 and kv_cache_usage only ~21%, so it is not
# concurrency and not KV exhaustion: the transient chunked-prefill workspace
# scales with context and needs more than the ~22 GiB of unreserved headroom
# 0.90 leaves. 0.88 leaves ~28 GiB and served 987 requests at ISL p90 374,550 /
# p95 511,146 with a 0/165 error rate (run 30453589555, byte-identical serve
# command apart from this one flag).
#
# The extra 443K tokens of pool 0.90 buys are unusable anyway -- peak usage at
# death was 21%. Raising this again requires bounding the prefill workspace
# first (the build_mla_chunked_context_metadata aggregate-chunk cap that only
# exists in hyukjleeamd/kimi-k3-mla-b128:fp8-kv-fixed, not in the mla_gluon
# gist patched below, which is a decode-path fix only).
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"

echo "Starting vllm server..."
export PYTHONNOUSERSITE=1

# Long-context forward passes (~370K tokens with fp8 KV + DRAM offload) can exceed
# vLLM's default 300s worker RPC timeout, killing the engine with
# "RPC call to sample_tokens timed out". Widen it.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1200}"

# Patch aiter's Gluon MLA kernel with a fixed version. The b128 kernel is only
# exercised on the fp8 KV path, so only patch when KV_CACHE_DTYPE is fp8.
if [ "${KV_CACHE_DTYPE:-}" = "fp8" ]; then
    MLA_GLUON_DST="/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/gluon/mla_gluon.py"
    # Pinned to the exact gist revision that produced the first clean fp8 KV
    # run (30442578333: c8, 3600s, 696 total tok/s/GPU, TTFT 2.6s). The
    # unpinned .../raw/mla_gluon.py URL silently follows the gist HEAD, so a
    # later edit would change the kernel under us with no diff in this repo.
    MLA_GLUON_SRC="https://gist.githubusercontent.com/seungrokj/f64cb547829360bfb304f5e794d284ac/raw/4b0088c5fecbeffa6544d2da1006b45380aac896/mla_gluon.py"
    if [ -f "$MLA_GLUON_DST" ]; then
        echo "Patching $MLA_GLUON_DST from gist..."
        curl --silent --fail --location "$MLA_GLUON_SRC" -o "$MLA_GLUON_DST" \
            && echo "Patched mla_gluon.py" \
            || echo "WARN: failed to patch mla_gluon.py; leaving the image version in place" >&2
    else
        echo "WARN: $MLA_GLUON_DST not found; skipping mla_gluon.py patch" >&2
    fi
fi

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
    "${LMCACHE_MAMBA_CACHE_MODE_ARGS[@]}"
    "${KV_CACHE_DTYPE_ARGS[@]}"
    "${KV_BLOCK_SIZE_ARGS[@]}"
    "${EAGER_ARGS[@]}"
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
