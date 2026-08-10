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
#   MAX_NUM_BATCHED_TOKENS   4096   (reference)
#   AITER_A8W4               1      (reference; 0 = aiter a16w4 MoE path)
#   LANGUAGE_MODEL_ONLY      false  (reference loads the vision tower)
#   KV_CACHE_DTYPE           fp8    (default for every arm; =auto for a bf16 A/B)
#   KV_BLOCK_SIZE            unset  (unset -> vLLM sizes the page; 128 under fp8)
#   MAX_MODEL_LEN            unset  (unset -> vLLM derives K3's 1M context)
#   SPEC_DECODE              true   (this is the _mtp DSpark recipe; =false for a no-spec A/B)
#   SPEC_NUM_TOKENS          2      (DSpark draft length; validated by the _mtp config)

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
#
# BOTH names must be set. vllm-project/vllm#50582 ("[ROCm][Kimi-K3] aiter moe
# environment variable cleanup") renamed the vLLM-side flag
# AITER_SITUV2_A8W4 -> VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4 with no
# back-compat alias, while the AITER runtime still reads the old name
# directly. Setting only the old name splits the two halves of the path:
# vLLM's mxfp4.py shuffles w13 in the SEPARATED layout and passes
# GateMode.SEPARATED (guinterleave=False), but AITER still dispatches the
# gate/up-interleaved fp8 kernels (flydsl_moe1_afp8_wfp4_bf16_..._gui_...).
# Gate and up are then read from the wrong halves of w13 -- no crash, pure
# numerical garbage. That is what took gsm8k to 0.0 on
# nightly-cb8104839c141609d99f1254459ef3a4f1bd4263 while
# nightly-124154a8843d1f8e4d4e2d5d466e2d3ebc3716da (pre-#50582) scored 1.0.
# Setting only the new name would split it the other way. Keep them equal.
export AITER_SITUV2_A8W4="${AITER_A8W4:-1}"
export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4="$AITER_SITUV2_A8W4"
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
# Likewise --max-num-batched-tokens: empty everywhere except the LMCache arms.
#
# 60da246d2 dropped the flag so vLLM picks its own budget (8192 on the GPU-only
# c8 run 30928884574, which is 5.6x the value the LMCache path derives). That is
# right for GPU-only and wrong for LMCache: `align` requires
# N <= max_num_batched_tokens < 2N, and vLLM's 8192 is far outside [768, 1536)
# for the bf16 N=768. Leaving the flag off on an LMCache arm silently violates
# the constraint the block below spends thirty lines deriving and guarding.
#
# So the flag comes back, but only where the constraint applies -- the GPU-only
# arm keeps vLLM's default and stays byte-identical to run 30928884574.
MNBT_ARGS=()

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
# fp8 by default on this branch. It has to be decided HERE, not at the serve
# command: the LMCache block below derives vLLM's unified block N from it
# (768 bf16 / 1536 fp8) and `align` then pins max_num_batched_tokens into
# [N, 2N). Setting fp8 only at the serve line would leave LMCache on N=768 and
# mnbt 1464 while the engine actually ran fp8 -- the 5.6x-too-small prefill
# chunk that pinned every earlier LMCache arm. At fp8 the same derivation gives
# mnbt 3000.
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
    # Chunk size is NOT the attention block. K3 is hybrid (KDA + MLA) and the
    # KDA cache group carries its own, larger tokens_per_block; LMCache requires
    # chunk_size to be a multiple of THAT. Run 30987890713 died at engine init:
    #   LMCache chunk size 1536 must be a multiple of engine group 14
    #   tokens_per_block 3072
    # with attention block 1536 and mnbt 3000, i.e. the mnbt derivation was
    # right and only the chunk was wrong. Under bf16 the two happened to agree
    # so chunk = N passed, which is why this never surfaced before fp8.
    #
    # 2N satisfies every constraint at once: LMCache's chunk % 3072 == 0, the
    # launcher's own chunk % N == 0, and N <= mnbt < 2N is untouched because
    # mnbt is still derived from N.
    if [ "${KV_CACHE_DTYPE:-}" = "fp8" ]; then
        LMCACHE_K3_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE_OVERRIDE:-$(( LMCACHE_UNIFIED_BLOCK * 2 ))}"
    else
        LMCACHE_K3_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE_OVERRIDE:-$LMCACHE_UNIFIED_BLOCK}"
    fi
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
    # Actually pass it. The value above is derived and guarded but, since
    # 60da246d2, was never reaching vllm serve.
    MNBT_ARGS=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")

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
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"

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
        # Upstream's validated K3 command, plus the worker and eviction tuning
        # needed by this TP8 agentic workload. The pinned LMCache revision
        # supports these flags, and four workers serialize pairs of TP ranks.
        # Upstream's original command was:
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
            --max-workers "${LMCACHE_MAX_WORKERS:-$TP}"
            --l1-size-gb "$LMCACHE_L1_SIZE_GB"
            --eviction-trigger-watermark "$LMCACHE_EVICTION_WATERMARK"
            --eviction-ratio "$LMCACHE_EVICTION_RATIO"
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

# ---- LLM server  ------------------------------------------------------------
# Apply the in-container filesystem patches this MTP recipe depends on (vLLM
# PR #50619 K3 fp8 MLA verify, aiter mla_gluon batch relax + PR #4474 int64 KV
# stride, Triton 3.7.0). The script is idempotent, so re-runs are safe.
bash "$(dirname "$0")/apply_k3_container_patches.sh"

# ---- Parallelism ------------------------------------------------------------
# TP8 or TEP8. No DP-attention arm: upstream strategy_min_gpus.multi_node_dep is
# 16, so DEP is not a single-node strategy for this model.
EP_ARGS=()
if [ "$EP_SIZE" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

# ---- Multimodal vs text-only ------------------------------------------------
LANGUAGE_MODEL_ONLY="true"
MM_ARGS=(--mm-encoder-tp-mode data)
if [ "$LANGUAGE_MODEL_ONLY" = "true" ]; then
    MM_ARGS=(--language-model-only)
fi

# ---- Optional axes ----------------------------------------------------------
# fp8 KV on the TARGET only. The drafter's own kv_cache_dtype stays "auto" in
# SPEC_ARGS below, deliberately -- see the note there.
#
# This is only reachable because verify runs on the asm PS route. The kimi_k3
# wrapper asserts `impl.supports_quant_query_input` whenever the KV cache is
# quantized (mla.py:615). ROCM_AITER_MLA inherits True, so the assert passes;
# it is Gluon that cannot take an fp8 query, which is the whole reason #51088
# exists. With verify on Gluon this arm would abort at startup.
#
# The kernel exists: hsa/gfx950/mla/mla_asm.csv registers
#   mla_a8w8_qh16_qseqlen4_gqaratio16_v3_ps.co   (fp8 q, fp8 kv, Gqa 16, qlen 4)
# which is exactly 12 heads padded to 16 at k=3 verify.
KV_CACHE_DTYPE_ARGS=(--kv-cache-dtype "$KV_CACHE_DTYPE")

# k=3, not 2, and the kernel registry is why.
#
# hsa/gfx950/mla/mla_asm.csv in aiter main has exactly one bf16 PS entry at
# Gqa=16 -- qSeqLen=4, mask1 (causal):
#   mla_a16w16_qh16_m16x4_n16x1_coex0_mask1_ps.co
# There is NO qSeqLen=3 kernel in either dtype; that is what the still-open
# aiter#4521 ("MLA PS mode fp8 -n 16,3 16,4") adds. DSpark verify runs at
# qlen = k+1, so k=2 asks for a kernel that does not exist and k=3 asks for one
# that does. It also sits exactly on asm_mla.cu:903's qo_len <= 4 boundary.
#
# k=3 is a fine place to be independently: the k=7 per-position acceptance
# measurements (0.306 / 0.191 / 0.096 / 0.034 / 0.019 / 0.010 / 0.004) say the
# first three positions carry 90% of the total gain.
SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-3}"
# The DRAFTER stays bf16 on purpose. It runs TRITON_MLA, which sets
# supports_quant_query_input=False, so giving it fp8 KV would trip the same
# kimi_k3 assert the target now clears -- the drafter has no asm route. vLLM
# keeps a separate kv_cache_dtype for the draft model precisely so the two can
# differ.
# Rejection sampling method. Default is now `synthetic` per the incoming recipe
# update.
#
# ###################################################################
# # WARNING: `synthetic` DOES NOT VERIFY DRAFT TOKENS.              #
# ###################################################################
#
# Upstream (vllm/config/speculative.py): "'synthetic' accepts draft tokens with
# a decaying probability calibrated to synthetic_acceptance_rate." No comparison
# against the target is performed, so:
#   - Emitted tokens are UNVERIFIED DRAFTS. Generated text is not the target
#     model's output, and gsm8k / any accuracy gate will fail.
#   - The reported "Mean acceptance length" echoes SYNTHETIC_ACCEPTANCE_LENGTH
#     rather than measuring anything. B300 run 30425131777 set 2.51 and reported
#     2.49/2.52, with ITL p50 0.07 ms -- physically impossible for a 2.8T model.
#   - Throughput from this arm is a SIMULATION of "what if acceptance were X",
#     not a measurement. Do not compare it against measured runs.
#
# For reference, the real measured acceptance on this trace at k=3:
#   1.77  probabilistic + block, temperature 1.0  (run 31065394450)
#   2.35  probabilistic + block, temperature 0    (run 31087323382)
#   1.59  greedy + block, temperature 1.0         (run 31087303391)
# so 2.45 sits ~38% above the current default-methodology result.
#
# Set SPEC_REJECTION_METHOD=block to restore real verification.
SPEC_REJECTION_METHOD="${SPEC_REJECTION_METHOD:-synthetic}"
SPEC_SYNTHETIC_ACCEPTANCE_LENGTH="${SPEC_SYNTHETIC_ACCEPTANCE_LENGTH:-2.51}"

SPEC_CFG="{\"model\":\"Inferact/Kimi-K3-DSpark\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"$SPEC_REJECTION_METHOD\""
if [ "$SPEC_REJECTION_METHOD" = "synthetic" ]; then
    SPEC_CFG="$SPEC_CFG,\"synthetic_acceptance_length\":$SPEC_SYNTHETIC_ACCEPTANCE_LENGTH"
    # Loud banner + an on-disk marker so a result directory is self-identifying
    # and nobody mines these numbers as measured perf months from now.
    echo "############################################################"
    echo "# SIMULATED ACCEPTANCE: rejection_sample_method=synthetic"
    echo "# synthetic_acceptance_length=$SPEC_SYNTHETIC_ACCEPTANCE_LENGTH"
    echo "# Draft tokens are NOT verified against the target model."
    echo "# Output text is unverified drafts; accuracy gates WILL fail."
    echo "# Throughput and acceptance from this run are SIMULATED."
    echo "############################################################"
    {
        echo "rejection_sample_method=synthetic"
        echo "synthetic_acceptance_length=$SPEC_SYNTHETIC_ACCEPTANCE_LENGTH"
        echo "NOT A MEASURED RESULT: draft tokens are accepted on a calibrated"
        echo "probability, never verified against the target. Acceptance echoes"
        echo "the configured value; throughput is a simulation."
    } > "$RESULT_DIR/SIMULATED_ACCEPTANCE.txt"
fi
SPEC_CFG="$SPEC_CFG}"

SPEC_ARGS=(
    --speculative-config
    "$SPEC_CFG"
)

# mns and the cudagraph capture ceiling are 2*CONC, capped at MNS_CAP.
#
# The cap exists because KimiGDNLinearAttention._forward faults with
# hipErrorIllegalAddress above mns 16. At CONC=16 (mns 32) the non-spec KDA
# decode dies in warmup_kernels/_run_decode_step -- reproduced twice, on two
# different nodes, on both the pre- and post-a8w4-fix builds:
#   run 30987878388 c16 (node g14) -> kimi_gdn_linear_attn.py:607
#                                     fused_recurrent_kda_packed_decode
#   run 31071548896 c16 (node g09) -> kimi_gdn_linear_attn.py:597
#                                     causal_conv1d_update
# Both dereference decode_conv_indices into the state cache; the slot load at
# fused_recurrent.py:479 is unmasked, so a bad index reads out of bounds.
#
# 16 is the largest value proven safe end to end: c8 (mns 16) clears warmup,
# capture and 3600 s of serving on the same nodes (runs 30982145794,
# 31065394450). Not a node issue -- g12 ran c4 (failed) and c8 (passed), g14 ran
# c16 (failed) and c1 (passed).
#
# This is a WORKAROUND. The real fix is bounds-checking the KDA state index
# upstream. It also does NOT help c4, which fails in a different phase
# (capture_model, kimi_gdn_linear_attn.py:478, the spec branch) and is not
# size-driven -- c8 captures a strict superset of c4's ladder and passes.
#
# Both values must move together: capture size may not exceed mns, or the
# capture ladder escapes mns*(k+1).
#
# NOTE: at CONC=16 this gives 1x scheduler headroom instead of 2x, so c16 is no
# longer "the c8 config at a different concurrency". Say so when comparing.
# Respect a value already set upstream (the LMCache block at ~391 sets 32, and
# MAX_NUM_SEQS is a documented knob); default to 2*CONC. Before this, both this
# line and GPU_MEM_UTIL below assigned unconditionally, so the LMCache arm's
# mns 32 / GMU 0.8 and the documented GPU_MEM_UTIL override were dead code --
# every LMCache run so far actually ran at GMU 0.9.
# PIN, not a cap. c4 (mns 8) dies capturing graph size 8 -- the exact graph c8
# (mns 16) captures successfully. Identical shape, different outcome, and mns is
# the only difference. So this is not a shape bug: it is a latent out-of-bounds
# read whose fault depends on allocation layout, since mns sizes the
# preallocated state/index buffers and therefore what sits adjacent to the bad
# address. That is also the only story consistent with the non-monotonic
# pass/fail (mns 2 pass, 8 fail, 16 pass, 32 fail).
#
#   mns 2  (c1)  capture path never exercised -- ladder [1,2] has no entry
#                divisible by k+1=4, so 0 FULL graphs. Pass is uninformative.
#   mns 8  (c4)  captures {4,8}; dies on size 8. 3/3 runs, nodes g10, g10, g12.
#   mns 16 (c8)  captures {4,8,16}; all pass. 2/2 runs, nodes g12, and one more.
#   mns 32 (c16) faults earlier, in warmup_kernels/_run_decode_step. 2/2 runs.
#
# 16 is the only value with a clean record, so pin every cell that exercises the
# capture path to it. c1 stays at 2 deliberately: raising it to 16 would start
# capturing spec graphs there and could expose the same fault on a cell that
# currently works.
MNS_PIN="${MNS_PIN:-16}"
MNS_BASE="${MAX_NUM_SEQS:-$(( 2 * CONC ))}"
# Pin unconditionally. The old `>= 4` guard left c1 (2*CONC = 2) on mns 2, the
# only cell in any sweep that runs a mns-2 drafter capture, and that is where
# run 31355411054's c1 cell died: aiter moe_sorting took an illegal address
# while capturing the DSpark speculator, after the target model had already
# captured PIECEWISE 4/4 and FULL 2/2 clean. A tiny-M expert sort is a shape no
# other cell produces.
#
# Pinning also makes c1 a real point on the curve. Under the guard it ran a
# different capture ladder (ceiling mns*(k+1) = 8 vs 64) and a different
# scheduler width from every other cell, so it was never comparable anyway.
MAX_NUM_SEQS="$MNS_PIN"
# Capture ceiling is mns * (k+1), NOT mns. A spec-decode graph is only usable at
# sizes divisible by k+1, and the decode step is running*(k+1) tokens wide, so a
# ceiling of mns only covers mns/(k+1) concurrent requests -- 4 of 16 at k=3.
# B300 sets 96 = 32*3 for exactly this reason; we were 4x under.
MAX_CUDAGRAPH_CAPTURE_SIZE=$(( MAX_NUM_SEQS * (SPEC_NUM_TOKENS + 1) ))
if [ "$MAX_NUM_SEQS" -ne "$MNS_BASE" ]; then
    echo "MNS_PIN=$MNS_PIN: max_num_seqs $MNS_BASE -> $MAX_NUM_SEQS (KDA illegal-address workaround)"
fi
COMPILATION_CONFIG_ARGS=(--compilation-config "{\"max_cudagraph_capture_size\":$MAX_CUDAGRAPH_CAPTURE_SIZE,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"+fused_rms_norm_gated\"]}")
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"

# vllm-project/vllm#51088 -- "[ROCm][MLA] Add small-head PS ASM decode route".
#
# K3 at TP8 is 12 heads/rank, under AITER MLA's 16-head minimum, so decode and
# DSpark verify both fall to the Gluon Triton kernel. #51088 adds
# VLLM_ROCM_MLA_FORCE_PS=1, which routes small-head decode to the persistent
# -scheduling ASM kernel instead, and fixes the head-padding helpers.
#
# Those helpers are the reason this is worth running. At 12 heads both are
# silent no-ops -- `16 // 12 == 1`, so get_mla_padded_q's repeat_interleave(1)
# pads nothing and get_mla_unpadded_o's o[:, ::1, :] returns 16 heads where 12
# are expected. That is the likeliest cause of the earlier "padded asm verify
# does not verify" result (574/574 accepted, per-position all 1.000), which we
# had recorded as asm verify being unusable for DSpark. #51088 replaces both
# with zero-pad and a contiguous slice.
#
# Unlike #50371 and #50578, nothing here gates on max_qo_len, and aiter's asm
# guard accepts qo_len <= 4 -- so at k=2 (qo_len 3) DSpark verify can take this
# path. That is the whole experiment.
#
# The PR does not compile as posted: it replaces the `return (...)` in
# get_mla_padded_q but leaves the outer closing paren behind as context, giving
# `SyntaxError: unmatched ')'`. The vendored diff deletes that line; it is
# otherwise byte-for-byte the PR. Verified to apply to this image's vLLM
# (nightly-cb8104839c... is byte-identical to the PR's base 12292d94b258 for
# this file).
#
# READ ACCEPTANCE SHAPE, NOT JUST THE MEAN: a corrupt asm verify previously
# presented as 100% acceptance, which looks like a win.
# Default ON for this branch -- it exists to test the PS ASM route. Every other
# branch leaves it at 0.
MLA_FORCE_PS="${MLA_FORCE_PS:-1}"
if [ "$MLA_FORCE_PS" = "1" ]; then
    # Absolute: the patch runs inside `cd "$VLLM_ROOT"`, and the `<` redirect is
    # resolved after that cd, so a relative path here resolves against
    # site-packages and fails (run 30970159866).
    # Resolve BOTH diffs to absolute paths here, before any subshell. Doing the
    # `$(cd ...)` inline in the redirect fails: the redirect is expanded after
    # the subshell's `cd "$VLLM_ROOT"`, so a relative $0 resolves against
    # site-packages. Cost one cell each way (30970159866, 30977723941).
    PATCH_DIR="$(cd "$(dirname "$0")/patches" && pwd)"
    PS_DIFF="$PATCH_DIR/vllm_pr51088_force_ps.diff"

    VLLM_ROOT="$(python3 -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
    for f in "$PS_DIFF" "$PATCH_DIR/vllm_dspark_ps_enable.py"; do
        [ -f "$f" ] || { echo "ERROR: $f missing" >&2; exit 1; }
    done
    # No `|| true`: an unpatched run would silently be a Gluon control arm
    # published under this experiment's name.
    ( cd "$VLLM_ROOT" && patch -p1 --forward --batch < "$PS_DIFF" ) \
        2>&1 | tee "$RESULT_DIR/vllm_pr51088.log"
    python3 -c "import ast,sys;ast.parse(open(sys.argv[1]).read())" \
        "$VLLM_ROOT/vllm/v1/attention/backends/mla/rocm_aiter_mla.py"
    # LOCAL EXTENSION, not part of #51088.
    #
    # #51088 adds `and not VLLM_ROCM_MLA_FORCE_PS` to use_gluon_decode, but not
    # to forward_mqa's small-head multi-token branch, which returns
    # unconditionally for num_heads < 16 and max_qo_len > 1. Under DSpark every
    # target step is a verify step at max_qo_len = k+1, so that branch always
    # wins and the asm route below is never reached -- which is why run
    # 30971662995 loaded a qseqlen1 PS kernel and then served every step on
    # Gluon, with acceptance identical to the Gluon baseline (1.21 / 10.6% vs
    # 1.21 / 10.8%). One condition, so verify falls through to asm.
    # An anchor-matched patcher, not a context diff. The installed file drifts
    # from the GitHub source for the same image tag -- #51088's own hunks land
    # at offset +1, and the context-diff version of this step applied with
    # `fuzz 2 (offset -28 lines)` in run 30978612788. Exact-match anchors with a
    # count check fail loudly instead of landing somewhere plausible.
    python3 "$PATCH_DIR/vllm_dspark_ps_enable.py" \
        --diff-out "$RESULT_DIR/vllm_dspark_ps_enable.diff" \
        2>&1 | tee -a "$RESULT_DIR/vllm_pr51088.log"
    export VLLM_ROCM_MLA_FORCE_PS=1
    echo "MLA_FORCE_PS=1: small-head decode routed to the PS ASM kernel"
else
    echo "MLA_FORCE_PS=0: leaving small-head decode on Gluon (upstream default)"
fi

# ---- upstream PRs under test: #51011 and #51040 ------------------------------
# Both are OPEN and neither is merged. They are ported as anchor-matched
# patchers rather than context diffs for the reason documented on
# vllm_dspark_ps_enable.py: the installed file drifts from the GitHub source for
# the same image tag. Each patcher verified offline against the exact file
# GitHub serves for cb8104839c, applied after #51088 + vllm_dspark_ps_enable.py,
# and is idempotent.
#
# Default OFF. Each arm of the c8 A/B flips exactly one of these, so a branch
# diff is one line and the cell it produces is attributable.
PR51011="${PR51011:-1}"
PR51040="${PR51040:-1}"
PATCH_DIR="${PATCH_DIR:-$(cd "$(dirname "$0")/patches" && pwd)}"
VLLM_ROOT="${VLLM_ROOT:-$(python3 -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))')}"

# vllm#51011 -- size _mtp_decode_qlen from reorder_batch_threshold. For DSpark
# at k=3 that is 1 + 2*3 = 7, where vllm_dspark_ps_enable.py hardcodes 4. The
# only behavioral surface of #51011 under MLA_FORCE_PS=1; see the patcher.
if [ "$PR51011" = "1" ]; then
    PR51011_PY="$PATCH_DIR/vllm_pr51011_reorder_qlen.py"
    [ -f "$PR51011_PY" ] || { echo "ERROR: $PR51011_PY missing" >&2; exit 1; }
    # No `|| true`: an unpatched run would be a silent baseline republished
    # under this experiment's name.
    python3 "$PR51011_PY" --diff-out "$RESULT_DIR/vllm_pr51011.diff" \
        2>&1 | tee "$RESULT_DIR/vllm_pr51011.log"
    echo "PR51011=1: _mtp_decode_qlen <- reorder_batch_threshold"
else
    echo "PR51011=0: _mtp_decode_qlen left at the whitelist value (k+1)"
fi

# vllm#51040 -- fp8 asm MLA prefill for non-divisor head counts (K3 = 12/rank at
# TP8). MAY BE INERT ON THIS TRACE: forward_mha bails to the bf16 route whenever
# chunked_context is set, and at ISL p50 ~60K with ~90% prefix hit most prefills
# are chunked. Grep server.log for "LOCAL PATCH vllm#51040" to see whether the
# gate actually opened before reading anything into the throughput.
if [ "$PR51040" = "1" ]; then
    PR51040_PY="$PATCH_DIR/vllm_pr51040_fp8_prefill_pad16.py"
    [ -f "$PR51040_PY" ] || { echo "ERROR: $PR51040_PY missing" >&2; exit 1; }
    python3 "$PR51040_PY" --diff-out "$RESULT_DIR/vllm_pr51040.diff" \
        2>&1 | tee "$RESULT_DIR/vllm_pr51040.log"
    echo "PR51040=1: fp8 asm prefill extended to 12 heads via pad-to-16"
else
    echo "PR51040=0: fp8 asm prefill stays gated on num_heads % 16 == 0"
fi

python3 -c "import ast,sys;ast.parse(open(sys.argv[1]).read())" \
    "$VLLM_ROOT/vllm/v1/attention/backends/mla/rocm_aiter_mla.py"

# vllm-project/vllm#51171 -- "[ROCm][MLA] Reach FULL cudagraphs for AITER MLA
# speculative decoding". ONLY the triton_mla.py hunk is taken.
#
# The DRAFTER runs TRITON_MLA, whose _cudagraph_support is a class constant
# reporting UNIFORM_SINGLE_TOKEN_DECODE. The engine takes the MINIMUM over
# attention groups, so that one group downgrades the WHOLE engine off full
# cudagraphs -- target included. The PR adds a get_cudagraph_support classmethod
# returning UNIFORM_BATCH when the KV cache group is
# non_causal_multi_token_decode, which a DSpark draft group is.
#
# Not taken: the PR's rocm_aiter_mla.py hunks move the small-head Gluon flatten
# expansion into _build_decode to remove device->host syncs. That is the exact
# path MLA_FORCE_PS bypasses (verify goes to the asm PS kernel instead), and it
# would collide with both #51088 and vllm_dspark_ps_enable.py on forward_mqa.
# Also not taken: the gpu_worker.py kv_cache_size_tokens fix, which is only
# load-bearing for the persistent verify buffers those hunks introduce.
#
# Verified against this image: non_causal_multi_token_decode exists on the spec
# (kv_cache_interface.py:397, OR'd across the group by MLAAttentionSpec.merge at
# 458), so this is not a no-op.
MLA_TRITON_CG="${MLA_TRITON_CG:-1}"
if [ "$MLA_TRITON_CG" = "1" ]; then
    TRITON_CG_DIFF="$(cd "$(dirname "$0")/patches" && pwd)/vllm_pr51171_triton_mla_cg.diff"
    TCG_ROOT="$(python3 -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
    [ -f "$TRITON_CG_DIFF" ] || { echo "ERROR: $TRITON_CG_DIFF missing" >&2; exit 1; }
    ( cd "$TCG_ROOT" && patch -p1 --forward --batch < "$TRITON_CG_DIFF" ) \
        2>&1 | tee "$RESULT_DIR/vllm_pr51171.log"
    python3 -c "import ast,sys;ast.parse(open(sys.argv[1]).read())" \
        "$TCG_ROOT/vllm/v1/attention/backends/mla/triton_mla.py"
    echo "MLA_TRITON_CG=1: drafter no longer downgrades the engine off FULL cudagraphs"
else
    echo "MLA_TRITON_CG=0: leaving the drafter's cudagraph support at upstream default"
fi

# ---- qSeqLen padding for k=2 -------------------------------------------------
# The gfx950 asm registry has kernels at qSeqLen 1, 2, 4, 8 -- no 3. For K3's
# bf16-query/fp8-KV combination there is exactly ONE row, qSeqLen=4, so k=2
# (verify qlen = k+1 = 3) has nothing to route to under MLA_FORCE_PS=1.
#
# This pads the verify block 3 -> 4, runs the qSeqLen=4 kernel, and trims the
# pad rows off the output. Padding is PREPENDED, not appended: the kernel
# derives row t's causal window from seq_len and max_qo_len, and seq_len does
# not change when the query tensor is padded, so raising max_qo_len shifts every
# window down by one. Prepending puts the real rows at t = 1..3 and restores
# their original offsets. Verified arithmetically:
#
#   real qlen 3    real-row windows = [ctx+1, ctx+2, ctx+3]
#   append         real-row windows = [ctx+0, ctx+1, ctx+2]   <- WRONG
#   prepend        real-row windows = [ctx+1, ctx+2, ctx+3]   <- correct
#
# THIS RESTS ON AN UNVERIFIED ASSUMPTION about how the asm kernel derives its
# causal window -- it is arithmetic, not a reading of the kernel source. A wrong
# window yields fluent, plausible output with subtly wrong attention, which no
# throughput number will catch. The gsm8k eval is the gate, not the perf run.
# That is also why this branch runs SPEC_REJECTION_METHOD=block: under synthetic
# the eval fails by construction and could not validate anything.
MLA_QLEN_PAD="${MLA_QLEN_PAD:-1}"
if [ "$MLA_QLEN_PAD" = "1" ]; then
    QLEN_PAD_PY="$(cd "$(dirname "$0")/patches" && pwd)/vllm_qlen_pad.py"
    [ -f "$QLEN_PAD_PY" ] || { echo "ERROR: $QLEN_PAD_PY missing" >&2; exit 1; }
    python3 "$QLEN_PAD_PY" --diff-out "$RESULT_DIR/vllm_qlen_pad.diff" \
        2>&1 | tee "$RESULT_DIR/vllm_qlen_pad.log"
    export VLLM_ROCM_MLA_QLEN_PAD=1
    echo "MLA_QLEN_PAD=1: verify qlen padded to the next supported asm qSeqLen"
else
    echo "MLA_QLEN_PAD=0: no qSeqLen padding (k must be one of 0,1,3,7)"
fi

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
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    "${MM_ARGS[@]}"
    --max-num-seqs "$MAX_NUM_SEQS"
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --max-model-len 1048576
    --enable-prefix-caching
    "${MNBT_ARGS[@]}"
    "${COMPILATION_CONFIG_ARGS[@]}"
    "${LMCACHE_MAMBA_CACHE_MODE_ARGS[@]}"
    "${KV_CACHE_DTYPE_ARGS[@]}"
    "${SPEC_ARGS[@]}"
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
