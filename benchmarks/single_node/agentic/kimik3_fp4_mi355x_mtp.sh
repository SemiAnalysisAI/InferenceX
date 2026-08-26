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
#   MAX_NUM_BATCHED_TOKENS   8192   (default; 2048 under K3_NATIVE_KV_PAGE)
#   AITER_A8W4               1      (reference; 0 = aiter a16w4 MoE path)
#   LANGUAGE_MODEL_ONLY      true   
#   KV_CACHE_DTYPE           fp8    (default for every arm; =auto for a bf16 A/B)
#   KV_BLOCK_SIZE            unset  (unset -> 1536 for LMCache/native-page,
#                                    4608 for other offloaded cells, else vLLM sizes it)
#   MAX_MODEL_LEN            1M     
#   SPEC_DECODE              true   (=false for the no-spec wrapper)
#   SPEC_NUM_TOKENS          2      (DSpark draft length; validated by the _mtp config)
#   DCP_SIZE                 1      (>1 sends the target's KV across the TP ranks;
#                                    needs vllm#51705, applied in-container)

source "$(dirname "$0")/../../benchmark_lib.sh"

wait_for_amd_gpu_clean

du -h /dev/shm

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

# ---- Decode context parallelism ----------------------------------------------
# Normalised here rather than in the parallelism block below because the patch
# script reads it: #51705 is applied only when DCP is actually on.
DCP_SIZE="${DCP_SIZE:-1}"
export DCP_SIZE

# ---- In-container patches ----------------------------------------------------
# ---- Out-of-tree overlay -----------------------------------------------------
# A unified diff of merged upstream PRs dropped into site-packages, for images
# newer than the ones k3_patches/ targets. Currently vllm#51705 (DSpark under
# DCP) + #52707 (KV block-pool clamp) + #53222 (AITER MoE chunking), merged onto
# the nightly-a9a17e70 base.
#
# On that base the squashed k3_patches/vllm_51705_dcp.patch no longer applies
# (the PR was rewritten: the DCP gate is now the capability flag
# supports_non_causal_multi_token_dcp on the builder class, not the old
# dcp_local_verify_row_lens plumbing), which is why this exists as a separate
# merged diff rather than another entry in the patch script.
#
# SELF-SELECTING: dry-run first, and skip quietly if the diff does not match the
# running image. That keeps every older cell on its existing patch path instead
# of hard-failing them, while a matching image gets the overlay with no config
# plumbing. `patch --forward` makes it idempotent.
K3_OVERLAY_APPLIED=0
# Absolute, always: the patch is fed to `patch` on the far side of a
# `cd "$SITE_PKGS"`, so a relative path resolves against site-packages and
# silently vanishes even though the -f test passed from the workspace root.
K3_OVERLAY_PATCH="${K3_OVERLAY_PATCH:-$(cd "$(dirname "$0")" && pwd)/k3_patches/vllm_nightly_a9a17e70_3pr.patch}"
case "$K3_OVERLAY_PATCH" in
    /*) ;;
    *) K3_OVERLAY_PATCH="$(cd "$(dirname "$K3_OVERLAY_PATCH")" && pwd)/$(basename "$K3_OVERLAY_PATCH")" ;;
esac
if [ -f "$K3_OVERLAY_PATCH" ]; then
    SITE_PKGS=$(python3 -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))')
    if ( cd "$SITE_PKGS" && patch -p1 --forward --batch --dry-run < "$K3_OVERLAY_PATCH" ) \
            >/tmp/k3_overlay_dryrun.log 2>&1; then
        echo "Applying K3 overlay $K3_OVERLAY_PATCH into $SITE_PKGS"
        if ( cd "$SITE_PKGS" && patch -p1 --forward --batch < "$K3_OVERLAY_PATCH" ); then
            K3_OVERLAY_APPLIED=1
        elif [ "${REQUIRE_K3_OVERLAY:-0}" = "1" ]; then
            exit 1
        fi
    else
        # Print why. A silent skip here costs a whole CI cycle to diagnose, and
        # the same diff can apply against the registry image while failing
        # against a pre-converted squashfs of nominally the same tag.
        echo "K3 overlay does not match this image, skipping: $K3_OVERLAY_PATCH"
        echo "--- overlay dry-run output (first 40 lines) ---"
        head -40 /tmp/k3_overlay_dryrun.log || true
        echo "--- installed vLLM ---"
        python3 -c 'import vllm;print("vllm",vllm.__version__)' || true
        echo "----------------------------------------------"
        if [ "${REQUIRE_K3_OVERLAY:-0}" = "1" ]; then
            exit 1
        fi
    fi
elif [ "${REQUIRE_K3_OVERLAY:-0}" = "1" ]; then
    echo "Required K3 overlay is missing: $K3_OVERLAY_PATCH" >&2
    exit 1
fi

# ---- In-container patches ----------------------------------------------------
# Four fixes, all confined to this container's site-packages, all idempotent
# and all self-disabling once the image ships them:
#   [1] aiter pybind11 internals mismatch  -> unblocks ROCM_AITER_FA prefill
#   [2] TritonMLA cudagraph support        -> FULL cudagraphs for DSpark (5.52x TPOT)
#   [3] KV block-pool negative-count clamp -> stops the mid-run engine crash
#   [4] vllm#51705                         -> DSpark under DCP (DCP_SIZE>1 only)
# Set SKIP_KIMI_PATCHES=1 to run stock.
#
# The legacy patch script edits triton_mla.py and single_type_kv_cache_manager.py,
# two of the files the overlay also carries. Running it first shifts their
# context so the overlay no longer applies and #51705 is silently lost, so the
# overlay goes first and supersedes the script when it lands.
if [ "$K3_OVERLAY_APPLIED" = "1" ]; then
    SKIP_KIMI_PATCHES=1
    export SKIP_KIMI_PATCHES
fi
bash "$(dirname "$0")/apply_k3_container_patches.sh" || true

# Fail closed. vLLM ACCEPTS dcp>1 with dspark and only dies later at model init,
# so a clean startup banner is not evidence DCP works -- assert the capability
# up front rather than discovering it 10 minutes into a bring-up.
if [ "$DCP_SIZE" -gt 1 ]; then
    python3 - <<'PYEOF' || exit 1
import sys
from vllm.v1.attention.backends.mla.triton_mla import TritonMLAMetadataBuilder as B
ok = bool(getattr(B, "supports_non_causal_multi_token_dcp", False))
print(f"DCP capability check: supports_non_causal_multi_token_dcp={ok}")
if not ok:
    print("vLLM lacks vllm#51705; dcp>1 with DSpark would fail at model init.",
          file=sys.stderr)
sys.exit(0 if ok else 1)
PYEOF
fi

# ---- Reference env block ----------------------------------------------------
# Keep ALL of these. Commenting them out does not avoid the AITER FMHA crash:
# that crash is gated on VLLM_ROCM_USE_AITER alone (AiterFlashAttnPrefillBackend
# .is_available() consults only rocm_aiter_ops.is_enabled()), so disabling the
# others just loses the MoE kernels while keeping the failure.
export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
export VLLM_ROCM_USE_AITER=1
export SAFETENSORS_FAST_GPU=1
export VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
# BOTH names. vllm#50582 renamed the vLLM-side flag with no back-compat alias,
# but aiter still reads the old one (aiter/fused_moe.py defaults it to "0").
# Setting only one makes vLLM shuffle w13 for one gate mode while aiter runs
# the kernels for the other: gsm8k 0.00 instead of 1.00, with no other symptom.
# An agentic cell CANNOT detect this -- DSpark runs synthetic acceptance here,
# so the accept rate is a supplied constant, not a measurement.
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

cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---- KV page size ------------------------------------------------------------
# K3 is a Mamba-hybrid (the KDA linear-attention layers), and attaching ANY KV
# connector makes LMCache's validate_mamba_step_alignment() require
#   block_size <= max_num_batched_tokens < 2 * block_size
# so every prefill step crosses exactly one block boundary and each boundary
# gets a recurrent-state snapshot (mamba_cache_mode=align only snapshots at
# step end, so a step spanning two blocks would store null-block garbage under
# a valid token hash and corrupt anything resuming from that prefix).
#
# The block_size it reads is cache_config.block_size -- the ATTENTION KV page,
# NOT the scheduler block that --mamba-block-size feeds. vLLM auto-sizes that
# page to 1536 here, whose legal window [1536, 3072) excludes the 8192 batch
# this recipe wants, and no mamba-side knob can move it.
#
# So set the page instead. 4608 = 3 * 1536 keeps it an exact multiple of the
# size vLLM picked for itself, and its window [4608, 9216) contains 8192.
# kvnone cells are unaffected: no connector, no constraint, no override.
# K3_NATIVE_KV_PAGE=1 opts out of the override and keeps vLLM's own page.
# That matters for LMCache: its _MambaUnifiedViewEdit views the KDA tensor as
# [num_blocks, page, 1, -1], and the per-block element count (832512 measured)
# divides by the native 1536 page exactly (542) but not by 4608 (180.667). So
# forcing the page to reach mnbt 8192 is what breaks the mamba view. The cost
# of opting out is mnbt, which must then fall in [1536, 3072).
KV_BLOCK_SIZE="${KV_BLOCK_SIZE:-}"
if [ -z "$KV_BLOCK_SIZE" ] && [ "${K3_AUTO_KV_PAGE:-0}" != "1" ] && agentic_kv_offload_enabled; then
    if [ "${KV_OFFLOAD_BACKEND:-}" = "lmcache" ] || [ "${K3_NATIVE_KV_PAGE:-0}" = "1" ]; then
        # Native page, stated explicitly rather than left unset. Leaving it
        # empty made the LMCache chunk alignment fall back to a literal 256 and
        # request 2048 where vLLM wanted 1536*8=12288 -- the page has to be a
        # known number here because the chunk is derived from it. 1536 is what
        # vLLM auto-sizes to for this model (confirmed twice: the first
        # step-alignment error reported block_size=1536, and the mamba view's
        # 832512 elems/block divides by it exactly).
        KV_BLOCK_SIZE=1536
        MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
    else
        KV_BLOCK_SIZE=4608
    fi
fi
KV_BLOCK_ARGS=()
if [ -n "$KV_BLOCK_SIZE" ]; then
    KV_BLOCK_ARGS=(--block-size "$KV_BLOCK_SIZE")
fi

# Optional, unset by default. Feeds the scheduler block via
# lcm(attention_block * dcp, mamba_block); forcing it interacts badly with a
# non-power-of-two page, so leave vLLM to size it unless deliberately sweeping.
MAMBA_BLOCK_ARGS=()
if [ -n "${MAMBA_BLOCK_SIZE:-}" ]; then
    MAMBA_BLOCK_ARGS=(--mamba-block-size "$MAMBA_BLOCK_SIZE")
fi

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
    SIMPLE_EXTRA_CONFIG="\"cpu_bytes_to_use_per_rank\":$CPU_BYTES_PER_RANK,\"lazy_offload\":$SIMPLE_LAZY_OFFLOAD"
    if [ -n "${SIMPLE_LAZY_OFFLOAD_WATERMARK_RATIO:-}" ]; then
        SIMPLE_EXTRA_CONFIG+=",\"lazy_offload_watermark_ratio\":$SIMPLE_LAZY_OFFLOAD_WATERMARK_RATIO"
    fi
    OFFLOAD_ARGS=(
        --kv-transfer-config
        "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{$SIMPLE_EXTRA_CONFIG}}"
    )
    echo "SimpleCPUOffloadConnector: ${CPU_BYTES_PER_RANK} B/rank x ${TP} ranks, lazy_offload=$SIMPLE_LAZY_OFFLOAD"
    ;;
  lmcache)
    require_agentic_kv_offload_backend "$KV_OFFLOAD_BACKEND"
    # Ported from dsv4_fp4_mi355x_vllm_mtp.sh, minus its from-source LMCache
    # build -- a released ROCm wheel is not a code fork, so install it rather
    # than compiling. Same --find-links pattern the minimaxm3 recipes use.
    #
    # The image ships 0.5.3; LMCACHE_VERSION overrides it. Default is the
    # nightly channel, which is the newest build but a ROLLING window: old dev
    # builds get pruned within days, so an exact rerun of this cell may become
    # unobtainable. Set LMCACHE_ROCM_TAG=v0.5.4rc4-rocm with the matching
    # LMCACHE_VERSION for a pinnable tagged release instead.
    #
    # MP connector, not the in-engine LMCacheConnectorV1: the integrated one
    # faults at conc>=4 on ROCm with a GPU memory access fault.
    # UNPINNED by default, deliberately. The nightly channel keeps exactly one
    # dev wheel and prunes the rest: 0.5.5.dev19 was published, used, and gone
    # within the hour, which is a hard install failure for anything that names
    # it. Resolve whatever the channel currently holds and record it instead.
    # Set LMCACHE_VERSION (with LMCACHE_ROCM_TAG=v0.5.4rc4-rocm for a tagged
    # release) when an exact rerun matters more than being current.
    LMCACHE_VERSION="${LMCACHE_VERSION:-}"
    LMCACHE_ROCM_TAG="${LMCACHE_ROCM_TAG:-nightly-rocm}"
    LMCACHE_ROCM_INDEX="https://github.com/LMCache/LMCache/releases/expanded_assets/${LMCACHE_ROCM_TAG}"
    if [ -n "$LMCACHE_VERSION" ]; then
        LMCACHE_SPEC="lmcache==${LMCACHE_VERSION}"
    else
        LMCACHE_SPEC="lmcache"
    fi
    echo "Installing ${LMCACHE_SPEC} from ${LMCACHE_ROCM_TAG}"
    # --pre: the ROCm wheels are dev builds and pip skips them otherwise.
    # --upgrade: the image already ships 0.5.3, and without it pip calls the
    # requirement satisfied and silently leaves that in place.
    pip install --no-cache-dir --pre --upgrade "$LMCACHE_SPEC" \
        --find-links "$LMCACHE_ROCM_INDEX"

    # NOTE: LMCache is NOT usable against a post-vllm#51718 nightly as of
    # 2026-08-26 (newest ROCm wheel 0.5.5.dev24). It calls the removed
    # get_kv_cache_layout, and shimming that back only moves the failure into
    # _MambaUnifiedViewEdit, which cannot view this vLLM's mamba KV tensor
    # under EITHER legacy layout: 1643378688/1974 = 832512 elems per block,
    # and 832512/4608 = 180.667 for both NHD and HND. The mismatch is
    # structural, not a layout choice, so it needs an upstream LMCache fix.
    # Do not re-add a layout shim; it lets LMCache past a check that is
    # correctly stopping it and risks mis-viewing KV rather than failing.

    # LMCache/LMCache#4729 "Restore vLLM KV cache layout discovery after
    # vllm#51718". #51718 removed vllm.v1.attention.backends.utils.
    # get_kv_cache_layout, which this integration imported; without the PR the
    # kv_layout hint silently vanishes and registration dies with
    # "Unsupported kv_layout: none". The PR reads the layout from its
    # post-#51718 home (CacheConfig.kv_cache_layout) and translates
    # LBNHC->NHD / LBHNC->HND, raising NotImplementedError on the four layouts
    # LMCache cannot transfer rather than guessing.
    #
    # Applied as a patch because the PR is still OPEN, so no wheel carries it.
    # Drop this once it lands in a nightly-rocm build.
    LMC_PATCH="$(cd "$(dirname "$0")" && pwd)/k3_patches/lmcache_pr4729_kv_layout.patch"
    LMC_SITE=$(python3 -c 'import lmcache,os;print(os.path.dirname(os.path.dirname(lmcache.__file__)))')
    echo "Applying LMCache #4729 from $LMC_PATCH into $LMC_SITE"
    ( cd "$LMC_SITE" && patch -p1 --forward --batch < "$LMC_PATCH" ) || true
    # Kimi-K3's unified Mamba tensor has dim-0 padding between blocks. The
    # upstream edit exposes it as a rank-4 attention view, whose LMCache copy
    # kernels assume tightly packed blocks and reject the registration. Expose
    # the opaque recurrent state through LMCache's rank-3 NB_BS_HS format,
    # which carries the real stride(0), without copying the multi-GiB allocation.
    LMC_MAMBA_PATCH="$(cd "$(dirname "$0")" && pwd)/k3_patches/lmcache_k3_mamba_padded_view.patch"
    echo "Applying Kimi-K3 padded Mamba view fix from $LMC_MAMBA_PATCH into $LMC_SITE"
    ( cd "$LMC_SITE" && patch -p1 --forward --batch < "$LMC_MAMBA_PATCH" ) || true
    # Fail closed: without the discovery fix the run dies ~40 min later at KV
    # registration. Also assert the padded Mamba view uses the stride-aware
    # rank-3 transfer format rather than the tight-only rank-4 attention path.
    python3 - <<'PYLMC' || exit 1
import sys
import inspect
from lmcache.integration.vllm import kv_cache_group_edits as edits
from lmcache.integration.vllm import utils as u
layout_ok = hasattr(u, "translate_vllm_kv_cache_layout")
mamba_source = inspect.getsource(edits._MambaUnifiedViewEdit.apply)
mamba_ok = "spec.block_size, -1" in mamba_source
print(f"LMCache #4729 check: translate_vllm_kv_cache_layout present={layout_ok}")
print(f"LMCache K3 Mamba view check: rank3_stride_aware={mamba_ok}")
sys.exit(0 if layout_ok and mamba_ok else 1)
PYLMC

    python3 -c "import lmcache.integration.vllm.lmcache_mp_connector" >/dev/null
    # Assert rather than trust: KV_OFFLOAD_BACKEND_METADATA reports a version
    # into the aggregated result, and a silent mismatch there mislabels the run.
    python3 - "$LMCACHE_VERSION" <<'PYVER' || exit 1
import sys, lmcache
want, got = sys.argv[1], lmcache.__version__
print(f"lmcache installed: {got}" + (f" (requested {want})" if want else " (unpinned)"))
# Only assert when pinned; unpinned resolves to whatever the channel holds, and
# that resolved string is what gets reported, so there is nothing to contradict.
sys.exit(0 if not want or got.split("+")[0] == want.split("+")[0] else 1)
PYVER

    LMCACHE_LOG="${LMCACHE_LOG:-$RESULT_DIR/lmcache.log}"
    LMCACHE_PID=""

    cleanup_lmcache_server() {
        if [[ -n "$LMCACHE_PID" ]] && kill -0 "$LMCACHE_PID" 2>/dev/null; then
            kill "$LMCACHE_PID" 2>/dev/null || true
            wait "$LMCACHE_PID" 2>/dev/null || true
        fi
    }
    cleanup_agentic_services() {
        local exit_code=$?
        trap - EXIT INT TERM
        set +e
        stop_background_process_tree "$SERVER_PID" "vLLM server" 60
        cleanup_lmcache_server
        exit "$exit_code"
    }
    trap cleanup_agentic_services EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    LMCACHE_HOST="${LMCACHE_HOST:-127.0.0.1}"
    LMCACHE_PORT="${LMCACHE_PORT:-5555}"
    LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
    # LMCacheMPConnector concatenates lmcache.mp.host and port into the ZMQ
    # endpoint, so the server binds a raw host but the connector gets tcp://.
    LMCACHE_CONNECT_HOST="${LMCACHE_CONNECT_HOST:-tcp://$LMCACHE_HOST}"
    # The MP server owns the whole pool; do not divide by TP the way the
    # integrated backend does. Must stay under container /dev/shm or LMCache
    # silently falls back to slow pickle transfers instead of the SHM path.
    LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-$TOTAL_CPU_DRAM_GB}"
    LMCACHE_L1_INIT_SIZE_GB="${LMCACHE_L1_INIT_SIZE_GB:-20}"
    # Read locks are leases on chunks lookup promised vLLM it could retrieve.
    # The 300s default expires mid-queue on long-context agentic turns.
    LMCACHE_L1_READ_TTL_SECONDS="${LMCACHE_L1_READ_TTL_SECONDS:-7200}"
    # LMCache requires the chunk to be a multiple of every hybrid KV group's
    # logical tokens_per_block. K3 has two relevant alignments on this stack:
    # attention is KV_BLOCK_SIZE scaled by DCP, while the replicated Mamba
    # state uses 24576 tokens per block. With the native 1536-token page and
    # DCP8 these are 12288 and 24576, so their LCM is 24576. Derive the LCM so
    # explicit chunk overrides are checked against both groups as well.
    if [ -z "${KV_BLOCK_SIZE:-}" ]; then
        echo "Error: KV_BLOCK_SIZE must be known to derive the LMCache chunk" >&2
        exit 1
    fi
    LMCACHE_BLOCK_ALIGN=$(( KV_BLOCK_SIZE * ${DCP_SIZE:-1} ))
    LMCACHE_MAMBA_BLOCK_ALIGN="${LMCACHE_MAMBA_BLOCK_ALIGN:-24576}"
    gcd_a=$LMCACHE_BLOCK_ALIGN
    gcd_b=$LMCACHE_MAMBA_BLOCK_ALIGN
    while [ "$gcd_b" -ne 0 ]; do
        gcd_tmp=$(( gcd_a % gcd_b ))
        gcd_a=$gcd_b
        gcd_b=$gcd_tmp
    done
    LMCACHE_CHUNK_ALIGN=$(( LMCACHE_BLOCK_ALIGN / gcd_a * LMCACHE_MAMBA_BLOCK_ALIGN ))
    LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-$LMCACHE_CHUNK_ALIGN}"
    if [ $((LMCACHE_CHUNK_SIZE % LMCACHE_CHUNK_ALIGN)) -ne 0 ]; then
        echo "Error: LMCACHE_CHUNK_SIZE=$LMCACHE_CHUNK_SIZE must be a multiple of" >&2
        echo "       ${LMCACHE_CHUNK_ALIGN} (LCM of attention ${LMCACHE_BLOCK_ALIGN}" >&2
        echo "       = block ${KV_BLOCK_SIZE} x dcp ${DCP_SIZE:-1}, and Mamba ${LMCACHE_MAMBA_BLOCK_ALIGN})" >&2
        exit 1
    fi
    echo "LMCache chunk ${LMCACHE_CHUNK_SIZE} (hybrid align ${LMCACHE_CHUNK_ALIGN}; attention ${LMCACHE_BLOCK_ALIGN}, Mamba ${LMCACHE_MAMBA_BLOCK_ALIGN})"
    LMCACHE_MAX_WORKERS="${LMCACHE_MAX_WORKERS:-$TP}"
    # Without this, identical prompts hash differently per process and the hit
    # rate is silently 0. Must be set on BOTH the server and vllm serve.
    export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
    export LMCACHE_BLOCKING_TIMEOUT_SECS=1200

    echo "Starting LMCache MP server (l1=${LMCACHE_L1_SIZE_GB} GB)..."
    LMCACHE_CMD=(
        lmcache server
        --host "$LMCACHE_HOST"
        --port "$LMCACHE_PORT"
        --http-host "$LMCACHE_HOST"
        --http-port "$LMCACHE_HTTP_PORT"
        --l1-size-gb "$LMCACHE_L1_SIZE_GB"
        --l1-init-size-gb "$LMCACHE_L1_INIT_SIZE_GB"
        --l1-read-ttl-seconds "$LMCACHE_L1_READ_TTL_SECONDS"
        --chunk-size "$LMCACHE_CHUNK_SIZE"
        --max-workers "$LMCACHE_MAX_WORKERS"
        --eviction-policy LRU
        --supported-transfer-mode lmcache_driven
    )
    printf '%q ' "${LMCACHE_CMD[@]}" > "$RESULT_DIR/lmcache_command.txt"
    printf '\n' >> "$RESULT_DIR/lmcache_command.txt"
    "${LMCACHE_CMD[@]}" > "$LMCACHE_LOG" 2>&1 &
    LMCACHE_PID=$!
    echo "LMCache server PID: $LMCACHE_PID"

    for ((i = 1; i <= ${LMCACHE_READY_ATTEMPTS:-300}; i++)); do
        if curl --output /dev/null --silent --fail \
            "http://127.0.0.1:${LMCACHE_HTTP_PORT}/healthcheck"; then
            echo "LMCache server healthy after ${i}s"
            break
        fi
        if [[ -n "$LMCACHE_PID" ]] && ! kill -0 "$LMCACHE_PID" 2>/dev/null; then
            echo "LMCache server died before becoming healthy. Log follows:" >&2
            cat "$LMCACHE_LOG" >&2 || true
            exit 1
        fi
        sleep 1
    done

    OFFLOAD_ARGS=(
        --kv-transfer-config
        "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"$LMCACHE_CONNECT_HOST\",\"lmcache.mp.port\":$LMCACHE_PORT,\"lmcache.mp.mq_timeout\":6000.0}}"
    )
    ;;
  *)
    echo "Error: unsupported KV_OFFLOAD_BACKEND '${KV_OFFLOAD_BACKEND:-}' (expected: vllm-simple, lmcache)" >&2
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

# DCP shards the target's KV cache across the TP ranks; the DSpark draft group
# stays replicated (the draft attends over the whole sequence and throws its
# decode LSE away), which is the substance of vllm#51705 -- unpatched, vLLM
# rejects the combination at config time. Interleave 1 is the round-robin
# sharding that PR's per-row verify window assumes. GPU count is unchanged: DCP
# reuses the TP ranks.
#
# NB on this recipe DCP is a KV capacity REGRESSION, not the +40.6% #51705
# reports: replicating the draft group adds 19 padding layers ("may waste at
# most 380.00%") and 2,795,917 tokens at dcp=1 becomes 1,932,335 at dcp=8. The
# PR's gain is measured against a dcp=1 baseline 2.2x worse than ours.
#
# Comm backend: a2a is what #51705 validated, but run 32215249474 took an
# HSA_STATUS_ERROR_EXCEPTION 0x1016 on all 8 ranks inside aiter moe_sorting
# (apply_routed_input_transform) on the first real ~99k-token prefill, on both
# c4 and c8, where the dcp=1 baseline on the same image aborts zero times. ag_rs
# is vLLM's default and the less exotic path, so it is the default here while
# that is isolated; set DCP_COMM_BACKEND=a2a to go back.
DCP_COMM_BACKEND="${DCP_COMM_BACKEND:-ag_rs}"
DCP_ARGS=()
if [ "$DCP_SIZE" -gt 1 ]; then
    if [ $(( TP % DCP_SIZE )) -ne 0 ]; then
        echo "Error: TP=$TP must be divisible by DCP_SIZE=$DCP_SIZE." >&2
        exit 1
    fi
    DCP_ARGS=(
        --decode-context-parallel-size "$DCP_SIZE"
        --dcp-comm-backend "$DCP_COMM_BACKEND"
        --cp-kv-cache-interleave-size 1
    )
fi

# ---- Speculative ------------------------------------------------------------
SPEC_ARGS=()
if [ "${SPEC_DECODE:-true}" = "true" ]; then
    SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-2}"
    SYNTHETIC_ACCEPT_LEN=2.51
    # Hosts that pre-stage the draft as a plain directory rather than an HF hub
    # cache cannot resolve the repo id, and downloading it needs egress + a token.
    DRAFT_MODEL="${DRAFT_MODEL:-Inferact/Kimi-K3-DSpark}"

    if [ "${EVAL_ONLY:-false}" = "true" ]; then
        SPEC_ARGS=(
            --speculative-config
            "{\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\": \"block\"}"
        )
    else
        SPEC_ARGS=(
            --speculative-config
            "{\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":$SPEC_NUM_TOKENS,\"method\":\"dspark\",\"attention_backend\":\"TRITON_MLA\",\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
        )
    fi
fi

# ---- Async scheduling / KV block-pool stability ------------------------------
# DSpark is the ONLY spec method exempted from vLLM's async-scheduling disable
# list (config/vllm.py:1181), so async_scheduling resolves True here. That gives
# max_concurrent_batches = pp_size + 1 = 2 (vllm.py:563-569), and with
# kv_role=kv_both (is_kv_consumer=True) the scheduler sets defer_block_free=True
# (sched/scheduler.py:155-157). Its own comment: "a step may still be writing a
# freed request's KV blocks. A consumer KV Connector can reallocate and fill
# those blocks via a load that isn't ordered against that write."
#
# That limbo state matches our crash signature exactly -- the engine dies with
#   block_pool.py:667  assert block.ref_cnt == 0
# i.e. a block sitting on the FREE list that is still referenced. Crash time
# scales inversely with concurrency: c10 survived 3612 s, c12 died at 487 s,
# c16 at 354 s. Note vLLM already disables async scheduling for ROCm DeepEP DBO
# because "that combination can corrupt" state.
#
# Setting max_concurrent_batches back to 1 makes defer_block_free unreachable.
# Cost: async scheduling exists to fill GPU-utilisation gaps, so expect to give
# some throughput back. Set ASYNC_SCHEDULING=1 to restore the default.
ASYNC_SCHED_ARGS=()
if [ "${ASYNC_SCHEDULING:-0}" != "1" ]; then
    ASYNC_SCHED_ARGS=(--no-async-scheduling)
fi

# ---- MLA prefill backend -----------------------------------------------------
# On ROCm the prefill priority is [ROCM_AITER_FA, FLASH_ATTN]. ROCM_AITER_FA
# JIT-builds module_fmha_fwd_bf16_opus at runtime; that module registers its own
# aiter_tensor_t, distinct from the one in the prebuilt module_aiter_core, so the
# first call dies with:
#   TypeError: fmha_fwd_bf16_opus_fwd(): incompatible function arguments
# during compile_or_warm_up_model -> _dummy_run, before the server binds.
# Pinning FLASH_ATTN keeps every AITER MoE kernel (and its throughput) while
# skipping only the broken FMHA prefill path.
# UPDATE: the AITER packaging issue is now fixed at source by
# apply_kimi_k3_patches.sh (run above), so ROCM_AITER_FA is usable again and
# is the default. Measured on 8x MI355X / Kimi-K3 MXFP4 TP8, cold prefill:
#   ~24k ctx  FLASH_ATTN 12,953 -> AITER 13,524 tok/s  (+4.4%)
#   ~93k ctx  FLASH_ATTN 11,174 -> AITER 13,423 tok/s  (+20.1%)
# This workload averages ~99k input tokens, so the ~93k figure is the relevant
# one. Set MLA_PREFILL_BACKEND=FLASH_ATTN to fall back if AITER regresses.
#
# Under DCP the default flips back to FLASH_ATTN. Runs 32215249474 and
# 32216365989 both died with an HSA_STATUS_ERROR_EXCEPTION 0x1016 on all 8 ranks
# on the first real ~99k-token prefill -- never in warmup, never at dcp=1. The
# reported kernel moved between runs (aiter moe_sorting, then the attn_res
# triton kernel) with a generic `unspecified launch failure`, which is what an
# ASYNC fault in an earlier kernel looks like: the next launch is simply the one
# that finds the queue dead. Both reported sites sit downstream of the MLA
# prefill, and mla_attention.py:809-811,839-840 already route dcp_world_size>1
# away from the fast paths onto the chunked-context merge, so AITER's FMHA
# prefill is the standing suspect. Set MLA_PREFILL_BACKEND explicitly to
# override either default.
if [ "$DCP_SIZE" -gt 1 ]; then
    MLA_PREFILL_BACKEND="${MLA_PREFILL_BACKEND:-FLASH_ATTN}"
fi
MLA_PREFILL_BACKEND="${MLA_PREFILL_BACKEND:-ROCM_AITER_FA}"
ATTENTION_ARGS=()
if [ -n "${ATTENTION_BACKEND:-}" ]; then
    ATTENTION_ARGS+=(--attention-backend "$ATTENTION_BACKEND")
fi
if [ -n "${ATTENTION_CONFIG_JSON:-}" ]; then
    ATTENTION_ARGS+=(--attention-config "$ATTENTION_CONFIG_JSON")
elif [ -n "$MLA_PREFILL_BACKEND" ]; then
    ATTENTION_ARGS+=(
        --attention-config
        "{\"mla_prefill_backend\":\"$MLA_PREFILL_BACKEND\"}"
    )
fi

# ---- HIP graph ------------------------------------------------------------
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$(( CONC * 2 ))}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-$(( MAX_NUM_SEQS * 3 ))}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-$(seq -s, 1 "$MAX_CUDAGRAPH_CAPTURE_SIZE")}"
COMPILATION_CUSTOM_OPS="${COMPILATION_CUSTOM_OPS:-\"+fused_rms_norm_gated\"}"
COMPILATION_CONFIG_ARGS=(--compilation-config "{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"max_cudagraph_capture_size\":$MAX_CUDAGRAPH_CAPTURE_SIZE,\"custom_ops\":[$COMPILATION_CUSTOM_OPS],\"cudagraph_capture_sizes\":[$CUDAGRAPH_CAPTURE_SIZES]}")

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"

STREAM_ARGS=()
if [ -n "${STREAM_INTERVAL:-}" ]; then
    STREAM_ARGS=(--stream-interval "$STREAM_INTERVAL")
fi

PREFIX_CACHE_ARGS=()
if [ -n "${PREFIX_MATCH_UNIT:-}" ]; then
    PREFIX_CACHE_ARGS+=(--prefix-match-unit "$PREFIX_MATCH_UNIT")
fi
if [ -n "${PREFIX_CACHING_HASH_ALGO:-}" ]; then
    PREFIX_CACHE_ARGS+=(--prefix-caching-hash-algo "$PREFIX_CACHING_HASH_ALGO")
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
    "${DCP_ARGS[@]}"
    --load-format fastsafetensors
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --language-model-only
    --max-num-seqs "$MAX_NUM_SEQS"
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
    --max-model-len 1048576
    "${STREAM_ARGS[@]}"
    --enable-prefix-caching
    "${PREFIX_CACHE_ARGS[@]}"
    --kv-cache-dtype "fp8"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}"
    "${KV_BLOCK_ARGS[@]}"
    "${MAMBA_BLOCK_ARGS[@]}"
    "${ASYNC_SCHED_ARGS[@]}"
    "${ATTENTION_ARGS[@]}"
    "${COMPILATION_CONFIG_ARGS[@]}"
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
