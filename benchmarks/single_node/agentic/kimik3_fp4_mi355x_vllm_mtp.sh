#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"

# Reproduction of the accepted C1 baseline and the compiled C16 candidate on
# nightly-46638857.
if [ "${SPEC_DECODING:-none}" != "mtp" ]; then
    echo "Error: this recipe requires DSpark speculative decoding." >&2
    exit 1
fi

if [ -r /proc/sys/kernel/numa_balancing ]; then
    read -r numa_balancing </proc/sys/kernel/numa_balancing
    if [ "$numa_balancing" != "0" ]; then
        echo "Error: Kimi-K3 tuning requires kernel.numa_balancing=0." >&2
        exit 1
    fi
fi

export SPEC_DECODE=true
export REQUIRE_K3_OVERLAY=1
export AMD_GPU_CLEAN_VRAM_MAX_PERCENT=2
unset K3_OVERLAY_PATCH_SHA256
unset K3_POST_OVERLAY_PATCH K3_POST_OVERLAY_PATCH_SHA256
unset K3_CPRR_RUNTIME_BUNDLE K3_CPRR_RUNTIME_MANIFEST_SHA256

case "${CONC:?CONC is required}:${DCP_SIZE:-1}:${KV_OFFLOADING:-none}" in
    1:1:none)
        export SPEC_NUM_TOKENS=6
        export SYNTHETIC_ACCEPT_LEN=3.84
        export K3_OVERLAY_PATCH="$script_dir/k3_patches/vllm_nightly_46638857_k3_c1_current.patch"
        export K3_OVERLAY_PATCH_SHA256=554ec6384b4ae143df42b223af66a8365e2b466c7ea691ed6c5a26a8749a4e6d
        export DCP_COMM_BACKEND=ag_rs
        export GPU_MEM_UTIL=0.875
        export MAX_NUM_BATCHED_TOKENS=8192
        export MAX_NUM_SEQS=2
        export MAX_CUDAGRAPH_CAPTURE_SIZE=16
        export CUDAGRAPH_CAPTURE_SIZES="$(seq -s, 1 16)"
        ;;
    16:8:none)
        export SPEC_NUM_TOKENS=3
        export SYNTHETIC_ACCEPT_LEN=3.00
        export K3_OVERLAY_PATCH="$script_dir/k3_patches/vllm_nightly_46638857_k3_c16_c52_current.patch"
        export K3_OVERLAY_PATCH_SHA256=90f975fad15722494366153ec3f32a14c4445bfa88c51ec53043b88eaf64dcc0
        export K3_POST_OVERLAY_PATCH="$script_dir/k3_patches/vllm_nightly_46638857_k3_compile_52190_delta.patch"
        export K3_POST_OVERLAY_PATCH_SHA256=de1ac272820122281f865c4f81d3f7a87e03c0cb42feb59390d9012b9bb88c00
        export K3_CPRR_RUNTIME_BUNDLE="$script_dir/k3_patches/aiter_pr4521_plus_4964_runtime"
        export K3_CPRR_RUNTIME_MANIFEST_SHA256=cb6f7ab6210d876e674f276cbaacf638936358cc12c1f89622084a611bb1d342
        export DCP_COMM_BACKEND=a2a
        # CPRR removes the dominant DCP8 verification cost. Keep only six
        # active sequences so the long-context C16 tail stays interactive;
        # the synthetic screen still clears 8.1K total tok/s/GPU.
        export GPU_MEM_UTIL=0.84
        export MAX_NUM_BATCHED_TOKENS=8192
        export MAX_NUM_SEQS=6
        export MAX_CUDAGRAPH_CAPTURE_SIZE=24
        export CUDAGRAPH_CAPTURE_SIZES=1,2,4,6,8,12,16,20,24
        ;;
    *)
        echo "Error: unsupported pinned Kimi-K3 MTP arm: C${CONC}, DCP=${DCP_SIZE:-1}, KV_OFFLOADING=${KV_OFFLOADING:-none}" >&2
        exit 1
        ;;
esac

export K3_AUTO_KV_PAGE=1
export ASYNC_SCHEDULING=0

export ATTENTION_BACKEND=ROCM_AITER_MLA
export ATTENTION_CONFIG_JSON='{"use_prefill_query_quantization":true}'
export STREAM_INTERVAL=10
export PREFIX_MATCH_UNIT=128
export PREFIX_CACHING_HASH_ALGO=sha256

export VLLM_ALLOW_DCP_FULL_CUDAGRAPH=1
export COMPILATION_CUSTOM_OPS='"+fused_rms_norm_gated","+quant_fp8","+grouped_topk","+sparse_attn_indexer","none"'

export HSA_NO_SCRATCH_RECLAIM=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900
export VLLM_ROCM_FORCE_SHARED_EXPERTS_STREAM=0

exec bash "$script_dir/kimik3_fp4_mi355x_mtp.sh" "$@"
