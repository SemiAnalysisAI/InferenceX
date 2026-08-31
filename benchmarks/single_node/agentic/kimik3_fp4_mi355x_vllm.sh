#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"

# Reproduction of the accepted C16 and C52 baselines on nightly-46638857.
if [ "${SPEC_DECODING:-none}" != "none" ]; then
    echo "Error: this recipe is the no-spec Kimi-K3 configuration." >&2
    exit 1
fi

if [ "${DCP_SIZE:-1}" -ne 8 ]; then
    echo "Error: this pinned recipe requires TP8/DCP8." >&2
    exit 1
fi

if [ -r /proc/sys/kernel/numa_balancing ]; then
    read -r numa_balancing </proc/sys/kernel/numa_balancing
    if [ "$numa_balancing" != "0" ]; then
        echo "Error: Kimi-K3 tuning requires kernel.numa_balancing=0." >&2
        exit 1
    fi
fi

export SPEC_DECODE=false
export AMD_GPU_CLEAN_VRAM_MAX_PERCENT=2
export K3_OVERLAY_PATCH="$script_dir/k3_patches/vllm_nightly_46638857_k3_c16_c52_current.patch"
export K3_OVERLAY_PATCH_SHA256=90f975fad15722494366153ec3f32a14c4445bfa88c51ec53043b88eaf64dcc0
export REQUIRE_K3_OVERLAY=1
unset K3_POST_OVERLAY_PATCH K3_POST_OVERLAY_PATCH_SHA256

export DCP_COMM_BACKEND=a2a
case "${CONC:?CONC is required}:${KV_OFFLOADING:-none}:${KV_OFFLOAD_BACKEND:-}" in
    16:none:)
        export K3_POST_OVERLAY_PATCH="$script_dir/k3_patches/vllm_nightly_46638857_k3_compile_52190_delta.patch"
        export K3_POST_OVERLAY_PATCH_SHA256=de1ac272820122281f865c4f81d3f7a87e03c0cb42feb59390d9012b9bb88c00
        export GPU_MEM_UTIL=0.86
        export MAX_NUM_BATCHED_TOKENS=8192
        export ASYNC_SCHEDULING=0
        export MAX_CUDAGRAPH_CAPTURE_SIZE=80
        export CUDAGRAPH_CAPTURE_SIZES="$(seq -s, 1 80)"
        ;;
    52:dram:vllm-simple)
        export GPU_MEM_UTIL=0.88
        export MAX_NUM_BATCHED_TOKENS=16384
        export ASYNC_SCHEDULING=1
        export MAX_CUDAGRAPH_CAPTURE_SIZE=4096
        export CUDAGRAPH_CAPTURE_SIZES="$(seq -s, 1 80),128,256,512,1024,2048,4096"
        ;;
    *)
        echo "Error: unsupported pinned Kimi-K3 baseline arm: C${CONC}, KV_OFFLOADING=${KV_OFFLOADING:-none}, KV_OFFLOAD_BACKEND=${KV_OFFLOAD_BACKEND:-}" >&2
        exit 1
        ;;
esac
export MAX_NUM_SEQS=80
export K3_AUTO_KV_PAGE=1
export SIMPLE_LAZY_OFFLOAD=true
export SIMPLE_LAZY_OFFLOAD_WATERMARK_RATIO=1.0

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
