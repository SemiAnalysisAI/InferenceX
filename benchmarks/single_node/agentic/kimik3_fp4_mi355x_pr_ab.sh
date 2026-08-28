#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"

if [ -z "${K3_OVERLAY_PATCH:-}" ]; then
    echo "Error: K3_OVERLAY_PATCH must select the PR stack under test." >&2
    exit 1
fi

# Both comparison arms need the upstream stale partial-prefix fix discovered
# while exercising C16. Keep it separate so the original four-vs-five-PR
# performance delta remains explicit in K3_OVERLAY_PATCH.
export K3_POST_OVERLAY_PATCH="${K3_POST_OVERLAY_PATCH:-$script_dir/k3_patches/vllm_pr52972_stale_partial_hash.patch}"
export REQUIRE_K3_POST_OVERLAY=1

case "${SPEC_DECODING:-none}:${CONC:?CONC is required}:${DCP_SIZE:-1}:${KV_OFFLOADING:-none}:${KV_OFFLOAD_BACKEND:-}" in
    mtp:1:1:none:)
        export SPEC_DECODE=true
        export SPEC_NUM_TOKENS=6
        export GPU_MEM_UTIL=0.90
        export MAX_NUM_BATCHED_TOKENS=8192
        export MAX_NUM_SEQS=20
        capture_size=60
        ;;
    mtp:16:8:none:)
        export SPEC_DECODE=true
        export SPEC_NUM_TOKENS=3
        export GPU_MEM_UTIL=0.86
        export MAX_NUM_BATCHED_TOKENS=8192
        export MAX_NUM_SEQS=32
        capture_size=64
        ;;
    none:52:8:none:)
        export SPEC_DECODE=false
        export GPU_MEM_UTIL=0.88
        export MAX_NUM_BATCHED_TOKENS=4096
        export MAX_NUM_SEQS=72
        capture_size=64
        ;;
    none:52:8:dram:vllm-simple)
        export SPEC_DECODE=false
        export GPU_MEM_UTIL=0.88
        export MAX_NUM_BATCHED_TOKENS=16384
        export MAX_NUM_SEQS=80
        export SIMPLE_LAZY_OFFLOAD=true
        export SIMPLE_LAZY_OFFLOAD_WATERMARK_RATIO=1.0
        capture_size=80
        ;;
    *)
        echo "Error: unsupported Kimi-K3 PR A/B cell: spec=${SPEC_DECODING:-none}, conc=${CONC:-}, dcp=${DCP_SIZE:-1}, offload=${KV_OFFLOADING:-none}, backend=${KV_OFFLOAD_BACKEND:-}" >&2
        exit 1
        ;;
esac

export REQUIRE_K3_OVERLAY=1
export DCP_COMM_BACKEND=a2a
export K3_AUTO_KV_PAGE=1
export ASYNC_SCHEDULING=1

export ATTENTION_BACKEND=ROCM_AITER_MLA
export ATTENTION_CONFIG_JSON='{"use_prefill_query_quantization":true}'
export STREAM_INTERVAL=10
export PREFIX_MATCH_UNIT=128
export PREFIX_CACHING_HASH_ALGO=sha256

export VLLM_ALLOW_DCP_FULL_CUDAGRAPH=1
export MAX_CUDAGRAPH_CAPTURE_SIZE="$capture_size"
export CUDAGRAPH_CAPTURE_SIZES="$(seq -s, 1 "$capture_size")"
export COMPILATION_CUSTOM_OPS='"+fused_rms_norm_gated","+quant_fp8","+grouped_topk","+sparse_attn_indexer","none"'

export HSA_NO_SCRATCH_RECLAIM=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900

exec bash "$script_dir/kimik3_fp4_mi355x_pr_ab_runner.sh" "$@"
