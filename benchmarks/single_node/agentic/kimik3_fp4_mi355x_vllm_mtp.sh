#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"

# DSpark K=3 counterpart of the tuned nightly-46638857 no-spec path. Keep the
# target, cache, graph, and offload settings aligned with the validated c40
# configuration so speculative decoding is the deliberate low-concurrency
# delta.
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
export SPEC_NUM_TOKENS=3
export K3_OVERLAY_PATCH="$script_dir/k3_patches/vllm_nightly_46638857_k3_tuned.patch"
export REQUIRE_K3_OVERLAY=1

export DCP_COMM_BACKEND=a2a
export GPU_MEM_UTIL=0.90
export MAX_NUM_BATCHED_TOKENS=16384
export MAX_NUM_SEQS=80
export K3_AUTO_KV_PAGE=1
export SIMPLE_LAZY_OFFLOAD=true
export SIMPLE_LAZY_OFFLOAD_WATERMARK_RATIO=1.0
export ASYNC_SCHEDULING=1

export ATTENTION_BACKEND=ROCM_AITER_MLA
export ATTENTION_CONFIG_JSON='{"use_prefill_query_quantization":true}'
export STREAM_INTERVAL=10
export PREFIX_MATCH_UNIT=128
export PREFIX_CACHING_HASH_ALGO=sha256

export VLLM_ALLOW_DCP_FULL_CUDAGRAPH=1
export MAX_CUDAGRAPH_CAPTURE_SIZE=4096
export CUDAGRAPH_CAPTURE_SIZES="$(seq -s, 1 80),128,256,512,1024,2048,4096"
export COMPILATION_CUSTOM_OPS='"+fused_rms_norm_gated","+quant_fp8","+grouped_topk","+sparse_attn_indexer","none"'

export HSA_NO_SCRATCH_RECLAIM=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900

exec bash "$script_dir/kimik3_fp4_mi355x_mtp.sh" "$@"
