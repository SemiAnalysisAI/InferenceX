#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"

# DSpark counterpart of the tuned nightly-46638857 no-spec path. Low-concurrency
# GPU-resident TP8-only cells use K=6; DCP8/SimpleCPU cells use K=3.
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
case "${DCP_SIZE:-1}:${KV_OFFLOADING:-none}:${KV_OFFLOAD_BACKEND:-}" in
    1:none:)
        export SPEC_NUM_TOKENS=6
        ;;
    8:dram:vllm-simple)
        export SPEC_NUM_TOKENS=3
        export DCP_COMM_BACKEND=a2a
        export SIMPLE_LAZY_OFFLOAD=true
        export SIMPLE_LAZY_OFFLOAD_WATERMARK_RATIO=1.0
        ;;
    *)
        echo "Error: unsupported Kimi-K3 DSpark topology: DCP_SIZE=${DCP_SIZE:-1}, KV_OFFLOADING=${KV_OFFLOADING:-none}, KV_OFFLOAD_BACKEND=${KV_OFFLOAD_BACKEND:-}" >&2
        exit 1
        ;;
esac
export K3_OVERLAY_PATCH="$script_dir/k3_patches/vllm_nightly_46638857_k3_tuned.patch"
export REQUIRE_K3_OVERLAY=1

if [ "${DCP_SIZE:-1}" -eq 1 ] && [ "${KV_OFFLOADING:-none}" = "none" ]; then
    default_max_num_seqs=8
    default_cudagraph_capture_size=64
    default_cudagraph_capture_sizes="$(seq -s, 1 64)"
else
    default_max_num_seqs=24
    default_cudagraph_capture_size=128
    default_cudagraph_capture_sizes="$(seq -s, 1 128)"
fi

export GPU_MEM_UTIL=0.90
export MAX_NUM_BATCHED_TOKENS=16384
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-$default_max_num_seqs}"
export K3_AUTO_KV_PAGE=1
export ASYNC_SCHEDULING=1

export ATTENTION_BACKEND=ROCM_AITER_MLA
export ATTENTION_CONFIG_JSON='{"use_prefill_query_quantization":true}'
export STREAM_INTERVAL=10
export PREFIX_MATCH_UNIT=128
export PREFIX_CACHING_HASH_ALGO=sha256

export VLLM_ALLOW_DCP_FULL_CUDAGRAPH=1
export MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-$default_cudagraph_capture_size}"
export CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-$default_cudagraph_capture_sizes}"
export COMPILATION_CUSTOM_OPS='"+fused_rms_norm_gated","+quant_fp8","+grouped_topk","+sparse_attn_indexer","none"'

export HSA_NO_SCRATCH_RECLAIM=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900

exec bash "$script_dir/kimik3_fp4_mi355x_mtp.sh" "$@"
