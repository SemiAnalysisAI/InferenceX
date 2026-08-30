#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"

# Reproduction of the accepted C1 baseline on nightly-46638857.
if [ "${SPEC_DECODING:-none}" != "mtp" ]; then
    echo "Error: this recipe requires DSpark speculative decoding." >&2
    exit 1
fi

if [ "${CONC:?CONC is required}" -ne 1 ] \
        || [ "${DCP_SIZE:-1}" -ne 1 ] \
        || [ "${KV_OFFLOADING:-none}" != "none" ]; then
    echo "Error: this pinned recipe supports only C1 TP8/DCP1 without offload." >&2
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
export SPEC_NUM_TOKENS=6
export SYNTHETIC_ACCEPT_LEN=3.84
export K3_OVERLAY_PATCH="$script_dir/k3_patches/vllm_nightly_46638857_k3_c1_current.patch"
export REQUIRE_K3_OVERLAY=1

export DCP_COMM_BACKEND=ag_rs
export GPU_MEM_UTIL=0.875
export MAX_NUM_BATCHED_TOKENS=8192
export MAX_NUM_SEQS=2
export K3_AUTO_KV_PAGE=1
export ASYNC_SCHEDULING=0

export ATTENTION_BACKEND=ROCM_AITER_MLA
export ATTENTION_CONFIG_JSON='{"use_prefill_query_quantization":true}'
export STREAM_INTERVAL=10
export PREFIX_MATCH_UNIT=128
export PREFIX_CACHING_HASH_ALGO=sha256

export VLLM_ALLOW_DCP_FULL_CUDAGRAPH=1
export MAX_CUDAGRAPH_CAPTURE_SIZE=16
export CUDAGRAPH_CAPTURE_SIZES="$(seq -s, 1 16)"
export COMPILATION_CUSTOM_OPS='"+fused_rms_norm_gated","+quant_fp8","+grouped_topk","+sparse_attn_indexer","none"'

export HSA_NO_SCRATCH_RECLAIM=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900
export VLLM_ROCM_FORCE_SHARED_EXPERTS_STREAM=0

exec bash "$script_dir/kimik3_fp4_mi355x_mtp.sh" "$@"
