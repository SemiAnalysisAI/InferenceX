#!/usr/bin/env bash

# Source at the workflow boundary before master-config additional-settings.
# Receivers validate these inputs; recipe overrides are applied afterward.
source "$(dirname "${BASH_SOURCE[0]}")/../benchmark_lib.sh" --validation-only
check_env_vars FRAMEWORK IS_AGENTIC MODEL_PREFIX

export BENCH_NUM_PROMPTS_MULTIPLIER=10 DRY_RUN=0 KEEP_CONTAINERS=0
export AIPERF_DRAIN_TIMEOUT_SECONDS=1800 AIPERF_DRAIN_POLL_SECONDS=10

case "$FRAMEWORK" in
    sglang-disagg|vllm-disagg|atom-disagg)
        export VLLM_ROUTER_IMAGE=vllm/vllm-router:nightly-20260716-1fbcde7 SKIP_RDMA_CHECK=0 SKIP_GPU_SANITY=0
        export ROUTER_TYPE=vllm-router ROUTER_PORT=30000 PROXY_PING_PORT=36367
        export DECODE_MTP_SIZE=0
        export HEADNODE_PORT=20000 SERVER_PORT=2584 PROXY_STREAM_IDLE_TIMEOUT=300
        export ENABLE_METRICS=0 PREFILL_ROUTER_POLICY=random DECODE_ROUTER_POLICY=random
        export FLUSH_DRAIN_TIMEOUT=120 CLEAR_CACHE_BETWEEN_CONC=1
        export ROCM_PATH=/opt/rocm UCX_HOME=/usr/local/ucx RIXL_HOME=/usr/local/rixl
        export MORI_IO_SQ_BACKOFF_TIMEOUT_US=50000 MORI_IO_QP_MAX_SEND_WR=16384
        export MORI_IO_QP_MAX_CQE=32768 MORI_IO_QP_MAX_SGE=2 MORI_IO_TC_DISABLE=0
        export UCX_IB_GID_INDEX=1 MORI_APP_LOG_LEVEL=WARNING SGLANG_ROUTER_STDOUT_LOGS=0
        export TORCH_NCCL_BLOCKING_WAIT=1 NCCL_BLOCKING_WAIT=1 SGLANG_OPT_USE_AITER_INDEXER=true
        export HICACHE_HOST_POOL_COUNT=1 HICACHE_PAGE_SIZE=1 HICACHE_PREFETCH_POLICY=wait_complete
        export HICACHE_L2_MEM_LAYOUT=page_first_direct HICACHE_L3_MEM_LAYOUT=page_first
        export HICACHE_IO_BACKEND=direct HICACHE_WRITE_POLICY=write_through HICACHE_RATIO=5
        export FORCE_HICACHE_RATIO=0 MC_MASTER_PORT=50061 MC_METADATA_PORT=8080 MC_METRICS_PORT=9003
        export MC_MASTER_THREADS=64 MC_EVICTION_HIGH_WATERMARK=0.95 MC_PROTOCOL=tcp MC_GLOBAL_SEG=64gb
        export ROUTER_CACHE_THRESHOLD=0.3 ROUTER_BALANCE_ABS_THRESHOLD=2 ROUTER_BALANCE_REL_THRESHOLD=1.1
        export ROUTER_CANARY_TIMEOUT=600 ROUTER_CANARY_REQ_TIMEOUT=120 ROUTER_READINESS_CANARY=1
        export ROUTER_CB_ARGS='--cb-timeout-duration-secs 15 --retry-max-retries 3'
        export ROUTER_DEFAULT_POLICY_FLAGS='--policy random --prefill-policy random --decode-policy random'
        if [[ "$IS_AGENTIC" == 1 || "$IS_AGENTIC" == true ]]; then
            export PREFILL_ROUTER_POLICY=consistent_hashing ENABLE_METRICS=1
            export ROUTER_RESILIENCE_FLAGS='--disable-circuit-breaker --health-failure-threshold 100 --health-check-timeout-secs 600 --health-check-interval-secs 30'
            if [[ "$MODEL_PREFIX" == dsv4 ]]; then
                export TIME_LIMIT=08:00:00 MAX_MODEL_LEN=1000000 DISABLE_CUSTOM_ALL_REDUCE=0
                export HICACHE_TIER=L2 HICACHE_PAGE_SIZE=256 HICACHE_RATIO=3 HICACHE_MEM_LAYOUT=page_first
                export HICACHE_PREFETCH_POLICY=best_effort FORCE_HICACHE_RATIO=1
                export MC_MASTER_PORT=58137 MC_METRICS_PORT=19003 
                export MORI_IO_SQ_BACKOFF_TIMEOUT_US=500000 MORI_IO_QP_MAX_SEND_WR=32768
            fi
        fi
        if [[ "$FRAMEWORK" == atom-disagg ]]; then
            export PREFILL_PORT=8010 DECODE_PORT=8020 HANDSHAKE_PORT=6301
            export MEM_FRAC_STATIC=0.85 KV_CACHE_DTYPE=fp8 BLOCK_SIZE=16 MAX_NUM_SEQS=256
            export WAIT_SERVER_TIMEOUT=2500 WAIT_LOCAL_ROUTER_TIMEOUT=300 WAIT_REMOTE_ROUTER_TIMEOUT=2800
        fi
        ;;
    tilert)
        check_env_vars GITHUB_WORKSPACE
        export BENCHMARK_LOGS_DIR="$GITHUB_WORKSPACE" RESULT_DIR=/workspace
        export GPU_MEM_UTIL=0.75 DECODE_CTRL_PORT=5556 DECODE_HTTP_PORT=5557 PREFILL_PORT=8000
        export DECODE_WAIT=3600 PREFILL_WAIT=3600 TILERT_QUEUE_TIMEOUT=0
        export TILERT_RDMA_STRICT=0 TILERT_CONVERT_LOCK_WAIT=21600 TILERT_DECODE_DRAIN=60
        export TILERT_VERSION=0.1.5.post3 TILERT_HTTP_DEPS='fastapi uvicorn httpx' TILERT_NIXL_VERSION=1.3.1
        export B200_SQUASH_DIR=/home/sa-shared/containers
        if [[ "$IS_AGENTIC" == 1 || "$IS_AGENTIC" == true ]]; then
            export TILERT_QUEUE_TIMEOUT=1800
        fi
        ;;
    llmd-vllm)
        export LLMD_CONTAINER_ENGINE=docker VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1
        export VLLM_ENGINE_READY_TIMEOUT_S=1800 VLLM_LOGGING_LEVEL=INFO UCX_TLS=cuda_copy,cuda_ipc,rc
        export NVSHMEM_REMOTE_TRANSPORT=ibgda NVSHMEM_IB_ENABLE_IBGDA=true NVSHMEM_SYMMETRIC_SIZE=16G
        export LLMD_API_SERVER_COUNT=4
        ;;
esac
