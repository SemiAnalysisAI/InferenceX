#!/usr/bin/env bash

# Source at the workflow boundary before master-config additional-settings.
source "$(dirname "${BASH_SOURCE[0]}")/../check_env.sh"
check_env_vars FRAMEWORK

case "$FRAMEWORK" in
    sglang-disagg)
        # Native workers inherit these engine/library settings through Slurm.
        check_env_vars IS_AGENTIC MODEL_PREFIX
        export ROCM_PATH=/opt/rocm UCX_HOME=/usr/local/ucx RIXL_HOME=/usr/local/rixl
        export MORI_IO_SQ_BACKOFF_TIMEOUT_US=50000 MORI_IO_QP_MAX_SEND_WR=16384
        export MORI_IO_QP_MAX_CQE=32768 MORI_IO_QP_MAX_SGE=2 MORI_IO_TC_DISABLE=0
        export UCX_IB_GID_INDEX=1 MORI_APP_LOG_LEVEL=WARNING SGLANG_ROUTER_STDOUT_LOGS=0
        export TORCH_NCCL_BLOCKING_WAIT=1 NCCL_BLOCKING_WAIT=1 SGLANG_OPT_USE_AITER_INDEXER=true
        if [[ "$IS_AGENTIC" == 1 || "$IS_AGENTIC" == true ]]; then
            if [[ "$MODEL_PREFIX" == dsv4 ]]; then
                export MORI_IO_SQ_BACKOFF_TIMEOUT_US=500000 MORI_IO_QP_MAX_SEND_WR=32768
            fi
        fi
        ;;
esac
