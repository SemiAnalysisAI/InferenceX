#!/usr/bin/env bash
set -eo pipefail
source benchmarks/benchmark_lib.sh --validation-only
check_env_vars INFERENCEX_EXPERIMENT IS_AGENTIC MODEL_PREFIX PRECISION FRAMEWORK TP DURATION
if [[ "$INFERENCEX_EXPERIMENT" != agentx-offload || "$IS_AGENTIC" != 1 || "$MODEL_PREFIX" != minimaxm3 || "$PRECISION" != fp4 || "$FRAMEWORK" != vllm || "$TP" != 4 || "$DURATION" != 3600 ]]; then
    echo 'The offload experiment requires canonical MiniMax-M3 NVFP4 TP4 B200 AgentX.' >&2
    exit 1
fi
exec bash benchmarks/single_node/agentic/minimaxm3_fp4_b200_mtp.sh
