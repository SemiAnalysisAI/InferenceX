#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/../../benchmark_lib.sh"

check_env_vars CONC_LIST IMAGE MODEL_PATH PREFILL_TP PREFILL_PP_SIZE \
    PREFILL_DCP_SIZE PREFILL_NUM_WORKERS RANDOM_RANGE_RATIO DURATION FRAMEWORK

cd "$GITHUB_WORKSPACE/benchmarks/multi_node/amd_utils"

# Reuse the mature AMD two-node allocator and artifact fan-in.  The synthetic
# P/D geometry only communicates a two-node allocation; server_native_pp.sh
# starts one aggregated engine and does not create P/D workers or a router.
export K3_NATIVE_PP=1
export CONTAINER_IMAGE="$IMAGE"
export MODEL_NAME="${MODEL##*/}"
export MODEL_PREFIX="${MODEL_PREFIX:-kimik3}"
export PRECISION="${PRECISION:-fp4}"
export RESULT_FILENAME="${RESULT_FILENAME:-${RUNNER_NAME:-kimik3-native-pp-c56}}"
export IS_AGENTIC="${IS_AGENTIC:-1}"
export KV_OFFLOADING="${KV_OFFLOADING:-dram}"
export ENABLE_METRICS="${ENABLE_METRICS:-1}"
# Vultri's compute-node /tmp is not visible from the controller.  Bind the
# shared CI archive at the same absolute path so native PP startup logs can be
# inspected live without entering the allocation.
export NATIVE_PP_SHARED_LOG_ROOT="${NATIVE_PP_SHARED_LOG_ROOT:-/shared/data/R7N/InferenceX_CI/live}"
export EXTRA_DOCKER_MOUNTS="${EXTRA_DOCKER_MOUNTS:-} -v /shared/data/R7N/InferenceX_CI:/shared/data/R7N/InferenceX_CI"
export PREFILL_ENABLE_EP=false PREFILL_ENABLE_DP=false
export DECODE_ENABLE_EP=false DECODE_ENABLE_DP=false
export PREFILL_NODES=1 DECODE_NODES=1
export PREFILL_NUM_WORKERS=1 DECODE_NUM_WORKERS=1
export PREFILL_TP=4 DECODE_TP=4
export PREFILL_EP=1 DECODE_EP=1
export PREFILL_DP_ATTN=false DECODE_DP_ATTN=false
export PREFILL_PP_SIZE=2 DECODE_PP_SIZE=2
export PREFILL_DCP_SIZE=4 DECODE_DCP_SIZE=4
export PREFILL_PCP_SIZE=1 DECODE_PCP_SIZE=1
export DECODE_MTP_SIZE=4

job_id=$(bash ./submit.sh 1 1 1 1 0 0 "${CONC_LIST// /x}" inf \
    false false false false 8 8 "$RANDOM_RANGE_RATIO")
printf '%s\n' "$job_id"
