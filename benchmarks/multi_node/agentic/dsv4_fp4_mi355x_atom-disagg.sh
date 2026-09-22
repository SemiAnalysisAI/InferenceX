#!/usr/bin/env bash
# 1P1D, eight GPUs per role, DPA AgentX baseline at C128/C192/C256.
# Use the shared AMD submit.sh -> job.slurm -> server_atom.sh chain,
# following the GLM-5.3 P/D recipe's CI integration.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../benchmark_lib.sh"

check_env_vars \
    CONC_LIST ISL OSL IMAGE SPEC_DECODING MODEL_PATH MODEL_NAME \
    PREFILL_NUM_WORKERS PREFILL_TP PREFILL_EP PREFILL_DP_ATTN PREFILL_NODES \
    DECODE_NUM_WORKERS DECODE_TP DECODE_EP DECODE_DP_ATTN DECODE_NODES \
    DECODE_MTP_SIZE RANDOM_RANGE_RATIO DURATION KV_OFFLOADING IS_AGENTIC \
    FRAMEWORK GITHUB_WORKSPACE MODEL_PREFIX PRECISION RESULT_FILENAME

if [[ "$KV_OFFLOADING" != "none" ]]; then
    echo "ERROR: V4 DPA baseline requires KV_OFFLOADING=none" >&2
    exit 1
fi
if [[ "$IS_AGENTIC" != 1 && "$IS_AGENTIC" != true ]]; then
    echo "ERROR: this V4 DPA baseline requires an AgentX workload" >&2
    exit 1
fi
if [[ "$PREFILL_NODES" != 1 || "$DECODE_NODES" != 1 ||
      "$PREFILL_NUM_WORKERS" != 1 || "$DECODE_NUM_WORKERS" != 1 ||
      "$PREFILL_TP" != 8 || "$DECODE_TP" != 8 ||
      "$PREFILL_EP" != 1 || "$DECODE_EP" != 1 ||
      "$PREFILL_DP_ATTN" != true || "$DECODE_DP_ATTN" != true ]]; then
    echo "ERROR: expected 1P1D with TP8, EP1 and DPA enabled on both roles" >&2
    exit 1
fi
for concurrency in $CONC_LIST; do
    case "$concurrency" in
        128|192|256) ;;
        *) echo "ERROR: unsupported DPA concurrency: $concurrency" >&2; exit 1 ;;
    esac
done
if [[ "$SPEC_DECODING" != mtp || "$DECODE_MTP_SIZE" != 3 ]]; then
    echo "ERROR: this V4 DPA baseline requires the DSpark K3 configuration" >&2
    exit 1
fi

export TIME_LIMIT=08:00:00
export CONTAINER_IMAGE="$IMAGE"
export MAX_MODEL_LEN=1000000
export PREFILL_ENABLE_EP=false DECODE_ENABLE_EP=false
export PREFILL_ENABLE_DP=true DECODE_ENABLE_DP=true
export CLEAR_CACHE_BETWEEN_CONC=0

cd "$GITHUB_WORKSPACE/benchmarks/multi_node/amd_utils"
JOB_ID=$(bash ./submit.sh \
    "$PREFILL_NODES" "$PREFILL_NUM_WORKERS" \
    "$DECODE_NODES" "$DECODE_NUM_WORKERS" \
    "$ISL" "$OSL" "${CONC_LIST// /x}" inf \
    "$PREFILL_ENABLE_EP" "$PREFILL_ENABLE_DP" \
    "$DECODE_ENABLE_EP" "$DECODE_ENABLE_DP" \
    "$PREFILL_TP" "$DECODE_TP" "$RANDOM_RANGE_RATIO")
if [[ ! "$JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: submit.sh did not return a numeric Slurm job id: $JOB_ID" >&2
    exit 1
fi
echo "$JOB_ID"
