#!/usr/bin/env bash
#
# Multi-node ATOM disaggregated benchmark entry script for
# .github/configs/amd-master.yaml :: dsv4-fp4-mi355x-atom-disagg.
#
# Dispatched by runners/launch_mi355x-amds.sh (IS_MULTINODE=true branch)
# when FRAMEWORK==atom-disagg. Structurally mirrors the sglang-disagg
# entry script (benchmarks/multi_node/dsr1_fp4_mi355x_sglang-disagg.sh)
# but delegates SLURM submission to atom_disagg_utils/submit.sh, which
# launches the ATOM equivalent of SGLang's --disaggregation-mode setup
# defined in benchmarks/multi_node/atom_disagg_utils/server.sh.

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    CONC_LIST \
    ISL \
    OSL \
    IMAGE \
    SPEC_DECODING \
    MODEL_PATH \
    PREFILL_NUM_WORKERS \
    PREFILL_TP \
    PREFILL_EP \
    PREFILL_DP_ATTN \
    DECODE_NUM_WORKERS \
    DECODE_TP \
    DECODE_EP \
    DECODE_DP_ATTN \
    PREFILL_NODES \
    DECODE_NODES \
    RANDOM_RANGE_RATIO

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

set -x

cd "$GITHUB_WORKSPACE/benchmarks/multi_node/atom_disagg_utils" || exit 1

# Forwarded to job.slurm / server.sh
export TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
export MODEL_PATH=$MODEL_PATH
export MODEL_NAME=$MODEL_NAME
export CONTAINER_IMAGE=$IMAGE

# Translate matrix EP/DP-ATTN flags into the boolean strings the SLURM job
# expects (same convention as amd_utils/submit.sh + server.sh).
if [[ "${PREFILL_EP:-1}" -eq 1 ]]; then
    export PREFILL_ENABLE_EP=false
else
    export PREFILL_ENABLE_EP=true
fi

if [[ "$PREFILL_DP_ATTN" == "true" ]]; then
    export PREFILL_ENABLE_DP=true
else
    export PREFILL_ENABLE_DP=false
fi

if [[ "${DECODE_EP:-1}" -eq 1 ]]; then
    export DECODE_ENABLE_EP=false
else
    export DECODE_ENABLE_EP=true
fi

if [[ "$DECODE_DP_ATTN" == "true" ]]; then
    export DECODE_ENABLE_DP=true
else
    export DECODE_ENABLE_DP=false
fi

# Launch jobs based on ISL/OSL. Replace ' ' in CONC_LIST with 'x' such that
# the concurrency list is represented by a list of numbers delimited by 'x'.
# This is because of how the underlying launch script expects the concurrencies.
# The 16th positional arg is NODELIST (comma-separated hostnames); empty =
# let SLURM auto-pick.
JOB_ID=$(bash ./submit.sh "$PREFILL_NODES" \
    "$PREFILL_NUM_WORKERS" \
    "$DECODE_NODES" \
    "$DECODE_NUM_WORKERS" \
    "$ISL" "$OSL" "${CONC_LIST// /x}" inf \
    "${PREFILL_ENABLE_EP}" "${PREFILL_ENABLE_DP}" \
    "${DECODE_ENABLE_EP}" "${DECODE_ENABLE_DP}" \
    "${PREFILL_TP}" "${DECODE_TP}" \
    "${RANDOM_RANGE_RATIO}" \
    "${NODELIST:-}")

if [[ $? -ne 0 ]]; then
    echo "Failed to submit job" >&2
    exit 1
fi

echo "$JOB_ID"
