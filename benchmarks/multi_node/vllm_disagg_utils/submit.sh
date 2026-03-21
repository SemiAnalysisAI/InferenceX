#!/bin/bash
#
# Cluster Configuration Template for Multi-Node vLLM Disaggregated Serving
#
# This script submits a multi-node vLLM disaggregated benchmark job to SLURM.
# It must be configured for your specific cluster before use.
#
# Router is co-located with the first prefill node (same as SGLang), so
# NUM_NODES = PREFILL_NODES + DECODE_NODES.

usage() {
    cat << 'USAGE'
Usage:
  bash submit.sh <PREFILL_NODES> <PREFILL_WORKERS> <DECODE_NODES> <DECODE_WORKERS> \
                 <ISL> <OSL> <CONCURRENCIES> <REQUEST_RATE> [NODE_LIST] [RANDOM_RANGE_RATIO]

Arguments:
  PREFILL_NODES       Number of prefill nodes
  PREFILL_WORKERS     Number of prefill workers (usually 1)
  DECODE_NODES        Number of decode nodes
  DECODE_WORKERS      Number of decode workers (usually 1)
  ISL                 Input sequence length
  OSL                 Output sequence length
  CONCURRENCIES       Concurrency levels, delimited by 'x' (e.g., "8x16x32")
  REQUEST_RATE        Request rate ("inf" for max throughput)
  NODE_LIST           Optional: comma-separated hostnames
  RANDOM_RANGE_RATIO  Optional: random range ratio for benchmark (default 0.8)

Required environment variables:
  SLURM_ACCOUNT    SLURM account name
  SLURM_PARTITION  SLURM partition
  TIME_LIMIT       Job time limit (e.g., "08:00:00")
  MODEL_PATH       Path to model directory (e.g., /nfsdata)
  MODEL_NAME       Model name directory
  CONTAINER_IMAGE  Docker image name (e.g., vllm_disagg_pd:latest)
  RUNNER_NAME      Runner identifier (for job name)
USAGE
}

check_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "Error: ${name} not specified" >&2
        usage >&2
        exit 1
    fi
}

check_env SLURM_ACCOUNT
check_env SLURM_PARTITION
check_env TIME_LIMIT

check_env MODEL_PATH
check_env MODEL_NAME
check_env CONTAINER_IMAGE
check_env RUNNER_NAME

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

# COMMAND_LINE ARGS
PREFILL_NODES=$1
PREFILL_WORKERS=${2:-1}
DECODE_NODES=$3
DECODE_WORKERS=${4:-1}
ISL=$5
OSL=$6
CONCURRENCIES=$7
REQUEST_RATE=$8
NODE_LIST=${9}
RANDOM_RANGE_RATIO=${10}

# Router co-located with first prefill: xP + yD nodes total
NUM_NODES=$((PREFILL_NODES + DECODE_NODES))
profiler_args="${ISL} ${OSL} ${CONCURRENCIES} ${REQUEST_RATE}"

# Export variables for the SLURM job
export MODEL_DIR=$MODEL_PATH
export DOCKER_IMAGE_NAME=$CONTAINER_IMAGE
export PROFILER_ARGS=$profiler_args

# For vLLM, each worker = 1 node (TP=8 per node).
# xP/yD must match the node counts so NUM_NODES = xP+yD is correct.
export xP=$PREFILL_NODES
export yD=$DECODE_NODES
export NUM_NODES=$NUM_NODES
export GPUS_PER_NODE=$GPUS_PER_NODE
export MODEL_NAME=$MODEL_NAME
export BENCH_INPUT_LEN=${ISL}
export BENCH_OUTPUT_LEN=${OSL}
export BENCH_NUM_PROMPTS_MULTIPLIER=${BENCH_NUM_PROMPTS_MULTIPLIER:-10}
export BENCH_MAX_CONCURRENCY=${CONCURRENCIES}
export BENCH_REQUEST_RATE=${REQUEST_RATE}
export BENCH_RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.8}

export PROXY_STREAM_IDLE_TIMEOUT=${PROXY_STREAM_IDLE_TIMEOUT:-300}
export VLLM_MORIIO_CONNECTOR_READ_MODE=${VLLM_MORIIO_CONNECTOR_READ_MODE:-0}

# Log directory: must be on NFS (shared filesystem) so the submit host can read SLURM output.
export BENCHMARK_LOGS_DIR="${BENCHMARK_LOGS_DIR:-$(pwd)/benchmark_logs}"
mkdir -p "$BENCHMARK_LOGS_DIR"

# Optional: pass an explicit node list to sbatch.
NODELIST_OPT=()
if [[ -n "${NODE_LIST//[[:space:]]/}" ]]; then
    IFS=',' read -r -a NODE_ARR <<< "$NODE_LIST"
    if [[ "${#NODE_ARR[@]}" -ne "$NUM_NODES" ]]; then
        echo "Error: NODE_LIST has ${#NODE_ARR[@]} nodes but NUM_NODES=${NUM_NODES}" >&2
        echo "Error: NODE_LIST='${NODE_LIST}'" >&2
        exit 1
    fi
    NODELIST_CSV="$(IFS=,; echo "${NODE_ARR[*]}")"
    NODELIST_OPT=(--nodelist "$NODELIST_CSV")
fi

# Optional: exclude specific nodes (e.g. nodes with broken Docker sockets).
# Set SLURM_EXCLUDE_NODES env var to a comma-separated list of hostnames.
EXCLUDE_OPT=()
if [[ -n "${SLURM_EXCLUDE_NODES:-}" ]]; then
    EXCLUDE_OPT=(--exclude "$SLURM_EXCLUDE_NODES")
fi

# Construct the sbatch command
sbatch_cmd=(
    sbatch
    --parsable
    -N "$NUM_NODES"
    -n "$NUM_NODES"
    "${NODELIST_OPT[@]}"
    "${EXCLUDE_OPT[@]}"
    --time "$TIME_LIMIT"
    --partition "$SLURM_PARTITION"
    --account "$SLURM_ACCOUNT"
    --job-name "$RUNNER_NAME"
    --output "${BENCHMARK_LOGS_DIR}/slurm_job-%j.out"
    --error "${BENCHMARK_LOGS_DIR}/slurm_job-%j.err"
    "$(dirname "$0")/job.slurm"
)

JOB_ID=$("${sbatch_cmd[@]}")
if [[ $? -ne 0 ]]; then
    echo "Error: Failed to submit job with sbatch" >&2
    exit 1
fi
echo "$JOB_ID"
