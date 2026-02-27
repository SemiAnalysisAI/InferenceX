#!/bin/bash
#
# Cluster Configuration Template for Multi-Node Disaggregated Serving
#
# This script submits a multi-node SGLang disaggregated benchmark job to SLURM.
# It must be configured for your specific cluster before use.

usage() {
    cat << 'USAGE'
This script aims to provide a one-liner call to the submit_job_script.py,
so that the deployment process can be further simplified.

To use this script, fill in the following script and run it under your `slurm_jobs` directory:
======== begin script area ========
# REQUIRED: Cluster-specific configuration
export SLURM_ACCOUNT=              # Your SLURM account name
export SLURM_PARTITION=            # SLURM partition to submit to
export TIME_LIMIT=                 # Job time limit (e.g., "08:00:00")

# REQUIRED: Model and container paths
export MODEL_PATH=                 # Path to model directory (e.g., /mnt/models, /nfsdata)
export CONTAINER_IMAGE=            # Path to container squash file

# REQUIRED: Hardware configuration
export GPUS_PER_NODE=              # GPUs per node (e.g., 8 for MI355X, 4 for MI325X)

# OPTIONAL: RDMA/Network configuration (set in runners/launch_mi355x-amds.sh for AMD)
# export IBDEVICES=                # RDMA device names (e.g., ionic_0,ionic_1,... or mlx5_0,mlx5_1,...)
# export MORI_RDMA_TC=             # RDMA traffic class (e.g., 96, 104)

bash submit.sh \
$PREFILL_NODES $PREFILL_WORKERS $DECODE_NODES $DECODE_WORKERS \
$ADDITIONAL_FRONTENDS \
$ISL $OSL $CONCURRENCIES $REQUEST_RATE
======== end script area ========
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

# GPUS_PER_NODE defaults to 8 (MI355X). Set to 4 for MI325X if needed.
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
PREFILL_ENABLE_EP=${9:-1}
PREFILL_ENABLE_DP=${10:-1}
DECODE_ENABLE_EP=${11:-1}
DECODE_ENABLE_DP=${12:-1}
RANDOM_RANGE_RATIO=${13}
NODE_LIST=${14}


NUM_NODES=$((PREFILL_NODES + DECODE_NODES))
profiler_args="${ISL} ${OSL} ${CONCURRENCIES} ${REQUEST_RATE}"

# Export variables for the SLURM job
export MODEL_DIR=$MODEL_PATH
export DOCKER_IMAGE_NAME=$CONTAINER_IMAGE
export PROFILER_ARGS=$profiler_args



export xP=$PREFILL_WORKERS
export yD=$DECODE_WORKERS
export NUM_NODES=$NUM_NODES
export GPUS_PER_NODE=$GPUS_PER_NODE
export MODEL_NAME=$MODEL_NAME
export PREFILL_TP_SIZE=$(( $PREFILL_NODES * $GPUS_PER_NODE / $PREFILL_WORKERS ))
export PREFILL_ENABLE_EP=${PREFILL_ENABLE_EP}
export PREFILL_ENABLE_DP=${PREFILL_ENABLE_DP}
export DECODE_TP_SIZE=$(( $DECODE_NODES * $GPUS_PER_NODE / $DECODE_WORKERS ))
export DECODE_ENABLE_EP=${DECODE_ENABLE_EP}
export DECODE_ENABLE_DP=${DECODE_ENABLE_DP}
export DECODE_MTP_SIZE=${DECODE_MTP_SIZE}
export BENCH_INPUT_LEN=${ISL}
export BENCH_OUTPUT_LEN=${OSL}
export BENCH_RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO}
export BENCH_NUM_PROMPTS_MULTIPLIER=10
export BENCH_MAX_CONCURRENCY=${CONCURRENCIES}
export BENCH_REQUEST_RATE=${REQUEST_RATE}

# Log directory: must be on NFS (shared filesystem) so the submit host can read SLURM output.
# SLURM writes output files on the batch node, so /tmp won't work (node-local).
# Defaults to a sibling directory of the submit working directory.
export BENCHMARK_LOGS_DIR="${BENCHMARK_LOGS_DIR:-$(pwd)/benchmark_logs}"
mkdir -p "$BENCHMARK_LOGS_DIR"

# Optional: pass an explicit node list to sbatch.
# NODE_LIST is expected to be comma-separated hostnames.
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

# Construct the sbatch command
sbatch_cmd=(
    sbatch
    --parsable
    -N "$NUM_NODES"
    -n "$NUM_NODES"
    "${NODELIST_OPT[@]}"
    --time "$TIME_LIMIT"
    --partition "$SLURM_PARTITION"
    --account "$SLURM_ACCOUNT"
    --job-name "$RUNNER_NAME"
    --output "${BENCHMARK_LOGS_DIR}/slurm_job-%j.out"
    --error "${BENCHMARK_LOGS_DIR}/slurm_job-%j.err"
    "$(dirname "$0")/job.slurm"
)

# todo: --parsable outputs only the jobid and cluster name, test if jobid;clustername is correct
JOB_ID=$("${sbatch_cmd[@]}")
if [[ $? -ne 0 ]]; then
    echo "Error: Failed to submit job with sbatch" >&2
    exit 1
fi
echo "$JOB_ID"
