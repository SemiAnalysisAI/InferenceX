#!/bin/bash
#
# SLURM job submission for ATOM disaggregated benchmark (multi-node).
#
# Adapted verbatim from benchmarks/multi_node/amd_utils/submit.sh; the only
# semantic differences are:
#   1. Submits this directory's job.slurm (atom_disagg_utils/job.slurm)
#      rather than amd_utils/job.slurm.
#   2. Drops DECODE_MTP_SIZE (ATOM disagg enablement phase does not yet
#      enable MTP; the variable is still threaded through so future PRs
#      can switch it on without script-level changes).
# Everything else — argument order, env var exports, sbatch construction —
# mirrors amd_utils/submit.sh exactly so the SGLang-side disagg sbatch
# wrapping stays a usable reference.

usage() {
    cat << 'USAGE'
Multi-node ATOM disaggregated benchmark submission.

Positional arguments (must match dsv4_fp4_mi355x_atom-disagg.sh):
  $1  PREFILL_NODES
  $2  PREFILL_WORKERS
  $3  DECODE_NODES
  $4  DECODE_WORKERS
  $5  ISL
  $6  OSL
  $7  CONCURRENCIES  ('x'-separated list, e.g. "1x2x4")
  $8  REQUEST_RATE   (typically "inf")
  $9  PREFILL_ENABLE_EP   (true|false)
  $10 PREFILL_ENABLE_DP   (true|false)
  $11 DECODE_ENABLE_EP    (true|false)
  $12 DECODE_ENABLE_DP    (true|false)
  $13 PREFILL_TP
  $14 DECODE_TP
  $15 RANDOM_RANGE_RATIO
  $16 NODE_LIST  (optional; comma-separated hostnames)

Required env vars (set by runners/launch_mi355x-amds.sh):
  SLURM_ACCOUNT, SLURM_PARTITION, TIME_LIMIT, MODEL_PATH, MODEL_NAME,
  CONTAINER_IMAGE, RUNNER_NAME, GPUS_PER_NODE
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

PREFILL_NODES=$1
PREFILL_WORKERS=${2:-1}
DECODE_NODES=$3
DECODE_WORKERS=${4:-1}
ISL=$5
OSL=$6
CONCURRENCIES=$7
REQUEST_RATE=$8
PREFILL_ENABLE_EP=${9:-false}
PREFILL_ENABLE_DP=${10:-false}
DECODE_ENABLE_EP=${11:-false}
DECODE_ENABLE_DP=${12:-false}
PREFILL_TP=${13:-8}
DECODE_TP=${14:-8}
RANDOM_RANGE_RATIO=${15}
NODE_LIST=${16}

NUM_NODES=$((PREFILL_NODES + DECODE_NODES))
profiler_args="${ISL} ${OSL} ${CONCURRENCIES} ${REQUEST_RATE}"

export MODEL_DIR=$MODEL_PATH
export DOCKER_IMAGE_NAME=$CONTAINER_IMAGE
export PROFILER_ARGS=$profiler_args

export xP=$PREFILL_WORKERS
export yD=$DECODE_WORKERS
export NUM_NODES=$NUM_NODES
export GPUS_PER_NODE=$GPUS_PER_NODE
export MODEL_NAME=$MODEL_NAME
export PREFILL_TP_SIZE=$(( PREFILL_NODES * PREFILL_TP / PREFILL_WORKERS ))
export PREFILL_ENABLE_EP=${PREFILL_ENABLE_EP}
export PREFILL_ENABLE_DP=${PREFILL_ENABLE_DP}
export DECODE_TP_SIZE=$(( DECODE_NODES * DECODE_TP / DECODE_WORKERS ))
export DECODE_ENABLE_EP=${DECODE_ENABLE_EP}
export DECODE_ENABLE_DP=${DECODE_ENABLE_DP}
export BENCH_INPUT_LEN=${ISL}
export BENCH_OUTPUT_LEN=${OSL}
export BENCH_RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO}
export BENCH_NUM_PROMPTS_MULTIPLIER=10
export BENCH_MAX_CONCURRENCY=${CONCURRENCIES}
export BENCH_REQUEST_RATE=${REQUEST_RATE}

# Eval-related env vars (threaded from workflow → runner → here → job.slurm → Docker)
export RUN_EVAL="${RUN_EVAL:-false}"
export EVAL_ONLY="${EVAL_ONLY:-false}"
export EVAL_CONC="${EVAL_CONC:-}"
export FRAMEWORK="${FRAMEWORK:-atom-disagg}"
export PRECISION="${PRECISION:-}"
export MODEL_PREFIX="${MODEL_PREFIX:-}"
export RUNNER_TYPE="${RUNNER_TYPE:-}"
export RESULT_FILENAME="${RESULT_FILENAME:-}"
export SPEC_DECODING="${SPEC_DECODING:-}"
export IS_MULTINODE="${IS_MULTINODE:-true}"

# Log directory: must be on NFS (shared filesystem) so the submit host can read SLURM output.
export BENCHMARK_LOGS_DIR="${BENCHMARK_LOGS_DIR:-$(pwd)/benchmark_logs}"
mkdir -p "$BENCHMARK_LOGS_DIR"

# Optional explicit node list to sbatch (comma-separated hostnames).
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

# Optional exclude list to sbatch. sbatch only recognises a couple of env
# vars (SBATCH_NODELIST, SBATCH_HOSTFILE) and *not* SBATCH_EXCLUDE_NODES on
# this cluster, so honour our own EXCLUDE_NODES env var by passing it
# explicitly as --exclude. Use this to keep jobs off nodes that are known
# to be unusable for the current user (e.g. nodes where the user is not
# in the host-side docker group → `sg docker` prompts for a password and
# the SLURM srun hangs).
EXCLUDE_OPT=()
if [[ -n "${EXCLUDE_NODES//[[:space:]]/}" ]]; then
    EXCLUDE_OPT=(--exclude "$EXCLUDE_NODES")
fi

sbatch_cmd=(
    sbatch
    --parsable
    --exclusive
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
