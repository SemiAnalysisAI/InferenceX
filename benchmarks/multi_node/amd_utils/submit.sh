#!/bin/bash
#
# Cluster Configuration Template for Multi-Node Disaggregated Serving
#
# This script submits a multi-node disaggregated benchmark job to SLURM.
# It must be configured for your specific cluster before use.
#
# ENGINE=sglang (default): SGLang disaggregated serving
# ENGINE=vllm:             vLLM disaggregated serving
#
# Router is co-located with the first prefill node (same for both engines),
# so NUM_NODES = PREFILL_NODES + DECODE_NODES.

usage() {
    cat << 'USAGE'
Usage:
  bash submit.sh <PREFILL_NODES> <PREFILL_WORKERS> <DECODE_NODES> <DECODE_WORKERS> \
                 <ISL> <OSL> <CONCURRENCIES> <REQUEST_RATE> \
                 <PREFILL_ENABLE_EP> <PREFILL_ENABLE_DP> \
                 <DECODE_ENABLE_EP> <DECODE_ENABLE_DP> \
                 <PREFILL_TP> <DECODE_TP> \
                 <RANDOM_RANGE_RATIO> [NODE_LIST]

Arguments:
  PREFILL_NODES        Number of prefill nodes
  PREFILL_WORKERS      Number of prefill workers (usually 1)
  DECODE_NODES         Number of decode nodes
  DECODE_WORKERS       Number of decode workers (usually 1)
  ISL                  Input sequence length
  OSL                  Output sequence length
  CONCURRENCIES        Concurrency levels, delimited by 'x' (e.g., "8x16x32")
  REQUEST_RATE         Request rate ("inf" for max throughput)
  PREFILL_ENABLE_EP    true/false or 1/0 (expert parallelism on prefill)
  PREFILL_ENABLE_DP    true/false or 1/0 (data-parallel attention on prefill)
  DECODE_ENABLE_EP     true/false or 1/0 (expert parallelism on decode)
  DECODE_ENABLE_DP     true/false or 1/0 (data-parallel attention on decode)
  PREFILL_TP           Tensor parallel size per prefill node
  DECODE_TP            Tensor parallel size per decode node
  RANDOM_RANGE_RATIO   Random range ratio for benchmark client
  NODE_LIST            Optional: comma-separated hostnames (must match NUM_NODES)

Required environment variables:
  SLURM_ACCOUNT    SLURM account name
  SLURM_PARTITION  SLURM partition
  TIME_LIMIT       Job time limit (e.g., "08:00:00")
  MODEL_PATH       Path to model directory (e.g., /nfsdata)
  MODEL_NAME       Model name directory
  CONTAINER_IMAGE  Docker image name (e.g., vllm_disagg_pd:latest)
  RUNNER_NAME      Runner identifier (for job name)

Optional environment variables:
  DRY_RUN          1 = echo composed server/router launch commands instead of
                   running them (preview a recipe against a real allocation).
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
check_env FRAMEWORK

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
PREFILL_ENABLE_EP=${9:-true}
PREFILL_ENABLE_DP=${10:-true}
DECODE_ENABLE_EP=${11:-true}
DECODE_ENABLE_DP=${12:-true}
PREFILL_TP=${13:-8}
DECODE_TP=${14:-8}
RANDOM_RANGE_RATIO=${15:-0.8}
NODE_LIST=${16}

NUM_NODES=$((PREFILL_NODES + DECODE_NODES))
profiler_args="${ISL} ${OSL} ${CONCURRENCIES} ${REQUEST_RATE}"

# Export variables for the SLURM job
export ENGINE="${FRAMEWORK:-sglang}"
export MODEL_DIR=$MODEL_PATH
export DOCKER_IMAGE_NAME=$CONTAINER_IMAGE
export PROFILER_ARGS=$profiler_args

# Engine-specific xP/yD semantics and TP exports
if [[ "$ENGINE" == "vllm-disagg" ]]; then
    export PROXY_STREAM_IDLE_TIMEOUT=${PROXY_STREAM_IDLE_TIMEOUT:-300}
fi
# xP = prefill workers, yD = decode workers (may span multiple nodes)
export xP=$PREFILL_WORKERS
export yD=$DECODE_WORKERS
export PREFILL_TP_SIZE=$(( $PREFILL_NODES * $PREFILL_TP / $PREFILL_WORKERS ))
export PREFILL_ENABLE_EP=${PREFILL_ENABLE_EP}
export PREFILL_ENABLE_DP=${PREFILL_ENABLE_DP}
export DECODE_TP_SIZE=$(( $DECODE_NODES * $DECODE_TP / $DECODE_WORKERS ))
export DECODE_ENABLE_EP=${DECODE_ENABLE_EP}
export DECODE_ENABLE_DP=${DECODE_ENABLE_DP}
export DECODE_MTP_SIZE=${DECODE_MTP_SIZE}

export NUM_NODES=$NUM_NODES
export GPUS_PER_NODE=$GPUS_PER_NODE
export MODEL_NAME=$MODEL_NAME
export BENCH_INPUT_LEN=${ISL}
export BENCH_OUTPUT_LEN=${OSL}
export BENCH_NUM_PROMPTS_MULTIPLIER=${BENCH_NUM_PROMPTS_MULTIPLIER:-10}
export BENCH_MAX_CONCURRENCY=${CONCURRENCIES}
export BENCH_REQUEST_RATE=${REQUEST_RATE}
export BENCH_RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.8}

# DRY_RUN=1 makes server_sglang.sh echo the composed prefill/decode/router launch
# commands instead of executing them (useful for previewing a recipe against a real
# allocation). Threaded here → job.slurm → Docker (-e DRY_RUN) → server_sglang.sh.
# sbatch defaults to --export=ALL, so exporting it is what carries it into the job.
export DRY_RUN="${DRY_RUN:-0}"

# Eval-related env vars (threaded from workflow → runner → here → job.slurm → Docker)
export RUN_EVAL="${RUN_EVAL:-false}"
export EVAL_ONLY="${EVAL_ONLY:-false}"
export EVAL_CONC="${EVAL_CONC:-}"
export FRAMEWORK="${FRAMEWORK:-}"
export PRECISION="${PRECISION:-}"
export MODEL_PREFIX="${MODEL_PREFIX:-}"
export RUNNER_TYPE="${RUNNER_TYPE:-}"
export RESULT_FILENAME="${RESULT_FILENAME:-}"
export SPEC_DECODING="${SPEC_DECODING:-}"
export IS_MULTINODE="${IS_MULTINODE:-false}"

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
SLURM_EXCLUDE_NODES="${SLURM_EXCLUDE_NODES:-mia1-p01-g11,mia1-p01-g12,mia1-p01-g15}"
if [[ -n "${SLURM_EXCLUDE_NODES:-}" ]]; then
    EXCLUDE_OPT=(--exclude "$SLURM_EXCLUDE_NODES")
fi

# =============================================================================
# Reuse existing allocation (skip sbatch)
# =============================================================================
# When SLURM_REUSE_JOBID is set, run job.slurm directly in the current shell,
# attaching to the existing allocation. Inner `srun` calls pick up the
# allocation via SLURM_JOB_ID; SLURM_OVERLAP=1 lets them share task slots with
# the interactive shell already holding the allocation.
if [[ -n "${SLURM_REUSE_JOBID:-}" ]]; then
    REUSE_JID="$SLURM_REUSE_JOBID"
    echo "Reusing existing Slurm allocation ${REUSE_JID} (skipping sbatch)" >&2

    # Resolve allocation's nodelist if not already provided.
    ALLOC_NODELIST="${SLURM_JOB_NODELIST:-$(squeue -h -j "$REUSE_JID" -o '%N' 2>/dev/null)}"
    if [[ -z "$ALLOC_NODELIST" ]]; then
        echo "Error: could not resolve nodelist for job ${REUSE_JID}" >&2
        exit 1
    fi
    ALLOC_NNODES=$(scontrol show hostnames "$ALLOC_NODELIST" | wc -l)
    if [[ "$ALLOC_NNODES" -lt "$NUM_NODES" ]]; then
        echo "Error: allocation ${REUSE_JID} has ${ALLOC_NNODES} nodes, need ${NUM_NODES}" >&2
        exit 1
    fi

    export SLURM_JOB_ID="$REUSE_JID"
    export SLURM_JOBID="$REUSE_JID"
    export SLURM_JOB_NODELIST="$ALLOC_NODELIST"
    export SLURM_NODELIST="$ALLOC_NODELIST"
    export SLURM_NNODES="$ALLOC_NNODES"
    export SLURM_JOB_NUM_NODES="$ALLOC_NNODES"
    export SLURM_NTASKS="$ALLOC_NNODES"
    export SLURM_NPROCS="$ALLOC_NNODES"
    export SLURM_NTASKS_PER_NODE=1
    export SLURM_TASKS_PER_NODE="1(x${ALLOC_NNODES})"
    export SLURM_OVERLAP=1
    export SLURM_SUBMIT_DIR="$(pwd)"

    STDOUT_LOG="${BENCHMARK_LOGS_DIR}/slurm_job-${REUSE_JID}.out"
    STDERR_LOG="${BENCHMARK_LOGS_DIR}/slurm_job-${REUSE_JID}.err"
    rm -f "$STDOUT_LOG" "$STDERR_LOG"

    nohup bash "$(dirname "$0")/job.slurm" >"$STDOUT_LOG" 2>"$STDERR_LOG" &
    INLINE_PID=$!
    echo "$INLINE_PID" > "${BENCHMARK_LOGS_DIR}/slurm_job-${REUSE_JID}.pid"
    echo "Started job.slurm (pid=${INLINE_PID}); logs: ${STDOUT_LOG}" >&2

    echo "$REUSE_JID"
    exit 0
fi

# Construct the sbatch command
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
