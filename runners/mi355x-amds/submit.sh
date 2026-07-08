#!/bin/bash
#
# MI355X AMDS Slurm adapter for the cluster-agnostic multi-node contract.
#
# This script submits a multi-node disaggregated benchmark job to SLURM.
# It must be configured for your specific cluster before use.
#
# ENGINE=sglang (default): SGLang disaggregated serving
# ENGINE=vllm:             vLLM disaggregated serving
#
# Router is co-located with the first prefill node (same for both engines),
# so NUM_NODES = PREFILL_NODES + DECODE_NODES.
set -euo pipefail

usage() {
    cat << 'USAGE'
Usage:
  MULTINODE_LAUNCHER=runners/mi355x-amds/submit.sh \
    bash benchmarks/multi_node/<recipe>.sh

Required environment variables:
  Benchmark contract:
    CONC_LIST, ISL, OSL, PREFILL_NODES, PREFILL_NUM_WORKERS,
    PREFILL_TP, PREFILL_EP, PREFILL_DP_ATTN, PREFILL_ENABLE_EP,
    PREFILL_ENABLE_DP, DECODE_NODES, DECODE_NUM_WORKERS, DECODE_TP,
    DECODE_EP, DECODE_DP_ATTN, DECODE_ENABLE_EP, DECODE_ENABLE_DP,
    RANDOM_RANGE_RATIO, FRAMEWORK, CONTAINER_IMAGE, MULTINODE_BENCHMARK_DIR

  Cluster contract:
    SLURM_ACCOUNT, SLURM_PARTITION, TIME_LIMIT, MODEL_PATH, MODEL_NAME,
    RUNNER_NAME, INFERENCEX_REPO_ROOT, GPUS_PER_NODE
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

for required in \
    SLURM_ACCOUNT SLURM_PARTITION TIME_LIMIT MODEL_PATH MODEL_NAME \
    CONTAINER_IMAGE RUNNER_NAME FRAMEWORK GPUS_PER_NODE \
    INFERENCEX_REPO_ROOT MULTINODE_BENCHMARK_DIR \
    CONC_LIST ISL OSL RANDOM_RANGE_RATIO \
    PREFILL_NODES PREFILL_NUM_WORKERS PREFILL_TP PREFILL_EP PREFILL_DP_ATTN \
    PREFILL_ENABLE_EP PREFILL_ENABLE_DP \
    DECODE_NODES DECODE_NUM_WORKERS DECODE_TP DECODE_EP DECODE_DP_ATTN \
    DECODE_ENABLE_EP DECODE_ENABLE_DP; do
    check_env "$required"
done

for numeric in \
    GPUS_PER_NODE \
    PREFILL_NODES PREFILL_NUM_WORKERS PREFILL_TP \
    DECODE_NODES DECODE_NUM_WORKERS DECODE_TP; do
    if ! [[ "${!numeric}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: ${numeric} must be a positive integer, got '${!numeric}'" >&2
        exit 1
    fi
done

if (( (PREFILL_NODES * PREFILL_TP) % PREFILL_NUM_WORKERS != 0 )); then
    echo "Error: PREFILL_NODES * PREFILL_TP must be divisible by PREFILL_NUM_WORKERS" >&2
    echo "Error: ${PREFILL_NODES} * ${PREFILL_TP} / ${PREFILL_NUM_WORKERS} is not integral" >&2
    exit 1
fi
if (( (DECODE_NODES * DECODE_TP) % DECODE_NUM_WORKERS != 0 )); then
    echo "Error: DECODE_NODES * DECODE_TP must be divisible by DECODE_NUM_WORKERS" >&2
    echo "Error: ${DECODE_NODES} * ${DECODE_TP} / ${DECODE_NUM_WORKERS} is not integral" >&2
    exit 1
fi

PREFILL_WORKERS="$PREFILL_NUM_WORKERS"
DECODE_WORKERS="$DECODE_NUM_WORKERS"
CONCURRENCIES="${CONC_LIST// /x}"
REQUEST_RATE="${BENCH_REQUEST_RATE:-inf}"
NODE_LIST="${NODE_LIST:-}"

NUM_NODES=$((PREFILL_NODES + DECODE_NODES))
profiler_args="${ISL} ${OSL} ${CONCURRENCIES} ${REQUEST_RATE}"

# Export variables for the SLURM job
export ENGINE="${FRAMEWORK:-sglang}"
export MODEL_DIR="$MODEL_PATH"
export DOCKER_IMAGE_NAME="$CONTAINER_IMAGE"
export PROFILER_ARGS="$profiler_args"

# Engine-specific xP/yD semantics and TP exports
if [[ "$ENGINE" == "vllm-disagg" ]]; then
    export PROXY_STREAM_IDLE_TIMEOUT=${PROXY_STREAM_IDLE_TIMEOUT:-300}
fi
# xP = prefill workers, yD = decode workers (may span multiple nodes)
export xP="$PREFILL_WORKERS"
export yD="$DECODE_WORKERS"
export PREFILL_TP_SIZE=$((PREFILL_NODES * PREFILL_TP / PREFILL_WORKERS))
export PREFILL_NODES PREFILL_NUM_WORKERS PREFILL_TP PREFILL_EP PREFILL_DP_ATTN
export PREFILL_ENABLE_EP
export PREFILL_ENABLE_DP
export DECODE_TP_SIZE=$((DECODE_NODES * DECODE_TP / DECODE_WORKERS))
export DECODE_NODES DECODE_NUM_WORKERS DECODE_TP DECODE_EP DECODE_DP_ATTN
export DECODE_ENABLE_EP
export DECODE_ENABLE_DP
export DECODE_MTP_SIZE="${DECODE_MTP_SIZE:-0}"

export NUM_NODES
export GPUS_PER_NODE
export MODEL_NAME
export BENCH_INPUT_LEN="$ISL"
export BENCH_OUTPUT_LEN="$OSL"
export BENCH_NUM_PROMPTS_MULTIPLIER=${BENCH_NUM_PROMPTS_MULTIPLIER:-10}
export BENCH_MAX_CONCURRENCY="$CONCURRENCIES"
export BENCH_REQUEST_RATE="$REQUEST_RATE"
export BENCH_RANDOM_RANGE_RATIO="$RANDOM_RANGE_RATIO"

# Eval-related env vars (threaded from workflow → runner → here → job.slurm → Docker)
export RUN_EVAL="${RUN_EVAL:-false}"
export EVAL_ONLY="${EVAL_ONLY:-false}"
export EVAL_CONC="${EVAL_CONC:-}"
export FRAMEWORK
export PRECISION="${PRECISION:-}"
export MODEL_PREFIX="${MODEL_PREFIX:-}"
export RUNNER_TYPE="${RUNNER_TYPE:-}"
export RESULT_FILENAME="${RESULT_FILENAME:-}"
export SPEC_DECODING="${SPEC_DECODING:-none}"
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

# Optional runner policy: exclude cluster nodes that are under maintenance.
EXCLUDE_OPT=()
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
    --gres "gpu:$GPUS_PER_NODE"
)
if (( ${#NODELIST_OPT[@]} > 0 )); then
    sbatch_cmd+=("${NODELIST_OPT[@]}")
fi
if (( ${#EXCLUDE_OPT[@]} > 0 )); then
    sbatch_cmd+=("${EXCLUDE_OPT[@]}")
fi
sbatch_cmd+=(
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
