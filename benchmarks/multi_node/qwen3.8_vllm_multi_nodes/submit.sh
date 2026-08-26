#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_ACCOUNT:?SLURM_ACCOUNT must be set}"
: "${SLURM_PARTITION:?SLURM_PARTITION must be set}"
: "${RUNNER_NAME:?RUNNER_NAME must be set}"
: "${BENCHMARK_LOGS_DIR:?BENCHMARK_LOGS_DIR must be set}"
: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE must be set}"
: "${IMAGE:?IMAGE must be set}"
: "${QWEN38_SCENARIO:?QWEN38_SCENARIO must be set}"

TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
NODE_LIST="${NODE_LIST:-${NODELIST:-}}"
SLURM_EXCLUDE_NODES="${SLURM_EXCLUDE_NODES:-mia1-p01-g09,mia1-p01-g10,mia1-p01-g11,mia1-p01-g12}"

mkdir -p "$BENCHMARK_LOGS_DIR"

node_args=()
if [[ -n "${NODE_LIST//[[:space:]]/}" ]]; then
    IFS=',' read -r -a nodes <<< "$NODE_LIST"
    if [[ "${#nodes[@]}" -ne 2 ]]; then
        echo "ERROR: NODE_LIST must contain exactly two nodes, got: $NODE_LIST" >&2
        exit 1
    fi
    node_args=(--nodelist "$(IFS=,; echo "${nodes[*]}")")
fi

exclude_args=()
if [[ -z "$NODE_LIST" && -n "$SLURM_EXCLUDE_NODES" ]]; then
    exclude_args=(--exclude "$SLURM_EXCLUDE_NODES")
fi

reservation_args=()
if [[ -n "${SLURM_RESERVATION:-}" ]]; then
    reservation_args=(--reservation "$SLURM_RESERVATION")
fi

job_id=$(
    sbatch \
        --parsable \
        --exclusive \
        --nodes=2 \
        --ntasks=2 \
        --ntasks-per-node=1 \
        --cpus-per-task=128 \
        --time="$TIME_LIMIT" \
        --partition="$SLURM_PARTITION" \
        --account="$SLURM_ACCOUNT" \
        --job-name="$RUNNER_NAME" \
        --output="$BENCHMARK_LOGS_DIR/slurm_job-%j.out" \
        --error="$BENCHMARK_LOGS_DIR/slurm_job-%j.err" \
        "${node_args[@]}" \
        "${exclude_args[@]}" \
        "${reservation_args[@]}" \
        "$(dirname "$0")/job.slurm"
)

echo "$job_id"
