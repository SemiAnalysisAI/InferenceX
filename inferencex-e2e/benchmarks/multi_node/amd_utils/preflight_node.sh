#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../benchmark_lib.sh" --validation-only
check_env_vars DOCKER_CMD_DETECT DI_REPO_DIR SLURM_JOB_ID
CONT_FILTER="$1"
SKIP_GPU_SANITY="$2"
check_env_vars CONT_FILTER SKIP_GPU_SANITY

preflight_node() {
    eval "$DOCKER_CMD_DETECT"

    # Preserve the existing pre-clean scope and ordering. Moving it into this
    # separate Slurm step prevents one node starting while another still drains.
    $DOCKER_CMD ps -aq --filter "$CONT_FILTER" | xargs -r $DOCKER_CMD rm -f || true
    $DOCKER_CMD ps -aq | xargs -r $DOCKER_CMD stop -t 15 || true
    $DOCKER_CMD ps -aq | xargs -r $DOCKER_CMD rm -f || true
    sleep 2

    if [[ "$SKIP_GPU_SANITY" == "1" ]]; then
        echo "[INFO] SKIP_GPU_SANITY=1 set; skipping GPU pre-flight drain check"
    else
        # Avoid benchmark-only agentic initialization on the host, as before.
        bash -c 'unset IS_AGENTIC SCENARIO_TYPE; source "$DI_REPO_DIR/benchmarks/benchmark_lib.sh" && wait_for_amd_gpu_clean'
    fi
}

NODE_LOG_DIR="/tmp/slurm_job-${SLURM_JOB_ID}"
mkdir -p "$NODE_LOG_DIR"
preflight_node 2>&1 | tee "$NODE_LOG_DIR/preflight_$(hostname).log"
