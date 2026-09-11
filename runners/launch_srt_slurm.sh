#!/usr/bin/env bash
set -euo pipefail

# Shared workflow adapter. Cluster entry points supply paths; srt-slurm owns
# allocation, container startup, workers, router, and benchmark execution.
: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE must be set by Actions}"
: "${RESULT_FILENAME:?RESULT_FILENAME must be set by the benchmark workflow}"
: "${CONFIG_FILE:?CONFIG_FILE must select an srt-slurm recipe}"
: "${IMAGE:?IMAGE must identify the serving container}"
: "${MODEL:?MODEL must identify the checkpoint}"
: "${SRT_SLURM_CLUSTER_CONFIG:?Cluster profile is required}"
: "${SRT_SLURM_SHARED_BASE:?Shared runtime directory is required}"
: "${AIPERF_MMAP_CACHE_HOST_PATH:?Shared AIPerf cache is required}"
SRT_SLURM_REPOSITORY="${SRT_SLURM_REPOSITORY:-https://github.com/SemiAnalysisAI/srt-slurm.git}"
SRT_SLURM_COMMIT="${SRT_SLURM_COMMIT:-81d46274f508e18ab14d1f123b75132005818dcf}"
RUN_KEY="${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-0}-${RUNNER_NAME:-runner}"
WORK_DIR=$(mktemp -d "${GITHUB_WORKSPACE}/.srt-slurm-${RUN_KEY}.XXXXXX")
SRT_REPO_DIR="${WORK_DIR}/srt-slurm"
SHARED_RESULTS="${SRT_SLURM_SHARED_BASE}/results"
CONFIG_PATH="${CONFIG_FILE%%:*}"
LOCAL_RECIPE="${GITHUB_WORKSPACE}/benchmarks/multi_node/srt-slurm-recipes/${CONFIG_PATH#recipes/}"
ADAPTER="${GITHUB_WORKSPACE}/utils/srt_slurm.py"
mkdir -p "$SHARED_RESULTS" "$AIPERF_MMAP_CACHE_HOST_PATH"

git clone "$SRT_SLURM_REPOSITORY" "$SRT_REPO_DIR"
git -C "$SRT_REPO_DIR" checkout --detach "$SRT_SLURM_COMMIT"
[[ "$(git -C "$SRT_REPO_DIR" rev-parse HEAD)" == "$SRT_SLURM_COMMIT" ]]
cd "$SRT_REPO_DIR"
make setup-compute ARCH="${SRT_SLURM_COMPUTE_ARCH:-x86_64}"
export PATH="$SRT_REPO_DIR/bin:$PATH"
uv venv --python 3.12
uv pip install -e .
source .venv/bin/activate

python "$ADAPTER" prepare \
    --recipe "$LOCAL_RECIPE" --profile "$SRT_SLURM_CLUSTER_CONFIG" \
    --work-dir "$WORK_DIR" --workspace "$GITHUB_WORKSPACE" \
    --results-root "$SHARED_RESULTS" --aiperf-cache "$AIPERF_MMAP_CACHE_HOST_PATH" \
    --image-cache "${SRT_SLURM_SHARED_BASE}/containers"
export SRTSLURM_CONFIG="${WORK_DIR}/srtslurm.yaml"
export SRTCTL_RUNTIME_SOURCE_DIR="$SRT_REPO_DIR"
PREPARED_RECIPE="${WORK_DIR}/recipe.yaml"
[[ "$CONFIG_FILE" != *:* ]] || PREPARED_RECIPE="${PREPARED_RECIPE}:${CONFIG_FILE#*:}"
SUBMISSION="${WORK_DIR}/submission.json"
JOB_ID=""
cleanup() {
    local rc=$? collect_rc=0
    trap - EXIT INT TERM
    if [[ -z "$JOB_ID" && -s "$SUBMISSION" ]]; then
        JOB_ID=$(jq -er '.slurm_job_id' "$SUBMISSION") || JOB_ID=""
    fi
    if [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
        # This ID comes only from this invocation's srtctl submission.
        if [[ "$rc" -ne 0 ]]; then scancel "$JOB_ID" || true; fi
        python "$ADAPTER" collect --submission "$SUBMISSION" \
            --workspace "$GITHUB_WORKSPACE" --results-root "$SHARED_RESULTS" || collect_rc=$?
        [[ "$rc" -ne 0 ]] || rc=$collect_rc
    fi
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Keep the CI host's bytecode-cache path out of the container environment.
env -u PYTHONPYCACHEPREFIX srtctl apply -f "$PREPARED_RECIPE" \
    --tags "inferencex,github-actions,${RUN_KEY}" --json > "$SUBMISSION"
JOB_ID=$(jq -er '.slurm_job_id' "$SUBMISSION")
[[ "$JOB_ID" =~ ^[0-9]+$ ]]
OUTPUT_DIR=$(jq -er '.output_dir' "$SUBMISSION")
echo "SRT_SLURM_JOB_ID=$JOB_ID"
printf '%s\n' "$SRT_SLURM_COMMIT" > "$GITHUB_WORKSPACE/srt-slurm-producer-sha.txt"
srtctl wait "$JOB_ID" --log-file "${OUTPUT_DIR}/logs/sweep_${JOB_ID}.log"
