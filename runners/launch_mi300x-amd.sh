#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_CACHE_MOUNT="/raid/inferencex/hf-hub-cache/"

PARTITION="compute"
SQUASH_DIR="/raid/inferencex/squash"
SQUASH_FILE="$SQUASH_DIR/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
LOCK_FILE="${SQUASH_FILE}.lock"
COMPUTE_TMPDIR="$SQUASH_DIR/tmp-${UID}"
# Some MI300X compute images do not provision the login user's /run/user/$UID
# directory. Keep Enroot's runtime state node-local instead of inheriting the
# controller's XDG_RUNTIME_DIR, which is not valid on those compute nodes.
export XDG_RUNTIME_DIR="/tmp/enroot-runtime-${UID}"

# Route spec-decoding=mtp configs to the _mtp benchmark script (parity with
# the h200 launchers, which have carried SPEC_SUFFIX since #392).
SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')

export GPU_COUNT="${GPU_COUNT:-${TP:?TP must be set}}"

set -x

JOB_ID=$(set +o pipefail; salloc --partition=$PARTITION --gres=gpu:$GPU_COUNT --cpus-per-task=128 --time=180 --no-shell --job-name="$RUNNER_NAME" 2>&1 | tee /dev/stderr | grep -oP 'Granted job allocation \K[0-9]+')

if [ -z "$JOB_ID" ]; then
    echo "ERROR: salloc failed to allocate a job" >&2
    exit 1
fi

export PORT=$((40000 + (JOB_ID % 10000)))

# The GitHub runners live on the Slurm controller, whose /home filesystem is
# not mounted on compute nodes. Stage the checked-out source with sbcast, then
# copy only benchmark artifacts back over srun stdout before releasing the job.
LOCAL_WORKSPACE_TAR=$(mktemp "/tmp/inferencex-${JOB_ID}.XXXXXX.tar.gz")
REMOTE_WORKSPACE_TAR="/tmp/inferencex-${JOB_ID}.tar.gz"
REMOTE_WORKSPACE="/tmp/inferencex-${JOB_ID}"

cleanup() {
    local rc=$?
    srun --jobid="$JOB_ID" bash -c "rm -rf '$REMOTE_WORKSPACE' '$REMOTE_WORKSPACE_TAR'" >/dev/null 2>&1 || true
    scancel "$JOB_ID" 2>/dev/null || true
    rm -f "$LOCAL_WORKSPACE_TAR"
    exit "$rc"
}
trap cleanup EXIT

tar -C "$GITHUB_WORKSPACE" \
    --exclude=.git --exclude='*/.git' \
    -czf "$LOCAL_WORKSPACE_TAR" .
sbcast --jobid="$JOB_ID" --force "$LOCAL_WORKSPACE_TAR" "$REMOTE_WORKSPACE_TAR"
srun --jobid="$JOB_ID" --job-name="$RUNNER_NAME" bash -c "
    set -euo pipefail
    mkdir -p '$XDG_RUNTIME_DIR'
    chmod 700 '$XDG_RUNTIME_DIR'
    mkdir -p '$COMPUTE_TMPDIR'
    chmod 700 '$COMPUTE_TMPDIR'
    rm -rf '$REMOTE_WORKSPACE'
    mkdir -p '$REMOTE_WORKSPACE'
    tar -C '$REMOTE_WORKSPACE' -xzf '$REMOTE_WORKSPACE_TAR'
"

# Use flock to serialize concurrent imports to the same squash file
srun --jobid="$JOB_ID" --job-name="$RUNNER_NAME" bash -c "
    set -euo pipefail
    export TMPDIR='$COMPUTE_TMPDIR'
    mkdir -p '$SQUASH_DIR'
    exec 9>\"$LOCK_FILE\"
    flock -w 600 9 || { echo 'Failed to acquire lock for $SQUASH_FILE'; exit 1; }
    if unsquashfs -l \"$SQUASH_FILE\" > /dev/null 2>&1; then
        echo 'Squash file already exists and is valid, skipping import'
    else
        rm -f \"$SQUASH_FILE\"
        enroot import -o \"$SQUASH_FILE\" docker://$IMAGE
    fi
"

set +e
srun --jobid="$JOB_ID" \
--container-image=$SQUASH_FILE \
--container-mounts=$REMOTE_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE,/dev/kfd:/dev/kfd,/dev/dri:/dev/dri \
--container-mount-home \
--container-writable \
--container-remap-root \
--container-workdir=/workspace/ \
--no-container-entrypoint --export=ALL \
bash benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_mi300x${SPEC_SUFFIX}.sh
BENCHMARK_RC=$?
set -e

# Stream runtime artifacts back to the controller-side Actions workspace. The
# binary archive is stdout-only; Slurm diagnostics remain on stderr.
srun --jobid="$JOB_ID" bash -c "
    set -euo pipefail
    cd '$REMOTE_WORKSPACE'
    shopt -s nullglob
    artifacts=()
    for path in results LOGS *.json *.log gpu_metrics.csv profile_*.trace.json.gz eval_results*; do
        [[ -e \"\$path\" ]] && artifacts+=(\"\$path\")
    done
    if (( \${#artifacts[@]} == 0 )); then
        tar -czf - --files-from /dev/null
    else
        tar -czf - \"\${artifacts[@]}\"
    fi
" | tar -C "$GITHUB_WORKSPACE" -xzf -

exit "$BENCHMARK_RC"
