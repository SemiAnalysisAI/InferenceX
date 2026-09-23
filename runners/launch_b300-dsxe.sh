#!/usr/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/../benchmarks/benchmark_lib.sh" --validation-only || exit 1
check_env_vars ENROOT_IMPORT_TIME_LIMIT EVAL_ONLY IS_AGENTIC IS_MULTINODE RUN_EVAL SALLOC_TIME_LIMIT

# shellcheck source=runners/slurm_utils.sh
source "$(dirname "${BASH_SOURCE[0]}")/slurm_utils.sh" || exit 1

# B300 DSXE Slurm cluster (dsxe-sa-b300-prd0); runners run as sa-gha-runner.
# Cluster-specific facts live in this block. Multi-node jobs go through
# srt-slurm/srtctl, single-node jobs through salloc + pyxis.

SLURM_PARTITION="batch_1"
SLURM_ACCOUNT="benchmark"

# This lane's interactive allocation notifications fail on login-02, while
# batch submission and steps launched from the allocated node work. Keep the
# workaround scoped to this recipe and use normal Slurm resource accounting.
if [[ "$IS_MULTINODE" != true && "${MODEL_PREFIX:-}" == dsv41flash &&
      "${FRAMEWORK:-}" == sglang && "${IS_AGENTIC:-}" == 1 &&
      "${B300_AGENTX_BATCH:-}" != 1 ]]; then
    check_env_vars GITHUB_WORKSPACE GPU_COUNT RUNNER_NAME
    BATCH_SCRIPT=$(mktemp "${RUNNER_TEMP:-$GITHUB_WORKSPACE}/b300-agentx.XXXXXX.sh") || exit 1
    BATCH_LOG="${BATCH_SCRIPT%.sh}.log"
    {
        printf '#!/usr/bin/env bash\nexport B300_AGENTX_BATCH=1\nexec bash '
        printf '%q\n' "$GITHUB_WORKSPACE/runners/launch_b300-dsxe.sh"
    } > "$BATCH_SCRIPT"
    BATCH_ARGS=(--parsable --partition="$SLURM_PARTITION" --account="$SLURM_ACCOUNT"
        --nodes=1 --ntasks=1 --gres="gpu:$GPU_COUNT" --exclusive --mem=0
        --time="$SALLOC_TIME_LIMIT" --job-name="$RUNNER_NAME" --export=ALL
        --chdir="$GITHUB_WORKSPACE" --output="$BATCH_LOG")
    if [[ -n "${SALLOC_EXCLUDE:-}" ]]; then
        BATCH_ARGS+=(--exclude="$SALLOC_EXCLUDE")
    fi
    JOB_ID=$(sbatch "${BATCH_ARGS[@]}" "$BATCH_SCRIPT") || { rm -f "$BATCH_SCRIPT"; exit 1; }
    JOB_ID="${JOB_ID%%;*}"
    [[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo 'ERROR: B300 batch allocation unavailable' >&2; rm -f "$BATCH_SCRIPT"; exit 1; }
    trap 'rc=$?; scancel "$JOB_ID" 2>/dev/null || true; rm -f "$BATCH_SCRIPT"; exit "$rc"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    echo "B300 AgentX batch job $JOB_ID; log: $BATCH_LOG"
    stream_slurm_job_log "$JOB_ID" "$BATCH_LOG" || exit 1
    verify_slurm_job_status "$JOB_ID"
    exit $?
fi

# enroot squash images. Must be on storage every compute node mounts and writable
# by the runner user (/data/squash is root-owned, hence the per-user default).
SQUASH_DIR="/data/home/sa-gha-runner/squash"

# Weights. MODEL_ROOT is node-local NVMe with the same layout on every compute node;
# it is read-only from the job's point of view. Anything not in STAGED_MODELS is
# downloaded into WRITABLE_MODELS_DIR (shared Lustre) by the single-node scripts.
MODEL_ROOT="/scratch/models"
SHARED_MODEL_ROOT="/data/models"
WRITABLE_MODELS_DIR="/data/home/sa-gha-runner/models"


# Directory names under MODEL_ROOT (upstream HF repo basenames).
STAGED_MODELS=(
    DeepSeek-R1-0528
    DeepSeek-R1-0528-NVFP4-v2
    DeepSeek-V4-Pro
    DeepSeek-V4-Pro-0813
    DeepSeek-V4-Pro-NVFP4
    GLM-5.2-FP8
    GLM-5.2-NVFP4
    Kimi-K2.6-NVFP4 # Retained Kimi SPEED-Bench collector staging.
    Kimi-K3
    MiniMax-M3
    MiniMax-M3-MXFP8
    MiniMax-M3-NVFP4
    Qwen3.5-397B-A17B-FP8
    Qwen3.5-397B-A17B-NVFP4
    Qwen3.5-397B-A17B-NVFP4-V2
    Qwen3.8-2.4T-A95B-FP8
)

mkdir -p "$SQUASH_DIR"
set -x

# Keep this definition above the IS_MULTINODE branch: both paths call it, and
# bash only defines a function when execution reaches it.
#
# Concurrent callers target the same squash path, so serialize on a per-file
# lock. The import must run on a compute node (enroot builds the squashfs over
# an overlay mount the shared FS cannot back, and the login host is too small),
# but reading a finished file is plain I/O, so probe here first; a warm cache
# then costs no allocation. --time bounds the step because an unbounded srun
# hangs the job if its step is lost.
import_squash_image() {
    local image_ref="$1"
    local sqsh="$2"
    local lock="${2}.lock"

    if unsquashfs -l "$sqsh" > /dev/null 2>&1; then
        echo "Squash file already present, skipping import: $sqsh"
        return 0
    fi

    local import_launcher=(srun -N 1 -A "$SLURM_ACCOUNT" -p "$SLURM_PARTITION"
        --time="${ENROOT_IMPORT_TIME_LIMIT}")
    if [[ "${B300_AGENTX_BATCH:-}" == 1 ]]; then
        # Already inside our exclusive compute-node allocation.
        import_launcher=()
    fi
    "${import_launcher[@]}" bash -c "
        set -eo pipefail
        exec 9>\"$lock\"
        flock -w 3600 9
        if unsquashfs -l \"$sqsh\" > /dev/null 2>&1; then
            exit 0
        fi
        rm -f \"$sqsh\"
        enroot import -o \"$sqsh\" \"docker://$image_ref\"
        unsquashfs -l \"$sqsh\" > /dev/null
    " || { echo "Error: enroot import failed for $image_ref -> $sqsh" >&2; exit 1; }

    test -r "$sqsh" || { echo "Error: squash file not readable: $sqsh" >&2; exit 1; }
}

if [[ "$IS_MULTINODE" == "true" ]]; then

if [[ $FRAMEWORK != "dynamo-sglang" && $FRAMEWORK != "dynamo-trt" && $FRAMEWORK != "dynamo-vllm" ]]; then
    echo "Unsupported framework: $FRAMEWORK. Supported frameworks are: dynamo-trt, dynamo-sglang, dynamo-vllm"
    exit 1
fi

USES_DCGM_POWER=0
_RECIPE_REL="${CONFIG_FILE%%:*}"
_RECIPE_SRC="$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/${_RECIPE_REL#recipes/}"
if [[ -n "$CONFIG_FILE" && -f "$_RECIPE_SRC" ]] && awk '
    /^telemetry:/ { t = 1; next }
    t && /^[^ ]/  { t = 0 }
    t && /^  dcgm_exporter:/ { p = 1 }
    t && /^  enabled: true$/        { e = 1 }
    END { exit !(p && e) }
' "$_RECIPE_SRC"; then
    USES_DCGM_POWER=1
fi
if [[ "$USES_DCGM_POWER" == "1" && (
    "${IS_AGENTIC}" == "1" ||
    "$MODEL_PREFIX" != "dsv4" ||
    "$PRECISION" != "fp4" ||
    ( "$FRAMEWORK" != "dynamo-sglang" && "$FRAMEWORK" != "dynamo-vllm" )
) ]]; then
    echo "Error: B300 dcgm-power is limited to fixed-sequence DSV4 FP4 dynamo-sglang/vllm" >&2
    exit 1
fi

SRT_REPO_DIR="srt-slurm"
rm -rf "$SRT_REPO_DIR"
setup_srt_slurm "$SRT_REPO_DIR" "$FRAMEWORK" "$USES_DCGM_POWER" || exit 1

echo "Installing srtctl..."
export UV_INSTALL_DIR="$GITHUB_WORKSPACE/.local/bin"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$UV_INSTALL_DIR:$PATH"

uv venv "$GITHUB_WORKSPACE/.venv"
source "$GITHUB_WORKSPACE/.venv/bin/activate"
uv pip install -e .

if ! command -v srtctl &> /dev/null; then
    echo "Error: Failed to install srtctl"
    exit 1
fi

NGINX_IMAGE="nginx:1.27.4"
SQUASH_FILE="$SQUASH_DIR/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
NGINX_SQUASH_FILE="$SQUASH_DIR/$(echo "$NGINX_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

import_squash_image "$IMAGE" "$SQUASH_FILE"
import_squash_image "$NGINX_IMAGE" "$NGINX_SQUASH_FILE"

if [[ "$USES_DCGM_POWER" == "1" ]]; then
    DCGM_EXPORTER_IMAGE="nvcr.io/nvidia/k8s/dcgm-exporter:4.6.0-4.8.3-distroless"
    # enroot resolves bare paths against Docker Hub; nvcr.io pulls need the registry# form
    DCGM_EXPORTER_ENROOT_REF="${DCGM_EXPORTER_IMAGE/nvcr.io\//nvcr.io#}"
    DCGM_EXPORTER_SQSH="$SQUASH_DIR/$(echo "$DCGM_EXPORTER_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    import_squash_image "$DCGM_EXPORTER_ENROOT_REF" "$DCGM_EXPORTER_SQSH"
    sha256sum "$DCGM_EXPORTER_SQSH" > "$GITHUB_WORKSPACE/exporter-image.sha256"
fi

export ISL="$ISL"
export OSL="$OSL"

SRTCTL_ROOT="${GITHUB_WORKSPACE}/${SRT_REPO_DIR}"
echo "Creating srtslurm.yaml configuration..."
write_srt_cluster_config b300-dsxe srtslurm.yaml "$USES_DCGM_POWER" \
    --var MODEL_ROOT "$MODEL_ROOT" || exit 1

echo "Generated srtslurm.yaml:"
cat srtslurm.yaml

echo "Running make setup..."
make setup ARCH=x86_64

# Read by srt-slurm's post-benchmark eval.
export INFMAX_WORKSPACE="$GITHUB_WORKSPACE"

echo "Submitting job with srtctl..."

if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: CONFIG_FILE is not set. The srt-slurm path requires a CONFIG_FILE in additional-settings." >&2
    echo "Config: MODEL_PREFIX=${MODEL_PREFIX} PRECISION=${PRECISION} FRAMEWORK=${FRAMEWORK}" >&2
    exit 1
fi

# CONFIG_FILE may carry an srt-slurm matrix selector such as :zip_override_dep4_dep8[0].
CONFIG_PATH="${CONFIG_FILE%%:*}"
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: CONFIG_FILE does not exist after srt-slurm setup: $CONFIG_PATH" >&2
    exit 1
fi

sed -i "s/^name:.*/name: \"${RUNNER_NAME}\"/" "$CONFIG_PATH"

# Weights live on node-local MODEL_ROOT, which this login host cannot stat, so
# srtctl's preflight model.path check is always skipped. Runtime loading still
# validates the path on the compute nodes.
SRTCTL_APPLY_ARGS=(
    -f "$CONFIG_FILE"
    --no-preflight
    --tags "b300,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)"
)
if [[ "$MODEL_PREFIX" == "glm5.2" && "$FRAMEWORK" == "dynamo-sglang" ]]; then
    SRTCTL_OUTPUT=$(unset UCX_TLS; apply_srt_recipe "$CONFIG_FILE" "$FRAMEWORK" "${SRTCTL_EVAL_ARGS[@]}" "${SRTCTL_APPLY_ARGS[@]}" 2>&1)
else
    SRTCTL_OUTPUT=$(apply_srt_recipe "$CONFIG_FILE" "$FRAMEWORK" "${SRTCTL_EVAL_ARGS[@]}" "${SRTCTL_APPLY_ARGS[@]}" 2>&1)
fi
echo "$SRTCTL_OUTPUT"

JOB_ID=$(echo "$SRTCTL_OUTPUT" | grep -oP '✅ Job \K[0-9]+' || echo "$SRTCTL_OUTPUT" | grep -oP 'Job \K[0-9]+')

set +x

if [ -z "$JOB_ID" ]; then
    echo "Error: Failed to extract JOB_ID from srtctl output"
    exit 1
fi

echo "Extracted JOB_ID: $JOB_ID"

LOGS_DIR="outputs/$JOB_ID/logs"
LOG_FILE="$LOGS_DIR/sweep_${JOB_ID}.log"

SRT_JOB_RC=0
stream_slurm_job_log "$JOB_ID" "$LOG_FILE" || SRT_JOB_RC=$?
verify_slurm_job_status "$JOB_ID" || SRT_JOB_RC=$?

set -x

echo "Job $JOB_ID completed!"
echo "Collecting results..."

if [ ! -d "$LOGS_DIR" ]; then
    echo "Warning: Logs directory not found at $LOGS_DIR"
    exit 1
fi

echo "Found logs directory: $LOGS_DIR"

if [[ "$USES_DCGM_POWER" == "1" ]]; then
    mkdir -p "$LOGS_DIR/power"
    cp "$GITHUB_WORKSPACE/exporter-image.sha256" "$LOGS_DIR/power/exporter-image.sha256"
    cp "$GITHUB_WORKSPACE/power-producer-sha.txt" "$LOGS_DIR/power/power-producer-sha.txt"
fi

cp -r "$LOGS_DIR" "$GITHUB_WORKSPACE/LOGS"
tar czf "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz" -C "$LOGS_DIR" .

if [[ "${EVAL_ONLY}" != "true" ]]; then
    copy_fixed_sequence_results "$LOGS_DIR" "$GITHUB_WORKSPACE" "$RESULT_FILENAME" || exit 1
else
    echo "EVAL_ONLY=true: Skipping benchmark result collection"
fi

if [[ "${RUN_EVAL}" == "true" || "${EVAL_ONLY}" == "true" ]]; then
    EVAL_DIR="$LOGS_DIR/eval_results"
    if [ -d "$EVAL_DIR" ]; then
        echo "Extracting eval results from $EVAL_DIR"
        shopt -s nullglob
        for eval_file in "$EVAL_DIR"/*; do
            [ -f "$eval_file" ] || continue
            cp "$eval_file" "$GITHUB_WORKSPACE/"
            echo "Copied eval artifact: $(basename "$eval_file")"
        done
        shopt -u nullglob
    else
        echo "WARNING: RUN_EVAL=true but no eval results found at $EVAL_DIR"
    fi
fi

# Clean up srt-slurm outputs to prevent NFS silly-rename lock files
# from blocking the next job's checkout on this runner
echo "Cleaning up srt-slurm outputs..."
for i in 1 2 3 4 5; do
    rm -rf outputs 2>/dev/null && break
    echo "Retry $i/5: Waiting for NFS locks to release..."
    sleep 10
done
find . -name '.nfs*' -delete 2>/dev/null || true
# Preserve diagnostics and eval outputs before propagating a failed allocation.
exit "$SRT_JOB_RC"

else
    # AgentX trace datasets need a writable persistent cache. Keep the host and
    # container paths separate so the cache remains valid with
    # --no-container-mount-home.
    check_env_vars B300_HF_CACHE_HOST_DIR
    HF_CACHE_HOST_DIR="${B300_HF_CACHE_HOST_DIR}"
    check_env_vars B300_HF_CACHE_CONTAINER_DIR
    HF_CACHE_CONTAINER_DIR="${B300_HF_CACHE_CONTAINER_DIR}"
    mkdir -p "$HF_CACHE_HOST_DIR/hub" "$HF_CACHE_HOST_DIR/xet"
    export HF_HOME="$HF_CACHE_CONTAINER_DIR"
    export HF_HUB_CACHE="$HF_CACHE_CONTAINER_DIR/hub"
    export HF_XET_CACHE="$HF_CACHE_CONTAINER_DIR/xet"

    # MODEL stays the HF id for the client; MODEL_PATH is where the server reads
    # weights. Only the root holding MODEL_PATH is mounted -- mounting both roots
    # makes pyxis fail whenever the unused one is absent on the node.
    MODEL_BASENAME="${MODEL##*/}"
    if [[ "$MODEL_BASENAME" == "DeepSeek-V4-Pro-0813" ]]; then
        MODEL_MOUNT_DIR="$SHARED_MODEL_ROOT"
    elif [[ " ${STAGED_MODELS[*]} " == *" ${MODEL_BASENAME} "* ]]; then
        MODEL_MOUNT_DIR="$MODEL_ROOT"
    else
        MODEL_MOUNT_DIR="$WRITABLE_MODELS_DIR"
        mkdir -p "$WRITABLE_MODELS_DIR"
    fi
    export MODEL_PATH="${MODEL_MOUNT_DIR}/${MODEL_BASENAME}"

    SQUASH_FILE="$SQUASH_DIR/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" || "$SPEC_DECODING" == "draft_model" ]] && printf '_mtp' || printf '')
    # Prefer a framework-tagged script (dsv4_fp4_b300_sglang.sh) so engines can
    # coexist; fall back to the untagged name for scripts not yet retagged.
    BENCH_BASE="benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_b300"
    BENCH_SCRIPT="${BENCH_BASE}_${FRAMEWORK}${SPEC_SUFFIX}.sh"
    if [[ ! -f "$BENCH_SCRIPT" ]]; then
        LEGACY_FW_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')
        BENCH_SCRIPT="${BENCH_BASE}${LEGACY_FW_SUFFIX}${SPEC_SUFFIX}.sh"
    fi

    # Allow callers (e.g. the speedbench-al.yml AL-collection workflow) to run a
    # specific script instead of the auto-selected throughput benchmark.
    if [[ -n "${BENCH_SCRIPT_OVERRIDE:-}" ]]; then
        BENCH_SCRIPT="$BENCH_SCRIPT_OVERRIDE"
    fi

    # These images install sglang editable under /workspace, so the default
    # workspace bind-mount masks the install and breaks `import sglang`. Mount at
    # /ix instead; drop this once the images stop installing there.
    if [[ "$IMAGE" == *deepseek-v4-blackwell* || "$IMAGE" == *deepseek-v4-bw-ultra* || "$IMAGE" == *deepseek-v4-b300* || "$IMAGE" == *sglang-b300* ]]; then
        CONTAINER_MOUNT_DIR=/ix
    else
        CONTAINER_MOUNT_DIR=/workspace
    fi

    # Keep all new AgentX runtime directories outside /workspace.
    if [[ "$MODEL_PREFIX" == "dsv41flash" && ( "$FRAMEWORK" == "vllm" || "$FRAMEWORK" == "sglang" ) ]]; then
        CONTAINER_MOUNT_DIR=/ix
        export INFMAX_CONTAINER_WORKSPACE=/ix
        export RESULT_DIR=/ix/results
        # Cover DSpark5 verification for concurrent AgentX subagents at c1/c2/c4.
        export DSV41_MIN_CUDAGRAPH_CAPTURE_SIZE=64
    fi

    import_squash_image "$IMAGE" "$SQUASH_FILE"

    check_env_vars GPU_COUNT

    SALLOC_ARGS=(
        --partition="$SLURM_PARTITION"
        --account="$SLURM_ACCOUNT"
        -N 1
        --gres="gpu:$GPU_COUNT"
        --exclusive
        --mem=0
        --time="${SALLOC_TIME_LIMIT}"
        --no-shell
        --job-name="$RUNNER_NAME"
    )
    # Optional escape hatch for taking a bad node out of rotation without a code change.
    if [[ -n "${SALLOC_EXCLUDE:-}" ]]; then
        SALLOC_ARGS+=(--exclude="$SALLOC_EXCLUDE")
    fi
    # Capture this allocation's ID; a runner name can also match an older job.
    if [[ "${B300_AGENTX_BATCH:-}" == 1 ]]; then
        JOB_ID="${SLURM_JOB_ID:?B300 batch execution requires a Slurm allocation}"
    else
        JOB_ID=$(
        set -o pipefail
        LC_ALL=C salloc "${SALLOC_ARGS[@]}" 2>&1 | tee /dev/stderr |
            sed -n 's/.*Granted job allocation \([0-9][0-9]*\)$/\1/p'
        ) || exit 1
        [[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo 'ERROR: B300 allocation unavailable' >&2; exit 1; }
        trap 'rc=$?; scancel "$JOB_ID" 2>/dev/null || true; exit "$rc"' EXIT
    fi
    if [[ "$MODEL_MOUNT_DIR" == "$MODEL_ROOT" ]]; then
        # MODEL_ROOT is node-local: probe the allocated compute node, not the login host.
        srun --jobid="$JOB_ID" test -r "$MODEL_PATH/config.json" || {
            echo 'ERROR: readiness-blocked: staged model config is unavailable on the allocated node' >&2
            exit 1
        }
    fi

    CONTAINER_MOUNTS=(
        "$GITHUB_WORKSPACE:$CONTAINER_MOUNT_DIR"
        "$MODEL_MOUNT_DIR:$MODEL_MOUNT_DIR"
        "$HF_CACHE_HOST_DIR:$HF_CACHE_CONTAINER_DIR"
    )
    if [[ "$MODEL_PREFIX" == "kimik3" && "$FRAMEWORK" == "vllm" && "${IS_AGENTIC}" == "1" ]]; then
        # The pre-staged target is read-only; DSpark needs the writable,
        # persistent model root as a separate mount.
        mkdir -p "$WRITABLE_MODELS_DIR"
        export WRITABLE_MODELS_DIR
        if [[ "$MODEL_MOUNT_DIR" != "$WRITABLE_MODELS_DIR" ]]; then
            CONTAINER_MOUNTS+=("$WRITABLE_MODELS_DIR:$WRITABLE_MODELS_DIR")
        fi
    fi
    CONTAINER_MOUNTS_ARG=$(IFS=,; printf '%s' "${CONTAINER_MOUNTS[*]}")

    B300_CONTAINER_MPI=none
    if [[ "${B300_AGENTX_BATCH:-}" == 1 ]]; then
        # The installed Enroot hook sees PMIx variables in batch jobs. Use the
        # supported plugin so its required per-step mount directories exist.
        B300_CONTAINER_MPI=pmix
    fi
    srun --jobid="$JOB_ID" \
        --mpi="$B300_CONTAINER_MPI" \
        --container-image="$SQUASH_FILE" \
        --container-mounts="$CONTAINER_MOUNTS_ARG" \
        --no-container-mount-home \
        --container-remap-root \
        --container-workdir="$CONTAINER_MOUNT_DIR" \
        --no-container-entrypoint --export=ALL,PORT=8888 \
        bash "$BENCH_SCRIPT"

fi
