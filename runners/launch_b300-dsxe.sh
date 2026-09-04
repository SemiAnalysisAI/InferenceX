#!/usr/bin/bash

# Launcher for the B300 DSXE Slurm cluster (dsxe-sa-b300-prd0), runners run as sa-gha-runner.
#
# Every cluster-specific fact lives in this block. The rest of the file is generic:
# multi-node jobs go through srt-slurm/srtctl, single-node jobs through salloc + pyxis.

SLURM_PARTITION="batch_1"
SLURM_ACCOUNT="benchmark"

# enroot squash images. Must be on storage every compute node mounts and writable
# by the runner user (/data/squash is root-owned, hence the per-user default).
SQUASH_DIR="/data/home/sa-gha-runner/squash"

# Weights. MODEL_ROOT is node-local NVMe with the same layout on every compute node;
# it is read-only from the job's point of view. Anything not in STAGED_MODELS is
# downloaded into WRITABLE_MODELS_DIR (shared Lustre) by the single-node scripts.
MODEL_ROOT="/scratch/models"
WRITABLE_MODELS_DIR="/data/home/sa-gha-runner/models"

# Official power (dcgm-power) runs use a separate, pinned producer; CI derives
# POWER_PRODUCER_SHA from the stamp this script writes. Keep in sync with the other launchers.
POWER_SRT_SLURM_URL="https://github.com/edwingao28/srt-slurm.git"
POWER_SRT_SLURM_PIN="e5c837f06a362dc888dfea2ee588e9f19c298270"

# Directory names under MODEL_ROOT (upstream HF repo basenames).
STAGED_MODELS=(
    DeepSeek-R1-0528
    DeepSeek-R1-0528-NVFP4-v2
    DeepSeek-V4-Pro
    DeepSeek-V4-Pro-0813
    DeepSeek-V4-Pro-NVFP4
    GLM-5.2-NVFP4
    Kimi-K2.6-NVFP4
    Kimi-K3
    MiniMax-M3
    MiniMax-M3-MXFP8
    MiniMax-M3-NVFP4
    Qwen3.5-397B-A17B-FP8
    Qwen3.5-397B-A17B-NVFP4
    Qwen3.5-397B-A17B-NVFP4-V2
    Qwen3.8-2.4T-A95B-FP8
)

# srt-slurm recipes refer to models by alias (model.path in the recipe yaml). Every
# alias below is written into srtslurm.yaml, so no per-model branching is needed.
# Several aliases map to the same directory because recipes are not consistent.
declare -A MODEL_ALIASES=(
    [dsr1]="DeepSeek-R1-0528-NVFP4-v2"
    [dsr1-fp8]="DeepSeek-R1-0528"
    [deepseek-v4-pro]="DeepSeek-V4-Pro"
    [deepseek-ai/DeepSeek-V4-Pro]="DeepSeek-V4-Pro"
    [glm-5.2-fp4]="GLM-5.2-NVFP4"
    [nvidia/GLM-5.2-NVFP4]="GLM-5.2-NVFP4"
    [kimi-k2.6-nvfp4]="Kimi-K2.6-NVFP4"
    [kimi-k3]="Kimi-K3"
    [kimik3]="Kimi-K3"
    [moonshotai/Kimi-K3]="Kimi-K3"
    [minimax-m3-nvfp4]="MiniMax-M3-NVFP4"
    [nvidia/MiniMax-M3-NVFP4]="MiniMax-M3-NVFP4"
    [minimax-m3-mxfp8]="MiniMax-M3-MXFP8"
    [MiniMaxAI/MiniMax-M3-MXFP8]="MiniMax-M3-MXFP8"
    [qwen3.5-fp4]="Qwen3.5-397B-A17B-NVFP4-V2"
    [qwen3.5-fp8]="Qwen3.5-397B-A17B-FP8"
    [nvidia/Qwen3.5-397B-A17B-NVFP4-V2]="Qwen3.5-397B-A17B-NVFP4-V2"
)


mkdir -p "$SQUASH_DIR"
set -x

# !! KEEP THIS DEFINITION ABOVE THE IS_MULTINODE BRANCH BELOW. !!
# Both the multi-node and single-node paths call it. Bash only defines a function
# when execution reaches it, so moving this inside either branch silently removes
# it from the other and the job dies on "command not found" at import time.
#
# Import a container image into the shared squash dir. Concurrent callers target the
# same path, so serialize on a per-file lock and skip when a valid squash file exists.
# --time bounds the step; an unbounded srun hangs the job if its step is lost.
#
# The import itself must run on a compute node: enroot builds the squashfs over an
# overlay mount, which the shared filesystem cannot back, and the login host is too
# small to unpack a multi-GB image. Reading the finished file is just I/O, so probe
# it here first -- a warm cache then costs no Slurm allocation at all. The in-srun
# check under the lock stays authoritative, so a stale probe only costs one step.
import_squash_image() {
    local image_ref="$1"
    local sqsh="$2"
    local lock="${2}.lock"

    if unsquashfs -l "$sqsh" > /dev/null 2>&1; then
        echo "Squash file already present, skipping import: $sqsh"
        return 0
    fi

    srun -N 1 -A "$SLURM_ACCOUNT" -p "$SLURM_PARTITION" \
        --time="${ENROOT_IMPORT_TIME_LIMIT:-120}" bash -c "
        set -euo pipefail
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

# Validate framework
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
    t && /^  provider: dcgm-power$/ { p = 1 }
    t && /^  enabled: true$/        { e = 1 }
    END { exit !(p && e) }
' "$_RECIPE_SRC"; then
    USES_DCGM_POWER=1
fi
if [[ "$USES_DCGM_POWER" == "1" && (
    "${IS_AGENTIC:-0}" == "1" ||
    "$MODEL_PREFIX" != "dsv4" ||
    "$PRECISION" != "fp4" ||
    ( "$FRAMEWORK" != "dynamo-sglang" && "$FRAMEWORK" != "dynamo-vllm" )
) ]]; then
    echo "Error: B300 dcgm-power is limited to fixed-sequence DSV4 FP4 dynamo-sglang/vllm" >&2
    exit 1
fi

# Default is the newest tag. Add a branch here to pin a ref per model / precision /
# framework when a recipe needs one, so results stay reproducible.
select_srt_slurm_version() {
    if false; then
        :
    else
        SRT_SLURM_REPO="https://github.com/NVIDIA/srt-slurm.git"
        SRT_SLURM_REF="v1.0.87"
    fi
}

# ---------------------------------------------------------------------------
# srt-slurm checkout: one clone at the selected ref, plus every in-repo recipe.
# ---------------------------------------------------------------------------
SRT_REPO_DIR="srt-slurm"
rm -rf "$SRT_REPO_DIR"

if [[ "$USES_DCGM_POWER" == "1" ]]; then
    SRT_SLURM_REPO="$POWER_SRT_SLURM_URL"
    SRT_SLURM_REF="$POWER_SRT_SLURM_PIN"
else
    select_srt_slurm_version
fi

echo "Cloning srt-slurm ($SRT_SLURM_REPO @ $SRT_SLURM_REF)..."
git clone "$SRT_SLURM_REPO" "$SRT_REPO_DIR" || exit 1
cd "$SRT_REPO_DIR" || exit 1
git checkout --quiet "$SRT_SLURM_REF" || exit 1
git rev-parse HEAD > "$GITHUB_WORKSPACE/srt-slurm-sha.txt"

# Dynamo installation enables enroot root remapping by default. Disable it for
# DSV4 TensorRT-LLM MPI workers so the container can use the host PMIx socket.
if [[ $FRAMEWORK == "dynamo-trt" && $MODEL_PREFIX == "dsv4" && $PRECISION == "fp4" ]]; then
    grep -Fq 'CONTAINER_REMAP_ROOT_EXPORT = {"ENROOT_REMAP_ROOT": "yes"}' \
        src/srtctl/core/slurm.py || exit 1
    sed -i 's/CONTAINER_REMAP_ROOT_EXPORT = {"ENROOT_REMAP_ROOT": "yes"}/CONTAINER_REMAP_ROOT_EXPORT = {"ENROOT_REMAP_ROOT": "no"}/' \
        src/srtctl/core/slurm.py
    grep -Fq 'CONTAINER_REMAP_ROOT_EXPORT = {"ENROOT_REMAP_ROOT": "no"}' \
        src/srtctl/core/slurm.py || exit 1
fi

if [[ "$USES_DCGM_POWER" == "1" ]]; then
    test "$(git rev-parse HEAD)" = "$POWER_SRT_SLURM_PIN" \
        || { echo "Error: srt-slurm HEAD does not match POWER_SRT_SLURM_PIN=$POWER_SRT_SLURM_PIN" >&2; exit 1; }
    cp "$GITHUB_WORKSPACE/srt-slurm-sha.txt" "$GITHUB_WORKSPACE/power-producer-sha.txt"
fi

# Recipes live in this repo; overlay all of them onto the checkout's recipes/ dir.
mkdir -p recipes
cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes" recipes || exit 1

if [[ "${EVAL_FRAMEWORK:-lm-eval}" != "lm-eval" ]]; then
    python3 "$GITHUB_WORKSPACE/runners/patch_srt_eval_dispatch.py" "$(pwd)" || exit 1
fi

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

# Map container images to local squash files
NGINX_IMAGE="nginx:1.27.4"
SQUASH_FILE="$SQUASH_DIR/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
NGINX_SQUASH_FILE="$SQUASH_DIR/$(echo "$NGINX_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

# Import containers via enroot
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
export EVAL_ONLY="${EVAL_ONLY:-false}"

# ---------------------------------------------------------------------------
# srtslurm.yaml: cluster defaults, every model alias, container aliases.
# ---------------------------------------------------------------------------
SRTCTL_ROOT="${GITHUB_WORKSPACE}/${SRT_REPO_DIR}"
echo "Creating srtslurm.yaml configuration..."
{
    cat <<EOF
# SRT SLURM Configuration for B300 DSXE (generated by launch_b300-dsxe.sh)
default_account: "${SLURM_ACCOUNT}"
default_partition: "${SLURM_PARTITION}"
gpus_per_node: 8
network_interface: ""
srtctl_root: "${SRTCTL_ROOT}"
model_paths:
EOF
    for alias in "${!MODEL_ALIASES[@]}"; do
        printf '  "%s": "%s/%s"\n' "$alias" "$MODEL_ROOT" "${MODEL_ALIASES[$alias]}"
    done | sort
    cat <<EOF
containers:
  dynamo-trtllm: "${SQUASH_FILE}"
  dynamo-sglang: "${SQUASH_FILE}"
  dynamo-vllm: "${SQUASH_FILE}"
  "${IMAGE}": "${SQUASH_FILE}"
  nginx-sqsh: "${NGINX_SQUASH_FILE}"
EOF
    if [[ "$USES_DCGM_POWER" == "1" ]]; then
        printf '  dcgm-exporter: "%s"\n' "$DCGM_EXPORTER_SQSH"
    fi
    echo "use_exclusive_sbatch_directive: true"
} > srtslurm.yaml

echo "Generated srtslurm.yaml:"
cat srtslurm.yaml

echo "Running make setup..."
make setup ARCH=x86_64

# Export eval-related env vars for srt-slurm post-benchmark eval
export INFMAX_WORKSPACE="$GITHUB_WORKSPACE"

echo "Submitting job with srtctl..."

if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: CONFIG_FILE is not set. The srt-slurm path requires a CONFIG_FILE in additional-settings." >&2
    echo "Config: MODEL_PREFIX=${MODEL_PREFIX} PRECISION=${PRECISION} FRAMEWORK=${FRAMEWORK}" >&2
    exit 1
fi

# Resolve the recipe path before editing it. CONFIG_FILE may include an
# srt-slurm matrix selector such as :zip_override_dep4_dep8[0].
CONFIG_PATH="${CONFIG_FILE%%:*}"
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: CONFIG_FILE does not exist after srt-slurm setup: $CONFIG_PATH" >&2
    exit 1
fi

# Override the job name in the recipe with the runner name.
sed -i "s/^name:.*/name: \"${RUNNER_NAME}\"/" "$CONFIG_PATH"
if [[ "${EVAL_ONLY:-false}" == "true" ]]; then
    python3 "$GITHUB_WORKSPACE/runners/inject_synthetic_acceptance.py" \
        "$CONFIG_PATH" "$FRAMEWORK" || exit 1
fi

# Weights live on node-local MODEL_ROOT, which this login host cannot stat, so
# srtctl's preflight model.path check is always skipped. Runtime loading still
# validates the path on the compute nodes.
SRTCTL_APPLY_ARGS=(
    -f "$CONFIG_FILE"
    --no-preflight
    --tags "b300,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)"
)
SRTCTL_OUTPUT=$(srtctl apply "${SRTCTL_APPLY_ARGS[@]}" 2>&1)
echo "$SRTCTL_OUTPUT"

# Extract JOB_ID from srtctl output
JOB_ID=$(echo "$SRTCTL_OUTPUT" | grep -oP '✅ Job \K[0-9]+' || echo "$SRTCTL_OUTPUT" | grep -oP 'Job \K[0-9]+')

set +x

if [ -z "$JOB_ID" ]; then
    echo "Error: Failed to extract JOB_ID from srtctl output"
    exit 1
fi

echo "Extracted JOB_ID: $JOB_ID"

# Use the JOB_ID to find the logs directory
# srtctl creates logs in outputs/JOB_ID/logs/
LOGS_DIR="outputs/$JOB_ID/logs"
LOG_FILE="$LOGS_DIR/sweep_${JOB_ID}.log"

# Wait for log file to appear (also check job is still alive)
while ! ls "$LOG_FILE" &>/dev/null; do
    if ! squeue -j "$JOB_ID" --noheader 2>/dev/null | grep -q "$JOB_ID"; then
        echo "ERROR: Job $JOB_ID failed before creating log file"
        scontrol show job "$JOB_ID"
        exit 1
    fi
    echo "Waiting for JOB_ID $JOB_ID to begin and $LOG_FILE to appear..."
    sleep 5
done

# Poll for job completion in background
(
    while squeue -j "$JOB_ID" --noheader 2>/dev/null | grep -q "$JOB_ID"; do
        sleep 10
    done
) &
POLL_PID=$!

echo "Tailing LOG_FILE: $LOG_FILE"

# Stream the log file until job completes (-F follows by name, polls instead of inotify for NFS)
tail -F -s 2 -n+1 "$LOG_FILE" --pid=$POLL_PID 2>/dev/null

wait $POLL_PID

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

if [[ "${EVAL_ONLY:-false}" != "true" ]]; then
    # Find all result subdirectories
    RESULT_SUBDIRS=$(find "$LOGS_DIR" -maxdepth 1 -type d -name "*isl*osl*" 2>/dev/null)

    if [ -z "$RESULT_SUBDIRS" ]; then
        echo "Warning: No result subdirectories found in $LOGS_DIR"
    else
        # Process results from all configurations
        for result_subdir in $RESULT_SUBDIRS; do
            echo "Processing result subdirectory: $result_subdir"

            # Extract configuration info from directory name
            CONFIG_NAME=$(basename "$result_subdir")

            # Find all result JSON files
            RESULT_FILES=$(find "$result_subdir" -name "results_concurrency_*.json" 2>/dev/null)

            for result_file in $RESULT_FILES; do
                if [ -f "$result_file" ]; then
                    # Extract metadata from filename
                    # Files may be "results_concurrency_N_gpus_G_ctx_C_gen_D.json" (disagg) or "results_concurrency_N_gpus_G.json" (non-disagg)
                    filename=$(basename "$result_file")
                    concurrency=$(echo "$filename" | sed -n 's/results_concurrency_\([0-9]*\)_gpus_.*/\1/p')
                    gpus=$(echo "$filename" | sed -n 's/results_concurrency_[0-9]*_gpus_\([0-9][0-9]*\).*/\1/p')
                    ctx=$(echo "$filename" | sed -n 's/.*_ctx_\([0-9]*\)_gen_.*/\1/p')
                    gen=$(echo "$filename" | sed -n 's/.*_gen_\([0-9]*\)\.json/\1/p')

                    echo "Processing concurrency $concurrency with $gpus GPUs (ctx: $ctx, gen: $gen): $result_file"

                    if [ -n "$ctx" ] && [ -n "$gen" ]; then
                        WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${CONFIG_NAME}_conc${concurrency}_gpus_${gpus}_ctx_${ctx}_gen_${gen}.json"
                    else
                        WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${CONFIG_NAME}_conc${concurrency}_gpus_${gpus}.json"
                    fi
                    cp "$result_file" "$WORKSPACE_RESULT_FILE"

                    echo "Copied result file to: $WORKSPACE_RESULT_FILE"
                fi
            done
        done
    fi

    echo "All result files processed"
else
    echo "EVAL_ONLY=true: Skipping benchmark result collection"
fi

# Collect eval results if eval was requested
if [[ "${RUN_EVAL:-false}" == "true" || "${EVAL_ONLY:-false}" == "true" ]]; then
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

else
    # HF_HUB_CACHE is set to help with dataset download inside the container
    # for eval jobs.
    export HF_HUB_CACHE="$HOME/.cache/huggingface"

    # MODEL stays the HF id for the client; MODEL_PATH is where the server reads
    # weights. Only the root holding MODEL_PATH is mounted -- mounting both roots
    # makes pyxis fail whenever the unused one is absent on the node.
    MODEL_BASENAME="${MODEL##*/}"
    if [[ " ${STAGED_MODELS[*]} " == *" ${MODEL_BASENAME} "* ]]; then
        MODEL_MOUNT_DIR="$MODEL_ROOT"
    else
        MODEL_MOUNT_DIR="$WRITABLE_MODELS_DIR"
        mkdir -p "$WRITABLE_MODELS_DIR"
    fi
    export MODEL_PATH="${MODEL_MOUNT_DIR}/${MODEL_BASENAME}"

    SQUASH_FILE="$SQUASH_DIR/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')
    # Prefer a framework-tagged script (e.g. dsv4_fp4_b300_sglang.sh); fall back to
    # the untagged historical name for scripts that haven't been retagged yet.
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

    import_squash_image "$IMAGE" "$SQUASH_FILE"

    export GPU_COUNT="${GPU_COUNT:-${TP:?TP must be set}}"

    SALLOC_ARGS=(
        --partition="$SLURM_PARTITION"
        --account="$SLURM_ACCOUNT"
        -N 1
        --gres="gpu:$GPU_COUNT"
        --exclusive
        --mem=0
        --time="${SALLOC_TIME_LIMIT:-480}"
        --no-shell
        --job-name="$RUNNER_NAME"
    )
    # Optional escape hatch for taking a bad node out of rotation without a code change.
    if [[ -n "${SALLOC_EXCLUDE:-}" ]]; then
        SALLOC_ARGS+=(--exclude="$SALLOC_EXCLUDE")
    fi
    salloc "${SALLOC_ARGS[@]}"
    JOB_ID=$(squeue --name="$RUNNER_NAME" -u "$USER" -h -o %A | head -n1)

    CONTAINER_MOUNTS=(
        "$GITHUB_WORKSPACE:$CONTAINER_MOUNT_DIR"
        "$MODEL_MOUNT_DIR:$MODEL_MOUNT_DIR"
    )
    CONTAINER_MOUNTS_ARG=$(IFS=,; printf '%s' "${CONTAINER_MOUNTS[*]}")

    srun --jobid="$JOB_ID" \
        --mpi=none \
        --container-image="$SQUASH_FILE" \
        --container-mounts="$CONTAINER_MOUNTS_ARG" \
        --no-container-mount-home \
        --container-remap-root \
        --container-workdir="$CONTAINER_MOUNT_DIR" \
        --no-container-entrypoint --export=ALL,PORT=8888 \
        bash "$BENCH_SCRIPT"

fi
