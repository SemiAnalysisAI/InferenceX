#!/usr/bin/bash

# Launches multi-node Dynamo + vLLM benchmarks on the gb300-cw (CoreWeave)
# cluster. Mirrors launch_gb200-nv.sh but adjusted for cw's filesystem
# layout: /mnt/vast (10T shared VAST PVC) replaces Lustre/NUMA-local NVMe,
# and the SLURM partition is `all`. cw is 2x 18-node racks; srtctl's
# auto-segment is disabled (use_segment_sbatch_directive: false) and each
# recipe pins its own segment via sbatch_directives — the largest
# topology (14p1d-dep4-dep16, 18 nodes) fills exactly one rack.
#
# srt-slurm is checked out at NVIDIA/srt-slurm PR #84 head; that PR ships
# the dynamo 1.0.2 install path + the vLLM patches the new recipes
# require, so we use upstream's configs/vllm-container-deps.sh and
# configs/patches/* unchanged (no local overlay).

set -x

if [[ $FRAMEWORK == "dynamo-vllm" && $MODEL_PREFIX == "dsv4" && $PRECISION == "fp4" ]]; then
    # Weights staged on the shared VAST mount; no compute-node-local NVMe on cr.
    export MODEL_PATH="/mnt/vast/models/dsv4/"
    export SRT_SLURM_MODEL_PREFIX="deepseek-v4-pro"
else
    echo "Unsupported model prefix/precision/framework combination on gb300-cw: $MODEL_PREFIX/$PRECISION/$FRAMEWORK. Currently supported: dsv4/fp4/dynamo-vllm"
    exit 1
fi

# CoreWeave cluster has a single `all` partition; no separate batch queue.
# Account `cw-sup` is what `sacctmgr show assoc user=$USER` returns on this
# cluster — `benchmark` (inherited from gb200-nv) does not exist here.
export SLURM_PARTITION="all"
export SLURM_ACCOUNT="cw-sup"

# Pyxis/enroot's NVIDIA prestart hook reads these from the runtime env to
# decide which host driver libraries (libcuda.so.1, libnvidia-*.so) to
# mount into the container. cw doesn't set them by default — without them
# the container has no libcuda and `import vllm._C` dies with
# "libcuda.so.1: cannot open shared object file". SLURM's default
# --export=ALL propagates these from this shell through sbatch+srun
# into the enroot environment.
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

NGINX_IMAGE="nginx:1.27.4"

# Squash files live alongside models on /mnt/vast (shared across nodes).
# The deepseekv4-cu130 vLLM image is pre-staged at /mnt/vast/squash_dupe/
# (manual upload — enroot import of the ~25 GB image takes too long to
# repeat each run). nginx is small enough to import on-demand into
# /mnt/vast/squash/.
SQUASH_DIR="/mnt/vast/squash"
mkdir -p "$SQUASH_DIR"
SQUASH_FILE="/mnt/vast/squash_dupe/vllm_vllm-openai_d29a90b13bb9.sqsh"
NGINX_SQUASH_FILE="$SQUASH_DIR/$(echo "$NGINX_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

if [ ! -f "$SQUASH_FILE" ]; then
    echo "ERROR: pre-staged vLLM squash not found at $SQUASH_FILE" >&2
    echo "Re-stage it from docker://$IMAGE or repoint SQUASH_FILE." >&2
    exit 1
fi
enroot import -o $NGINX_SQUASH_FILE docker://$NGINX_IMAGE

export EVAL_ONLY="${EVAL_ONLY:-false}"

export ISL="$ISL"
export OSL="$OSL"

# srt-slurm path requires a CONFIG_FILE pointing to a recipe YAML.
# Without it, srtctl apply scans every YAML in the repo and submits hundreds of jobs.
if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: CONFIG_FILE is not set. The srt-slurm path requires a CONFIG_FILE in additional-settings." >&2
    echo "Config: MODEL_PREFIX=${MODEL_PREFIX} PRECISION=${PRECISION} FRAMEWORK=${FRAMEWORK}" >&2
    exit 1
fi

echo "Cloning srt-slurm repository..."
SRT_REPO_DIR="srt-slurm"
if [ -d "$SRT_REPO_DIR" ]; then
    echo "Removing existing $SRT_REPO_DIR..."
    rm -rf "$SRT_REPO_DIR"
fi

git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
cd "$SRT_REPO_DIR"
# Pin to NVIDIA/srt-slurm PR #84 (ywang96/gb300-vllm) head SHA. PR 84
# carries the configs/patches/* (cumem expandable_segments fix, MegaMoE
# free_orig, nvlink one-sided bf16 fix, numa-bind hash fix) and the
# matching configs/vllm-container-deps.sh that wires them up. Released
# dynamo 1.0.2 wheel + sleep-mode + safetensors prefetch make the
# prebuild infrastructure unnecessary, so we use upstream's setup
# script directly — no overlay.
git fetch origin pull/84/head:pr-84
git checkout 228febcfe9c76347cd619a7622af83ca52ca35a4
# Use `cp -rT` so if the upstream branch ever ships a stub
# `recipes/vllm/deepseek-v4/` directory, we overlay our recipes onto it
# rather than nesting (`cp -r src dst` would create
# `recipes/vllm/deepseek-v4/deepseek-v4/...` in that case).
mkdir -p recipes/vllm/deepseek-v4
cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/deepseek-v4" recipes/vllm/deepseek-v4

# ─── Stage ai-dynamo dev wheels for in-container install ─────────────────────
# Stable ai-dynamo 1.0.2 imports vllm.inputs.data, which vllm-project/vllm#35182
# (2026-03-26) deleted; this container is post-deletion. The fix shipped in
# 1.2.0.dev wheels but PR #84's DynamoConfig has no wheel: field, so we can't
# tell srtctl to install them directly. Instead we set dynamo.install: false in
# every gb300 recipe and append a pip install to the upstream
# configs/vllm-container-deps.sh — which srtctl runs in each worker container
# before launching dynamo.vllm.
#
# Wheels are staged on /mnt/vast (shared aarch64-readable NFS) once per version
# and re-symlinked into srt-slurm/configs/dynamo-wheels/<version>/ each run, so
# they end up at /configs/dynamo-wheels/<version>/ inside the container.
DYNAMO_DEV_VERSION="1.2.0.dev20260426"
DYNAMO_WHEEL_CACHE="/mnt/vast/dynamo-wheels/${DYNAMO_DEV_VERSION}"
DYNAMO_WHEEL_LOCK="/mnt/vast/dynamo-wheels/${DYNAMO_DEV_VERSION}.lock"
mkdir -p "$(dirname "$DYNAMO_WHEEL_CACHE")"

# mkdir-as-lock — flock is unreliable on this VAST mount per prior runs.
# First runner to win the mkdir downloads; others spin until the cache is ready.
if mkdir "$DYNAMO_WHEEL_LOCK" 2>/dev/null; then
    trap 'rmdir "$DYNAMO_WHEEL_LOCK" 2>/dev/null || true' EXIT
    if [ ! -f "${DYNAMO_WHEEL_CACHE}/.complete" ]; then
        echo "Staging ai-dynamo ${DYNAMO_DEV_VERSION} aarch64 wheels to ${DYNAMO_WHEEL_CACHE}..."
        rm -rf "$DYNAMO_WHEEL_CACHE"
        mkdir -p "$DYNAMO_WHEEL_CACHE"
        # Runner pod is x86 but compute nodes are aarch64, so force --platform.
        # --pre is required: dev wheels aren't picked up otherwise.
        python3 -m pip download \
            --no-deps --pre --only-binary=:all: \
            --implementation cp --python-version 3.12 \
            --platform manylinux_2_28_aarch64 --platform manylinux2014_aarch64 \
            --extra-index-url https://pypi.nvidia.com \
            --dest "$DYNAMO_WHEEL_CACHE" \
            "ai-dynamo-runtime==${DYNAMO_DEV_VERSION}" \
            "ai-dynamo==${DYNAMO_DEV_VERSION}"
        touch "${DYNAMO_WHEEL_CACHE}/.complete"
    fi
    trap - EXIT
    rmdir "$DYNAMO_WHEEL_LOCK" 2>/dev/null || true
else
    echo "Another runner is staging ai-dynamo ${DYNAMO_DEV_VERSION}; waiting..."
    while [ -d "$DYNAMO_WHEEL_LOCK" ] && [ ! -f "${DYNAMO_WHEEL_CACHE}/.complete" ]; do
        sleep 5
    done
fi

if [ ! -f "${DYNAMO_WHEEL_CACHE}/.complete" ]; then
    echo "ERROR: ai-dynamo wheel cache at ${DYNAMO_WHEEL_CACHE} is missing the .complete marker — staging failed." >&2
    exit 1
fi

# Surface the cache to the container at /configs/dynamo-wheels/<version>/. Must be a
# real directory copy, not a symlink — only srt-slurm/configs is bind-mounted
# into the worker container, so a symlink whose target is /mnt/vast/... dangles
# from inside the container and pip's --find-links silently turns into "no such
# location" (which then falls through to a network lookup that --no-index blocks).
mkdir -p configs/dynamo-wheels
rm -rf "configs/dynamo-wheels/${DYNAMO_DEV_VERSION}"
mkdir -p "configs/dynamo-wheels/${DYNAMO_DEV_VERSION}"
cp "${DYNAMO_WHEEL_CACHE}"/*.whl "configs/dynamo-wheels/${DYNAMO_DEV_VERSION}/"

# Append the pip install to upstream's vllm-container-deps.sh. The recipe sets
# dynamo.install: false so srtctl emits no install line; this hook installs
# from the wheel cache instead. --no-index keeps pip off the network entirely.
# Drop --quiet and fail loudly if pip exits non-zero — upstream's script has
# no `set -e`, so silent failure here propagates as "module not found" much
# later when the worker tries `python3 -m dynamo.vllm`.
cat >> configs/vllm-container-deps.sh <<EOF

# ─── Local dynamo install (added by runners/launch_gb300-cw.sh) ─────────────
# Install ai-dynamo ${DYNAMO_DEV_VERSION} from staged wheels. We do this here
# rather than via srtctl's DynamoConfig.get_install_commands because PR #84's
# schema only knows about \`version\` (PyPI) and \`hash\`/\`top_of_tree\` (source
# build). The 1.2.0.dev wheels are pre-release on pypi.nvidia.com; --no-index
# forces pip off the network so worker startup is offline-deterministic.
echo "Installing ai-dynamo ${DYNAMO_DEV_VERSION} from /configs/dynamo-wheels/${DYNAMO_DEV_VERSION}/..."
ls -la "/configs/dynamo-wheels/${DYNAMO_DEV_VERSION}/" >&2
if ! pip install --break-system-packages --no-deps --no-index \\
    --find-links "/configs/dynamo-wheels/${DYNAMO_DEV_VERSION}/" \\
    "ai-dynamo-runtime==${DYNAMO_DEV_VERSION}" \\
    "ai-dynamo==${DYNAMO_DEV_VERSION}"; then
    echo "ERROR: ai-dynamo ${DYNAMO_DEV_VERSION} install from /configs/dynamo-wheels/ failed" >&2
    exit 1
fi
echo "ai-dynamo ${DYNAMO_DEV_VERSION} installed"
EOF

echo "Installing srtctl..."
# CRITICAL — uv install location.
# Runner pod is x86 but compute nodes are aarch64, and /mnt/home is shared
# NFS across both. srtctl's slurm template (job_script_minimal.j2) does
# `if ! command -v uv` and skips its own ARM64 install when uv is already
# on PATH; on compute nodes $HOME/.local/bin is on PATH by default, so a
# stray x86 binary at $HOME/.local/bin/uv from this runner shadows the
# template's install and crashes the orchestrator with
# `cannot execute binary file: Exec format error`. Install to a
# runner-pod-local /tmp path (tmpfs, not NFS) and scrub any stale x86
# uv left in the shared path by prior runs.
rm -f "$HOME/.local/bin/uv" "$HOME/.local/bin/uvx"
export XDG_BIN_HOME="/tmp/uv-runner-${RUNNER_NAME:-default}/bin"
mkdir -p "$XDG_BIN_HOME"
curl -LsSf https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 sh
export PATH="$XDG_BIN_HOME:$PATH"

# Sanity: confirm the install landed where we expect, not in $HOME/.local/bin.
if [ ! -x "$XDG_BIN_HOME/uv" ]; then
    echo "ERROR: uv not at $XDG_BIN_HOME/uv after install — install script may not honor XDG_BIN_HOME on this version. Aborting before x86 uv leaks onto NFS." >&2
    exit 1
fi
if [ -e "$HOME/.local/bin/uv" ]; then
    echo "ERROR: uv install leaked to shared $HOME/.local/bin/uv. Remove it and re-run." >&2
    exit 1
fi

uv venv
source .venv/bin/activate
uv pip install -e .

if ! command -v srtctl &> /dev/null; then
    echo "Error: Failed to install srtctl"
    exit 1
fi

echo "Configs available at: $SRT_REPO_DIR/"

# Create srtslurm.yaml for srtctl
SRTCTL_ROOT="${GITHUB_WORKSPACE}/srt-slurm"
echo "Creating srtslurm.yaml configuration..."
cat > srtslurm.yaml <<EOF
# SRT SLURM Configuration for GB300-CW

# Default SLURM settings
default_account: "${SLURM_ACCOUNT}"
default_partition: "${SLURM_PARTITION}"
default_time_limit: "6:00:00"

# Resource defaults
gpus_per_node: 4
network_interface: ""

# Path to srtctl repo root (where the configs live)
srtctl_root: "${SRTCTL_ROOT}"

# Model path aliases
model_paths:
  "${SRT_SLURM_MODEL_PREFIX}": "${MODEL_PATH}"
containers:
  dynamo-trtllm: ${SQUASH_FILE}
  dynamo-sglang: ${SQUASH_FILE}
  "${IMAGE}": ${SQUASH_FILE}
  nginx-sqsh: ${NGINX_SQUASH_FILE}
# Auto-emission of #SBATCH --segment={total_nodes} is turned off here
# because each gb300 recipe sets its own segment via sbatch_directives.
# (Avoid backticks in this comment — heredoc is unquoted, so backtick
# content would be command-substituted by bash and produce noisy errors.)
use_segment_sbatch_directive: false
EOF

echo "Generated srtslurm.yaml:"
cat srtslurm.yaml

echo "Running make setup..."
make setup ARCH=aarch64

# Export eval-related env vars for srt-slurm post-benchmark eval
export INFMAX_WORKSPACE="$GITHUB_WORKSPACE"

echo "Submitting job with srtctl..."

# Override the job name in the config file with the runner name
sed -i "s/^name:.*/name: \"${RUNNER_NAME}\"/" "$CONFIG_FILE"

SRTCTL_OUTPUT=$(srtctl apply -f "$CONFIG_FILE" --tags "gb300,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)" 2>&1)
echo "$SRTCTL_OUTPUT"

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

if [ -d "$LOGS_DIR" ]; then
    echo "Found logs directory: $LOGS_DIR"
    cp -r "$LOGS_DIR" "$GITHUB_WORKSPACE/LOGS"
    tar czf "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz" -C "$LOGS_DIR" .
else
    echo "Warning: Logs directory not found at $LOGS_DIR"
fi

if [[ "${EVAL_ONLY:-false}" != "true" ]]; then
    if [ ! -d "$LOGS_DIR" ]; then
        exit 1
    fi

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
                    # Files are of the format "results_concurrency_gpus_{num gpus}_ctx_{num ctx}_gen_{num gen}.json"
                    filename=$(basename "$result_file")
                    concurrency=$(echo "$filename" | sed -n 's/results_concurrency_\([0-9]*\)_gpus_.*/\1/p')
                    gpus=$(echo "$filename" | sed -n 's/results_concurrency_[0-9]*_gpus_\([0-9]*\)_ctx_.*/\1/p')
                    ctx=$(echo "$filename" | sed -n 's/.*_ctx_\([0-9]*\)_gen_.*/\1/p')
                    gen=$(echo "$filename" | sed -n 's/.*_gen_\([0-9]*\)\.json/\1/p')

                    echo "Processing concurrency $concurrency with $gpus GPUs (ctx: $ctx, gen: $gen): $result_file"

                    WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${CONFIG_NAME}_conc${concurrency}_gpus_${gpus}_ctx_${ctx}_gen_${gen}.json"
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
