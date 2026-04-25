#!/usr/bin/bash

# Launches multi-node Dynamo + vLLM benchmarks on the gb300-cw (CoreWeave)
# cluster. Mirrors launch_gb200-nv.sh but adjusted for cr's filesystem
# layout: /mnt/vast (10T shared VAST PVC) replaces Lustre/NUMA-local NVMe,
# the SLURM partition is `all`, and srtctl auto-emits `--segment={total_nodes}`
# to keep each job rack-local (cr is 2x18-node racks, so any of our recipes
# at ≤18 nodes fits within a single rack).

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
SQUASH_DIR="/mnt/vast/squash"
mkdir -p "$SQUASH_DIR"
SQUASH_FILE="$SQUASH_DIR/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
NGINX_SQUASH_FILE="$SQUASH_DIR/$(echo "$NGINX_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

enroot import -o $SQUASH_FILE docker://$IMAGE
enroot import -o $NGINX_SQUASH_FILE docker://$NGINX_IMAGE

# Pre-build dynamo wheel ONCE on a single compute node, BEFORE submitting
# the main sbatch. The DP+EP path inside sbatch spawns one container per
# GPU (~60 ranks for the 18-node 7p1d topology), and trying to coordinate
# a one-time build across that many containers via filesystem locks is
# unreliable on /mnt/vast (NFS) — flock silently no-ops, mkdir caches
# negatively, etc. Building once here on a dedicated single-node srun
# eliminates all per-rank coordination: every worker just pip-installs
# from the cache (~30 s) and the timing across ranks stays tight.
DYNAMO_HASH="6a159fedd8e4a1563aa647c31f622aedbf254b5b"
DYNAMO_CACHE_ROOT="/mnt/vast/dynamo_cache"
DYNAMO_CACHE_DIR="$DYNAMO_CACHE_ROOT/$DYNAMO_HASH"
DYNAMO_DONE_MARKER="$DYNAMO_CACHE_DIR/.done"
mkdir -p "$DYNAMO_CACHE_ROOT"

if [ ! -f "$DYNAMO_DONE_MARKER" ]; then
    echo "[dynamo-prebuild] cold cache, building wheel + source archive on a single compute node..."
    # Build into a unique temp dir, then atomically mv into place. Two
    # concurrent runners may both build; the first to finish the rename
    # wins, the loser cleans up. Same-directory rename() is atomic on
    # NFS (unlike flock).
    TEMP_BUILD=$(mktemp -d "$DYNAMO_CACHE_ROOT/$DYNAMO_HASH.tmp.XXXXXX")
    # --mem=0: claim full node memory. Default cgroup is much smaller and
    # the moxcms / dynamo-llm rustc invocations OOM-killed the previous
    # attempt. CARGO_BUILD_JOBS=8 caps parallelism so peak rustc memory
    # stays bounded even on a 72-core Grace node, and `-C debuginfo=0`
    # cuts per-process memory further (default debuginfo=2 from cargo
    # is what makes the link phase memory-hungry).
    srun --partition=$SLURM_PARTITION --account=$SLURM_ACCOUNT \
         --nodes=1 --ntasks=1 --mem=0 --time=00:45:00 \
         --job-name="${RUNNER_NAME}-prebuild" \
         --container-image="$SQUASH_FILE" \
         --no-container-entrypoint --no-container-mount-home \
         --container-mounts="$DYNAMO_CACHE_ROOT:$DYNAMO_CACHE_ROOT" \
         bash -c "
            set -e
            apt-get update -qq
            apt-get install -y -qq git curl libclang-dev protobuf-compiler >/dev/null 2>&1
            if ! command -v cargo &>/dev/null; then
              curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
              . \$HOME/.cargo/env
            fi
            if ! command -v maturin &>/dev/null; then
              pip install --break-system-packages maturin
            fi
            rm -rf /tmp/dynamo_build
            mkdir -p /tmp/dynamo_build
            cd /tmp/dynamo_build
            git clone https://github.com/ai-dynamo/dynamo.git
            cd dynamo
            git checkout $DYNAMO_HASH
            cd lib/bindings/python/
            export CARGO_BUILD_JOBS=8
            export RUSTFLAGS='-C target-cpu=native -C debuginfo=0 --cfg tokio_unstable'
            maturin build -o '$TEMP_BUILD'
            cd /tmp/dynamo_build/dynamo
            tar czf '$TEMP_BUILD/dynamo-source.tar.gz' \
                --exclude='lib/bindings/python/target' \
                --exclude='.git' \
                .
            touch '$TEMP_BUILD/.done'
        "
    if [ -f "$TEMP_BUILD/.done" ]; then
        # Atomic publish. If another runner already published, mv fails
        # and we just discard our copy.
        if mv "$TEMP_BUILD" "$DYNAMO_CACHE_DIR" 2>/dev/null; then
            echo "[dynamo-prebuild] published cache at $DYNAMO_CACHE_DIR"
        else
            echo "[dynamo-prebuild] another runner published first, discarding our copy"
            rm -rf "$TEMP_BUILD"
        fi
    else
        echo "[dynamo-prebuild] BUILD FAILED — no .done in $TEMP_BUILD" >&2
        rm -rf "$TEMP_BUILD"
        exit 1
    fi
else
    echo "[dynamo-prebuild] cache hit at $DYNAMO_CACHE_DIR"
fi

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
git checkout sa-submission-q2-2026
# Use `cp -rT` so if the upstream branch ever ships a stub
# `recipes/vllm/deepseek-v4/` directory, we overlay our recipes onto it
# rather than nesting (`cp -r src dst` would create
# `recipes/vllm/deepseek-v4/deepseek-v4/...` in that case).
mkdir -p recipes/vllm/deepseek-v4
cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/deepseek-v4" recipes/vllm/deepseek-v4

# Replace the upstream stub setup script with our flock-cached dynamo
# installer. See runners/gb300-cw-vllm-container-deps.sh for why. Used
# together with `dynamo.install: false` in the gb300 recipes.
cp "$GITHUB_WORKSPACE/runners/gb300-cw-vllm-container-deps.sh" configs/vllm-container-deps.sh
chmod +x configs/vllm-container-deps.sh

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
