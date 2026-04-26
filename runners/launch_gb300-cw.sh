#!/usr/bin/bash

# Launches multi-node Dynamo + SGLang benchmarks on the gb300-cw
# (CoreWeave) cluster. Adapted from the dynamo-vllm sibling launcher in
# the dsv4-fp4-gb300-dynamo-vllm-disagg branch (PR #1150). Compared to
# that script, the SGLang flow is simpler: no dynamo wheel prebuild and
# no vllm-container-deps.sh override, because the SGLang recipes pin
# `dynamo.version: 0.8.1` and srtctl pip-installs from PyPI per rank.

set -x

if [[ $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "dsv4" && $PRECISION == "fp4" ]]; then
    # Weights staged on the shared VAST mount; no compute-node-local
    # NVMe on cw. SRT_SLURM_MODEL_PREFIX matches the model.path alias in
    # benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4/.
    export MODEL_PATH="/mnt/vast/models/dsv4/"
    export SRT_SLURM_MODEL_PREFIX="deepseek-v4-pro"
else
    echo "Unsupported model prefix/precision/framework combination on gb300-cw: $MODEL_PREFIX/$PRECISION/$FRAMEWORK. Currently supported: dsv4/fp4/dynamo-sglang"
    exit 1
fi

# CoreWeave cluster has a single `all` partition; account `cw-sup` is
# what `sacctmgr show assoc user=$USER` returns there. `benchmark`
# (inherited from gb200-nv) does not exist on cw.
export SLURM_PARTITION="all"
export SLURM_ACCOUNT="cw-sup"

# Pyxis/enroot's NVIDIA prestart hook reads these from the runtime env
# to decide which host driver libraries (libcuda.so.1, libnvidia-*.so)
# to mount into the container. cw doesn't set them by default — without
# them the container has no libcuda and CUDA init fails. SLURM's default
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

export EVAL_ONLY="${EVAL_ONLY:-false}"

export ISL="$ISL"
export OSL="$OSL"

# srt-slurm path requires a CONFIG_FILE pointing to a recipe YAML.
# Without it, srtctl apply scans every YAML in the repo and submits
# hundreds of jobs.
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

# Overlay our hand-rolled DSv4 SGLang recipes. NVIDIA/srt-slurm has no
# upstream sglang DSv4 disagg recipe yet beyond PR #75's 1P1D-TP4
# entry, so we ship the recipe locally and copy it in here. `cp -rT`
# overlays onto a possibly-existing upstream stub instead of nesting.
mkdir -p recipes/sglang/deepseek-v4
cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4" recipes/sglang/deepseek-v4

echo "Installing srtctl..."
# CRITICAL — uv install location.
# Runner pod is x86 but compute nodes are aarch64, and /mnt/home is
# shared NFS across both. srtctl's slurm template (job_script_minimal.j2)
# does `if ! command -v uv` and skips its own ARM64 install when uv is
# already on PATH; on compute nodes $HOME/.local/bin is on PATH by
# default, so a stray x86 binary at $HOME/.local/bin/uv from this
# runner shadows the template's install and crashes the orchestrator
# with `cannot execute binary file: Exec format error`. Install to a
# runner-pod-local /tmp path (tmpfs, not NFS) and scrub any stale x86
# uv left in the shared path by prior runs.
rm -f "$HOME/.local/bin/uv" "$HOME/.local/bin/uvx"
export XDG_BIN_HOME="/tmp/uv-runner-${RUNNER_NAME:-default}/bin"
mkdir -p "$XDG_BIN_HOME"
curl -LsSf https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 sh
export PATH="$XDG_BIN_HOME:$PATH"

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

SRTCTL_ROOT="${GITHUB_WORKSPACE}/srt-slurm"
echo "Creating srtslurm.yaml configuration..."
cat > srtslurm.yaml <<EOF
# SRT SLURM Configuration for GB300-CW (SGLang)

default_account: "${SLURM_ACCOUNT}"
default_partition: "${SLURM_PARTITION}"
default_time_limit: "6:00:00"

gpus_per_node: 4
network_interface: ""

srtctl_root: "${SRTCTL_ROOT}"

model_paths:
  "${SRT_SLURM_MODEL_PREFIX}": "${MODEL_PATH}"
containers:
  dynamo-trtllm: ${SQUASH_FILE}
  dynamo-sglang: ${SQUASH_FILE}
  "${IMAGE}": ${SQUASH_FILE}
  nginx-sqsh: ${NGINX_SQUASH_FILE}
# Auto-emission of #SBATCH --segment={total_nodes} is turned off here
# because each gb300 recipe sets its own segment via sbatch_directives
# (rack-pinning on cw's 2x18-node racks).
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

LOGS_DIR="outputs/$JOB_ID/logs"
LOG_FILE="$LOGS_DIR/sweep_${JOB_ID}.log"

while ! ls "$LOG_FILE" &>/dev/null; do
    if ! squeue -j "$JOB_ID" --noheader 2>/dev/null | grep -q "$JOB_ID"; then
        echo "ERROR: Job $JOB_ID failed before creating log file"
        scontrol show job "$JOB_ID"
        exit 1
    fi
    echo "Waiting for JOB_ID $JOB_ID to begin and $LOG_FILE to appear..."
    sleep 5
done

(
    while squeue -j "$JOB_ID" --noheader 2>/dev/null | grep -q "$JOB_ID"; do
        sleep 10
    done
) &
POLL_PID=$!

echo "Tailing LOG_FILE: $LOG_FILE"

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

    RESULT_SUBDIRS=$(find "$LOGS_DIR" -maxdepth 1 -type d -name "*isl*osl*" 2>/dev/null)

    if [ -z "$RESULT_SUBDIRS" ]; then
        echo "Warning: No result subdirectories found in $LOGS_DIR"
    else
        for result_subdir in $RESULT_SUBDIRS; do
            echo "Processing result subdirectory: $result_subdir"

            CONFIG_NAME=$(basename "$result_subdir")

            RESULT_FILES=$(find "$result_subdir" -name "results_concurrency_*.json" 2>/dev/null)

            for result_file in $RESULT_FILES; do
                if [ -f "$result_file" ]; then
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
