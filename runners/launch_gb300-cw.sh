#!/usr/bin/bash

# Launches multi-node Dynamo + SGLang benchmarks on the gb300-cw
# (CoreWeave) cluster. Adapted from the dynamo-vllm sibling launcher in
# the dsv4-fp4-gb300-dynamo-vllm-disagg branch (PR #1150). The SGLang
# recipes are copied exactly from the pinned srt-slurm commit below.

set -x

if [[ $MODEL_PREFIX == "dsv4" && $PRECISION == "fp4" ]]; then
    # Weights staged on shared storage; avoid node-local /scratch symlink drift.
    export MODEL_PATH="/mnt/vast/models/dsv4"

    if [[ $FRAMEWORK == "dynamo-sglang" ]]; then
        # Pinned to our SemiAnalysisAI fork of NVIDIA/srt-slurm to pick up
        # PR #35 (per-node nvidia-smi monitoring during the benchmark sweep)
        # ahead of its upstream merge. The branch tracks PR #35's head SHA:
        # to bump, re-fetch refs/pull/35/head from NVIDIA/srt-slurm and force-
        # push to SemiAnalysisAI/srt-slurm:feat/inferencex-perfmon.
        SRT_SLURM_RECIPES_REPO="https://github.com/SemiAnalysisAI/srt-slurm.git"
        SRT_SLURM_RECIPES_REF="feat/inferencex-perfmon"
        SRT_RECIPE_SRC="$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4"
        SRT_RECIPE_DST="recipes/sglang/deepseek-v4"
    elif [[ $FRAMEWORK == "dynamo-vllm" ]]; then
        SRT_SLURM_RECIPES_REPO="https://github.com/NVIDIA/srt-slurm.git"
        SRT_SLURM_RECIPES_REF="aflowers/gb200-dsv4-recipes"
        SRT_RECIPE_SRC="$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/deepseek-v4"
        SRT_RECIPE_DST="recipes/vllm/deepseek-v4"
    else
        echo "Unsupported framework on gb300-cw for dsv4/fp4: $FRAMEWORK. Currently supported: dynamo-sglang, dynamo-vllm"
        exit 1
    fi
else
    echo "Unsupported model prefix/precision combination on gb300-cw: $MODEL_PREFIX/$PRECISION. Currently supported: dsv4/fp4"
    exit 1
fi

# CoreWeave cluster has a single `all` partition; account `cw-sup` is
# what `sacctmgr show assoc user=$USER` returns there. `benchmark`
# (inherited from gb300-nv) does not exist on cw.
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
# `squash_dupe` instead of `squash` to use '_'-separated names: srtctl /
# pyxis rejects '+' in image paths with "Invalid image format", and the
# old /mnt/vast/squash dir contains '+'-separated files from prior runs.
SQUASH_DIR="/mnt/vast/squash_dupe"
mkdir -p "$SQUASH_DIR"
SQUASH_FILE="$SQUASH_DIR/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g')_arm64.sqsh"
NGINX_SQUASH_FILE="$SQUASH_DIR/$(echo "$NGINX_IMAGE" | sed 's/[\/:@#]/_/g')_arm64.sqsh"

# Run the import on a compute node via srun, not on the runner pod:
# the runner pod is x86_64 while the compute nodes are aarch64, so the
# arm64 squash file has to be built on a compute node.
import_squash() {
    local squash="$1" image="$2"
    local lock="${squash}.lock"
    srun --partition=$SLURM_PARTITION --account=$SLURM_ACCOUNT --exclusive --time=180 bash -c "
        exec 9>\"$lock\"
        flock -w 600 9 || { echo 'Failed to acquire lock for $squash' >&2; exit 1; }
        if unsquashfs -l \"$squash\" > /dev/null 2>&1; then
            echo 'Squash file already exists and is valid, skipping import: $squash'
        else
            rm -f \"$squash\"
            enroot import -o \"$squash\" docker://$image
        fi
    "
}

import_squash "$SQUASH_FILE" "$IMAGE"
import_squash "$NGINX_SQUASH_FILE" "$NGINX_IMAGE"

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

git clone "$SRT_SLURM_RECIPES_REPO" "$SRT_REPO_DIR"
cd "$SRT_REPO_DIR"
git checkout "$SRT_SLURM_RECIPES_REF"

# Overlay the hand-rolled DSV4 recipes onto the selected srt-slurm checkout.
mkdir -p "$SRT_RECIPE_DST"
cp -rT "$SRT_RECIPE_SRC" "$SRT_RECIPE_DST"

# Enable per-node GPU perfmon (PR #35) on every overlaid recipe. `monitoring`
# is a top-level SrtConfig field and defaults to None, so without this the
# orchestrator's _start_perf_monitor short-circuits and no perf_samples_*.csv
# are ever written — multinode measured-power aggregation would silently
# skip. Idempotent: skips recipes that already declare `monitoring:`.
for recipe in "$SRT_RECIPE_DST"/*.yaml; do
    [ -f "$recipe" ] || continue
    if ! grep -q '^monitoring:' "$recipe"; then
        printf '\nmonitoring:\n  enabled: true\n  sample_interval: 1.0\n' >> "$recipe"
        echo "[perfmon] enabled monitoring in recipe: $recipe"
    fi
done

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

# Persistent cluster-wide cache for `dynamo: hash:` source builds. The
# upstream cache root (_DYNAMO_CACHE_ROOT in srtctl/core/schema.py) is
# `/configs/dynamo-wheels`; without an override that dir lives inside
# `srt-slurm/configs`, which the launcher wipes via `rm -rf` every CI
# run, so each run does a cold ~10-20 min rust+pyo3 build. Stage the
# cache on /mnt/vast (NFS, shared by all gb300-cw_N runners) and have
# srtctl bind-mount it over `/configs/dynamo-wheels` via the cluster
# `default_mounts` setting. flock inside srtctl serializes cold-cache
# builds across concurrent matrix jobs.
DYNAMO_WHEELS_CACHE_HOST="/mnt/vast/dynamo-wheels-cache"
mkdir -p "$DYNAMO_WHEELS_CACHE_HOST"
mkdir -p configs/dynamo-wheels

echo "Creating srtslurm.yaml configuration..."
cat > srtslurm.yaml <<EOF
# SRT SLURM Configuration for GB300-CW

default_account: "${SLURM_ACCOUNT}"
default_partition: "${SLURM_PARTITION}"
default_time_limit: "8:00:00"

gpus_per_node: 4
network_interface: ""

srtctl_root: "${SRTCTL_ROOT}"

default_mounts:
  ${DYNAMO_WHEELS_CACHE_HOST}: /configs/dynamo-wheels

model_paths:
  dspro: "${MODEL_PATH}"
  dsv4-pro: "${MODEL_PATH}"
  # Our hand-rolled DSV4 sglang recipes use `model.path: deepseek-v4-pro`
  # (matches the alias in launch_gb200-nv.sh's srtslurm.yaml). Without
  # this entry srtctl preflight rejects with "Model 'deepseek-v4-pro'
  # is not a local model path and is not defined in srtslurm.yaml
  # model_paths".
  deepseek-v4-pro: "${MODEL_PATH}"
containers:
  dynamo-trtllm: ${SQUASH_FILE}
  dynamo-sglang: ${SQUASH_FILE}
  dynamo-vllm: ${SQUASH_FILE}
  dspro-0426: ${SQUASH_FILE}
  dspro-0426-nixl: ${SQUASH_FILE}
  dsv4-grace-blackwell: ${SQUASH_FILE}
  "${IMAGE}": ${SQUASH_FILE}
  nginx: ${NGINX_SQUASH_FILE}
  nginx-sqsh: ${NGINX_SQUASH_FILE}
# Use one contiguous CW segment for the full allocation. This is a
# cluster-level setting, not a recipe overlay; the copied recipe files
# stay byte-identical to the pinned upstream commit.
use_segment_sbatch_directive: true
# Cluster-wide bash preamble — runs before every container srun. Raises
# NOFILE so the dynamo frontend / sglang servers can accept high
# concurrency (8192 in the 7p1d sweep) without EMFILE / "too many open
# files". Mirrors what `yangminl@slurm-login-0:~/srt-slurm/srtslurm.yaml`
# already uses for manual runs on this cluster.
default_bash_preamble: "ulimit -n 1048576 && ulimit -a"
EOF

echo "Generated srtslurm.yaml:"
cat srtslurm.yaml

echo "Running make setup..."
make setup ARCH=aarch64

# Export eval-related env vars for srt-slurm post-benchmark eval
export INFMAX_WORKSPACE="$GITHUB_WORKSPACE"

echo "Submitting job with srtctl..."

# Use the runner name for the submitted job. Some exact upstream recipes do
# not define `name`, so insert it into only the cloned runtime copy.
if grep -q '^name:' "$CONFIG_FILE"; then
    sed -i "s/^name:.*/name: \"${RUNNER_NAME}\"/" "$CONFIG_FILE"
else
    TMP_CONFIG_FILE="$(mktemp)"
    awk -v runner_name="${RUNNER_NAME}" 'BEGIN { print "name: \"" runner_name "\"" } { print }' "$CONFIG_FILE" > "$TMP_CONFIG_FILE"
    mv "$TMP_CONFIG_FILE" "$CONFIG_FILE"
fi

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

# Hand the per-node perfmon CSVs off to the downstream "Process result" step
# in benchmark-multinode-tmpl.yml. srt-slurm's perfmon (PR #35) writes
# perf_samples_{node}.csv straight into $LOGS_DIR on the host. process_result.py
# already invokes aggregate_power.run() inline; teaching it to read
# GPU_METRICS_CSV_GLOB lets utils/aggregate_power.py do the multi-CSV
# aggregation (each agg JSON gets avg_power_w / joules_per_*_token patched in
# place). Use an absolute glob because process_result.py runs from
# $GITHUB_WORKSPACE, not from this srt-slurm checkout.
if [ -d "$LOGS_DIR" ]; then
    perf_glob_dir="$(pwd)/$LOGS_DIR"
    perf_csv_count=$(ls "$perf_glob_dir"/perf_samples_*.csv 2>/dev/null | wc -l | tr -d ' ')
    if [ "$perf_csv_count" -gt 0 ]; then
        echo "[perfmon] Found $perf_csv_count per-node perf_samples_*.csv under $perf_glob_dir/"
        echo "GPU_METRICS_CSV_GLOB=$perf_glob_dir/perf_samples_*.csv" >> "$GITHUB_ENV"
    else
        echo "[perfmon] WARNING: monitoring enabled but no perf_samples_*.csv found in $perf_glob_dir — measured power aggregation will be skipped"
    fi
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
