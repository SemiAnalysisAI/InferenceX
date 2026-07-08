#!/usr/bin/bash

source "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE must be set}/runners/lib/srt_slurm.sh"

# This script sets up the environment and launches multi-node benchmarks

set -x

load_runner_model gb200-nv || exit 1

# Set up environment variables for SLURM
export SLURM_PARTITION="batch"
export SLURM_ACCOUNT="benchmark"

NGINX_IMAGE="nginx:1.27.4"

uses_watchtower_shared_fs() {
    case "$MODEL_PREFIX" in
        minimaxm2.5|minimaxm3|kimik2.5) return 0 ;;
        *) return 1 ;;
    esac
}

# === Cluster diagnostic probe for watchtower-hosted sweeps ===
# The gb200-nv_* runners may be hosted on different physical clusters
# (e.g., the legacy NVIDIA Lustre cluster vs Oracle Cloud "watchtower").
# Print enough info to identify the layout, then pick a writable
# squash dir on a path that's also visible to compute nodes. Falls
# back to the legacy sa-shared path so other configs are untouched.
SQUASH_DIR="$RUNNER_PATH_SQUASH_ROOT"
if uses_watchtower_shared_fs; then
    echo "=== cluster diagnostic (watchtower sweep) ==="
    echo "USER=$(id -un) UID=$(id -u) GID=$(id -g) GROUPS=$(id -Gn)"
    echo "HOME=$HOME"
    echo "HOSTNAME=$(hostname -f 2>/dev/null || hostname)"
    echo "GITHUB_WORKSPACE=$GITHUB_WORKSPACE"
    echo "--- mount summary ---"
    mount | grep -E 'lustre|nfs|home|shared|/mnt' || true
    echo "--- /mnt contents ---"
    ls -ld /mnt/* 2>/dev/null || true
    echo "--- /mnt/lustre01 user dirs ---"
    ls -ld /mnt/lustre01/users/* 2>/dev/null || true
    ls -ld /mnt/lustre01/users-public/* 2>/dev/null || true
    ls -ld /mnt/lustre01/groups/* 2>/dev/null || true
    echo "--- /nfs contents (if present) ---"
    ls -ld /nfs/* 2>/dev/null || true
    echo "--- /home contents ---"
    ls -ld /home/* 2>/dev/null || true
    echo "=== end diagnostic ==="

    # Probe candidate squash dirs in order, pick first writable one.
    SQUASH_DIR=""
    for cand in \
        /mnt/lustre01/users/slurm-shared/squash \
        /mnt/lustre01/users-public/slurm-shared/squash \
        /mnt/lustre01/groups/slurm-shared/squash \
        /mnt/lustre01/users-public/sa-shared \
        /nfs/slurm-shared/squash \
        /home/slurm-shared/gharunners/squash
    do
        if mkdir -p "$cand" 2>/dev/null && touch "$cand/.write-probe.$$" 2>/dev/null; then
            rm -f "$cand/.write-probe.$$" 2>/dev/null
            SQUASH_DIR="$cand"
            echo "Selected SQUASH_DIR=$SQUASH_DIR (first writable candidate)"
            break
        else
            echo "  not writable: $cand"
        fi
    done
    if [ -z "$SQUASH_DIR" ]; then
        echo "Error: no writable squash dir candidate found on this cluster" >&2
        exit 1
    fi
fi
SQUASH_FILE="${SQUASH_DIR}/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
NGINX_SQUASH_FILE="${SQUASH_DIR}/$(echo "$NGINX_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

# Concurrent matrix jobs import to the same shared-FS squash path.
# Serialize imports and atomically replace invalid images so readers never
# observe a partially written squash file.
import_squash() {
    local squash="$1" image="$2"
    local lock="${squash}.lock"
    (
        exec 9>"$lock"
        flock -w 1800 9 || { echo "Failed to acquire lock for $squash" >&2; exit 1; }
        if unsquashfs -l "$squash" > /dev/null 2>&1; then
            echo "Squash file already exists and is valid, skipping import: $squash"
        else
            rm -f "$squash" "$squash".tmp.*
            enroot import -o "${squash}.tmp.$$" "docker://$image"
            mv -f "${squash}.tmp.$$" "$squash"
        fi
    ) || exit 1
}

import_squash "$SQUASH_FILE" "$IMAGE"
import_squash "$NGINX_SQUASH_FILE" "$NGINX_IMAGE"

export EVAL_ONLY="${EVAL_ONLY:-false}"

export ISL="$ISL"
export OSL="$OSL"

# Legacy path that doesn't use srt-slurm
if [[ $FRAMEWORK == "dynamo-sglang" && -z "$CONFIG_FILE" ]]; then
    export IMAGE=$SQUASH_FILE
    export SGL_SLURM_JOBS_PATH="dynamo/examples/backends/sglang/slurm_jobs"
    SCRIPT_NAME="${EXP_NAME%%_*}_${PRECISION}_gb200_${FRAMEWORK}.sh"
    if [[ "$FRAMEWORK" == "dynamo-sglang" ]] || [[ "$FRAMEWORK" == "dynamo-trt" ]]; then
        BENCHMARK_SUBDIR="multi_node"
    else
        BENCHMARK_SUBDIR="single_node"
    fi
    bash "benchmarks/${BENCHMARK_SUBDIR}/${SCRIPT_NAME}"
    # Wait for all jobs to complete
    echo "Waiting for all jobs to complete..."
    while [ -n "$(squeue -u $USER --noheader --format='%i')" ]; do
        echo "Jobs still running..."
        squeue --steps -u $USER
        sleep 30
    done

        # Find the latest log directory that contains the data
    cat > collect_latest_results.py <<'PY'
import os, sys
sgl_job_dir, isl, osl, nexp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
for path in sorted([f"{sgl_job_dir}/logs/{name}/vllm_isl_{isl}_osl_{osl}" for name in os.listdir(f"{sgl_job_dir}/logs/") if os.path.isdir(f"{sgl_job_dir}/logs/{name}/vllm_isl_{isl}_osl_{osl}")], key=os.path.getmtime, reverse=True)[:nexp]:
    print(path)
PY

    LOGS_DIR=$(python3 collect_latest_results.py "$SGL_SLURM_JOBS_PATH" $ISL $OSL 1)
    if [ -z "$LOGS_DIR" ]; then
        echo "No logs directory found for ISL=${ISL}, OSL=${OSL}"
        exit 1
    fi

    echo "Found logs directory: $LOGS_DIR"
    ls -la $LOGS_DIR

    # Result JSON are contained within the result directory
    for result_file in $(find $LOGS_DIR -type f); do
        # result_file should directly be isl_ISL_osl_OSL_concurrency_CONC_req_rate_R_gpus_N_ctx_M_gen_N.json
        file_name=$(basename $result_file)
        if [ -f $result_file ]; then
            # Copy the result file to workspace with a unique name
            WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${file_name}"
            echo "Found result file ${result_file}. Copying them to ${WORKSPACE_RESULT_FILE}"
            cp $result_file $WORKSPACE_RESULT_FILE
        fi
    done

    exit 0
fi


# srt-slurm path requires a CONFIG_FILE pointing to a recipe YAML.
# Without it, srtctl apply scans every YAML in the repo and submits hundreds of jobs.
if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: CONFIG_FILE is not set. The srt-slurm path requires a CONFIG_FILE in additional-settings." >&2
    echo "Config: MODEL_PREFIX=${MODEL_PREFIX} PRECISION=${PRECISION} FRAMEWORK=${FRAMEWORK}" >&2
    exit 1
fi

echo "Cloning srt-slurm repository..."
SRT_REPO_DIR="srt-slurm"
SRTCTL_SETUP_SCRIPT=""
# On the watchtower (Oracle) gb200 cluster, /home/slurm-shared is not
# cross-mounted to compute nodes. Put the srt-slurm workspace and staged
# InferenceX checkout on a writable shared-FS path that compute can see.
# Per-run-unique paths avoid races between parallel sweep jobs.
if uses_watchtower_shared_fs; then
    SHARED_BASE=""
    for cand in \
        /mnt/lustre01/users-public/sa-shared/gha-runs \
        /mnt/lustre01/users/slurm-shared/gha-runs \
        /mnt/lustre01/users-public/slurm-shared/gha-runs \
        /mnt/lustre01/groups/slurm-shared/gha-runs \
        /nfs/slurm-shared/gha-runs \
        /home/slurm-shared/gharunners/gha-runs
    do
        if mkdir -p "$cand" 2>/dev/null && touch "$cand/.write-probe.$$" 2>/dev/null; then
            rm -f "$cand/.write-probe.$$" 2>/dev/null
            SHARED_BASE="$cand"
            echo "Selected SHARED_BASE=$SHARED_BASE (first writable candidate)"
            break
        else
            echo "  not writable: $cand"
        fi
    done
    if [ -z "$SHARED_BASE" ]; then
        echo "Error: no writable shared run directory candidate found on this cluster" >&2
        exit 1
    fi
    RUN_KEY="${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-0}-${RUNNER_NAME:-gb200-nv}-$$"
    SRT_REPO_DIR="${SHARED_BASE}/srt-slurm-${RUN_KEY}"
    echo "Using shared-FS SRT_REPO_DIR=$SRT_REPO_DIR (compute-visible)"
fi
if [ -d "$SRT_REPO_DIR" ]; then
    echo "Removing existing $SRT_REPO_DIR..."
    rm -rf "$SRT_REPO_DIR"
fi

clone_srt_slurm "$SRT_REPO_DIR" || exit 1
cd "$SRT_REPO_DIR" || exit 1

if [[ $FRAMEWORK == "dynamo-vllm" && $MODEL_PREFIX == "minimaxm3" && $PRECISION == "fp8" ]]; then
    SRTCTL_SETUP_SCRIPT="minimax-m3-gb200-vllm-fixes.sh"
    cp \
        "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/configs/$SRTCTL_SETUP_SCRIPT" \
        "configs/$SRTCTL_SETUP_SCRIPT" || exit 1
fi

CONFIG_FILE="$(stage_srt_recipe "$CONFIG_FILE")" || exit 1
CONFIG_FILE="$(prepare_srt_benchmark "$CONFIG_FILE")" || exit 1

echo "Installing srtctl..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Watchtower: the launcher runs on the head node but compute nodes
# inherit the activated .venv (via VIRTUAL_ENV) through SRT_REPO_DIR
# which is now on shared FS. If uv's default python install lives
# under a head-node-only path, .venv/bin/python3 becomes a broken
# symlink on compute. Pin the venv to /usr/bin/python3 — a system
# path that exists at the same location on both head and compute.
if uses_watchtower_shared_fs && [[ -x /usr/bin/python3 ]]; then
    uv venv --seed --python /usr/bin/python3
else
    uv venv --seed
fi
source .venv/bin/activate
uv pip install -e .

if ! command -v srtctl &> /dev/null; then
    echo "Error: Failed to install srtctl"
    exit 1
fi

echo "Configs available at: $SRT_REPO_DIR/"

# Create srtslurm.yaml for srtctl (used by both frameworks)
SRTCTL_ROOT="${GITHUB_WORKSPACE}/srt-slurm"
# Watchtower-hosted sweeps: SRT_REPO_DIR was moved to a shared-FS path
# above so srtctl's outputs/ directory (which lives under
# SRTCTL_ROOT) is visible to compute nodes.
if uses_watchtower_shared_fs; then
    SRTCTL_ROOT="$SRT_REPO_DIR"
fi

# Agentic runs bind-mount two persistent caches into every worker container
# (Lustre, shared across nodes): aiperf's content-addressed dataset mmap
# cache (~65 GB per corpus, re-tokenized from scratch without it) and the
# HF hub cache holding the trace dataset download. The container-side paths
# are referenced by the agentic recipes' benchmark.env
# (AIPERF_DATASET_MMAP_CACHE_DIR=/aiperf_mmap_cache, HF_HUB_CACHE=/hf_hub_cache).
DEFAULT_MOUNTS_BLOCK=""
if [[ "$IS_AGENTIC" == "1" ]]; then
    AIPERF_MMAP_CACHE_HOST_PATH="$RUNNER_PATH_AIPERF_CACHE"
    HF_HUB_CACHE_HOST_PATH="$RUNNER_PATH_HF_CACHE"
    mkdir -p "$AIPERF_MMAP_CACHE_HOST_PATH" "$HF_HUB_CACHE_HOST_PATH"
    chmod 777 "$AIPERF_MMAP_CACHE_HOST_PATH" "$HF_HUB_CACHE_HOST_PATH" 2>/dev/null || true
    DEFAULT_MOUNTS_BLOCK="default_mounts:
  ${AIPERF_MMAP_CACHE_HOST_PATH}: /aiperf_mmap_cache
  ${HF_HUB_CACHE_HOST_PATH}: /hf_hub_cache"
fi

echo "Creating srtslurm.yaml configuration..."
cat > srtslurm.yaml <<EOF
# SRT SLURM Configuration for GB200

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
  inferencex-model: "${MODEL_PATH}"
  "${SRT_SLURM_MODEL_PREFIX}": "${MODEL_PATH}"
containers:
  inferencex-workload: ${SQUASH_FILE}
  inferencex-nginx: ${NGINX_SQUASH_FILE}
  dynamo-trtllm: ${SQUASH_FILE}
  dynamo-sglang: ${SQUASH_FILE}
  "${IMAGE}": ${SQUASH_FILE}
  nginx-sqsh: ${NGINX_SQUASH_FILE}
# srtctl defaults this to true, which adds #SBATCH --segment=<total_nodes>.
# On watchtower the whole batch partition (blue-cn01-18) is a single NVL72
# rack, so segment contiguity buys nothing for MNNVL — but it DOES make
# jobs unschedulable when the partition is fragmented: Slurm backfills a
# non-contiguous node set, fails segment placement at start, and the job
# dies with "CANCELLED Reason=Resources" at RunTime=0 (hit by the first
# gb200 agentic run, job 18582). Mirror launch_gb300-nv.sh and disable.
use_segment_sbatch_directive: false
${DEFAULT_MOUNTS_BLOCK}
EOF

echo "Generated srtslurm.yaml:"
cat srtslurm.yaml

echo "Running make setup..."
make setup ARCH=aarch64 || exit 1

# Export eval-related env vars for srt-slurm post-benchmark eval
export INFMAX_WORKSPACE="$GITHUB_WORKSPACE"
# Watchtower: pyxis mounts INFMAX_WORKSPACE into the container, but
# GITHUB_WORKSPACE is under /home/slurm-shared/ which compute nodes
# can't see. Stage the relevant subset to shared FS and repoint
# INFMAX_WORKSPACE there. rsync excludes the srt-slurm clone (already
# on shared FS) and .git (not needed in container) for speed.
if uses_watchtower_shared_fs; then
    SHARED_INFMAX_WORKSPACE="${SHARED_BASE}/infmax-workspace-${RUN_KEY}"
    mkdir -p "$SHARED_INFMAX_WORKSPACE" || exit 1
    rsync -a --delete \
        --exclude='.git/' \
        --exclude='srt-slurm*/' \
        --exclude='outputs/' \
        --exclude='LOGS/' \
        --exclude='*.sqsh' \
        "${GITHUB_WORKSPACE}/" "${SHARED_INFMAX_WORKSPACE}/" || exit 1
    export INFMAX_WORKSPACE="$SHARED_INFMAX_WORKSPACE"
    echo "Using shared-FS INFMAX_WORKSPACE=$INFMAX_WORKSPACE (compute-visible)"
fi

echo "Submitting job with srtctl..."

# Resolve the recipe path before editing or submitting it.
CONFIG_PATH="${CONFIG_FILE%%:*}"
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: CONFIG_FILE does not exist after srt-slurm setup: $CONFIG_PATH" >&2
    echo "Current directory: $(pwd)" >&2
    exit 1
fi

# Keep the Slurm job name aligned with the GitHub runner name.
sed -i "s/^name:.*/name: \"${RUNNER_NAME}\"/" "$CONFIG_PATH"

# Don't leak the login-node venv to the compute-node orchestrator. sbatch's
# default --export=ALL propagates VIRTUAL_ENV (set by `source
# .venv/bin/activate` above) into job_script_minimal.j2, whose
# `uv run` step then tries to inspect the *active* venv — and dies with
# "Broken symlink at .venv/bin/python3" because the login-node interpreter
# path doesn't exist on compute nodes (gb200 agentic R2, job 18587).
# srtctl itself still resolves through PATH (.venv/bin is on it).
unset VIRTUAL_ENV

# --no-preflight is only used on the agentic path, where the recipe resolves
# model.path to /mnt/numa1 (compute-node-only NVMe) that the login-node
# runner can't see. Fixed-seq-len recipes keep enforcement on.
PREFLIGHT_ARGS=()
if [[ "$IS_AGENTIC" == "1" ]]; then
    PREFLIGHT_ARGS=(--no-preflight)
fi

SRTCTL_APPLY_ARGS=(
    "${PREFLIGHT_ARGS[@]}"
    -f "$CONFIG_PATH"
    --tags "gb200,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)"
)
if [[ "$FRAMEWORK" == "dynamo-sglang" ]]; then
    SRTCTL_APPLY_ARGS+=(--setup-script install-torchao.sh)
elif [[ -n "$SRTCTL_SETUP_SCRIPT" ]]; then
    SRTCTL_APPLY_ARGS+=(--setup-script "$SRTCTL_SETUP_SCRIPT")
fi
SRTCTL_OUTPUT=$(srtctl apply "${SRTCTL_APPLY_ARGS[@]}" 2>&1)
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

require_slurm_job_succeeded "$JOB_ID" || exit 1

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
