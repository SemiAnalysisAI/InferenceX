#!/usr/bin/bash

# Standalone launcher for the B200 nscale Slurm cluster.
#
# Self-contained on purpose: it does not source or exec launch_b200-dgxc.sh.
# The two clusters share a shape but not their storage layout — dgxc resolves
# models and caches under /lustre/fsw, which does not exist here — so coupling
# them would force every dgxc path to become an override hook and would let a
# dgxc-only change silently alter nscale runs.
#
# Scope: multi-node Dynamo-vLLM DeepSeek-V4-Pro and Kimi K2.6 FP4 runs, plus
# DeepSeek-V4-Pro FP4 Dynamo-SGLang MTP, on the b200-nscale/b200-new runner
# labels. Anything else exits non-zero.

SLURM_PARTITION="batch_1"
SLURM_ACCOUNT="benchmark"

# Node-local NVMe, not a shared filesystem: much faster for the ~1.6T
# DeepSeek-V4-Pro load, and already pre-staged on every nscale compute node.
NSCALE_MODEL_ROOT="/scratch/models"
SQUASH_DIR="/data/home/sa-shared/containers"
AIPERF_MMAP_CACHE_HOST_PATH="/data/home/sa-shared/gharunners/aiperf-cache"
HF_HUB_CACHE_HOST_PATH="/data/home/sa-shared/gharunners/hf-hub-cache"
# Importing the vLLM image over this cluster's shared home is slower than on
# dgxc, so allow well beyond the 600s that suffices there.
SQUASH_LOCK_TIMEOUT=3600

# shellcheck source=runners/slurm_utils.sh
source "$(dirname "${BASH_SOURCE[0]}")/slurm_utils.sh"

set -x

export AIPERF_MMAP_CACHE_HOST_PATH

if [[ "$IS_MULTINODE" != "true" ]]; then
    echo "launch_b200-nscale-slurm.sh only supports multi-node runs (IS_MULTINODE=$IS_MULTINODE)" >&2
    exit 1
fi

if [[ $MODEL_PREFIX == "dsv4" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH="${MODEL_PATH:-$NSCALE_MODEL_ROOT/DeepSeek-V4-Pro}"
    export SRT_SLURM_MODEL_PREFIX="deepseek-v4-pro"
elif [[ $MODEL_PREFIX == "kimik2.6" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH="${MODEL_PATH:-$NSCALE_MODEL_ROOT/Kimi-K2.6-NVFP4}"
    export SRT_SLURM_MODEL_PREFIX="kimi-k2.6-nvfp4"
elif [[ $MODEL_PREFIX == "kimik3" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH="${MODEL_PATH:-$NSCALE_MODEL_ROOT/Kimi-K3}"
    export SRT_SLURM_MODEL_PREFIX="kimik3"
else
    echo "Unsupported model prefix/precision for b200-nscale: $MODEL_PREFIX/$PRECISION" >&2
    echo "Models staged under $NSCALE_MODEL_ROOT:" >&2
    ls -la "$NSCALE_MODEL_ROOT" >&2 || true
    exit 1
fi

if [[ $FRAMEWORK != "dynamo-vllm" ]] &&
   [[ $MODEL_PREFIX != "dsv4" || $PRECISION != "fp4" || $FRAMEWORK != "dynamo-sglang" || $SPEC_DECODING != "mtp" ]]; then
    echo "Unsupported framework/configuration for b200-nscale: $MODEL_PREFIX/$PRECISION/$FRAMEWORK/$SPEC_DECODING" >&2
    exit 1
fi

export SERVED_MODEL_NAME=$MODEL

echo "Cloning srt-slurm repository..."
SRT_REPO_DIR="srt-slurm"
rm -rf "$SRT_REPO_DIR"
if [[ "$IS_AGENTIC" == "1" && $MODEL_PREFIX == "kimik3" ]]; then
    # Pin the tested renderer so branch movement cannot change generated rank
    # commands between sweep points.
    git clone --branch main --single-branch https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR" || exit 1
    cd "$SRT_REPO_DIR" || exit 1
    git checkout 217f9438 || exit 1
    mkdir -p recipes/vllm/kimi-k3/agentic || exit 1
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/kimi-k3/agentic" \
        recipes/vllm/kimi-k3/agentic || exit 1
elif [[ $MODEL_PREFIX == "dsv4" && $FRAMEWORK == "dynamo-sglang" ]]; then
    git clone --branch main --single-branch https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR" || exit 1
    cd "$SRT_REPO_DIR" || exit 1
    # Pin the srt-slurm revision used by these checked-in recipes.
    git checkout 04e87fcc505d6d851451781a5499ca19a02ec2b4 || exit 1
    mkdir -p recipes/sglang/deepseek-v4
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4" recipes/sglang/deepseek-v4
elif [[ $MODEL_PREFIX == "dsv4" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR" || exit 1
    cd "$SRT_REPO_DIR" || exit 1
    git checkout aflowers/vllm-gb200-v0.20.0 || exit 1
    mkdir -p recipes/vllm/deepseek-v4
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/deepseek-v4" recipes/vllm/deepseek-v4
else
    git clone --branch main --single-branch https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR" || exit 1
    cd "$SRT_REPO_DIR" || exit 1
    git checkout c180328b98c3793ca84a1e24a030f90545eb7d5d || exit 1
    mkdir -p recipes/vllm/kimi-k2.6
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/kimi-k2.6" recipes/vllm/kimi-k2.6
fi

echo "Installing srtctl..."
export UV_INSTALL_DIR="$GITHUB_WORKSPACE/.local/bin"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$UV_INSTALL_DIR:$PATH"
uv venv "$GITHUB_WORKSPACE/.venv"
source "$GITHUB_WORKSPACE/.venv/bin/activate"
uv pip install -e .

if ! command -v srtctl &> /dev/null; then
    echo "Error: Failed to install srtctl" >&2
    exit 1
fi

# Map container images to local squash files
NGINX_IMAGE="nginx:1.27.4"
if ! mkdir -p "$SQUASH_DIR" 2>/dev/null || [[ ! -w "$SQUASH_DIR" ]]; then
    echo "Warning: $SQUASH_DIR is not writable; using workspace-local squash cache" >&2
    SQUASH_DIR="$GITHUB_WORKSPACE/.container-squash"
    mkdir -p "$SQUASH_DIR"
fi
chmod a+rx "$SQUASH_DIR" || true

SQUASH_FILE="$SQUASH_DIR/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
NGINX_SQUASH_FILE="$SQUASH_DIR/$(echo "$NGINX_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

# Import containers via enroot, serialized so concurrent runners on this
# cluster don't race on the same squash file.
import_squash() {
    local squash_file="$1"
    local image_ref="$2"
    local image_key
    image_key=$(echo "$image_ref" | sed 's/[\/:@#]/_/g')
    local lock_dir="${SQUASH_DIR}/.locks"
    mkdir -p "$lock_dir"
    local lock_file="${lock_dir}/${image_key}.lock"

    (
        flock -w "$SQUASH_LOCK_TIMEOUT" 9 || { echo "Failed to acquire lock for $squash_file" >&2; exit 1; }
        if unsquashfs -l "$squash_file" > /dev/null 2>&1; then
            echo "Squash file already exists and is valid, skipping import: $squash_file"
        else
            rm -f "$squash_file"
            enroot import -o "$squash_file" "docker://$image_ref"
            if ! unsquashfs -l "$squash_file" > /dev/null 2>&1; then
                echo "Error: enroot import did not produce a valid squash file: $squash_file" >&2
                exit 1
            fi
            chmod a+r "$squash_file" || true
        fi
    ) 9>"$lock_file"
}

import_squash "$SQUASH_FILE" "$IMAGE" || exit 1
import_squash "$NGINX_SQUASH_FILE" "$NGINX_IMAGE" || exit 1

export ISL="$ISL"
export OSL="$OSL"
export EVAL_ONLY="${EVAL_ONLY:-false}"

# Agentic runs bind-mount two persistent caches into every worker container:
# aiperf's content-addressed dataset mmap cache and the HF hub cache holding
# the trace dataset. Container-side paths are referenced by the agentic
# recipes' benchmark.env.
DEFAULT_MOUNTS_BLOCK=""
if [[ "$IS_AGENTIC" == "1" ]]; then
    mkdir -p "$AIPERF_MMAP_CACHE_HOST_PATH" "$HF_HUB_CACHE_HOST_PATH"
    chmod 777 "$AIPERF_MMAP_CACHE_HOST_PATH" "$HF_HUB_CACHE_HOST_PATH" 2>/dev/null || true
    DEFAULT_MOUNTS_BLOCK="default_mounts:
  ${AIPERF_MMAP_CACHE_HOST_PATH}: /aiperf_mmap_cache
  ${HF_HUB_CACHE_HOST_PATH}: /hf_hub_cache"
fi

SRTCTL_ROOT="${GITHUB_WORKSPACE}/${SRT_REPO_DIR}"
echo "Creating srtslurm.yaml configuration..."
cat > srtslurm.yaml <<EOF
# SRT SLURM Configuration for B200 nscale

# Default SLURM settings
default_account: "${SLURM_ACCOUNT}"
default_partition: "${SLURM_PARTITION}"
default_time_limit: "4:00:00"
# Resource defaults
gpus_per_node: 8
network_interface: ""
# Path to srtctl repo root (where the configs live)
srtctl_root: "${SRTCTL_ROOT}"
# Model path aliases
model_paths:
  "${SRT_SLURM_MODEL_PREFIX}": "${MODEL_PATH}"
# Container aliases
containers:
  dynamo-vllm: "${SQUASH_FILE}"
  dynamo-sglang: "${SQUASH_FILE}"
  "${IMAGE}": "${SQUASH_FILE}"
  nginx-sqsh: "${NGINX_SQUASH_FILE}"
use_exclusive_sbatch_directive: true
${DEFAULT_MOUNTS_BLOCK}
EOF

echo "Generated srtslurm.yaml:"
cat srtslurm.yaml

echo "Running make setup..."
make setup ARCH=x86_64

# Export eval-related env vars for srt-slurm post-benchmark eval
export INFMAX_WORKSPACE="$GITHUB_WORKSPACE"

echo "Submitting job with srtctl..."
echo "MODEL_PATH=$MODEL_PATH"

# An eval row may point at a committed real-verification recipe while its
# throughput row keeps synthetic golden acceptance. Only configs that set
# EVAL_CONFIG_FILE opt into this selection; all other configs keep using
# CONFIG_FILE unchanged.
if [[ "${EVAL_ONLY:-false}" == "true" && -n "${EVAL_CONFIG_FILE:-}" ]]; then
    CONFIG_FILE="$EVAL_CONFIG_FILE"
    echo "EVAL_ONLY=true: selecting real-verification recipe $CONFIG_FILE"
fi

if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: CONFIG_FILE is not set. The srt-slurm path requires a CONFIG_FILE in additional-settings." >&2
    echo "Config: MODEL_PREFIX=${MODEL_PREFIX} PRECISION=${PRECISION} FRAMEWORK=${FRAMEWORK}" >&2
    exit 1
fi

# Strip any :override[N] selector so sed and the injector operate on the file.
CONFIG_PATH="${CONFIG_FILE%%:*}"

# Override the job name in the config file with the runner name
sed -i "s/^name:.*/name: \"${RUNNER_NAME}\"/" "$CONFIG_PATH"
# Bump recipe health-check timeout from 360x10s=3600s to 720x10s=7200s so
# large-model loads finish in time.
sed -i 's/^  max_attempts: [0-9]*/  max_attempts: 720/' "$CONFIG_PATH"

inject_synthetic_acceptance "$CONFIG_PATH" "$FRAMEWORK" || exit 1

SRTCTL_PREFLIGHT_ARGS=()
# These weights are staged on the Slurm compute nodes, not the login node.
if [[ $MODEL_PREFIX == "kimik2.6" ]] ||
   [[ $MODEL_PREFIX == "kimik3" ]] ||
   [[ $MODEL_PREFIX == "dsv4" && $FRAMEWORK == "dynamo-sglang" ]]; then
    SRTCTL_PREFLIGHT_ARGS+=(--no-preflight)
fi

SRTCTL_OUTPUT=$(srtctl apply -f "$CONFIG_FILE" "${SRTCTL_PREFLIGHT_ARGS[@]}" --tags "b200,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)" 2>&1)
echo "$SRTCTL_OUTPUT"

JOB_ID=$(echo "$SRTCTL_OUTPUT" | grep -oP '✅ Job \K[0-9]+' || echo "$SRTCTL_OUTPUT" | grep -oP 'Job \K[0-9]+')

set +x

if [ -z "$JOB_ID" ]; then
    echo "Error: Failed to extract JOB_ID from srtctl output" >&2
    exit 1
fi

echo "Extracted JOB_ID: $JOB_ID"

LOGS_DIR="outputs/$JOB_ID/logs"
LOG_FILE="$LOGS_DIR/sweep_${JOB_ID}.log"

# Waits for the log file to appear, fails fast if the job dies first, then
# streams until the job leaves the queue.
stream_slurm_job_log "$JOB_ID" "$LOG_FILE" || exit 1

set -x

echo "Job $JOB_ID completed!"
echo "Collecting results..."

if [ ! -d "$LOGS_DIR" ]; then
    echo "Warning: Logs directory not found at $LOGS_DIR" >&2
    exit 1
fi

cp -r "$LOGS_DIR" "$GITHUB_WORKSPACE/LOGS"
bundle_server_logs "$LOGS_DIR" "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz"

if [[ "${EVAL_ONLY:-false}" != "true" ]]; then
    RESULT_SUBDIRS=$(find "$LOGS_DIR" -maxdepth 1 -type d -name "*isl*osl*" 2>/dev/null)

    if [ -z "$RESULT_SUBDIRS" ]; then
        echo "Warning: No result subdirectories found in $LOGS_DIR" >&2
    else
        for result_subdir in $RESULT_SUBDIRS; do
            echo "Processing result subdirectory: $result_subdir"
            CONFIG_NAME=$(basename "$result_subdir")
            RESULT_FILES=$(find "$result_subdir" -name "results_concurrency_*.json" 2>/dev/null)

            for result_file in $RESULT_FILES; do
                [ -f "$result_file" ] || continue
                # Files may be "results_concurrency_N_gpus_G_ctx_C_gen_D.json"
                # (disagg) or "results_concurrency_N_gpus_G.json" (non-disagg).
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
            done
        done
    fi

    echo "All result files processed"
else
    echo "EVAL_ONLY=true: Skipping benchmark result collection"
fi

# Collect eval results if eval was requested. copy_eval_artifacts warns and
# returns 0 when the directory is absent.
if [[ "${RUN_EVAL:-false}" == "true" || "${EVAL_ONLY:-false}" == "true" ]]; then
    copy_eval_artifacts "$LOGS_DIR/eval_results" "$GITHUB_WORKSPACE"
fi

# Clean up srt-slurm outputs to prevent NFS silly-rename lock files from
# blocking the next job's checkout on this runner.
echo "Cleaning up srt-slurm outputs..."
for i in 1 2 3 4 5; do
    rm -rf outputs 2>/dev/null && break
    echo "Retry $i/5: Waiting for NFS locks to release..."
    sleep 10
done
find . -name '.nfs*' -delete 2>/dev/null || true
