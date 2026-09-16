#!/usr/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/../benchmarks/benchmark_lib.sh" --validation-only || exit 1
check_env_vars EVAL_ONLY IS_AGENTIC IS_MULTINODE RUN_EVAL SALLOC_TIME_LIMIT


set -exo pipefail

# shellcheck source=runners/slurm_utils.sh
source "$(dirname "${BASH_SOURCE[0]}")/slurm_utils.sh" || exit 1

check_env_vars SLURM_PARTITION
export SBATCH_PARTITION="$SLURM_PARTITION"
export SLURM_ACCOUNT="benchmark"
export ENROOT_ROOTFS_WRITABLE=1

# aiperf's dataset mmap cache, mounted at /aiperf_mmap_cache via default_mounts
# below and read through AIPERF_DATASET_MMAP_CACHE_DIR in each agentic recipe's
# benchmark.env. Without it every run rewrites ~65 GB of mmap files per dataset.
export AIPERF_MMAP_CACHE_HOST_PATH="/data/home/sa-shared/gharunners/ai-perf-cache"

export HF_HUB_CACHE_HOST_PATH="/data/home/sa-shared/gharunners/hf-hub-cache"
mkdir -p "$HF_HUB_CACHE_HOST_PATH"

# srtctl's hash-pinned dynamo install (_hash_cached_source_install) caches the
# built wheel at /configs/dynamo-wheels/<hash>. In CI /configs is the per-job
# checkout, so a cold build would need apt and root, which the non-root server
# containers lack; share this host dir there instead. Seed it once with a
# --container-remap-root build.
export DYNAMO_WHEELS_CACHE_HOST_PATH="/data/home/sa-shared/gharunners/dynamo-wheels"
mkdir -p "$DYNAMO_WHEELS_CACHE_HOST_PATH"

export MODEL_PATH=$MODEL

if [[ "$MODEL_PREFIX" == "dsv41flash" && "$PRECISION" == "fp4" && "$FRAMEWORK" == "vllm" && "${IS_MULTINODE}" != "true" ]]; then
    # Download the new checkpoint into the persistent shared HF cache.
    export MODEL_PATH="$MODEL"
elif [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp4" ]]; then
    export SERVED_MODEL_NAME="deepseek-r1-fp4"
    export MODEL_PATH=/scratch/models/DeepSeek-R1-0528-NVFP4-v2
    export SRT_SLURM_MODEL_PREFIX="dsr1"
elif [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp8" ]]; then
    export SERVED_MODEL_NAME="deepseek-r1-fp8"
    export MODEL_PATH=/scratch/models/DeepSeek-R1-0528
    export SRT_SLURM_MODEL_PREFIX="dsr1-fp8"
elif [[ $MODEL_PREFIX == "dsv4" && $PRECISION == "fp4" && $MODEL == "deepseek-ai/DeepSeek-V4-Pro-0813" ]]; then
    export MODEL_PATH="/scratch/models/DeepSeek-V4-Pro-0813"
    export SRT_SLURM_MODEL_PREFIX="deepseek-v4-pro-0813"
elif [[ $MODEL_PREFIX == "dsv4" && $PRECISION == "fp4" ]]; then
    # Node-local /scratch SSD for the 806 GB checkpoint, faster than Vast NFS.
    # It exists only on compute nodes, so srtctl's preflight (which stats from
    # the runner pod) can fail with "path is unavailable".
    export MODEL_PATH=/scratch/models/DeepSeek-V4-Pro
    export SRT_SLURM_MODEL_PREFIX="deepseek-v4-pro"
elif [[ $MODEL_PREFIX == "glm5" && $PRECISION == "fp4" && $FRAMEWORK == "dynamo-trt" ]]; then
    export SERVED_MODEL_NAME="glm-5-nvfp4"
    export MODEL_PATH=/scratch/models/GLM-5-NVFP4
    export SRT_SLURM_MODEL_PREFIX="nvidia/GLM-5-NVFP4"
elif [[ $MODEL_PREFIX == "glm5.1" && $PRECISION == "fp4" ]]; then
    # The GLM-5.1 sglang recipes reuse the glm-5-fp4 alias.
    export MODEL_PATH=/scratch/models/GLM-5.1-NVFP4
    export SRT_SLURM_MODEL_PREFIX="glm-5-fp4"
elif [[ $MODEL_PREFIX == "glm5.2" && $PRECISION == "fp4" && $FRAMEWORK == "dynamo-trt" ]]; then
    export SERVED_MODEL_NAME="GLM-5.2-NVFP4"
    export MODEL_PATH=/scratch/models/GLM-5.2-NVFP4
    export SRT_SLURM_MODEL_PREFIX="nvidia/GLM-5.2-NVFP4"
elif [[ $MODEL_PREFIX == "glm5.2" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH=/scratch/models/GLM-5.2-NVFP4
    export SRT_SLURM_MODEL_PREFIX="glm-5.2-fp4"
elif [[ $MODEL_PREFIX == "glm5" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH=/scratch/models/GLM-5-NVFP4
    export SRT_SLURM_MODEL_PREFIX="glm-5-fp4"
elif [[ $MODEL_PREFIX == "glm5" && $PRECISION == "fp8" ]]; then
    export MODEL_PATH=/scratch/models/GLM-5-FP8
    export SRT_SLURM_MODEL_PREFIX="glm-5-fp8"
elif [[ $MODEL_PREFIX == "minimaxm2.5" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH=/data/models/MiniMax-M2.5-NVFP4
    export SRT_SLURM_MODEL_PREFIX="minimax-m2.5-nvfp4"
elif [[ $MODEL_PREFIX == "minimaxm2.5" && $PRECISION == "fp8" ]]; then
    export MODEL_PATH=/data/models/MiniMax-M2.5
    export SRT_SLURM_MODEL_PREFIX="minimax-m2.5-fp8"
elif [[ $MODEL_PREFIX == "minimaxm3" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH=/scratch/models/MiniMax-M3-NVFP4
    export SRT_SLURM_MODEL_PREFIX="nvidia/MiniMax-M3-NVFP4"
elif [[ $MODEL_PREFIX == "minimaxm3" && $PRECISION == "fp8" ]]; then
    export MODEL_PATH=/data/models/MiniMax-M3-MXFP8
    export SRT_SLURM_MODEL_PREFIX="minimax-m3-mxfp8"
elif [[ $MODEL_PREFIX == "kimik2.5" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH=/scratch/models/Kimi-K2.5-NVFP4
    export SRT_SLURM_MODEL_PREFIX="nvidia/Kimi-K2.5-NVFP4"
elif [[ $MODEL_PREFIX == "kimik3" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH=/scratch/models/Kimi-K3
    export SRT_SLURM_MODEL_PREFIX="moonshotai/Kimi-K3"
elif [[ $MODEL_PREFIX == "qwen3.5" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH=/scratch/models/Qwen3.5-397B-A17B-NVFP4-V2
    export SRT_SLURM_MODEL_PREFIX="qwen3.5-fp4"
elif [[ $MODEL_PREFIX == "qwen3.5" && $PRECISION == "fp8" ]]; then
    export MODEL_PATH=/scratch/models/Qwen3.5-397B-A17B-FP8
    export SRT_SLURM_MODEL_PREFIX="qwen3.5-fp8"
else
    echo "Unsupported model: $MODEL_PREFIX-$PRECISION. Supported models are: dsr1-fp4, dsr1-fp8, dsv4-fp4, glm5-fp4, glm5-fp8, glm5.2-fp4, minimaxm2.5-fp4, minimaxm2.5-fp8, minimaxm3-fp4, minimaxm3-fp8, kimik2.5-fp4, kimik3-fp4, qwen3.5-fp4, qwen3.5-fp8"
    exit 1
fi

NGINX_IMAGE="nginx:1.27.4"

# Use the /data/ mount, not /home/sa-shared/: same Vast NFS backing store, but
# the /home mount has a chronic ELOOP ("Too many levels of symbolic links") bug
# from workflow worker NFS sessions, and /data/ has a separate client cache.
# See feedback_gb300_nfs_eloop_workaround.
SQUASH_FILE="/data/home/sa-shared/gharunners/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
NGINX_SQUASH_FILE="/data/home/sa-shared/gharunners/squash/$(echo "$NGINX_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

# The login node is x86_64 and the compute nodes aarch64, so import on a compute node.
import_squash() {
    local squash="$1" image="$2"
    local lock="${squash}.lock"
    srun --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" --exclusive --time=180 bash -c "
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
# Keep this branch before the nginx import and srtctl setup.
if [[ "$MODEL_PREFIX" == "dsv41flash" && "$FRAMEWORK" == "vllm" && "${IS_MULTINODE}" != "true" ]]; then
    BENCH_SCRIPT="benchmarks/single_node/agentic/${MODEL_PREFIX}_${PRECISION}_gb300_${FRAMEWORK}_mtp.sh"
    # Cover DSpark5 verification for concurrent AgentX subagents at c1/c2/c4.
    export DSV41_MIN_CUDAGRAPH_CAPTURE_SIZE=64
    [[ "${IS_AGENTIC}" == "1" && "${SPEC_DECODING:-}" == "mtp" && -f "$BENCH_SCRIPT" ]] || {
        echo "Unsupported single-node recipe: $BENCH_SCRIPT" >&2
        exit 1
    }
    export HF_HUB_CACHE=/hf-cache
    export INFMAX_CONTAINER_WORKSPACE=/ix
    export RESULT_DIR=/ix/results
    # Cold model loading and graph capture exceeded the one-hour frontend deadline.
    export VLLM_ENGINE_READY_TIMEOUT_S=7200
    srun --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
        --nodes=1 --ntasks=1 --gpus="${TP:?}" --cpus-per-task=144 --exclusive --mem=0 \
        --time="${SALLOC_TIME_LIMIT}" --job-name="$RUNNER_NAME" \
        --mpi=none --container-image="$SQUASH_FILE" \
        --container-mounts="$GITHUB_WORKSPACE:/ix,$HF_HUB_CACHE_HOST_PATH:/hf-cache" \
        --no-container-mount-home --container-remap-root \
        --container-workdir=/ix --no-container-entrypoint \
        --export=ALL,PORT=8888 bash "$BENCH_SCRIPT"
    exit $?
fi

import_squash "$NGINX_SQUASH_FILE" "$NGINX_IMAGE"

# A recipe opts into the power lane via an enabled dcgm-power telemetry block.
# The srt-slurm checkout does not exist yet, so read the workspace mirror;
# recipes that exist only upstream stay non-power.
USES_DCGM_POWER=0
_RECIPE_REL="${CONFIG_FILE%%:*}"
_RECIPE_SRC="$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/${_RECIPE_REL#recipes/}"
# Scoped match: a stray "enabled: true" outside the telemetry block must not flip the lane.
if [[ -n "$CONFIG_FILE" && -f "$_RECIPE_SRC" ]] && awk '
    /^telemetry:/ { t = 1; next }
    t && /^[^ ]/  { t = 0 }
    t && /^  dcgm_exporter:/ { p = 1 }
    t && /^  enabled: true$/        { e = 1 }
    END { exit !(p && e) }
' "$_RECIPE_SRC"; then
    USES_DCGM_POWER=1
fi

USES_AGENTX_POWER=0
if [[ "$USES_DCGM_POWER" == "1" && "$IS_AGENTIC" == "1" &&
    "$MODEL_PREFIX" == "kimik3" && "$PRECISION" == "fp4" &&
    "$FRAMEWORK" == "dynamo-vllm" &&
    "$_RECIPE_REL" == recipes/kimik3/vllm/*/agentx/* ]]; then
    USES_AGENTX_POWER=1
fi
if [[ "$USES_DCGM_POWER" == "1" && "$FRAMEWORK" != "dynamo-sglang" && "$USES_AGENTX_POWER" != "1" ]]; then
    echo "Error: dcgm-power requires dynamo-sglang or the supported Kimi-K3 AgentX route" >&2
    exit 1
fi


if [[ "$USES_DCGM_POWER" == "1" ]]; then
    DCGM_EXPORTER_IMAGE="nvcr.io/nvidia/k8s/dcgm-exporter:4.6.0-4.8.3-distroless"
    # enroot resolves bare paths against Docker Hub; nvcr.io pulls need the registry# form
    DCGM_EXPORTER_ENROOT_REF="${DCGM_EXPORTER_IMAGE/nvcr.io\//nvcr.io#}"
    DCGM_EXPORTER_SQSH="/data/home/sa-shared/gharunners/squash/$(echo "$DCGM_EXPORTER_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    # import_squash does not re-validate a fresh import, so check explicitly on
    # a compute node (login node is x86, nodes aarch64).
    import_squash "$DCGM_EXPORTER_SQSH" "$DCGM_EXPORTER_ENROOT_REF"
    test -r "$DCGM_EXPORTER_SQSH" || { echo "Error: DCGM exporter squash not readable: $DCGM_EXPORTER_SQSH" >&2; exit 1; }
    srun --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" --exclusive --time=30 bash -c "unsquashfs -l \"$DCGM_EXPORTER_SQSH\" > /dev/null" || { echo "Error: DCGM exporter squash invalid: $DCGM_EXPORTER_SQSH" >&2; exit 1; }
    sha256sum "$DCGM_EXPORTER_SQSH" > "$GITHUB_WORKSPACE/exporter-image.sha256"
fi

if [[ "$EVAL_ONLY" == "true" && -n "${EVAL_CONFIG_FILE:-}" ]]; then
    CONFIG_FILE="$EVAL_CONFIG_FILE"
    echo "EVAL_ONLY=true: selecting real-verification recipe $CONFIG_FILE"
fi

export ISL="$ISL"
export OSL="$OSL"

echo "Preparing job-local srt-slurm checkout..."
check_env_vars RESULT_FILENAME
RUN_KEY=$(printf "%s" "${RESULT_FILENAME}" | sha1sum | cut -c1-12)
check_env_vars GITHUB_RUN_ID GITHUB_RUN_ATTEMPT
SRT_REPO_DIR="${GITHUB_WORKSPACE}/srt-slurm-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}-${RUN_KEY}"
rm -rf "$SRT_REPO_DIR"

setup_srt_slurm "$SRT_REPO_DIR" "$FRAMEWORK" "$USES_DCGM_POWER" || exit 1

if [[ "$FRAMEWORK" == "dynamo-trt" && "$MODEL_PREFIX" == "dsv4" ]]; then
    SRT_SLURM_MODEL_PREFIX="deepseek-ai/DeepSeek-V4-Pro"
fi
# Accuracy runs use real speculative verification and a frontend colocated
# with the post-eval client on the allocation head.
if [[ "$IS_AGENTIC" == "1" && "$FRAMEWORK" == "dynamo-trt" && "$EVAL_ONLY" == "true" ]]; then
    find recipes -path 'recipes/*/trtllm/*' -name '*.yaml' -exec sed -i '/TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS/d' {} +
    if [[ "$MODEL_PREFIX" == "dsv4" ]]; then
        SRTCTL_EVAL_ARGS+=(--set frontend.placement.node=head)
    fi
fi

echo "Installing srtctl..."
export UV_INSTALL_DIR="$GITHUB_WORKSPACE/.local/bin"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$UV_INSTALL_DIR:$PATH"

check_env_vars GITHUB_RUN_ID GITHUB_RUN_ATTEMPT
VENV_DIR="${GITHUB_WORKSPACE}/.venv-srt-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}-${RUN_KEY}"
rm -rf "$VENV_DIR"
# --seed installs pip; srtctl's prefetch-ai-dynamo-wheel.sh (recipes with
# dynamo.wheel) otherwise fails with "No module named pip".
uv venv --seed "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv pip install -e .

if ! command -v srtctl &> /dev/null; then
    echo "Error: Failed to install srtctl"
    exit 1
fi

echo "Configs available at: $SRT_REPO_DIR/"

SRTCTL_ROOT="${SRT_REPO_DIR}"
echo "Creating srtslurm.yaml configuration..."
SRT_DEFAULT_TIME_LIMIT="4:00:00"
if [[ "$IS_AGENTIC" == "1" && "$MODEL_PREFIX" == "dsv4" && ( "$FRAMEWORK" == "dynamo-sglang" || "$FRAMEWORK" == "dynamo-trt" ) ]]; then
    SRT_DEFAULT_TIME_LIMIT="8:00:00"
fi
cat > srtslurm.yaml <<EOF
# SRT SLURM Configuration for GB300

# Default SLURM settings
default_account: "${SLURM_ACCOUNT}"
default_partition: "${SLURM_PARTITION}"
default_time_limit: "${SRT_DEFAULT_TIME_LIMIT}"

# Resource defaults
gpus_per_node: 4
network_interface: ""

# Path to srtctl repo root (where the configs live)
srtctl_root: "${SRTCTL_ROOT}"

# Cluster-level bind mounts applied to every worker container
# (see srtctl/core/runtime.py — get_srtslurm_setting("default_mounts")).
# Used here for aiperf's persistent mmap cache so the dataset isn't
# re-tokenized + re-written every job.
default_mounts:
  "${AIPERF_MMAP_CACHE_HOST_PATH}": "/aiperf_mmap_cache"
  "${HF_HUB_CACHE_HOST_PATH}": "/hf_hub_cache"
  # Warm dynamo source-build cache (nested over the auto /configs mount) so the
  # hash-pinned install is a cache hit (pip-only, no apt/root) on every job.
  "${DYNAMO_WHEELS_CACHE_HOST_PATH}": "/configs/dynamo-wheels"

# Model path aliases
model_paths:
  "${SRT_SLURM_MODEL_PREFIX}": "${MODEL_PATH}"
containers:
  dynamo-trtllm: ${SQUASH_FILE}
  dynamo-sglang: ${SQUASH_FILE}
  v0.5.11: ${SQUASH_FILE}
  v0.5.13.post1: ${SQUASH_FILE}
  "${IMAGE}": ${SQUASH_FILE}
  nginx-sqsh: ${NGINX_SQUASH_FILE}
use_segment_sbatch_directive: false
EOF

# Appended via sed so non-power lanes' generated yaml stays byte-identical.
if [[ "$USES_DCGM_POWER" == "1" ]]; then
    sed -i "/^  nginx-sqsh:/a\\  dcgm-exporter: ${DCGM_EXPORTER_SQSH}" srtslurm.yaml
    # sed's append is a silent no-op if the anchor drifts.
    grep -q "^  dcgm-exporter: " srtslurm.yaml || { echo "Error: dcgm-exporter injection failed: nginx-sqsh anchor not found in srtslurm.yaml" >&2; exit 1; }
fi

echo "Generated srtslurm.yaml:"
cat srtslurm.yaml

echo "Running make setup..."
make setup ARCH=aarch64

# Read by srt-slurm's post-benchmark eval.
export INFMAX_WORKSPACE="$GITHUB_WORKSPACE"

echo "Submitting job with srtctl..."

if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: CONFIG_FILE is not set. The srt-slurm path requires a CONFIG_FILE in additional-settings." >&2
    echo "Config: MODEL_PREFIX=${MODEL_PREFIX} PRECISION=${PRECISION} FRAMEWORK=${FRAMEWORK}" >&2
    exit 1
fi

# CONFIG_FILE may carry a ":zip_override_...[i]" selector that only
# `srtctl apply -f` parses; strip it for the sed, pass the full value to srtctl.
CONFIG_PATH="${CONFIG_FILE%%:*}"
sed -i "s/^name:.*/name: \"${RUNNER_NAME}\"/" "$CONFIG_PATH"

# Throughput recipes opt into synthetic acceptance via the master config;
# eval-only jobs strip it so tokens get real target-model verification.
inject_synthetic_acceptance "$CONFIG_PATH" "$FRAMEWORK" || exit 1

if [[ "$USES_AGENTX_POWER" == "1" ]]; then
    read -r -a POWER_CONCURRENCIES <<< "$CONC_LIST"
    python3 "$GITHUB_WORKSPACE/runners/inject_srt_power_concurrencies.py" \
        "$CONFIG_PATH" "${POWER_CONCURRENCIES[@]}" || exit 1
fi

# Skip the login-host model check for checkpoints staged only on compute
# nodes. The worker still validates the model path during startup.
SRTCTL_APPLY_ARGS=(
    -f "$CONFIG_FILE"
    --tags "gb300,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)"
)
if [[ "$IS_AGENTIC" == "1" || "$MODEL_PREFIX" == "glm5.1" || ( "$MODEL_PREFIX" == "qwen3.5" && "$PRECISION" == "fp8" ) || ( "$MODEL_PREFIX" == "qwen3.5" && "$PRECISION" == "fp4" && ( "$FRAMEWORK" == "dynamo-trt" || "$USES_DCGM_POWER" == "1" ) ) || ( "$USES_DCGM_POWER" == "1" && "$MODEL_PREFIX" == "dsv4" && "$FRAMEWORK" == "dynamo-sglang" ) ]]; then
    SRTCTL_APPLY_ARGS+=(--no-preflight)
fi

SRTCTL_OUTPUT=$(srtctl apply "${SRTCTL_EVAL_ARGS[@]}" "${SRTCTL_APPLY_ARGS[@]}" 2>&1)
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

# Snapshot worker logs on every exit path, including the SIGTERM that
# `gh run cancel` sends during the tail wait; otherwise the Upload server logs
# step finds nothing.
_snapshot_server_logs() {
    if [ -n "${LOGS_DIR:-}" ] && [ -d "$LOGS_DIR" ] && [ -n "${GITHUB_WORKSPACE:-}" ]; then
        # Provenance markers ride in the trap so cancel paths still bundle them for the audit.
        if [[ "$USES_DCGM_POWER" == "1" ]]; then
            mkdir -p "$LOGS_DIR/power" 2>/dev/null || true
            cp "$GITHUB_WORKSPACE/exporter-image.sha256" "$LOGS_DIR/power/exporter-image.sha256" 2>/dev/null || true
            cp "$GITHUB_WORKSPACE/power-producer-sha.txt" "$LOGS_DIR/power/power-producer-sha.txt" 2>/dev/null || true
        fi
        # Independent best-effort steps: an in-flight worker .out write at SIGTERM
        # would otherwise abort the script before either succeeds.
        cp -r "$LOGS_DIR" "$GITHUB_WORKSPACE/LOGS" 2>/dev/null || true
        tar czf "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz" -C "$LOGS_DIR" . 2>/dev/null || true
    fi
}
trap _snapshot_server_logs EXIT

AGENTX_POWER_RC=0
if [[ "$USES_AGENTX_POWER" == "1" ]]; then
    stream_slurm_job_log "$JOB_ID" "$LOG_FILE" || AGENTX_POWER_RC=$?
else
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

    # -F follows by name and polls; inotify does not work on NFS.
    tail -F -s 2 -n+1 "$LOG_FILE" --pid=$POLL_PID 2>/dev/null

    wait $POLL_PID
fi

set -x

echo "Job $JOB_ID completed!"
echo "Collecting results..."

if [[ "$USES_AGENTX_POWER" == "1" && "${EVAL_ONLY}" != "true" ]]; then
    read -r -a POWER_CONCURRENCIES <<< "$CONC_LIST"
    collect_agentic_power_results "$JOB_ID" "$LOGS_DIR" "$INFMAX_WORKSPACE" \
        "$GITHUB_WORKSPACE" "$RESULT_FILENAME" "$SRT_SLURM_COMMIT" \
        "${POWER_CONCURRENCIES[@]}" || AGENTX_POWER_RC=$?
fi

if [ -d "$LOGS_DIR" ]; then
    echo "Found logs directory: $LOGS_DIR"
    # The EXIT trap produces the tarball, LOGS copy, and provenance markers.
    echo "multinode_server_logs.tar.gz will be (re)produced on script EXIT."
else
    echo "Warning: Logs directory not found at $LOGS_DIR"
fi

if [[ "$AGENTX_POWER_RC" != "0" ]]; then
    echo "ERROR: AgentX job or power validation failed; EXIT will stage audit artifacts" >&2
    exit "$AGENTX_POWER_RC"
fi

if [[ "${EVAL_ONLY}" != "true" ]]; then
    if [ ! -d "$LOGS_DIR" ]; then
        exit 1
    fi

    copy_fixed_sequence_results "$LOGS_DIR" "$GITHUB_WORKSPACE" "$RESULT_FILENAME"
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
            eval_dest="$GITHUB_WORKSPACE/$(basename "$eval_file")"
            rm -f "$eval_dest"
            if cp "$eval_file" "$eval_dest"; then
                echo "Copied eval artifact: $(basename "$eval_file")"
            else
                echo "WARNING: Failed to copy eval artifact, continuing: $(basename "$eval_file")"
            fi
        done
        shopt -u nullglob
    else
        echo "WARNING: RUN_EVAL=true but no eval results found at $EVAL_DIR"
    fi

    # srt-slurm stages eval artifacts but not the metadata file score validation
    # consumes; the canonical writer keeps topology aligned with workflow inputs.
    check_env_vars EVAL_CONC
    eval_conc_value="$EVAL_CONC"
    (
        export IS_MULTINODE=true
        # shellcheck source=benchmarks/benchmark_lib.sh
        source "$GITHUB_WORKSPACE/benchmarks/benchmark_lib.sh"
        _write_lm_eval_meta_json \
            "$GITHUB_WORKSPACE/meta_env.json" "" "$eval_conc_value"
    )
    echo "Wrote meta_env.json (conc=${eval_conc_value}, prefix=${MODEL_PREFIX:-unknown})"
fi

# The EXIT trap fires after the rm below, when its LOGS_DIR guard is already
# false, so snapshot here first.
_snapshot_server_logs

# Clean up srt-slurm outputs to prevent NFS silly-rename lock files
# from blocking the next job's checkout on this runner
echo "Cleaning up srt-slurm outputs..."
for i in 1 2 3 4 5; do
    rm -rf outputs 2>/dev/null && break
    echo "Retry $i/5: Waiting for NFS locks to release..."
    sleep 10
done
find . -name '.nfs*' -delete 2>/dev/null || true
