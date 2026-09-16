#!/usr/bin/bash

# B200 nscale Slurm launcher for the native srt-slurm lanes selected below;
# everything else falls through to launch_b200-nscale-compat.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../benchmarks/benchmark_lib.sh" --validation-only || exit 1
check_env_vars EVAL_ONLY IS_AGENTIC RUN_EVAL


SLURM_PARTITION="batch_1"
SLURM_ACCOUNT="benchmark"

# Node-local NVMe, not a shared filesystem: much faster for the ~1.6T
# DeepSeek-V4-Pro load, and already pre-staged on every nscale compute node.
NSCALE_MODEL_ROOT="/scratch/models"
SQUASH_DIR="/data/home/sa-shared/containers"
AIPERF_MMAP_CACHE_HOST_PATH="/data/home/sa-shared/gharunners/aiperf-cache"
HF_HUB_CACHE_HOST_PATH="/data/home/sa-shared/gharunners/hf-hub-cache"
# Importing the vLLM image over this cluster's shared home can take a while.
SQUASH_LOCK_TIMEOUT=3600

# shellcheck source=runners/slurm_utils.sh
source "$(dirname "${BASH_SOURCE[0]}")/slurm_utils.sh" || exit 1

set -x

export AIPERF_MMAP_CACHE_HOST_PATH

run_compat_launcher() {
    exec bash "$(dirname "${BASH_SOURCE[0]}")/launch_b200-nscale-compat.sh"
}

if [[ "$IS_MULTINODE" != "true" ]]; then
    run_compat_launcher
fi

if [[ "$FRAMEWORK" == "tilert" && "${IS_AGENTIC}" != "1" ]]; then
    run_compat_launcher
fi

if [[ $MODEL_PREFIX == "dsv4" && $PRECISION == "fp4" ]]; then
    check_env_vars MODEL_PATH
    export SRT_SLURM_MODEL_PREFIX="deepseek-v4-pro"
elif [[ $MODEL_PREFIX == "kimik2.6" && $PRECISION == "fp4" ]]; then
    check_env_vars MODEL_PATH
    export SRT_SLURM_MODEL_PREFIX="kimi-k2.6-nvfp4"
elif [[ $MODEL_PREFIX == "kimik3" && $PRECISION == "fp4" ]]; then
    check_env_vars MODEL_PATH
    export SRT_SLURM_MODEL_PREFIX="kimik3"
elif [[ $MODEL_PREFIX == "glm5.2" && $PRECISION == "fp4" ]]; then
    check_env_vars MODEL_PATH
    # This alias must match model.path in the checked-in GLM-5.2 recipes.
    export SRT_SLURM_MODEL_PREFIX="glm-5.2-fp4"
elif [[ $MODEL_PREFIX == "glm5.1" && $PRECISION == "fp8" && $FRAMEWORK == "tilert" ]]; then
    export SRT_SLURM_MODEL_PREFIX="glm5.1-fp8"
else
    run_compat_launcher
fi

if [[ $FRAMEWORK != "dynamo-vllm" ]] &&
   [[ $MODEL_PREFIX != "dsv4" || $PRECISION != "fp4" || $FRAMEWORK != "dynamo-sglang" ||
      ( $SPEC_DECODING != "none" && $SPEC_DECODING != "mtp" ) ]] &&
   [[ $MODEL_PREFIX != "glm5.2" || $PRECISION != "fp4" || $FRAMEWORK != "dynamo-sglang" || $SPEC_DECODING != "mtp" ]] &&
   [[ $MODEL_PREFIX != "glm5.1" || $PRECISION != "fp8" || $FRAMEWORK != "tilert" || $SPEC_DECODING != "mtp" ]]; then
    run_compat_launcher
fi

USES_DCGM_POWER=0
USES_AGENTX_POWER=0
_POWER_CONFIG_FILE="${CONFIG_FILE:-}"
if [[ "${EVAL_ONLY}" == "true" && -n "${EVAL_CONFIG_FILE:-}" ]]; then
    _POWER_CONFIG_FILE="$EVAL_CONFIG_FILE"
fi
_RECIPE_REL="${_POWER_CONFIG_FILE%%:*}"
_RECIPE_SRC="$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/${_RECIPE_REL#recipes/}"
if [[ -n "$_POWER_CONFIG_FILE" && -f "$_RECIPE_SRC" ]] && awk '
    /^telemetry:/ { t = 1; next }
    t && /^[^ ]/  { t = 0 }
    t && /^  dcgm_exporter:/ { p = 1 }
    t && /^  enabled: true$/        { e = 1 }
    END { exit !(p && e) }
' "$_RECIPE_SRC"; then
    USES_DCGM_POWER=1
fi
if [[ "$USES_DCGM_POWER" == "1" && "$IS_AGENTIC" == "1" &&
    "$MODEL_PREFIX" == "kimik3" && "$PRECISION" == "fp4" && "$FRAMEWORK" == "dynamo-vllm" ]]; then
    USES_AGENTX_POWER=1
elif [[ "$USES_DCGM_POWER" == "1" && (
    "${IS_AGENTIC}" == "1" ||
    "$PRECISION" != "fp4" ||
    ( "$MODEL_PREFIX" == "dsv4" && "$FRAMEWORK" != "dynamo-sglang" && "$FRAMEWORK" != "dynamo-vllm" ) ||
    ( "$MODEL_PREFIX" == "kimik2.6" && "$FRAMEWORK" != "dynamo-vllm" ) ||
    ( "$MODEL_PREFIX" != "dsv4" && "$MODEL_PREFIX" != "kimik2.6" )
) ]]; then
    echo "Error: B200 nscale dcgm-power requires a supported fixed-sequence lane or Kimi-K3 AgentX vLLM" >&2
    exit 1
fi

export SERVED_MODEL_NAME=$MODEL

echo "Preparing job-local srt-slurm checkout..."
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
    echo "Error: Failed to install srtctl" >&2
    exit 1
fi

NGINX_IMAGE="nginx:1.27.4"
if ! mkdir -p "$SQUASH_DIR" 2>/dev/null || [[ ! -w "$SQUASH_DIR" ]]; then
    echo "Warning: $SQUASH_DIR is not writable; using workspace-local squash cache" >&2
    SQUASH_DIR="$GITHUB_WORKSPACE/.container-squash"
    mkdir -p "$SQUASH_DIR"
fi
chmod a+rx "$SQUASH_DIR" || true

SQUASH_FILE="$SQUASH_DIR/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
NGINX_SQUASH_FILE="$SQUASH_DIR/$(echo "$NGINX_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

# Enroot treats docker://foo/bar as a Docker Hub path. Preserve that form for
# ordinary Docker Hub images, but use Enroot's explicit registry separator for
# fully qualified references such as ghcr.io/tile-ai/tilert.
enroot_uri_for_image() {
    local image_ref="$1"
    local first_component="${image_ref%%/*}"

    if [[ "$image_ref" == */* && (
        "$first_component" == *.* ||
        "$first_component" == *:* ||
        "$first_component" == "localhost"
    ) ]]; then
        printf 'docker://%s#%s\n' "$first_component" "${image_ref#*/}"
    else
        printf 'docker://%s\n' "$image_ref"
    fi
}

# Import containers via enroot, serialized so concurrent runners on this
# cluster don't race on the same squash file.
import_squash() {
    local squash_file="$1"
    local image_ref="$2"
    local image_key enroot_uri
    image_key=$(echo "$image_ref" | sed 's/[\/:@#]/_/g')
    enroot_uri=$(enroot_uri_for_image "$image_ref") || exit 1
    local lock_dir="${SQUASH_DIR}/.locks"
    mkdir -p "$lock_dir"
    local lock_file="${lock_dir}/${image_key}.lock"

    (
        flock -w "$SQUASH_LOCK_TIMEOUT" 9 || { echo "Failed to acquire lock for $squash_file" >&2; exit 1; }
        if unsquashfs -l "$squash_file" > /dev/null 2>&1; then
            echo "Squash file already exists and is valid, skipping import: $squash_file"
        else
            rm -f "$squash_file"
            enroot import -o "$squash_file" "$enroot_uri"
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

PREFILL_SQUASH_FILE=""
TILERT_CONTAINER_BLOCK=""
if [[ $FRAMEWORK == "tilert" ]]; then
    : "${PREFILL_IMAGE:?PREFILL_IMAGE is required for TileRT prefill}"
    PREFILL_SQUASH_FILE="$SQUASH_DIR/$(echo "$PREFILL_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    import_squash "$PREFILL_SQUASH_FILE" "$PREFILL_IMAGE" || exit 1
    TILERT_CONTAINER_BLOCK="
  tilert-decode: ${SQUASH_FILE}
  tilert-prefill: ${PREFILL_SQUASH_FILE}"
fi

if [[ "$USES_DCGM_POWER" == "1" ]]; then
    DCGM_EXPORTER_IMAGE="nvcr.io/nvidia/k8s/dcgm-exporter:4.6.0-4.8.3-distroless"
    DCGM_EXPORTER_SQSH="$SQUASH_DIR/$(echo "$DCGM_EXPORTER_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    import_squash "$DCGM_EXPORTER_SQSH" "$DCGM_EXPORTER_IMAGE" || exit 1
    test -r "$DCGM_EXPORTER_SQSH" || { echo "Error: DCGM exporter squash not readable: $DCGM_EXPORTER_SQSH" >&2; exit 1; }
    unsquashfs -l "$DCGM_EXPORTER_SQSH" > /dev/null || { echo "Error: DCGM exporter squash invalid: $DCGM_EXPORTER_SQSH" >&2; exit 1; }
    sha256sum "$DCGM_EXPORTER_SQSH" > "$GITHUB_WORKSPACE/exporter-image.sha256"
fi

export ISL="$ISL"
export OSL="$OSL"

# Persistent caches for aiperf's dataset mmap files and the HF trace dataset;
# the container paths are referenced by the agentic recipes' benchmark.env.
DEFAULT_MOUNTS_BLOCK=""
if [[ "$IS_AGENTIC" == "1" ]]; then
    mkdir -p "$AIPERF_MMAP_CACHE_HOST_PATH" "$HF_HUB_CACHE_HOST_PATH"
    chmod 777 "$AIPERF_MMAP_CACHE_HOST_PATH" "$HF_HUB_CACHE_HOST_PATH" 2>/dev/null || true
    DEFAULT_MOUNTS_BLOCK="default_mounts:
  ${AIPERF_MMAP_CACHE_HOST_PATH}: /aiperf_mmap_cache
  ${HF_HUB_CACHE_HOST_PATH}: /hf_hub_cache"
fi
if [[ $FRAMEWORK == "tilert" ]]; then
    TILERT_WEIGHTS_HOST_PATH="/data/home/sa-shared/gharunners/tilert-cache"
    mkdir -p "$TILERT_WEIGHTS_HOST_PATH"
    DEFAULT_MOUNTS_BLOCK="${DEFAULT_MOUNTS_BLOCK}
  ${GITHUB_WORKSPACE}: /infmax-workspace
  ${TILERT_WEIGHTS_HOST_PATH}: ${TILERT_WEIGHTS_HOST_PATH}"
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
${TILERT_CONTAINER_BLOCK}
use_exclusive_sbatch_directive: true
${DEFAULT_MOUNTS_BLOCK}
EOF

if [[ "$USES_DCGM_POWER" == "1" ]]; then
    sed -i "/^  nginx-sqsh:/a\\  dcgm-exporter: ${DCGM_EXPORTER_SQSH}" srtslurm.yaml
    grep -q "^  dcgm-exporter: " srtslurm.yaml || { echo "Error: dcgm-exporter injection failed: nginx-sqsh anchor not found in srtslurm.yaml" >&2; exit 1; }
fi

echo "Generated srtslurm.yaml:"
cat srtslurm.yaml

echo "Running make setup..."
make setup ARCH=x86_64

# Read by srt-slurm's post-benchmark eval.
export INFMAX_WORKSPACE="$GITHUB_WORKSPACE"

echo "Submitting job with srtctl..."
echo "MODEL_PATH=$MODEL_PATH"

# An eval row may use a real-verification recipe while its throughput row
# keeps synthetic acceptance; only configs setting EVAL_CONFIG_FILE opt in.
if [[ "${EVAL_ONLY}" == "true" && -n "${EVAL_CONFIG_FILE:-}" ]]; then
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

sed -i "s/^name:.*/name: \"${RUNNER_NAME}\"/" "$CONFIG_PATH"
# Give recipes at least 720 attempts without shortening a larger model-specific
# load budget (GLM-5.2 intentionally requests 1440x10s).
RECIPE_MAX_ATTEMPTS=$(sed -n 's/^  max_attempts: \([0-9][0-9]*\)$/\1/p' "$CONFIG_PATH" | head -1)
if [[ $RECIPE_MAX_ATTEMPTS =~ ^[0-9]+$ ]] && (( RECIPE_MAX_ATTEMPTS < 720 )); then
    sed -i 's/^  max_attempts: [0-9]*/  max_attempts: 720/' "$CONFIG_PATH"
fi

inject_synthetic_acceptance "$CONFIG_PATH" "$FRAMEWORK" || exit 1

if [[ "$USES_AGENTX_POWER" == "1" ]]; then
    read -r -a POWER_CONCURRENCIES <<< "$CONC_LIST"
    python "$GITHUB_WORKSPACE/runners/inject_srt_power_concurrencies.py" \
        "$CONFIG_PATH" "${POWER_CONCURRENCIES[@]}" || exit 1
fi

SRTCTL_PREFLIGHT_ARGS=()
# These weights are staged on the Slurm compute nodes, not the login node.
if [[ $MODEL_PREFIX == "kimik2.6" ]] ||
   [[ $MODEL_PREFIX == "kimik3" ]] ||
   [[ $MODEL_PREFIX == "glm5.2" ]] ||
   [[ $MODEL_PREFIX == "dsv4" ]]; then
    SRTCTL_PREFLIGHT_ARGS+=(--no-preflight)
fi

SRTCTL_OUTPUT=$(srtctl apply "${SRTCTL_EVAL_ARGS[@]}" -f "$CONFIG_FILE" "${SRTCTL_PREFLIGHT_ARGS[@]}" --tags "b200,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)" 2>&1)
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

SRT_JOB_RC=0
stream_slurm_job_log "$JOB_ID" "$LOG_FILE" || SRT_JOB_RC=$?
if [[ "$SRT_JOB_RC" != "0" && "$USES_AGENTX_POWER" != "1" ]]; then
    exit "$SRT_JOB_RC"
fi

set -x

echo "Job $JOB_ID completed!"
echo "Collecting results..."

if [ ! -d "$LOGS_DIR" ]; then
    echo "Warning: Logs directory not found at $LOGS_DIR" >&2
    exit 1
fi

AGENTX_POWER_RC="$SRT_JOB_RC"
if [[ "$USES_AGENTX_POWER" == "1" && "${EVAL_ONLY}" != "true" ]]; then
    read -r -a POWER_CONCURRENCIES <<< "$CONC_LIST"
    collect_agentic_power_results "$JOB_ID" "$LOGS_DIR" \
        "$GITHUB_WORKSPACE" "$GITHUB_WORKSPACE" "$RESULT_FILENAME" \
        "$SRT_SLURM_COMMIT" "${POWER_CONCURRENCIES[@]}" || AGENTX_POWER_RC=$?
fi

if [[ "$USES_DCGM_POWER" == "1" ]]; then
    mkdir -p "$LOGS_DIR/power"
    cp "$GITHUB_WORKSPACE/exporter-image.sha256" "$LOGS_DIR/power/exporter-image.sha256"
    cp "$GITHUB_WORKSPACE/power-producer-sha.txt" "$LOGS_DIR/power/power-producer-sha.txt"
fi

cp -r "$LOGS_DIR" "$GITHUB_WORKSPACE/LOGS"
bundle_server_logs "$LOGS_DIR" "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz"

if [[ "$AGENTX_POWER_RC" != "0" ]]; then
    echo "ERROR: AgentX power validation failed; available audit and server artifacts were staged" >&2
    exit "$AGENTX_POWER_RC"
fi

if [[ "${EVAL_ONLY}" != "true" ]]; then
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

if [[ "${RUN_EVAL}" == "true" || "${EVAL_ONLY}" == "true" ]]; then
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
