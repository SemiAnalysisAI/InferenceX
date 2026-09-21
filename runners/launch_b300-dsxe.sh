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

    srun -N 1 -A "$SLURM_ACCOUNT" -p "$SLURM_PARTITION" \
        --time="${ENROOT_IMPORT_TIME_LIMIT}" bash -c "
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

SRT_EXTRA_CLUSTER_CONFIG=""
if [[ "$IS_AGENTIC" == "1" && "$FRAMEWORK" == "dynamo-trt" && "$MODEL_PREFIX" == "glm5.2" ]]; then
    # TRT-LLM rc26 ships NIXL v1.4.0 without its LIBFABRIC plugin. Stage the
    # matching wheel plugin and EFA userspace runtime in a shared, immutable
    # cache, then mount only those runtime libraries.
    NIXL_LIBFABRIC_HOST_DIR="/data/home/sa-gha-runner/nixl-libfabric/nixl-1.4.0-efa-1.47.0"
    mkdir -p "$(dirname "$NIXL_LIBFABRIC_HOST_DIR")"
    (
        exec 9>"${NIXL_LIBFABRIC_HOST_DIR}.lock"
        flock -w 1800 9 || exit 1

        if [[ ! -r "$NIXL_LIBFABRIC_HOST_DIR/nixl/libplugin_LIBFABRIC.so" ||
              ! -r "$NIXL_LIBFABRIC_HOST_DIR/efa/opt/amazon/efa/lib/libfabric.so.1" ||
              ! -r "$NIXL_LIBFABRIC_HOST_DIR/efa/usr/lib/x86_64-linux-gnu/libibverbs/libefa-rdmav59.so" ]]; then
            if [[ -e "$NIXL_LIBFABRIC_HOST_DIR" ]]; then
                echo "Error: incomplete NIXL LIBFABRIC cache: $NIXL_LIBFABRIC_HOST_DIR" >&2
                exit 1
            fi

            _nixl_stage=$(mktemp -d "${NIXL_LIBFABRIC_HOST_DIR}.tmp.XXXXXX")
            trap 'rm -rf -- "$_nixl_stage"' EXIT
            mkdir -p "$_nixl_stage/runtime/nixl" "$_nixl_stage/runtime/efa" \
                "$_nixl_stage/installer"

            curl -LfsS --retry 3 -o "$_nixl_stage/nixl.whl" \
                "https://files.pythonhosted.org/packages/8b/7c/b79fb09e832233c90f1e9d9b953e88c2b92096d968f2444839c6aa92b645/nixl_cu13-1.4.0-cp312-cp312-manylinux_2_28_x86_64.whl"
            echo "3e606fbe80c39ce14899726fad0cb0fec53c6bac9f34168492692c4166b2fabb  $_nixl_stage/nixl.whl" | sha256sum -c -
            unzip -p "$_nixl_stage/nixl.whl" \
                nixl_cu13.libs/nixl/libplugin_LIBFABRIC.so \
                > "$_nixl_stage/runtime/nixl/libplugin_LIBFABRIC.so"
            unzip -p "$_nixl_stage/nixl.whl" \
                nixl_cu13.libs/libnuma-3387f5e3.so.1.0.0 \
                > "$_nixl_stage/runtime/nixl/libnuma-3387f5e3.so.1.0.0"

            curl -LfsS --retry 3 -o "$_nixl_stage/efa.tar.gz" \
                "https://efa-installer.amazonaws.com/aws-efa-installer-1.47.0.tar.gz"
            echo "2df4201e046833c7dc8160907bee7f52b76ff80ed147376a2d0ed8a0dd66b2db  $_nixl_stage/efa.tar.gz" | sha256sum -c -
            tar -xzf "$_nixl_stage/efa.tar.gz" -C "$_nixl_stage/installer" \
                aws-efa-installer/DEBS/UBUNTU2404/x86_64/libfabric1-aws_2.4.0amzn1.0_amd64.deb \
                aws-efa-installer/DEBS/UBUNTU2404/x86_64/rdma-core/ibverbs-providers_61.0-1_amd64.deb \
                aws-efa-installer/DEBS/UBUNTU2404/x86_64/rdma-core/libibverbs1_61.0-1_amd64.deb \
                aws-efa-installer/DEBS/UBUNTU2404/x86_64/rdma-core/librdmacm1_61.0-1_amd64.deb \
                aws-efa-installer/DEBS/UBUNTU2404/x86_64/rdma-core/rdma-core_61.0-1_amd64.deb
            while IFS= read -r -d '' _efa_deb; do
                dpkg-deb -x "$_efa_deb" "$_nixl_stage/runtime/efa"
            done < <(find "$_nixl_stage/installer" -name '*.deb' -print0)
            mv "$_nixl_stage/runtime" "$NIXL_LIBFABRIC_HOST_DIR"
        fi
    ) || exit 1

    # The cluster has EFA and Mellanox HCAs. Its Mellanox Enroot hook otherwise
    # masks the EFA sysfs devices before the LIBFABRIC backend starts.
    export MELLANOX_VISIBLE_DEVICES=void

    AIPERF_MMAP_CACHE_HOST_PATH="/data/home/sa-gha-runner/aiperf-cache"
    HF_HUB_CACHE_HOST_PATH="/data/home/sa-gha-runner/hf-hub-cache"
    TRTLLM_JIT_CACHE_HOST_PATH="/data/home/sa-gha-runner/trtllm-jit-cache"
    mkdir -p \
        "$AIPERF_MMAP_CACHE_HOST_PATH" \
        "$HF_HUB_CACHE_HOST_PATH" \
        "$TRTLLM_JIT_CACHE_HOST_PATH"
    chmod 0777 \
        "$AIPERF_MMAP_CACHE_HOST_PATH" \
        "$HF_HUB_CACHE_HOST_PATH" \
        "$TRTLLM_JIT_CACHE_HOST_PATH" 2>/dev/null || true

    SRT_EXTRA_CLUSTER_CONFIG=$(cat <<EOF
default_mounts:
  "${NIXL_LIBFABRIC_HOST_DIR}": "/nixl-libfabric"
  "${AIPERF_MMAP_CACHE_HOST_PATH}": "/aiperf_mmap_cache"
  "${HF_HUB_CACHE_HOST_PATH}": "/hf_hub_cache"
  "${TRTLLM_JIT_CACHE_HOST_PATH}": "/trtllm-jit-cache"
default_bash_preamble: "if command -v trtllm-llmapi-launch >/dev/null 2>&1; then source /configs/prepare-dynamo-venv.sh || exit; fi; export NIXL_PLUGIN_DIR=/nixl-libfabric/nixl; export LD_LIBRARY_PATH=/nixl-libfabric/efa/opt/amazon/efa/lib:/nixl-libfabric/efa/usr/lib/x86_64-linux-gnu:/nixl-libfabric/nixl:\${LD_LIBRARY_PATH:-}; export IBV_DRIVERS_PATH=/nixl-libfabric/efa/usr/lib/x86_64-linux-gnu/libibverbs; export FI_PROVIDER=efa; export FI_EFA_USE_DEVICE_RDMA=1"
EOF
)
fi

SRTCTL_ROOT="${GITHUB_WORKSPACE}/${SRT_REPO_DIR}"
echo "Creating srtslurm.yaml configuration..."
write_srt_cluster_config b300-dsxe srtslurm.yaml "$USES_DCGM_POWER" \
    --var MODEL_ROOT "$MODEL_ROOT" || exit 1
if [[ -n "$SRT_EXTRA_CLUSTER_CONFIG" ]]; then
    printf '%s\n' "$SRT_EXTRA_CLUSTER_CONFIG" >> srtslurm.yaml
fi

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
if [[ "$IS_AGENTIC" == "1" && "$EVAL_ONLY" != "true" ]]; then
    check_env_vars RESULT_FILENAME
    SRTCTL_APPLY_ARGS+=(--set "benchmark.env.RESULT_FILENAME=$RESULT_FILENAME")
fi
SRTCTL_OUTPUT=$(apply_srt_recipe "$CONFIG_FILE" "$FRAMEWORK" "${SRTCTL_EVAL_ARGS[@]}" "${SRTCTL_APPLY_ARGS[@]}" 2>&1)
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

# -F follows by name and polls; inotify does not work on NFS.
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

RESULT_COLLECTION_RC=0
if [[ "${EVAL_ONLY}" != "true" ]]; then
    if [[ "$IS_AGENTIC" == "1" ]]; then
        check_slurm_job_success "$JOB_ID" "$LOGS_DIR" || RESULT_COLLECTION_RC=1
        copy_agentic_results "$GITHUB_WORKSPACE" "$GITHUB_WORKSPACE" "$RESULT_FILENAME" || RESULT_COLLECTION_RC=1
    else
        copy_fixed_sequence_results "$LOGS_DIR" "$GITHUB_WORKSPACE" "$RESULT_FILENAME" || RESULT_COLLECTION_RC=1
    fi
else
    echo "EVAL_ONLY=true: Skipping benchmark result collection"
fi

cp -r "$LOGS_DIR" "$GITHUB_WORKSPACE/LOGS"
tar czf "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz" -C "$LOGS_DIR" .

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
if [[ "$RESULT_COLLECTION_RC" != "0" ]]; then
    exit "$RESULT_COLLECTION_RC"
fi

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
    if [[ "$MODEL_PREFIX" == "dsv41flash" && "$FRAMEWORK" == "vllm" ]]; then
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
    JOB_ID=$(
        set -o pipefail
        LC_ALL=C salloc "${SALLOC_ARGS[@]}" 2>&1 | tee /dev/stderr |
            sed -n 's/.*Granted job allocation \([0-9][0-9]*\)$/\1/p'
    ) || exit 1
    [[ "$JOB_ID" =~ ^[0-9]+$ ]] || { echo 'ERROR: B300 allocation unavailable' >&2; exit 1; }
    trap 'rc=$?; scancel "$JOB_ID" 2>/dev/null || true; exit "$rc"' EXIT
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
