#!/usr/bin/bash

# B200 Nscale launcher.
#
# The reusable workflows run runners/launch_${RUNNER_NAME%%_*}.sh, so every
# b200-nscale-slurm_* runner enters here and this is the pool's only launcher.
# Three execution paths share the file and are selected once, below:
#   native-srt     multi-node lanes whose srt-slurm recipes are maintained
#                  against this cluster (DSV4 / Kimi K2.6 / Kimi K3 / GLM-5.2
#                  FP4 and GLM-5.1 FP8 TileRT)
#   multinode-srt  every other multi-node job, through srt-slurm with the
#                  cluster-wide model table
#   single-node    salloc + srun of the benchmarks/single_node script
source "$(dirname "${BASH_SOURCE[0]}")/../benchmarks/benchmark_lib.sh" --validation-only || exit 1
check_env_vars EVAL_ONLY IS_AGENTIC IS_MULTINODE RUN_EVAL
# Exported for this pool by runners/runtime_settings.sh.
check_env_vars SLURM_PARTITION SLURM_ACCOUNT

# shellcheck source=runners/slurm_utils.sh
source "$(dirname "${BASH_SOURCE[0]}")/slurm_utils.sh" || exit 1

set -x

export AIPERF_MMAP_CACHE_HOST_PATH="/data/home/sa-shared/gharunners/aiperf-cache"

# ---------------------------------------------------------------------------
# Path selection
# ---------------------------------------------------------------------------

uses_native_srt_lane() {
    [[ "$IS_MULTINODE" == "true" ]] || return 1
    if [[ "$FRAMEWORK" == "tilert" && "${IS_AGENTIC}" != "1" ]]; then
        return 1
    fi
    case "${MODEL_PREFIX}/${PRECISION}" in
        dsv4/fp4|kimik2.6/fp4|kimik3/fp4|glm5.2/fp4) ;;
        glm5.1/fp8) [[ "$FRAMEWORK" == "tilert" ]] || return 1 ;;
        *) return 1 ;;
    esac
    [[ "$FRAMEWORK" == "dynamo-vllm" ]] && return 0
    [[ "$MODEL_PREFIX" == "dsv4" && "$PRECISION" == "fp4" && "$FRAMEWORK" == "dynamo-sglang" &&
       ( "$SPEC_DECODING" == "none" || "$SPEC_DECODING" == "mtp" ) ]] && return 0
    [[ "$MODEL_PREFIX" == "glm5.2" && "$PRECISION" == "fp4" && "$FRAMEWORK" == "dynamo-sglang" && "$SPEC_DECODING" == "mtp" ]] && return 0
    [[ "$MODEL_PREFIX" == "glm5.1" && "$PRECISION" == "fp8" && "$FRAMEWORK" == "tilert" && "$SPEC_DECODING" == "mtp" ]] && return 0
    return 1
}

if uses_native_srt_lane; then
    LAUNCH_PATH="native-srt"
elif [[ "$IS_MULTINODE" == "true" ]]; then
    LAUNCH_PATH="multinode-srt"
else
    LAUNCH_PATH="single-node"
fi
echo "B200 Nscale launch path: $LAUNCH_PATH"

# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------
# Bench scripts and srt-slurm recipes name HF model IDs; resolve them to paths
# pre-staged on the compute nodes' local NVMe so no node re-downloads (the
# ~1.6T DeepSeek-V4-Pro load is much faster from there than from a shared
# filesystem). SRT_SLURM_MODEL_PREFIX must match the recipe's model.path alias.

if [[ "$LAUNCH_PATH" == "native-srt" ]]; then
    case "${MODEL_PREFIX}/${PRECISION}" in
        dsv4/fp4)
            check_env_vars MODEL_PATH
            export SRT_SLURM_MODEL_PREFIX="deepseek-v4-pro"
            ;;
        kimik2.6/fp4)
            check_env_vars MODEL_PATH
            export SRT_SLURM_MODEL_PREFIX="kimi-k2.6-nvfp4"
            ;;
        kimik3/fp4)
            check_env_vars MODEL_PATH
            export SRT_SLURM_MODEL_PREFIX="kimik3"
            ;;
        glm5.2/fp4)
            check_env_vars MODEL_PATH
            # This alias must match model.path in the checked-in GLM-5.2 recipes.
            export SRT_SLURM_MODEL_PREFIX="glm-5.2-fp4"
            ;;
        glm5.1/fp8)
            export SRT_SLURM_MODEL_PREFIX="glm5.1-fp8"
            ;;
    esac
elif [[ "$MODEL_PREFIX" == "dsv41flash" && "$PRECISION" == "fp4" && "$FRAMEWORK" == "vllm" && "$IS_MULTINODE" != "true" ]]; then
    export MODEL_PATH="$MODEL"
    export HF_HUB_CACHE_HOST_PATH="/data/home/sa-shared/gharunners/hf-hub-cache"
    mkdir -p "$HF_HUB_CACHE_HOST_PATH"
elif [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH="/scratch/models/DeepSeek-R1-0528-NVFP4-v2"
    export SRT_SLURM_MODEL_PREFIX="dsr1"
elif [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp8" ]]; then
    export MODEL_PATH="/scratch/models/DeepSeek-R1-0528"
    export SRT_SLURM_MODEL_PREFIX="dsr1-fp8"
elif [[ $MODEL_PREFIX == "dsv4" && $PRECISION == "fp4" && $MODEL == "deepseek-ai/DeepSeek-V4-Pro-0813" ]]; then
    check_env_vars MODEL_PATH
elif [[ $MODEL_PREFIX == "dsv4" && $PRECISION == "fp4" ]]; then
    # Node-local weights are not visible on the runner/login node.
    export MODEL_PATH="/scratch/models/DeepSeek-V4-Pro-NVFP4"
    export SRT_SLURM_MODEL_PREFIX="deepseek-v4-pro"
elif [[ $MODEL_PREFIX == "qwen3.5" && $PRECISION == "bf16" ]]; then
    export MODEL_PATH="/scratch/models/Qwen3.5-397B-A17B"
    export SRT_SLURM_MODEL_PREFIX="qwen3.5"
elif [[ $MODEL_PREFIX == "qwen3.5" && $PRECISION == "fp8" ]]; then
    export MODEL_PATH="/scratch/models/Qwen3.5-397B-A17B-FP8"
    export SRT_SLURM_MODEL_PREFIX="qwen3.5-fp8"
# qwen3.5 fp4 spans two checkpoints: sglang keys moved to NVFP4-V2 while the TRT
# configs still declare plain NVFP4. Branch on the checkpoint, because the
# `export MODEL="$MODEL_PATH"` in the single-node path would otherwise serve V2
# weights under the old name.
elif [[ $MODEL_PREFIX == "qwen3.5" && $PRECISION == "fp4" && $MODEL == *NVFP4-V2 ]]; then
    export MODEL_PATH="/scratch/models/Qwen3.5-397B-A17B-NVFP4-V2"
    export SRT_SLURM_MODEL_PREFIX="qwen3.5-fp4"
elif [[ $MODEL_PREFIX == "qwen3.5" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH="/scratch/models/Qwen3.5-397B-A17B-NVFP4"
    export SRT_SLURM_MODEL_PREFIX="qwen3.5-fp4"
elif [[ $MODEL_PREFIX == "glm5" && $PRECISION == "fp8" ]]; then
    export MODEL_PATH="/scratch/models/GLM-5-FP8"
    export SRT_SLURM_MODEL_PREFIX="glm5-fp8"
elif [[ $MODEL_PREFIX == "glm5.1" && $PRECISION == "fp8" ]]; then
    check_env_vars MODEL_PATH
    export SRT_SLURM_MODEL_PREFIX="glm5.1-fp8"
elif [[ $MODEL_PREFIX == "glm5" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH="/scratch/models/GLM-5-NVFP4"
    export SRT_SLURM_MODEL_PREFIX="glm5-fp4"
elif [[ $MODEL_PREFIX == "glm5.2" && $PRECISION == "fp4" ]]; then
    check_env_vars MODEL_PATH
    export SRT_SLURM_MODEL_PREFIX="glm5.2-fp4"
elif [[ $MODEL_PREFIX == "glm5.2" && $PRECISION == "fp8" ]]; then
    export MODEL_PATH="${MODEL_PATH:-/scratch/models/GLM-5.2-FP8}"
    export SRT_SLURM_MODEL_PREFIX="glm5.2-fp8"
elif [[ $MODEL_PREFIX == "kimik2.5" && $PRECISION == "int4" ]]; then
    export MODEL_PATH="/scratch/models/Kimi-K2.5"
    export SRT_SLURM_MODEL_PREFIX="kimik2.5"
elif [[ $MODEL_PREFIX == "kimik2.5" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH="/scratch/models/Kimi-K2.5-NVFP4"
    export SRT_SLURM_MODEL_PREFIX="kimik2.5-fp4"
elif [[ $MODEL_PREFIX == "kimik2.6" && $PRECISION == "fp4" ]]; then
    check_env_vars MODEL_PATH
    export SRT_SLURM_MODEL_PREFIX="kimi-k2.6-nvfp4"
elif [[ $MODEL_PREFIX == "minimaxm2.5" && $PRECISION == "fp8" ]]; then
    export MODEL_PATH="/scratch/models/MiniMax-M2.5"
    export SRT_SLURM_MODEL_PREFIX="minimax-m2.5-fp8"
elif [[ $MODEL_PREFIX == "minimaxm2.5" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH="/scratch/models/MiniMax-M2.5-NVFP4"
    export SRT_SLURM_MODEL_PREFIX="minimax-m2.5-nvfp4"
elif [[ $MODEL_PREFIX == "gptoss" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH="/scratch/models/gpt-oss-120b"
    export SRT_SLURM_MODEL_PREFIX="gptoss"
elif [[ $MODEL_PREFIX == "minimaxm3" && $PRECISION == "fp8" ]]; then
    export MODEL_PATH="/scratch/models/MiniMax-M3-MXFP8"
    export SRT_SLURM_MODEL_PREFIX="minimax-m3-mxfp8"
elif [[ $MODEL_PREFIX == "minimaxm3" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH="/scratch/models/MiniMax-M3-NVFP4"
    export SRT_SLURM_MODEL_PREFIX="nvidia/MiniMax-M3-NVFP4"
elif [[ $MODEL_PREFIX == "kimik3" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH="/scratch/models/Kimi-K3"
    export SRT_SLURM_MODEL_PREFIX="kimik3"
elif [[ $MODEL_PREFIX == "qwen3.8next" && $PRECISION == "fp4" ]]; then
    check_env_vars MODEL_PATH
    if [[ -n "${MODEL_PATH}" && -d "$MODEL_PATH" ]]; then
        :
    else
        export MODEL_PATH="/scratch/models/Qwen3.8-Flash-Next-NVFP4"
    fi
    export SRT_SLURM_MODEL_PREFIX="qwen3.8next-fp4"
else
    echo "Unsupported model prefix/precision: $MODEL_PREFIX/$PRECISION"
    echo "Available models under /scratch/models:"
    ls -la /scratch/models
    exit 1
fi

# ---------------------------------------------------------------------------
# Container import helpers shared by both srt-slurm paths
# ---------------------------------------------------------------------------
# Callers set SQUASH_DIR and SQUASH_LOCK_TIMEOUT before importing.

# Enroot treats docker://foo/bar as a Docker Hub path. Preserve that form for
# ordinary Docker Hub images, but use Enroot's explicit registry separator for
# fully qualified references such as ghcr.io/tile-ai/tilert or nvcr.io/....
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

# Use a workspace-local squash cache when the shared one is not writable.
ensure_writable_squash_dir() {
    if ! mkdir -p "$SQUASH_DIR" 2>/dev/null || [[ ! -w "$SQUASH_DIR" ]]; then
        echo "Warning: $SQUASH_DIR is not writable; using workspace-local squash cache" >&2
        SQUASH_DIR="$GITHUB_WORKSPACE/.container-squash"
        mkdir -p "$SQUASH_DIR"
    fi
    chmod a+rx "$SQUASH_DIR" || true
}

# ---------------------------------------------------------------------------
# native-srt: cluster-maintained multi-node srt-slurm lanes
# ---------------------------------------------------------------------------

run_native_srt_lane() {
    SQUASH_DIR="/data/home/sa-shared/containers"
    HF_HUB_CACHE_HOST_PATH="/data/home/sa-shared/gharunners/hf-hub-cache"
    # Importing the vLLM image over this cluster's shared home can take a while.
    SQUASH_LOCK_TIMEOUT=3600

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
    ensure_writable_squash_dir

    SQUASH_FILE="$SQUASH_DIR/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    NGINX_SQUASH_FILE="$SQUASH_DIR/$(echo "$NGINX_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

    import_squash "$SQUASH_FILE" "$IMAGE" || exit 1
    import_squash "$NGINX_SQUASH_FILE" "$NGINX_IMAGE" || exit 1

    PREFILL_SQUASH_FILE=""
    SRT_CLUSTER_ARGS=()
    if [[ $FRAMEWORK == "tilert" ]]; then
        : "${PREFILL_IMAGE:?PREFILL_IMAGE is required for TileRT prefill}"
        PREFILL_SQUASH_FILE="$SQUASH_DIR/$(echo "$PREFILL_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
        import_squash "$PREFILL_SQUASH_FILE" "$PREFILL_IMAGE" || exit 1
        SRT_CLUSTER_ARGS+=(
            --container tilert-decode "$SQUASH_FILE"
            --container tilert-prefill "$PREFILL_SQUASH_FILE"
        )
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
    if [[ "$IS_AGENTIC" == "1" ]]; then
        mkdir -p "$AIPERF_MMAP_CACHE_HOST_PATH" "$HF_HUB_CACHE_HOST_PATH"
        chmod 777 "$AIPERF_MMAP_CACHE_HOST_PATH" "$HF_HUB_CACHE_HOST_PATH" 2>/dev/null || true
        SRT_CLUSTER_ARGS+=(
            --mount "$AIPERF_MMAP_CACHE_HOST_PATH" /aiperf_mmap_cache
            --mount "$HF_HUB_CACHE_HOST_PATH" /hf_hub_cache
        )
    fi
    if [[ $FRAMEWORK == "tilert" ]]; then
        TILERT_WEIGHTS_HOST_PATH="/data/home/sa-shared/gharunners/tilert-cache"
        mkdir -p "$TILERT_WEIGHTS_HOST_PATH"
        SRT_CLUSTER_ARGS+=(
            --mount "$GITHUB_WORKSPACE" /infmax-workspace
            --mount "$TILERT_WEIGHTS_HOST_PATH" "$TILERT_WEIGHTS_HOST_PATH"
        )
    fi

    SRTCTL_ROOT="${GITHUB_WORKSPACE}/${SRT_REPO_DIR}"
    echo "Creating srtslurm.yaml configuration..."
    write_srt_cluster_config b200-nscale-slurm srtslurm.yaml "$USES_DCGM_POWER" \
        --model "$SRT_SLURM_MODEL_PREFIX" "$MODEL_PATH" \
        "${SRT_CLUSTER_ARGS[@]}" || exit 1

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

    SRTCTL_OUTPUT=$(apply_srt_recipe "$CONFIG_FILE" "$FRAMEWORK" "${SRTCTL_EVAL_ARGS[@]}" -f "$CONFIG_FILE" "${SRTCTL_PREFLIGHT_ARGS[@]}" --tags "b200,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)" 2>&1)
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
}

# ---------------------------------------------------------------------------
# multinode-srt: every other multi-node job, through srt-slurm
# ---------------------------------------------------------------------------

run_multinode_srt() {
    if [[ "$FRAMEWORK" == "tilert" ]]; then
        export SLURM_PARTITION SLURM_ACCOUNT
        check_env_vars TILERT_WEIGHTS_DIR
        # Nscale exposes eight RoCE HCAs, mlx5_0..mlx5_7.
        check_env_vars UCX_NET_DEVICES
        check_env_vars UCX_MEMTYPE_CACHE
        check_env_vars UCX_MEMTYPE_REG_WHOLE
        TILERT_SUBDIR="multi_node"
        [[ "${SCENARIO_SUBDIR}" == "agentic/" ]] && TILERT_SUBDIR="multi_node/agentic"
        TILERT_DISAGG="$GITHUB_WORKSPACE/benchmarks/${TILERT_SUBDIR}/${EXP_NAME%%_*}_${PRECISION}_b200_${FRAMEWORK}-disagg.sh"
        [[ -f "$TILERT_DISAGG" ]] || { echo "tilert disagg script not found: $TILERT_DISAGG"; exit 1; }
        exec bash "$TILERT_DISAGG"
        exit 1
    fi

    if [[ $FRAMEWORK != "dynamo-sglang" && $FRAMEWORK != "dynamo-trt" && $FRAMEWORK != "dynamo-vllm" ]]; then
        echo "Unsupported framework: $FRAMEWORK. Supported frameworks are: dynamo-trt, dynamo-sglang, dynamo-vllm"
        exit 1
    fi

    if [[ $MODEL_PREFIX == "dsv4" && $FRAMEWORK != "dynamo-vllm" ]]; then
        echo "Unsupported framework for multinode dsv4: $FRAMEWORK (only dynamo-vllm)"
        exit 1
    fi

    USES_DCGM_POWER=0
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
    if [[ "$USES_DCGM_POWER" == "1" && (
        "${IS_AGENTIC}" == "1" ||
        "$MODEL_PREFIX" != "dsv4" ||
        "$PRECISION" != "fp4" ||
        "$FRAMEWORK" != "dynamo-vllm"
    ) ]]; then
        echo "Error: B200 Nscale dcgm-power is limited to fixed-sequence DSV4 FP4 dynamo-vllm" >&2
        exit 1
    fi

    export SERVED_MODEL_NAME=$MODEL

    echo "Preparing job-local srt-slurm checkout..."
    SRT_REPO_DIR="srt-slurm"
    if [ -d "$SRT_REPO_DIR" ]; then
        echo "Removing existing $SRT_REPO_DIR..."
        rm -rf "$SRT_REPO_DIR"
    fi

    setup_srt_slurm "$SRT_REPO_DIR" "$FRAMEWORK" "$USES_DCGM_POWER" || exit 1

    echo "Installing srtctl..."
    export UV_INSTALL_DIR="$GITHUB_WORKSPACE/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$UV_INSTALL_DIR:$PATH"

    if [[ $MODEL_PREFIX == "minimaxm2.5" && $FRAMEWORK == "dynamo-vllm" ]]; then
        uv venv --seed "$GITHUB_WORKSPACE/.venv"
    else
        uv venv "$GITHUB_WORKSPACE/.venv"
    fi
    source "$GITHUB_WORKSPACE/.venv/bin/activate"
    uv pip install -e .

    if ! command -v srtctl &> /dev/null; then
        echo "Error: Failed to install srtctl"
        exit 1
    fi

    NGINX_IMAGE="nginx:1.27.4"
    # Set by runners/runtime_settings.sh (a different squash dir for MiniMax-M2.5 dynamo-vllm).
    check_env_vars B200_SQUASH_DIR B200_SQUASH_LOCK_TIMEOUT
    SQUASH_DIR="${B200_SQUASH_DIR}"
    SQUASH_LOCK_TIMEOUT="${B200_SQUASH_LOCK_TIMEOUT}"
    ensure_writable_squash_dir

    SQUASH_FILE="$SQUASH_DIR/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    NGINX_SQUASH_FILE="$SQUASH_DIR/$(echo "$NGINX_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

    import_squash "$SQUASH_FILE" "$IMAGE" || exit 1
    import_squash "$NGINX_SQUASH_FILE" "$NGINX_IMAGE" || exit 1

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

    # Persistent Lustre caches for aiperf's dataset mmap files and the HF trace
    # dataset; the container paths are referenced by the agentic recipes'
    # benchmark.env.
    SRT_CLUSTER_ARGS=()
    if [[ "$IS_AGENTIC" == "1" ]]; then
        HF_HUB_CACHE_HOST_PATH="/data/home/sa-shared/gharunners/hf-hub-cache"
        mkdir -p "$AIPERF_MMAP_CACHE_HOST_PATH" "$HF_HUB_CACHE_HOST_PATH"
        chmod 777 "$AIPERF_MMAP_CACHE_HOST_PATH" "$HF_HUB_CACHE_HOST_PATH" 2>/dev/null || true
        SRT_CLUSTER_ARGS+=(
            --mount "$AIPERF_MMAP_CACHE_HOST_PATH" /aiperf_mmap_cache
            --mount "$HF_HUB_CACHE_HOST_PATH" /hf_hub_cache
        )
    fi

    SRTCTL_ROOT="${GITHUB_WORKSPACE}/${SRT_REPO_DIR}"
    echo "Creating srtslurm.yaml configuration..."
    write_srt_cluster_config b200-nscale-slurm srtslurm.yaml "$USES_DCGM_POWER" \
        --model "$SRT_SLURM_MODEL_PREFIX" "$MODEL_PATH" \
        --container dynamo-trtllm "$SQUASH_FILE" \
        --container sglang-v0.5.11-cu130 "$SQUASH_FILE" \
        "${SRT_CLUSTER_ARGS[@]}" || exit 1

    echo "Generated srtslurm.yaml:"
    cat srtslurm.yaml

    echo "Running make setup..."
    make setup ARCH=x86_64

    # Read by srt-slurm's post-benchmark eval.
    export INFMAX_WORKSPACE="$GITHUB_WORKSPACE"

    echo "Submitting job with srtctl..."
    echo "MODEL_PATH=$MODEL_PATH (exists=$(test -d "$MODEL_PATH" && echo yes || echo NO))"
    ls -ld "$MODEL_PATH" 2>&1 || ls /scratch/models/ 2>&1 | head -40

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

    sed -i "s/^name:.*/name: \"${RUNNER_NAME}\"/" "${CONFIG_FILE%%:*}"
    # 720x10s health-check budget so large loads (DSR1-FP8 ~680GB off shared FS)
    # finish. CONFIG_FILE may carry an :override[N] suffix.
    sed -i 's/^  max_attempts: [0-9]*/  max_attempts: 720/' "${CONFIG_FILE%%:*}"

    SRTCTL_PREFLIGHT_ARGS=()
    # These weights are staged on the Slurm compute nodes, not the login node.
    # SRT still checks the resolved model path when the worker starts.
    if [[ $FRAMEWORK == "dynamo-vllm" && $MODEL_PREFIX == "kimik2.6" && $PRECISION == "fp4" ]] ||
       [[ $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "qwen3.5" && $PRECISION == "fp8" ]]; then
        SRTCTL_PREFLIGHT_ARGS+=(--no-preflight)
    fi

    SRTCTL_OUTPUT=$(apply_srt_recipe "$CONFIG_FILE" "$FRAMEWORK" "${SRTCTL_EVAL_ARGS[@]}" -f "$CONFIG_FILE" "${SRTCTL_PREFLIGHT_ARGS[@]}" --tags "b200,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)" 2>&1)
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

    local srt_job_rc=0
    stream_slurm_job_log "$JOB_ID" "$LOG_FILE" || srt_job_rc=$?
    if [[ "$srt_job_rc" -eq 0 ]]; then
        verify_slurm_job_status "$JOB_ID" || srt_job_rc=$?
    fi

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

    if [[ "${EVAL_ONLY}" != "true" ]]; then
        copy_fixed_sequence_results "$LOGS_DIR" "$GITHUB_WORKSPACE" "$RESULT_FILENAME" || exit 1
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
    # Failed runs still provide the diagnostics and eval outputs above.
    return "$srt_job_rc"
}

# ---------------------------------------------------------------------------
# single-node: salloc + srun of the benchmarks/single_node script
# ---------------------------------------------------------------------------

run_single_node() {
    # The runner lease reserves the Slurm nodes before this single-node job is
    # submitted to the Nscale batch_1 partition.
    check_env_vars SALLOC_TIME_LIMIT GPU_COUNT

    if [[ "$IS_AGENTIC" == "1" && "$MODEL_PREFIX" == "qwen3.5" &&
          "$PRECISION" == "fp8" && "$FRAMEWORK" == "sglang" && "$SPEC_DECODING" == "mtp" ]]; then
        check_env_vars IMAGE
        export QWEN35_HICACHE_BUDGET_MODE=legacy
        if [[ "$IMAGE" == "lmsysorg/sglang:nightly-dev-cu13-20260918-20518d85" ]]; then
            export QWEN35_HICACHE_BUDGET_MODE=combined
        fi
    fi

    SQUASH_FILE="/data/home/sa-shared/containers/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')
    SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" || "$SPEC_DECODING" == "draft_model" ]] && printf '_mtp' || printf '')
    # Prefer a framework-tagged script (e.g. dsv4_fp4_b200_vllm.sh) so models
    # with multiple inference engines can coexist; fall back to the historical
    # name without an engine suffix (`_trt` for trt, bare for everyone else).
    BENCH_BASE="benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_b200"
    BENCH_SCRIPT="${BENCH_BASE}_${FRAMEWORK}${SPEC_SUFFIX}.sh"
    if [[ ! -f "$BENCH_SCRIPT" ]]; then
        BENCH_SCRIPT="${BENCH_BASE}${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh"
    fi
    LOCK_FILE="${SQUASH_FILE}.lock"

    # TODO(Cam): lmsysorg/sglang:deepseek-v4-blackwell installs sglang editable at
    # /workspace/sglang/python (prior sglang tags used /sgl-workspace/sglang), so
    # the default $GITHUB_WORKSPACE:/workspace/ bind-mount masks the install and
    # breaks `import sglang`. Mount this one image at /ix instead; drop the
    # conditional once the image stops installing editable under /workspace.
    if [[ "$IMAGE" == *deepseek-v4-blackwell* ]]; then
        CONTAINER_MOUNT_DIR=/ix
    else
        CONTAINER_MOUNT_DIR=/workspace
    fi

    if [[ "$MODEL_PREFIX" == "dsv41flash" ]]; then
        # Cover DSpark5 verification for concurrent AgentX subagents at c1/c2/c4.
        export DSV41_MIN_CUDAGRAPH_CAPTURE_SIZE=64
        CONTAINER_MOUNT_DIR=/ix
        export INFMAX_CONTAINER_WORKSPACE=/ix
        export RESULT_DIR=/ix/results
        export HF_HUB_CACHE=/hf-cache
        CONTAINER_MOUNTS="$GITHUB_WORKSPACE:/ix,$HF_HUB_CACHE_HOST_PATH:/hf-cache,$AIPERF_MMAP_CACHE_HOST_PATH:/aiperf_mmap_cache"
    else
        CONTAINER_MOUNTS="$GITHUB_WORKSPACE:$CONTAINER_MOUNT_DIR,$MODEL_PATH:$MODEL_PATH,$AIPERF_MMAP_CACHE_HOST_PATH:/aiperf_mmap_cache"
    fi

    salloc --partition=$SLURM_PARTITION --account=$SLURM_ACCOUNT --gres=gpu:$GPU_COUNT --exclusive --mem=0 --time="$SALLOC_TIME_LIMIT" --no-shell --job-name="$RUNNER_NAME"
    JOB_ID=$(squeue --name="$RUNNER_NAME" -u "$USER" -h -o %A | head -n1)

    # Bench scripts skip `hf download` when MODEL is a local path.
    export MODEL="$MODEL_PATH"

    # Serialize concurrent imports of the same squash file. ENROOT_CACHE_PATH
    # avoids permission issues with the system-wide cache on worker nodes.
    srun --jobid=$JOB_ID bash -c "
        export ENROOT_CACHE_PATH=\$HOME/.cache/enroot
        mkdir -p \$ENROOT_CACHE_PATH
        exec 9>\"$LOCK_FILE\"
        flock -w 600 9 || { echo 'Failed to acquire lock for $SQUASH_FILE'; exit 1; }
        if unsquashfs -l \"$SQUASH_FILE\" > /dev/null 2>&1; then
            echo 'Squash file already exists and is valid, skipping import'
        else
            rm -f \"$SQUASH_FILE\"
            enroot import -o \"$SQUASH_FILE\" docker://$IMAGE
        fi
    "

    srun --jobid=$JOB_ID \
        --container-image=$SQUASH_FILE \
        --container-mounts="$CONTAINER_MOUNTS" \
        --no-container-mount-home \
        --container-workdir=$CONTAINER_MOUNT_DIR \
        --no-container-entrypoint --export=ALL,PORT=8888,AIPERF_DATASET_MMAP_CACHE_DIR=/aiperf_mmap_cache \
        bash "$BENCH_SCRIPT"
}

case "$LAUNCH_PATH" in
    native-srt) run_native_srt_lane ;;
    multinode-srt) run_multinode_srt ;;
    single-node) run_single_node ;;
esac
