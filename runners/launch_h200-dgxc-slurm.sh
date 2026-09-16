#!/usr/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/../benchmarks/benchmark_lib.sh" --validation-only || exit 1
check_env_vars EVAL_ONLY IS_MULTINODE REQUIRE_POWER RUN_EVAL
set -eo pipefail

SLURM_PARTITION="main"
SLURM_ACCOUNT="sa-shared"
check_env_vars HF_HUB_CACHE_MOUNT
check_env_vars AIPERF_MMAP_CACHE_HOST_PATH
DSV4_MODEL_REPO="deepseek-ai/DeepSeek-V4-Pro-0813"


set -x

source "$(dirname "${BASH_SOURCE[0]}")/slurm_utils.sh" || exit 1

if [[ "$IS_MULTINODE" == "true" ]]; then

    if [[ -z "${CONFIG_FILE:-}" ]]; then
        echo "Error: CONFIG_FILE is not set. The srt-slurm path requires a CONFIG_FILE in additional-settings." >&2
        exit 1
    fi
    CONFIG_PATH="${CONFIG_FILE%%:*}"
    LOCAL_CONFIG_FILE="$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/${CONFIG_PATH#recipes/}"

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

    USES_KIMIK3_POWER=0
    if [[ "$USES_DCGM_POWER" == "1" && "$IS_AGENTIC" == "1" &&
        "$MODEL_PREFIX" == "kimik3" && "$PRECISION" == "fp4" && "$FRAMEWORK" == "vllm" ]]; then
        USES_KIMIK3_POWER=1

    elif [[ "$USES_DCGM_POWER" == "1" && (
        "$IS_AGENTIC" != "1" ||
        "$FRAMEWORK" != "dynamo-sglang" ||
        ( "$MODEL_PREFIX" != "glm5.2" && "$MODEL_PREFIX" != "dsv4" ) ||
        "$PRECISION" != "fp8"
    ) ]]; then
        echo "Error: H200 dcgm-power requires AgentX dynamo-sglang glm5.2/dsv4 FP8 or Kimi-K3 vLLM FP4" >&2
        exit 1
    fi

    # Recipes name HF model IDs; resolve them to pre-staged paths so the shared
    # cluster does not re-download. SRT_SLURM_MODEL_PREFIX must match the
    # recipe's model.path alias.
    if [[ $FRAMEWORK == "dynamo-sglang" ]]; then
        if [[ $MODEL_PREFIX == "dsv4" && $PRECISION == "fp8" ]]; then
            # Stage the dated checkpoint into shared storage below before
            # srtctl preflight. DSV4_MODEL_PATH remains available for clusters
            # that manage the checkpoint out of band.
            if [[ -n "${DSV4_MODEL_PATH:-}" ]]; then
                export MODEL_PATH="$DSV4_MODEL_PATH"
                DSV4_STAGE_MODEL=0
            else
                export MODEL_PATH="${HF_HUB_CACHE_MOUNT}/DeepSeek-V4-Pro-0813"
                DSV4_STAGE_MODEL=1
            fi
            export SRT_SLURM_MODEL_PREFIX="deepseek-v4-pro-0813"
        elif [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp8" ]]; then
            export MODEL_PATH="/models/DeepSeek-R1-0528"
            export SRT_SLURM_MODEL_PREFIX="dsr1-fp8"
        elif [[ $MODEL_PREFIX == "glm5.2" && $PRECISION == "fp8" ]]; then
            check_env_vars GLM52_FP8_MODEL_PATH
            export MODEL_PATH="${GLM52_FP8_MODEL_PATH}"
            if [[ ! -d "$MODEL_PATH" ]]; then
                export MODEL_PATH="hf:zai-org/GLM-5.2-FP8"
            fi
            export SRT_SLURM_MODEL_PREFIX="glm5.2-fp8"
        else
            echo "Unsupported model prefix/precision for dynamo-sglang: $MODEL_PREFIX/$PRECISION"
            exit 1
        fi
    elif [[ $FRAMEWORK == "dynamo-trt" ]]; then
        if [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp8" ]]; then
            export MODEL_PATH="/models/DeepSeek-R1-0528"
            export SERVED_MODEL_NAME="DeepSeek-R1-0528"
            export SRT_SLURM_MODEL_PREFIX="DeepSeek-R1-0528"
        else
            echo "Unsupported model prefix/precision for dynamo-trt: $MODEL_PREFIX/$PRECISION"
            exit 1
        fi
    elif [[ $FRAMEWORK == "vllm" ]]; then
        if [[ $MODEL_PREFIX == "kimik3" && $PRECISION == "fp4" ]]; then
            export MODEL_PATH="/models/gharunners/hf-hub-cache/Kimi-K3"
            export SRT_SLURM_MODEL_PREFIX="kimik3"
        else
            echo "Unsupported model prefix/precision for vllm: $MODEL_PREFIX/$PRECISION"
            exit 1
        fi
    else
        echo "Unsupported framework: $FRAMEWORK. Supported frameworks are: dynamo-trt, dynamo-sglang, vllm"
        exit 1
    fi

    echo "Preparing job-local srt-slurm checkout..."
    SRT_REPO_DIR="srt-slurm"
    if [ -d "$SRT_REPO_DIR" ]; then
        echo "Removing existing $SRT_REPO_DIR..."
        rm -rf "$SRT_REPO_DIR"
    fi

    setup_srt_slurm "$SRT_REPO_DIR" "$FRAMEWORK" "$USES_DCGM_POWER" || exit 1

    echo "Installing srtctl..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env

    uv venv
    source .venv/bin/activate
    uv pip install -e .

    # A full sweep starts several independent runner jobs at once. Serialize
    # the initial DSV4 download into the shared model directory so those jobs
    # cannot race on Hugging Face's per-file locks. The completion marker is
    # written only after `hf download` verifies every repository file, making
    # interrupted downloads resumable by the next job.
    if [[ $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "dsv4" && $PRECISION == "fp8" && $DSV4_STAGE_MODEL == "1" ]]; then
        DSV4_MODEL_READY="${MODEL_PATH}/.inference-max-download-complete"
        DSV4_MODEL_LOCK="${MODEL_PATH}.download.lock"
        if [[ ! -f "$DSV4_MODEL_READY" ]]; then
            uv pip install huggingface-hub
            mkdir -p "$(dirname "$MODEL_PATH")"
            (
                exec 9>"$DSV4_MODEL_LOCK"
                flock -w 14400 9 || { echo "Error: Timed out waiting for $DSV4_MODEL_LOCK" >&2; exit 1; }
                if [[ ! -f "$DSV4_MODEL_READY" ]]; then
                    hf download "$DSV4_MODEL_REPO" --local-dir "$MODEL_PATH"
                    touch "$DSV4_MODEL_READY"
                fi
            )
        fi
    fi
    if [[ $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "dsv4" && $PRECISION == "fp8" ]]; then
        test -r "$MODEL_PATH/config.json" || { echo "Error: DSV4 model path is unavailable: $MODEL_PATH" >&2; exit 1; }
    fi

    if ! command -v srtctl &> /dev/null; then
        echo "Error: Failed to install srtctl"
        exit 1
    fi

    echo "Configs available at: $SRT_REPO_DIR/"

    NGINX_SQUASH_FILE="/data/containers/nginx+1.27.4.sqsh"

    if [[ $FRAMEWORK == "dynamo-sglang" ]]; then
        if [[ $MODEL_PREFIX == "glm5.2" ]]; then
            SQUASH_FILE="/data/gharunners/containers/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
        else
            SQUASH_FILE="/data/containers/$(echo "$IMAGE" | sed 's/[\/:@#]/+/g').sqsh"
        fi
        CONTAINER_KEY="$IMAGE"
    elif [[ $FRAMEWORK == "dynamo-trt" ]]; then
        CONTAINER_KEY=$(echo "$IMAGE" | sed 's|nvcr.io/|nvcr.io#|')
        SQUASH_FILE="/data/containers/$(echo "$IMAGE" | sed 's|nvcr.io/||' | sed 's/[\/:@#]/+/g').sqsh"
    elif [[ $FRAMEWORK == "vllm" ]]; then
        CONTAINER_KEY="$IMAGE"
        SQUASH_FILE="/data/gharunners/containers/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    fi

    if [[ $FRAMEWORK == "dynamo-sglang" && (
        $MODEL_PREFIX == "glm5.2" || $MODEL_PREFIX == "dsv4"
    ) ]] && ! unsquashfs -l "$SQUASH_FILE" >/dev/null 2>&1; then
        DOCKER_IMAGE=$(echo "$IMAGE" | sed 's/#/\//g')
        LOCK_FILE="${SQUASH_FILE}.lock"
        mkdir -p "$(dirname "$SQUASH_FILE")"
        srun --partition="$SLURM_PARTITION" --account="$SLURM_ACCOUNT" \
            --nodes=1 --ntasks=1 --time=30 --job-name="$RUNNER_NAME" \
            bash -c "
                set -eo pipefail
                exec 9>\"$LOCK_FILE\"
                flock -w 1800 9
                if unsquashfs -l \"$SQUASH_FILE\" >/dev/null 2>&1; then
                    exit 0
                fi
                rm -f \"$SQUASH_FILE\"
                export ENROOT_CACHE_PATH=\${HOME}/.cache/enroot
                mkdir -p \"\$ENROOT_CACHE_PATH\"
                enroot import -o \"$SQUASH_FILE\" docker://$DOCKER_IMAGE
            "
    fi
    if [[ $FRAMEWORK == "dynamo-sglang" && (
        $MODEL_PREFIX == "glm5.2" || $MODEL_PREFIX == "dsv4"
    ) ]]; then
        test -r "$SQUASH_FILE" || { echo "Error: SGLang squash is not readable: $SQUASH_FILE" >&2; exit 1; }
        unsquashfs -l "$SQUASH_FILE" >/dev/null || { echo "Error: SGLang squash is invalid: $SQUASH_FILE" >&2; exit 1; }
    fi

    if [[ "$USES_DCGM_POWER" == "1" ]]; then
        DCGM_EXPORTER_IMAGE="nvcr.io/nvidia/k8s/dcgm-exporter:4.6.0-4.8.3-distroless"
        # enroot resolves bare paths against Docker Hub; nvcr.io pulls need the registry# form
        DCGM_EXPORTER_ENROOT_REF="${DCGM_EXPORTER_IMAGE/nvcr.io\//nvcr.io#}"
        DCGM_EXPORTER_SQSH="/data/gharunners/containers/$(echo "$DCGM_EXPORTER_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
        if ! unsquashfs -l "$DCGM_EXPORTER_SQSH" >/dev/null 2>&1; then
            DCGM_EXPORTER_LOCK="${DCGM_EXPORTER_SQSH}.lock"
            mkdir -p "$(dirname "$DCGM_EXPORTER_SQSH")"
            srun --partition="$SLURM_PARTITION" --account="$SLURM_ACCOUNT" \
                --nodes=1 --ntasks=1 --time=30 --job-name="$RUNNER_NAME" \
                bash -c "
                    set -eo pipefail
                    exec 9>\"$DCGM_EXPORTER_LOCK\"
                    flock -w 1800 9
                    if unsquashfs -l \"$DCGM_EXPORTER_SQSH\" >/dev/null 2>&1; then
                        exit 0
                    fi
                    rm -f \"$DCGM_EXPORTER_SQSH\"
                    export ENROOT_CACHE_PATH=\${HOME}/.cache/enroot
                    mkdir -p \"\$ENROOT_CACHE_PATH\"
                    enroot import -o \"$DCGM_EXPORTER_SQSH\" \"docker://$DCGM_EXPORTER_ENROOT_REF\"
                "
        fi
        test -r "$DCGM_EXPORTER_SQSH" || { echo "Error: DCGM exporter squash is not readable: $DCGM_EXPORTER_SQSH" >&2; exit 1; }
        unsquashfs -l "$DCGM_EXPORTER_SQSH" >/dev/null || { echo "Error: DCGM exporter squash is invalid: $DCGM_EXPORTER_SQSH" >&2; exit 1; }
        sha256sum "$DCGM_EXPORTER_SQSH" > "$GITHUB_WORKSPACE/exporter-image.sha256"
    fi

    export ISL="$ISL"
    export OSL="$OSL"

    SRTCTL_ROOT="${GITHUB_WORKSPACE}/${SRT_REPO_DIR}"
    DEFAULT_MOUNTS_BLOCK=""
    if [[ "$IS_AGENTIC" == "1" ]]; then
        AIPERF_MMAP_CACHE_HOST_PATH="/home/sa-shared/gharunners/ai-perf-cache"
        HF_HUB_CACHE_HOST_PATH="/models/gharunners/hf-hub-cache"
        mkdir -p "$AIPERF_MMAP_CACHE_HOST_PATH"
        DEFAULT_MOUNTS_BLOCK="default_mounts:
  ${AIPERF_MMAP_CACHE_HOST_PATH}: /aiperf_mmap_cache
  ${HF_HUB_CACHE_HOST_PATH}: /hf_hub_cache"
    fi
    echo "Creating srtslurm.yaml configuration..."
    SRT_DEFAULT_TIME_LIMIT="4:00:00"
    if [[ "$IS_AGENTIC" == "1" && "$MODEL_PREFIX" == "dsv4" && "$FRAMEWORK" == "dynamo-sglang" ]]; then
        SRT_DEFAULT_TIME_LIMIT="8:00:00"
    fi
    cat > srtslurm.yaml <<EOF
# SRT SLURM Configuration for H200

# Default SLURM settings
default_account: "${SLURM_ACCOUNT}"
default_partition: "${SLURM_PARTITION}"
default_time_limit: "${SRT_DEFAULT_TIME_LIMIT}"
# Resource defaults
gpus_per_node: 8
network_interface: ""
# Path to srtctl repo root (where the configs live)
srtctl_root: "${SRTCTL_ROOT}"
# Persistent AgentX dataset and Hugging Face caches mounted into every
# server and benchmark container.
default_mounts:
  "${AIPERF_MMAP_CACHE_HOST_PATH}": "/aiperf_mmap_cache"
  "${HF_HUB_CACHE_MOUNT}": "/hf_hub_cache"
# Model path aliases
model_paths:
  "${SRT_SLURM_MODEL_PREFIX}": "${MODEL_PATH}"
  "${MODEL_PREFIX}": "${MODEL_PATH}"
containers:
  dynamo-trtllm: "${SQUASH_FILE}"
  dynamo-sglang: "${SQUASH_FILE}"
  dynamo-vllm: "${SQUASH_FILE}"
  nginx-sqsh: "${NGINX_SQUASH_FILE}"
  latest: "${SQUASH_FILE}"
  "${CONTAINER_KEY}": "${SQUASH_FILE}"
# SLURM directive compatibility
use_gpus_per_node_directive: true
use_segment_sbatch_directive: false
use_exclusive_sbatch_directive: false
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

    if [[ -f "$LOCAL_CONFIG_FILE" ]]; then
        mkdir -p "$(dirname "$CONFIG_PATH")"
        cp "$LOCAL_CONFIG_FILE" "$CONFIG_PATH"
    fi

    if [[ "$USES_DCGM_POWER" == "1" ]]; then
        read -r -a POWER_CONCURRENCIES <<< "$CONC_LIST"
        python "$GITHUB_WORKSPACE/runners/inject_srt_power_concurrencies.py" \
            "$CONFIG_PATH" "${POWER_CONCURRENCIES[@]}"
    fi

    # Read by srt-slurm's post-benchmark eval.
    export INFMAX_WORKSPACE="$GITHUB_WORKSPACE"

    echo "Submitting job with srtctl..."

    sed -i "s/^name:.*/name: \"${RUNNER_NAME}\"/" "$CONFIG_PATH"
    sed -i '/^health_check:/,/^[^ ]/{ /^health_check:/d; /^  /d; }' "$CONFIG_PATH"
    printf '\nhealth_check:\n  max_attempts: 720\n  interval_seconds: 10\n' >> "$CONFIG_PATH"
    if [[ "${EVAL_ONLY}" == "true" ]]; then
        python3 "$GITHUB_WORKSPACE/runners/inject_synthetic_acceptance.py" \
            "$CONFIG_PATH" "$FRAMEWORK" || exit 1
    fi
    WORKLOAD_TAG="${ISL}x${OSL}"
    if [[ "$IS_AGENTIC" == "1" ]]; then
        WORKLOAD_TAG="agentic"
    fi
    SRTCTL_OUTPUT=$(srtctl apply "${SRTCTL_EVAL_ARGS[@]}" -f "$CONFIG_FILE" --tags "h200,${MODEL_PREFIX},${PRECISION},${WORKLOAD_TAG},infmax-$(date +%Y%m%d)" 2>&1)
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
    trap 'rc=$?; bundle_server_logs "$LOGS_DIR" "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz"; scancel "$JOB_ID" 2>/dev/null || true; exit "$rc"' EXIT INT TERM HUP

    SRT_JOB_RC=0
    stream_slurm_job_log "$JOB_ID" "$LOG_FILE" || SRT_JOB_RC=$?
    if [[ "$SRT_JOB_RC" != "0" && "$USES_KIMIK3_POWER" != "1" ]]; then
        exit "$SRT_JOB_RC"
    fi

    set -x

    echo "Job $JOB_ID completed!"
    echo "Collecting results..."

    if [ ! -d "$LOGS_DIR" ]; then
        echo "Warning: Logs directory not found at $LOGS_DIR"
        exit 1
    fi

    echo "Found logs directory: $LOGS_DIR"

    AGENTX_POWER_RC="$SRT_JOB_RC"
    if [[ "$USES_KIMIK3_POWER" == "1" && "${EVAL_ONLY}" != "true" ]]; then
        read -r -a POWER_CONCURRENCIES <<< "$CONC_LIST"
        collect_agentic_power_results "$JOB_ID" "$LOGS_DIR" \
            "$GITHUB_WORKSPACE" "$GITHUB_WORKSPACE" "$RESULT_FILENAME" \
            "$SRT_SLURM_COMMIT" "${POWER_CONCURRENCIES[@]}" || AGENTX_POWER_RC=$?
    elif [[ "$USES_DCGM_POWER" == "1" && "${EVAL_ONLY}" != "true" ]]; then
        POWER_LOGS_ROOT=$(cd "$LOGS_DIR" && pwd -P)
        read -r -a POWER_CONCURRENCIES <<< "$CONC_LIST"
        for concurrency in "${POWER_CONCURRENCIES[@]}"; do
            power_args=(
                --result-dir "$POWER_LOGS_ROOT/agentic/conc_${concurrency}"
                --agg-result "$GITHUB_WORKSPACE/${RESULT_FILENAME}_conc${concurrency}.json"
                --power-dir "$POWER_LOGS_ROOT/power"
                --logs-root "$POWER_LOGS_ROOT"
                --expected-producer-sha "$SRT_SLURM_COMMIT"
            )
            case "${REQUIRE_POWER}" in
                1|true|TRUE|yes|YES) power_args+=(--require-power) ;;
            esac
            (
                cd "$GITHUB_WORKSPACE"
                python -m infx.results.agentic.power_adapter "${power_args[@]}"
            ) || AGENTX_POWER_RC=$?
        done
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

else
    SQUASH_FILE="/data/containers/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

    # enroot import wants nvcr.io/path, not the pyxis nvcr.io#path spelling.
    DOCKER_IMAGE=$(echo "$IMAGE" | sed 's/#/\//g')
    LOCK_FILE="${SQUASH_FILE}.lock"

    check_env_vars GPU_COUNT

    salloc --partition=$SLURM_PARTITION --account=$SLURM_ACCOUNT --gres=gpu:$GPU_COUNT --exclusive --time=180 --no-shell --job-name="$RUNNER_NAME"
    JOB_ID=$(squeue --name="$RUNNER_NAME" -u "$USER" -h -o %A | head -n1)
    if [[ -z "$JOB_ID" ]]; then
        echo "ERROR: failed to resolve H200 Slurm allocation" >&2
        exit 1
    fi
    trap 'rc=$?; scancel "$JOB_ID" 2>/dev/null || true; exit "$rc"' EXIT

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
            enroot import -o \"$SQUASH_FILE\" docker://$DOCKER_IMAGE
        fi
    "

    SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')
    BENCH_BASE="benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_h200"
    BENCH_SCRIPT="${BENCH_BASE}_${FRAMEWORK}${SPEC_SUFFIX}.sh"
    if [[ ! -f "$BENCH_SCRIPT" ]]; then
        LEGACY_FW_SUFFIX=$([[ "$FRAMEWORK" == "trt" ]] && printf '_trt' || printf '')
        BENCH_SCRIPT="${BENCH_BASE}${LEGACY_FW_SUFFIX}${SPEC_SUFFIX}.sh"
    fi

    # DeepSeek-V4.1-Flash creates AgentX runtime directories next to the
    # repository, which must not land under /workspace.
    if [[ "$IMAGE" == *deepseek-v4-hopper* || "$MODEL_PREFIX" == "dsv41flash" ]]; then
        CONTAINER_MOUNT_DIR=/ix
    else
        CONTAINER_MOUNT_DIR=/workspace
    fi
    if [[ "$MODEL_PREFIX" == "dsv41flash" ]]; then
        # Cover DSpark5 verification for concurrent AgentX subagents at c1/c2/c4.
        export DSV41_MIN_CUDAGRAPH_CAPTURE_SIZE=64
        export INFMAX_CONTAINER_WORKSPACE=/ix
        export RESULT_DIR=/ix/results
        # The HF cache here is a VIRTIOFS mount, which vLLM does not treat as a
        # network FS, so it memory-maps the 475 GiB checkpoint lazily. On the
        # 2026-09-15 nightly that path loaded 19/48 shards in the 3600 s
        # readiness window (run 35012494184). Stream the shards into page cache
        # with parallel readers first, and give cold loads the same two-hour
        # deadline the GB300 launcher uses.
        export VLLM_SAFETENSORS_LOAD_STRATEGY=prefetch
        export VLLM_ENGINE_READY_TIMEOUT_S=7200
    fi

    srun --jobid=$JOB_ID \
        --container-image=$SQUASH_FILE \
        --container-mounts=$GITHUB_WORKSPACE:$CONTAINER_MOUNT_DIR/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE,$AIPERF_MMAP_CACHE_HOST_PATH:/aiperf_mmap_cache \
        --no-container-mount-home \
        --container-remap-root \
        --container-workdir=$CONTAINER_MOUNT_DIR/ \
        --no-container-entrypoint --export=ALL,PORT=8888,AIPERF_DATASET_MMAP_CACHE_DIR=/aiperf_mmap_cache \
        bash $BENCH_SCRIPT

    scancel $JOB_ID

fi
