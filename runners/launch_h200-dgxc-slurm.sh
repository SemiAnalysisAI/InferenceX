#!/usr/bin/bash
set -eo pipefail

# System-specific configuration for H200 DGXC Slurm cluster
SLURM_PARTITION="main"
SLURM_ACCOUNT="sa-shared"

set -x

# Native multi-node vLLM: keep Slurm/Pyxis on the host and pass only rank and
# rendezvous information to the hardware-independent benchmark entrypoint.
if [[ "$IS_MULTINODE" == "true" && "$NATIVE_MULTINODE" == "1" ]]; then
    if [[ "$IS_AGENTIC" != "1" || "$FRAMEWORK" != "vllm" ]]; then
        echo "Native multi-node H200 currently supports AgentX vLLM jobs" >&2
        exit 1
    fi

    case "$MODEL_PREFIX/$PRECISION" in
        kimik3/fp4)
            model_host_path="/models/gharunners/hf-hub-cache/Kimi-K3"
            export MODEL_PATH="/models/hf-hub-cache/Kimi-K3"
            ;;
        *)
            echo "No native multi-node model mapping for $MODEL_PREFIX/$PRECISION on H200 DGXC" >&2
            exit 1
            ;;
    esac

    hf_cache_host_path="/models/gharunners/hf-hub-cache"
    hf_cache_container_path="/models/hf-hub-cache"
    aiperf_cache_host_path="/home/sa-shared/gharunners/ai-perf-cache"
    image_cache_dir="/data/gharunners/containers"
    server_log_dir="$GITHUB_WORKSPACE/multinode_server_logs"
    gpus_per_node=8
    gpu_count=$((PREFILL_NUM_WORKERS * PREFILL_TP))
    if (( gpu_count % gpus_per_node != 0 )); then
        echo "Native multi-node GPU count ($gpu_count) must use full 8xH200 nodes" >&2
        exit 1
    fi
    node_count=$((gpu_count / gpus_per_node))
    server_script="benchmarks/multi_node/${SCENARIO_SUBDIR}${MODEL_PREFIX}_${PRECISION}_${FRAMEWORK}.sh"
    client_script="benchmarks/multi_node/agentic_srt.sh"
    squash_file="$image_cache_dir/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
    lock_file="${squash_file}.lock"
    container_mounts="$GITHUB_WORKSPACE:/workspace,$hf_cache_host_path:$hf_cache_container_path,$aiperf_cache_host_path:/aiperf_mmap_cache"

    if [[ ! -d "$model_host_path" || ! -f "$server_script" ]]; then
        echo "Missing model or benchmark entrypoint: $model_host_path / $server_script" >&2
        exit 1
    fi
    mkdir -p "$aiperf_cache_host_path" "$image_cache_dir" "$server_log_dir"

    export PORT=8888
    export HF_HUB_CACHE="$hf_cache_container_path"
    export AIPERF_DATASET_MMAP_CACHE_DIR="/aiperf_mmap_cache"
    export INFMAX_CONTAINER_WORKSPACE="/workspace"
    export RESULT_DIR="/workspace/LOGS/agentic"
    export GLOO_SOCKET_IFNAME=eth0
    export NCCL_SOCKET_IFNAME=eth0

    salloc \
        --partition="$SLURM_PARTITION" \
        --account="$SLURM_ACCOUNT" \
        --nodes="$node_count" \
        --ntasks-per-node=1 \
        --gres="gpu:$gpus_per_node" \
        --exclusive \
        --time=240 \
        --no-shell \
        --job-name="$RUNNER_NAME"
    job_id=$(squeue --name="$RUNNER_NAME" -u "$USER" -h -o %A | sed -n '1p')
    if [[ -z "$job_id" ]]; then
        echo "Failed to resolve native multi-node Slurm allocation" >&2
        exit 1
    fi

    cleanup_native_multinode() {
        local exit_code=$?
        trap - EXIT INT TERM
        kill "$server_step_pid" 2>/dev/null || true
        wait "$server_step_pid" 2>/dev/null || true
        tar czf "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz" \
            -C "$server_log_dir" . 2>/dev/null || true
        scancel "$job_id" 2>/dev/null || true
        exit "$exit_code"
    }
    trap cleanup_native_multinode EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    allocated_nodelist=$(squeue -j "$job_id" -h -o %N)
    allocated_nodes=$(scontrol show hostnames "$allocated_nodelist")
    # Slurm task rank order is not guaranteed to match scontrol's hostname order.
    head_node=$(
        srun \
            --jobid="$job_id" \
            --nodes="$node_count" \
            --ntasks="$node_count" \
            --ntasks-per-node=1 \
            bash -c 'if [[ "$SLURM_PROCID" == "0" ]]; then hostname; fi'
    )
    if [[ -z "$head_node" ]]; then
        echo "Failed to resolve native multi-node head node" >&2
        exit 1
    fi
    server_nodes="$head_node"
    if [[ "$PREFILL_DP_ATTN" == "true" ]]; then
        server_nodes="$allocated_nodes"
    fi
    endpoint_urls=
    metrics_urls=
    for server_node in $server_nodes; do
        if [[ -n "$endpoint_urls" ]]; then
            endpoint_urls+=","
            metrics_urls+=","
        fi
        endpoint_urls+="http://$server_node:$PORT"
        metrics_urls+="http://$server_node:$PORT/metrics"
    done
    export AIPERF_ENDPOINT_URLS="$endpoint_urls"
    export AIPERF_SERVER_METRICS_URLS="$metrics_urls"

    srun --jobid="$job_id" --nodes=1 --ntasks=1 --nodelist="$head_node" bash -c "
        export ENROOT_CACHE_PATH=\$HOME/.cache/enroot
        mkdir -p \$ENROOT_CACHE_PATH
        exec 9>'$lock_file'
        flock -w 1800 9
        if ! unsquashfs -l '$squash_file' >/dev/null 2>&1; then
            rm -f '$squash_file'
            enroot import -o '$squash_file' 'docker://${IMAGE//#//}'
        fi
    "

    export MULTINODE_NODE_COUNT="$node_count"
    export MULTINODE_GPUS_PER_NODE="$gpus_per_node"
    export MULTINODE_MASTER_ADDR="$head_node"

    # shellcheck disable=SC2016 # SLURM_PROCID is created inside each srun task.
    srun \
        --jobid="$job_id" \
        --nodes="$node_count" \
        --ntasks="$node_count" \
        --ntasks-per-node=1 \
        --kill-on-bad-exit=1 \
        --container-image="$squash_file" \
        --container-mounts="$container_mounts" \
        --no-container-mount-home \
        --container-remap-root \
        --container-workdir=/workspace \
        --no-container-entrypoint \
        --export=ALL \
        bash -c 'export MULTINODE_NODE_RANK="$SLURM_PROCID"; exec bash "$1"' \
        bash "$server_script" \
        >"$server_log_dir/combined.log" 2>&1 &
    server_step_pid=$!

    for ((attempt = 1; attempt <= 720; attempt++)); do
        servers_ready=true
        for server_node in $server_nodes; do
            if ! curl --output /dev/null --silent --fail "http://$server_node:$PORT/health"; then
                servers_ready=false
                break
            fi
        done
        if [[ "$servers_ready" == "true" ]]; then
            break
        fi
        if ! kill -0 "$server_step_pid" 2>/dev/null; then
            echo "Native multi-node server exited before becoming ready" >&2
            tail -n 200 "$server_log_dir/combined.log" >&2 || true
            exit 1
        fi
        sleep 10
    done
    if (( attempt > 720 )); then
        echo "Native multi-node server did not become ready within 7200 seconds" >&2
        exit 1
    fi

    set +e
    srun \
        --jobid="$job_id" \
        --overlap \
        --nodes=1 \
        --ntasks=1 \
        --nodelist="$head_node" \
        --container-image="$squash_file" \
        --container-mounts="$container_mounts" \
        --no-container-mount-home \
        --container-remap-root \
        --container-workdir=/workspace \
        --no-container-entrypoint \
        --export=ALL,MULTINODE_NODE_RANK=0 \
        bash "$client_script"
    client_rc=$?
    set -e
    exit "$client_rc"
fi

if [[ "$IS_MULTINODE" == "true" ]]; then

    # MODEL_PATH: Override with pre-downloaded paths on H200 runner
    # The yaml files specify HuggingFace model IDs for portability, but we use
    # local paths to avoid repeated downloading on the shared H200 cluster.
    if [[ $FRAMEWORK == "dynamo-sglang" ]]; then
        if [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp8" ]]; then
            export MODEL_PATH="/models/DeepSeek-R1-0528"
            export SRT_SLURM_MODEL_PREFIX="dsr1-fp8"
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
    else
        echo "Unsupported framework: $FRAMEWORK. Supported frameworks are: dynamo-trt, dynamo-sglang"
        exit 1
    fi

    echo "Cloning srt-slurm repository..."
    SRT_REPO_DIR="srt-slurm"
    if [ -d "$SRT_REPO_DIR" ]; then
        echo "Removing existing $SRT_REPO_DIR..."
        rm -rf "$SRT_REPO_DIR"
    fi

    # TODO(CJQ): make first class upon srt-slurm upstream refactor
    if [[ "$IS_AGENTIC" == "1" ]]; then
        git clone --branch cam/sa-submission-q2-2026 --single-branch https://github.com/cquil11/srt-slurm-nv.git "$SRT_REPO_DIR"
        cd "$SRT_REPO_DIR"
    else
        git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
        cd "$SRT_REPO_DIR"
        git checkout sa-submission-q2-2026
    fi

    echo "Installing srtctl..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env

    uv venv
    source .venv/bin/activate
    uv pip install -e .

    if ! command -v srtctl &> /dev/null; then
        echo "Error: Failed to install srtctl"
        exit 1
    fi

    echo "Configs available at: $SRT_REPO_DIR/"

    # Map container images to local squash files based on framework
    NGINX_SQUASH_FILE="/data/containers/nginx+1.27.4.sqsh"

    if [[ $FRAMEWORK == "dynamo-sglang" ]]; then
        # SGLang container mapping
        SQUASH_FILE="/data/containers/$(echo "$IMAGE" | sed 's/[\/:@#]/+/g').sqsh"
        CONTAINER_KEY="$IMAGE"
    elif [[ $FRAMEWORK == "dynamo-trt" ]]; then
        # TRT-LLM container mapping - convert IMAGE to srt-slurm format (nvcr.io/ -> nvcr.io#)
        CONTAINER_KEY=$(echo "$IMAGE" | sed 's|nvcr.io/|nvcr.io#|')
        SQUASH_FILE="/data/containers/$(echo "$IMAGE" | sed 's|nvcr.io/||' | sed 's/[\/:@#]/+/g').sqsh"
    fi

    export ISL="$ISL"
    export OSL="$OSL"
    export EVAL_ONLY="${EVAL_ONLY:-false}"

    # Create srtslurm.yaml for srtctl (used by both frameworks)
    SRTCTL_ROOT="${GITHUB_WORKSPACE}/${SRT_REPO_DIR}"
    echo "Creating srtslurm.yaml configuration..."
    cat > srtslurm.yaml <<EOF
# SRT SLURM Configuration for H200

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
  "${MODEL_PREFIX}": "${MODEL_PATH}"
containers:
  dynamo-trtllm: "${SQUASH_FILE}"
  dynamo-sglang: "${SQUASH_FILE}"
  nginx-sqsh: "${NGINX_SQUASH_FILE}"
  latest: "${SQUASH_FILE}"
  "${CONTAINER_KEY}": "${SQUASH_FILE}"
# SLURM directive compatibility
use_gpus_per_node_directive: true
use_segment_sbatch_directive: false
use_exclusive_sbatch_directive: false
EOF

    echo "Generated srtslurm.yaml:"
    cat srtslurm.yaml

    echo "Running make setup..."
    make setup ARCH=x86_64

    # Export eval-related env vars for srt-slurm post-benchmark eval
    export INFMAX_WORKSPACE="$GITHUB_WORKSPACE"

    echo "Submitting job with srtctl..."

    if [[ -z "$CONFIG_FILE" ]]; then
        echo "Error: CONFIG_FILE is not set. The srt-slurm path requires a CONFIG_FILE in additional-settings." >&2
        echo "Config: MODEL_PREFIX=${MODEL_PREFIX} PRECISION=${PRECISION} FRAMEWORK=${FRAMEWORK}" >&2
        exit 1
    fi

    # Override the job name in the config file with the runner name
    sed -i "s/^name:.*/name: \"${RUNNER_NAME}\"/" "$CONFIG_FILE"
    sed -i '/^health_check:/,/^[^ ]/{ /^health_check:/d; /^  /d; }' "${CONFIG_FILE%%:*}"
    printf '\nhealth_check:\n  max_attempts: 720\n  interval_seconds: 10\n' >> "${CONFIG_FILE%%:*}"
    SRTCTL_OUTPUT=$(srtctl apply -f "$CONFIG_FILE" --tags "h200,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)" 2>&1)
    echo "$SRTCTL_OUTPUT"

    # Extract JOB_ID from srtctl output
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

    if [ ! -d "$LOGS_DIR" ]; then
        echo "Warning: Logs directory not found at $LOGS_DIR"
        exit 1
    fi

    echo "Found logs directory: $LOGS_DIR"

    cp -r "$LOGS_DIR" "$GITHUB_WORKSPACE/LOGS"
    tar czf "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz" -C "$LOGS_DIR" .

    if [[ "${EVAL_ONLY:-false}" != "true" ]]; then
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

    HF_HUB_CACHE_MOUNT="/models/gharunners/hf-hub-cache"
    AIPERF_MMAP_CACHE_HOST_PATH="/home/sa-shared/gharunners/ai-perf-cache"
    SQUASH_FILE="/data/gharunners/containers/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

    # Convert pyxis image format (nvcr.io#path) to docker format (nvcr.io/path) for enroot import
    DOCKER_IMAGE=$(echo "$IMAGE" | sed 's/#/\//g')
    LOCK_FILE="${SQUASH_FILE}.lock"

    export GPU_COUNT="${GPU_COUNT:-${TP:?TP must be set}}"

    salloc --partition=$SLURM_PARTITION --account=$SLURM_ACCOUNT --gres=gpu:$GPU_COUNT --exclusive --time=180 --no-shell --job-name="$RUNNER_NAME"
    JOB_ID=$(squeue --name="$RUNNER_NAME" -u "$USER" -h -o %A | head -n1)
    if [[ -z "$JOB_ID" ]]; then
        echo "ERROR: failed to resolve H200 Slurm allocation" >&2
        exit 1
    fi
    trap 'rc=$?; scancel "$JOB_ID" 2>/dev/null || true; exit "$rc"' EXIT

    # Use flock to serialize concurrent imports to the same squash file
    # Override ENROOT_CACHE_PATH to avoid permission issues with system-wide cache on worker nodes
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

    if [[ "$IMAGE" == *deepseek-v4-hopper* ]]; then
        CONTAINER_MOUNT_DIR=/ix
    else
        CONTAINER_MOUNT_DIR=/workspace
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
