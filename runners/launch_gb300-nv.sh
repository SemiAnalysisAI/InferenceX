#!/usr/bin/bash

# This script sets up the environment and launches multi-node benchmarks

set -exo pipefail

export SLURM_PARTITION="batch_1"
export SLURM_ACCOUNT="benchmark"
export ENROOT_ROOTFS_WRITABLE=1

# Host-side directory holding aiperf's content-addressed dataset mmap cache.
# Bind-mounted into worker containers at /aiperf_mmap_cache via the
# default_mounts: block in srtslurm.yaml below; aiperf reads it via
# AIPERF_DATASET_MMAP_CACHE_DIR (set in each agentic recipe's benchmark.env).
# Without it, every run re-tokenizes and re-writes ~65 GB of mmap files
# per dataset on first use. 777 mode so all gharunner_X SLURM users can
# write to it.
export AIPERF_MMAP_CACHE_HOST_PATH="/data/home/sa-shared/gharunners/ai-perf-cache"

export MODEL_PATH=$MODEL

if [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp4" ]]; then
    export SERVED_MODEL_NAME="deepseek-r1-fp4"
    export MODEL_PATH=/scratch/models/DeepSeek-R1-0528-NVFP4-v2
    export SRT_SLURM_MODEL_PREFIX="dsr1"
elif [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp8" ]]; then
    export SERVED_MODEL_NAME="deepseek-r1-fp8"
    export MODEL_PATH=/scratch/models/DeepSeek-R1-0528
    export SRT_SLURM_MODEL_PREFIX="dsr1-fp8"
elif [[ $MODEL_PREFIX == "dsv4" && $PRECISION == "fp4" ]]; then
    # Use the node-local /scratch SSD for the 806 GB DSv4-Pro
    # checkpoint. Faster than the Vast NFS path, but this dir only
    # exists on compute nodes — the GHA runner pod's view does NOT
    # have /scratch/models, so srtctl preflight (which stats the path
    # from the runner pod) may fail with "Model alias resolved to
    # /scratch/models/DeepSeek-V4-Pro, but that path is unavailable."
    # If that happens, the next step is either to (a) patch srt-slurm
    # to add a skip_model_preflight recipe field, or (b) stub a
    # symlink on the runner pod that points at the NFS copy.
    export MODEL_PATH=/scratch/models/DeepSeek-V4-Pro
    export SRT_SLURM_MODEL_PREFIX="deepseek-v4-pro"
elif [[ $MODEL_PREFIX == "glm5" && $PRECISION == "fp4" && $FRAMEWORK == "dynamo-trt" ]]; then
    export SERVED_MODEL_NAME="glm-5-nvfp4"
    export MODEL_PATH=/scratch/models/GLM-5-NVFP4
    export SRT_SLURM_MODEL_PREFIX="nvidia/GLM-5-NVFP4"
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
elif [[ $MODEL_PREFIX == "minimaxm3" && $PRECISION == "fp8" ]]; then
    export MODEL_PATH=/data/models/MiniMax-M3-MXFP8
    export SRT_SLURM_MODEL_PREFIX="minimax-m3-mxfp8"
elif [[ $MODEL_PREFIX == "kimik2.5" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH=/scratch/models/Kimi-K2.5-NVFP4
    export SRT_SLURM_MODEL_PREFIX="nvidia/Kimi-K2.5-NVFP4"
elif [[ $MODEL_PREFIX == "qwen3.5" && $PRECISION == "fp4" ]]; then
    # SRT_SLURM_MODEL_PREFIX must match the model.path alias used in our
    # Qwen3.5 sglang recipes (qwen3.5-fp4).
    export MODEL_PATH=/scratch/models/Qwen3.5-397B-A17B-NVFP4
    export SRT_SLURM_MODEL_PREFIX="qwen3.5-fp4"
else
    echo "Unsupported model: $MODEL_PREFIX-$PRECISION. Supported models are: dsr1-fp4, dsr1-fp8, dsv4-fp4, glm5-fp4, glm5-fp8, minimaxm2.5-fp4, minimaxm2.5-fp8, kimik2.5-fp4, qwen3.5-fp4"
    exit 1
fi

NGINX_IMAGE="nginx:1.27.4"

# Squash files live on the Vast NFS storage; use the /data/ mount
# (not /home/sa-shared/) — both are the same backing storage but the
# /home/sa-shared/ mount has a chronic ELOOP / "Too many levels of
# symbolic links" bug from workflow worker NFS sessions on lockfiles
# AND data files. /data/ has a separate NFS client cache that isn't
# poisoned. See feedback_gb300_nfs_eloop_workaround for diagnosis.
SQUASH_FILE="/data/home/sa-shared/gharunners/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
NGINX_SQUASH_FILE="/data/home/sa-shared/gharunners/squash/$(echo "$NGINX_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

# Run the import on a compute node via srun, not on the login node:
# the login node is x86_64 while the compute nodes are aarch64, so the
# arm64 squash file has to be built on a compute node.
import_squash() {
    local squash="$1" image="$2"
    local lock="${squash}.lock"
    srun --partition=$SLURM_PARTITION --exclusive --time=180 bash -c "
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

# Native TRT-LLM disaggregated generation-worker benchmark. The end-to-end
# workflow exports CONFIG_FILE from the selected master-config row.
# The wrapper renders cluster-local paths and invokes the bundled generation-only harness.
if [[ "$IS_MULTINODE" == "true" &&
      "$SPEC_DECODING" == "offline" &&
      "$FRAMEWORK" == "trt" &&
      "$MODEL_PREFIX" == "dsv4" ]]; then
    export TRT_GEN_ONLY_SQUASH_FILE="$SQUASH_FILE"
    bash "benchmarks/multi_node/offline/dsv4_fp4_gb300_trt.sh"
    exit $?
fi

# ---------------------------------------------------------------------------
# Single-node path (multinode: false configs, e.g. the offline decode-step DSV4
# offline bench). Mirrors the launch_gb200-nv.sh single-node branch:
# salloc one 4-GPU GB300 tray and run the bench script inside the
# container via pyxis. MODEL_PATH was resolved above (node-local
# /scratch for DSV4); the squash import already ran on a compute node.
# ---------------------------------------------------------------------------
if [[ "$IS_MULTINODE" != "true" ]]; then
    case "$SPEC_DECODING" in
        mtp)     SPEC_SUFFIX='_mtp' ;;
        offline) SPEC_SUFFIX='_offline' ;;
        *)       SPEC_SUFFIX='' ;;
    esac
    BENCH_BASE="benchmarks/single_node/${SCENARIO_SUBDIR}${EXP_NAME%%_*}_${PRECISION}_gb300"
    BENCH_SCRIPT="${BENCH_BASE}_${FRAMEWORK}${SPEC_SUFFIX}.sh"
    if [[ ! -f "$BENCH_SCRIPT" ]]; then
        BENCH_SCRIPT="${BENCH_BASE}${SPEC_SUFFIX}.sh"
    fi

    # A GB300 NVL72 Slurm node is one 4-GPU tray. TP<=4 fits on a single
    # tray (the DEP=4 offline bench). TP>4 (e.g. DEP=16) spans multiple
    # trays in the same rack: allocate TP/4 nodes, 4 GPUs each, and run the
    # bench script once per node (distinct SLURM_PROCID = DP node_rank).
    # Cross-tray DP/EP NCCL rides the rack NVLink domain; only the initial
    # TCP rendezvous uses the inter-node IP network (known-good here — this
    # cluster runs multi-node srt-slurm jobs). Mirrors launch_h200-dgxc-slurm.sh.
    GPUS_PER_NODE=4
    if [[ $TP -gt $GPUS_PER_NODE ]]; then
        ALLOC_NODES=$(( TP / GPUS_PER_NODE ))
        ALLOC_GPUS="gpu:${GPUS_PER_NODE}"
        SRUN_MULTI="--nodes=${ALLOC_NODES} --ntasks=${ALLOC_NODES} --ntasks-per-node=1"
    else
        ALLOC_NODES=1
        ALLOC_GPUS="gpu:${TP}"
        SRUN_MULTI=""
    fi

    salloc --partition=$SLURM_PARTITION --account=$SLURM_ACCOUNT -N "$ALLOC_NODES" --gres=$ALLOC_GPUS --exclusive --time=180 --no-shell --job-name="$RUNNER_NAME"
    JOB_ID=$(squeue --name="$RUNNER_NAME" -u "$USER" -h -o %A | head -n1)

    # Multi-node DP rendezvous needs the head node's hostname. scontrol runs
    # on the login node (compute lacks it), so resolve here and export into
    # the container via --export=ALL.
    if [[ $ALLOC_NODES -gt 1 ]]; then
        MASTER_ADDR=$(scontrol show hostname "$(squeue -j "$JOB_ID" -o '%N' -h)" | head -n1)
        export MASTER_ADDR
        echo "Resolved MASTER_ADDR=$MASTER_ADDR for job $JOB_ID"
    fi

    # MODEL_PATH is node-local (/scratch) on compute; bind-mount it so the
    # bench script's directory check sees it inside the container.
    MODEL_MOUNT=""
    if [[ "${MODEL_PATH:-}" == /* ]]; then
        MODEL_MOUNT=",$MODEL_PATH:$MODEL_PATH"
    fi

    # Pre-flight: /scratch is node-local, so the model must be staged on
    # every allocated tray. Fail fast with the offending hostname rather
    # than letting one node OOM/crash deep into vLLM startup.
    if [[ $ALLOC_NODES -gt 1 && "${MODEL_PATH:-}" == /* ]]; then
        if ! srun --jobid=$JOB_ID --nodes=$ALLOC_NODES --ntasks=$ALLOC_NODES --ntasks-per-node=1 \
                bash -c "test -d '$MODEL_PATH' || { echo \"MISSING $MODEL_PATH on \$(hostname)\"; exit 1; }"; then
            echo "Model not staged on all $ALLOC_NODES nodes; aborting"
            scancel "$JOB_ID"
            exit 1
        fi
    fi

    RC=0
    srun --jobid=$JOB_ID $SRUN_MULTI \
        --mpi=none \
        --container-image=$SQUASH_FILE \
        --container-mounts=$GITHUB_WORKSPACE:/workspace${MODEL_MOUNT} \
        --no-container-mount-home \
        --container-workdir=/workspace \
        --no-container-entrypoint --export=ALL,PORT=8888 \
        bash "$BENCH_SCRIPT" || RC=$?

    scancel "$JOB_ID"
    exit $RC
fi

echo "Cloning srt-slurm repository..."
RUN_KEY=$(printf "%s" "${RESULT_FILENAME:-${RUNNER_NAME:-gb300-nv}}" | sha1sum | cut -c1-12)
SRT_REPO_DIR="${GITHUB_WORKSPACE}/srt-slurm-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-0}-${RUN_KEY}"
SRTCTL_SETUP_SCRIPT=""
rm -rf "$SRT_REPO_DIR"

if [[ "$IS_AGENTIC" == "1" ]]; then
    # Agentic multi-node uses cquil11/srt-slurm-nv@cam/no-preflight-flag,
    # a thin branch off NVIDIA/srt-slurm@127597c that adds one CLI flag
    # (`srtctl apply --no-preflight`) — needed because:
    #
    #   - We want MODEL_PATH=/scratch/models/DeepSeek-V4-Pro (node-local
    #     NVMe, fast) instead of the NFS path under /data/home/sa-shared.
    #   - /scratch only exists on GB300 compute nodes; it is NOT mounted
    #     on the GHA runner pod that invokes srtctl.
    #   - srtctl's pre-submit model check (_preflight_model in
    #     src/srtctl/core/validation.py) does a Path.is_dir() in-process
    #     on the invoking node — so it fails before sbatch is ever
    #     called with "Model alias 'X' resolved to '/scratch/...',
    #     but that path is unavailable".
    #   - --no-preflight skips just the optional Python-level FS check.
    #     vLLM still fails loudly at runtime if the path is genuinely
    #     missing on the compute node.
    #
    # All other upstream schema features we need are inherited from
    # NVIDIA HEAD:
    #   - BenchmarkType.CUSTOM + benchmark.command + benchmark.env
    #     (hook that hands off to benchmarks/multi_node/agentic_srt.sh)
    #   - DynamoConfig.wheel (so vllm recipes can pin the ai-dynamo wheel)
    #   - sbatch_directives / srun_options (top-level recipe fields)
    git clone https://github.com/cquil11/srt-slurm-nv.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    # 854b3fd = --no-preflight flag
    # 6e34b8b = benchmark_stage propagates srun_options (needed for
    #           container-remap-root to reach the agentic_srt.sh srun)
    git checkout 6e34b8b83229634d732e41a4e2d6595f46ef60b5
    mkdir -p recipes/vllm/deepseek-v4/agentic
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/deepseek-v4/agentic" \
        recipes/vllm/deepseek-v4/agentic
elif [[ $FRAMEWORK == "dynamo-vllm" && $MODEL_PREFIX == "dsv4" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout aflowers/gb200-dsv4-recipes
    mkdir -p recipes/vllm/deepseek-v4
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/deepseek-v4" recipes/vllm/deepseek-v4
elif [[ $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "glm5" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout sa-submission-q2-2026
    mkdir -p recipes/sglang/glm5
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/glm5" recipes/sglang/glm5
elif [[ $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "qwen3.5" ]]; then
    # Same srt-slurm tooling as glm5: NVIDIA/srt-slurm @ sa-submission-q2-2026.
    # Overlay our version-controlled Qwen3.5 recipes on top (upstream has none).
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout sa-submission-q2-2026
    mkdir -p recipes/sglang/qwen3.5
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/qwen3.5" recipes/sglang/qwen3.5
elif [[ $FRAMEWORK == "dynamo-vllm" && $MODEL_PREFIX == "minimaxm2.5" && $PRECISION == "fp8" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout main
    mkdir -p recipes/vllm/minimax-m2.5-fp8
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/minimax-m2.5-fp8" recipes/vllm/minimax-m2.5-fp8
elif [[ $FRAMEWORK == "dynamo-vllm" && $MODEL_PREFIX == "minimaxm2.5" && $PRECISION == "fp4" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout main
    mkdir -p recipes/vllm/minimax-m2.5
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/minimax-m2.5" recipes/vllm/minimax-m2.5
elif [[ $FRAMEWORK == "dynamo-vllm" && $MODEL_PREFIX == "minimaxm3" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout sa-submission-q2-2026
    mkdir -p recipes/vllm/minimax-m3-gb300-fp8
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/minimax-m3-gb300-fp8" recipes/vllm/minimax-m3-gb300-fp8
    SRTCTL_SETUP_SCRIPT="minimax-m3-gb300-vllm-fixes.sh"
    cp \
        "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/configs/$SRTCTL_SETUP_SCRIPT" \
        "configs/$SRTCTL_SETUP_SCRIPT"
elif [[ $FRAMEWORK == "dynamo-vllm" && $MODEL_PREFIX == "kimik2.5" && $PRECISION == "fp4" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout main
    mkdir -p recipes/vllm/kimi-k2.5-fp4
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/kimi-k2.5-fp4" recipes/vllm/kimi-k2.5-fp4
elif [[ $FRAMEWORK == "dynamo-trt" && $MODEL_PREFIX == "dsv4" ]]; then
    # DSv4 dynamo-trt recipes use the HuggingFace model ID as model.path,
    # so override SRT_SLURM_MODEL_PREFIX to match the recipe's model path key.
    SRT_SLURM_MODEL_PREFIX="deepseek-ai/DeepSeek-V4-Pro"
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout sa-submission-q2-2026
else
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout sa-submission-q2-2026
fi

echo "Installing srtctl..."
export UV_INSTALL_DIR="$GITHUB_WORKSPACE/.local/bin"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$UV_INSTALL_DIR:$PATH"

VENV_DIR="${GITHUB_WORKSPACE}/.venv-srt-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-0}-${RUN_KEY}"
rm -rf "$VENV_DIR"
# --seed installs pip+setuptools+wheel into the venv. Without it, the
# upstream prefetch-ai-dynamo-wheel.sh script (called by srtctl when a
# recipe has dynamo.wheel set) fails with "No module named pip" because
# uv venv defaults to no-pip.
uv venv --seed "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv pip install -e .

if ! command -v srtctl &> /dev/null; then
    echo "Error: Failed to install srtctl"
    exit 1
fi

echo "Configs available at: $SRT_REPO_DIR/"

# Create srtslurm.yaml for srtctl (used by both frameworks)
SRTCTL_ROOT="${SRT_REPO_DIR}"
echo "Creating srtslurm.yaml configuration..."
cat > srtslurm.yaml <<EOF
# SRT SLURM Configuration for GB300

# Default SLURM settings
default_account: "${SLURM_ACCOUNT}"
default_partition: "${SLURM_PARTITION}"
default_time_limit: "4:00:00"

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

# Model path aliases
model_paths:
  "${SRT_SLURM_MODEL_PREFIX}": "${MODEL_PATH}"
containers:
  dynamo-trtllm: ${SQUASH_FILE}
  dynamo-sglang: ${SQUASH_FILE}
  "${IMAGE}": ${SQUASH_FILE}
  nginx-sqsh: ${NGINX_SQUASH_FILE}
use_segment_sbatch_directive: false
EOF

echo "Generated srtslurm.yaml:"
cat srtslurm.yaml

echo "Running make setup..."
make setup ARCH=aarch64

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

# --no-preflight is only safe on the agentic path, where the recipe
# resolves model.path to /scratch (compute-node-only NVMe) and the
# srtctl process running on the GHA runner pod can't see it. Fixed-
# seq-len recipes still resolve model.path to an NFS-visible location
# where the precheck is a useful sanity guard, so keep enforcement on
# for them.
SRTCTL_APPLY_ARGS=(
    -f "$CONFIG_FILE"
    --tags "gb300,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)"
)
if [[ "$IS_AGENTIC" == "1" ]]; then
    SRTCTL_APPLY_ARGS+=(--no-preflight)
fi
if [[ -n "$SRTCTL_SETUP_SCRIPT" ]]; then
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

# Snapshot worker logs on any exit path — normal completion, error,
# SIGTERM (gh run cancel sends this to the launcher), even SIGKILL of
# our parent. Without this trap, the cancel-time tar lives only in the
# main flow below (after `wait $POLL_PID`), so a manual `gh run cancel`
# during the tail wait skips it entirely and the
# `Upload server logs` workflow step finds nothing to upload.
# Idempotent: the main-flow tar at the bottom of this script is now a
# no-op because the trap already produced the artifact, but it stays
# for narrative continuity in normal (non-cancel) runs.
_snapshot_server_logs() {
    if [ -n "${LOGS_DIR:-}" ] && [ -d "$LOGS_DIR" ] && [ -n "${GITHUB_WORKSPACE:-}" ]; then
        # Copy + tar are independent best-effort; an in-flight write
        # from a worker .out file at SIGTERM time would otherwise abort
        # the whole script before either succeeds.
        cp -r "$LOGS_DIR" "$GITHUB_WORKSPACE/LOGS" 2>/dev/null || true
        tar czf "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz" -C "$LOGS_DIR" . 2>/dev/null || true
    fi
}
trap _snapshot_server_logs EXIT

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

if [ -d "$LOGS_DIR" ]; then
    echo "Found logs directory: $LOGS_DIR"
    # Tarball + LOGS copy are produced by the EXIT trap defined near
    # JOB_ID extraction (so cancel paths also get them); just log here.
    echo "multinode_server_logs.tar.gz will be (re)produced on script EXIT."
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
fi

# Snapshot logs to GITHUB_WORKSPACE BEFORE cleanup, so the EXIT trap's
# `[ -d "$LOGS_DIR" ]` guard isn't already false by the time it fires
# (it runs AFTER the rm below, since EXIT traps are last-thing-before-exit).
# Without this inline call, R25 lost both 1p6d shards' logs.
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
