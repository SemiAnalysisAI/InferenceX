#!/usr/bin/bash

# This script sets up the environment and launches multi-node benchmarks

set -x

# Helper: download HuggingFace models to a shared cache so all Slurm nodes can read them.
# If MODEL is already a local path, it is returned as-is.
hf_download_to_shared_cache() {
    local model_id="$1"
    local cache_root="${HF_MODEL_CACHE_ROOT:-/mnt/lustre01/users-public/sa-shared/hf-models}"

    # Local path provided
    if [[ "$model_id" == /* ]]; then
        echo "$model_id"
        return 0
    fi

    local safe_id="${model_id//\//__}"
    local dst="${cache_root}/${safe_id}"

    mkdir -p "$dst"

    # Heuristic: if the directory is empty, download into it.
    if [[ -z "$(ls -A "$dst" 2>/dev/null)" ]]; then
        echo "[INFO] Downloading HuggingFace model '${model_id}' to '${dst}'"
        if command -v hf >/dev/null 2>&1; then
            hf download "$model_id" --local-dir "$dst" --local-dir-use-symlinks False
        elif command -v huggingface-cli >/dev/null 2>&1; then
            huggingface-cli download "$model_id" --local-dir "$dst" --local-dir-use-symlinks False
        else
            echo "[ERROR] Neither 'hf' nor 'huggingface-cli' is available to download '${model_id}'."
            exit 1
        fi
    else
        echo "[INFO] Reusing cached model directory '${dst}'"
    fi

    echo "$dst"
}

# MODEL_PATH: Override with pre-downloaded paths on GB200 runner
# The yaml files specify HuggingFace model IDs for portability, but we use
# local paths to avoid repeated downloading on the shared GB200 cluster.
if [[ $FRAMEWORK == "dynamo-sglang" ]]; then
    export CONFIG_DIR="/mnt/lustre01/artifacts/sglang-configs/1k1k"
    if [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp8" ]]; then
        export MODEL_PATH="/mnt/lustre01/models/deepseek-r1-0528"
        export SRT_SLURM_MODEL_PREFIX="dsr1-fp8"
    elif [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp4" ]]; then
        export MODEL_PATH="/mnt/lustre01/models/deepseek-r1-0528-fp4-v2/"
        export SRT_SLURM_MODEL_PREFIX="dsr1-fp4"
    elif [[ $MODEL_PREFIX == "qwen3.5" ]]; then
        # Pull the model once to shared storage so all Slurm nodes can access it.
        export SRT_SLURM_MODEL_PREFIX="qwen3.5-${PRECISION}"
        export MODEL_PATH="/mnt/lustre01/users-public/sa-shared/hf-models/qwen3.5-397b-a17b"   
    else
        export MODEL_PATH=$MODEL
        export SRT_SLURM_MODEL_PREFIX="${SRT_SLURM_MODEL_PREFIX:-$MODEL_PREFIX}"
    fi
elif [[ $FRAMEWORK == "dynamo-trt" ]]; then
    if [[ $MODEL_PREFIX == "gptoss" ]]; then
        export MODEL_PATH="/mnt/lustre01/models/gpt-oss-120b"
        export SERVED_MODEL_NAME="gpt-oss-120b"
    elif [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp4" ]]; then
        export MODEL_PATH="/mnt/lustre01/models/deepseek-r1-0528-fp4-v2/"
        export SERVED_MODEL_NAME="deepseek-r1-fp4"
        export SRT_SLURM_MODEL_PREFIX="dsr1"
    elif [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp8" ]]; then
        export MODEL_PATH="/mnt/numa1/groups/sa-shared/models/deepseek-r1-0528/"
        export SERVED_MODEL_NAME="deepseek-r1-fp8"
        export SRT_SLURM_MODEL_PREFIX="dsr1-fp8"
    else
        echo "Unsupported model prefix: $MODEL_PREFIX. Supported prefixes are: gptoss or dsr1"
        exit 1
    fi
elif [[ $FRAMEWORK == "dynamo-vllm" ]]; then
    if [[ $MODEL_PREFIX == "kimik2.5" && $PRECISION == "fp4" ]]; then
        export MODEL_PATH="/mnt/lustre01/models/kimi-k2.5-nvfp4"
        export SRT_SLURM_MODEL_PREFIX="kimi-k2.5-nvfp4"
    else
        echo "Unsupported model prefix/precision combination: $MODEL_PREFIX/$PRECISION. Supported combinations for dynamo-vllm: kimik2.5/fp4"
        exit 1
    fi
else
    export MODEL_PATH=$MODEL
fi

# Set up environment variables for SLURM
export SLURM_PARTITION="batch"
export SLURM_ACCOUNT="benchmark"

NGINX_IMAGE="nginx:1.27.4"

SQUASH_FILE="/mnt/lustre01/users-public/sa-shared/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
NGINX_SQUASH_FILE="/mnt/lustre01/users-public/sa-shared/$(echo "$NGINX_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

enroot import -o $SQUASH_FILE docker://$IMAGE
enroot import -o $NGINX_SQUASH_FILE docker://$NGINX_IMAGE

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


echo "Cloning srt-slurm repository..."
SRT_REPO_DIR="srt-slurm"
if [ -d "$SRT_REPO_DIR" ]; then
    echo "Removing existing $SRT_REPO_DIR..."
    rm -rf "$SRT_REPO_DIR"
fi

if [[ $FRAMEWORK == "dynamo-vllm" ]] || [[ "$MODEL_PREFIX" == "qwen3.5" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout sa-submission-q2-2026
else
    git clone https://github.com/ishandhanani/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout sa-submission-q1-2026
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

# Apply InferenceX patches to srt-slurm
PATCH_DIR="${GITHUB_WORKSPACE}/runners/patches"
if [ -d "$PATCH_DIR" ]; then
    for patch_file in "$PATCH_DIR"/*.patch; do
        [ -f "$patch_file" ] || continue
        echo "Applying patch: $(basename "$patch_file")"
        git apply --recount "$patch_file" || echo "Warning: patch ... did not apply cleanly"
    done
fi

echo "Configs available at: $SRT_REPO_DIR/"

# Create srtslurm.yaml for srtctl (used by both frameworks)
SRTCTL_ROOT="${GITHUB_WORKSPACE}/srt-slurm"
echo "Creating srtslurm.yaml configuration..."

# Ensure we always have a model alias for srtslurm.yaml
export SRT_SLURM_MODEL_PREFIX="${SRT_SLURM_MODEL_PREFIX:-$MODEL_PREFIX}"

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
  "${SRT_SLURM_MODEL_PREFIX}": "${MODEL_PATH}"
containers:
  dynamo-trtllm: ${SQUASH_FILE}
  dynamo-sglang: ${SQUASH_FILE}
  dev: ${SQUASH_FILE}
  "${IMAGE}": ${SQUASH_FILE}
  nginx-sqsh: ${NGINX_SQUASH_FILE}
EOF

echo "Generated srtslurm.yaml:"
cat srtslurm.yaml

echo "Running make setup..."
make setup ARCH=aarch64

echo "Submitting job with srtctl..."

SETUP_SCRIPT=""

# --- Qwen3.5 EPD (Encoder/Prefill/Decode) disaggregation support ---
# Generates a recipe based on the upstream DEP4 nixl config but adds EPD-specific
# fields: language-only mode, encoder-urls, and a dedicated infra/encoder node.
if [[ "${EPD:-}" == "1" || "${EPD:-}" == "true" ]]; then
    if [[ "$MODEL_PREFIX" == "qwen3.5" ]]; then
        echo "[INFO] EPD enabled for qwen3.5: generating recipe at '${CONFIG_FILE}'"
        mkdir -p "$(dirname "${CONFIG_FILE}")"

        GEN_PREFILL_WORKERS=${PREFILL_NUM_WORKERS:-1}
        GEN_DECODE_WORKERS=${DECODE_NUM_WORKERS:-1}

        # Convert space-separated CONC_LIST to srtctl format
        EPD_CONC_LIST="$(echo "${CONC_LIST:-64}" | tr ' ' 'x')"

        cat > "${CONFIG_FILE}" <<EOF
name: qwen3.5-epd-fp8-gb200
model:
  path: ${SRT_SLURM_MODEL_PREFIX}
  container: dev
  precision: fp8
resources:
  gpu_type: gb200
  gpus_per_node: 4
  prefill_nodes: ${GEN_PREFILL_WORKERS}
  prefill_workers: ${GEN_PREFILL_WORKERS}
  decode_nodes: ${GEN_DECODE_WORKERS}
  decode_workers: ${GEN_DECODE_WORKERS}
infra:
  etcd_nats_dedicated_node: true
backend:
  prefill_environment:
    TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"
    PYTHONUNBUFFERED: "1"
    NCCL_MNNVL_ENABLE: "1"
    NCCL_CUMEM_ENABLE: "1"
    MC_FORCE_MNNVL: "1"
    SGLANG_DG_CACHE_DIR: "/configs/deepgemm-cache"
    FLASHINFER_WORKSPACE_BASE: "/configs/flashinfer-cache"
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE: "100000"
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT: "100000"
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT: "100000"
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL: "True"
    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER: "0"
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK: "1"
  decode_environment:
    TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"
    PYTHONUNBUFFERED: "1"
    NCCL_MNNVL_ENABLE: "1"
    NCCL_CUMEM_ENABLE: "1"
    MC_FORCE_MNNVL: "1"
    SGLANG_DG_CACHE_DIR: "/configs/deepgemm-cache"
    FLASHINFER_WORKSPACE_BASE: "/configs/flashinfer-cache"
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE: "100000"
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT: "100000"
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT: "100000"
    SGLANG_DECODE_BOOTSTRAP_TIMEOUT: "1000"
    SGLANG_HACK_SEQ_BOOTSTRAP_ROOM: "1"
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL: "True"
    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER: "0"
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK: "1"
  sglang_config:
    prefill:
      served-model-name: "Qwen/Qwen3.5-397B-A17B-FP8"
      model-path: "/model/"
      attention-backend: trtllm_mha
      quantization: fp8
      kv-cache-dtype: fp8_e4m3
      moe-runner-backend: flashinfer_trtllm
      tensor-parallel-size: 4
      data-parallel-size: 4
      expert-parallel-size: 4
      enable-dp-attention: true
      enable-dp-lm-head: true
      moe-dense-tp-size: 1
      mamba-scheduler-strategy: no_buffer
      mamba-track-interval: 2048
      mamba-ssm-dtype: bfloat16
      disaggregation-mode: prefill
      disaggregation-transfer-backend: nixl
      disable-radix-cache: true
      disaggregation-decode-tp: 4
      disaggregation-decode-dp: 4
      language-only: true
      encoder-urls:
        - "http://{head_node_ip}:40000"
        - "http://{head_node_ip}:40001"
        - "http://{head_node_ip}:40002"
        - "http://{head_node_ip}:40003"
      trust-remote-code: true
      mem-fraction-static: 0.80
      chunked-prefill-size: 16384
      context-length: 2020
      load-balance-method: round_robin
      watchdog-timeout: 1000000
      disable-cuda-graph: true
    decode:
      served-model-name: "Qwen/Qwen3.5-397B-A17B-FP8"
      model-path: "/model/"
      attention-backend: trtllm_mha
      quantization: fp8
      kv-cache-dtype: fp8_e4m3
      moe-runner-backend: flashinfer_trtllm
      tensor-parallel-size: 4
      data-parallel-size: 4
      expert-parallel-size: 4
      enable-dp-attention: true
      enable-dp-lm-head: true
      moe-dense-tp-size: 1
      mamba-scheduler-strategy: no_buffer
      mamba-track-interval: 2048
      mamba-ssm-dtype: bfloat16
      disaggregation-mode: decode
      disaggregation-transfer-backend: nixl
      disable-radix-cache: true
      language-only: true
      encoder-urls:
        - "http://{head_node_ip}:40000"
        - "http://{head_node_ip}:40001"
        - "http://{head_node_ip}:40002"
        - "http://{head_node_ip}:40003"
      trust-remote-code: true
      mem-fraction-static: 0.80
      chunked-prefill-size: 16384
      context-length: 2020
      cuda-graph-max-bs: 1024
      watchdog-timeout: 1000000
      decode-log-interval: 1
      stream-interval: 50
benchmark:
  type: sa-bench
  isl: ${ISL}
  osl: ${OSL}
  concurrencies: "${EPD_CONC_LIST}"
  req_rate: "inf"
EOF

        # Setup script: start 4 encoder-only servers on the infra node.
        cat > configs/qwen3.5-epd-setup.sh <<'EOF'
#!/usr/bin/env bash
set -euxo pipefail

if [[ -f /configs/install-torchao.sh ]]; then
  bash /configs/install-torchao.sh
fi

# Start encoder-only servers on the first allocated node (reserved when infra.etcd_nats_dedicated_node=true)
if command -v scontrol >/dev/null 2>&1; then
  HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
else
  HEAD_NODE=""
fi
THIS_NODE="$(hostname -s)"

if [[ -n "${HEAD_NODE_IP:-}" ]]; then
  if command -v ip >/dev/null 2>&1; then
    ip -4 addr show | grep -qw "${HEAD_NODE_IP}" || exit 0
  else
    (hostname -I 2>/dev/null || true) | grep -qw "${HEAD_NODE_IP}" || exit 0
  fi
else
  if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
    [[ "${THIS_NODE}" == "${HEAD_NODE}" ]] || exit 0
  else
    echo "[EPD] WARNING: cannot determine head node; skipping encoder launch"
    exit 0
  fi
fi

echo "[EPD] Starting encoder-only servers on ${THIS_NODE}"
for GPU_ID in 0 1 2 3; do
  PORT=$((40000 + GPU_ID))
  if (echo > /dev/tcp/127.0.0.1/${PORT}) >/dev/null 2>&1; then
    echo "[EPD] Port ${PORT} already listening; skipping GPU ${GPU_ID}"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python3 -m sglang.launch_server \
      --model-path /model \
      --encoder-only \
      --tp-size 1 \
      --host 0.0.0.0 \
      --port "${PORT}" \
      --trust-remote-code \
      >"/logs/encoder_${GPU_ID}.log" 2>&1 &
done

for GPU_ID in 0 1 2 3; do
  PORT=$((40000 + GPU_ID))
  for _ in $(seq 1 120); do
    (echo > /dev/tcp/127.0.0.1/${PORT}) >/dev/null 2>&1 && break || true
    sleep 1
  done
done
echo "[EPD] Encoder servers ready"

EOF
        chmod +x configs/qwen3.5-epd-setup.sh
        SETUP_SCRIPT="qwen3.5-epd-setup.sh"
    fi
fi

# Default setup script for dynamo-sglang (fp8 tooling)
if [[ "$FRAMEWORK" == "dynamo-sglang" && -z "${SETUP_SCRIPT}" ]]; then
    SETUP_SCRIPT="install-torchao.sh"
fi

# Override the job name in the config file with the runner name
sed -i "s/^name:.*/name: \"${RUNNER_NAME}\"/" "$CONFIG_FILE"

if [[ "$FRAMEWORK" == "dynamo-sglang" ]]; then
    SRTCTL_OUTPUT=$(srtctl apply -f "$CONFIG_FILE" --tags "gb200,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)" --setup-script "${SETUP_SCRIPT}" 2>&1)
else
    SRTCTL_OUTPUT=$(srtctl apply -f "$CONFIG_FILE" --tags "gb200,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)" 2>&1)
fi
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
                # Files are of the format "results_concurrency_gpus_{num gpus}_ctx_{num ctx}_gen_{num gen}.json"
                filename=$(basename "$result_file")
                concurrency=$(echo "$filename" | sed -n 's/results_concurrency_\([0-9]*\)_gpus_.*/\1/p')
                gpus=$(echo "$filename" | sed -n 's/results_concurrency_[0-9]*_gpus_\([0-9]*\)_ctx_.*/\1/p')
                ctx=$(echo "$filename" | sed -n 's/.*_ctx_\([0-9]*\)_gen_.*/\1/p')
                gen=$(echo "$filename" | sed -n 's/.*_gen_\([0-9]*\)\.json/\1/p')

                echo "Processing concurrency $concurrency with $gpus GPUs (ctx: $ctx, gen: $gen): $result_file"

                WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${CONFIG_NAME}_conc${concurrency}_gpus_${gpus}_ctx_${ctx}_gen_${gen}.json"
                cp "$result_file" "$WORKSPACE_RESULT_FILE"

                echo "Copied result file to: $WORKSPACE_RESULT_FILE"
            fi
        done
    done
fi

echo "All result files processed"
