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
else
    export MODEL_PATH=$MODEL
fi

# Set up environment variables for SLURM
export SLURM_PARTITION="batch"
export SLURM_ACCOUNT="benchmark"
export SLURM_JOB_NAME="benchmark-dynamo.job"

NGINX_IMAGE="nginx:1.27.4"

SQUASH_FILE="/mnt/lustre01/users-public/sa-shared/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
NGINX_SQUASH_FILE="/mnt/lustre01/users-public/sa-shared/$(echo "$NGINX_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

srun -N 1 -A $SLURM_ACCOUNT -p $SLURM_PARTITION bash -c "enroot import -o $SQUASH_FILE docker://$IMAGE"
srun -N 1 -A $SLURM_ACCOUNT -p $SLURM_PARTITION bash -c "enroot import -o $NGINX_SQUASH_FILE docker://$NGINX_IMAGE"

export ISL="$ISL"
export OSL="$OSL"

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

git clone https://github.com/ishandhanani/srt-slurm.git "$SRT_REPO_DIR"
cd "$SRT_REPO_DIR"
git checkout sa-submission-q1-2026

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
  nginx-sqsh: ${NGINX_SQUASH_FILE}
EOF

echo "Generated srtslurm.yaml:"
cat srtslurm.yaml

echo "Running make setup..."
make setup ARCH=aarch64

# these 2 lines are for debugging
# TODO: remove when merge
echo "Make setup complete"
ls configs/

echo "Submitting job with srtctl..."

SETUP_SCRIPT=""

# --- Qwen3.5 EPD (Encoder/Prefill/Decode) disaggregation support ---
# If EPD is enabled, we generate a minimal srtctl config on the fly and start a dedicated
# encoder-only node (1 full GB200 node = 4 GPUs). This is intentionally *not* optimized; it is
# meant to be a working reference.
if [[ "${EPD:-}" == "1" || "${EPD:-}" == "true" ]]; then
    if [[ "$MODEL_PREFIX" == "qwen3.5" ]]; then
        echo "[INFO] EPD enabled for qwen3.5: generating recipe at '${CONFIG_FILE}'"
        mkdir -p "$(dirname "${CONFIG_FILE}")"

        # Compute stage node counts (heuristic): nodes_per_worker = ceil(max(tp, ep) / 4)
        GPUS_PER_NODE=4
        PREFILL_TP_VAL=${PREFILL_TP:-4}
        PREFILL_EP_VAL=${PREFILL_EP:-${PREFILL_TP_VAL}}
        DECODE_TP_VAL=${DECODE_TP:-4}
        DECODE_EP_VAL=${DECODE_EP:-${DECODE_TP_VAL}}

        PREFILL_WORLD=${PREFILL_TP_VAL}
        if [[ ${PREFILL_EP_VAL} -gt ${PREFILL_WORLD} ]]; then PREFILL_WORLD=${PREFILL_EP_VAL}; fi
        DECODE_WORLD=${DECODE_TP_VAL}
        if [[ ${DECODE_EP_VAL} -gt ${DECODE_WORLD} ]]; then DECODE_WORLD=${DECODE_EP_VAL}; fi

        PREFILL_NODES_PER_WORKER=$(( (PREFILL_WORLD + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
        DECODE_NODES_PER_WORKER=$(( (DECODE_WORLD + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))

        GEN_PREFILL_WORKERS=${PREFILL_NUM_WORKERS:-1}
        GEN_DECODE_WORKERS=${DECODE_NUM_WORKERS:-1}
        GEN_PREFILL_NODES=$(( PREFILL_NODES_PER_WORKER * GEN_PREFILL_WORKERS ))
        GEN_DECODE_NODES=$(( DECODE_NODES_PER_WORKER * GEN_DECODE_WORKERS ))

        # Build quantization args based on precision
        if [[ "${PRECISION}" == "fp8" ]]; then
            EPD_QUANT_ARGS="quantization: fp8
      kv-cache-dtype: fp8_e4m3"
        else
            EPD_QUANT_ARGS=""
        fi

        # Pick stable disaggregation bootstrap ports once so all nodes agree.
        PREFILL_BOOTSTRAP_PORT=${EPD_PREFILL_BOOTSTRAP_PORT:-$((52000 + RANDOM % 1000))}
        DECODE_BOOTSTRAP_PORT=${EPD_DECODE_BOOTSTRAP_PORT:-$((53000 + RANDOM % 1000))}
        echo "[INFO] Using EPD bootstrap ports: prefill=${PREFILL_BOOTSTRAP_PORT}, decode=${DECODE_BOOTSTRAP_PORT}"

        cat > "${CONFIG_FILE}" <<EOF
name: qwen3.5-epd-${PRECISION}-gb200
model:
  path: ${SRT_SLURM_MODEL_PREFIX}
  container: dynamo-sglang
  precision: ${PRECISION}
resources:
  gpu_type: gb200
  gpus_per_node: 4
  prefill_nodes: ${GEN_PREFILL_NODES}
  prefill_workers: ${GEN_PREFILL_WORKERS}
  decode_nodes: ${GEN_DECODE_NODES}
  decode_workers: ${GEN_DECODE_WORKERS}
infra:
  # Reserve one full node for infra (etcd/nats). We also run the vision encoder there.
  etcd_nats_dedicated_node: true
backend:
  type: sglang
  sglang_config:
    prefill:
      tp-size: ${PREFILL_TP_VAL}
      ep-size: ${PREFILL_EP_VAL}
      disaggregation-mode: prefill
      disaggregation-transfer-backend: nixl
      disaggregation-bootstrap-port: ${DECODE_BOOTSTRAP_PORT}   
      language-only: true
      # 4 encoder instances (1 GPU each) on the dedicated infra node
      encoder-urls:
        - "http://{head_node_ip}:40000"
        - "http://{head_node_ip}:40001"
        - "http://{head_node_ip}:40002"
        - "http://{head_node_ip}:40003"
      trust-remote-code: true
      ${EPD_QUANT_ARGS}
      context-length: ${MAX_MODEL_LEN}
    decode:
      tp-size: ${DECODE_TP_VAL}
      ep-size: ${DECODE_EP_VAL}
      disaggregation-mode: decode
      disaggregation-transfer-backend: nixl
      disaggregation-bootstrap-port: ${DECODE_BOOTSTRAP_PORT}
      language-only: true
      encoder-urls:
        - "http://{head_node_ip}:40000"
        - "http://{head_node_ip}:40001"
        - "http://{head_node_ip}:40002"
        - "http://{head_node_ip}:40003"
      trust-remote-code: true
      ${EPD_QUANT_ARGS}
      context-length: ${MAX_MODEL_LEN}
benchmark:
  type: sa-bench
  isl: ${ISL}
  osl: ${OSL}
  concurrencies: [${MAX_CONCURRENCY}]
EOF

        # Setup script: install torchao and start 4 encoder-only servers on the infra node.
        cat > configs/qwen3.5-epd-setup.sh <<'EOF'
#!/usr/bin/env bash
set -euxo pipefail

# Install torchao if the repo provides the helper (used for fp8 runs)
if [[ -f /configs/install-torchao.sh ]]; then
  bash /configs/install-torchao.sh
fi

# Start encoder-only servers on the first allocated node (reserved when infra.etcd_nats_dedicated_node=true)
# Prefer scontrol if available; otherwise fall back to SLURM_NODEID==0.
if command -v scontrol >/dev/null 2>&1; then
  HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
else
  echo "[EPD] WARNING: scontrol not found; using SLURM_NODEID==0 to pick a single encoder node"
  HEAD_NODE=""
fi
THIS_NODE="$(hostname -s)"

# Prefer HEAD_NODE_IP (present in env from srtctl/srt-slurm)
if [[ -n "${HEAD_NODE_IP:-}" ]]; then
  if command -v ip >/dev/null 2>&1; then
    ip -4 addr show | grep -qw "${HEAD_NODE_IP}" || exit 0
  else
    # fallback
    (hostname -I 2>/dev/null || true) | grep -qw "${HEAD_NODE_IP}" || exit 0
  fi
else
  # last resort if HEAD_NODE_IP isn't set
  if command -v scontrol >/dev/null 2>&1 && [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
    [[ "${THIS_NODE}" == "${HEAD_NODE}" ]] || exit 0
  else
    echo "[EPD] WARNING: cannot determine head node; skipping encoder launch to avoid port conflicts"
    exit 0
  fi
fi

echo "[EPD] Starting encoder-only servers on ${THIS_NODE}"
PORT_BASE=$((40000 + (SLURM_JOB_ID % 1000) * 10))

for GPU_ID in 0 1 2 3; do
  PORT=$((PORT_BASE + GPU_ID))

  # Idempotency: if something is already listening, don't try to bind again.
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

# Best-effort wait for ports to open (no hard failure)
for GPU_ID in 0 1 2 3; do
  PORT=$((PORT_BASE + GPU_ID))
  for _ in $(seq 1 120); do
    (echo > /dev/tcp/127.0.0.1/${PORT}) >/dev/null 2>&1 && break || true
    sleep 1
  done
done

echo "[EPD] Encoder servers ready"

python3 - <<'PY'
from pathlib import Path
import re

p = Path("/sgl-workspace/sglang/python/sglang/srt/disaggregation/utils.py")
txt = p.read_text()

# Only change the bootstrap_room tensor dtype
txt2 = re.sub(
    r"(self\.bootstrap_room\s*=\s*torch\.zeros\(\s*\(size,\s*8\),\s*dtype=torch\.)int64",
    r"\1uint64",
    txt,
    count=1,
)

if txt2 == txt:
    raise SystemExit(f"[bootstrap_room patch] Pattern not found in {p} (file changed?)")

p.write_text(txt2)
print(f"[bootstrap_room patch] Patched {p} to use torch.uint64 for bootstrap_room")
PY

EOF
        chmod +x configs/qwen3.5-epd-setup.sh
        SETUP_SCRIPT="qwen3.5-epd-setup.sh"
    fi
fi

# Default setup script for dynamo-sglang (fp8 tooling)
if [[ "$FRAMEWORK" == "dynamo-sglang" && -z "${SETUP_SCRIPT}" ]]; then
    SETUP_SCRIPT="install-torchao.sh"
fi

if [[ "$FRAMEWORK" == "dynamo-sglang" ]]; then
    SRTCTL_OUTPUT=$(srtctl apply -f "$CONFIG_FILE" --tags "gb200,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)" --setup-script "${SETUP_SCRIPT}" 2>&1)
else
    SRTCTL_OUTPUT=$(srtctl apply -f "$CONFIG_FILE" --tags "gb200,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)" 2>&1)
fi
echo "$SRTCTL_OUTPUT"

JOB_ID=$(echo "$SRTCTL_OUTPUT" | grep -oP '✅ Job \K[0-9]+' || echo "$SRTCTL_OUTPUT" | grep -oP 'Job \K[0-9]+')

if [ -z "$JOB_ID" ]; then
    echo "Error: Failed to extract JOB_ID from srtctl output"
    exit 1
fi

echo "Extracted JOB_ID: $JOB_ID"

# Wait for this specific job to complete
echo "Waiting for job $JOB_ID to complete..."
while [ -n "$(squeue -j $JOB_ID --noheader 2>/dev/null)" ]; do
    echo "Job $JOB_ID still running..."
    squeue -j $JOB_ID
    sleep 30
done
echo "Job $JOB_ID completed!"

echo "Collecting results..."

# Use the JOB_ID to find the logs directory
# srtctl creates logs in outputs/JOB_ID/logs/
LOGS_DIR="outputs/$JOB_ID/logs"

if [ ! -d "$LOGS_DIR" ]; then
    echo "Warning: Logs directory not found at $LOGS_DIR"
    exit 1
fi

echo "Found logs directory: $LOGS_DIR"

cat $LOGS_DIR/sweep_$JOB_ID.log

for file in $LOGS_DIR/*; do
    if [ -f "$file" ]; then
        tail -n 500 $file
    fi
done

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

# Cleanup
echo "Cleaning up..."
deactivate 2>/dev/null || true
rm -rf .venv
echo "Cleanup complete"

echo "All result files processed"