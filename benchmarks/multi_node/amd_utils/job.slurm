#!/bin/bash
#SBATCH --job-name=1p2d_bench-serving    # Specify a custom string for your slurm batch job
#SBATCH -N 3            # CHECK this to be right in batch jobs
#SBATCH -n 3          # CHECK this to be right in batch jobs
#SBATCH --ntasks-per-node=1
#SBATCH --spread-job
#SBATCH --gres=gpu:8      # Request 8 GPUs and 8 NICs (use --gres if specific GPU resources are needed)
#SBATCH --time=24:00:00         # Set a time limit for the job (HH:MM:SS)
# --output and --error are set by submit.sh via BENCHMARK_LOGS_DIR


# ------------------------
# Print current time in UTC and PST formats
# ------------------------
echo "=== Job Start Time ==="
echo "UTC Time: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo "PST Time: $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')"
echo "======================="
echo ""

# =============================================================================
# Model validation from models.yaml (replaces hardcoded VALID_MODELS array)
# =============================================================================
# DI_REPO_DIR is set below from $(pwd); use the submit-time working directory
# because sbatch copies this script to /var/spool/slurmd/ at runtime.
MODELS_YAML="$(pwd)/models.yaml"

if [[ ! -f "$MODELS_YAML" ]]; then
    echo "Error: models.yaml not found at $MODELS_YAML"
    exit 1
fi

# Validate MODEL_NAME exists as a top-level key in models.yaml
if ! grep -q "^${MODEL_NAME}:" "$MODELS_YAML"; then
    echo "Error: Model '$MODEL_NAME' not found in models.yaml"
    echo "Available models:"
    grep -E '^[A-Za-z]' "$MODELS_YAML" | sed 's/:.*$//' | sed 's/^/  - /'
    exit 1
fi
echo "Model found: $MODEL_NAME"

# All models use server.sh as the entrypoint
RUN_FILE="server.sh"
echo "Runfile set: $RUN_FILE"

if [[ -z "${DOCKER_IMAGE_NAME:-}" ]]; then
    echo "Error: DOCKER_IMAGE_NAME is not set."
    exit 1
fi

# Set current directory to be REPO directory with all relevant scripts
export DI_REPO_DIR=$(pwd)

xP="${xP:-1}" #-> Number of Prefill Workers
yD="${yD:-1}" #-> Number of Decode Workers

# Parallelism Configuration with defaults
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-8}"
PREFILL_ENABLE_EP="${PREFILL_ENABLE_EP:-true}"
PREFILL_ENABLE_DP="${PREFILL_ENABLE_DP:-true}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-8}"
DECODE_ENABLE_EP="${DECODE_ENABLE_EP:-true}"
DECODE_ENABLE_DP="${DECODE_ENABLE_DP:-true}"
DECODE_MTP_SIZE=${DECODE_MTP_SIZE:-0} # 0 for disabling MTP

# Benchmark Configuration with defaults
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-1024}"
BENCH_RANDOM_RANGE_RATIO="${BENCH_RANDOM_RANGE_RATIO:-1}"
BENCH_NUM_PROMPTS_MULTIPLIER="${BENCH_NUM_PROMPTS_MULTIPLIER:-10}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-512}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

MODEL_NAME="${MODEL_NAME:-None}"
MODEL_DIR="${MODEL_DIR:-"/nfsdata"}"

# ------------------------
# Model path validation and selection across all nodes
# ------------------------
echo "Looking for model: $MODEL_NAME"
echo "Checking model availability across all allocated nodes..."

# Get all allocated nodes
ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
TOTAL_NODES=$(echo "$ALL_NODES" | wc -l)

echo "Total allocated nodes: $TOTAL_NODES"
echo "Nodes: $(echo "$ALL_NODES" | tr '\n' ' ')"

# Function to check model path on all nodes
check_model_path() {
    local path=$1
    local check_name=$2

    echo "Checking $check_name: $path"

    # Run check on all nodes in parallel
    srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES /bin/bash -c "
        if [ -d '$path' ]; then
            echo \"\$(hostname): ✓ Found $path\"
            exit 0
        else
            echo \"\$(hostname): ✗ Missing $path\"
            exit 1
        fi
    "

    # Check if all nodes succeeded (exit code 0)
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ $check_name available on ALL nodes"
        return 0
    else
        echo "✗ $check_name NOT available on all nodes"
        return 1
    fi
}

# Check model weights exist on "$MODEL_DIR/$MODEL_NAME"
if check_model_path "$MODEL_DIR/$MODEL_NAME" "$MODEL_DIR"; then
    MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
    echo ""
    echo "✓ Selected MODEL_PATH: $MODEL_PATH (available on all nodes)"
else
    echo ""
    echo "✗ FATAL ERROR: Model '$MODEL_NAME' not found on ALL allocated nodes in the following:"
    echo "  - $MODEL_DIR/$MODEL_NAME"
    echo ""
    echo "Model must be accessible from all nodes for distributed execution."
    echo "Please ensure the model is available on all allocated nodes."
    exit 1
fi

echo "Final MODEL_PATH: $MODEL_PATH"
echo ""

NUM_NODES="${NUM_NODES}"

# ------------------------
# Extract first NUM_NODES from SLURM allocation and update SLURM variables
# ------------------------
echo "Original SLURM allocation:"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"

# Get the full nodelist and extract first NUM_NODES
FULL_NODELIST=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
SELECTED_NODES=$(echo "$FULL_NODELIST" | head -n $NUM_NODES)
SELECTED_NODELIST_STR=$(echo "$SELECTED_NODES" | tr '\n' ',' | sed 's/,$//')

# Create new nodelist in SLURM format
# This is a simplified approach - for complex ranges, you might need more sophisticated parsing
NEW_SLURM_NODELIST=$(echo "$SELECTED_NODES" | paste -sd, | sed 's/,/,/g')

# Update SLURM environment variables
export SLURM_NNODES=$NUM_NODES
export SLURM_NTASKS=$NUM_NODES
export SLURM_JOB_NUM_NODES=$NUM_NODES
export SLURM_NPROCS=$NUM_NODES
export SLURM_JOB_NODELIST="$NEW_SLURM_NODELIST"
export SLURM_NODELIST="$NEW_SLURM_NODELIST"

# Keep other SLURM variables as they were or set defaults
export SLURM_TASKS_PER_NODE="1(x$NUM_NODES)"
export SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/home/bill}"
export SLURM_CLUSTER_NAME="${SLURM_CLUSTER_NAME:-m2m}"
export SLURM_JOB_CPUS_PER_NODE="${SLURM_JOB_CPUS_PER_NODE}"
export SLURM_JOB_PARTITION="${SLURM_JOB_PARTITION:-amd-rccl}"
export SLURM_JOBID="${SLURM_JOBID:-$SLURM_JOB_ID}"
export SLURM_JOB_QOS="${SLURM_JOB_QOS:-normal}"
export SLURM_JOB_ACCOUNT="${SLURM_JOB_ACCOUNT:-amd-rccl}"
export SLURM_NTASKS_PER_NODE=1
export SLURM_SUBMIT_HOST="${SLURM_SUBMIT_HOST}"
export SLURM_JOB_ID="${SLURM_JOB_ID}"
export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"
export SLURM_JOB_NAME="${SLURM_JOB_NAME:-1p1d_bench-serving}"

echo ""
echo "Updated SLURM Environment Variables:"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "SLURM_JOB_CPUS_PER_NODE: $SLURM_JOB_CPUS_PER_NODE"
echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_JOB_QOS: $SLURM_JOB_QOS"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_JOB_ACCOUNT: $SLURM_JOB_ACCOUNT"
echo "SLURM_NPROCS: $SLURM_NPROCS"
echo "SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "SLURM_CONF: $SLURM_CONF"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "SLURM_CLUSTER_NAME: $SLURM_CLUSTER_NAME"
echo "ulimit: $(ulimit -a)"
echo ""
echo "Selected nodes for execution:"
echo "$SELECTED_NODES"
echo ""

# Node information
USER_NAME=$(whoami)
MASTER_NODE=$(echo "$SELECTED_NODES" | head -n 1)
NODE0_ADDR=$(srun --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="$MASTER_NODE" bash -c 'ip route get 1.1.1.1')
NODE0_ADDR=$(echo "$NODE0_ADDR" | awk '/src/ {print $7}')

IPS=()

GW_NIC=$(ip route | awk '/^default/ {print $5; exit}')
for NODE in $SELECTED_NODES; do
    IP=$(srun --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="$NODE" bash -c 'ip route get 1.1.1.1')
    IP=$(echo "$IP" | awk '/src/ {print $7}')
    IPS+=("$IP")
done

echo "Selected node IPs: ${IPS[*]}" | sed 's/ /,/g'

SGL_WS_PATH="/multi_node/amd_utils"
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

NNODES=$NUM_NODES

echo "MASTER_NODE is ${MASTER_NODE}"
echo "NODE0_ADDR is ${NODE0_ADDR}"
echo "NNODES is ${NNODES}"
echo "REPO Directory is ${DI_REPO_DIR}"
echo "USER_NAME is ${USER_NAME}"

# Get the RDMA priority and DSCP value from the NIC
if ! command -v nicctl >/dev/null 2>&1; then
    echo "Error: nicctl command not found. Please ensure nicctl is installed and available." >&2
    exit 1
fi


export DI_REPO_DIR=$DI_REPO_DIR
export SGL_WS_PATH=$SGL_WS_PATH
export NNODES=$NNODES
export NODE0_ADDR=$NODE0_ADDR
export MODEL_PATH=$MODEL_PATH
export MODEL_DIR=$MODEL_DIR
export xP=$xP
export yD=$yD
export MODEL_NAME=$MODEL_NAME
export USER_NAME=$USER_NAME
export IPADDRS="$(echo "${IPS[*]}" | sed 's/ /,/g')"
export PREFILL_TP_SIZE=$PREFILL_TP_SIZE
export PREFILL_ENABLE_EP=$PREFILL_ENABLE_EP
export PREFILL_ENABLE_DP=$PREFILL_ENABLE_DP
export DECODE_TP_SIZE=$DECODE_TP_SIZE
export DECODE_ENABLE_EP=$DECODE_ENABLE_EP
export DECODE_ENABLE_DP=$DECODE_ENABLE_DP
export DECODE_MTP_SIZE=$DECODE_MTP_SIZE
export GPUS_PER_NODE=$GPUS_PER_NODE
export BENCH_INPUT_LEN=$BENCH_INPUT_LEN
export BENCH_OUTPUT_LEN=$BENCH_OUTPUT_LEN
export BENCH_RANDOM_RANGE_RATIO=$BENCH_RANDOM_RANGE_RATIO
export BENCH_NUM_PROMPTS_MULTIPLIER=$BENCH_NUM_PROMPTS_MULTIPLIER
export BENCH_MAX_CONCURRENCY=$BENCH_MAX_CONCURRENCY
export DRY_RUN="${DRY_RUN:-0}"
export BENCHMARK_LOGS_DIR="${BENCHMARK_LOGS_DIR:-$(pwd)/benchmark_logs}"

SANITIZED_USER=$(echo "$USER_NAME" | tr -c 'a-zA-Z0-9_.-' '_')
export DOCKER_CONT_NAME="container_sbatch_${SANITIZED_USER}_${MODEL_NAME}_${SLURM_JOB_ID}"
export RUN_FILE_FULL="$SGL_WS_PATH/${RUN_FILE}"


# Use only the selected nodes for srun execution
SELECTED_NODELIST_SRUN=$(echo "$SELECTED_NODES" | paste -sd,)


cleanup() {
  echo "[${SLURM_JOB_ID}] termination received on $(hostname); cleaning stale logs folder..."
  # clean up the logs folder
  sudo rm -rf ${SLURM_SUBMIT_DIR}/logs 2>/dev/null || true

  echo "[${SLURM_JOB_ID}] cleanup done."
}

trap cleanup INT TERM HUP


# Force NFS cache refresh on all nodes before running Docker to avoid stale file handle errors
echo "Refreshing NFS caches on all nodes..."
srun --nodelist="$SELECTED_NODELIST_SRUN" bash -c '
    sync
    # Force re-stat of the mounted directory to refresh NFS handles
    ls -la '"$DI_REPO_DIR"' > /dev/null 2>&1
    stat '"$DI_REPO_DIR"'/server.sh > /dev/null 2>&1
    cat '"$DI_REPO_DIR"'/server.sh > /dev/null 2>&1
    # Drop caches if we have permission (optional, requires root)
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true
    echo "NFS cache refreshed on $(hostname)"
'

srun \
  --nodelist="$SELECTED_NODELIST_SRUN" \
  --kill-on-bad-exit=1 \
  --signal=TERM@30 \
  --unbuffered \
  bash -lc "
set -euo pipefail

echo \"Rank \$SLURM_PROCID on \$(hostname)\"

# Pre-clean (idempotent)
sudo docker ps -aq --filter \"name=^container_sbatch_\" | xargs -r sudo docker rm -f || true
sudo docker ps -aq | xargs -r sudo docker stop || true

exec sudo docker run --rm \
    --init \
    --stop-timeout 10 \
    --device /dev/dri \
    --device /dev/kfd \
    --device /dev/infiniband \
    --network host \
    --ipc host \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    -v ${MODEL_DIR}:/models \
    -v \$HOME/.ssh:/root/.ssh \
    -v $(which nicctl):/usr/sbin/nicctl \
    --shm-size 128G \
    -v /tmp:/run_logs \
    -v ${BENCHMARK_LOGS_DIR}:/benchmark_logs \
    -v ${DI_REPO_DIR}:${SGL_WS_PATH} \
    -e SLURM_JOB_ID=\$SLURM_JOB_ID \
    -e SLURM_JOB_NODELIST=\$SLURM_JOB_NODELIST \
    -e NNODES=\$NNODES \
    -e NODE_RANK=\$SLURM_PROCID \
    -e NODE0_ADDR=\$NODE0_ADDR \
    -e MODEL_DIR=/models \
    -e SGL_WS_PATH=${SGL_WS_PATH} \
    -e GPUS_PER_NODE=\$GPUS_PER_NODE \
    -e xP=\$xP \
    -e yD=\$yD \
    -e MODEL_NAME=\$MODEL_NAME \
    -e IPADDRS=\$IPADDRS \
    -e PREFILL_TP_SIZE=\$PREFILL_TP_SIZE \
    -e PREFILL_ENABLE_EP=\$PREFILL_ENABLE_EP \
    -e PREFILL_ENABLE_DP=\$PREFILL_ENABLE_DP \
    -e DECODE_TP_SIZE=\$DECODE_TP_SIZE \
    -e DECODE_ENABLE_EP=\$DECODE_ENABLE_EP \
    -e DECODE_ENABLE_DP=\$DECODE_ENABLE_DP \
    -e DECODE_MTP_SIZE=\$DECODE_MTP_SIZE \
    -e BENCH_INPUT_LEN=\$BENCH_INPUT_LEN \
    -e BENCH_OUTPUT_LEN=\$BENCH_OUTPUT_LEN \
    -e BENCH_RANDOM_RANGE_RATIO=\$BENCH_RANDOM_RANGE_RATIO \
    -e BENCH_NUM_PROMPTS_MULTIPLIER=\$BENCH_NUM_PROMPTS_MULTIPLIER \
    -e BENCH_MAX_CONCURRENCY=\$BENCH_MAX_CONCURRENCY \
    -e DRY_RUN=\$DRY_RUN \
    -e BENCHMARK_LOGS_DIR=/benchmark_logs \
    --name \"$DOCKER_CONT_NAME\" \
    \"$DOCKER_IMAGE_NAME\" bash -lc '
        mkdir -p /run_logs/slurm_job-'\"\$SLURM_JOB_ID\"'
        '"$RUN_FILE_FULL"' 2>&1 | tee /run_logs/slurm_job-'\"\$SLURM_JOB_ID\"'/server_\$(hostname).log
    '

DOCKER_EXIT_CODE=\$?
if [[ \$DOCKER_EXIT_CODE -ne 0 ]]; then
  echo \"ERROR: docker exited rc=\$DOCKER_EXIT_CODE on \$(hostname)\"
  exit \$DOCKER_EXIT_CODE
fi
"

srun --nodelist="$SELECTED_NODELIST_SRUN" bash -c 'sudo docker rm -f $DOCKER_CONT_NAME 2>/dev/null || true'
