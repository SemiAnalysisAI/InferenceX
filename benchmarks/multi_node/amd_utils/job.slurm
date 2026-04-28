#!/bin/bash
#SBATCH --job-name=disagg-bench
#SBATCH -N 3            # Overridden by submit.sh -N flag
#SBATCH -n 3            # Overridden by submit.sh -n flag
#SBATCH --ntasks-per-node=1
#SBATCH --spread-job
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
# --output and --error are set by submit.sh via BENCHMARK_LOGS_DIR

ENGINE="${ENGINE:-sglang-disagg}"

echo "=== Job Start Time ==="
echo "UTC Time: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo "PST Time: $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')"
echo "ENGINE: $ENGINE"
echo "======================="
echo ""

# =============================================================================
# Model Validation
# =============================================================================

# Use $(pwd) not BASH_SOURCE — sbatch copies the script to /var/spool/slurmd/
# at runtime, but the CWD remains the submit-time directory (amd_utils/).
if [[ "$ENGINE" == "vllm-disagg" ]]; then
    MODELS_YAML="$(pwd)/models_vllm.yaml"
else
    MODELS_YAML="$(pwd)/models.yaml"
fi

if [[ ! -f "$MODELS_YAML" ]]; then
    echo "Error: models YAML not found at $MODELS_YAML"
    exit 1
fi

if [[ -z "${DOCKER_IMAGE_NAME:-}" ]]; then
    echo "Error: DOCKER_IMAGE_NAME is not set."
    exit 1
fi

MODEL_NAME="${MODEL_NAME:-None}"
if ! grep -q "^${MODEL_NAME}:" "$MODELS_YAML"; then
    echo "Error: Model '$MODEL_NAME' not found in $MODELS_YAML"
    echo "Available models:"
    grep -E '^[A-Za-z]' "$MODELS_YAML" | sed 's/:.*$//' | sed 's/^/  - /'
    exit 1
fi
echo "Model found: $MODEL_NAME"

RUN_FILE="server.sh"
echo "Runfile set: $RUN_FILE"

# DI_REPO_DIR points to the repo root.
# $(pwd) is amd_utils/ (the sbatch submit dir); go up 3 levels to reach the repo root.
export DI_REPO_DIR=$(cd "$(pwd)/../../.." && pwd)

xP="${xP:-1}"
yD="${yD:-1}"

# Benchmark configuration
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-1024}"
BENCH_RANDOM_RANGE_RATIO="${BENCH_RANDOM_RANGE_RATIO:-1}"
BENCH_NUM_PROMPTS_MULTIPLIER="${BENCH_NUM_PROMPTS_MULTIPLIER:-10}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-512}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-inf}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

# Engine-specific defaults
PREFILL_ENABLE_EP="${PREFILL_ENABLE_EP:-false}"
PREFILL_ENABLE_DP="${PREFILL_ENABLE_DP:-false}"
DECODE_ENABLE_EP="${DECODE_ENABLE_EP:-false}"
DECODE_ENABLE_DP="${DECODE_ENABLE_DP:-false}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-8}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-8}"
DECODE_MTP_SIZE=${DECODE_MTP_SIZE:-0}

# Router selection: "vllm-router" (external container) or "moriio" (in-container proxy)
ROUTER_TYPE="${ROUTER_TYPE:-vllm-router}"
ROUTER_PORT="${ROUTER_PORT:-30000}"
PROXY_PING_PORT="${PROXY_PING_PORT:-36367}"

# =============================================================================
# Docker privilege detection
# =============================================================================
# Detect on the batch host. Per-node detection happens inside srun below.
if docker ps &>/dev/null; then
    DOCKER_CMD="docker"
else
    DOCKER_CMD="sudo docker"
fi
export DOCKER_CMD

# =============================================================================
# Model Path Resolution
# =============================================================================

# MODEL_DIR detection: prefer env var, fall back to hostname detection
if [[ -z "$MODEL_DIR" ]]; then
    NODENAME=$(hostname -s)
    if [[ $NODENAME == GPU* ]] || [[ $NODENAME == smci355-ccs-aus* ]]; then
        MODEL_DIR="/nfsdata"
    elif [[ $NODENAME == mia1* ]]; then
        MODEL_DIR="/it-share/data"
    else
        MODEL_DIR="/nfsdata"
    fi
    echo "[INFO] Auto-detected MODEL_DIR=$MODEL_DIR from hostname $(hostname -s)"
fi
export MODEL_DIR

if [[ "$ENGINE" == "vllm-disagg" ]]; then
    # vLLM: Extract hf_dir from models.yaml, search multiple paths, resolve HF cache snapshots
    DISK_DIR_NAME=$(awk '/^'"$MODEL_NAME"':/{found=1; next}
        found && /^[^ ]/{exit}
        found && /hf_dir:/{gsub(/[" ]/, "", $2); print $2; exit}' "$MODELS_YAML")
    DISK_DIR_NAME="${DISK_DIR_NAME:-$MODEL_NAME}"
    echo "Looking for model: $MODEL_NAME (disk dir: $DISK_DIR_NAME)"

    resolve_hf_cache_path() {
        local base_path=$1
        if [[ -d "${base_path}/snapshots" ]]; then
            local snapshot=$(ls -1 "${base_path}/snapshots" 2>/dev/null | head -1)
            if [[ -n "$snapshot" ]]; then
                echo "${base_path}/snapshots/${snapshot}"
                return 0
            fi
        fi
        echo "$base_path"
        return 1
    }

    MODEL_PATH=""
    SEARCH_PATHS=(
        "${MODEL_DIR}/${DISK_DIR_NAME}"
        "${MODEL_DIR}/${MODEL_NAME}"
        "/nfsdata/hf_hub_cache-0/${DISK_DIR_NAME}"
        "/nfsdata/hf_hub_cache-0/${MODEL_NAME}"
    )

    for search_path in "${SEARCH_PATHS[@]}"; do
        if [[ -d "$search_path" ]]; then
            RESOLVED=$(resolve_hf_cache_path "$search_path")
            MODEL_PATH="$RESOLVED"
            echo "Found MODEL_PATH: $MODEL_PATH"
            break
        fi
    done

    if [[ -z "$MODEL_PATH" ]]; then
        echo "FATAL: Model '$MODEL_NAME' not found. Searched:"
        for p in "${SEARCH_PATHS[@]}"; do echo "  - $p"; done
        exit 1
    fi
    echo "Final MODEL_PATH: $MODEL_PATH"
else
    # SGLang: Validate model path across all allocated nodes
    echo "Looking for model: $MODEL_NAME"
    echo "Checking model availability across all allocated nodes..."

    ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    TOTAL_NODES=$(echo "$ALL_NODES" | wc -l)
    echo "Total allocated nodes: $TOTAL_NODES"
    echo "Nodes: $(echo "$ALL_NODES" | tr '\n' ' ')"

    check_model_path() {
        local path=$1
        local check_name=$2
        echo "Checking $check_name: $path"
        srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES /bin/bash -c "
            if [ -d '$path' ]; then
                echo \"\$(hostname): Found $path\"
                exit 0
            else
                echo \"\$(hostname): Missing $path\"
                exit 1
            fi
        "
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "$check_name available on ALL nodes"
            return 0
        else
            echo "$check_name NOT available on all nodes"
            return 1
        fi
    }

    if check_model_path "$MODEL_DIR/$MODEL_NAME" "$MODEL_DIR"; then
        MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
        echo "Selected MODEL_PATH: $MODEL_PATH (available on all nodes)"
    else
        echo "FATAL ERROR: Model '$MODEL_NAME' not found on ALL allocated nodes in:"
        echo "  - $MODEL_DIR/$MODEL_NAME"
        exit 1
    fi
    echo "Final MODEL_PATH: $MODEL_PATH"
fi

# =============================================================================
# Node Selection
# =============================================================================

NUM_NODES=$((xP + yD))
echo "NUM_NODES: $NUM_NODES (xP=$xP + yD=$yD)"

FULL_NODELIST=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
SELECTED_NODES=$(echo "$FULL_NODELIST" | head -n $NUM_NODES)
SELECTED_NODELIST_STR=$(echo "$SELECTED_NODES" | tr '\n' ',' | sed 's/,$//')

# Update SLURM environment variables
export SLURM_NNODES=$NUM_NODES
export SLURM_NTASKS=$NUM_NODES
export SLURM_JOB_NUM_NODES=$NUM_NODES
export SLURM_NPROCS=$NUM_NODES
export SLURM_JOB_NODELIST="$SELECTED_NODELIST_STR"
export SLURM_NODELIST="$SELECTED_NODELIST_STR"
export SLURM_TASKS_PER_NODE="1(x$NUM_NODES)"
export SLURM_NTASKS_PER_NODE=1

echo ""
echo "Selected nodes: $SELECTED_NODELIST_STR"

# =============================================================================
# IP Resolution
# =============================================================================

USER_NAME=$(whoami)
MASTER_NODE=$(echo "$SELECTED_NODES" | head -n 1)
NODE0_ADDR=$(srun --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="$MASTER_NODE" bash -c 'ip route get 1.1.1.1')
NODE0_ADDR=$(echo "$NODE0_ADDR" | awk '/src/ {print $7}')

IPS=()
for NODE in $SELECTED_NODES; do
    IP=$(srun --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="$NODE" bash -c 'ip route get 1.1.1.1')
    IP=$(echo "$IP" | awk '/src/ {print $7}')
    IPS+=("$IP")
done

echo "Node IPs: ${IPS[*]}"

DOCKER_MOUNT_PATH="/workspace"
WS_PATH="${DOCKER_MOUNT_PATH}/benchmarks/multi_node/amd_utils"

NNODES=$NUM_NODES

echo "MASTER_NODE: ${MASTER_NODE}"
echo "NODE0_ADDR:  ${NODE0_ADDR}"
echo "NNODES:      ${NNODES}"
echo "REPO DIR:    ${DI_REPO_DIR}"
echo "USER:        ${USER_NAME}"

# Reduce log spam
export TQDM_MININTERVAL=20

# Translate the host-resolved MODEL_PATH to the Docker mount namespace
DOCKER_MODEL_PATH="${MODEL_PATH/#$MODEL_DIR//models}"

export DI_REPO_DIR=$DI_REPO_DIR
export WS_PATH=$WS_PATH
export NNODES=$NNODES
export NODE0_ADDR=$NODE0_ADDR
export MODEL_PATH=$MODEL_PATH
export MODEL_DIR=$MODEL_DIR
export xP=$xP
export yD=$yD
export MODEL_NAME=$MODEL_NAME
export USER_NAME=$USER_NAME
export IPADDRS="$(echo "${IPS[*]}" | sed 's/ /,/g')"
export GPUS_PER_NODE=$GPUS_PER_NODE
export BENCH_INPUT_LEN=$BENCH_INPUT_LEN
export BENCH_OUTPUT_LEN=$BENCH_OUTPUT_LEN
export BENCH_RANDOM_RANGE_RATIO=$BENCH_RANDOM_RANGE_RATIO
export BENCH_NUM_PROMPTS_MULTIPLIER=$BENCH_NUM_PROMPTS_MULTIPLIER
export BENCH_MAX_CONCURRENCY=$BENCH_MAX_CONCURRENCY
export BENCH_REQUEST_RATE=$BENCH_REQUEST_RATE
export DRY_RUN="${DRY_RUN:-0}"
export BENCHMARK_LOGS_DIR="${BENCHMARK_LOGS_DIR:-$(pwd)/benchmark_logs}"
export KEEP_CONTAINERS="${KEEP_CONTAINERS:-0}"
export ENGINE=$ENGINE

# Eval-related env vars (threaded from submit.sh)
export RUN_EVAL="${RUN_EVAL:-false}"
export EVAL_ONLY="${EVAL_ONLY:-false}"
export EVAL_CONC="${EVAL_CONC:-}"
export FRAMEWORK="${FRAMEWORK:-}"
export PRECISION="${PRECISION:-}"
export MODEL_PREFIX="${MODEL_PREFIX:-}"
export RUNNER_TYPE="${RUNNER_TYPE:-}"
export RESULT_FILENAME="${RESULT_FILENAME:-}"
export SPEC_DECODING="${SPEC_DECODING:-}"

SANITIZED_USER=$(echo "$USER_NAME" | tr -c 'a-zA-Z0-9_.-' '_')
export DOCKER_CONT_NAME="container_${ENGINE}_${SANITIZED_USER}_${MODEL_NAME}_${SLURM_JOB_ID}"

# vLLM external router container
VLLM_ROUTER_IMAGE="${VLLM_ROUTER_IMAGE:-ghcr.io/simondanielsson/vllm-router:dev-streaming-cn-cjy}"
ROUTER_CONT_NAME="router_vllm_${SANITIZED_USER}_${SLURM_JOB_ID}"
export RUN_FILE_FULL="$WS_PATH/${RUN_FILE}"

SELECTED_NODELIST_SRUN=$(echo "$SELECTED_NODES" | paste -sd,)

cleanup() {
  echo "[${SLURM_JOB_ID}] termination received on $(hostname); cleaning up..."
  rm -rf ${SLURM_SUBMIT_DIR}/logs 2>/dev/null || true
  echo "[${SLURM_JOB_ID}] cleanup done."
}

trap cleanup INT TERM HUP

# Force NFS cache refresh on all nodes
echo "Refreshing NFS caches on all nodes..."
srun --nodelist="$SELECTED_NODELIST_SRUN" bash -c '
    sync
    ls -la '"$DI_REPO_DIR"'/benchmarks/multi_node/amd_utils > /dev/null 2>&1
    stat '"$DI_REPO_DIR"'/benchmarks/multi_node/amd_utils/server.sh > /dev/null 2>&1
    cat '"$DI_REPO_DIR"'/benchmarks/multi_node/amd_utils/server.sh > /dev/null 2>&1
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true
    echo "NFS cache refreshed on $(hostname)"
'

# =============================================================================
# Build engine-specific Docker environment variables
# =============================================================================

# Common env vars (always passed)
DOCKER_ENV_COMMON=(
    -e SLURM_JOB_ID=\$SLURM_JOB_ID
    -e SLURM_JOB_NODELIST=\$SLURM_JOB_NODELIST
    -e NNODES=\$NNODES
    -e NODE_RANK=\$SLURM_PROCID
    -e NODE0_ADDR=\$NODE0_ADDR
    -e MODEL_DIR=/models
    -e MODEL_NAME=\$MODEL_NAME
    -e GPUS_PER_NODE=\$GPUS_PER_NODE
    -e xP=\$xP
    -e yD=\$yD
    -e IPADDRS=\$IPADDRS
    -e BENCH_INPUT_LEN=\$BENCH_INPUT_LEN
    -e BENCH_OUTPUT_LEN=\$BENCH_OUTPUT_LEN
    -e BENCH_RANDOM_RANGE_RATIO=\$BENCH_RANDOM_RANGE_RATIO
    -e BENCH_NUM_PROMPTS_MULTIPLIER=\$BENCH_NUM_PROMPTS_MULTIPLIER
    -e BENCH_MAX_CONCURRENCY=\$BENCH_MAX_CONCURRENCY
    -e TQDM_MININTERVAL=\$TQDM_MININTERVAL
    -e DRY_RUN=\$DRY_RUN
    -e BENCHMARK_LOGS_DIR=/benchmark_logs
    -e ENGINE=\$ENGINE
    -e WS_PATH=${WS_PATH}
    -e RUN_EVAL=\$RUN_EVAL
    -e EVAL_ONLY=\$EVAL_ONLY
    -e EVAL_CONC=\$EVAL_CONC
    -e FRAMEWORK=\$FRAMEWORK
    -e PRECISION=\$PRECISION
    -e MODEL_PREFIX=\$MODEL_PREFIX
    -e RUNNER_TYPE=\$RUNNER_TYPE
    -e RESULT_FILENAME=\$RESULT_FILENAME
    -e SPEC_DECODING=\$SPEC_DECODING
    -e PREFILL_TP_SIZE=\$PREFILL_TP_SIZE
    -e PREFILL_ENABLE_EP=\$PREFILL_ENABLE_EP
    -e PREFILL_ENABLE_DP=\$PREFILL_ENABLE_DP
    -e DECODE_TP_SIZE=\$DECODE_TP_SIZE
    -e DECODE_ENABLE_EP=\$DECODE_ENABLE_EP
    -e DECODE_ENABLE_DP=\$DECODE_ENABLE_DP
    -e DECODE_MTP_SIZE=\$DECODE_MTP_SIZE
)

# Engine-specific env vars
if [[ "$ENGINE" == "vllm-disagg" ]]; then
    DOCKER_ENV_ENGINE=(
        -e VLLM_WS_PATH=${WS_PATH}
        -e MODEL_PATH=$DOCKER_MODEL_PATH
        -e UCX_TLS=tcp,self,shm,rocm_ipc,rocm_copy,cma
        -e UCX_SOCKADDR_TLS_PRIORITY=tcp
        -e UCX_MEMTYPE_CACHE=y
        -e UCX_RNDV_SCHEME=get_zcopy
        -e UCX_RNDV_THRESH=4k
        -e UCX_ROCM_IPC_MIN_ZCOPY=0
        -e UCX_LOG_LEVEL=warn
        -e HSA_ENABLE_SDMA=1
        -e PROXY_STREAM_IDLE_TIMEOUT=\${PROXY_STREAM_IDLE_TIMEOUT:-300}
        -e VLLM_MORIIO_CONNECTOR_READ_MODE=\${VLLM_MORIIO_CONNECTOR_READ_MODE:-1}
        -e PYTHONPYCACHEPREFIX=/tmp/pycache
    )
else
    DOCKER_ENV_ENGINE=(
        -e SGLANG_WS_PATH=${WS_PATH}
    )
fi

# Engine-specific container filter for pre-clean
CONT_FILTER="name=^container_${ENGINE}_"

srun \
  --nodelist="$SELECTED_NODELIST_SRUN" \
  --kill-on-bad-exit=1 \
  --signal=TERM@30 \
  --unbuffered \
  bash -lc "
set -euo pipefail

echo \"Rank \$SLURM_PROCID on \$(hostname)\"

# Pre-clean (idempotent)
\$DOCKER_CMD ps -aq --filter \"$CONT_FILTER\" | xargs -r \$DOCKER_CMD rm -f || true
\$DOCKER_CMD ps -aq | xargs -r \$DOCKER_CMD stop || true

# Start vLLM external router container on node 0
if [[ \"$ENGINE\" == \"vllm-disagg\" && \"$ROUTER_TYPE\" == \"vllm-router\" && \"\$SLURM_PROCID\" == \"0\" ]]; then
    \$DOCKER_CMD rm -f \"$ROUTER_CONT_NAME\" 2>/dev/null || true
    \$DOCKER_CMD run -d \
        --name \"$ROUTER_CONT_NAME\" \
        --network host \
        -v /tmp:/run_logs \
        \"$VLLM_ROUTER_IMAGE\" \
        bash -lc \"mkdir -p /run_logs/slurm_job-${SLURM_JOB_ID} && exec vllm-router \
            --vllm-pd-disaggregation \
            --kv-connector moriio \
            --vllm-discovery-address 0.0.0.0:${PROXY_PING_PORT} \
            --port ${ROUTER_PORT} \
            --host 0.0.0.0 \
            --policy consistent_hash \
            --prefill-policy consistent_hash \
            --decode-policy consistent_hash \
            --log-level info 2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/vllm_router_\$(hostname).log \"
fi

exec \$DOCKER_CMD run \
    --init \
    --stop-timeout 10 \
    --device /dev/dri \
    --device /dev/kfd \
    --device /dev/infiniband \
    --device=/dev/infiniband/rdma_cm \
    --device=/dev/infiniband/uverbs0 \
    --device=/dev/infiniband/uverbs1 \
    --device=/dev/infiniband/uverbs2 \
    --device=/dev/infiniband/uverbs3 \
    --device=/dev/infiniband/uverbs4 \
    --device=/dev/infiniband/uverbs5 \
    --device=/dev/infiniband/uverbs6 \
    --device=/dev/infiniband/uverbs7 \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --network host \
    --ipc host \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    -v /sys:/sys \
    $(command -v nicctl >/dev/null 2>&1 && echo "-v $(which nicctl):/usr/sbin/nicctl") \
    -v ${MODEL_DIR}:/models \
    -v \$HOME/.ssh:/root/.ssh \
    --shm-size 128G \
    -v /tmp:/run_logs \
    -v ${BENCHMARK_LOGS_DIR}:/benchmark_logs \
    -v ${DI_REPO_DIR}:${DOCKER_MOUNT_PATH} \
    ${DOCKER_ENV_COMMON[*]} \
    ${DOCKER_ENV_ENGINE[*]} \
    --name \"$DOCKER_CONT_NAME\" \
    --entrypoint \"\" \
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

if [[ "${KEEP_CONTAINERS}" != "1" ]]; then
    srun --nodelist="$SELECTED_NODELIST_SRUN" bash -c '$DOCKER_CMD rm -f '"$DOCKER_CONT_NAME"' 2>/dev/null || true'

    # Clean up vLLM external router container on node 0
    if [[ "$ENGINE" == "vllm-disagg" && "$ROUTER_TYPE" == "vllm-router" ]]; then
        srun --nodes=1 --ntasks=1 --nodelist="$MASTER_NODE" bash -c '
            '"$DOCKER_CMD"' rm -f '"$ROUTER_CONT_NAME"' 2>/dev/null || true
        '
    fi
fi