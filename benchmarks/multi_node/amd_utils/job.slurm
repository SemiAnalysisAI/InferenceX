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
elif [[ "$ENGINE" == "atom-disagg" ]]; then
    MODELS_YAML="$(pwd)/models_atom.yaml"
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
# Resolve the models.yaml entry the same way server_sglang.sh does: agentic runs
# (IS_AGENTIC) use the '<model>-AgentX' recipe, non-agentic disaggregated runs use
# '<model>-DI'. Fall back to the bare model name if the variant key is absent.
# MODEL_NAME itself is left unchanged so env.sh/server_sglang.sh still see the base name.
if [[ "${IS_AGENTIC:-0}" == "1" || "${IS_AGENTIC:-}" == "true" ]]; then
    MODEL_YAML_KEY="${MODEL_NAME}-AgentX"
else
    MODEL_YAML_KEY="${MODEL_NAME}-DI"
fi
if ! grep -q "^${MODEL_YAML_KEY}:" "$MODELS_YAML"; then
    if grep -q "^${MODEL_NAME}:" "$MODELS_YAML"; then
        MODEL_YAML_KEY="$MODEL_NAME"
    else
        echo "Error: Model '$MODEL_YAML_KEY' (nor bare '$MODEL_NAME') not found in $MODELS_YAML"
        echo "Available models:"
        grep -E '^[A-Za-z]' "$MODELS_YAML" | sed 's/:.*$//' | sed 's/^/  - /'
        exit 1
    fi
fi
echo "Model found: $MODEL_YAML_KEY (MODEL_NAME=$MODEL_NAME, IS_AGENTIC=${IS_AGENTIC:-0})"

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

    # Extract hf_dir from models.yaml (same as vllm-disagg path above)
    SGL_DISK_DIR_NAME=$(awk '/^'"$MODEL_NAME"':/{found=1; next}
        found && /^[^ ]/{exit}
        found && /hf_dir:/{gsub(/[" ]/, "", $2); print $2; exit}' "$MODELS_YAML")
    SGL_DISK_DIR_NAME="${SGL_DISK_DIR_NAME:-$MODEL_NAME}"

    # Prefer the caller-supplied MODEL_PATH (recipe scripts set this explicitly);
    # fall back to MODEL_DIR/hf_dir then MODEL_DIR/MODEL_NAME.
    if [[ -n "${MODEL_PATH:-}" && "$MODEL_PATH" != "$MODEL_DIR" ]]; then
        # Caller already resolved the path (e.g. MODEL_PATH=/it-share/hf_cache/models--...)
        # Use it directly if it exists on all nodes, otherwise try subdirectory combos.
        if check_model_path "$MODEL_PATH" "MODEL_PATH (caller-supplied)"; then
            echo "Selected MODEL_PATH: $MODEL_PATH (caller-supplied, available on all nodes)"
        elif check_model_path "$MODEL_PATH/$SGL_DISK_DIR_NAME" "$MODEL_PATH/$SGL_DISK_DIR_NAME"; then
            MODEL_PATH="$MODEL_PATH/$SGL_DISK_DIR_NAME"
            echo "Selected MODEL_PATH: $MODEL_PATH (available on all nodes)"
        elif check_model_path "$MODEL_PATH/$MODEL_NAME" "$MODEL_PATH/$MODEL_NAME"; then
            MODEL_PATH="$MODEL_PATH/$MODEL_NAME"
            echo "Selected MODEL_PATH: $MODEL_PATH (available on all nodes)"
        else
            echo "FATAL ERROR: Model '$MODEL_NAME' not found on ALL allocated nodes in:"
            echo "  - $MODEL_PATH"
            echo "  - $MODEL_PATH/$SGL_DISK_DIR_NAME"
            echo "  - $MODEL_PATH/$MODEL_NAME"
            exit 1
        fi
    elif check_model_path "$MODEL_DIR/$SGL_DISK_DIR_NAME" "$MODEL_DIR/$SGL_DISK_DIR_NAME"; then
        MODEL_PATH="$MODEL_DIR/$SGL_DISK_DIR_NAME"
        echo "Selected MODEL_PATH: $MODEL_PATH (available on all nodes)"
    elif check_model_path "$MODEL_DIR/$MODEL_NAME" "$MODEL_DIR"; then
        MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
        echo "Selected MODEL_PATH: $MODEL_PATH (available on all nodes)"
    else
        echo "FATAL ERROR: Model '$MODEL_NAME' not found on ALL allocated nodes in:"
        echo "  - $MODEL_DIR/$SGL_DISK_DIR_NAME"
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

# Docker privilege detection — evaluated per-node since group membership varies.
# Exported as a snippet so every srun participant resolves it locally.
export DOCKER_CMD_DETECT='if docker ps &>/dev/null 2>&1; then DOCKER_CMD=docker; else DOCKER_CMD="sudo docker"; fi'

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
export IS_MULTINODE="${IS_MULTINODE:-false}"
export INFERENCEX_MOONCAKE_MAX_TRANSFER_BATCH_KEYS="${INFERENCEX_MOONCAKE_MAX_TRANSFER_BATCH_KEYS:-}"
export MC_WORKERS_PER_CTX="${MC_WORKERS_PER_CTX:-}"

SANITIZED_USER=$(echo "$USER_NAME" | tr -c 'a-zA-Z0-9_.-' '_')
export DOCKER_CONT_NAME="container_${ENGINE}_${SANITIZED_USER}_${MODEL_NAME}_${SLURM_JOB_ID}"

# vLLM external router container.
# NOTE: vllm/vllm-router only retains ~16 recent nightlies on Docker Hub; older
# dated tags are garbage-collected (manifest unknown)
VLLM_ROUTER_IMAGE="${VLLM_ROUTER_IMAGE:-vllm/vllm-router:nightly-20260629-e667ebb}"
ROUTER_CONT_NAME="router_vllm_${SANITIZED_USER}_${SLURM_JOB_ID}"
# Separate agentic benchmark-client container (see CLIENT_IMAGE handling below).
CLIENT_CONT_NAME="container_${ENGINE}_${SANITIZED_USER}_client_${SLURM_JOB_ID}"
export RUN_FILE_FULL="$WS_PATH/${RUN_FILE}"

SELECTED_NODELIST_SRUN=$(echo "$SELECTED_NODES" | paste -sd,)

cleanup() {
  echo "[${SLURM_JOB_ID}] termination received on $(hostname); cleaning up container + stale logs..."
  # Backstop: on scancel/timeout/step-hang the foreground `exec docker run`
  # client is killed before --rm can fire, so the container (and its GPU/ports/
  # :30000) leaks and the step never returns. Force-remove THIS job's container
  # on every allocated node. Scoped to $DOCKER_CONT_NAME so it never touches
  # other users' containers. (Ported from InferenceY 51ebfa88.)
  srun --nodelist="$SELECTED_NODELIST_SRUN" \
    bash -c 'eval "$DOCKER_CMD_DETECT"; $DOCKER_CMD rm -f '"$DOCKER_CONT_NAME"' 2>/dev/null || true' 2>/dev/null || true
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
    -e BENCH_REQUEST_RATE=\$BENCH_REQUEST_RATE
    -e TQDM_MININTERVAL=\$TQDM_MININTERVAL
    -e BENCHMARK_LOGS_DIR=/benchmark_logs
    -e ENGINE=\$ENGINE
    -e WS_PATH=${WS_PATH}
    -e RUN_EVAL=\$RUN_EVAL
    -e EVAL_ONLY=\$EVAL_ONLY
    -e \"EVAL_CONC=\$EVAL_CONC\"
    -e FRAMEWORK=\$FRAMEWORK
    -e PRECISION=\$PRECISION
    -e MODEL_PREFIX=\$MODEL_PREFIX
    -e RUNNER_TYPE=\$RUNNER_TYPE
    -e RESULT_FILENAME=\$RESULT_FILENAME
    -e SPEC_DECODING=\$SPEC_DECODING
    # DISAGG was never forwarded into the container at all (not even under a
    # different name), so process_agentic_result.py's env_bool("DISAGG") always
    # defaulted to false in the result JSON regardless of the actual topology.
    -e DISAGG=\${DISAGG:-false}
    -e PREFILL_TP_SIZE=\$PREFILL_TP_SIZE
    # PREFILL_TP/DECODE_TP/*_NUM_WORKERS (below, undecorated -- distinct from the
    # *_SIZE vars server_sglang.sh uses for launch args) are what
    # process_agentic_result.py's _gpu_shape() reads for multinode runs. Without
    # these the container never sees them (only *_SIZE was passed), so agentic
    # result JSONs silently recorded tp=0 / prefill_tp=0 / prefill_num_workers=0
    # for every multinode run.
    -e PREFILL_TP=\$PREFILL_TP
    -e PREFILL_NUM_WORKERS=\$PREFILL_NUM_WORKERS
    -e PREFILL_ENABLE_EP=\$PREFILL_ENABLE_EP
    -e PREFILL_ENABLE_DP=\$PREFILL_ENABLE_DP
    -e PREFILL_CONTEXT_LENGTH=\${PREFILL_CONTEXT_LENGTH:-}
    -e PREFILL_CHUNKED_PREFILL_SIZE=\${PREFILL_CHUNKED_PREFILL_SIZE:-}
    -e SGLANG_AITER_MLA_PERSIST=\${SGLANG_AITER_MLA_PERSIST:-0}
    -e DISABLE_CUSTOM_ALL_REDUCE=\${DISABLE_CUSTOM_ALL_REDUCE:-0}
    -e MAX_MODEL_LEN=\${MAX_MODEL_LEN:-}
    -e GPU_MEMORY_UTILIZATION=\${GPU_MEMORY_UTILIZATION:-}
    -e ENABLE_PREFIX_CACHING=\${ENABLE_PREFIX_CACHING:-0}
    -e DURATION=\${DURATION:-1800}
    -e IS_AGENTIC=\${IS_AGENTIC:-0}
    -e KV_OFFLOADING=\${KV_OFFLOADING:-none}
    -e KV_OFFLOAD_BACKEND=\${KV_OFFLOAD_BACKEND:-}
    -e KV_OFFLOAD_BACKEND_METADATA=\"\${KV_OFFLOAD_BACKEND_METADATA:-}\"
    -e TOTAL_CPU_DRAM_GB=\${TOTAL_CPU_DRAM_GB:-}
    -e HF_HUB_CACHE=\${HF_HUB_CACHE:-/tmp/mooncake-hf-cache}
    -e INFERENCEX_MOONCAKE_MAX_TRANSFER_BATCH_KEYS=\${INFERENCEX_MOONCAKE_MAX_TRANSFER_BATCH_KEYS:-}
    -e MC_WORKERS_PER_CTX=\${MC_WORKERS_PER_CTX:-}
    -e VLLM_PATCH_46240=\${VLLM_PATCH_46240:-}
    -e ENABLE_METRICS=\${ENABLE_METRICS:-0}
    -e PREFILL_ROUTER_POLICY=\${PREFILL_ROUTER_POLICY:-random}
    -e DECODE_ROUTER_POLICY=\${DECODE_ROUTER_POLICY:-random}
    -e MORI_IO_SQ_BACKOFF_TIMEOUT_US=\${MORI_IO_SQ_BACKOFF_TIMEOUT_US:-}
    -e MORI_IO_QP_MAX_SEND_WR=\${MORI_IO_QP_MAX_SEND_WR:-}
    -e DECODE_TP_SIZE=\$DECODE_TP_SIZE
    -e DECODE_TP=\$DECODE_TP
    -e DECODE_NUM_WORKERS=\$DECODE_NUM_WORKERS
    -e DECODE_ENABLE_EP=\$DECODE_ENABLE_EP
    -e DECODE_ENABLE_DP=\$DECODE_ENABLE_DP
    -e DECODE_MTP_SIZE=\$DECODE_MTP_SIZE
    -e IS_MULTINODE=\$IS_MULTINODE
    -e IBDEVICES=\${IBDEVICES:-}
    -e MORI_RDMA_TC=\${MORI_RDMA_TC:-}
    -e DRY_RUN=\${DRY_RUN:-0}
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
        -e PYTHONPYCACHEPREFIX=/tmp/pycache
    )
elif [[ "$ENGINE" == "atom-disagg" ]]; then
    DOCKER_ENV_ENGINE=(
        -e ATOM_WS_PATH=${WS_PATH}
        -e PREFILL_PORT=${PREFILL_PORT:-8010}
        -e DECODE_PORT=${DECODE_PORT:-8020}
        -e ROUTER_PORT=${ROUTER_PORT:-30000}
        -e HANDSHAKE_PORT=${HANDSHAKE_PORT:-6301}
        -e MEM_FRAC_STATIC=${MEM_FRAC_STATIC:-0.85}
        -e KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8}
        -e BLOCK_SIZE=${BLOCK_SIZE:-16}
        -e MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
        -e MAX_MODEL_LEN=${MAX_MODEL_LEN:-}
        -e MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-}
        -e EXTRA_SERVER_ARGS=\${EXTRA_SERVER_ARGS:-}
        -e IBDEVICES=${IBDEVICES:-}
    )
else
    DOCKER_ENV_ENGINE=(
        -e SGLANG_WS_PATH=${WS_PATH}
    )
fi

# HiCache / Mooncake settings are delivered via a bind-mounted config file rather
# than a long list of docker -e flags. Write it once to the shared benchmark-logs
# dir (already a host path, visible on every node) and mount it read-only at
# /config/hicache_mc.env, where env.sh sources it before applying its defaults.
# Empty values are preserved so env.sh's "${VAR:-default}" fallbacks still apply.
HICACHE_MC_CONFIG="${BENCHMARK_LOGS_DIR}/hicache_mc_${SLURM_JOB_ID}.env"
cat > "$HICACHE_MC_CONFIG" <<EOF
HICACHE_RATIO=${HICACHE_RATIO:-}
HICACHE_HOST_POOL_COUNT=${HICACHE_HOST_POOL_COUNT:-}
HICACHE_PAGE_SIZE=${HICACHE_PAGE_SIZE:-}
HICACHE_IO_BACKEND=${HICACHE_IO_BACKEND:-}
HICACHE_MEM_LAYOUT=${HICACHE_MEM_LAYOUT:-}
HICACHE_WRITE_POLICY=${HICACHE_WRITE_POLICY:-}
HICACHE_STORAGE_BACKEND=${HICACHE_STORAGE_BACKEND:-}
HICACHE_PREFETCH_POLICY=${HICACHE_PREFETCH_POLICY:-}
MC_PATCH_HOSTPOOL=${MC_PATCH_HOSTPOOL:-}
MC_MASTER_PORT=${MC_MASTER_PORT:-}
MC_METADATA_PORT=${MC_METADATA_PORT:-}
MC_METRICS_PORT=${MC_METRICS_PORT:-}
MC_MASTER_THREADS=${MC_MASTER_THREADS:-}
MC_EVICTION_HIGH_WATERMARK=${MC_EVICTION_HIGH_WATERMARK:-}
MC_PROTOCOL=${MC_PROTOCOL:-}
MC_GLOBAL_SEG=${MC_GLOBAL_SEG:-}
MC_DEVICE=${MC_DEVICE:-}
MC_MASTER_ADDR=${MC_MASTER_ADDR:-}
MC_METADATA_SERVER=${MC_METADATA_SERVER:-}
EOF
echo "[config] wrote HiCache/Mooncake settings -> $HICACHE_MC_CONFIG"

# Engine-specific container filter for pre-clean
CONT_FILTER="name=^container_${ENGINE}_"

# =============================================================================
# Optional: separate benchmark-client image (agentic runs) — node-0 sibling
# =============================================================================
# When CLIENT_IMAGE is set, node 0 runs the aiperf trace replay in its own
# sibling container built from CLIENT_IMAGE (which ships a pre-baked aiperf +
# deps), instead of rebuilding the aiperf venv inside the server container every
# run. Give the server container access to the host docker socket + CLI and the
# host paths the sibling needs for its bind mounts. These fragments are expanded
# at submit time and injected into the server `docker run` below; empty (no-op)
# when CLIENT_IMAGE is unset, so the in-container aiperf path is unchanged.
CLIENT_DOCKER_MOUNTS=""
CLIENT_DOCKER_ENV=""
if [[ -n "${CLIENT_IMAGE:-}" ]]; then
    HOST_DOCKER_BIN="$(command -v docker || echo /usr/bin/docker)"
    CLIENT_DOCKER_MOUNTS="-v /var/run/docker.sock:/var/run/docker.sock -v ${HOST_DOCKER_BIN}:/usr/bin/docker"
    CLIENT_DOCKER_ENV="-e CLIENT_IMAGE=${CLIENT_IMAGE} -e HOST_REPO_DIR=${DI_REPO_DIR} -e HOST_MODEL_DIR=${MODEL_DIR} -e HOST_BENCH_LOGS=${BENCHMARK_LOGS_DIR} -e CLIENT_CONT_NAME=${CLIENT_CONT_NAME}"
    echo "[client] node-0 sibling benchmark-client image enabled: ${CLIENT_IMAGE}"
    # Best-effort pre-pull on all nodes so node 0's sibling launch doesn't stall.
    srun --nodelist="$SELECTED_NODELIST_SRUN" bash -c 'eval "$DOCKER_CMD_DETECT"; $DOCKER_CMD pull '"$CLIENT_IMAGE"' >/dev/null 2>&1 || true' 2>/dev/null || true
fi

srun \
  --nodelist="$SELECTED_NODELIST_SRUN" \
  --kill-on-bad-exit=1 \
  --signal=TERM@30 \
  --unbuffered \
  bash -lc "
set -euo pipefail

echo \"Rank \$SLURM_PROCID on \$(hostname)\"

# Per-node docker privilege detection
eval \"\$DOCKER_CMD_DETECT\"
echo \"[docker-detect] rank \$SLURM_PROCID: DOCKER_CMD=\$DOCKER_CMD\"

# Enable out-of-tree RDMA library mounts on MI355X ionic/bnxt nodes. Required for
# atom-disagg (mooncake) and vllm-disagg (MoRIIO MultiConnector); container bundled
# libibverbs/libionic may be ABI-incompatible with the host kernel drivers.
RDMA_MOUNTS=()
if [[ "$ENGINE" == "atom-disagg" || "$ENGINE" == "vllm-disagg" ]]; then

# When the container base OS differs from the host (e.g. Ubuntu 24.04 image
# on a 22.04 host), the container's bundled libibverbs/libionic may be
# ABI-incompatible with the host kernel drivers.  Detect the NIC type and
# bind-mount the host's out-of-tree RDMA userspace libraries into the
# container so the RDMA stack always matches the running kernel.
_detect_nic_type() {
    if [[ -n \"\${MORI_NIC_TYPE:-}\" ]]; then echo \"\$MORI_NIC_TYPE\"; return; fi
    local bnxt=0 mlx5=0 ionic=0
    if [[ -d /sys/class/infiniband ]]; then
        for dev in /sys/class/infiniband/*; do
            local name; name=\$(basename \"\$dev\")
            case \"\$name\" in
                bnxt_re*) ((bnxt++)) ;; mlx5*) ((mlx5++)) ;; ionic*) ((ionic++)) ;;
                *)
                    local drv; drv=\$(basename \"\$(readlink -f \"\$dev/device/driver\" 2>/dev/null)\" 2>/dev/null || true)
                    case \"\$drv\" in bnxt*) ((bnxt++)) ;; mlx5*) ((mlx5++)) ;; ionic*) ((ionic++)) ;; esac ;;
            esac
        done
    fi
    if (( bnxt >= mlx5 && bnxt >= ionic && bnxt > 0 )); then echo bnxt
    elif (( ionic >= mlx5 && ionic > 0 )); then echo ionic
    else echo mlx5; fi
}

_find_host_ibverbs() {
    for c in /usr/lib64/libibverbs.so.1 /lib/x86_64-linux-gnu/libibverbs.so.1 /usr/lib/x86_64-linux-gnu/libibverbs.so.1.14.39.0 /usr/lib/x86_64-linux-gnu/libibverbs.so.1; do
        local r; r=\$(readlink -f \"\$c\" 2>/dev/null || true)
        [[ \"\$r\" == *libibverbs.so.1.14.57.0 ]] && continue
        if [[ -f \"\$r\" ]]; then echo \"\$r\"; return; fi
    done
}

_NIC_TYPE=\$(_detect_nic_type)
echo \"[rdma] NIC type: \${_NIC_TYPE} on \$(hostname)\"

if [[ \"\$_NIC_TYPE\" == \"ionic\" || \"\$_NIC_TYPE\" == \"bnxt\" ]]; then
    _host_ibv=\$(_find_host_ibverbs)
    if [[ -n \"\$_host_ibv\" ]]; then
        RDMA_MOUNTS+=(-v \"\$_host_ibv:/lib/x86_64-linux-gnu/libibverbs.so.1\")
    fi
fi

if [[ \"\$_NIC_TYPE\" == \"ionic\" ]]; then
    for _dir in /usr/local/lib /usr/lib/x86_64-linux-gnu; do
        for _lib in \"\$_dir\"/libionic*.so; do
            [[ -f \"\$_lib\" ]] || continue
            _real=\$(readlink -f \"\$_lib\")
            [[ -f \"\$_real\" ]] && RDMA_MOUNTS+=(-v \"\$_real:\$_real\")
            RDMA_MOUNTS+=(-v \"\$_lib:/usr/lib/x86_64-linux-gnu/\$(basename \"\$_lib\")\")
        done
    done
    if [[ -d /usr/lib/x86_64-linux-gnu/libibverbs ]]; then
        for _lib in /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav*.so; do
            [[ -f \"\$_lib\" ]] && RDMA_MOUNTS+=(-v \"\$_lib:\$_lib\")
        done
    fi
    [[ -d /etc/libibverbs.d ]] && RDMA_MOUNTS+=(-v /etc/libibverbs.d:/etc/libibverbs.d:ro)
elif [[ \"\$_NIC_TYPE\" == \"bnxt\" ]]; then
    for _lib in /usr/local/lib/libbnxt_re-rdmav*.so; do
        [[ -f \"\$_lib\" ]] && RDMA_MOUNTS+=(-v \"\$_lib:/usr/lib/x86_64-linux-gnu/libibverbs/\$(basename \"\$_lib\")\")
    done
    for _lib in /usr/local/lib/libbnxt_re.so; do
        [[ -f \"\$_lib\" ]] && RDMA_MOUNTS+=(-v \"\$_lib:/usr/lib/x86_64-linux-gnu/\$(basename \"\$_lib\")\")
    done
    [[ -d /etc/libibverbs.d ]] && RDMA_MOUNTS+=(-v /etc/libibverbs.d:/etc/libibverbs.d:ro)
fi

if [[ \${#RDMA_MOUNTS[@]} -gt 0 ]]; then
    echo \"[rdma] bind-mounts: \${RDMA_MOUNTS[*]}\"
else
    echo \"[rdma] no out-of-tree RDMA mounts needed\"
fi
fi  # end: if ENGINE needs host RDMA bind-mounts

# Pre-clean (idempotent)
\$DOCKER_CMD ps -aq --filter \"$CONT_FILTER\" | xargs -r \$DOCKER_CMD rm -f || true
\$DOCKER_CMD ps -aq | xargs -r \$DOCKER_CMD stop || true

# Start vLLM external router container on node 0
if [[ \"$ENGINE\" == \"vllm-disagg\" && \"$ROUTER_TYPE\" == \"vllm-router\" && \"\$SLURM_PROCID\" == \"0\" ]]; then
    \$DOCKER_CMD rm -f \"$ROUTER_CONT_NAME\" 2>/dev/null || true
    \$DOCKER_CMD run -d \
        --name \"$ROUTER_CONT_NAME\" \
        --network host \
        --ulimit nofile=1048576:1048576 \
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

# Skip exec on vllm-disagg rank 0 so we can stop the router after the main
# container exits.  Without this, decode nodes block forever waiting for the
# router port to close (the router is a separate container).
MAYBE_EXEC=exec
if [[ \"$ENGINE\" == \"vllm-disagg\" && \"$ROUTER_TYPE\" == \"vllm-router\" && \"\$SLURM_PROCID\" == \"0\" ]]; then
    MAYBE_EXEC=
    set +e
fi

\$MAYBE_EXEC \$DOCKER_CMD run \
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
    --ulimit nofile=1048576:1048576 \
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
    -v ${HICACHE_MC_CONFIG}:/config/hicache_mc.env:ro \
    ${EXTRA_DOCKER_MOUNTS:-} \
    ${CLIENT_DOCKER_MOUNTS} \
    \${RDMA_MOUNTS[@]+"\${RDMA_MOUNTS[@]}"} \
    ${DOCKER_ENV_COMMON[*]} \
    ${DOCKER_ENV_ENGINE[*]} \
    ${CLIENT_DOCKER_ENV} \
    --name \"$DOCKER_CONT_NAME\" \
    --entrypoint \"\" \
    \"$DOCKER_IMAGE_NAME\" bash -lc '
        set -o pipefail
        mkdir -p /run_logs/slurm_job-'\"\$SLURM_JOB_ID\"'
        '"$RUN_FILE_FULL"' 2>&1 | tee /run_logs/slurm_job-'\"\$SLURM_JOB_ID\"'/server_\$(hostname).log
    '

# Only reached when exec was skipped (vllm-disagg rank 0)
DOCKER_EXIT_CODE=\$?
echo \"[rank 0] Main container exited (rc=\$DOCKER_EXIT_CODE). Stopping vllm-router...\"
\$DOCKER_CMD rm -f \"$ROUTER_CONT_NAME\" 2>/dev/null || true
exit \$DOCKER_EXIT_CODE
"

if [[ "${KEEP_CONTAINERS}" != "1" ]]; then
    srun --nodelist="$SELECTED_NODELIST_SRUN" bash -c 'eval "$DOCKER_CMD_DETECT"; $DOCKER_CMD rm -f '"$DOCKER_CONT_NAME"' '"$CLIENT_CONT_NAME"' 2>/dev/null || true'

    # Clean up vLLM external router container on node 0
    if [[ "$ENGINE" == "vllm-disagg" && "$ROUTER_TYPE" == "vllm-router" ]]; then
        srun --nodes=1 --ntasks=1 --nodelist="$MASTER_NODE" bash -c '
            eval "$DOCKER_CMD_DETECT"; $DOCKER_CMD rm -f '"$ROUTER_CONT_NAME"' 2>/dev/null || true
        '
    fi
fi
