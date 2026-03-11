#!/bin/bash
# vLLM Disaggregated Server Launcher with Model-Specific Configurations
# =============================================================================
#
# Node role assignment (by NODE_RANK):
#   0            -> Proxy/Router node
#   1..xP        -> Prefill nodes  (kv_producer)
#   xP+1..xP+yD -> Decode nodes   (kv_consumer)

# =============================================================================
# Environment Configuration
# =============================================================================

NODE0_ADDR="${NODE0_ADDR:-localhost}"
NODE_RANK="${NODE_RANK:-0}"
MODEL_DIR="${MODEL_DIR:-}"
MODEL_NAME="${MODEL_NAME:-}"

xP="${xP:-1}"
yD="${yD:-1}"

IPADDRS="${IPADDRS:-localhost}"

# Benchmark Configuration
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-1024}"
BENCH_RANDOM_RANGE_RATIO="${BENCH_RANDOM_RANGE_RATIO:-1}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-inf}"
BENCH_NUM_PROMPTS_MULTIPLIER="${BENCH_NUM_PROMPTS_MULTIPLIER:-10}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-512}"

DRY_RUN="${DRY_RUN:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

ROUTER_PORT="${ROUTER_PORT:-2584}"
SERVER_PORT="${SERVER_PORT:-2584}"
ENGINE_ID="${ENGINE_ID:-${MODEL_NAME}-pd-run}"

# Prefer MODEL_PATH from job.slurm (handles HF cache snapshot resolution)
MODEL_PATH="${MODEL_PATH:-${MODEL_DIR}/${MODEL_NAME}}"

# =============================================================================
# Dependencies and Environment Setup
# =============================================================================
source $VLLM_WS_PATH/env.sh

host_ip=$(ip route get 1.1.1.1 2>/dev/null | awk '/src/ {print $7}')
# RDMA IP for Nixl KV transfer (prefer 192.168.x.x subnet if available)
rdma_ip=$(hostname -I | tr ' ' '\n' | grep '^192\.168\.' | head -1)
rdma_ip="${rdma_ip:-$host_ip}"
host_name=$(hostname)

echo "[INFO] Management IP (barriers/proxy): $host_ip"
echo "[INFO] RDMA IP (Nixl KV transfer): $rdma_ip"

# ---------------------------------------------------------------------------
# RDMA route setup for Pensando ionic (RoCEv2) point-to-point /31 links.
# Each benic interface has a /31 to the TOR switch. Without explicit routes,
# traffic to other nodes' RDMA IPs falls through to the management network
# (no RDMA capability). Fix: add a /24 route via the TOR gateway so RoCEv2
# stays on the ionic fabric.
# ---------------------------------------------------------------------------
if [[ "$rdma_ip" =~ ^192\.168\.([0-9]+)\.([0-9]+)$ ]]; then
    rdma_subnet="${BASH_REMATCH[1]}"
    rdma_host="${BASH_REMATCH[2]}"
    rdma_gw="192.168.${rdma_subnet}.$(( rdma_host | 1 ))"  # /31 peer = TOR switch
    rdma_iface=$(ip -o addr show | awk -v ip="$rdma_ip" '$4 ~ ip {print $2}' | head -1)
    if [[ -n "$rdma_iface" ]]; then
        ip route replace "192.168.${rdma_subnet}.0/24" via "$rdma_gw" dev "$rdma_iface" 2>/dev/null && \
            echo "[RDMA-ROUTE] Added 192.168.${rdma_subnet}.0/24 via $rdma_gw dev $rdma_iface" || \
            echo "[RDMA-ROUTE] Route add failed for 192.168.${rdma_subnet}.0/24"
    fi
fi

# Patch Nixl UCX backend: set ucx_error_handling_mode=none for shared-memory
# transport compatibility (Pensando ionic NICs don't support rdmacm, so the
# default UCP_ERR_HANDLING_MODE_PEER causes "no active messages transport" errors)
NIXL_API_FILE=$(python3 -c "import rixl._api; print(rixl._api.__file__)" 2>/dev/null)
if [[ -n "$NIXL_API_FILE" ]]; then
    if ! grep -q 'ucx_error_handling_mode' "$NIXL_API_FILE"; then
        sed -i '/init\["num_threads"\] = str(nixl_conf.num_threads)/a\                        init["ucx_error_handling_mode"] = "none"' "$NIXL_API_FILE"
        echo "[PATCH] Added ucx_error_handling_mode=none to $NIXL_API_FILE"
    else
        echo "[PATCH] ucx_error_handling_mode already set in $NIXL_API_FILE"
    fi
fi

if [[ -z "$UCX_NET_DEVICES" ]]; then
    echo "Error: UCX_NET_DEVICES is empty after env.sh detection" >&2
    exit 1
fi

# =============================================================================
# Model-Specific Configuration Maps
# =============================================================================

declare -A MODEL_PREFILL_CONFIGS=(
    ["Llama-3.1-405B-Instruct-FP8-KV"]="--tensor-parallel-size 8 --kv-cache-dtype fp8"
    ["amd-Llama-3.3-70B-Instruct-FP8-KV"]="--tensor-parallel-size 8 --max-model-len 65536 --kv-cache-dtype fp8"
    ["DeepSeek-V3"]="--tensor-parallel-size 8 --compilation-config '{\"cudagraph_mode\":\"PIECEWISE\"}' --no-enable-prefix-caching --block-size 1"
    ["DeepSeek-R1-0528"]="--tensor-parallel-size 8 --compilation-config '{\"cudagraph_mode\":\"PIECEWISE\"}' --no-enable-prefix-caching --block-size 1"
    ["gpt-oss-120b"]="--tensor-parallel-size 8"
)

declare -A MODEL_DECODE_CONFIGS=(
    ["Llama-3.1-405B-Instruct-FP8-KV"]="--tensor-parallel-size 8 --kv-cache-dtype fp8"
    ["amd-Llama-3.3-70B-Instruct-FP8-KV"]="--tensor-parallel-size 8 --max-model-len 65536 --kv-cache-dtype fp8"
    ["DeepSeek-V3"]="--tensor-parallel-size 8 --compilation-config '{\"cudagraph_mode\":\"PIECEWISE\"}' --no-enable-prefix-caching --block-size 1"
    ["DeepSeek-R1-0528"]="--tensor-parallel-size 8 --compilation-config '{\"cudagraph_mode\":\"PIECEWISE\"}' --no-enable-prefix-caching --block-size 1"
    ["gpt-oss-120b"]="--tensor-parallel-size 8"
)

declare -A MODEL_ENVS=(
    ["amd-Llama-3.3-70B-Instruct-FP8-KV"]="VLLM_USE_V1=1 VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 AMDGCN_USE_BUFFER_OPS=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_RMSNORM=1 VLLM_USE_AITER_TRITON_ROPE=1 TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 TRITON_HIP_USE_ASYNC_COPY=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 TRITON_HIP_ASYNC_FAST_SWIZZLE=1"
    ["Llama-3.1-405B-Instruct-FP8-KV"]="VLLM_USE_V1=1 VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 AMDGCN_USE_BUFFER_OPS=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_RMSNORM=1 VLLM_USE_AITER_TRITON_ROPE=1 TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 TRITON_HIP_USE_ASYNC_COPY=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 TRITON_HIP_ASYNC_FAST_SWIZZLE=1"
    ["DeepSeek-V3"]="VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_PAGED_ATTN=0 VLLM_ROCM_USE_AITER_RMSNORM=1 VLLM_USE_AITER_TRITON_SILU_MUL=0"
    ["DeepSeek-R1-0528"]="VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_PAGED_ATTN=0 VLLM_ROCM_USE_AITER_RMSNORM=1 VLLM_USE_AITER_TRITON_SILU_MUL=0"
    ["gpt-oss-120b"]="VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_TRITON_BF16_GEMM=0 VLLM_USE_AITER_UNIFIED_ATTENTION=1 VLLM_ROCM_USE_AITER_MHA=0 ROCM_TRITON_MOE_PRESHUFFLE_SCALES=0"
)

get_model_config() {
    local mode="$1"
    local model_name="$2"
    if [[ "$mode" == "prefill" ]]; then
        echo "${MODEL_PREFILL_CONFIGS[$model_name]:-"--tensor-parallel-size 8"}"
    elif [[ "$mode" == "decode" ]]; then
        echo "${MODEL_DECODE_CONFIGS[$model_name]:-"--tensor-parallel-size 8"}"
    fi
}

get_model_envs() {
    echo "${MODEL_ENVS[$1]:-""}"
}

if [[ -z "$MODEL_NAME" ]]; then
    echo "ERROR: MODEL_NAME is not set"; exit 1
fi

PREFILL_SERVER_CONFIG=$(get_model_config "prefill" "$MODEL_NAME")
DECODE_SERVER_CONFIG=$(get_model_config "decode" "$MODEL_NAME")
PREFILL_MODEL_ENVS=$(get_model_envs "$MODEL_NAME")
DECODE_MODEL_ENVS=$(get_model_envs "$MODEL_NAME")
echo "Using model-specific configuration for: $MODEL_NAME"

# =============================================================================
# Container Synchronization
# =============================================================================

echo "Waiting at the container creation barrier on $host_name"
python3 $VLLM_WS_PATH/sync.py barrier \
    --local-ip ${host_ip} \
    --local-port 5000 \
    --enable-port \
    --node-ips ${IPADDRS} \
    --node-ports 5000 \
    --wait-for-all-ports \
    --timeout 300

# =============================================================================
# ETCD Server Setup
# =============================================================================

echo "Proceeding to start etcd server on $host_name"
bash ${VLLM_WS_PATH}/start_etcd.sh > /dev/null &
etcd_pid=$!

echo "Waiting at etcd server barrier on $host_name"
python3 $VLLM_WS_PATH/sync.py barrier \
    --node-ips ${IPADDRS} \
    --node-ports 2379 \
    --wait-for-all-ports \
    --timeout 300

echo "All etcd servers are up : $host_name"
sleep 3

echo "etcd endpoint health=================="
etcdctl endpoint health 2>&1 || /usr/local/bin/etcd/etcdctl endpoint health 2>&1 || true
echo "======================================"

python3 $VLLM_WS_PATH/sync.py barrier \
    --node-ips ${IPADDRS} \
    --node-ports 2379 \
    --wait-for-all-ports \
    --timeout 300

# =============================================================================
# Cluster Topology Configuration
# =============================================================================
IFS=',' read -ra IP_ARRAY <<< "$IPADDRS"

PREFILL_ARGS=""
DECODE_ARGS=""

for ((i=1; i<=xP && i<${#IP_ARRAY[@]}; i++)); do
    PREFILL_ARGS+="${IP_ARRAY[$i]} "
done

for ((i=xP+1; i<${#IP_ARRAY[@]}; i++)); do
    DECODE_ARGS+="${IP_ARRAY[$i]} "
done

echo "Prefill node IPs: ${PREFILL_ARGS}"
echo "Decode  node IPs: ${DECODE_ARGS}"

# Common UCX/Nixl environment for prefill and decode workers
setup_ucx_env() {
    export UCX_TLS=all
    export UCX_SOCKADDR_TLS_PRIORITY=tcp
    export UCX_MEMTYPE_CACHE=y
    export UCX_RNDV_SCHEME=get_zcopy
    export UCX_RNDV_THRESH=4k
    export UCX_ROCM_IPC_MIN_ZCOPY=0
    export HSA_ENABLE_SDMA=1
    export UCX_LOG_LEVEL=info
    export VLLM_USE_V1=1
    export VLLM_SERVER_DEV_MODE=0
    export VLLM_NIXL_SIDE_CHANNEL_HOST=${host_ip}
    export VLLM_NIXL_SIDE_CHANNEL_PORT=5557
}

# =============================================================================
# Node Role Assignment and Server Launch
# =============================================================================

if [ "$NODE_RANK" -eq 0 ]; then
    echo "NODE INFO ======================================="
    echo "================================================"
    echo "Node List : ${SLURM_JOB_NODELIST}"
    echo "Node IPs  : ${IPADDRS}"
    echo "Model     : ${MODEL_NAME:-'Not specified'}"
    echo "================================================"

    echo "CLUSTER INFO ===================================="
    echo "================================================"
    echo "${host_name}:${host_ip} is Proxy Node"
    echo "Prefill servers: ${PREFILL_ARGS}"
    echo "Decode  servers: ${DECODE_ARGS}"
    echo "================================================"

    PD_IPADDRS="${IPADDRS#*,}"
    echo "Waiting for all prefill and decode servers to be up . . ."
    python3 $VLLM_WS_PATH/sync.py barrier \
        --node-ips ${PD_IPADDRS} \
        --node-ports $SERVER_PORT \
        --wait-for-all-ports \
        --timeout 1800

    echo "Congratulations!!! All prefill and decode servers are up . . ."

    echo "Starting vLLM Router..."
    [ -f /root/.cargo/env ] && source /root/.cargo/env

    PREFILL_URLS=""
    DECODE_URLS=""
    for ip in ${PREFILL_ARGS}; do
        PREFILL_URLS+="--prefill http://${ip}:${SERVER_PORT} "
    done
    for ip in ${DECODE_ARGS}; do
        DECODE_URLS+="--decode http://${ip}:${SERVER_PORT} "
    done

    ROUTER_CMD="UCX_TLS=tcp,self,shm VLLM_USE_V1=1 \
    vllm-router \
        --host 0.0.0.0 \
        --port $ROUTER_PORT \
        --vllm-pd-disaggregation \
        $PREFILL_URLS \
        $DECODE_URLS \
        --policy round_robin \
        --prefill-policy round_robin \
        --decode-policy round_robin \
        --intra-node-data-parallel-size 1 \
        --retry-max-retries 3 \
        --health-check-endpoint /health \
        --prometheus-port 29000"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $ROUTER_CMD"
    else
        ROUTER_LOG_FILE="/run_logs/slurm_job-${SLURM_JOB_ID}/vllm_router_${host_name}.log"
        set -x
        eval "$ROUTER_CMD" 2>&1 | tee "$ROUTER_LOG_FILE" &
        set +x
        proxy_pid=$!

        HEALTH_BARRIER_CMD="python3 $VLLM_WS_PATH/sync.py barrier \
            --node-ips ${NODE0_ADDR} \
            --node-ports ${ROUTER_PORT} \
            --wait-for-all-health \
            --health-endpoint /health \
            --timeout 1800"

        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "DRY RUN: $HEALTH_BARRIER_CMD"
        else
            eval "$HEALTH_BARRIER_CMD"
        fi

        echo "Router is ready for benchmarking"
    fi

    echo "Ready for benchmarking on ${host_name}:${host_ip}"
    echo "Benchmarking on ${host_name}:${host_ip}"
    cd $VLLM_WS_PATH

    export ROUTER_PORT=$ROUTER_PORT
    BENCH_CMD="bash $VLLM_WS_PATH/bench.sh ${xP} ${yD} $((GPUS_PER_NODE*xP)) $((GPUS_PER_NODE*yD)) \
        $MODEL_DIR $MODEL_NAME /run_logs/slurm_job-${SLURM_JOB_ID} ${BENCH_INPUT_LEN} \
        ${BENCH_OUTPUT_LEN} \"${BENCH_MAX_CONCURRENCY}\" ${BENCH_REQUEST_RATE} \
        ${BENCH_RANDOM_RANGE_RATIO} ${BENCH_NUM_PROMPTS_MULTIPLIER}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BENCH_CMD"
    else
        set -x
        eval "$BENCH_CMD"
        set +x
    fi

    # Copy benchmark results to BENCHMARK_LOGS_DIR (mounted from host)
    LOGS_OUTPUT="${BENCHMARK_LOGS_DIR:-/run_logs}/logs"
    mkdir -p "$LOGS_OUTPUT"

    if [[ "$DRY_RUN" -eq 0 ]]; then
        cp -r /run_logs/slurm_job-${SLURM_JOB_ID} "$LOGS_OUTPUT/"
        echo "Copied results to $LOGS_OUTPUT/slurm_job-${SLURM_JOB_ID}"
    fi

    echo "Killing the proxy server"
    [[ "$DRY_RUN" -eq 0 ]] && kill $proxy_pid

elif [ "$NODE_RANK" -gt 0 ] && [ "$NODE_RANK" -le "$xP" ]; then
    echo "${host_name}:${host_ip} is Prefill Node (Model: ${MODEL_NAME})"
    echo "Using prefill config: $PREFILL_SERVER_CONFIG"

    setup_ucx_env
    for env_pair in ${PREFILL_MODEL_ENVS}; do
        export "$env_pair"
    done

    PREFILL_CMD="vllm serve ${MODEL_PATH} \
        --port $SERVER_PORT \
        --trust-remote-code \
        --disable-log-requests \
        --kv-transfer-config '{\"kv_connector\": \"NixlConnector\", \"engine_id\": \"${ENGINE_ID}\", \"kv_role\": \"kv_producer\", \"kv_parallel_size\": 8, \"kv_rank\": 0, \"kv_buffer_size\": 5000000000, \"kv_buffer_device\": \"cuda\", \"kv_ip\": \"'\"${rdma_ip}\"'\", \"kv_port\": 14600}' \
        ${PREFILL_SERVER_CONFIG}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $PREFILL_CMD"
    else
        set -x
        eval "$PREFILL_CMD" \
            2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/prefill_${host_name}.log &
        set +x
        prefill_pid=$!
    fi

    echo "Waiting for proxy server to be up..."
    BARRIER_CMD="python3 $VLLM_WS_PATH/sync.py barrier \
        --node-ips ${NODE0_ADDR} \
        --node-ports ${ROUTER_PORT} \
        --wait-for-all-ports \
        --timeout 1800"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BARRIER_CMD"
    else
        eval "$BARRIER_CMD"
    fi

    echo "Waiting until proxy server closes..."
    WAIT_CMD="python3 $VLLM_WS_PATH/sync.py wait \
        --remote-ip ${NODE0_ADDR} \
        --remote-port ${ROUTER_PORT}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $WAIT_CMD"
    else
        eval "$WAIT_CMD"
    fi

    echo "Killing the prefill server"
    [[ "$DRY_RUN" -eq 0 ]] && kill $prefill_pid

else
    echo "${host_name}:${host_ip} is Decode Node (Model: ${MODEL_NAME})"
    echo "Using decode config: $DECODE_SERVER_CONFIG"

    setup_ucx_env
    for env_pair in ${DECODE_MODEL_ENVS}; do
        export "$env_pair"
    done

    DECODE_CMD="vllm serve ${MODEL_PATH} \
        --port $SERVER_PORT \
        --trust-remote-code \
        --disable-log-requests \
        --kv-transfer-config '{\"kv_connector\": \"NixlConnector\", \"engine_id\": \"${ENGINE_ID}\", \"kv_role\": \"kv_consumer\", \"kv_parallel_size\": 8, \"kv_rank\": 0, \"kv_buffer_size\": 5000000000, \"kv_buffer_device\": \"cuda\", \"kv_ip\": \"'\"${rdma_ip}\"'\", \"kv_port\": 14600}' \
        ${DECODE_SERVER_CONFIG}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $DECODE_CMD"
    else
        set -x
        eval "$DECODE_CMD" \
            2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/decode_${host_name}.log &
        set +x
        decode_pid=$!
    fi

    echo "Waiting for proxy server to be up..."
    BARRIER_CMD="python3 $VLLM_WS_PATH/sync.py barrier \
        --node-ips ${NODE0_ADDR} \
        --node-ports ${ROUTER_PORT} \
        --wait-for-all-ports \
        --timeout 1800"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BARRIER_CMD"
    else
        eval "$BARRIER_CMD"
    fi

    echo "Waiting until proxy server closes..."
    WAIT_CMD="python3 $VLLM_WS_PATH/sync.py wait \
        --remote-ip ${NODE0_ADDR} \
        --remote-port ${ROUTER_PORT}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $WAIT_CMD"
    else
        eval "$WAIT_CMD"
    fi

    echo "Killing the decode server"
    [[ "$DRY_RUN" -eq 0 ]] && kill $decode_pid
fi

echo "Killing the etcd server"
kill $etcd_pid

echo "Script completed successfully"
exit 0
