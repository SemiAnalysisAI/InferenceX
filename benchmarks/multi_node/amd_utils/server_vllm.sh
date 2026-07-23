#!/bin/bash
# vLLM Disaggregated Server Launcher with Model-Specific Configurations
# =============================================================================
#
# Node role assignment (by NODE_RANK):
#   0           -> Proxy/Router + first Prefill node  (kv_producer)
#   1..xP-1     -> Additional Prefill nodes            (kv_producer)
#   xP..xP+yD-1 -> Decode nodes                        (kv_consumer)
#
# Total nodes = xP + yD (router co-located with first prefill, like SGLang).

# =============================================================================
# Dependency Setup (idempotent; required when using base vLLM image)
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/setup_deps.sh"

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

PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-$GPUS_PER_NODE}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-$GPUS_PER_NODE}"

ROUTER_PORT="${ROUTER_PORT:-30000}"
SERVER_PORT="${SERVER_PORT:-2584}"
ENGINE_ID="${ENGINE_ID:-${MODEL_NAME}-pd-run}"

# Prefer MODEL_PATH from job.slurm (handles HF cache snapshot resolution)
MODEL_PATH="${MODEL_PATH:-${MODEL_DIR}/${MODEL_NAME}}"

# =============================================================================
# Dependencies and Environment Setup
# =============================================================================
source $WS_PATH/env.sh

host_ip=$(ip route get 1.1.1.1 2>/dev/null | awk '/src/ {print $7}')
# RDMA IP for Nixl KV transfer (prefer 192.168.x.x subnet if available)
rdma_ip=$(hostname -I | tr ' ' '\n' | grep '^192\.168\.' | head -1)
rdma_ip="${rdma_ip:-$host_ip}"
host_name=$(hostname)

echo "[INFO] Management IP (barriers/proxy): $host_ip"
echo "[INFO] RDMA IP (Nixl KV transfer): $rdma_ip"

# =============================================================================
# RDMA / Nixl Workarounds
# =============================================================================

setup_rdma_env() {
    # Pensando ionic (RoCEv2) point-to-point /31 route fix.
    # Each benic interface has a /31 to the TOR switch. Without explicit routes,
    # traffic to other nodes' RDMA IPs falls through to the management network.
    if [[ "$rdma_ip" =~ ^192\.168\.([0-9]+)\.([0-9]+)$ ]]; then
        local rdma_subnet="${BASH_REMATCH[1]}"
        local rdma_host="${BASH_REMATCH[2]}"
        local rdma_gw="192.168.${rdma_subnet}.$(( rdma_host | 1 ))"
        local rdma_iface
        rdma_iface=$(ip -o addr show | awk -v ip="$rdma_ip" '$4 ~ ip {print $2}' | head -1)
        if [[ -n "$rdma_iface" ]]; then
            ip route replace "192.168.${rdma_subnet}.0/24" via "$rdma_gw" dev "$rdma_iface" 2>/dev/null && \
                echo "[RDMA-ROUTE] Added 192.168.${rdma_subnet}.0/24 via $rdma_gw dev $rdma_iface" || \
                echo "[RDMA-ROUTE] Route add failed for 192.168.${rdma_subnet}.0/24"
        fi
    fi

    # Patch Nixl UCX backend: set ucx_error_handling_mode=none.
    # Required for ALL NIC types under high concurrency (C512+). Without this,
    # UCX's default UCP_ERR_HANDLING_MODE_PEER triggers transport-level error
    # recovery on ibv_post_send failures, preventing RIXL RDMA READ retries from
    # recovering gracefully. This causes the prefill KV cache to fill to 100%
    # and deadlock the pipeline. On ionic NICs this was already applied (rdmacm
    # incompatibility); on mlx5 NICs it was incorrectly skipped.
    local nixl_api
    nixl_api=$(python3 -c "import rixl._api; print(rixl._api.__file__)" 2>/dev/null)
    if [[ -n "$nixl_api" ]]; then
        if ! grep -q 'ucx_error_handling_mode' "$nixl_api"; then
            sed -i '/self\.create_backend(bknd, init)/i\                init["ucx_error_handling_mode"] = "none"' "$nixl_api"
            echo "[PATCH] Added ucx_error_handling_mode=none to $nixl_api (IBDEVICES=${IBDEVICES:-unset})"
        else
            echo "[PATCH] ucx_error_handling_mode already set in $nixl_api"
        fi
    fi
}

setup_rdma_env

if [[ -z "$UCX_NET_DEVICES" ]]; then
    echo "Error: UCX_NET_DEVICES is empty after env.sh detection" >&2
    exit 1
fi

# =============================================================================
# Model-Specific Configuration from YAML
# =============================================================================
MODELS_YAML="${WS_PATH}/models_vllm.yaml"

if [[ ! -f "$MODELS_YAML" ]]; then
    echo "ERROR: models.yaml not found at $MODELS_YAML"
    exit 1
fi

if [[ -z "$MODEL_NAME" ]]; then
    echo "ERROR: MODEL_NAME is not set"; exit 1
fi

eval "$(python3 -c "
import yaml, sys

with open('${MODELS_YAML}') as f:
    models = yaml.safe_load(f)

model_name = '${MODEL_NAME}'
if model_name not in models:
    print(f'echo \"ERROR: Model {model_name} not in models.yaml\"; exit 1')
    sys.exit(0)

m = models[model_name]

def bash_escape(s):
    \"\"\"Escape a value for safe embedding in a bash double-quoted assignment.\"\"\"
    return s.replace('\\\\', '\\\\\\\\').replace('\"', '\\\\\"').replace('\$', '\\\\\$').replace('\`', '\\\\\`')

pf = bash_escape(m.get('prefill_flags', '--tensor-parallel-size 8'))
df = bash_escape(m.get('decode_flags', '--tensor-parallel-size 8'))
ev = bash_escape(m.get('env', ''))
dev = bash_escape(m.get('decode_env', ''))
print(f'PREFILL_SERVER_CONFIG=\"{pf}\"')
print(f'DECODE_SERVER_CONFIG=\"{df}\"')
print(f'MODEL_ENVS=\"{ev}\"')
print(f'DECODE_MODEL_ENVS=\"{dev}\"')
")"

echo "Loaded model configuration for: $MODEL_NAME"

# Apply tensor-parallel size and EP/DP flags from submit pipeline.
if [[ -n "${PREFILL_TP_SIZE:-}" ]]; then
    if echo "$PREFILL_SERVER_CONFIG" | grep -q -- '--tensor-parallel-size'; then
        PREFILL_SERVER_CONFIG=$(echo "$PREFILL_SERVER_CONFIG" | sed -E "s/--tensor-parallel-size[[:space:]]+[0-9]+/--tensor-parallel-size ${PREFILL_TP_SIZE}/g")
    else
        PREFILL_SERVER_CONFIG+=" --tensor-parallel-size ${PREFILL_TP_SIZE}"
    fi
fi
if [[ -n "${DECODE_TP_SIZE:-}" ]]; then
    if echo "$DECODE_SERVER_CONFIG" | grep -q -- '--tensor-parallel-size'; then
        DECODE_SERVER_CONFIG=$(echo "$DECODE_SERVER_CONFIG" | sed -E "s/--tensor-parallel-size[[:space:]]+[0-9]+/--tensor-parallel-size ${DECODE_TP_SIZE}/g")
    else
        DECODE_SERVER_CONFIG+=" --tensor-parallel-size ${DECODE_TP_SIZE}"
    fi
fi
if [[ "${PREFILL_ENABLE_EP:-false}" == "true" ]] && ! echo "$PREFILL_SERVER_CONFIG" | grep -q -- '--enable-expert-parallel'; then
    PREFILL_SERVER_CONFIG+=" --enable-expert-parallel"
fi
if [[ "${DECODE_ENABLE_EP:-false}" == "true" ]] && ! echo "$DECODE_SERVER_CONFIG" | grep -q -- '--enable-expert-parallel'; then
    DECODE_SERVER_CONFIG+=" --enable-expert-parallel"
fi

# DEP8 on ROCm vLLM (mori-0625): TP1 + data-parallel-size + EP, not --enable-dp-attention
# (same as benchmarks/single_node/fixed_seq_len/minimaxm3_fp4_mi355x_vllm.sh).
apply_vllm_dp_config() {
    local cfg="$1"
    local tp_size="$2"
    local enable_dp="${3:-false}"

    cfg=$(echo "$cfg" | sed -E 's/[[:space:]]*--enable-dp-attention//g')
    cfg=$(echo "$cfg" | sed -E 's/[[:space:]]*--data-parallel-size[[:space:]]+[0-9]+//g')

    if [[ "$enable_dp" != "true" ]]; then
        echo "$cfg"
        return
    fi

    if echo "$cfg" | grep -q -- '--tensor-parallel-size'; then
        echo "$cfg" | sed -E "s/--tensor-parallel-size[[:space:]]+[0-9]+/--tensor-parallel-size 1 --data-parallel-size ${tp_size}/"
    else
        echo "$cfg --tensor-parallel-size 1 --data-parallel-size ${tp_size}"
    fi
}

PREFILL_SERVER_CONFIG="$(apply_vllm_dp_config "$PREFILL_SERVER_CONFIG" "${PREFILL_TP_SIZE:-8}" "${PREFILL_ENABLE_DP:-false}")"
DECODE_SERVER_CONFIG="$(apply_vllm_dp_config "$DECODE_SERVER_CONFIG" "${DECODE_TP_SIZE:-8}" "${DECODE_ENABLE_DP:-false}")"

apply_gpu_memory_utilization() {
    local cfg="$1"
    local gmu="${GPU_MEMORY_UTILIZATION:-}"
    if [[ -z "$gmu" ]]; then
        echo "$cfg"
        return
    fi
    if echo "$cfg" | grep -q -- '--gpu-memory-utilization'; then
        echo "$cfg" | sed -E "s/--gpu-memory-utilization[[:space:]]+[0-9.]+/--gpu-memory-utilization ${gmu}/g"
    else
        echo "$cfg --gpu-memory-utilization ${gmu}"
    fi
}

if [[ -n "${GPU_MEMORY_UTILIZATION:-}" ]]; then
    PREFILL_SERVER_CONFIG="$(apply_gpu_memory_utilization "$PREFILL_SERVER_CONFIG")"
    DECODE_SERVER_CONFIG="$(apply_gpu_memory_utilization "$DECODE_SERVER_CONFIG")"
    echo "Applied GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
fi

echo "PREFILL_SERVER_CONFIG (after TP/EP/DP): $PREFILL_SERVER_CONFIG"
echo "DECODE_SERVER_CONFIG (after TP/EP/DP): $DECODE_SERVER_CONFIG"

apply_max_model_len() {
    local cfg="$1"
    if [[ -n "${MAX_MODEL_LEN:-}" && "${MAX_MODEL_LEN}" != "0" ]]; then
        if echo "$cfg" | grep -q -- '--max-model-len'; then
            echo "$cfg" | sed -E "s/--max-model-len[[:space:]]+[0-9]+/--max-model-len ${MAX_MODEL_LEN}/g"
        else
            echo "$cfg --max-model-len ${MAX_MODEL_LEN}"
        fi
    else
        echo "$cfg"
    fi
}

enable_prefix_caching=false
if [[ "${IS_AGENTIC:-0}" == "1" || "${IS_AGENTIC:-}" == "true" ]]; then
    enable_prefix_caching=true
fi
if [[ "${ENABLE_PREFIX_CACHING:-0}" == "1" || "${ENABLE_PREFIX_CACHING:-}" == "true" ]]; then
    enable_prefix_caching=true
fi
if [[ "$enable_prefix_caching" == "true" ]]; then
    PREFILL_SERVER_CONFIG="${PREFILL_SERVER_CONFIG//--no-enable-prefix-caching/--enable-prefix-caching}"
    DECODE_SERVER_CONFIG="${DECODE_SERVER_CONFIG//--no-enable-prefix-caching/--enable-prefix-caching}"
fi
if [[ -n "${MAX_MODEL_LEN:-}" && "${MAX_MODEL_LEN}" != "0" ]]; then
    PREFILL_SERVER_CONFIG="$(apply_max_model_len "$PREFILL_SERVER_CONFIG")"
    DECODE_SERVER_CONFIG="$(apply_max_model_len "$DECODE_SERVER_CONFIG")"
    echo "Applied MAX_MODEL_LEN=${MAX_MODEL_LEN}"
fi
if [[ "$enable_prefix_caching" == "true" || -n "${MAX_MODEL_LEN:-}" ]]; then
    echo "PREFILL_SERVER_CONFIG (overrides): $PREFILL_SERVER_CONFIG"
    echo "DECODE_SERVER_CONFIG (overrides): $DECODE_SERVER_CONFIG"
fi

install_mooncake_rocm() {
    local mooncake_tag="v0.3.11.post1"
    local mooncake_src="/tmp/Mooncake-$mooncake_tag"
    local mooncake_stage="/tmp/mooncake-stage-$mooncake_tag"
    local build_jobs cache_root cache_key cache_archive cache_tmp engine_path
    local os_version python_abi rocm_version

    build_jobs=$(nproc)
    if ((build_jobs > 32)); then
        build_jobs=32
    fi

    os_version=$(. /etc/os-release && printf '%s-%s' "$ID" "$VERSION_ID")
    python_abi=$(python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
    rocm_version=$(sed -n '1p' /opt/rocm/.info/version 2>/dev/null || true)
    if [[ -z "$rocm_version" ]]; then
        rocm_version=$(hipconfig --version)
    fi
    rocm_version=${rocm_version//[^[:alnum:]._-]/_}
    local hf_hub_cache="${HF_HUB_CACHE:-${MODEL_DIR}/.cache/huggingface/hub}"
    cache_root="${hf_hub_cache}/inferencex/mooncake"
    cache_key="${mooncake_tag}-${os_version}-${python_abi}-${rocm_version}-$(uname -m)-hip"
    cache_archive="$cache_root/$cache_key.tar.gz"
    mkdir -p "$cache_root"

    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake git libasio-dev libboost-dev libcurl4-openssl-dev \
        libgflags-dev libgoogle-glog-dev libibverbs-dev libjsoncpp-dev \
        libnuma-dev libpython3-dev libssl-dev libunwind-dev liburing-dev \
        libxxhash-dev libyaml-cpp-dev libzstd-dev ninja-build pybind11-dev

    exec 9>"$cache_archive.lock"
    flock -w 1800 9
    if [[ -f "$cache_archive" ]] && ! tar -tzf "$cache_archive" >/dev/null 2>&1; then
        rm -f "$cache_archive"
    fi
    if [[ ! -f "$cache_archive" ]]; then
        echo "[Mooncake] Building HIP cache artifact: $cache_archive"
        rm -rf "$mooncake_src" "$mooncake_stage"
        git clone --depth 1 --branch "$mooncake_tag" --recurse-submodules \
            --shallow-submodules https://github.com/kvcache-ai/Mooncake.git "$mooncake_src"
        cmake -S "$mooncake_src/extern/yalantinglibs" \
            -B "$mooncake_src/extern/yalantinglibs/build" \
            -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF
        cmake --build "$mooncake_src/extern/yalantinglibs/build" -j "$build_jobs"
        cmake --install "$mooncake_src/extern/yalantinglibs/build"
        cmake -S "$mooncake_src" -B "$mooncake_src/build" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF -DUSE_HIP=ON \
            -DWITH_EP=OFF -DWITH_STORE=ON -DWITH_STORE_RUST=OFF \
            -DWITH_RUST_EXAMPLE=OFF -DBUILD_EXAMPLES=OFF -DBUILD_UNIT_TESTS=OFF
        cmake --build "$mooncake_src/build" -j "$build_jobs"
        mkdir -p "$mooncake_stage"
        DESTDIR="$mooncake_stage" cmake --install "$mooncake_src/build"
        cache_tmp=$(mktemp "$cache_root/$cache_key.tmp.XXXXXX")
        tar -C "$mooncake_stage" -czf "$cache_tmp" .
        mv -f "$cache_tmp" "$cache_archive"
    else
        echo "[Mooncake] Using HIP cache artifact: $cache_archive"
    fi
    tar -C / -xzf "$cache_archive"
    engine_path=$(python3 -c 'import mooncake.engine; print(mooncake.engine.__file__)')
    ldd "$engine_path" | grep -q 'libamdhip64.so'
    exec 9>&-
}

# MiniMax-M3 agentic DRAM offload: per-node mooncake_master + MooncakeStoreConnector.
# MoRIIOConnector still handles P/D transfer via vLLM MultiConnector.
ensure_mooncake_kv_offload() {
    local tp_size="$1"
    if [[ "${KV_OFFLOADING:-none}" != "dram" || "${KV_OFFLOAD_BACKEND:-}" != "mooncake" ]]; then
        return 0
    fi
    if [[ -n "${MOONCAKE_SETUP_DONE:-}" ]]; then
        return 0
    fi
    if [[ ! "${TOTAL_CPU_DRAM_GB:-}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: KV_OFFLOADING=dram with KV_OFFLOAD_BACKEND=mooncake requires positive TOTAL_CPU_DRAM_GB" >&2
        exit 1
    fi
    if ! python3 -c "from mooncake.store import MooncakeDistributedStore" >/dev/null 2>&1; then
        install_mooncake_rocm
    fi
    python3 -c "from mooncake.store import MooncakeDistributedStore" >/dev/null

    local per_rank_gb=$((TOTAL_CPU_DRAM_GB / tp_size))
    MOONCAKE_MASTER_PORT=$((SERVER_PORT + 12000))
    MOONCAKE_CONFIG_PATH="/run_logs/slurm_job-${SLURM_JOB_ID}/mooncake_config_${host_name}.json"
    mkdir -p "$(dirname "$MOONCAKE_CONFIG_PATH")"
    cat > "$MOONCAKE_CONFIG_PATH" <<EOF
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:${MOONCAKE_MASTER_PORT}",
  "global_segment_size": "${per_rank_gb}GB",
  "local_buffer_size": "4GB",
  "protocol": "tcp",
  "device_name": "",
  "enable_offload": false
}
EOF
    export MOONCAKE_CONFIG_PATH PYTHONHASHSEED=0 MC_SLICE_SIZE=1048576
    export MC_TCP_ENABLE_CONNECTION_POOL=1

    local transfer_batch_keys_log="off"
    local mc_workers_log="default"
    if [[ -n "${INFERENCEX_MOONCAKE_MAX_TRANSFER_BATCH_KEYS:-}" ]]; then
        export MC_WORKERS_PER_CTX="${MC_WORKERS_PER_CTX:-4}"
        transfer_batch_keys_log="${INFERENCEX_MOONCAKE_MAX_TRANSFER_BATCH_KEYS}"
        mc_workers_log="${MC_WORKERS_PER_CTX}"

        MOONCAKE_BATCH_PATCH_SCRIPT="$(dirname "${BASH_SOURCE[0]}")/patches/apply_vllm_mooncake_transfer_batches.py"
        if [[ ! -f "$MOONCAKE_BATCH_PATCH_SCRIPT" ]]; then
            echo "ERROR: Mooncake transfer batch patch missing: $MOONCAKE_BATCH_PATCH_SCRIPT" >&2
            exit 1
        fi
        python3 "$MOONCAKE_BATCH_PATCH_SCRIPT"
    fi

    local mooncake_master_cmd="mooncake_master --port ${MOONCAKE_MASTER_PORT} --default_kv_lease_ttl=120s --eviction_high_watermark_ratio=0.80 --eviction_ratio=0.10"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $mooncake_master_cmd"
    else
        MOONCAKE_MASTER_LOG="/run_logs/slurm_job-${SLURM_JOB_ID}/mooncake_master_${host_name}.log"
        $mooncake_master_cmd > "$MOONCAKE_MASTER_LOG" 2>&1 &
        MOONCAKE_MASTER_PID=$!
        sleep 2
        kill -0 "$MOONCAKE_MASTER_PID"
    fi

    echo "Applied Mooncake DRAM KV offload on ${host_name}: TOTAL_CPU_DRAM_GB=${TOTAL_CPU_DRAM_GB} tp=${tp_size} per_rank=${per_rank_gb}GB master_port=${MOONCAKE_MASTER_PORT} transfer_batch_keys=${transfer_batch_keys_log} mc_workers_per_ctx=${mc_workers_log}"
    MOONCAKE_SETUP_DONE=1
}

build_kv_transfer_config_json() {
    local mori_role="$1"
    if [[ "${KV_OFFLOADING:-none}" == "dram" && "${KV_OFFLOAD_BACKEND:-}" == "mooncake" ]]; then
        NODE0_ADDR="$NODE0_ADDR" PROXY_PING_PORT="$PROXY_PING_PORT" SERVER_PORT="$SERVER_PORT" \
            MORI_KV_ROLE="$mori_role" python3 -c '
import json, os

mori_extra = {
    "proxy_ip": os.environ["NODE0_ADDR"],
    "proxy_ping_port": os.environ["PROXY_PING_PORT"],
    "http_port": os.environ["SERVER_PORT"],
    "read_mode": True,
}
print(json.dumps({
    "kv_connector": "MultiConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "connectors": [
            {
                "kv_connector": "MoRIIOConnector",
                "kv_role": os.environ["MORI_KV_ROLE"],
                "kv_load_failure_policy": "fail",
                "kv_connector_extra_config": mori_extra,
            },
            {
                "kv_connector": "MooncakeStoreConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "load_async": True,
                    "lookup_async": True,
                },
            },
        ],
    },
}))
'
        return
    fi

    cat <<EOF
{"kv_connector": "MoRIIOConnector", "kv_role": "${mori_role}", "kv_connector_extra_config": {"proxy_ip": "${NODE0_ADDR}", "proxy_ping_port": "${PROXY_PING_PORT}", "http_port": "${SERVER_PORT}", "read_mode": true}}
EOF
}

if [[ "${KV_OFFLOADING:-none}" == "dram" && "${KV_OFFLOAD_BACKEND:-}" == "native" ]]; then
    echo "ERROR: KV_OFFLOAD_BACKEND=native is not supported for vLLM disagg (use mooncake for MiniMax-M3 agentic DRAM offload)" >&2
    exit 1
fi

# vLLM #46240: skip stale KV xfer completions instead of assert-killing EngineCore.
# https://github.com/vllm-project/vllm/issues/46240
if [[ "${VLLM_PATCH_46240:-${KV_OFFLOADING:-none}}" == "dram" || "${VLLM_PATCH_46240:-}" == "1" ]]; then
    PATCH_SCRIPT="$(dirname "${BASH_SOURCE[0]}")/patches/apply_vllm_46240_scheduler_patch.py"
    if [[ ! -f "$PATCH_SCRIPT" ]]; then
        echo "ERROR: VLLM_PATCH_46240 enabled but missing $PATCH_SCRIPT" >&2
        exit 1
    fi
    python3 "$PATCH_SCRIPT"
fi

# =============================================================================
# Container Synchronization
# =============================================================================

echo "Waiting at the container creation barrier on $host_name"
python3 $WS_PATH/sync.py barrier \
    --local-ip ${host_ip} \
    --local-port 5000 \
    --enable-port \
    --node-ips ${IPADDRS} \
    --node-ports 5000 \
    --wait-for-all-ports \
    --timeout 600

# =============================================================================
# Cluster Topology Configuration
# =============================================================================
IFS=',' read -ra IP_ARRAY <<< "$IPADDRS"

PREFILL_ARGS=""
DECODE_ARGS=""

for ((i=0; i<xP && i<${#IP_ARRAY[@]}; i++)); do
    PREFILL_ARGS+="${IP_ARRAY[$i]} "
done

for ((i=xP; i<${#IP_ARRAY[@]}; i++)); do
    DECODE_ARGS+="${IP_ARRAY[$i]} "
done

echo "Prefill node IPs: ${PREFILL_ARGS}"
echo "Decode  node IPs: ${DECODE_ARGS}"

# Per-worker Prometheus /metrics and cache-flush base URLs for agentic replay.
# vLLM workers listen on SERVER_PORT; the vllm-router on ROUTER_PORT does not
# expose Prometheus or fan out cache resets.
SERVER_METRICS_URLS=()
SERVER_FLUSH_URLS=()
for ((i=0; i<xP && i<${#IP_ARRAY[@]}; i++)); do
    SERVER_METRICS_URLS+=("http://${IP_ARRAY[$i]}:${SERVER_PORT}/metrics")
    SERVER_FLUSH_URLS+=("http://${IP_ARRAY[$i]}:${SERVER_PORT}")
done
for ((i=0; i<yD; i++)); do
    idx=$((xP + i))
    if (( idx < ${#IP_ARRAY[@]} )); then
        SERVER_METRICS_URLS+=("http://${IP_ARRAY[$idx]}:${SERVER_PORT}/metrics")
        SERVER_FLUSH_URLS+=("http://${IP_ARRAY[$idx]}:${SERVER_PORT}")
    fi
done

# MoRI-IO proxy ZMQ registration port (must match vllm-router --vllm-discovery-address)
PROXY_PING_PORT="${PROXY_PING_PORT:-36367}"

# vLLM runtime environment (static vars moved to env.sh; these depend on per-node state)
setup_vllm_env() {
    export VLLM_NIXL_SIDE_CHANNEL_HOST=${rdma_ip}
    export VLLM_NIXL_SIDE_CHANNEL_PORT=5600
    for env_pair in ${MODEL_ENVS}; do
        export "$env_pair"
    done
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
    echo "${host_name}:${host_ip} is Proxy Node and Prefill Node"
    echo "Using prefill config: $PREFILL_SERVER_CONFIG"
    echo "Prefill servers: ${PREFILL_ARGS}"
    echo "Decode  servers: ${DECODE_ARGS}"
    echo "================================================"

    setup_vllm_env
    ensure_mooncake_kv_offload "$PREFILL_TP_SIZE"
    KV_TRANSFER_JSON=$(build_kv_transfer_config_json kv_producer)

    # Router is started as an external container by job.slurm (VLLM_ROUTER_IMAGE)
    echo "Using external vllm-router container (started by job.slurm on this node)"

    SERVED_MODEL="${MODEL_NAME}"
    PREFILL_CMD="vllm serve ${MODEL_PATH} \
        --served-model-name ${SERVED_MODEL} \
        --port $SERVER_PORT \
        --trust-remote-code \
        --kv-transfer-config '${KV_TRANSFER_JSON}' \
        ${PREFILL_SERVER_CONFIG}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $PREFILL_CMD"
    else
        PREFILL_LOG_FILE="/run_logs/slurm_job-${SLURM_JOB_ID}/prefill_${host_name}.log"
        set -x
        eval "$PREFILL_CMD" > "$PREFILL_LOG_FILE" 2>&1 &
        set +x
        prefill_pid=$!
    fi

    echo "Waiting for all prefill and decode servers to be up . . ."
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: skipping barrier (wait-for-all-ports)"
    else
        python3 $WS_PATH/sync.py barrier \
            --node-ips ${IPADDRS} \
            --node-ports $SERVER_PORT \
            --wait-for-all-ports \
            --timeout 1800
    fi

    echo "Congratulations!!! All prefill and decode servers are up . . ."

    # Wait for proxy /health to confirm it is accepting requests
    HEALTH_BARRIER_CMD="python3 $WS_PATH/sync.py barrier \
        --node-ips ${NODE0_ADDR} \
        --node-ports ${ROUTER_PORT} \
        --wait-for-all-health \
        --health-endpoint /health \
        --timeout 1800"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $HEALTH_BARRIER_CMD"
    else
        eval "$HEALTH_BARRIER_CMD"
        echo "MoRI-IO proxy is ready for benchmarking"
    fi

    echo "Ready for benchmarking on ${host_name}:${host_ip}"
    echo "Benchmarking on ${host_name}:${host_ip}"
    cd $WS_PATH

    export ROUTER_PORT=$ROUTER_PORT

    # IS_AGENTIC=1/true  → agentic trace replay (trace_replay.sh)
    # IS_AGENTIC unset/0 → fixed-seq-len throughput benchmark (bench.sh)
    if [[ "${IS_AGENTIC:-0}" == "1" || "${IS_AGENTIC:-}" == "true" ]]; then
        if [[ "${ENABLE_METRICS:-0}" == "1" && "${#SERVER_METRICS_URLS[@]}" -gt 0 ]]; then
            AIPERF_SERVER_METRICS_URLS=$(IFS=,; echo "${SERVER_METRICS_URLS[*]}")
            export AIPERF_SERVER_METRICS_URLS
            echo "AIPERF_SERVER_METRICS_URLS=${AIPERF_SERVER_METRICS_URLS}"
        fi
        if [[ "${#SERVER_FLUSH_URLS[@]}" -gt 0 ]]; then
            SERVER_FLUSH_URLS_CSV=$(IFS=,; echo "${SERVER_FLUSH_URLS[*]}")
            export SERVER_FLUSH_URLS_CSV
            echo "SERVER_FLUSH_URLS_CSV=${SERVER_FLUSH_URLS_CSV}"
        fi
        export ENGINE="${FRAMEWORK:-vllm-disagg}"
        BENCH_CMD="bash $WS_PATH/trace_replay.sh \
            $MODEL_DIR $MODEL_NAME $BENCH_MAX_CONCURRENCY /run_logs/slurm_job-${SLURM_JOB_ID}"
        echo "Benchmark runner: trace_replay.sh (agentic, KV_OFFLOADING=${KV_OFFLOADING:-none}, CONC=${BENCH_MAX_CONCURRENCY})"
    else
        BENCH_CMD="bash $WS_PATH/bench.sh ${xP} ${yD} $((PREFILL_TP_SIZE*xP)) $((DECODE_TP_SIZE*yD)) \
            $MODEL_DIR $MODEL_NAME /run_logs/slurm_job-${SLURM_JOB_ID} ${BENCH_INPUT_LEN} \
            ${BENCH_OUTPUT_LEN} \"${BENCH_MAX_CONCURRENCY}\" ${BENCH_REQUEST_RATE} \
            ${BENCH_RANDOM_RANGE_RATIO} ${BENCH_NUM_PROMPTS_MULTIPLIER}"
        echo "Benchmark runner: bench.sh (fixed-seq-len)"
    fi

    if [[ "${EVAL_ONLY:-false}" == "true" ]]; then
        echo "EVAL_ONLY mode: skipping throughput benchmark"
    elif [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BENCH_CMD"
    else
        set -x
        eval "$BENCH_CMD"
        set +x
    fi

    # Run evaluation if requested (before killing router)
    if [[ "${RUN_EVAL:-false}" == "true" ]]; then
        echo "Running lm-eval evaluation on Node 0..."

        EVAL_HEALTH_OK=false
        for _attempt in 1 2 3; do
            if curl -sf --max-time 10 "http://0.0.0.0:${ROUTER_PORT}/health" >/dev/null 2>&1; then
                EVAL_HEALTH_OK=true
                break
            fi
            echo "Eval health check attempt $_attempt failed, retrying in 10s..."
            sleep 10
        done

        if [[ "$EVAL_HEALTH_OK" != "true" ]]; then
            echo "WARNING: Router health check failed after 3 attempts. Skipping eval."
        else
            pushd /workspace

            source /workspace/benchmarks/benchmark_lib.sh

            if [[ -n "${EVAL_CONC:-}" ]]; then
                export EVAL_CONCURRENT_REQUESTS="${EVAL_CONC}"
            else
                export EVAL_CONCURRENT_REQUESTS=$(echo "$BENCH_MAX_CONCURRENCY" | tr 'x' '\n' | sort -n | tail -1)
            fi

            if [[ "$DRY_RUN" -eq 1 ]]; then
                echo "DRY RUN: run_eval --framework lm-eval --port $ROUTER_PORT (conc=${EVAL_CONCURRENT_REQUESTS}, ctx=${EVAL_MAX_MODEL_LEN:-auto})"
            else
                run_eval --framework lm-eval --port "$ROUTER_PORT"
                eval_rc=$?

                if [[ $eval_rc -ne 0 ]]; then
                    echo "ERROR: run_eval exited rc=$eval_rc; skipping metadata write and eval artifact staging" >&2
                    EVAL_FAILED=1
                else
                    export TP="${PREFILL_TP_SIZE}"
                    export CONC="${EVAL_CONCURRENT_REQUESTS}"
                    export EP_SIZE=1
                    [[ "${PREFILL_ENABLE_EP}" == "true" ]] && EP_SIZE="${PREFILL_TP_SIZE}"
                    export PREFILL_TP="${PREFILL_TP_SIZE}"
                    export PREFILL_EP=1
                    [[ "${PREFILL_ENABLE_EP}" == "true" ]] && PREFILL_EP="${PREFILL_TP_SIZE}"
                    export PREFILL_NUM_WORKERS="${xP}"
                    export DECODE_TP="${DECODE_TP_SIZE}"
                    export DECODE_EP=1
                    [[ "${DECODE_ENABLE_EP}" == "true" ]] && DECODE_EP="${DECODE_TP_SIZE}"
                    export DECODE_NUM_WORKERS="${yD}"
                    export DP_ATTENTION="${PREFILL_ENABLE_DP}"
                    export PREFILL_DP_ATTENTION="${PREFILL_ENABLE_DP}"
                    export DECODE_DP_ATTENTION="${DECODE_ENABLE_DP}"
                    export ISL="${BENCH_INPUT_LEN}"
                    export OSL="${BENCH_OUTPUT_LEN}"

                    append_lm_eval_summary

                    EVAL_COPY_DIR="/run_logs/slurm_job-${SLURM_JOB_ID}/eval_results"
                    mkdir -p "$EVAL_COPY_DIR"
                    for f in meta_env.json; do
                        [ -e "/workspace/$f" ] && cp -f "/workspace/$f" "$EVAL_COPY_DIR/"
                    done
                    find /workspace -maxdepth 1 -name 'results*.json' -exec cp -f {} "$EVAL_COPY_DIR/" \;
                    find /workspace -maxdepth 1 -name 'sample*.jsonl' -exec cp -f {} "$EVAL_COPY_DIR/" \;

                    echo "Eval completed. Artifacts staged in $EVAL_COPY_DIR"
                fi
            fi

            popd
        fi
    fi

    # Copy benchmark/eval results to BENCHMARK_LOGS_DIR (mounted from host)
    LOGS_OUTPUT="${BENCHMARK_LOGS_DIR:-/run_logs}/logs"
    mkdir -p "$LOGS_OUTPUT"

    if [[ "$DRY_RUN" -eq 0 ]]; then
        cp -r /run_logs/slurm_job-${SLURM_JOB_ID} "$LOGS_OUTPUT/"
        echo "Copied results to $LOGS_OUTPUT/slurm_job-${SLURM_JOB_ID}"
    fi

    echo "Killing the prefill server"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        [[ -n "${prefill_pid:-}" ]] && kill $prefill_pid 2>/dev/null || true
        sleep 2
        pkill -f "vllm serve" 2>/dev/null || true
    fi

    if [[ "${EVAL_FAILED:-0}" -eq 1 ]]; then
        echo "ERROR: eval failed; exiting node-0 with rc=1"
        exit 1
    fi

elif [ "$NODE_RANK" -gt 0 ] && [ "$NODE_RANK" -lt "$xP" ]; then
    echo "${host_name}:${host_ip} is Additional Prefill Node (Model: ${MODEL_NAME})"
    echo "Using prefill config: $PREFILL_SERVER_CONFIG"

    setup_vllm_env
    ensure_mooncake_kv_offload "$PREFILL_TP_SIZE"
    KV_TRANSFER_JSON=$(build_kv_transfer_config_json kv_producer)

    SERVED_MODEL="${MODEL_NAME}"
    PREFILL_CMD="vllm serve ${MODEL_PATH} \
        --served-model-name ${SERVED_MODEL} \
        --port $SERVER_PORT \
        --trust-remote-code \
        --kv-transfer-config '${KV_TRANSFER_JSON}' \
        ${PREFILL_SERVER_CONFIG}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $PREFILL_CMD"
    else
        PREFILL_LOG_FILE="/run_logs/slurm_job-${SLURM_JOB_ID}/prefill_${host_name}.log"
        set -x
        eval "$PREFILL_CMD" > "$PREFILL_LOG_FILE" 2>&1 &
        set +x
        prefill_pid=$!
    fi

    echo "Waiting for proxy server to be up..."
    BARRIER_CMD="python3 $WS_PATH/sync.py barrier \
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
    WAIT_CMD="python3 $WS_PATH/sync.py wait \
        --remote-ip ${NODE0_ADDR} \
        --remote-port ${ROUTER_PORT}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $WAIT_CMD"
    else
        eval "$WAIT_CMD"
    fi

    echo "Killing the prefill server"
    [[ "$DRY_RUN" -eq 0 ]] && kill $prefill_pid 2>/dev/null || true

else
    echo "${host_name}:${host_ip} is Decode Node (Model: ${MODEL_NAME})"
    echo "Using decode config: $DECODE_SERVER_CONFIG"

    setup_vllm_env
    ensure_mooncake_kv_offload "$DECODE_TP_SIZE"
    KV_TRANSFER_JSON=$(build_kv_transfer_config_json kv_consumer)

    for env_pair in ${DECODE_MODEL_ENVS}; do
        export "$env_pair"
        echo "[DECODE_ENV] $env_pair"
    done

    SERVED_MODEL="${MODEL_NAME}"
    DECODE_CMD="vllm serve ${MODEL_PATH} \
        --served-model-name ${SERVED_MODEL} \
        --port $SERVER_PORT \
        --trust-remote-code \
        --kv-transfer-config '${KV_TRANSFER_JSON}' \
        ${DECODE_SERVER_CONFIG}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $DECODE_CMD"
    else
        DECODE_LOG_FILE="/run_logs/slurm_job-${SLURM_JOB_ID}/decode_${host_name}.log"
        set -x
        eval "$DECODE_CMD" > "$DECODE_LOG_FILE" 2>&1 &
        set +x
        decode_pid=$!
    fi

    echo "Waiting for proxy server to be up..."
    BARRIER_CMD="python3 $WS_PATH/sync.py barrier \
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
    WAIT_CMD="python3 $WS_PATH/sync.py wait \
        --remote-ip ${NODE0_ADDR} \
        --remote-port ${ROUTER_PORT}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $WAIT_CMD"
    else
        eval "$WAIT_CMD"
    fi

    echo "Killing the decode server"
    [[ "$DRY_RUN" -eq 0 ]] && kill $decode_pid 2>/dev/null || true
fi

# echo "Killing the etcd server"
# kill $etcd_pid 2>/dev/null || true
# pkill -f etcd 2>/dev/null || true

echo "Script completed successfully"
exit 0
