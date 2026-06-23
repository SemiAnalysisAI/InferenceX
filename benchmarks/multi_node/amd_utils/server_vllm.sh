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
if [[ "${PREFILL_ENABLE_DP:-false}" == "true" ]] && ! echo "$PREFILL_SERVER_CONFIG" | grep -q -- '--enable-dp-attention'; then
    PREFILL_SERVER_CONFIG+=" --enable-dp-attention"
fi
if [[ "${DECODE_ENABLE_EP:-false}" == "true" ]] && ! echo "$DECODE_SERVER_CONFIG" | grep -q -- '--enable-expert-parallel'; then
    DECODE_SERVER_CONFIG+=" --enable-expert-parallel"
fi
if [[ "${DECODE_ENABLE_DP:-false}" == "true" ]] && ! echo "$DECODE_SERVER_CONFIG" | grep -q -- '--enable-dp-attention'; then
    DECODE_SERVER_CONFIG+=" --enable-dp-attention"
fi

echo "PREFILL_SERVER_CONFIG (after TP/EP/DP): $PREFILL_SERVER_CONFIG"
echo "DECODE_SERVER_CONFIG (after TP/EP/DP): $DECODE_SERVER_CONFIG"

# =============================================================================
# Optional: force-enable APC / prefix caching (vLLM "radix cache" = L1 tier).
# Some model entries ship --no-enable-prefix-caching (e.g. Kimi-K2.5-MXFP4);
# ENABLE_PREFIX_CACHING=1 strips that opt-out and adds --enable-prefix-caching
# on both prefill and decode. Gated so other recipes are unaffected.
# =============================================================================
if [[ "${ENABLE_PREFIX_CACHING:-0}" == "1" ]]; then
    for _cfg in PREFILL_SERVER_CONFIG DECODE_SERVER_CONFIG; do
        _val="${!_cfg}"
        _val="${_val//--no-enable-prefix-caching/}"
        if ! echo "$_val" | grep -q -- '--enable-prefix-caching'; then
            _val+=" --enable-prefix-caching"
        fi
        printf -v "$_cfg" '%s' "$_val"
    done
    echo "[vLLM] ENABLE_PREFIX_CACHING=1 -> prefix/radix cache (L1) enabled on prefill + decode"
fi

# =============================================================================
# KV connector selection: MoRIIO (default) | LMCache + Mooncake distributed store
# -----------------------------------------------------------------------------
# Default = MoRIIOConnector (unchanged upstream behavior). When
# KV_CONNECTOR=lmcache-mooncake we instead run a single LMCacheConnectorV1
# (kv_both on every node) backed by a Mooncake store, with all reuse tiers:
#   L1 vLLM APC (ENABLE_PREFIX_CACHING) + L2 LMCache local_cpu host DRAM +
#   L3 mooncakestore (cross-node). Mooncake-TCP is the default L3 transport on
#   MI355X ionic NICs (host RDMA MR registration is capped <=64MiB/MR,
#   ~3.8GiB/node; TCP has no NIC registration limit so the L3 pool can be large).
# =============================================================================
KV_CONNECTOR="${KV_CONNECTOR:-moriio}"
_MORIIO_EXTRA="\"kv_connector_extra_config\": {\"proxy_ip\": \"${NODE0_ADDR}\", \"proxy_ping_port\": \"${PROXY_PING_PORT:-36367}\", \"http_port\": \"${SERVER_PORT}\"}"
KVT_PREFILL="{\"kv_connector\": \"MoRIIOConnector\", \"kv_role\": \"kv_producer\", ${_MORIIO_EXTRA}}"
KVT_DECODE="{\"kv_connector\": \"MoRIIOConnector\", \"kv_role\": \"kv_consumer\", ${_MORIIO_EXTRA}}"

if [[ "$KV_CONNECTOR" == "lmcache-mooncake" ]]; then
    # Override the container's mooncake .so with the hipseg-patched build, if a
    # patched-.so dir is mounted (EXTRA_DOCKER_MOUNTS). The patch stops
    # HipTransport::install() from clobbering the segment protocol to "hip",
    # which otherwise breaks cross-node segment open on both TCP and RDMA.
    MC_PATCHED_SO_DIR="${MC_PATCHED_SO_DIR:-/mc_dmabuf_so}"
    if [[ -d "$MC_PATCHED_SO_DIR" ]]; then
        _mcpkg=$(python3 -c "import mooncake, os; print(os.path.dirname(mooncake.__file__))" 2>/dev/null || true)
        if [[ -n "$_mcpkg" && -d "$_mcpkg" ]]; then
            for _so in engine store; do
                _src=$(ls "$MC_PATCHED_SO_DIR"/${_so}*.so 2>/dev/null | head -1)
                [[ -n "$_src" ]] && cp -f "$_src" "$_mcpkg/" && echo "[Mooncake] overrode ${_so}.so from $MC_PATCHED_SO_DIR"
            done
        fi
    fi
    MC_MASTER_ADDR_EFF="${MC_MASTER_ADDR:-${NODE0_ADDR}:${MC_MASTER_PORT:-50051}}"
    MC_METADATA_URL="http://${NODE0_ADDR}:${MC_METADATA_PORT:-8080}/metadata"
    LMC_CFG_DIR="/run_logs/slurm_job-${SLURM_JOB_ID}"
    mkdir -p "$LMC_CFG_DIR"
    LMCACHE_CONFIG_FILE="${LMC_CFG_DIR}/lmcache_rank${NODE_RANK}.yaml"
    # Decoders pull the remote (prefill) segment; prefill prefers local alloc.
    _prefer_local=true
    [[ "$NODE_RANK" -ge "$xP" ]] && _prefer_local=false
    cat > "$LMCACHE_CONFIG_FILE" <<LMCEOF
local_cpu: ${LMCACHE_LOCAL_CPU:-True}
max_local_cpu_size: ${LMCACHE_MAX_LOCAL_CPU_GB:-150}
numa_mode: "auto"
remote_url: "mooncakestore://${MC_MASTER_ADDR_EFF}/"
pre_caching_hash_algorithm: sha256_cbor_64bit
extra_config:
  protocol: "${MC_PROTOCOL:-tcp}"
  device_name: "${MC_DEVICE:-}"
  local_hostname: "${host_ip}"
  mooncake_master_server_addr: "${MC_MASTER_ADDR_EFF}"
  master_server_address: "${MC_MASTER_ADDR_EFF}"
  metadata_server: "${MC_METADATA_URL}"
  global_segment_size: ${MC_GLOBAL_SEG:-274877906944}
  local_buffer_size: ${MC_LOCAL_BUFFER:-4294967296}
  save_chunk_meta: True
  use_exists_sync: true
  mooncake_prefer_local_alloc: ${_prefer_local}
LMCEOF
    echo "[LMCache] rank=${NODE_RANK} wrote $LMCACHE_CONFIG_FILE (master=$MC_MASTER_ADDR_EFF proto=${MC_PROTOCOL:-tcp} prefer_local=$_prefer_local)"
    export LMCACHE_CONFIG_FILE
    export LMCACHE_USE_EXPERIMENTAL=True
    export PYTHONHASHSEED=0
    [[ "${MC_PROTOCOL:-tcp}" == "tcp" ]] && export MC_FORCE_TCP=1
    # CheckRegisterMemoryParams enforces MC_MAX_MR_SIZE even for TCP; keep it
    # large for TCP, or 64MiB for the (capped) ionic host-RDMA path.
    export MC_MAX_MR_SIZE="${MC_MAX_MR_SIZE:-137438953472}"
    KVT_PREFILL="{\"kv_connector\": \"LMCacheConnectorV1\", \"kv_role\": \"kv_both\", \"kv_load_failure_policy\": \"recompute\"}"
    KVT_DECODE="$KVT_PREFILL"
    echo "[KV] KV_CONNECTOR=lmcache-mooncake -> LMCacheConnectorV1 (kv_both) on all roles"
fi

# =============================================================================
# KV connector: LMCache over MoRI-IO P2P  (L2 + 1P1D, NO mooncake / NO L3 store)
# -----------------------------------------------------------------------------
# A single LMCacheConnectorV1 with transfer_channel=mori does BOTH the
# cross-node prefill->decode KV transfer (P2P RDMA over the lmcache MoriChannel,
# registering GPU memory via dmabuf -> sidesteps the ionic host-MR cap) AND L2
# prefix reuse (local_cpu host DRAM). One connector, no MultiConnector
# block-ownership corruption, no mooncake master/store. Roles split into
# kv_producer (prefill, pd_role=sender) and kv_consumer (decode, pd_role=
# receiver). The LMCache PD proxy (disagg_proxy_server.py) injects the per-request
# disagg_spec that tells the prefiller which decoder to push KV to.
# =============================================================================
if [[ "$KV_CONNECTOR" == "lmcache-mori" ]]; then
    # Inject the lmcache MoriChannel into the installed LMCache + register it in
    # the transfer-channel factory (idempotent).
    MORI_CHANNEL_SRC="${MORI_CHANNEL_SRC:-$WS_PATH/mori_channel.py}" \
        python3 "$WS_PATH/patch_lmcache.py" || echo "[mori] WARN: patch_lmcache.py failed"

    export PYTHONHASHSEED=0           # consistent KV hashing across procs
    export LMCACHE_USE_EXPERIMENTAL=True
    export VLLM_HOST_IP="${rdma_ip}"  # mori IOEngine binds here for cross-node RDMA

    # libionic so mori/libibverbs enumerate the ionic NICs (if repo is mounted)
    if ls /ainic-repo/libionic1*.deb >/dev/null 2>&1; then
        dpkg -i /ainic-repo/ionic-common*.deb /ainic-repo/libionic1*.deb 2>/dev/null || true
    fi

    # PD control-plane ports (single set; 1P1D TP8 validated). The decoder binds
    # init/alloc listeners; the prefiller (sender) connects per-request.
    LMC_PD_PROXY_HOST="${NODE0_ADDR}"
    LMC_PD_PROXY_PORT="${LMC_PD_PROXY_PORT:-7500}"
    LMC_PD_INIT_PORT="${LMC_PD_INIT_PORT:-7300}"
    LMC_PD_ALLOC_PORT="${LMC_PD_ALLOC_PORT:-7400}"
    LMC_PD_BUFFER_SIZE="${LMC_PD_BUFFER_SIZE:-8589934592}"   # 8 GiB GPU staging
    LMC_CHUNK_SIZE="${LMC_CHUNK_SIZE:-256}"

    LMC_CFG_DIR="/run_logs/slurm_job-${SLURM_JOB_ID}"
    mkdir -p "$LMC_CFG_DIR"
    LMCACHE_CONFIG_FILE="${LMC_CFG_DIR}/lmcache_mori_rank${NODE_RANK}.yaml"

    if [[ "$NODE_RANK" -lt "$xP" ]]; then
        # ---- prefiller (KV producer / PD sender) ----
        cat > "$LMCACHE_CONFIG_FILE" <<LMMORIEOF
local_cpu: ${LMCACHE_LOCAL_CPU:-True}
max_local_cpu_size: ${LMCACHE_MAX_LOCAL_CPU_GB:-150}
chunk_size: ${LMC_CHUNK_SIZE}
enable_pd: True
transfer_channel: "mori"
pd_role: "sender"
pd_proxy_host: "${LMC_PD_PROXY_HOST}"
pd_proxy_port: ${LMC_PD_PROXY_PORT}
pd_buffer_size: ${LMC_PD_BUFFER_SIZE}
pd_buffer_device: "cuda"
pd_backend_mode: "async"
pd_max_prefill_len: ${LMC_PD_MAX_PREFILL_LEN:-16384}
LMMORIEOF
    else
        # ---- decoder (KV consumer / PD receiver) ----
        # pd_peer_host MUST be a LOCAL bind addr (the receiver BINDS init/alloc on
        # {pd_peer_host}:{port}); the sender reaches us via the proxy-injected
        # disagg_spec (our real IP). Using the prefiller IP here would fail to bind.
        cat > "$LMCACHE_CONFIG_FILE" <<LMMORIEOF
local_cpu: ${LMCACHE_LOCAL_CPU:-True}
max_local_cpu_size: ${LMCACHE_MAX_LOCAL_CPU_GB:-150}
chunk_size: ${LMC_CHUNK_SIZE}
enable_pd: True
transfer_channel: "mori"
pd_role: "receiver"
pd_peer_host: "0.0.0.0"
pd_peer_init_port: ${LMC_PD_INIT_PORT}
pd_peer_alloc_port: ${LMC_PD_ALLOC_PORT}
pd_buffer_size: ${LMC_PD_BUFFER_SIZE}
pd_buffer_device: "cuda"
pd_backend_mode: "async"
pd_max_prefill_len: ${LMC_PD_MAX_PREFILL_LEN:-16384}
LMMORIEOF
    fi
    export LMCACHE_CONFIG_FILE
    echo "[LMCache] rank=${NODE_RANK} wrote $LMCACHE_CONFIG_FILE (transfer_channel=mori, L2 local_cpu=${LMCACHE_LOCAL_CPU:-True}/${LMCACHE_MAX_LOCAL_CPU_GB:-150}GB)"

    KVT_PREFILL="{\"kv_connector\": \"LMCacheConnectorV1\", \"kv_role\": \"kv_producer\", \"kv_connector_extra_config\": {\"discard_partial_chunks\": false, \"lmcache_rpc_port\": \"producer1\"}}"
    KVT_DECODE="{\"kv_connector\": \"LMCacheConnectorV1\", \"kv_role\": \"kv_consumer\", \"kv_connector_extra_config\": {\"skip_last_n_tokens\": 1, \"discard_partial_chunks\": false, \"lmcache_rpc_port\": \"consumer1\"}}"
    echo "[KV] KV_CONNECTOR=lmcache-mori -> LMCacheConnectorV1 P2P (kv_producer/kv_consumer) + L2 local_cpu, no L3"
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

    if [[ "$KV_CONNECTOR" == "lmcache-mooncake" ]]; then
        # Start the Mooncake master + HTTP metadata server in-container on node 0.
        # LMCache store clients (all ranks) connect here during engine init, so it
        # must be up before the prefill/decode vllm serve processes start.
        MC_MASTER_LOG="/run_logs/slurm_job-${SLURM_JOB_ID}/mooncake_master.log"
        echo "[Mooncake] starting master on 0.0.0.0:${MC_MASTER_PORT:-50051} (metadata :${MC_METADATA_PORT:-8080})"
        ( mooncake_master --enable_http_metadata_server=1 \
            --http_metadata_server_host=0.0.0.0 --http_metadata_server_port=${MC_METADATA_PORT:-8080} \
            --rpc_address=0.0.0.0 --port=${MC_MASTER_PORT:-50051} -v=1 > "$MC_MASTER_LOG" 2>&1 & ) || \
            echo "[Mooncake] WARN: master failed to start (see $MC_MASTER_LOG)"
        sleep 6
    elif [[ "$KV_CONNECTOR" == "lmcache-mori" ]]; then
        # No master/store: the in-container LMCache PD proxy (started after the
        # engines are up) owns ROUTER_PORT. job.slurm skips the external
        # vllm-router because ROUTER_TYPE=lmc-proxy (not "vllm-router").
        echo "[mori] no external router / no mooncake master; in-container PD proxy will own ROUTER_PORT"
    else
        # Router is started as an external container by job.slurm (VLLM_ROUTER_IMAGE)
        echo "Using external vllm-router container (started by job.slurm on this node)"
    fi

    SERVED_MODEL="${MODEL_NAME}"
    PREFILL_CMD="vllm serve ${MODEL_PATH} \
        --served-model-name ${SERVED_MODEL} \
        --port $SERVER_PORT \
        --trust-remote-code \
        --kv-transfer-config '${KVT_PREFILL}' \
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

    if [[ "$KV_CONNECTOR" == "lmcache-mooncake" ]]; then
        # Start the minimal 1P1D PD proxy in-container on node 0 (owns ROUTER_PORT).
        # prefiller = this node; decoder = first decode node (IP_ARRAY[xP]).
        MC_PROXY_DECODER_IP="${IP_ARRAY[$xP]:-${NODE0_ADDR}}"
        MC_PROXY_LOG="/run_logs/slurm_job-${SLURM_JOB_ID}/mc_pd_proxy.log"
        echo "[Mooncake] starting mc_pd_proxy :${ROUTER_PORT} (P=${NODE0_ADDR}:${SERVER_PORT} D=${MC_PROXY_DECODER_IP}:${SERVER_PORT})"
        python3 -c 'import httpx,fastapi,uvicorn' 2>/dev/null || pip install -q httpx fastapi uvicorn
        ( python3 "$WS_PATH/mc_pd_proxy.py" --host 0.0.0.0 --port "${ROUTER_PORT}" \
            --prefiller-host "${NODE0_ADDR}" --prefiller-port "${SERVER_PORT}" \
            --decoder-host "${MC_PROXY_DECODER_IP}" --decoder-port "${SERVER_PORT}" \
            > "$MC_PROXY_LOG" 2>&1 & ) || echo "[Mooncake] WARN: proxy failed to start (see $MC_PROXY_LOG)"
        sleep 4
    fi

    if [[ "$KV_CONNECTOR" == "lmcache-mori" ]]; then
        # Start the LMCache PD proxy in-container on node 0 (owns ROUTER_PORT).
        # It tokenizes, runs prefill (max_tokens=1) with an injected disagg_spec
        # (receiver = first decode node init/alloc ports), waits for the prefiller's
        # "KV ready" zmq notify on LMC_PD_PROXY_PORT, then streams the decode.
        # prefiller = this node; decoder = first decode node (IP_ARRAY[xP]).
        LMC_PROXY_DECODER_IP="${IP_ARRAY[$xP]:-${NODE0_ADDR}}"
        LMC_PROXY_LOG="/run_logs/slurm_job-${SLURM_JOB_ID}/lmc_mori_proxy.log"
        echo "[mori] starting disagg_proxy_server :${ROUTER_PORT} (P=${NODE0_ADDR}:${SERVER_PORT} D=${LMC_PROXY_DECODER_IP}:${SERVER_PORT} reg=${LMC_PD_PROXY_PORT})"
        python3 -c 'import httpx,fastapi,uvicorn,msgspec' 2>/dev/null || pip install -q httpx fastapi uvicorn msgspec
        ( python3 "$WS_PATH/disagg_proxy_server.py" --host 0.0.0.0 --port "${ROUTER_PORT}" \
            --prefiller-host "${NODE0_ADDR}" --prefiller-port "${SERVER_PORT}" \
            --decoder-host "${LMC_PROXY_DECODER_IP}" --decoder-port "${SERVER_PORT}" \
            --decoder-init-port "${LMC_PD_INIT_PORT}" --decoder-alloc-port "${LMC_PD_ALLOC_PORT}" \
            --proxy-host "${NODE0_ADDR}" --proxy-port "${LMC_PD_PROXY_PORT}" \
            --pd-buffer-size "${LMC_PD_BUFFER_SIZE}" --chunk-size "${LMC_CHUNK_SIZE}" \
            --model "${MODEL_PATH}" > "$LMC_PROXY_LOG" 2>&1 & ) || echo "[mori] WARN: proxy failed to start (see $LMC_PROXY_LOG)"
        sleep 6
    fi

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
    BENCH_CMD="bash $WS_PATH/bench.sh ${xP} ${yD} $((GPUS_PER_NODE*xP)) $((GPUS_PER_NODE*yD)) \
        $MODEL_DIR $MODEL_NAME /run_logs/slurm_job-${SLURM_JOB_ID} ${BENCH_INPUT_LEN} \
        ${BENCH_OUTPUT_LEN} \"${BENCH_MAX_CONCURRENCY}\" ${BENCH_REQUEST_RATE} \
        ${BENCH_RANDOM_RANGE_RATIO} ${BENCH_NUM_PROMPTS_MULTIPLIER}"

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

    SERVED_MODEL="${MODEL_NAME}"
    PREFILL_CMD="vllm serve ${MODEL_PATH} \
        --served-model-name ${SERVED_MODEL} \
        --port $SERVER_PORT \
        --trust-remote-code \
        --kv-transfer-config '${KVT_PREFILL}' \
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

    for env_pair in ${DECODE_MODEL_ENVS}; do
        export "$env_pair"
        echo "[DECODE_ENV] $env_pair"
    done

    SERVED_MODEL="${MODEL_NAME}"
    DECODE_CMD="vllm serve ${MODEL_PATH} \
        --served-model-name ${SERVED_MODEL} \
        --port $SERVER_PORT \
        --trust-remote-code \
        --kv-transfer-config '${KVT_DECODE}' \
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
