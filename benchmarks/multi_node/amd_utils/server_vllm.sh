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
IFS=',' read -ra NODE_IP_ARRAY <<< "$IPADDRS"
node_ip="${NODE_IP_ARRAY[$NODE_RANK]:-${NODE0_ADDR}}"
host_ip="${host_ip:-$node_ip}"
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
if [[ "${PREFILL_ENABLE_DP:-false}" == "true" ]] && ! echo "$PREFILL_SERVER_CONFIG" | grep -q -- '--decode-context-parallel-size'; then
    PREFILL_SERVER_CONFIG+=" --decode-context-parallel-size ${PREFILL_TP_SIZE:-1}"
fi
if [[ "${DECODE_ENABLE_EP:-false}" == "true" ]] && ! echo "$DECODE_SERVER_CONFIG" | grep -q -- '--enable-expert-parallel'; then
    DECODE_SERVER_CONFIG+=" --enable-expert-parallel"
fi
if [[ "${DECODE_ENABLE_DP:-false}" == "true" ]] && ! echo "$DECODE_SERVER_CONFIG" | grep -q -- '--decode-context-parallel-size'; then
    DECODE_SERVER_CONFIG+=" --decode-context-parallel-size ${DECODE_TP_SIZE:-1}"
fi

echo "PREFILL_SERVER_CONFIG (after TP/EP/DP): $PREFILL_SERVER_CONFIG"
echo "DECODE_SERVER_CONFIG (after TP/EP/DP): $DECODE_SERVER_CONFIG"

if [[ "${ENABLE_PREFIX_CACHING:-0}" == "1" ]]; then
    for _cfg in PREFILL_SERVER_CONFIG DECODE_SERVER_CONFIG; do
        _val="${!_cfg}"
        _val="${_val//--no-enable-prefix-caching/}"
        if ! echo "$_val" | grep -q -- '--enable-prefix-caching'; then
            _val+=" --enable-prefix-caching"
        fi
        printf -v "$_cfg" '%s' "$_val"
    done
    echo "[vLLM] ENABLE_PREFIX_CACHING=1 -> prefix cache enabled on prefill + decode"
fi

if [[ -n "${MAX_MODEL_LEN:-}" && "${MAX_MODEL_LEN}" != "0" ]]; then
    for _cfg in PREFILL_SERVER_CONFIG DECODE_SERVER_CONFIG; do
        _val="${!_cfg}"
        if echo "$_val" | grep -q -- '--max-model-len'; then
            _val=$(echo "$_val" | sed -E "s/--max-model-len[=[:space:]]+[0-9]+/--max-model-len ${MAX_MODEL_LEN}/g")
        else
            _val+=" --max-model-len ${MAX_MODEL_LEN}"
        fi
        printf -v "$_cfg" '%s' "$_val"
    done
    echo "[vLLM] MAX_MODEL_LEN=${MAX_MODEL_LEN}"
fi

if [[ -n "${MAX_NUM_SEQS:-}" && "${MAX_NUM_SEQS}" != "0" ]]; then
    for _cfg in PREFILL_SERVER_CONFIG DECODE_SERVER_CONFIG; do
        _val="${!_cfg}"
        if echo "$_val" | grep -q -- '--max-num-seqs'; then
            _val=$(echo "$_val" | sed -E "s/--max-num-seqs[=[:space:]]+[0-9]+/--max-num-seqs ${MAX_NUM_SEQS}/g")
        else
            _val+=" --max-num-seqs ${MAX_NUM_SEQS}"
        fi
        printf -v "$_cfg" '%s' "$_val"
    done
    echo "[vLLM] MAX_NUM_SEQS=${MAX_NUM_SEQS}"
fi

# =============================================================================
# Container Synchronization
# =============================================================================

CONTAINER_BARRIER_PORT="${CONTAINER_BARRIER_PORT:-5000}"
echo "Waiting at the container creation barrier on $host_name (port ${CONTAINER_BARRIER_PORT})"
python3 $WS_PATH/sync.py barrier \
    --local-ip ${host_ip} \
    --local-port ${CONTAINER_BARRIER_PORT} \
    --enable-port \
    --node-ips ${IPADDRS} \
    --node-ports ${CONTAINER_BARRIER_PORT} \
    --wait-for-all-ports \
    --timeout 600

# =============================================================================
# Cluster Topology Configuration
# =============================================================================
IP_ARRAY=("${NODE_IP_ARRAY[@]}")

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

# =============================================================================
# KV connector selection
# =============================================================================
_MORIIO_EXTRA="\"kv_connector_extra_config\": {\"proxy_ip\": \"${NODE0_ADDR}\", \"proxy_ping_port\": \"${PROXY_PING_PORT}\", \"http_port\": \"${SERVER_PORT}\"}"
MORIIO_PREFILL_CONN="{\"kv_connector\": \"MoRIIOConnector\", \"kv_role\": \"kv_producer\", ${_MORIIO_EXTRA}}"
MORIIO_DECODE_CONN="{\"kv_connector\": \"MoRIIOConnector\", \"kv_role\": \"kv_consumer\", ${_MORIIO_EXTRA}}"
KVT_PREFILL="$MORIIO_PREFILL_CONN"
KVT_DECODE="$MORIIO_DECODE_CONN"

LMCACHE_CONNECT_HOST="${LMCACHE_CONNECT_HOST:-tcp://${LMCACHE_CONNECT_HOST_IP:-127.0.0.1}}"
LMC_CONN="{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"${LMCACHE_CONNECT_HOST}\",\"lmcache.mp.port\":${LMCACHE_PORT:-5555},\"lmcache.mp.mq_timeout\":${LMCACHE_MP_MQ_TIMEOUT:-1200}}}"

# -----------------------------------------------------------------------------
# LMCache MP + native NIXL/RIXL P/D transfer.
# -----------------------------------------------------------------------------
# The current LMCache MP docs compose vLLM P/D with MultiConnector:
#   NixlConnector       : per-request prefill -> decode KV handoff
#   LMCacheMPConnector  : local LMCache server offload/load for reuse
# P2P sharing is implemented by lmcache server + coordinator.
LMC_NIXL_PREFILL_XFER="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_producer\",\"kv_load_failure_policy\":\"fail\"}"
LMC_NIXL_DECODE_XFER="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_consumer\",\"kv_load_failure_policy\":\"fail\"}"
LMC_NIXL_PREFILL_CONN="{\"kv_connector\":\"MultiConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"connectors\":[${LMC_NIXL_PREFILL_XFER},${LMC_CONN}]}}"
LMC_NIXL_DECODE_CONN="{\"kv_connector\":\"MultiConnector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{\"connectors\":[${LMC_NIXL_DECODE_XFER},${LMC_CONN}]}}"

case "${PREFILL_KV_CONNECTOR:-moriio}" in
    moriio-lmcachemp)
        KVT_PREFILL="{\"kv_connector\":\"MultiConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"connectors\":[${MORIIO_PREFILL_CONN},${LMC_CONN}]}}"
        ;;
    lmcache-nixl)
        KVT_PREFILL="$LMC_NIXL_PREFILL_CONN"
        ;;
    moriio|"")
        ;;
    *)
        echo "ERROR: unsupported PREFILL_KV_CONNECTOR=${PREFILL_KV_CONNECTOR}" >&2
        exit 1
        ;;
esac

case "${DECODE_KV_CONNECTOR:-moriio}" in
    moriio-lmcachemp)
        KVT_DECODE="{\"kv_connector\":\"MultiConnector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{\"connectors\":[${MORIIO_DECODE_CONN},${LMC_CONN}]}}"
        ;;
    lmcache-nixl)
        KVT_DECODE="$LMC_NIXL_DECODE_CONN"
        ;;
    moriio|"")
        ;;
    *)
        echo "ERROR: unsupported DECODE_KV_CONNECTOR=${DECODE_KV_CONNECTOR}" >&2
        exit 1
        ;;
esac

if [[ "$NODE_RANK" -lt "$xP" ]]; then
    ROLE_KV_CONNECTOR="${PREFILL_KV_CONNECTOR:-moriio}"
else
    ROLE_KV_CONNECTOR="${DECODE_KV_CONNECTOR:-moriio}"
fi
echo "[KV] PREFILL_KV_CONNECTOR=${PREFILL_KV_CONNECTOR:-moriio}; DECODE_KV_CONNECTOR=${DECODE_KV_CONNECTOR:-moriio}; rank connector=${ROLE_KV_CONNECTOR}"


# vLLM runtime environment (static vars moved to env.sh; these depend on per-node state)
setup_vllm_env() {
    local bind_ip="${VLLM_BIND_IP:-${rdma_ip}}"
    export VLLM_HOST_IP="${bind_ip}"
    export VLLM_NIXL_SIDE_CHANNEL_HOST="${bind_ip}"
    export VLLM_NIXL_SIDE_CHANNEL_PORT="${VLLM_NIXL_SIDE_CHANNEL_PORT:-$((5600 + NODE_RANK))}"
    if [[ "$ROLE_KV_CONNECTOR" == "lmcache-nixl" ]]; then
        # LMCache MP P2P docs recommend UCX_NET_DEVICES=all. The generic
        # MoRI/UCX auto-detect path may set ionic_0:1, which is not visible to
        # LMCache's NIXL backend in the current nightly container.
        export UCX_NET_DEVICES="${LMCACHE_NIXL_UCX_NET_DEVICES:-all}"
        export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
    fi
    for env_pair in ${MODEL_ENVS}; do
        export "$env_pair"
    done
    echo "[vLLM] VLLM_HOST_IP=${VLLM_HOST_IP} VLLM_NIXL_SIDE_CHANNEL_HOST=${VLLM_NIXL_SIDE_CHANNEL_HOST} VLLM_NIXL_SIDE_CHANNEL_PORT=${VLLM_NIXL_SIDE_CHANNEL_PORT}"
}

# LMCache MP P2P uses the coordinator for peer discovery. It is a control-plane
# service only; KV data still moves peer-to-peer via the server transfer channel.
start_lmcache_coordinator_if_needed() {
    if [[ "${LMCACHE_ENABLE_P2P:-false}" != "true" ]]; then
        return 0
    fi
    if [[ "$ROLE_KV_CONNECTOR" != *"-lmcachemp" && "$ROLE_KV_CONNECTOR" != "lmcache-nixl" ]]; then
        return 0
    fi
    if [[ "$NODE_RANK" -ne 0 ]]; then
        return 0
    fi

    local coord_host="${LMCACHE_COORDINATOR_HOST:-0.0.0.0}"
    local coord_port="${LMCACHE_COORDINATOR_PORT:-9300}"
    local coord_log="/run_logs/slurm_job-${SLURM_JOB_ID}/lmcache_coordinator_${host_name}.log"
    echo "[LMCacheMP] starting coordinator on ${coord_host}:${coord_port}"
    lmcache coordinator --host "$coord_host" --port "$coord_port" > "$coord_log" 2>&1 &
    LMCACHE_COORDINATOR_PID=$!

    for _i in $(seq 1 60); do
        if curl -sf --max-time 3 "http://${NODE0_ADDR}:${coord_port}/healthz" >/dev/null 2>&1; then
            echo "[LMCacheMP] coordinator healthy"
            return 0
        fi
        sleep 1
    done
    echo "ERROR: LMCache coordinator failed to become healthy; tailing $coord_log" >&2
    tail -n 80 "$coord_log" >&2 || true
    return 1
}

build_vllm_metrics_urls() {
    local urls=()
    local ip
    for ip in "${IP_ARRAY[@]}"; do
        [[ -n "$ip" ]] || continue
        urls+=("http://${ip}:${SERVER_PORT}/metrics")
    done
    (IFS=,; echo "${urls[*]}")
}

capture_vllm_cache_metrics() {
    local label="${1:-snapshot}"
    local out="/run_logs/slurm_job-${SLURM_JOB_ID}/vllm_cache_metrics_${label}.txt"
    local urls="${AIPERF_SERVER_METRICS_URLS:-$(build_vllm_metrics_urls)}"
    local url
    mkdir -p "$(dirname "$out")"
    {
        echo "=== vLLM cache metrics snapshot: ${label} $(date --iso-8601=seconds) ==="
        echo "AIPERF_SERVER_METRICS_URLS=${urls}"
        IFS=',' read -r -a _metric_urls <<< "$urls"
        for url in "${_metric_urls[@]}"; do
            [[ -n "$url" ]] || continue
            echo "--- ${url} ---"
            curl -fsS --max-time 5 "$url" 2>/dev/null \
                | grep -E '^(vllm:num_requests_running|vllm:num_requests_waiting|vllm:num_requests_waiting_by_reason|vllm:prompt_tokens$|vllm:generation_tokens|vllm:prompt_tokens_cached|vllm:prompt_tokens_by_source|vllm:external_prefix_cache_hits|vllm:external_prefix_cache_queries|vllm:kv_cache_usage_perc|vllm:gpu_cache_usage_perc|vllm:prefix_cache_hits|vllm:prefix_cache_queries|vllm:num_preemptions|vllm:request_success|vllm:kv_offload|vllm:cpu_kv|vllm:.*lmcache)' \
                || true
        done
    } | tee -a "$out"
}

capture_lmcache_metrics() {
    local label="${1:-snapshot}"
    if [[ "$ROLE_KV_CONNECTOR" != *"-lmcachemp" && "$ROLE_KV_CONNECTOR" != "lmcache-nixl" ]]; then
        return 0
    fi

    local http_host="${LMCACHE_HOST:-127.0.0.1}"
    local http_port="${LMCACHE_HTTP_PORT:-8080}"
    local base_url="http://${http_host}:${http_port}"
    local out="/run_logs/slurm_job-${SLURM_JOB_ID}/lmcache_metrics_${host_name}_${label}.txt"
    mkdir -p "$(dirname "$out")"

    {
        echo "=== LMCacheMP metrics snapshot: ${label} $(date --iso-8601=seconds) ==="
        echo "ROLE_KV_CONNECTOR=${ROLE_KV_CONNECTOR}"
        echo "LMCACHE_URL=${base_url}"
        echo "--- /healthcheck ---"
        curl -fsS --max-time 5 "${base_url}/healthcheck" 2>&1 || true
        echo
        echo "--- /metrics ---"
        if ! curl -fsS --max-time 10 "${base_url}/metrics" 2>&1; then
            echo "[LMCacheMP] /metrics unavailable"
        fi
        echo
        echo "--- filtered cache keywords from /metrics ---"
        curl -fsS --max-time 10 "${base_url}/metrics" 2>/dev/null \
            | grep -Ei 'hit|miss|evict|eviction|lookup|retrieve|store|read|write|cache|l1|chunk|token|byte|latency|duration' \
            || true
        echo
        echo "--- lmcache log tail ---"
        tail -n 120 "/run_logs/slurm_job-${SLURM_JOB_ID}/lmcache_${host_name}.log" 2>/dev/null || true
    } | tee -a "$out"
}

start_lmcache_mp_if_needed() {
    if [[ "$ROLE_KV_CONNECTOR" != *"-lmcachemp" && "$ROLE_KV_CONNECTOR" != "lmcache-nixl" ]]; then
        return 0
    fi

    LMCACHE_PATCH_DIR="/run_logs/slurm_job-${SLURM_JOB_ID}/lmcache_mp_patch"
    mkdir -p "$LMCACHE_PATCH_DIR"
    cat > "$LMCACHE_PATCH_DIR/sitecustomize.py" <<'PY'
"""Keep LMCacheMP from producing proxy-visible PD transfer params.

MultiConnector permits only one child connector to return kv_transfer_params.
The PD transfer connector owns the prefill->decode protocol; LMCacheMP should
only provide local L2 lookup/retrieve/store for prefix reuse.
"""
import builtins
import sys

_orig_import = builtins.__import__


def _patch_module(mod):
    cls = getattr(mod, "LMCacheMPConnector", None)
    if cls is None or getattr(cls, "_inferencex_pd_params_patch", False):
        return
    orig = cls.request_finished

    def request_finished(self, request, block_ids):
        async_save, params = orig(self, request, block_ids)
        req_params = getattr(request, "kv_transfer_params", None)
        if req_params and (
            req_params.get("do_remote_decode") or req_params.get("do_remote_prefill")
        ):
            return async_save, None
        return async_save, params

    cls.request_finished = request_finished
    orig_process_new = cls._process_new_requests

    def _process_new_requests(self, scheduler_output, metadata):
        for new_request in scheduler_output.scheduled_new_reqs:
            block_ids = getattr(new_request, "block_ids", None)
            if not block_ids:
                continue
            request_tracker = self._get_request_tracker(new_request.req_id)
            try:
                allocated = request_tracker.num_allocated_blocks()
            except Exception:
                allocated = {}
            if not allocated:
                request_tracker.append_block_ids(block_ids)
        return orig_process_new(self, scheduler_output, metadata)

    cls._process_new_requests = _process_new_requests
    cls._inferencex_pd_params_patch = True


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _orig_import(name, globals, locals, fromlist, level)
    target = "lmcache.integration.vllm.lmcache_mp_connector"
    if name == target or target in sys.modules:
        _patch_module(sys.modules[target])
    return mod


builtins.__import__ = _import
if "lmcache.integration.vllm.lmcache_mp_connector" in sys.modules:
    _patch_module(sys.modules["lmcache.integration.vllm.lmcache_mp_connector"])

try:
    from vllm.distributed.kv_transfer.kv_connector.v1 import multi_connector
    _orig_observe = multi_connector.MultiConnectorPromMetrics.observe

    def _safe_observe(self, transfer_stats_data, engine_idx):
        if not isinstance(transfer_stats_data, dict):
            return _orig_observe(self, transfer_stats_data, engine_idx)
        filtered = {
            connector_id: stats_data
            for connector_id, stats_data in transfer_stats_data.items()
            if connector_id in getattr(self, "_prom_metrics", {})
        }
        if filtered:
            return _orig_observe(self, filtered, engine_idx)

    multi_connector.MultiConnectorPromMetrics.observe = _safe_observe
except Exception:
    pass
PY
    export PYTHONPATH="$LMCACHE_PATCH_DIR${PYTHONPATH:+:$PYTHONPATH}"

    python3 -c 'import lmcache.integration.vllm.lmcache_mp_connector; print("LMCacheMPConnector import OK")'

    LMCACHE_LOG="/run_logs/slurm_job-${SLURM_JOB_ID}/lmcache_${host_name}.log"
    local enable_p2p="${LMCACHE_ENABLE_P2P:-false}"
    local server_host="${LMCACHE_SERVER_HOST:-${LMCACHE_HOST:-127.0.0.1}}"
    if [[ "$enable_p2p" == "true" ]]; then
        server_host="${LMCACHE_SERVER_HOST:-0.0.0.0}"
    fi

    local server_args=(
        --host "$server_host" --port "${LMCACHE_PORT:-5555}"
        --http-port "${LMCACHE_HTTP_PORT:-8080}"
        --l1-size-gb "${LMCACHE_L1_SIZE_GB:-1200}"
        --l1-init-size-gb "${LMCACHE_L1_INIT_SIZE_GB:-20}"
        --l1-read-ttl-seconds "${LMCACHE_L1_READ_TTL_SECONDS:-7200}"
        --chunk-size "${LMCACHE_CHUNK_SIZE:-256}"
        --max-workers "${LMCACHE_MAX_WORKERS:-8}"
        --eviction-policy LRU
        --instance-id "${LMCACHE_INSTANCE_ID:-${host_name}-${SLURM_JOB_ID}}"
    )
    if [[ "$enable_p2p" == "true" ]]; then
        local coord_port="${LMCACHE_COORDINATOR_PORT:-9300}"
        server_args+=(
            --l1-align-bytes "${LMCACHE_L1_ALIGN_BYTES:-65536}"
            --coordinator-url "${LMCACHE_COORDINATOR_URL:-http://${NODE0_ADDR}:${coord_port}}"
            --coordinator-advertise-ip "${LMCACHE_COORDINATOR_ADVERTISE_IP:-${host_ip}}"
            --p2p-advertise-url "${LMCACHE_P2P_ADVERTISE_URL:-${rdma_ip}:${LMCACHE_P2P_PORT:-8500}}"
        )
        if [[ -n "${LMCACHE_P2P_LISTEN_URL:-}" ]]; then
            server_args+=(--p2p-listen-url "${LMCACHE_P2P_LISTEN_URL}")
        fi
    fi

    echo "[LMCacheMP] starting server on ${server_host}:${LMCACHE_PORT:-5555} (http-port ${LMCACHE_HTTP_PORT:-8080}) p2p=${enable_p2p}"
    lmcache server "${server_args[@]}" \
        > "$LMCACHE_LOG" 2>&1 &

    for _i in $(seq 1 120); do
        if curl -sf --max-time 3 "http://${LMCACHE_HOST:-127.0.0.1}:${LMCACHE_HTTP_PORT:-8080}/healthcheck" >/dev/null 2>&1; then
            echo "[LMCacheMP] server healthy"
            capture_lmcache_metrics "after_lmcache_start"
            return 0
        fi
        sleep 2
    done
    echo "ERROR: LMCache MP server failed to become healthy; tailing $LMCACHE_LOG" >&2
    tail -n 80 "$LMCACHE_LOG" >&2 || true
    return 1
}

dump_runtime_logs() {
    local label="${1:-runtime failure}"
    echo "ERROR: ${label}" >&2
    if [[ "$DRY_RUN" -eq 0 ]]; then
        local logs_output="${BENCHMARK_LOGS_DIR:-/run_logs}/logs"
        mkdir -p "$logs_output" 2>/dev/null || true
        cp -r "/run_logs/slurm_job-${SLURM_JOB_ID}" "$logs_output/" 2>/dev/null || \
            sudo cp -r "/run_logs/slurm_job-${SLURM_JOB_ID}" "$logs_output/" 2>/dev/null || true
        echo "==== staged runtime logs to ${logs_output}/slurm_job-${SLURM_JOB_ID} ====" >&2
    fi
    echo "==== /run_logs/slurm_job-${SLURM_JOB_ID} files ====" >&2
    find "/run_logs/slurm_job-${SLURM_JOB_ID}" -maxdepth 2 -type f -printf '%p %s bytes\n' 2>/dev/null | sort >&2 || true
    echo "==== recent server logs ====" >&2
    for _log in /run_logs/slurm_job-"${SLURM_JOB_ID}"/*.log; do
        [[ -f "$_log" ]] || continue
        echo "----- $_log -----" >&2
        tail -n 200 "$_log" >&2 || true
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
    start_lmcache_coordinator_if_needed || exit 1
    start_lmcache_mp_if_needed || exit 1

    # Router is started as an external container by job.slurm (VLLM_ROUTER_IMAGE)
    echo "Using external vllm-router container (started by job.slurm on this node)"

    SERVED_MODEL="${SERVED_MODEL_NAME:-${MODEL_NAME}}"
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
        if ! python3 $WS_PATH/sync.py barrier \
            --node-ips ${IPADDRS} \
            --node-ports $SERVER_PORT \
            --wait-for-all-ports \
            --timeout 1800; then
            dump_runtime_logs "prefill/decode server readiness barrier failed"
            [[ -n "${prefill_pid:-}" ]] && kill $prefill_pid 2>/dev/null || true
            exit 1
        fi
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
        if ! eval "$HEALTH_BARRIER_CMD"; then
            dump_runtime_logs "proxy health barrier failed"
            [[ -n "${prefill_pid:-}" ]] && kill $prefill_pid 2>/dev/null || true
            exit 1
        fi
        echo "MoRI-IO proxy is ready for benchmarking"
    fi

    echo "Ready for benchmarking on ${host_name}:${host_ip}"
    echo "Benchmarking on ${host_name}:${host_ip}"
    cd $WS_PATH

    export ROUTER_PORT=$ROUTER_PORT
    export AIPERF_SERVER_METRICS_URLS="${AIPERF_SERVER_METRICS_URLS:-$(build_vllm_metrics_urls)}"
    echo "AIPERF_SERVER_METRICS_URLS=${AIPERF_SERVER_METRICS_URLS}"
    capture_vllm_cache_metrics "before_benchmark"
    capture_lmcache_metrics "before_benchmark"

    if [[ "${EVAL_ONLY:-false}" == "true" ]]; then
        echo "EVAL_ONLY mode: skipping throughput benchmark"
    elif [[ "${IS_AGENTIC:-0}" == "1" || "${SCENARIO_TYPE:-}" == "agentic-coding" ]]; then
        AGENTIC_CMD='
            set -euo pipefail
            cd /workspace
            export AIPERF_RUNTIME_DIR="${AIPERF_RUNTIME_DIR:-/run_logs/slurm_job-'"${SLURM_JOB_ID}"'/aiperf_runtime}"
            source /workspace/benchmarks/benchmark_lib.sh

            export PORT='"${ROUTER_PORT}"'
            export MODEL="${SERVED_MODEL_NAME:-${MODEL:-${MODEL_NAME}}}"
            export MODEL_PREFIX="${MODEL_PREFIX:-'"${MODEL_NAME}"'}"
            export FRAMEWORK="${FRAMEWORK:-vllm-disagg}"
            export PRECISION="${PRECISION:-}"
            export KV_OFFLOADING="${KV_OFFLOADING:-none}"
            export DURATION="${DURATION:-1800}"
            export RESULT_FILENAME="${RESULT_FILENAME:-agentic_${MODEL_NAME}_${SLURM_JOB_ID}}"
            export AGENTIC_OUTPUT_DIR="/run_logs/slurm_job-'"${SLURM_JOB_ID}"'/agentic"
            export AIPERF_SERVER_METRICS_URLS="${AIPERF_SERVER_METRICS_URLS:-'"${AIPERF_SERVER_METRICS_URLS}"'}"

            IFS="x" read -r -a CONCURRENCIES <<< "${BENCH_MAX_CONCURRENCY:-1}"
            if [ "${#CONCURRENCIES[@]}" -eq 0 ]; then
                echo "ERROR: no agentic concurrency configured" >&2
                exit 1
            fi

            resolve_trace_source
            install_agentic_deps

            for concurrency in "${CONCURRENCIES[@]}"; do
                if ! [[ "$concurrency" =~ ^[1-9][0-9]*$ ]]; then
                    echo "ERROR: invalid agentic concurrency: $concurrency" >&2
                    exit 1
                fi
                export CONC="$concurrency"
                result_dir="/run_logs/slurm_job-'"${SLURM_JOB_ID}"'/agentic/conc_${concurrency}"
                mkdir -p "$result_dir"
                echo "Running vLLM disagg agentic replay at concurrency $concurrency"
                build_replay_cmd "$result_dir"
                run_agentic_replay_and_write_outputs "$result_dir"
            done
        '
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "DRY RUN: $AGENTIC_CMD"
        else
            set -x
            bash -lc "$AGENTIC_CMD"
            set +x
        fi
    elif [[ "$DRY_RUN" -eq 1 ]]; then
        BENCH_CMD="bash $WS_PATH/bench.sh ${xP} ${yD} $((PREFILL_TP_SIZE*xP)) $((DECODE_TP_SIZE*yD)) \
            $MODEL_DIR $MODEL_NAME /run_logs/slurm_job-${SLURM_JOB_ID} ${BENCH_INPUT_LEN} \
            ${BENCH_OUTPUT_LEN} \"${BENCH_MAX_CONCURRENCY}\" ${BENCH_REQUEST_RATE} \
            ${BENCH_RANDOM_RANGE_RATIO} ${BENCH_NUM_PROMPTS_MULTIPLIER}"
        echo "DRY RUN: $BENCH_CMD"
    else
        BENCH_CMD="bash $WS_PATH/bench.sh ${xP} ${yD} $((PREFILL_TP_SIZE*xP)) $((DECODE_TP_SIZE*yD)) \
            $MODEL_DIR $MODEL_NAME /run_logs/slurm_job-${SLURM_JOB_ID} ${BENCH_INPUT_LEN} \
            ${BENCH_OUTPUT_LEN} \"${BENCH_MAX_CONCURRENCY}\" ${BENCH_REQUEST_RATE} \
            ${BENCH_RANDOM_RANGE_RATIO} ${BENCH_NUM_PROMPTS_MULTIPLIER}"
        set -x
        eval "$BENCH_CMD"
        set +x
    fi

    capture_vllm_cache_metrics "after_benchmark"
    capture_lmcache_metrics "after_benchmark"

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
        capture_vllm_cache_metrics "after_eval"
        capture_lmcache_metrics "after_eval"
    fi

    # Stage workspace-level artifacts such as aggregated benchmark JSON and
    # AIPerf exports before the container exits.
    if [[ "$DRY_RUN" -eq 0 ]]; then
        WORKSPACE_ARTIFACT_DIR="/run_logs/slurm_job-${SLURM_JOB_ID}/workspace_artifacts"
        mkdir -p "$WORKSPACE_ARTIFACT_DIR"
        find /workspace -maxdepth 1 -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.png' \) \
            -exec cp -f {} "$WORKSPACE_ARTIFACT_DIR/" \; 2>/dev/null || true
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
    start_lmcache_coordinator_if_needed || exit 1
    start_lmcache_mp_if_needed || exit 1

    SERVED_MODEL="${SERVED_MODEL_NAME:-${MODEL_NAME}}"
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
        if ! eval "$BARRIER_CMD"; then
            dump_runtime_logs "proxy port barrier failed on additional prefill node"
            [[ -n "${prefill_pid:-}" ]] && kill $prefill_pid 2>/dev/null || true
            exit 1
        fi
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
    capture_lmcache_metrics "after_router_wait"

    echo "Killing the prefill server"
    [[ "$DRY_RUN" -eq 0 ]] && kill $prefill_pid 2>/dev/null || true

else
    echo "${host_name}:${host_ip} is Decode Node (Model: ${MODEL_NAME})"
    echo "Using decode config: $DECODE_SERVER_CONFIG"

    setup_vllm_env
    start_lmcache_coordinator_if_needed || exit 1
    start_lmcache_mp_if_needed || exit 1

    for env_pair in ${DECODE_MODEL_ENVS}; do
        export "$env_pair"
        echo "[DECODE_ENV] $env_pair"
    done

    SERVED_MODEL="${SERVED_MODEL_NAME:-${MODEL_NAME}}"
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
        if ! eval "$BARRIER_CMD"; then
            dump_runtime_logs "proxy port barrier failed on decode node"
            [[ -n "${decode_pid:-}" ]] && kill $decode_pid 2>/dev/null || true
            exit 1
        fi
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
