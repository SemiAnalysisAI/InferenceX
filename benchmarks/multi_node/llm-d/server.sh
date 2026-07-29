#!/usr/bin/env bash
#
# Per-node entrypoint for the llmd-vllm wide-EP P/D disagg benchmark.
# NODE_RANK is set by srun (= $SLURM_PROCID) in job.slurm.
#
# Roles:
#   Rank 0                         -> prefill leader (DP rank 0)
#   Ranks 1 .. PREFILL_NODES-1     -> prefill workers
#   Rank PREFILL_NODES             -> decode leader (DP rank 0) + pd-sidecar
#                                     + EPP + Envoy + benchmark client (coordinator)
#   Ranks PREFILL_NODES+1 ..       -> decode workers
#
# Each instance (prefill or decode) is one vLLM engine spanning its role's nodes
# via --data-parallel-hybrid-lb; the leader accepts traffic, workers serve their
# local DP ranks.

set -euo pipefail

source /workspace/benchmarks/benchmark_lib.sh

# ----------------------------------------------------------------
# Config + service ports
# ----------------------------------------------------------------
NODE_RANK="${NODE_RANK:-${SLURM_PROCID:-0}}"
PREFILL_NODES="${PREFILL_NODES:-1}"
DECODE_NODES="${DECODE_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
VLLM_PORT=8200
SIDECAR_PORT=8000
ENVOY_PORT=8080
EPP_GRPC_PORT=9002
EPP_HEALTH_PORT=9003
EPP_METRICS_PORT=9090

# Weights live at MODEL_DIR (/models, bind-mounted by job.slurm). MODEL_NAME is
# the served-model-name, not a filesystem path.
MODEL="${MODEL_DIR}"

# ----------------------------------------------------------------
# Host IP + default interface
# ----------------------------------------------------------------
# Resolved without iproute2 (`ip` is absent on the arm64 vLLM base); python3's
# socket layer exposes the kernel's source-IP / iface choice.
_HOST_INFO=$(python3 -c '
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("1.1.1.1", 80))
    ip = s.getsockname()[0]
finally:
    s.close()
iface = ""
try:
    with open("/proc/net/route") as f:
        f.readline()  # header
        for line in f:
            parts = line.split()
            if parts[1] == "00000000":  # default route dest
                iface = parts[0]; break
except OSError:
    pass
print(ip, iface)
' 2>/dev/null) || true
HOST_IP=$(echo "$_HOST_INFO" | awk '{print $1}')
DEFAULT_IFACE=$(echo "$_HOST_INFO" | awk '{print $2}')
DEFAULT_IFACE="${DEFAULT_IFACE:-eth0}"

VLLM_LOG="/benchmark_logs/vllm_rank${NODE_RANK}.log"
SIDECAR_LOG="/benchmark_logs/sidecar_rank${NODE_RANK}.log"
EPP_LOG="/benchmark_logs/epp.log"
ENVOY_LOG="/benchmark_logs/envoy.log"

echo "=== rank=$NODE_RANK host=$HOST_IP model=$MODEL ==="

# ----------------------------------------------------------------
# Role + topology (Option B engine grouping)
# ----------------------------------------------------------------
# A role's nodes split into PREFILL_WORKERS / DECODE_WORKERS independent DP/EP
# engines, each spanning (role_nodes / role_workers) nodes with its own DP
# coordinator (leader IP) and rank range. workers=1 => one engine over all role
# nodes (1P+1D / mid-curve); >1 => high-tpt (e.g. 2 prefill : 1 decode, DEP8 each).
PREFILL_WORKERS="${PREFILL_WORKERS:-1}"
DECODE_WORKERS="${DECODE_WORKERS:-1}"
IFS=',' read -r -a _ALL_IPS <<< "${ALL_IPS:-}"

if [[ "$NODE_RANK" -lt "$PREFILL_NODES" ]]; then
    ROLE="prefill"
    DP_SIZE="$PREFILL_DP_SIZE"
    _local_rank="$NODE_RANK"
    _nodes_per_worker=$(( PREFILL_NODES / PREFILL_WORKERS ))
    LWS_WORKER_INDEX=$(( _local_rank % _nodes_per_worker ))
    LWS_GROUP_SIZE="$_nodes_per_worker"
    _group_leader_rank=$(( (_local_rank / _nodes_per_worker) * _nodes_per_worker ))
elif [[ "$NODE_RANK" -lt $((PREFILL_NODES + DECODE_NODES)) ]]; then
    ROLE="decode"
    DP_SIZE="$DECODE_DP_SIZE"
    _local_rank=$(( NODE_RANK - PREFILL_NODES ))
    _nodes_per_worker=$(( DECODE_NODES / DECODE_WORKERS ))
    LWS_WORKER_INDEX=$(( _local_rank % _nodes_per_worker ))
    LWS_GROUP_SIZE="$_nodes_per_worker"
    _group_leader_rank=$(( PREFILL_NODES + (_local_rank / _nodes_per_worker) * _nodes_per_worker ))
else
    echo "ERROR: NODE_RANK=$NODE_RANK out of range" >&2
    exit 1
fi

# Each engine's DP coordinator = its leader node's IP (ALL_IPS[leader rank]);
# fall back to the role leader env when ALL_IPS is unset.
if [[ -n "${_ALL_IPS[${_group_leader_rank}]:-}" ]]; then
    DP_ADDR="${_ALL_IPS[${_group_leader_rank}]}"
elif [[ "$ROLE" == "prefill" ]]; then
    DP_ADDR="$PREFILL_DP_ADDR"
else
    DP_ADDR="$DECODE_DP_ADDR"
fi

DP_SIZE_LOCAL="$GPUS_PER_NODE"
START_RANK=$((LWS_WORKER_INDEX * DP_SIZE_LOCAL))

# Defaults: TP=1, DP=role_total, EP on (the H200 1P+1D shape). Recipe overrides below.
TP_SIZE=1
ROLE_ENABLE_EP=true

echo "ROLE=$ROLE DP_SIZE=$DP_SIZE DP_ADDR=$DP_ADDR LWS_WORKER_INDEX=$LWS_WORKER_INDEX START_RANK=$START_RANK"

# ----------------------------------------------------------------
# Recipe: per-role serve args + env (/etc/llmd-recipes/$CONFIG_FILE)
# ----------------------------------------------------------------
# Per-role keys: tp (int -> --tensor-parallel-size), enable-expert-parallel
# (bool -> --enable-expert-parallel + DP/wide-EP knobs), extra-args (appended
# verbatim), env (map, exported before vllm serve). Absent keys keep the
# defaults above, so a recipe with neither tp nor EP is a plain TP=1 DP+EP run.
ROLE_EXTRA_ARGS=""
if [[ -n "${CONFIG_FILE:-}" ]]; then
    RECIPE_PATH="/etc/llmd-recipes/${CONFIG_FILE}"
    if [[ -f "$RECIPE_PATH" ]]; then
        echo "Loading $ROLE recipe from $RECIPE_PATH"
        eval "$(python3 - <<PY
import yaml
recipe = yaml.safe_load(open('${RECIPE_PATH}'))
section = recipe.get('${ROLE}', {}) or {}
extra = (section.get('extra-args') or '').strip()
print(f'ROLE_EXTRA_ARGS={extra!r}')
tp = section.get('tp')
if tp is not None:
    print(f'TP_SIZE={int(tp)}')
ep = section.get('enable-expert-parallel')
if ep is not None:
    print(f'ROLE_ENABLE_EP={"true" if ep else "false"}')
for k, v in (section.get('env') or {}).items():
    print(f'export {k}={v!r}')
PY
)"
    else
        echo "WARNING: CONFIG_FILE=$CONFIG_FILE but $RECIPE_PATH not found; using defaults" >&2
    fi
fi
echo "Resolved $ROLE TP_SIZE=$TP_SIZE ROLE_ENABLE_EP=$ROLE_ENABLE_EP"

# ----------------------------------------------------------------
# Transport env (NCCL / UCX / NIXL), recipe-overridable
# ----------------------------------------------------------------
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$DEFAULT_IFACE}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-$DEFAULT_IFACE}
export VLLM_SKIP_P2P_CHECK=1
# Randomized DP dummy inputs make idle DP ranks fan their lockstep dummy passes
# across all experts (full MoE all-to-all), wasting prefill bandwidth; a recipe
# may set this to 0.
export VLLM_RANDOMIZE_DP_DUMMY_INPUTS=${VLLM_RANDOMIZE_DP_DUMMY_INPUTS:-1}
export VLLM_USE_DEEP_GEMM=1
# Cold-start budget for engine-core readiness. DSV4-Pro on GB200 cold-starts in
# ~9-11 min (weight load + DeepGEMM JIT warmup + cudagraph capture + NIXL/UCX
# handshake); the 600s vLLM default is too tight, so allow 30 min.
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-1800}
# DeepGEMM JIT links -l:libcuda.so.1 at warmup; the compat dir is on
# LD_LIBRARY_PATH (runtime) but not LIBRARY_PATH (link time). Prepend it, plus
# the arch-specific toolkit lib dir resolved from `uname -m`.
case "$(uname -m)" in
    aarch64|arm64) _NCT_LIB=/usr/lib/aarch64-linux-gnu ;;
    *)             _NCT_LIB=/usr/lib/x86_64-linux-gnu ;;
esac
export LIBRARY_PATH=/usr/local/cuda/compat:${_NCT_LIB}:${LIBRARY_PATH:-}
export VLLM_NIXL_SIDE_CHANNEL_HOST="$HOST_IP"
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}

# Pin NIXL/UCX to IB verbs (rc) so cross-node KV rides the IB HCAs (job.slurm
# exposes /dev/infiniband + IPC_LOCK); cuda_copy/cuda_ipc cover intra-node.
export UCX_TLS=${UCX_TLS:-cuda_copy,cuda_ipc,rc}

# ----------------------------------------------------------------
# Wide-EP NVSHMEM / ibgda env (only when an engine spans >1 node)
# ----------------------------------------------------------------
# Single-node-per-role recipes avoid DeepEP / NVSHMEM ibgda, so leave these off
# there to avoid triggering ibgda code paths that are not needed.
if [[ "$LWS_GROUP_SIZE" -gt 1 ]]; then
    export NVIDIA_GDRCOPY=enabled
    # ibgda default kept for future DeepEP/wide-EP recipes; a recipe may override
    # NVSHMEM_REMOTE_TRANSPORT to none.
    export NVSHMEM_REMOTE_TRANSPORT=${NVSHMEM_REMOTE_TRANSPORT:-ibgda}
    export NVSHMEM_IB_ENABLE_IBGDA=${NVSHMEM_IB_ENABLE_IBGDA:-true}
    export NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SYMMETRIC_SIZE:-16G}
    export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME:-$DEFAULT_IFACE}
    # NVSHMEM ignores NVSHMEM_HCA_PE_MAPPING when NVSHMEM_HCA_LIST is set, so
    # clear the latter when the recipe provides an explicit PE mapping.
    if [[ -n "${NVSHMEM_HCA_PE_MAPPING:-}" ]]; then
        unset NVSHMEM_HCA_LIST 2>/dev/null || true
    fi
fi

# ----------------------------------------------------------------
# Bring up vLLM engine (every node)
# ----------------------------------------------------------------
# KV role: prefill=producer, decode=consumer (override via KV_ROLE_OVERRIDE).
if [[ -n "${KV_ROLE_OVERRIDE:-}" ]]; then
    KV_ROLE="$KV_ROLE_OVERRIDE"
elif [[ "$ROLE" == "prefill" ]]; then
    KV_ROLE="kv_producer"
else
    KV_ROLE="kv_consumer"
fi
KV_TRANSFER_CONFIG="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"$KV_ROLE\",\"kv_load_failure_policy\":\"fail\"}"

COMMON_ARGS=(
    --port "$VLLM_PORT"
    --served-model-name "$MODEL_NAME"
    --trust-remote-code
    --disable-access-log-for-endpoints=/health,/metrics
    --tensor-parallel-size "$TP_SIZE"
    --kv_transfer_config "$KV_TRANSFER_CONFIG"
)
# A single frontend (HTTP + tokenize + DP load-balance) is CPU-bound and caps
# throughput, so run several. Incompatible with --headless, so it is the one
# flag the headless-worker branch below drops. Overridable via LLMD_API_SERVER_COUNT.
# LB is hybrid: --data-parallel-hybrid-lb; one api-server per node internally
# load-balances its local DP ranks -> ONE serving port (VLLM_PORT) per node, so
# the local rank-0 health port is always VLLM_PORT.
HEALTH_PORT="$VLLM_PORT"
API_SERVER_COUNT="${LLMD_API_SERVER_COUNT:-4}"
# Multiple frontends only help the DP (wide-EP) path, where they load-balance
# across the node's local DP ranks. A pure-TP engine has a single core with one
# frontend, so it keeps the default count (also avoids --api-server-count
# interacting with the --headless multi-node TP launch below). Every DEP8 node
# gets it; pure-TP nodes get none.
if [[ "$ROLE_ENABLE_EP" == "true" ]]; then
    COMMON_ARGS+=(--api-server-count "$API_SERVER_COUNT")
fi
# Set to 1 by the pure-TP multi-node branch below on --headless followers, which
# run no local api-server; gates the post-launch health wait.
IS_HEADLESS_FOLLOWER=0
# --moe-backend is model-specific (DSR1-FP8 wants deep_gemm, gpt-oss-MXFP4
# rejects it), so each recipe sets its own via extra-args.

# EP/DP knobs only when the recipe enables EP. Pure tensor-parallel roles skip
# them (vLLM rejects --data-parallel-size combined with TP>1).
if [[ "$ROLE_ENABLE_EP" == "true" ]]; then
    COMMON_ARGS+=(
        --enable-expert-parallel
        --data-parallel-size "$DP_SIZE"
    )
    if [[ "$LWS_GROUP_SIZE" -gt 1 ]]; then
        COMMON_ARGS+=(
            --data-parallel-hybrid-lb
            --data-parallel-size-local "$DP_SIZE_LOCAL"
            --data-parallel-address "$DP_ADDR"
            --data-parallel-rpc-port 5555
            --data-parallel-start-rank "$START_RANK"
        )
    fi
elif [[ "$LWS_GROUP_SIZE" -gt 1 ]]; then
    # Pure TP spanning >1 node (e.g. DSV4-Pro decode TP=8 on GB200's 4-GPU
    # nodes): use vLLM's headless multi-node API - leader binds --master-addr,
    # followers join --headless with matching --nnodes/--node-rank.
    COMMON_ARGS+=(
        --master-addr "$DP_ADDR"
        --nnodes "$LWS_GROUP_SIZE"
        --node-rank "$LWS_WORKER_INDEX"
    )
    if [[ "$LWS_WORKER_INDEX" -gt 0 ]]; then
        COMMON_ARGS+=(--headless)
        IS_HEADLESS_FOLLOWER=1
    fi
fi

echo "Starting vLLM ($ROLE) DP=$DP_SIZE local=$DP_SIZE_LOCAL start_rank=$START_RANK group_size=$LWS_GROUP_SIZE"
# shellcheck disable=SC2086
vllm serve "$MODEL" "${COMMON_ARGS[@]}" $ROLE_EXTRA_ARGS \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# Each rank waits for its own engine /health before continuing (for wide-EP this
# blocks the bench until worker DP shards are up; a no-op for single-node). A
# pure-TP --headless follower runs no local api-server (only the TP-group leader
# binds a health port, and its /health only reports ready once every TP worker
# has joined), so it skips the wait and stays alive via the final `wait`.
if [[ "$IS_HEADLESS_FOLLOWER" -eq 1 ]]; then
    echo "vLLM headless TP follower on rank $NODE_RANK (worker_index=$LWS_WORKER_INDEX): no local api-server, skipping health wait"
else
    wait_for_server_ready --port "$HEALTH_PORT" --server-log "$VLLM_LOG" --server-pid "$VLLM_PID"
    echo "vLLM ready on rank $NODE_RANK ($ROLE worker_index=$LWS_WORKER_INDEX, health port $HEALTH_PORT)"
fi

# ----------------------------------------------------------------
# Bring up pd-sidecar (every decode node)
# ----------------------------------------------------------------
# The sidecar forwards a prefill request, reads kv_transfer_params from vLLM's
# response, then hits its local decode vLLM, whose NIXLv2 connector pulls KV
# directly from prefill vLLM.
#
# DEP8 (EP on, hybrid-LB): every decode node runs an api-server for its local DP
# ranks, so every decode node runs a sidecar and endpoints.yaml lists one decode
# endpoint per node. Pure-TP: only the TP-group leader has an api-server
# (followers are --headless), so only the leader runs a sidecar and only leaders
# are listed as endpoints.
if [[ "$ROLE" == "decode" && ( "$ROLE_ENABLE_EP" == "true" || "$LWS_WORKER_INDEX" -eq 0 ) ]]; then
    SIDECAR_CONNECTOR="nixlv2"
    SIDECAR_FLAGS=(--port="$SIDECAR_PORT" --vllm-port="$VLLM_PORT"
                   --kv-connector="$SIDECAR_CONNECTOR" --secure-proxy=false
                   --enable-prefiller-sampling)
    SIDECAR_HEALTH_PORT="$SIDECAR_PORT"
    echo "Starting pd-sidecar (decode node_rank=$NODE_RANK worker_index=$LWS_WORKER_INDEX): ${SIDECAR_FLAGS[*]}"
    pd-sidecar "${SIDECAR_FLAGS[@]}" > "$SIDECAR_LOG" 2>&1 &
    SIDECAR_PID=$!
    wait_for_server_ready --port "$SIDECAR_HEALTH_PORT" --server-log "$SIDECAR_LOG" --server-pid "$SIDECAR_PID"
    echo "pd-sidecar ready on $HOST_IP:$SIDECAR_HEALTH_PORT"
fi

# ================================================================
# Coordinator (decode leader): endpoints, EPP, Envoy, bench, eval
# ================================================================
if [[ "$ROLE" == "decode" && "$LWS_WORKER_INDEX" -eq 0 ]]; then

    # Release the allocation whenever the coordinator exits.
    BENCH_DONE_MARKER="$BENCHMARK_LOGS_DIR/.bench_done.$SLURM_JOB_ID"
    trap 'touch "$BENCH_DONE_MARKER" 2>/dev/null || true' EXIT

    # ---- Write endpoints.yaml (file-discovery) ----
    # namespace must match EPP's --pool-namespace (file-discovery filters by it;
    # the schema default 'default' would drop every entry). See README.md.
    python3 - <<PY
import os, yaml
NS = 'inferencex'
all_ips = [x for x in os.environ.get('ALL_IPS', '').split(',') if x]
pn = int(os.environ.get('PREFILL_NODES', '1'))
dn = int(os.environ.get('DECODE_NODES', '1'))
decode_workers = max(1, int('$DECODE_WORKERS'))
# This block runs on the coordinator (a decode node), so ROLE_ENABLE_EP here
# reflects the DECODE role. EP on => DEP8 hybrid-LB (an api-server per node);
# EP off => pure-TP (only each TP-group leader has an api-server).
decode_ep = ('$ROLE_ENABLE_EP' == 'true')
VLLM_PORT = int('$VLLM_PORT')
SIDECAR_PORT = int('$SIDECAR_PORT')
# ALL_IPS is rank-ordered: ranks [0:pn] are prefill nodes, [pn:pn+dn] decode.
prefill_ips = all_ips[:pn] or [os.environ['PREFILL_LEADER_IP']]
decode_ips = all_ips[pn:pn + dn] or [os.environ['DECODE_LEADER_IP']]
endpoints = []

def add_role(role, ips, base_port, group_size=1):
    # group_size == 1: one endpoint per node (DEP8 hybrid-LB: each node's
    # api-server / sidecar load-balances its local DP ranks).
    # group_size  > 1: one endpoint per TP-group leader (pure-TP: followers are
    # --headless with no api-server), i.e. every group_size-th node IP.
    serving_ips = ips[::group_size] if group_size > 1 else ips
    for i, ip in enumerate(serving_ips):
        endpoints.append({'name': f'{role}-{i}', 'namespace': NS, 'address': ip,
                          'port': str(base_port), 'labels': {'llm-d.ai/role': role}})

# Prefill (DEP8 in every current recipe): one endpoint per node, EPP hits vLLM
# directly (VLLM_PORT). Decode: EPP hits the pd-sidecar (SIDECAR_PORT); one
# endpoint per node for DEP8, or one per TP-group leader for pure-TP.
add_role('prefill', prefill_ips, VLLM_PORT)
decode_group = 1 if decode_ep else max(1, dn // decode_workers)
add_role('decode', decode_ips, SIDECAR_PORT, group_size=decode_group)
yaml.safe_dump({'endpoints': endpoints}, open('/tmp/endpoints.yaml', 'w'))
print(f'endpoints.yaml ({len(endpoints)} endpoints):')
print(open('/tmp/endpoints.yaml').read())
PY

    # ---- Bring up EPP ----
    # Config: when a recipe is set, project it down to the keys EPP's strict
    # decoder accepts (it rejects the per-role vLLM / slurm keys); else use the
    # default mounted at /etc/epp/config.yaml.
    if [[ -n "$CONFIG_FILE" && -f "/etc/llmd-recipes/$CONFIG_FILE" ]]; then
        EPP_CONFIG="/tmp/epp-config-from-recipe.yaml"
        python3 - <<PY
import yaml
recipe = yaml.safe_load(open('/etc/llmd-recipes/${CONFIG_FILE}'))
keep = {'apiVersion', 'kind', 'plugins', 'schedulingProfiles', 'dataLayer'}
yaml.safe_dump({k: v for k, v in recipe.items() if k in keep},
               open('${EPP_CONFIG}', 'w'))
PY
    else
        EPP_CONFIG="/etc/epp/config.yaml"
    fi
    echo "EPP config: $EPP_CONFIG"

    # --secure-serving=false: Envoy's epp cluster is plaintext HTTP/2; EPP's TLS
    # default would fail every ext_proc dial (-> ext_proc trips, Envoy 500s).
    epp \
        --pool-name=epp \
        --pool-namespace=inferencex \
        --config-file="$EPP_CONFIG" \
        --grpc-port="$EPP_GRPC_PORT" \
        --grpc-health-port="$EPP_HEALTH_PORT" \
        --metrics-port="$EPP_METRICS_PORT" \
        --secure-serving=false \
        > "$EPP_LOG" 2>&1 &
    EPP_PID=$!

    # Wait for EPP's gRPC listener before starting Envoy (Envoy's ext_proc dials
    # it). gRPC has no plain HTTP /health, so probe the TCP listener directly.
    echo "Waiting for EPP on 127.0.0.1:$EPP_GRPC_PORT"
    EPP_WAIT_DEADLINE=$(( $(date +%s) + 60 ))
    until (echo > "/dev/tcp/127.0.0.1/$EPP_GRPC_PORT") 2>/dev/null; do
        if ! kill -0 "$EPP_PID" 2>/dev/null; then
            echo "ERROR: EPP died before binding $EPP_GRPC_PORT" >&2
            exit 1
        fi
        if [[ "$(date +%s)" -ge "$EPP_WAIT_DEADLINE" ]]; then
            echo "ERROR: EPP did not bind $EPP_GRPC_PORT within 60s" >&2
            exit 1
        fi
        sleep 1
    done
    echo "EPP listening on $EPP_GRPC_PORT"

    # ---- Bring up Envoy ----
    envoy -c /etc/envoy/envoy.yaml > "$ENVOY_LOG" 2>&1 &
    ENVOY_PID=$!

    # Probe admin /ready (9901); /health on :8080 routes through ext_proc -> EPP
    # and needs request routing metadata, so it would 503 until traffic flows.
    echo "Waiting for envoy admin on 127.0.0.1:9901/ready"
    ENVOY_WAIT_DEADLINE=$(( $(date +%s) + 120 ))
    until [[ "$(curl --output /dev/null --silent --write-out '%{http_code}' \
                "http://127.0.0.1:9901/ready" 2>/dev/null)" == "200" ]]; do
        if ! kill -0 "$ENVOY_PID" 2>/dev/null; then
            echo "ERROR: envoy died before admin /ready returned 200" >&2
            tail -n 80 "$ENVOY_LOG" >&2 || true
            exit 1
        fi
        if [[ "$(date +%s)" -ge "$ENVOY_WAIT_DEADLINE" ]]; then
            echo "ERROR: envoy admin /ready did not return 200 within 120s" >&2
            tail -n 80 "$ENVOY_LOG" >&2 || true
            exit 1
        fi
        sleep 2
    done
    echo "Envoy admin ready; listener should be on $ENVOY_PORT"

    # ---- Gate on ALL prefill vLLM /health endpoints (cross-node) ----
    # Prefill ranks wait on their own local /health; wait_for_server_ready only
    # probes localhost, so the decode leader polls the prefill nodes here.
    # endpoints.yaml lists one prefill endpoint per node, so with PREFILL_WORKERS>1
    # (multiple independent DP engines) EVERY prefill node must be probed, not just
    # IPS[0]. curl gets an explicit connect/max timeout so a blackholed endpoint
    # trips the deadline instead of hanging the whole run (a single timeout-less
    # curl once wedged a 2P run for 7h before it was cancelled).
    _prefill_ips=( "${_ALL_IPS[@]:0:${PREFILL_NODES}}" )
    [[ ${#_prefill_ips[@]} -gt 0 ]] || _prefill_ips=( "$PREFILL_LEADER_IP" )

    # On failure, dump enough to tell a server-not-ready problem (TCP connects but
    # /health is slow) apart from a network/subnet problem (TCP connect refused or
    # times out, host unreachable). Uses only bash builtins + coreutils: the arm64
    # serving image has no iproute2 / nc.
    _diag_prefill_endpoint() {
        local ip="$1" port="$2"
        {
            echo "=== NET DIAG: decode -> prefill ${ip}:${port} ==="
            echo "[diag] decode node: $(hostname -f 2>/dev/null || hostname) local-ips: $(hostname -I 2>/dev/null)"
            echo "[diag] ifaces: DEFAULT_IFACE=${DEFAULT_IFACE:-} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-}"
            # Local source address the kernel would pick to reach ip: reveals which
            # subnet/interface the route uses, without needing iproute2.
            python3 - "$ip" <<'PY' 2>&1 || true
import socket, struct, sys
ip = sys.argv[1]
# Source address the kernel picks to reach ip (which local iface/subnet).
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((ip, 80))
    print(f"[diag] local source IP toward {ip}: {s.getsockname()[0]}")
    s.close()
except Exception as e:
    print(f"[diag] no route to {ip}: {e}")
# Subnet check via /proc/net/route (no iproute2 needed). Fields are little-endian
# hex. On-link route (gateway 0.0.0.0) == same subnet; a non-zero gateway means the
# target is reached across a router == DIFFERENT subnet.
def _ntoa(v):  # little-endian int -> dotted quad
    return socket.inet_ntoa(struct.pack('<L', v))
try:
    tgt = struct.unpack('<L', socket.inet_aton(ip))[0]
    best = None
    with open('/proc/net/route') as f:
        next(f)
        for line in f:
            p = line.split()
            if len(p) < 8:
                continue
            iface, dest, gw, mask = p[0], int(p[1], 16), int(p[2], 16), int(p[7], 16)
            if (tgt & mask) == (dest & mask):
                plen = bin(mask).count('1')
                if best is None or plen > best[0]:
                    best = (plen, iface, gw, dest)
    if best is None:
        print(f"[diag] {ip}: NO matching route -> unreachable")
    else:
        plen, iface, gw, dest = best
        if gw == 0:
            print(f"[diag] {ip}: ON-LINK via {iface}, subnet {_ntoa(dest)}/{plen} -> SAME subnet (no router hop)")
        else:
            print(f"[diag] {ip}: via GATEWAY {_ntoa(gw)} on {iface}, subnet {_ntoa(dest)}/{plen} -> DIFFERENT subnet (crosses a router)")
except Exception as e:
    print(f"[diag] subnet classification failed: {e}")
PY
            # L4: raw TCP connect (bash /dev/tcp, 5s cap) - port reachable-but-slow
            # vs unreachable/filtered.
            if timeout 5 bash -c "exec 3<>/dev/tcp/${ip}/${port}" 2>/dev/null; then
                echo "[diag] TCP connect ${ip}:${port} OK -> port open; server up but /health slow/not-ready (NOT a network issue)"
            else
                echo "[diag] TCP connect ${ip}:${port} FAILED/timed out -> closed, filtered, or unreachable (LIKELY network/subnet/firewall issue)"
            fi
            # L3: ICMP reachability, if ping is present.
            if command -v ping >/dev/null 2>&1; then
                ping -c 2 -W 2 "$ip" 2>&1 || echo "[diag] ping ${ip} failed (ICMP blocked or host down)"
            fi
            # Verbose HTTP connect detail (DNS/connect/TLS timing, HTTP status).
            curl -v --connect-timeout 5 --max-time 8 "http://${ip}:${port}/health" 2>&1 || true
            echo "=== END NET DIAG ${ip}:${port} ==="
        } >&2
    }

    # Log the decode->prefill target layout up front so a subnet/interface
    # mismatch is visible even on a run that eventually succeeds. Every prefill
    # node serves on VLLM_PORT (hybrid LB).
    echo "[diag] decode-leader $(hostname 2>/dev/null) local-ips: $(hostname -I 2>/dev/null); prefill targets: ${_prefill_ips[*]}"
    echo "Waiting for prefill vLLM /health on ${#_prefill_ips[@]} node(s): ${_prefill_ips[*]}"
    PREFILL_WAIT_DEADLINE=$(( $(date +%s) + 300 ))
    for _pidx in "${!_prefill_ips[@]}"; do
        _pip="${_prefill_ips[$_pidx]}"
        _pport="$VLLM_PORT"
        until curl --output /dev/null --silent --fail \
                --connect-timeout 5 --max-time 10 \
                "http://$_pip:$_pport/health"; do
            if [[ "$(date +%s)" -ge "$PREFILL_WAIT_DEADLINE" ]]; then
                echo "ERROR: prefill vLLM at $_pip:$_pport not ready within 5 min" >&2
                _diag_prefill_endpoint "$_pip" "$_pport"
                exit 1
            fi
            sleep 5
        done
        echo "Prefill vLLM at $_pip:$_pport is ready"
    done
    echo "All ${#_prefill_ips[@]} prefill vLLM endpoint(s) ready"

    # ---- Benchmark sweep (one run per concurrency level) ----
    # BENCH_MAX_CONCURRENCY is an 'x'-delimited list from submit.sh (e.g. "1024x512").
    IFS='x' read -r -a CONCURRENCIES <<< "$BENCH_MAX_CONCURRENCY"
    # GPU counts embedded in the result filename as _gpus_/_ctx_/_gen_ tokens so the
    # CI "Process result" step (benchmark-multinode-tmpl.yml) can parse them and run
    # process_result.py for llm-d -- same filename convention as amd_utils/bench.sh.
    # ctx = prefill GPUs, gen = decode GPUs; nodes*GPUS_PER_NODE is correct for any
    # PREFILL_WORKERS/DECODE_WORKERS split (e.g. high-tpt 2P -> 16 prefill GPUs).
    _bench_prefill_gpus=$(( PREFILL_NODES * GPUS_PER_NODE ))
    _bench_decode_gpus=$(( DECODE_NODES * GPUS_PER_NODE ))
    _bench_total_gpus=$(( _bench_prefill_gpus + _bench_decode_gpus ))
    for max_concurrency in "${CONCURRENCIES[@]}"; do
        num_prompts=$(( max_concurrency * BENCH_NUM_PROMPTS_MULTIPLIER ))
        [[ "$num_prompts" -lt 16 ]] && num_prompts=16
        # Bench against Envoy (EPP routes to decode; the sidecar pulls from
        # prefill via NIXL). --bench-serving-dir = the /workspace repo bind-mount;
        # --tokenizer = /models (served-model-name is not a valid HF repo id).
        # DSV4-Pro needs trust-remote-code + tokenizer-mode deepseek_v4 (the older
        # transformers wheel does not register it) + chat template / --dsv4 to
        # match the dynamo-vllm bench prompt formatting.
        bench_extra_args=()
        if [[ "${MODEL_NAME,,}" == *"deepseek-v4"* ]]; then
            bench_extra_args+=(
                --trust-remote-code
                --tokenizer-mode deepseek_v4
                --use-chat-template
                --dsv4
            )
        fi

        # Non-fatal: a failed or timed-out conc point must not abort the sweep
        # or (under set -e) skip the allocation release below. The EXIT trap
        # releases the allocation regardless, but continuing here lets a
        # multi-conc sweep record every point it can.
        run_benchmark_serving \
            --bench-serving-dir /workspace \
            --tokenizer /models \
            --model "$MODEL_NAME" \
            --port "$ENVOY_PORT" \
            --backend openai \
            --input-len "$BENCH_INPUT_LEN" \
            --output-len "$BENCH_OUTPUT_LEN" \
            --random-range-ratio "$BENCH_RANDOM_RANGE_RATIO" \
            --num-prompts "$num_prompts" \
            --max-concurrency "$max_concurrency" \
            --result-filename "${RESULT_FILENAME}_c${max_concurrency}_gpus_${_bench_total_gpus}_ctx_${_bench_prefill_gpus}_gen_${_bench_decode_gpus}" \
            --result-dir "$BENCHMARK_LOGS_DIR/" \
            "${bench_extra_args[@]}" \
            || echo "WARNING: benchmark conc=$max_concurrency failed/timed out (rc=$?)"
    done

    # ---- Eval (optional) ----
    if [[ "${RUN_EVAL:-false}" == "true" ]]; then
        # Concurrency for the eval and, crucially, for the concurrency stamped
        # into meta_env.json. run_eval/append_lm_eval_summary read
        # EVAL_CONCURRENT_REQUESTS and CONC (not EVAL_CONC), so mirror the AMD
        # multi-node servers: use the workflow-provided EVAL_CONC when set, else
        # fall back to the max of the (x-delimited) BENCH_MAX_CONCURRENCY list.
        # Exporting CONC makes meta_env.json's "conc" match what
        # utils/evals/validate_scores.py --expected-concs verifies; without it
        # CONC is empty, the metadata records conc=1, and score verification
        # fails ("eval metadata concurrency does not match workflow request")
        # even when accuracy passes.
        if [[ -n "${EVAL_CONC:-}" ]]; then
            export EVAL_CONCURRENT_REQUESTS="${EVAL_CONC}"
        else
            export EVAL_CONCURRENT_REQUESTS=$(printf '%s' "$BENCH_MAX_CONCURRENCY" | tr 'x' '\n' | sort -n | tail -1)
        fi
        export CONC="${EVAL_CONCURRENT_REQUESTS}"
        # Run from /workspace (the repo bind-mount) so results*.json land where
        # the host-side workflow checks look; the subshell keeps the cd local.
        (
            cd /workspace
            run_eval --framework lm-eval --port "$ENVOY_PORT"
            append_lm_eval_summary
        )
    fi

    # Signal job.slurm (outside the container, where scancel exists) to release
    # the allocation; without it workers wait until TIME_LIMIT.
    touch "$BENCHMARK_LOGS_DIR/.bench_done.$SLURM_JOB_ID"
else
    # Workers (prefill leader, prefill/decode workers): keep vLLM alive.
    wait
fi
