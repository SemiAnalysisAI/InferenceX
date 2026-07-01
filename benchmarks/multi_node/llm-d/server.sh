#!/usr/bin/env bash
#
# Per-node entrypoint for the llm-d-vllm wide-EP P/D disagg benchmark.
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
# Transport-selection logging on so the rank logs record the chosen wire.
export UCX_TLS=${UCX_TLS:-cuda_copy,cuda_ipc,rc}
export UCX_LOG_LEVEL=${UCX_LOG_LEVEL:-info}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET,ENV}

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
API_SERVER_COUNT="${LLMD_API_SERVER_COUNT:-4}"
if [[ "$ROLE_ENABLE_EP" == "true" ]] || [[ "$LWS_GROUP_SIZE" -le 1 ]] || [[ "$LWS_WORKER_INDEX" -eq 0 ]]; then
    COMMON_ARGS+=(--api-server-count "$API_SERVER_COUNT")
fi
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
    fi
fi

echo "Starting vLLM ($ROLE) DP=$DP_SIZE local=$DP_SIZE_LOCAL start_rank=$START_RANK group_size=$LWS_GROUP_SIZE"
# shellcheck disable=SC2086
vllm serve "$MODEL" "${COMMON_ARGS[@]}" $ROLE_EXTRA_ARGS \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# Each rank waits for its own engine /health before continuing (for wide-EP this
# blocks the bench until worker DP shards are up; a no-op for single-node).
wait_for_server_ready --port "$VLLM_PORT" --server-log "$VLLM_LOG" --server-pid "$VLLM_PID"
echo "vLLM ready on rank $NODE_RANK ($ROLE worker_index=$LWS_WORKER_INDEX)"

# ----------------------------------------------------------------
# Bring up pd-sidecar (every decode node)
# ----------------------------------------------------------------
# Each decode node runs its own sidecar (SIDECAR_PORT -> local decode vLLM), and
# endpoints.yaml lists one decode endpoint per node so EPP fans out across all
# decode ranks. The sidecar forwards a prefill request, reads kv_transfer_params
# from vLLM's response, then hits its local decode vLLM, whose NIXLv2 connector
# pulls KV directly from prefill vLLM.
if [[ "$ROLE" == "decode" ]]; then
    SIDECAR_CONNECTOR="nixlv2"
    SIDECAR_FLAGS=(--port="$SIDECAR_PORT" --vllm-port="$VLLM_PORT"
                   --kv-connector="$SIDECAR_CONNECTOR" --secure-proxy=false
                   --enable-prefiller-sampling)
    echo "Starting pd-sidecar (decode node_rank=$NODE_RANK worker_index=$LWS_WORKER_INDEX): ${SIDECAR_FLAGS[*]}"
    pd-sidecar "${SIDECAR_FLAGS[@]}" > "$SIDECAR_LOG" 2>&1 &
    SIDECAR_PID=$!
    wait_for_server_ready --port "$SIDECAR_PORT" --server-log "$SIDECAR_LOG" --server-pid "$SIDECAR_PID"
    echo "pd-sidecar ready on $HOST_IP:$SIDECAR_PORT"
fi

# ================================================================
# Coordinator (decode leader): endpoints, EPP, Envoy, bench, eval
# ================================================================
if [[ "$ROLE" == "decode" && "$LWS_WORKER_INDEX" -eq 0 ]]; then

    # ---- Write endpoints.yaml (file-discovery) ----
    # namespace must match EPP's --pool-namespace (file-discovery filters by it;
    # the schema default 'default' would drop every entry). See README.md.
    python3 - <<PY
import os, yaml
NS = 'inferencex'
all_ips = [x for x in os.environ.get('ALL_IPS', '').split(',') if x]
pn = int(os.environ.get('PREFILL_NODES', '1'))
dn = int(os.environ.get('DECODE_NODES', '1'))
# ALL_IPS is rank-ordered: ranks [0:pn] are prefill nodes, [pn:pn+dn] decode.
prefill_ips = all_ips[:pn] or [os.environ['PREFILL_LEADER_IP']]
decode_ips = all_ips[pn:pn + dn] or [os.environ['DECODE_LEADER_IP']]
endpoints = []
# One prefill endpoint per prefill node (its vLLM, VLLM_PORT). Each node's API
# server load-balances only its local DP ranks, so listing every node lets EPP
# fan requests across all prefill engines.
for i, ip in enumerate(prefill_ips):
    endpoints.append({'name': f'prefill-{i}', 'namespace': NS, 'address': ip,
                      'port': '$VLLM_PORT', 'labels': {'llm-d.ai/role': 'prefill'}})
# One decode endpoint per decode node (its pd-sidecar, SIDECAR_PORT), so EPP
# fans out across all decode ranks.
for i, ip in enumerate(decode_ips):
    endpoints.append({'name': f'decode-{i}', 'namespace': NS, 'address': ip,
                      'port': '$SIDECAR_PORT', 'labels': {'llm-d.ai/role': 'decode'}})
yaml.safe_dump({'endpoints': endpoints}, open('/tmp/endpoints.yaml', 'w'))
print('endpoints.yaml:')
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
        --v=4 \
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

    # ---- Gate on the prefill leader's vLLM /health (cross-node) ----
    # Prefill ranks wait on their own local /health; wait_for_server_ready only
    # probes localhost, so the decode leader polls the prefill leader here.
    echo "Waiting for prefill vLLM at $PREFILL_LEADER_IP:$VLLM_PORT/health"
    PREFILL_WAIT_DEADLINE=$(( $(date +%s) + 300 ))
    until curl --output /dev/null --silent --fail \
            "http://$PREFILL_LEADER_IP:$VLLM_PORT/health"; do
        if [[ "$(date +%s)" -ge "$PREFILL_WAIT_DEADLINE" ]]; then
            echo "ERROR: prefill vLLM did not become ready within 5 min" >&2
            exit 1
        fi
        sleep 5
    done
    echo "Prefill vLLM at $PREFILL_LEADER_IP:$VLLM_PORT is ready"

    # ---- Scrape engine /metrics (KV-transfer + cache telemetry) ----
    # benchmark_serving only reports client-side TTFT/TPOT; scrape the full
    # /metrics from the prefill leader + local decode across the bench window
    # (timestamped, per-conc markers) into BENCHMARK_LOGS_DIR for the tarball.
    METRICS_SCRAPE_INTERVAL_S="${METRICS_SCRAPE_INTERVAL_S:-5}"
    PREFILL_METRICS_FILE="$BENCHMARK_LOGS_DIR/${RESULT_FILENAME}_metrics_prefill.prom"
    DECODE_METRICS_FILE="$BENCHMARK_LOGS_DIR/${RESULT_FILENAME}_metrics_decode.prom"
    _scrape_metrics() {
        local url="$1" out="$2"
        while true; do
            { echo "# scrape_ts=$(date +%s)"
              curl --silent --fail --max-time 4 "$url" 2>/dev/null \
                  || echo "# scrape_failed ts=$(date +%s)"
            } >> "$out"
            sleep "$METRICS_SCRAPE_INTERVAL_S"
        done
    }
    _scrape_metrics "http://$PREFILL_LEADER_IP:$VLLM_PORT/metrics" "$PREFILL_METRICS_FILE" &
    PREFILL_SCRAPE_PID=$!
    _scrape_metrics "http://127.0.0.1:$VLLM_PORT/metrics" "$DECODE_METRICS_FILE" &
    DECODE_SCRAPE_PID=$!
    echo "[metrics] scraping prefill($PREFILL_LEADER_IP) + decode(local) /metrics every ${METRICS_SCRAPE_INTERVAL_S}s -> $BENCHMARK_LOGS_DIR"

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
        # Tag the scraped metrics lines with this conc level.
        for _mf in "$PREFILL_METRICS_FILE" "$DECODE_METRICS_FILE"; do
            echo "# ==== conc=$max_concurrency num_prompts=$num_prompts start_ts=$(date +%s) ====" >> "$_mf"
        done
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
            "${bench_extra_args[@]}"
    done

    # ---- Stop scrapers (don't scrape during eval) ----
    kill "$PREFILL_SCRAPE_PID" "$DECODE_SCRAPE_PID" 2>/dev/null || true
    echo "[metrics] stopped scrapers; captured ${RESULT_FILENAME}_metrics_{prefill,decode}.prom"

    # ---- Eval (optional) ----
    if [[ "${RUN_EVAL:-false}" == "true" ]]; then
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
