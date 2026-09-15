#!/bin/bash
# SGLang Disaggregated Server Launcher with Model-Specific Configurations

NODE0_ADDR="${NODE0_ADDR:-localhost}"
NODE_RANK="${NODE_RANK:-0}"
MODEL_DIR="${MODEL_DIR:-}"
MODEL_NAME="${MODEL_NAME:-}"

xP="${xP:-1}" #-> Number of Prefill Workers
yD="${yD:-1}" #-> Number of Decode Workers

IPADDRS="${IPADDRS:-localhost}"
HEADNODE_PORT="${HEADNODE_PORT:-20000}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-8}"
PREFILL_ENABLE_EP="${PREFILL_ENABLE_EP:-true}"
PREFILL_ENABLE_DP="${PREFILL_ENABLE_DP:-true}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-8}"
DECODE_ENABLE_EP="${DECODE_ENABLE_EP:-true}"
DECODE_ENABLE_DP="${DECODE_ENABLE_DP:-true}"
DECODE_MTP_SIZE="${DECODE_MTP_SIZE:-0}"

BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-1024}"
BENCH_RANDOM_RANGE_RATIO="${BENCH_RANDOM_RANGE_RATIO:-1}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-inf}"
BENCH_NUM_PROMPTS_MULTIPLIER="${BENCH_NUM_PROMPTS_MULTIPLIER:-10}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-512}"

BENCH_MAX_CONC_VALUE=$(echo "$BENCH_MAX_CONCURRENCY" | tr 'x' '\n' | sort -n | tail -1)

DRY_RUN="${DRY_RUN:-0}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"


source $SGLANG_WS_PATH/setup_deps.sh
source $SGLANG_WS_PATH/env.sh

host_ip=$(ip route get 1.1.1.1 | awk '/src/ {print $7}')
host_name=$(hostname)

if [[ -n "${MORI_RDMA_TC}" ]]; then
    echo "[INFO] Using MORI_RDMA_TC=$MORI_RDMA_TC for RDMA traffic class configuration"
    echo "[INFO] Host '$host_name' configured with MORI_RDMA_TC=$MORI_RDMA_TC"
else
    echo "[INFO] MORI_RDMA_TC not set. Skipping RDMA traffic class configuration."
    echo "[INFO] This is normal for clusters without QoS requirements."
fi

# Model-specific configuration from models.yaml
MODELS_YAML="${SGLANG_WS_PATH}/models.yaml"

if [[ ! -f "$MODELS_YAML" ]]; then
    echo "ERROR: models.yaml not found at $MODELS_YAML"
    exit 1
fi

# Formula evaluation (e.g. "SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK * TP * xP") is
# done in Python so bash does not glob-expand the * characters.
eval "$(python3 -c "
import yaml, sys, os

config_path = '${MODELS_YAML}'
model_name = '${MODEL_NAME}'

# Select the models.yaml recipe variant by run type: agentic runs (IS_AGENTIC)
# use the '<model>-AgentX' entry, non-agentic disaggregated runs use '<model>-DI'.
# Fall back to the bare model name if the variant-specific key is absent.
is_agentic = '${IS_AGENTIC:-0}'.strip().lower() in ('1', 'true')
model_key = f'{model_name}-AgentX' if is_agentic else f'{model_name}-DI'

with open(config_path) as f:
    models = yaml.safe_load(f)

if model_key not in models:
    if model_name in models:
        model_key = model_name
    else:
        print(f'echo \"ERROR: Model {model_key} not in models.yaml\"; exit 1')
        sys.exit(0)

m = models[model_key]
print(f'echo \"Selected models.yaml entry: {model_key} (IS_AGENTIC={is_agentic})\"')

def eval_formula(val):
    \"\"\"Evaluate chunked_prefill_size: if string, resolve variable names from env and compute.\"\"\"
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val)
    # Build a namespace from env vars (convert numeric values to int)
    ns = {}
    for k, v in os.environ.items():
        try:
            ns[k] = int(v)
        except (ValueError, TypeError):
            pass
    try:
        return int(eval(s, {'__builtins__': {}}, ns))
    except Exception as e:
        print(f'echo \"WARNING: Cannot evaluate formula: {s} ({e})\"', file=sys.stderr)
        return val

def parse_range(cuda_range, default_start, default_end):
    if '-' in str(cuda_range):
        s, e = str(cuda_range).split('-')
        return s, e
    return str(default_start), str(default_end)

# Output shell variables
print(f'MODEL_BASE_FLAGS=\"{m.get(\"base_flags\", \"\")}\"')
print(f'MODEL_MTP_FLAGS=\"{m.get(\"mtp_flags\", \"\")}\"')
print(f'MODEL_DP_FLAGS=\"{m.get(\"dp_flags\", \"\")}\"')
print(f'MODEL_EP_FLAGS=\"{m.get(\"ep_flags\", \"\")}\"')

prefill = m.get('prefill', {})
decode = m.get('decode', {})

print(f'PREFILL_MEM_FRACTION_STATIC=\"{prefill.get(\"mem_fraction_static\", 0.8)}\"')
print(f'PREFILL_DISABLE_RADIX_CACHE=\"{prefill.get(\"disable_radix_cache\", True)}\"')
print(f'PREFILL_DISABLE_CUDA_GRAPH=\"{prefill.get(\"disable_cuda_graph\", False)}\"')

dp = prefill.get('dp', {})
no_dp = prefill.get('no_dp', {})
print(f'PREFILL_MAX_RUNNING_REQUESTS_DP=\"{dp.get(\"max_running_requests\", 24)}\"')
print(f'PREFILL_CHUNKED_PREFILL_SIZE_DP=\"{eval_formula(dp.get(\"chunked_prefill_size\", 262144))}\"')
print(f'PREFILL_CUDA_GRAPH_BS_DP=\"{dp.get(\"cuda_graph_bs\", \"1 2 3\")}\"')
print(f'PREFILL_CONTEXT_LENGTH_DP=\"{dp.get(\"context_length\", \"\")}\"')
print(f'PREFILL_MAX_TOTAL_TOKENS_DP=\"{dp.get(\"max_total_tokens\", \"\")}\"')
print(f'PREFILL_ENABLE_TWO_BATCH_OVERLAP_DP=\"{dp.get(\"enable_two_batch_overlap\", False)}\"')
print(f'PREFILL_MAX_RUNNING_REQUESTS_NO_DP=\"{no_dp.get(\"max_running_requests\", 128)}\"')
print(f'PREFILL_CHUNKED_PREFILL_SIZE_NO_DP=\"{eval_formula(no_dp.get(\"chunked_prefill_size\", 262144))}\"')
print(f'PREFILL_CONTEXT_LENGTH_NO_DP=\"{no_dp.get(\"context_length\", \"\")}\"')
print(f'PREFILL_MAX_TOTAL_TOKENS_NO_DP=\"{no_dp.get(\"max_total_tokens\", \"\")}\"')
s, e = parse_range(no_dp.get('cuda_graph_bs_range', '1-128'), 1, 128)
print(f'PREFILL_CUDA_GRAPH_BS_NO_DP_START=\"{s}\"')
print(f'PREFILL_CUDA_GRAPH_BS_NO_DP_END=\"{e}\"')

print(f'DECODE_MEM_FRACTION_STATIC=\"{decode.get(\"mem_fraction_static\", 0.85)}\"')
print(f'DECODE_PREFILL_ROUND_ROBIN_BALANCE=\"{decode.get(\"prefill_round_robin_balance\", True)}\"')
print(f'DECODE_DISAGG_ENABLE_RADIX_CACHE=\"{decode.get(\"disagg_decode_enable_radix_cache\", False)}\"')

dp = decode.get('dp', {})
ep_only = decode.get('ep_only', {})
no_dp = decode.get('no_dp', {})

# Decode DP config
print(f'DECODE_MAX_RUNNING_REQUESTS_DP=\"{dp.get(\"max_running_requests\", 4096)}\"')
print(f'DECODE_CHUNKED_PREFILL_SIZE_DP=\"{eval_formula(dp.get(\"chunked_prefill_size\", 262144))}\"')
print(f'DECODE_CONTEXT_LENGTH_DP=\"{dp.get(\"context_length\", \"\")}\"')
s, e = parse_range(dp.get('cuda_graph_bs_range', '1-160'), 1, 160)
print(f'DECODE_CUDA_GRAPH_BS_DP_START=\"{s}\"')
print(f'DECODE_CUDA_GRAPH_BS_DP_END=\"{e}\"')

# Decode EP-only config (EP enabled but DP disabled)
print(f'DECODE_MAX_RUNNING_REQUESTS_EP_ONLY=\"{ep_only.get(\"max_running_requests\", 256)}\"')
print(f'DECODE_CHUNKED_PREFILL_SIZE_EP_ONLY=\"{eval_formula(ep_only.get(\"chunked_prefill_size\", 262144))}\"')
print(f'DECODE_CONTEXT_LENGTH_EP_ONLY=\"{ep_only.get(\"context_length\", \"\")}\"')
s, e = parse_range(ep_only.get('cuda_graph_bs_range', '1-256'), 1, 256)
print(f'DECODE_CUDA_GRAPH_BS_EP_ONLY_START=\"{s}\"')
print(f'DECODE_CUDA_GRAPH_BS_EP_ONLY_END=\"{e}\"')

# Decode no-DP config
print(f'DECODE_MAX_RUNNING_REQUESTS_NO_DP=\"{no_dp.get(\"max_running_requests\", 128)}\"')
print(f'DECODE_CHUNKED_PREFILL_SIZE_NO_DP=\"{eval_formula(no_dp.get(\"chunked_prefill_size\", 262144))}\"')
print(f'DECODE_CONTEXT_LENGTH_NO_DP=\"{no_dp.get(\"context_length\", \"\")}\"')
s, e = parse_range(no_dp.get('cuda_graph_bs_range', '1-128'), 1, 128)
print(f'DECODE_CUDA_GRAPH_BS_NO_DP_START=\"{s}\"')
print(f'DECODE_CUDA_GRAPH_BS_NO_DP_END=\"{e}\"')
")"

echo "Loaded model configuration for: $MODEL_NAME"

if [[ "$PREFILL_ENABLE_DP" == "true" ]]; then
    prefill_cuda_graph_bs=($PREFILL_CUDA_GRAPH_BS_DP)
    prefill_max_running_requests=$PREFILL_MAX_RUNNING_REQUESTS_DP
    prefill_chunked_prefill_size=$PREFILL_CHUNKED_PREFILL_SIZE_DP
    prefill_context_length=$PREFILL_CONTEXT_LENGTH_DP
    prefill_max_total_tokens=$PREFILL_MAX_TOTAL_TOKENS_DP
    prefill_enable_two_batch_overlap=$PREFILL_ENABLE_TWO_BATCH_OVERLAP_DP
else
    prefill_cuda_graph_bs=($(seq $PREFILL_CUDA_GRAPH_BS_NO_DP_START $PREFILL_CUDA_GRAPH_BS_NO_DP_END))
    prefill_max_running_requests=$PREFILL_MAX_RUNNING_REQUESTS_NO_DP
    prefill_chunked_prefill_size=$PREFILL_CHUNKED_PREFILL_SIZE_NO_DP
    prefill_context_length=$PREFILL_CONTEXT_LENGTH_NO_DP
    prefill_max_total_tokens=$PREFILL_MAX_TOTAL_TOKENS_NO_DP
    prefill_enable_two_batch_overlap="false"
fi

if [[ "$PREFILL_ENABLE_DP" == "true" ]] && [[ "$PREFILL_ENABLE_EP" == "true" ]]; then
    prefill_max_running_requests=$BENCH_MAX_CONC_VALUE
    prefill_dp_ranks=$PREFILL_TP_SIZE
    echo "[DP+EP override] Prefill: max-running-requests=$prefill_max_running_requests, MOE_MAX_INPUT=$MORI_MOE_MAX_INPUT_TOKENS_PREFILL"
fi

if [[ "$DECODE_ENABLE_DP" == "true" ]]; then
    decode_cuda_graph_bs=($(seq $DECODE_CUDA_GRAPH_BS_DP_START $DECODE_CUDA_GRAPH_BS_DP_END))
    decode_max_running_requests=$((DECODE_CUDA_GRAPH_BS_DP_END * DECODE_TP_SIZE))
    decode_context_length=$DECODE_CONTEXT_LENGTH_DP
elif [[ "$DECODE_ENABLE_EP" == "true" ]]; then
    decode_cuda_graph_bs=($(seq $DECODE_CUDA_GRAPH_BS_EP_ONLY_START $DECODE_CUDA_GRAPH_BS_EP_ONLY_END))
    decode_max_running_requests=$DECODE_MAX_RUNNING_REQUESTS_EP_ONLY
    decode_context_length=$DECODE_CONTEXT_LENGTH_EP_ONLY
else
    decode_cuda_graph_bs=($(seq $DECODE_CUDA_GRAPH_BS_NO_DP_START $DECODE_CUDA_GRAPH_BS_NO_DP_END))
    decode_max_running_requests=$DECODE_MAX_RUNNING_REQUESTS_NO_DP
    decode_context_length=$DECODE_CONTEXT_LENGTH_NO_DP
fi
# In PD-disaggregation decode must admit requests against the SAME context length
# as prefill; otherwise decode accepts over-length requests that prefill rejects and
# they hang forever waiting for a KV transfer. Fall back to the prefill value.
if [[ -z "$decode_context_length" ]]; then
    decode_context_length=$prefill_context_length
fi

if [[ "$DECODE_ENABLE_DP" == "true" ]] && [[ "$DECODE_ENABLE_EP" == "true" ]]; then
    decode_max_running_requests=$BENCH_MAX_CONC_VALUE
    decode_dp_ranks=$DECODE_TP_SIZE
    MORI_MAX_DISPATCH_TOKENS_DECODE=$((BENCH_MAX_CONC_VALUE / decode_dp_ranks))
    SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD=$((MORI_MAX_DISPATCH_TOKENS_DECODE * 2))
    export SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD
    echo "[DP+EP override] Decode: max-running-requests=$decode_max_running_requests, DISPATCH_TOKENS=$MORI_MAX_DISPATCH_TOKENS_DECODE, MOE_MAX_INPUT=$MORI_MOE_MAX_INPUT_TOKENS_DECODE, INTER_KERNEL_SWITCH=$SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD"
fi

if [[ "$PREFILL_DISABLE_CUDA_GRAPH" == "True" ]] || [[ "$PREFILL_DISABLE_CUDA_GRAPH" == "true" ]]; then
    PREFILL_MODE_FLAGS="--mem-fraction-static ${PREFILL_MEM_FRACTION_STATIC} --max-running-requests ${prefill_max_running_requests} --chunked-prefill-size ${prefill_chunked_prefill_size} --disable-cuda-graph "
else
    PREFILL_MODE_FLAGS="--mem-fraction-static ${PREFILL_MEM_FRACTION_STATIC} --max-running-requests ${prefill_max_running_requests} --chunked-prefill-size ${prefill_chunked_prefill_size} --cuda-graph-bs ${prefill_cuda_graph_bs[*]} "
fi

if [[ "$PREFILL_DISABLE_RADIX_CACHE" == "True" ]] || [[ "$PREFILL_DISABLE_RADIX_CACHE" == "true" ]]; then
    PREFILL_MODE_FLAGS="$PREFILL_MODE_FLAGS --disable-radix-cache"
fi
# Agentic runs need the radix/prefix cache.
if [[ "${IS_AGENTIC:-0}" == "1" || "${IS_AGENTIC:-}" == "true" ]]; then
    PREFILL_MODE_FLAGS="${PREFILL_MODE_FLAGS//--disable-radix-cache/}"
fi
if [[ -n "$prefill_context_length" ]]; then
    PREFILL_MODE_FLAGS="$PREFILL_MODE_FLAGS --context-length ${prefill_context_length}"
fi
if [[ -n "$prefill_max_total_tokens" ]]; then
    PREFILL_MODE_FLAGS="$PREFILL_MODE_FLAGS --max-total-tokens ${prefill_max_total_tokens}"
fi
if [[ "$prefill_enable_two_batch_overlap" == "True" ]] || [[ "$prefill_enable_two_batch_overlap" == "true" ]]; then
    PREFILL_MODE_FLAGS="$PREFILL_MODE_FLAGS --enable-two-batch-overlap"
    PREFILL_SDMA_ENV="MORI_ENABLE_SDMA=true"
fi

DECODE_MODE_FLAGS="--mem-fraction-static ${DECODE_MEM_FRACTION_STATIC} --max-running-requests ${decode_max_running_requests} --cuda-graph-bs ${decode_cuda_graph_bs[*]} "

if [[ "$DECODE_PREFILL_ROUND_ROBIN_BALANCE" == "True" ]] || [[ "$DECODE_PREFILL_ROUND_ROBIN_BALANCE" == "true" ]]; then
    DECODE_MODE_FLAGS="$DECODE_MODE_FLAGS --prefill-round-robin-balance"
fi
if [[ -n "$decode_context_length" ]]; then
    DECODE_MODE_FLAGS="$DECODE_MODE_FLAGS --context-length ${decode_context_length}"
fi

if [[ "$DECODE_DISAGG_ENABLE_RADIX_CACHE" == "True" ]] || [[ "$DECODE_DISAGG_ENABLE_RADIX_CACHE" == "true" ]]; then
    DECODE_MODE_FLAGS="$DECODE_MODE_FLAGS --disaggregation-decode-enable-radix-cache"
fi

if [[ "$DECODE_MTP_SIZE" -gt 0 ]]; then
    MORI_MAX_DISPATCH_TOKENS_DECODE=$((MORI_MAX_DISPATCH_TOKENS_DECODE * (DECODE_MTP_SIZE + 1)))
fi

# Cluster topology
IFS=',' read -ra IP_ARRAY <<< "$IPADDRS"

PREFILL_NODES_PER_WORKER=$(((PREFILL_TP_SIZE + 7) / GPUS_PER_NODE))
DECODE_NODES_PER_WORKER=$(((DECODE_TP_SIZE + 7) / GPUS_PER_NODE))
NODE_OFFSET=$((PREFILL_NODES_PER_WORKER * xP))

PREFILL_HEADNODE_URLS=()
PREFILL_ARGS=""
# Per-worker Prometheus /metrics endpoints for aiperf's --server-metrics scrape;
# the router on :30000 does not serve Prometheus (see ENABLE_METRICS).
SERVER_METRICS_URLS=()
# Per-worker base URLs for cache flushing between concurrency points; the router
# does not fan /flush_cache out, so trace_replay.sh must POST to each worker.
SERVER_FLUSH_URLS=()
for i in $(seq 0 $((xP - 1))); do
    prefill_idx=$((i * PREFILL_NODES_PER_WORKER))
    PREFILL_HEADNODE_URLS[$i]="${IP_ARRAY[$prefill_idx]}:${HEADNODE_PORT}"
    PREFILL_ARGS="$PREFILL_ARGS --prefill http://${IP_ARRAY[$prefill_idx]}:8000"
    SERVER_METRICS_URLS+=("http://${IP_ARRAY[$prefill_idx]}:8000/metrics")
    SERVER_FLUSH_URLS+=("http://${IP_ARRAY[$prefill_idx]}:8000")
done

DECODE_HEADNODE_URLS=()
DECODE_ARGS=""
for i in $(seq 0 $((yD - 1))); do
    decode_idx=$((i * DECODE_NODES_PER_WORKER + NODE_OFFSET))
    DECODE_HEADNODE_URLS[$i]="${IP_ARRAY[$decode_idx]}:${HEADNODE_PORT}"
    DECODE_ARGS="$DECODE_ARGS --decode http://${IP_ARRAY[$decode_idx]}:8000"
    SERVER_METRICS_URLS+=("http://${IP_ARRAY[$decode_idx]}:8000/metrics")
    SERVER_FLUSH_URLS+=("http://${IP_ARRAY[$decode_idx]}:8000")
done

echo "Prefill worker headnode list: ${PREFILL_HEADNODE_URLS[@]}"
echo "Decode  worker headnode list: ${DECODE_HEADNODE_URLS[@]}"
echo "Server metrics endpoints:     ${SERVER_METRICS_URLS[@]}"
echo "Server flush endpoints:       ${SERVER_FLUSH_URLS[@]}"


# KV_P2P_TRANSFER (from amd-master.yaml kv-p2p-transfer) overrides the
# --disaggregation-transfer-backend baked into models.yaml base_flags.
apply_kv_p2p_transfer_override() {
    local flags="$1"
    if [[ -z "${KV_P2P_TRANSFER:-}" ]]; then
        printf '%s' "$flags"
        return 0
    fi
    local stripped
    stripped="$(echo "$flags" | sed -E 's/--disaggregation-transfer-backend[[:space:]]+[^[:space:]]+//g')"
    stripped="${stripped#"${stripped%%[![:space:]]*}"}"
    stripped="${stripped%"${stripped##*[![:space:]]}"}"
    echo "[KV_P2P] Using disaggregation-transfer-backend=${KV_P2P_TRANSFER} (KV_P2P_TRANSFER env)" >&2
    printf '%s --disaggregation-transfer-backend %s' "$stripped" "$KV_P2P_TRANSFER"
}

build_server_config() {
    local mode="$1"
    local model_name="$2"
    local tp_size="$3"
    local enable_ep="$4"
    local enable_dp="$5"
    local decode_mtp_size="$6"

    local ep_size=1
    local dp_size=1

    if [[ "$enable_ep" == "true" ]]; then
        ep_size=$tp_size
    fi

    if [[ "$enable_dp" == "true" ]]; then
        dp_size=$tp_size
    fi

    local parallel_args="--tp-size ${tp_size}"

    if [[ "$enable_ep" == "true" ]]; then
        parallel_args="$parallel_args --ep-size ${ep_size}"
    fi

    if [[ "$enable_dp" == "true" ]]; then
        parallel_args="$parallel_args --dp-size ${dp_size}"
    fi

    local base_config
    base_config="$(apply_kv_p2p_transfer_override "$MODEL_BASE_FLAGS")"
    local mtp_config=""
    local dp_config=""
    local ep_config=""
    local specific_config=""

    if [ "$decode_mtp_size" -gt 0 ]; then
        mtp_config="${MODEL_MTP_FLAGS} --speculative-num-steps ${decode_mtp_size} --speculative-num-draft-tokens $((decode_mtp_size + 1))"
    fi

    if [[ "$enable_dp" == "true" ]]; then
        dp_config="$MODEL_DP_FLAGS"
    fi

# Without EP the a2a backend / deepep mode / ep-dispatch flags are dropped, so the
# MoE runs tensor-parallel even when dp-attention is on.
    if [[ "$enable_ep" == "true" ]]; then
        ep_config="$MODEL_EP_FLAGS"
    fi

    if [[ "$mode" == "prefill" ]]; then
        specific_config="$PREFILL_MODE_FLAGS"
    elif [[ "$mode" == "decode" ]]; then
        specific_config="$DECODE_MODE_FLAGS"
    fi

    local full_config="$parallel_args"
    if [[ -n "$base_config" ]]; then
        full_config="$full_config $base_config"
    fi
    if [[ -n "$ep_config" ]]; then
        full_config="$full_config $ep_config"
    fi
# MTP/speculative flags go to BOTH prefill and decode: in PD-disaggregation the
# draft (nextn) layers take part in prefill KV computation, so the PD state component
# count must match. sglang v0.5.15+ rejects a mismatch ("state component count
# mismatch"); older builds silently fed decode uninitialized nextn state (lossy MTP).
    if [[ -n "$mtp_config" ]]; then
        full_config="$full_config $mtp_config"
    fi
    if [[ -n "$dp_config" ]]; then
        full_config="$full_config $dp_config"
    fi
    if [[ -n "$specific_config" ]]; then
        full_config="$full_config $specific_config"
    fi

    echo "$full_config"
}

PREFILL_SERVER_CONFIG=$(build_server_config "prefill" "$MODEL_NAME" "$PREFILL_TP_SIZE" "$PREFILL_ENABLE_EP" "$PREFILL_ENABLE_DP" "$DECODE_MTP_SIZE")
DECODE_SERVER_CONFIG=$(build_server_config "decode" "$MODEL_NAME" "$DECODE_TP_SIZE" "$DECODE_ENABLE_EP" "$DECODE_ENABLE_DP" "$DECODE_MTP_SIZE")

if [[ "${ENABLE_METRICS:-0}" == "1" ]]; then
    [[ "$PREFILL_SERVER_CONFIG" != *"--enable-metrics"* ]] && PREFILL_SERVER_CONFIG="$PREFILL_SERVER_CONFIG --enable-metrics"
    [[ "$DECODE_SERVER_CONFIG" != *"--enable-metrics"* ]] && DECODE_SERVER_CONFIG="$DECODE_SERVER_CONFIG --enable-metrics"
fi

if [[ -n "$MODEL_NAME" ]]; then
    echo "Using model-specific configuration for: $MODEL_NAME"
fi

# sync.py server-up barrier timeout; DSV4 needs more headroom.
if [[ -z "${SYNC_BARRIER_TIMEOUT:-}" ]]; then
    case "${MODEL_NAME}" in
        *DeepSeek-V4*) SYNC_BARRIER_TIMEOUT=3000 ;;
        *) SYNC_BARRIER_TIMEOUT=1800 ;;
    esac
fi
echo "SYNC_BARRIER_TIMEOUT=${SYNC_BARRIER_TIMEOUT}s (model=${MODEL_NAME:-unset})"

# HiCache KV offloading
KV_OFFLOADING="${KV_OFFLOADING:-none}"
KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-}"
if [[ "$KV_OFFLOADING" != "none" && "$KV_OFFLOAD_BACKEND" == "hicache" ]]; then
    HICACHE_HOST_POOL_COUNT="${HICACHE_HOST_POOL_COUNT:-1}"
    HICACHE_PAGE_SIZE="${HICACHE_PAGE_SIZE:-1}"
    HICACHE_PREFETCH_POLICY="${HICACHE_PREFETCH_POLICY:-wait_complete}"

    # Optional L3 storage tier behind the CPU-DRAM (L2) cache.
    #   ""        -> CPU DRAM only (default)
    #   "mooncake"-> Mooncake distributed KV store (needs a mooncake_master)
    HICACHE_STORAGE_BACKEND="${HICACHE_STORAGE_BACKEND:-}"

# The mooncake L3 store maps a page-contiguous segment for RDMA/zero-copy, so it
# needs the page_first layout with the direct IO backend; that layout asserts
# host_pool > device_pool, so it needs a large CPU-DRAM budget.
    if [[ "$HICACHE_STORAGE_BACKEND" == "mooncake" ]]; then
        HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first}"
        HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
        HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
    else
        HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first_direct}"
        HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
        HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
    fi

# The Mooncake master runs once on node 0; every prefill/decode server reaches it
# via NODE0_ADDR.
    MC_MASTER_PORT="${MC_MASTER_PORT:-50061}"
    MC_METADATA_PORT="${MC_METADATA_PORT:-8080}"
    MC_METRICS_PORT="${MC_METRICS_PORT:-9003}"
    MC_MASTER_THREADS="${MC_MASTER_THREADS:-64}"
    MC_EVICTION_HIGH_WATERMARK="${MC_EVICTION_HIGH_WATERMARK:-0.95}"
    MC_PROTOCOL="${MC_PROTOCOL:-tcp}"
    MC_GLOBAL_SEG="${MC_GLOBAL_SEG:-64gb}"
    MC_DEVICE="${MC_DEVICE:-$IBDEVICES}"
    MC_MASTER_ADDR="${MC_MASTER_ADDR:-${NODE0_ADDR}:${MC_MASTER_PORT}}"
    MC_METADATA_SERVER="${MC_METADATA_SERVER:-http://${NODE0_ADDR}:${MC_METADATA_PORT}/metadata}"

# The extra-config JSON is single-quoted so it survives the later eval of the
# launch command as a single argument.
    build_storage_flags() {
        [[ "$HICACHE_STORAGE_BACKEND" != "mooncake" ]] && return 0
        local extra="{\"master_server_address\": \"${MC_MASTER_ADDR}\", \"protocol\": \"${MC_PROTOCOL}\", \"device_name\": \"${MC_DEVICE}\", \"local_hostname\": \"${host_ip}\", \"global_segment_size\": \"${MC_GLOBAL_SEG}\", \"metadata_server\": \"${MC_METADATA_SERVER}\", \"check_server\": false}"
        echo "--hicache-storage-backend mooncake --hicache-storage-backend-extra-config '${extra}' --enable-metrics --enable-cache-report"
    }

# Prefer an absolute per-rank pool from the sweep generator's TOTAL_CPU_DRAM_GB
# budget; fall back to --hicache-ratio when no budget is given. FORCE_HICACHE_RATIO
# opts into ratio sizing without unsetting TOTAL_CPU_DRAM_GB, which benchmark_lib.sh
# requires client-side (KV_OFFLOADING=dram) and which is forwarded into client.env.
    HICACHE_RATIO="${HICACHE_RATIO:-5}"
    HICACHE_SIZING_FLAGS="--hicache-ratio ${HICACHE_RATIO}"
# DeepSeek V4's hybrid HiCache pool rejects --hicache-size (ratio only):
# https://github.com/sgl-project/sglang/blob/9dd57ef8c48e2cd82292d849f01e2130c5203e67/python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py#L262-L266
    if [[ "${FORCE_HICACHE_RATIO:-0}" != "1" && -n "${TOTAL_CPU_DRAM_GB:-}" && "${TOTAL_CPU_DRAM_GB}" -gt 0 && "${MODEL_NAME}" != *DeepSeek-V4* ]]; then
        # TOTAL_CPU_DRAM_GB is the prefill worker's per-node budget; --hicache-size is
        # per rank per host pool. A prefill server may span nodes, so divide by the
        # ranks that land on one node.
        prefill_ranks_per_node=$(( PREFILL_TP_SIZE < GPUS_PER_NODE ? PREFILL_TP_SIZE : GPUS_PER_NODE ))
        prefill_hicache_size_gb=$(( TOTAL_CPU_DRAM_GB / prefill_ranks_per_node / HICACHE_HOST_POOL_COUNT ))
        if (( prefill_hicache_size_gb < 1 )); then
            echo "Error: TOTAL_CPU_DRAM_GB=${TOTAL_CPU_DRAM_GB} / ranks_per_node=${prefill_ranks_per_node} / host_pools=${HICACHE_HOST_POOL_COUNT} rounds below 1 GB" >&2
            exit 1
        fi
        HICACHE_SIZING_FLAGS="--hicache-size ${prefill_hicache_size_gb}"
        echo "[HiCache] prefill CPU pool capped at ${prefill_hicache_size_gb} GB/rank (budget ${TOTAL_CPU_DRAM_GB} GB / ranks_per_node ${prefill_ranks_per_node} / host_pools ${HICACHE_HOST_POOL_COUNT})"
    fi

    build_hicache_flags() {
        echo "--page-size ${HICACHE_PAGE_SIZE} --enable-hierarchical-cache ${HICACHE_SIZING_FLAGS} --hicache-io-backend ${HICACHE_IO_BACKEND} --hicache-mem-layout ${HICACHE_MEM_LAYOUT} --hicache-write-policy ${HICACHE_WRITE_POLICY} --hicache-storage-prefetch-policy ${HICACHE_PREFETCH_POLICY} $(build_storage_flags)"
    }

    # HiCache requires RadixAttention; strip any --disable-radix-cache.
    PREFILL_SERVER_CONFIG="${PREFILL_SERVER_CONFIG//--disable-radix-cache/}"
    DECODE_SERVER_CONFIG="${DECODE_SERVER_CONFIG//--disable-radix-cache/}"

    PREFILL_SERVER_CONFIG="$PREFILL_SERVER_CONFIG $(build_hicache_flags "$PREFILL_TP_SIZE")"


    DECODE_SERVER_CONFIG="$DECODE_SERVER_CONFIG --page-size ${HICACHE_PAGE_SIZE}"
    echo "[HiCache] KV_OFFLOADING=${KV_OFFLOADING} backend=${KV_OFFLOAD_BACKEND} applied to prefill only; decode mirrors --page-size ${HICACHE_PAGE_SIZE} for transfer compatibility (chunk cache under the mori transfer backend)"
    echo "[HiCache] params: io_backend=${HICACHE_IO_BACKEND}, mem_layout=${HICACHE_MEM_LAYOUT}, page_size=${HICACHE_PAGE_SIZE}, write_policy=${HICACHE_WRITE_POLICY}, prefetch_policy=${HICACHE_PREFETCH_POLICY}, storage_backend=${HICACHE_STORAGE_BACKEND:-none}"
    if [[ "$HICACHE_STORAGE_BACKEND" == "mooncake" ]]; then
        echo "[HiCache] Mooncake store: master=${MC_MASTER_ADDR} metadata=${MC_METADATA_SERVER} protocol=${MC_PROTOCOL} device=${MC_DEVICE} segment=${MC_GLOBAL_SEG} threads=${MC_MASTER_THREADS} eviction_watermark=${MC_EVICTION_HIGH_WATERMARK}"
    fi
else
    echo "[HiCache] KV_OFFLOADING=${KV_OFFLOADING} backend=${KV_OFFLOAD_BACKEND:-none} (HiCache disabled)"
fi

if [[ "${EVAL_ONLY:-false}" == "true" ]] || [[ "${RUN_EVAL:-false}" == "true" ]]; then
    PREFILL_SERVER_CONFIG=$(echo "$PREFILL_SERVER_CONFIG" | sed 's/--ep-dispatch-algorithm fake//g')
    DECODE_SERVER_CONFIG=$(echo "$DECODE_SERVER_CONFIG" | sed 's/--ep-dispatch-algorithm fake//g')
    unset MORI_MOE_MAX_INPUT_TOKENS_PREFILL
    unset MORI_MOE_MAX_INPUT_TOKENS_DECODE
fi


# sync.py barrier exits 1 on timeout, but without an explicit check the script
# would continue past a timed-out barrier and launch the next stage against
# servers/routers that never came up.
run_barrier_or_die() {
    local desc="$1" cmd="$2"
    if ! eval "$cmd"; then
        echo "FATAL: ${desc} failed — see the sync.py timeout output above for which node/port never became ready." >&2
        exit 1
    fi
}

echo "Waiting at the container creation barrier on $host_name"
run_barrier_or_die "container creation barrier" "python3 $SGLANG_WS_PATH/sync.py barrier \
    --local-ip ${host_ip} \
    --local-port 5000 \
    --enable-port \
    --node-ips ${IPADDRS} \
    --node-ports 5000 \
    --wait-for-all-ports \
    --timeout 300"


# Node role assignment and server launch

# Run a blocking command while watching the local server PID. If the server dies
# the command is aborted and we return non-zero, so SLURM's --kill-on-bad-exit
# tears the job down in seconds instead of waiting out the barrier timeout.
wait_or_die() {            # $1 = server pid to watch; rest = blocking command
    local watch=$1; shift
    "$@" & local cmd=$!
    while kill -0 "$cmd" 2>/dev/null; do
        kill -0 "$watch" 2>/dev/null || {
            echo "FATAL: $(hostname) local sglang server (pid $watch) died; tearing down job" >&2
            kill "$cmd" 2>/dev/null || true
            return 1
        }
        sleep 5
    done
    wait "$cmd"
}

if [ "$NODE_RANK" -eq 0 ]; then
    echo "NODE INFO ======================================="
    echo "================================================"
    echo "Node List : ${SLURM_JOB_NODELIST}"
    echo "Node IPs : ${IPADDRS}"
    echo "Model Name : ${MODEL_NAME:-'Not specified'}"
    echo "================================================"

    echo "CLUSTER INFO ===================================="
    echo "================================================"
    echo "${host_name}:${host_ip} is Proxy Node and Prefill Node"
    echo "Using prefill config: $PREFILL_SERVER_CONFIG"
    echo "Prefill parallelism: TP=${PREFILL_TP_SIZE}, EP enabled: ${PREFILL_ENABLE_EP}, DP enabled: ${PREFILL_ENABLE_DP}, MTP size=${DECODE_MTP_SIZE}"
    echo "Decode  parallelism: TP=${DECODE_TP_SIZE},  EP enabled: ${DECODE_ENABLE_EP},  DP enabled: ${DECODE_ENABLE_DP},  MTP size=${DECODE_MTP_SIZE}"
    echo "Prefill servers ($((PREFILL_TP_SIZE/GPUS_PER_NODE)) nodes): ${PREFILL_ARGS}"
    echo "Decode servers  ($((DECODE_TP_SIZE/GPUS_PER_NODE))  nodes): ${DECODE_ARGS}"
    echo "Prefill env: SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_MAX_DISPATCH_TOKENS_PREFILL}"
    echo "Decode  env: SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_MAX_DISPATCH_TOKENS_DECODE} "
    echo "Decode  env: SGLANG_MORI_MOE_MAX_INPUT_TOKENS=${MORI_MOE_MAX_INPUT_TOKENS_DECODE} "

    echo "================================================"

    CMD_DUMP="/run_logs/slurm_job-${SLURM_JOB_ID}/commands_${host_name}.txt"
    dump_cmd() { echo -e "\n# ── $1 ──\n$2" >> "$CMD_DUMP"; }
    echo "# Commands dump — $(date -u '+%Y-%m-%d %H:%M:%S UTC')" > "$CMD_DUMP"
    echo "# Host: ${host_name} (${host_ip})  Node rank: ${NODE_RANK}" >> "$CMD_DUMP"
    echo "# Model: ${MODEL_NAME}  Image: ${DOCKER_IMAGE_NAME:-unknown}" >> "$CMD_DUMP"

    if [[ "${KV_OFFLOADING:-none}" != "none" && "${KV_OFFLOAD_BACKEND:-}" == "hicache" && "${HICACHE_STORAGE_BACKEND:-}" == "mooncake" ]]; then
        echo "Starting Mooncake master on ${host_ip}:${MC_MASTER_PORT} (metadata :${MC_METADATA_PORT}, metrics :${MC_METRICS_PORT})"
        MC_MASTER_CMD="mooncake_master \
        --enable_http_metadata_server=true \
        --http_metadata_server_host=0.0.0.0 \
        --http_metadata_server_port=${MC_METADATA_PORT} \
        --rpc_port=${MC_MASTER_PORT} \
        --rpc_thread_num=${MC_MASTER_THREADS} \
        --metrics_port=${MC_METRICS_PORT} \
        --enable_metric_reporting=true \
        --eviction_high_watermark_ratio=${MC_EVICTION_HIGH_WATERMARK}"
        dump_cmd "MOONCAKE MASTER" "$MC_MASTER_CMD"
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "DRY RUN: $MC_MASTER_CMD"
        else
            MC_MASTER_LOG="/run_logs/slurm_job-${SLURM_JOB_ID}/mooncake_master_${host_name}.log"
            mooncake_master \
                --enable_http_metadata_server=true \
                --http_metadata_server_host=0.0.0.0 \
                --http_metadata_server_port="${MC_METADATA_PORT}" \
                --rpc_port="${MC_MASTER_PORT}" \
                --rpc_thread_num="${MC_MASTER_THREADS}" \
                --metrics_port="${MC_METRICS_PORT}" \
                --enable_metric_reporting=true \
                --eviction_high_watermark_ratio="${MC_EVICTION_HIGH_WATERMARK}" \
                > "${MC_MASTER_LOG}" 2>&1 &
            mc_master_pid=$!
            sleep 3
            # On shared nodes the Mooncake RPC port may already be held by another
            # user's master; the metrics-port check below can then pass against the
            # foreign master while our RPC port is dead, and prefill hangs.
            if grep -qiE "Address already in use|bind .*error" "${MC_MASTER_LOG}" 2>/dev/null; then
                echo "ERROR: mooncake_master failed to bind port ${MC_MASTER_PORT} (already in use)."
                echo "       Set MC_MASTER_PORT/MC_METRICS_PORT to free ports and resubmit."
                grep -iE "Address already in use|bind .*error" "${MC_MASTER_LOG}" | tail -3
                exit 1
            fi
            for ((i=3; i<=60; i+=3)); do
                if curl -sf "http://127.0.0.1:${MC_METRICS_PORT}/get_all_segments" >/dev/null 2>&1; then
                    echo "  mooncake master OK at ${i}s"
                    break
                fi
                sleep 3
            done
        fi
    fi

    PREFILL_MORI_MOE_ENV=""
    set -x
    if [[ -n "$MORI_MOE_MAX_INPUT_TOKENS_PREFILL" ]]; then
        PREFILL_MORI_MOE_ENV="SGLANG_MORI_MOE_MAX_INPUT_TOKENS=${MORI_MOE_MAX_INPUT_TOKENS_PREFILL}"
    fi
    set +x
    PREFILL_CMD="SGLANG_MORI_COMBINE_DTYPE=${MORI_COMBINE_DTYPE_PREFILL} ${PREFILL_SDMA_ENV} ${PREFILL_MORI_MOE_ENV} SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK_PREFILL:-${MORI_MAX_DISPATCH_TOKENS_PREFILL}} MORI_IO_SQ_BACKOFF_TIMEOUT_US=${MORI_IO_SQ_BACKOFF_TIMEOUT_US} MORI_IO_QP_MAX_SEND_WR=${MORI_IO_QP_MAX_SEND_WR} ${LAUNCH_PREFIX:-} python3 -m sglang.launch_server \
        --model-path $MODEL_DIR/$MODEL_NAME \
        --disaggregation-mode prefill \
        --disaggregation-ib-device ${IBDEVICES} \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code \
        ${PREFILL_SERVER_CONFIG} "

    if [ "$PREFILL_NODES_PER_WORKER" -gt 1 ]; then
        PREFILL_CMD="$PREFILL_CMD --dist-init-addr ${PREFILL_HEADNODE_URLS[0]} --nnodes ${PREFILL_NODES_PER_WORKER} --node-rank 0"
    fi


    dump_cmd "PREFILL (node 0)" "$PREFILL_CMD"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $PREFILL_CMD"
    else
        set -x
        # setsid puts the server and its TP-scheduler children in one process group so
        # teardown can kill -- -$pgid the whole tree. Killing $prefill0_pid alone leaves
        # children holding the tee pipe, so the container's outer | tee never gets EOF
        # and the container never exits. Process substitution keeps $! as the setsid
        # group leader rather than tee's pid.
        setsid bash -c "$PREFILL_CMD" \
            > >(tee /run_logs/slurm_job-${SLURM_JOB_ID}/prefill_${host_name}.log >/dev/null) 2>&1 &
        set +x
        prefill0_pid=$!
        prefill0_pgid=$(ps -o pgid= -p "$prefill0_pid" 2>/dev/null | tr -d ' ')
        : "${prefill0_pgid:=$prefill0_pid}"
    fi


    echo "Waiting for all prefill and decode servers to be up . . ."


    BARRIER_CMD="python3 $SGLANG_WS_PATH/sync.py barrier \
        --node-ips ${IPADDRS} \
        --node-ports 8000 \
        --wait-for-all-ports \
        --timeout ${SYNC_BARRIER_TIMEOUT}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BARRIER_CMD"
    else
        wait_or_die "$prefill0_pid" bash -c "$BARRIER_CMD" || exit 1
    fi
    echo "Congratulations!!! All prefill and decode servers are up . . ."

    if [[ "${IS_AGENTIC:-0}" == "1" || "${IS_AGENTIC:-}" == "true" ]]; then
        # Long-context prefills can look unhealthy to the default circuit breaker during
        # a concurrent burst, so disable it and relax health checks. cache_aware prefill
        # routing exploits HiCache/radix prefix reuse across the agentic trace.
        ROUTER_RESILIENCE_FLAGS="${ROUTER_RESILIENCE_FLAGS:---disable-circuit-breaker --health-failure-threshold 100 --health-check-timeout-secs 600 --health-check-interval-secs 30}"
        ROUTER_PREFILL_POLICY="${PREFILL_ROUTER_POLICY:-consistent_hashing}"
        ROUTER_CACHE_THRESHOLD="${ROUTER_CACHE_THRESHOLD:-0.3}"
        ROUTER_BALANCE_ABS_THRESHOLD="${ROUTER_BALANCE_ABS_THRESHOLD:-2}"
        ROUTER_BALANCE_REL_THRESHOLD="${ROUTER_BALANCE_REL_THRESHOLD:-1.1}"
        ROUTER_POLICY_FLAGS="${ROUTER_POLICY_FLAGS:---policy ${ROUTER_PREFILL_POLICY} --dp-aware --cache-threshold ${ROUTER_CACHE_THRESHOLD} --balance-abs-threshold ${ROUTER_BALANCE_ABS_THRESHOLD} --balance-rel-threshold ${ROUTER_BALANCE_REL_THRESHOLD}}"
    else
        # Shorten the breaker's open->half-open window and let the router retry worker
        # selection so a transient trip re-closes inside lm_eval's retry budget
        # (max_retries=5) instead of 503ing every request until the client gives up.
        # Thresholds are unchanged; this only speeds recovery.
        ROUTER_CB_ARGS="${ROUTER_CB_ARGS:---cb-timeout-duration-secs 15 --retry-max-retries 3}"
        ROUTER_POLICY_FLAGS="${ROUTER_POLICY_FLAGS:---policy random --prefill-policy random --decode-policy random}"
        ROUTER_RESILIENCE_FLAGS="${ROUTER_RESILIENCE_FLAGS:-${ROUTER_CB_ARGS}}"
    fi

    echo "Router config: IS_AGENTIC=${IS_AGENTIC:-0} policy/resilience=${ROUTER_POLICY_FLAGS} ${ROUTER_RESILIENCE_FLAGS}"

    ROUTER_CMD="python -m sglang_router.launch_router \
        --pd-disaggregation \
        --port 30000 \
        ${ROUTER_POLICY_FLAGS} \
        ${ROUTER_RESILIENCE_FLAGS} \
        ${PREFILL_ARGS} \
        ${DECODE_ARGS}"


    dump_cmd "ROUTER" "$ROUTER_CMD"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $ROUTER_CMD"
    else
        ROUTER_LOG_FILE="/run_logs/slurm_job-${SLURM_JOB_ID}/router_${host_name}.log"
        # sgl-router (Rust/tracing) emits ANSI color codes; NO_COLOR asks it to stop and
        # the sed strip guarantees a clean file either way. Process substitution keeps $!
        # as the router pid. sglang-router >=0.5.14 spawns the Rust worker (binds :30000)
        # as a child and lets the python launcher exit, so the worker reparents to init
        # but keeps its process group: launch under setsid and record the pgid so teardown
        # can kill -- -$proxy_pgid after the launcher is gone.
        set -x
        if [[ "${SGLANG_ROUTER_STDOUT_LOGS:-0}" == "1" ]]; then
            NO_COLOR=1 setsid bash -c "exec $ROUTER_CMD" > >(sed -u -r 's/\x1b\[[0-9;]*[a-zA-Z]//g' | tee "$ROUTER_LOG_FILE") 2>&1 &
        else
            NO_COLOR=1 setsid bash -c "exec $ROUTER_CMD" > >(sed -u -r 's/\x1b\[[0-9;]*[a-zA-Z]//g' >"$ROUTER_LOG_FILE") 2>&1 &
        fi
        set +x
        proxy_pid=$!
        proxy_pgid=$(ps -o pgid= -p "$proxy_pid" 2>/dev/null | tr -d ' ')
        : "${proxy_pgid:=$proxy_pid}"

        HEALTH_BARRIER_CMD="python3 $SGLANG_WS_PATH/sync.py barrier \
            --node-ips ${NODE0_ADDR} \
            --node-ports 30000 \
            --wait-for-all-health \
            --health-endpoint /readiness \
            --timeout ${SYNC_BARRIER_TIMEOUT}"

        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "DRY RUN: $HEALTH_BARRIER_CMD"
        else
            wait_or_die "$prefill0_pid" bash -c "$HEALTH_BARRIER_CMD" || exit 1
        fi

        # /readiness only proves the router process is up, not that it can reach a
        # prefill worker and complete a generation; an eval started on /readiness alone
        # 503'd every request ("all circuits open or unhealthy") and produced no results.
        # Gate on one successful generation through the router. Runs under wait_or_die
        # so a prefill crash right after /readiness aborts in seconds instead of burning
        # ROUTER_CANARY_TIMEOUT on repeated 503s.
        run_router_canary() {
            local canary_url="http://${NODE0_ADDR}:30000/v1/chat/completions"
            local canary_model="${MODEL_DIR}/${MODEL_NAME}"
            local canary_deadline=$(( $(date +%s) + ${ROUTER_CANARY_TIMEOUT:-600} ))
            local canary_code
            while [ "$(date +%s)" -lt "$canary_deadline" ]; do
                canary_code=$(curl -s -o /tmp/router_canary.out -w '%{http_code}' \
                    -m "${ROUTER_CANARY_REQ_TIMEOUT:-120}" \
                    -X POST "$canary_url" -H 'Content-Type: application/json' \
                    -d "{\"model\":\"${canary_model}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1,\"temperature\":0}" 2>/dev/null)
                if [ "$canary_code" = "200" ] && \
                   ! grep -qE "circuits open|server_selection_failed|No available" /tmp/router_canary.out 2>/dev/null; then
                    echo "Router readiness canary passed (end-to-end generation OK)"
                    return 0
                fi
                echo "Router readiness canary not ready yet (http=${canary_code}); retrying in 5s . . ."
                sleep 5
            done
            echo "ERROR: router readiness canary failed after ${ROUTER_CANARY_TIMEOUT:-600}s -- the router cannot complete a generation through a prefill worker (all circuits open/unhealthy). Refusing to start the eval against a non-serving router."
            head -c 800 /tmp/router_canary.out 2>/dev/null
            return 1
        }
        if [[ "${ROUTER_READINESS_CANARY:-1}" == "1" ]]; then
            wait_or_die "$prefill0_pid" run_router_canary || exit 1
        fi

        echo "Router is ready for benchmarking"
    fi


    echo "Ready for benchmarking on ${host_name}:${host_ip}"

    echo "Benchmarking on ${host_name}:${host_ip}"
    cd $SGLANG_WS_PATH

    if [ "$DECODE_MTP_SIZE" -gt 0 ]; then
        export IS_MTP=true
    else
        export IS_MTP=false
    fi

    if [[ "${IS_AGENTIC:-0}" == "1" || "${IS_AGENTIC:-}" == "true" ]]; then
        # aiperf auto-detects the router from --url, which does not expose Prometheus;
        # point the scrape at the per-worker /metrics endpoints or every server-side
        # cache/KV field comes out null.
        if [[ "${ENABLE_METRICS:-0}" == "1" && "${#SERVER_METRICS_URLS[@]}" -gt 0 ]]; then
            AIPERF_SERVER_METRICS_URLS=$(IFS=,; echo "${SERVER_METRICS_URLS[*]}")
            export AIPERF_SERVER_METRICS_URLS
            echo "AIPERF_SERVER_METRICS_URLS=${AIPERF_SERVER_METRICS_URLS}"
        fi
        # trace_replay.sh flushes these workers directly when CLEAR_CACHE_BETWEEN_CONC=1.
        if [[ "${#SERVER_FLUSH_URLS[@]}" -gt 0 ]]; then
            SERVER_FLUSH_URLS_CSV=$(IFS=,; echo "${SERVER_FLUSH_URLS[*]}")
            export SERVER_FLUSH_URLS_CSV
            echo "SERVER_FLUSH_URLS_CSV=${SERVER_FLUSH_URLS_CSV}"
        fi
        # trace_replay.sh signature: model_path model_name concurrency_list log_path
        BENCH_CMD="bash $SGLANG_WS_PATH/trace_replay.sh \
            $MODEL_DIR $MODEL_NAME $BENCH_MAX_CONCURRENCY /run_logs/slurm_job-${SLURM_JOB_ID}"
        echo "Benchmark runner: trace_replay.sh (agentic, KV_OFFLOADING=${KV_OFFLOADING:-none}, backend=${KV_OFFLOAD_BACKEND:-none}, CONC=${BENCH_MAX_CONCURRENCY})"
    else
        # bench.sh signature:
        # n_prefill n_decode prefill_gpus decode_gpus model_dir model_name log_path
        # isl osl concurrency_list req_rate random_range_ratio num_prompts_multiplier
        BENCH_CMD="bash $SGLANG_WS_PATH/bench.sh ${xP} ${yD} $((PREFILL_TP_SIZE*xP)) $((DECODE_TP_SIZE*yD)) \
            $MODEL_DIR $MODEL_NAME /run_logs/slurm_job-${SLURM_JOB_ID} ${BENCH_INPUT_LEN} \
            ${BENCH_OUTPUT_LEN} \"${BENCH_MAX_CONCURRENCY}\" ${BENCH_REQUEST_RATE} \
            ${BENCH_RANDOM_RANGE_RATIO} ${BENCH_NUM_PROMPTS_MULTIPLIER}"
        echo "Benchmark runner: bench.sh (fixed-seq-len)"
    fi

    IS_AGENTIC_RUN=0
    if [[ "${IS_AGENTIC:-0}" == "1" || "${IS_AGENTIC:-}" == "true" ]]; then
        IS_AGENTIC_RUN=1
    fi

    if [[ "${EVAL_ONLY:-false}" == "true" ]]; then
        echo "EVAL_ONLY mode: skipping throughput benchmark"
    elif [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BENCH_CMD"
    elif [[ -n "${CLIENT_IMAGE:-}" && "$IS_AGENTIC_RUN" == "1" ]]; then
        # With CLIENT_IMAGE set, the aiperf trace replay runs in a sibling container
        # (pre-baked aiperf) on this node against the router over --network host.
        # job.slurm mounts the docker socket and forwards HOST_REPO_DIR / HOST_MODEL_DIR /
        # HOST_BENCH_LOGS / CLIENT_CONT_NAME for this.
        CLIENT_ENV_FILE="/run_logs/slurm_job-${SLURM_JOB_ID}/client.env"
        mkdir -p "/run_logs/slurm_job-${SLURM_JOB_ID}"
        # Unset vars are skipped so the client keeps its own defaults.
        {
            for _v in ENGINE MODEL_NAME MODEL_PREFIX PRECISION FRAMEWORK SPEC_DECODING \
                      DURATION MAX_MODEL_LEN RESULT_FILENAME RUNNER_NAME RUNNER_TYPE IMAGE \
                      AIPERF_SERVER_METRICS_URLS SERVER_FLUSH_URLS_CSV \
                      ENABLE_METRICS IS_AGENTIC CLEAR_CACHE_BETWEEN_CONC \
                      DISAGG IS_MULTINODE \
                      TP EP_SIZE DP_ATTENTION DCP_SIZE PCP_SIZE \
                      PREFILL_NUM_WORKERS PREFILL_TP PREFILL_EP PREFILL_DP_ATTN PREFILL_ENABLE_DP PREFILL_HARDWARE \
                      DECODE_NUM_WORKERS DECODE_TP DECODE_EP DECODE_DP_ATTN DECODE_ENABLE_DP DECODE_HARDWARE \
                      KV_OFFLOADING KV_OFFLOAD_BACKEND KV_OFFLOAD_BACKEND_METADATA TOTAL_CPU_DRAM_GB KV_P2P_TRANSFER \
                      WEKA_LOADER_OVERRIDE AIPERF_FAILED_REQUEST_THRESHOLD \
                      AIPERF_WARMUP_REQUESTS_PER_LANE AIPERF_TRACE_IDLE_GAP_CAP_SECONDS \
                      AIPERF_EXPERIMENTAL_FAST AIPERF_UNSAFE_OVERRIDE \
                      AIPERF_TRAJECTORY_START_MIN_RATIO AIPERF_TRAJECTORY_START_MAX_RATIO \
                      AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES ROUTER_PORT TQDM_MININTERVAL; do
                if [[ -n "${!_v+x}" ]]; then
                    _val="${!_v}"
                    # docker --env-file needs one KEY=VALUE per line; KV_OFFLOAD_BACKEND_METADATA
                    # carries pretty-printed multi-line JSON, so re-serialize it compact via
                    # json.loads/json.dumps. Empty/"none"/"null" means no metadata (job.slurm
                    # always sets the var) and must pass through untouched, matching
                    # optional_kv_offload_backend_metadata() in process_agentic_result.py.
                    if [[ "$_v" == "KV_OFFLOAD_BACKEND_METADATA" && -n "$_val" && "$_val" != "null" ]]; then
                        _val="$(python3 -c 'import json, sys
print(json.dumps(json.loads(sys.stdin.read())))' <<<"$_val")" || {
                            echo "KV_OFFLOAD_BACKEND_METADATA must contain valid JSON" >&2
                            exit 1
                        }
                    fi
                    printf '%s=%s\n' "$_v" "$_val"
                fi
            done
            echo "INFMAX_CONTAINER_WORKSPACE=/workspace"
            # AGENTIC_OUTPUT_DIR is deliberately not pinned: it must default to /workspace
            # (the host repo mount) so ${RESULT_FILENAME}_conc<N>.json lands where the
            # workflow guard globs it.
            echo "HF_HOME=/run_logs/hf_cache"
            echo "MODEL_DIR=/models"
            # Without a pre-baked venv (CLIENT_AIPERF_VENV unset, e.g. reusing the server
            # image) trace_replay builds aiperf from /workspace/utils/aiperf.
            if [[ -n "${CLIENT_AIPERF_VENV:-}" ]]; then
                echo "AIPERF_USE_PREBUILT=1"
                echo "AIPERF_VENV=${CLIENT_AIPERF_VENV}"
            fi
        } > "$CLIENT_ENV_FILE"

        echo "Launching agentic benchmark in separate client container: ${CLIENT_IMAGE}"
        docker rm -f "${CLIENT_CONT_NAME}" 2>/dev/null || true
        set -x
        docker run --rm --network host \
            --name "${CLIENT_CONT_NAME}" \
            --shm-size 32G \
            -v "${HOST_REPO_DIR}:/workspace" \
            -v "${HOST_MODEL_DIR}:/models" \
            -v /tmp:/run_logs \
            -v "${HOST_BENCH_LOGS}:/benchmark_logs" \
            --env-file "${CLIENT_ENV_FILE}" \
            --entrypoint "" \
            "${CLIENT_IMAGE}" \
            bash -lc "cd /workspace/benchmarks/multi_node/amd_utils && bash trace_replay.sh /models ${MODEL_NAME} \"${BENCH_MAX_CONCURRENCY}\" /run_logs/slurm_job-${SLURM_JOB_ID}"
        set +x
    else
        set -x
        eval "$BENCH_CMD"
        set +x
    fi

    if [[ "${RUN_EVAL:-false}" == "true" ]]; then
        echo "Running lm-eval (GSM8K) evaluation on Node 0..."

        # The throughput benchmark may have crashed decode workers; skip eval if so.
        EVAL_HEALTH_OK=false
        for _attempt in 1 2 3; do
            if curl -sf --max-time 10 "http://0.0.0.0:30000/readiness" >/dev/null 2>&1; then
                EVAL_HEALTH_OK=true
                break
            fi
            echo "Eval health check attempt $_attempt failed, retrying in 10s..."
            sleep 10
        done

        if [[ "$EVAL_HEALTH_OK" != "true" ]]; then
            echo "WARNING: Router health check failed after 3 attempts. Skipping eval."
        else
            # Must run from repo root so infx/evals/gsm8k.yaml resolves
            pushd /workspace

            source /workspace/benchmarks/benchmark_lib.sh

            # CONC must be exported before run_eval so meta_env.json matches validate_scores.py.
            if [[ -n "${EVAL_CONC:-}" ]]; then
                export EVAL_CONCURRENT_REQUESTS="${EVAL_CONC}"
            else
                export EVAL_CONCURRENT_REQUESTS=$(echo "$BENCH_MAX_CONCURRENCY" | tr 'x' '\n' | sort -n | tail -1)
            fi
            export CONC="${EVAL_CONCURRENT_REQUESTS}"

            if [[ -n "$prefill_context_length" ]]; then
                export EVAL_MAX_MODEL_LEN="$prefill_context_length"
            fi

            export ISL="${BENCH_INPUT_LEN:-0}"
            export OSL="${BENCH_OUTPUT_LEN:-0}"
            bridge_disagg_eval_metadata
            # IS_MULTINODE, FRAMEWORK, PRECISION, MODEL_PREFIX, RUNNER_TYPE, RESULT_FILENAME
            # arrive via Docker -e flags from job.slurm.

            if [[ "$DRY_RUN" -eq 1 ]]; then
                echo "DRY RUN: run_eval --port 30000 (framework=${EVAL_FRAMEWORK:-lm-eval}, conc=${EVAL_CONCURRENT_REQUESTS}, ctx=${EVAL_MAX_MODEL_LEN:-auto})"
            else
                run_eval --port 30000
                eval_rc=$?

                if [[ $eval_rc -ne 0 ]]; then
                    echo "ERROR: run_eval exited rc=$eval_rc; preserving failure artifacts" >&2
                    EVAL_FAILED=1
                else
                    # Always rewrite meta_env.json so EP/DPA match the workflow
                    # topology even when run_eval() staged artifacts internally.
                    rewrite_lm_eval_meta_env

                    # Fixed-seq-len post-bench eval still needs append to move
                    # results out of the temp EVAL_RESULT_DIR.
                    if [[ "${EVAL_ONLY:-false}" != "true" || "$IS_AGENTIC_RUN" != "1" ]]; then
                        append_lm_eval_summary
                    fi

                fi

                EVAL_COPY_DIR="/run_logs/slurm_job-${SLURM_JOB_ID}/eval_results"
                if stage_eval_artifacts \
                    "$EVAL_COPY_DIR" /workspace "${EVAL_RESULT_DIR:-}"; then
                    echo "Eval artifacts staged in $EVAL_COPY_DIR"
                else
                    echo "ERROR: failed to stage eval artifacts in $EVAL_COPY_DIR" >&2
                    EVAL_FAILED=1
                fi
            fi

            popd
        fi
    fi

    LOGS_OUTPUT="${BENCHMARK_LOGS_DIR:-/run_logs}/logs"
    mkdir -p "$LOGS_OUTPUT"

    if [[ "$DRY_RUN" -eq 0 ]]; then
        cp -r /run_logs/slurm_job-${SLURM_JOB_ID} "$LOGS_OUTPUT/"
        echo "Copied results to $LOGS_OUTPUT/slurm_job-${SLURM_JOB_ID}"
    fi

    echo "Killing the proxy server and prefill server"

    if [[ "$DRY_RUN" -eq 0 ]]; then
        # Group-kill the router (setsid at launch): the python launcher has usually
        # exited after spawning the Rust worker, which reparents to init but stays in
        # this group; kill $proxy_pid alone misses it and :30000 stays open.
        kill -TERM -"${proxy_pgid:-$proxy_pid}" 2>/dev/null || true
        # Group-kill the prefill tree so TP-scheduler children release the tee pipe
        # and the container can exit.
        kill -TERM -"${prefill0_pgid:-$prefill0_pid}" 2>/dev/null || true
    fi

    if [[ "${EVAL_FAILED:-0}" -eq 1 ]]; then
        echo "ERROR: eval failed; exiting node-0 with rc=1"
        exit 1
    fi

elif [ "$NODE_RANK" -gt 0 ] && [ "$NODE_RANK" -lt "$NODE_OFFSET" ]; then
    echo "${host_name}:${host_ip} is Prefill Node (Model: ${MODEL_NAME:-'default'})"
    echo "Using prefill config: $PREFILL_SERVER_CONFIG"
    echo "Prefill parallelism: TP=${PREFILL_TP_SIZE}, EP enabled: ${PREFILL_ENABLE_EP}, DP enabled: ${PREFILL_ENABLE_DP}"

    CMD_DUMP="/run_logs/slurm_job-${SLURM_JOB_ID}/commands_${host_name}.txt"
    dump_cmd() { echo -e "\n# ── $1 ──\n$2" >> "$CMD_DUMP"; }
    echo "# Commands dump — $(date -u '+%Y-%m-%d %H:%M:%S UTC')" > "$CMD_DUMP"
    echo "# Host: ${host_name} (${host_ip})  Node rank: ${NODE_RANK}" >> "$CMD_DUMP"

    PREFILL_MORI_MOE_ENV=""
    set -x
    if [[ -n "$MORI_MOE_MAX_INPUT_TOKENS_PREFILL" ]]; then
        PREFILL_MORI_MOE_ENV="SGLANG_MORI_MOE_MAX_INPUT_TOKENS=${MORI_MOE_MAX_INPUT_TOKENS_PREFILL}"
    fi
    set +x
    PREFILL_CMD="SGLANG_MORI_COMBINE_DTYPE=${MORI_COMBINE_DTYPE_PREFILL} ${PREFILL_SDMA_ENV} ${PREFILL_MORI_MOE_ENV} SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK_PREFILL:-${MORI_MAX_DISPATCH_TOKENS_PREFILL}} MORI_IO_SQ_BACKOFF_TIMEOUT_US=${MORI_IO_SQ_BACKOFF_TIMEOUT_US} MORI_IO_QP_MAX_SEND_WR=${MORI_IO_QP_MAX_SEND_WR} ${LAUNCH_PREFIX:-} python3 -m sglang.launch_server \
        --model-path $MODEL_DIR/${MODEL_NAME} \
        --disaggregation-mode prefill \
        --disaggregation-ib-device ${IBDEVICES} \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code \
        ${PREFILL_SERVER_CONFIG} "

    if [ "$PREFILL_NODES_PER_WORKER" -gt 1 ]; then
        rank=$((NODE_RANK % PREFILL_NODES_PER_WORKER))
        prefill_idx=$((NODE_RANK / PREFILL_NODES_PER_WORKER))
        PREFILL_CMD="$PREFILL_CMD --dist-init-addr ${PREFILL_HEADNODE_URLS[$prefill_idx]} --nnodes ${PREFILL_NODES_PER_WORKER} --node-rank $rank"
    fi

    dump_cmd "PREFILL (rank ${NODE_RANK})" "$PREFILL_CMD"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $PREFILL_CMD"
    else
        set -x
        # setsid isolates the server tree so teardown can group-kill python + TP-scheduler
        # children; otherwise they hold the tee pipe and the container never exits.
        setsid bash -c "$PREFILL_CMD" \
            > >(tee /run_logs/slurm_job-${SLURM_JOB_ID}/prefill_${host_name}.log >/dev/null) 2>&1 &
        set +x
        prefill_pid=$!
        prefill_pgid=$(ps -o pgid= -p "$prefill_pid" 2>/dev/null | tr -d ' ')
        : "${prefill_pgid:=$prefill_pid}"
    fi

    echo "Waiting for proxy server to be up..."
    BARRIER_CMD="python3 $SGLANG_WS_PATH/sync.py barrier \
        --node-ips ${NODE0_ADDR} \
        --node-ports 30000 \
        --wait-for-all-ports \
        --timeout ${SYNC_BARRIER_TIMEOUT}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BARRIER_CMD"
    else
        wait_or_die "$prefill_pid" bash -c "$BARRIER_CMD" || exit 1
    fi

    echo "Waiting until proxy server closes..."
    WAIT_CMD="python3 $SGLANG_WS_PATH/sync.py wait \
        --remote-ip ${NODE0_ADDR} \
        --remote-port 30000"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $WAIT_CMD"
    else
        wait_or_die "$prefill_pid" bash -c "$WAIT_CMD" || exit 1
    fi

    echo "Killing the rank $NODE_RANK prefill server"

    if [[ "$DRY_RUN" -eq 0 ]]; then
        # Group-kill so TP-scheduler children release the tee pipe and the container exits.
        kill -TERM -"${prefill_pgid:-$prefill_pid}" 2>/dev/null || true
    fi

else
    RANK=$((NODE_RANK - xP * PREFILL_NODES_PER_WORKER))
    echo "${host_name}:${host_ip} is Decode Node (Model: ${MODEL_NAME:-'default'})"
    echo "Using decode config: $DECODE_SERVER_CONFIG"
    echo "Decode node rank: $RANK"
    echo "Decode parallelism: TP=${DECODE_TP_SIZE}, EP enabled: ${DECODE_ENABLE_EP}, DP enabled: ${DECODE_ENABLE_DP}"

    CMD_DUMP="/run_logs/slurm_job-${SLURM_JOB_ID}/commands_${host_name}.txt"
    dump_cmd() { echo -e "\n# ── $1 ──\n$2" >> "$CMD_DUMP"; }
    echo "# Commands dump — $(date -u '+%Y-%m-%d %H:%M:%S UTC')" > "$CMD_DUMP"
    echo "# Host: ${host_name} (${host_ip})  Node rank: ${NODE_RANK}" >> "$CMD_DUMP"

    DECODE_MORI_MOE_ENV=""
    set -x
    if [[ -n "$MORI_MOE_MAX_INPUT_TOKENS_DECODE" ]]; then
        DECODE_MORI_MOE_ENV="SGLANG_MORI_MOE_MAX_INPUT_TOKENS=${MORI_MOE_MAX_INPUT_TOKENS_DECODE}"
    fi
    set +x

    # Agentic trace replay does not reproduce real token-by-token traffic, so measured
    # MTP acceptance there is not representative (PR #2309 review:
    # https://github.com/SemiAnalysisAI/InferenceX/pull/2309#pullrequestreview-4778348624).
    # Per golden_al_distribution/README.md, agentic throughput runs simulate acceptance at
    # the model's golden AL (golden_al_distribution/dsv4_mtp.yaml, thinking_on); eval
    # runs need real acceptance so GSM8K reflects actual MTP behavior.
    DECODE_SIM_ACC_ENV=""
    if [[ "$DECODE_MTP_SIZE" -gt 0 ]] && { [[ "${IS_AGENTIC:-0}" == "1" ]] || [[ "${IS_AGENTIC:-}" == "true" ]]; }; then
        if [[ "${EVAL_ONLY:-false}" == "true" ]] || [[ "${RUN_EVAL:-false}" == "true" ]]; then
            echo "[INFO] Eval mode: synthetic MTP disabled (using real acceptance)"
        else
            DSV4_GOLDEN_AL=""
            case "${MODEL_NAME}:${DECODE_MTP_SIZE}" in
                *DeepSeek-V4*:1) DSV4_GOLDEN_AL=1.79 ;;
                *DeepSeek-V4*:2) DSV4_GOLDEN_AL=2.27 ;;
                *DeepSeek-V4*:3) DSV4_GOLDEN_AL=2.49 ;;
            esac
            if [[ -n "$DSV4_GOLDEN_AL" ]]; then
                DECODE_SIM_ACC_ENV="SGLANG_SIMULATE_ACC_LEN=${DSV4_GOLDEN_AL} SGLANG_SIMULATE_ACC_METHOD=match-expected SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token"
            else
                echo "WARNING: agentic MTP run (model=${MODEL_NAME}, DECODE_MTP_SIZE=${DECODE_MTP_SIZE}) has no golden AL wired in server_sglang.sh -- falling back to real (unsimulated, non-representative) acceptance. Add a case in server_sglang.sh and golden_al_distribution/ before shipping this arm. See golden_al_distribution/README.md." >&2
            fi
        fi
    fi

    DECODE_CMD="SGLANG_MORI_COMBINE_DTYPE=${MORI_COMBINE_DTYPE_DECODE} ${DECODE_MORI_MOE_ENV} SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK_DECODE:-${MORI_MAX_DISPATCH_TOKENS_DECODE}} MORI_IO_SQ_BACKOFF_TIMEOUT_US=${MORI_IO_SQ_BACKOFF_TIMEOUT_US} MORI_IO_QP_MAX_SEND_WR=${MORI_IO_QP_MAX_SEND_WR} ${DECODE_SIM_ACC_ENV} ${LAUNCH_PREFIX:-} python3 -m sglang.launch_server \
        --model-path ${MODEL_DIR}/${MODEL_NAME} \
        --disaggregation-mode decode \
        --disaggregation-ib-device ${IBDEVICES} \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code \
        ${DECODE_SERVER_CONFIG} "

    if [ "$DECODE_NODES_PER_WORKER" -gt 1 ]; then
        rank=$((RANK % DECODE_NODES_PER_WORKER))
        decode_idx=$((RANK / DECODE_NODES_PER_WORKER))
        DECODE_CMD="$DECODE_CMD --dist-init-addr ${DECODE_HEADNODE_URLS[$decode_idx]} --nnodes ${DECODE_NODES_PER_WORKER} --node-rank $rank"
    fi

    dump_cmd "DECODE (rank ${NODE_RANK})" "$DECODE_CMD"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $DECODE_CMD"
    else
        set -x
        # setsid isolates the server tree so teardown can group-kill python + TP-scheduler
        # children; otherwise they hold the tee pipe and the container never exits.
        setsid bash -c "$DECODE_CMD" \
            > >(tee /run_logs/slurm_job-${SLURM_JOB_ID}/decode_${host_name}.log >/dev/null) 2>&1 &

        set +x
        decode_pid=$!
        decode_pgid=$(ps -o pgid= -p "$decode_pid" 2>/dev/null | tr -d ' ')
        : "${decode_pgid:=$decode_pid}"
    fi


    echo "Waiting for proxy server to be up..."
    BARRIER_CMD="python3 $SGLANG_WS_PATH/sync.py barrier \
        --node-ips ${NODE0_ADDR} \
        --node-ports 30000 \
        --wait-for-all-ports \
        --timeout ${SYNC_BARRIER_TIMEOUT}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BARRIER_CMD"
    else
        wait_or_die "$decode_pid" bash -c "$BARRIER_CMD" || exit 1
    fi


    echo "Waiting until proxy server closes..."
    WAIT_CMD="python3 $SGLANG_WS_PATH/sync.py wait \
        --remote-ip ${NODE0_ADDR} \
        --remote-port 30000"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $WAIT_CMD"
    else
        wait_or_die "$decode_pid" bash -c "$WAIT_CMD" || exit 1
    fi

    echo "Killing the rank $RANK decode server"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        # Group-kill so TP-scheduler children release the tee pipe and the container exits.
        kill -TERM -"${decode_pgid:-$decode_pid}" 2>/dev/null || true
    fi

fi

echo "Script completed successfully"
exit 0
