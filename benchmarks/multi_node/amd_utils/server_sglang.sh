#!/bin/bash
# SGLang Disaggregated Server Launcher with Model-Specific Configurations
# =============================================================================

# =============================================================================
# Environment Configuration
# =============================================================================

NODE0_ADDR="${NODE0_ADDR:-localhost}"
NODE_RANK="${NODE_RANK:-0}"
MODEL_DIR="${MODEL_DIR:-}"
MODEL_NAME="${MODEL_NAME:-}"

xP="${xP:-1}" #-> Number of Prefill Workers
yD="${yD:-1}" #-> Number of Decode Workers

IPADDRS="${IPADDRS:-localhost}"
HEADNODE_PORT="${HEADNODE_PORT:-20000}"
# Parallelism Configuration
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-8}"
PREFILL_ENABLE_EP="${PREFILL_ENABLE_EP:-true}"
PREFILL_ENABLE_DP="${PREFILL_ENABLE_DP:-true}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-8}"
DECODE_ENABLE_EP="${DECODE_ENABLE_EP:-true}"
DECODE_ENABLE_DP="${DECODE_ENABLE_DP:-true}"
DECODE_MTP_SIZE="${DECODE_MTP_SIZE:-0}"

# Benchmark Configuration
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-1024}"
BENCH_RANDOM_RANGE_RATIO="${BENCH_RANDOM_RANGE_RATIO:-1}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-inf}"
BENCH_NUM_PROMPTS_MULTIPLIER="${BENCH_NUM_PROMPTS_MULTIPLIER:-10}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-512}"

# Extract the maximum concurrency from the x-delimited list
BENCH_MAX_CONC_VALUE=$(echo "$BENCH_MAX_CONCURRENCY" | tr 'x' '\n' | sort -n | tail -1)

# Dry Run for debugging purpose
DRY_RUN="${DRY_RUN:-0}"

# GPU count (expandable for different hardware)
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"


# =============================================================================
# Dependencies and Environment Setup
# =============================================================================
source $SGLANG_WS_PATH/setup_deps.sh
source $SGLANG_WS_PATH/env.sh

host_ip=$(ip route get 1.1.1.1 | awk '/src/ {print $7}')
host_name=$(hostname)

# MORI_RDMA_TC configuration (optional)
# If set by runner, use it for RDMA traffic class configuration
# If not set, RDMA operations will proceed without QoS/traffic class settings
if [[ -n "${MORI_RDMA_TC}" ]]; then
    echo "[INFO] Using MORI_RDMA_TC=$MORI_RDMA_TC for RDMA traffic class configuration"
    echo "[INFO] Host '$host_name' configured with MORI_RDMA_TC=$MORI_RDMA_TC"
else
    echo "[INFO] MORI_RDMA_TC not set. Skipping RDMA traffic class configuration."
    echo "[INFO] This is normal for clusters without QoS requirements."
fi

# =============================================================================
# Model-Specific Configuration from YAML
# =============================================================================
MODELS_YAML="${SGLANG_WS_PATH}/models.yaml"

if [[ ! -f "$MODELS_YAML" ]]; then
    echo "ERROR: models.yaml not found at $MODELS_YAML"
    exit 1
fi

# Load model config via inline Python (PyYAML is available in SGLang containers)
# Formula evaluation (e.g. "SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK * TP * xP")
# is done here in Python to avoid bash glob-expanding the * characters.
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

# Compute DP-dependent prefill parameters
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

# When both DP and EP are enabled, override max-running-requests with max bench concurrency
if [[ "$PREFILL_ENABLE_DP" == "true" ]] && [[ "$PREFILL_ENABLE_EP" == "true" ]]; then
    prefill_max_running_requests=$BENCH_MAX_CONC_VALUE
    prefill_dp_ranks=$PREFILL_TP_SIZE
    # MORI_MAX_DISPATCH_TOKENS_PREFILL stays at 8192 (no change)
    # MORI_MOE_MAX_INPUT_TOKENS_PREFILL=$((MORI_MAX_DISPATCH_TOKENS_PREFILL * prefill_dp_ranks / 2))
    echo "[DP+EP override] Prefill: max-running-requests=$prefill_max_running_requests, MOE_MAX_INPUT=$MORI_MOE_MAX_INPUT_TOKENS_PREFILL"
fi

# Compute DP-dependent decode parameters (3-way: DP > EP-only > no_dp)
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
# In PD-disaggregation the decode must admit requests against the SAME context
# length as prefill; otherwise decode accepts over-length requests that prefill
# rejects, and those requests hang forever waiting for a KV transfer that never
# comes (the 8k1k conc-500 straggler). Fall back to the prefill value if the
# decode context_length is not set in the model config, so the two always agree.
if [[ -z "$decode_context_length" ]]; then
    decode_context_length=$prefill_context_length
fi

# When both DP and EP are enabled, override max-running-requests and dispatch tokens
if [[ "$DECODE_ENABLE_DP" == "true" ]] && [[ "$DECODE_ENABLE_EP" == "true" ]]; then
    decode_max_running_requests=$BENCH_MAX_CONC_VALUE
    decode_dp_ranks=$DECODE_TP_SIZE
    MORI_MAX_DISPATCH_TOKENS_DECODE=$((BENCH_MAX_CONC_VALUE / decode_dp_ranks))
    # MORI_MOE_MAX_INPUT_TOKENS_DECODE=$((MORI_MAX_DISPATCH_TOKENS_DECODE * decode_dp_ranks * 7 / 10))
    # Update derived variable
    SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD=$((MORI_MAX_DISPATCH_TOKENS_DECODE * 2))
    export SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD
    echo "[DP+EP override] Decode: max-running-requests=$decode_max_running_requests, DISPATCH_TOKENS=$MORI_MAX_DISPATCH_TOKENS_DECODE, MOE_MAX_INPUT=$MORI_MOE_MAX_INPUT_TOKENS_DECODE, INTER_KERNEL_SWITCH=$SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD"
fi

# Build the composed config strings (equivalent to the old MODEL_PREFILL_CONFIGS / MODEL_DECODE_CONFIGS)
# disable_cuda_graph (model-level) routes prefill to --disable-cuda-graph instead of --cuda-graph-bs.
if [[ "$PREFILL_DISABLE_CUDA_GRAPH" == "True" ]] || [[ "$PREFILL_DISABLE_CUDA_GRAPH" == "true" ]]; then
    PREFILL_MODE_FLAGS="--mem-fraction-static ${PREFILL_MEM_FRACTION_STATIC} --max-running-requests ${prefill_max_running_requests} --chunked-prefill-size ${prefill_chunked_prefill_size} --disable-cuda-graph "
else
    PREFILL_MODE_FLAGS="--mem-fraction-static ${PREFILL_MEM_FRACTION_STATIC} --max-running-requests ${prefill_max_running_requests} --chunked-prefill-size ${prefill_chunked_prefill_size} --cuda-graph-bs ${prefill_cuda_graph_bs[*]} "
fi

if [[ "$PREFILL_DISABLE_RADIX_CACHE" == "True" ]] || [[ "$PREFILL_DISABLE_RADIX_CACHE" == "true" ]]; then
    PREFILL_MODE_FLAGS="$PREFILL_MODE_FLAGS --disable-radix-cache"
fi
# Agentic runs: keep radix/prefix cache enabled by replacing --disable-radix-cache with empty.
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
    # MORI_MOE_MAX_INPUT_TOKENS_DECODE=$((MORI_MOE_MAX_INPUT_TOKENS_DECODE * (DECODE_MTP_SIZE + 1)))
fi

# =============================================================================
# Cluster Topology Configuration
# =============================================================================
IFS=',' read -ra IP_ARRAY <<< "$IPADDRS"

# Ceiling division by GPUS_PER_NODE for nodes-per-worker
PREFILL_NODES_PER_WORKER=$(((PREFILL_TP_SIZE + 7) / GPUS_PER_NODE))
DECODE_NODES_PER_WORKER=$(((DECODE_TP_SIZE + 7) / GPUS_PER_NODE))
NODE_OFFSET=$((PREFILL_NODES_PER_WORKER * xP))

# Build prefill arguments dynamically based on xP
PREFILL_HEADNODE_URLS=()
PREFILL_ARGS=""
# Per-worker Prometheus /metrics endpoints (port 8000) for aiperf's
# --server-metrics scrape. The router on :30000 does not serve Prometheus, so
# aiperf must scrape each prefill/decode worker directly (see ENABLE_METRICS).
SERVER_METRICS_URLS=()
# Per-worker base URLs (port 8000) for direct cache flushing between
# concurrency points. The router (:30000) does not fan /flush_cache out, so
# trace_replay.sh must POST to each prefill/decode worker directly.
SERVER_FLUSH_URLS=()
for i in $(seq 0 $((xP - 1))); do
    prefill_idx=$((i * PREFILL_NODES_PER_WORKER))
    PREFILL_HEADNODE_URLS[$i]="${IP_ARRAY[$prefill_idx]}:${HEADNODE_PORT}"
    PREFILL_ARGS="$PREFILL_ARGS --prefill http://${IP_ARRAY[$prefill_idx]}:8000"
    SERVER_METRICS_URLS+=("http://${IP_ARRAY[$prefill_idx]}:8000/metrics")
    SERVER_FLUSH_URLS+=("http://${IP_ARRAY[$prefill_idx]}:8000")
done

# Build decode arguments dynamically based on yD
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

# =============================================================================
# Configuration Builder Functions
# =============================================================================

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

    # Calculate EP and DP sizes based on enable flags
    local ep_size=1
    local dp_size=1

    if [[ "$enable_ep" == "true" ]]; then
        ep_size=$tp_size
    fi

    if [[ "$enable_dp" == "true" ]]; then
        dp_size=$tp_size
    fi

    # Build parallelism arguments
    local parallel_args="--tp-size ${tp_size}"

    if [[ "$enable_ep" == "true" ]]; then
        parallel_args="$parallel_args --ep-size ${ep_size}"
    fi

    if [[ "$enable_dp" == "true" ]]; then
        parallel_args="$parallel_args --dp-size ${dp_size}"
    fi

    # Get model-specific configuration from YAML-loaded variables
    local base_config
    base_config="$(apply_kv_p2p_transfer_override "$MODEL_BASE_FLAGS")"
    local mtp_config=""
    local dp_config=""
    local ep_config=""
    local specific_config=""

    # MTP config (only if MTP is enabled and mode is decode)
    if [ "$decode_mtp_size" -gt 0 ]; then
        mtp_config="${MODEL_MTP_FLAGS} --speculative-num-steps ${decode_mtp_size} --speculative-num-draft-tokens $((decode_mtp_size + 1))"
    fi

    # DP config (only if DP is enabled)
    if [[ "$enable_dp" == "true" ]]; then
        dp_config="$MODEL_DP_FLAGS"
    fi

    # EP config (only if EP is enabled): a2a backend, deepep mode, ep-dispatch algo.
    # With ep=1 (EP disabled) these are dropped, so the MoE runs tensor-parallel (TP)
    # instead of expert-parallel — even when dp-attention is on.
    if [[ "$enable_ep" == "true" ]]; then
        ep_config="$MODEL_EP_FLAGS"
    fi

    # Mode-specific config
    if [[ "$mode" == "prefill" ]]; then
        specific_config="$PREFILL_MODE_FLAGS"
    elif [[ "$mode" == "decode" ]]; then
        specific_config="$DECODE_MODE_FLAGS"
    fi

    # Combine: parallel args + base config + ep config + mtp config (decode only) + dp config + specific config
    local full_config="$parallel_args"
    if [[ -n "$base_config" ]]; then
        full_config="$full_config $base_config"
    fi
    if [[ -n "$ep_config" ]]; then
        full_config="$full_config $ep_config"
    fi
    if [[ -n "$mtp_config" ]] && [[ "$mode" == "decode" ]]; then
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

# Build complete server configurations
PREFILL_SERVER_CONFIG=$(build_server_config "prefill" "$MODEL_NAME" "$PREFILL_TP_SIZE" "$PREFILL_ENABLE_EP" "$PREFILL_ENABLE_DP" "$DECODE_MTP_SIZE")
DECODE_SERVER_CONFIG=$(build_server_config "decode" "$MODEL_NAME" "$DECODE_TP_SIZE" "$DECODE_ENABLE_EP" "$DECODE_ENABLE_DP" "$DECODE_MTP_SIZE")

# Expose Prometheus /metrics on the servers when requested (ENABLE_METRICS=1).
if [[ "${ENABLE_METRICS:-0}" == "1" ]]; then
    [[ "$PREFILL_SERVER_CONFIG" != *"--enable-metrics"* ]] && PREFILL_SERVER_CONFIG="$PREFILL_SERVER_CONFIG --enable-metrics"
    [[ "$DECODE_SERVER_CONFIG" != *"--enable-metrics"* ]] && DECODE_SERVER_CONFIG="$DECODE_SERVER_CONFIG --enable-metrics"
fi

if [[ -n "$MODEL_NAME" ]]; then
    echo "Using model-specific configuration for: $MODEL_NAME"
fi

# sync.py barrier timeout for server-up (port 8000). DSV4 needs more headroom.
# Override via SYNC_BARRIER_TIMEOUT if needed.
if [[ -z "${SYNC_BARRIER_TIMEOUT:-}" ]]; then
    case "${MODEL_NAME}" in
        *DeepSeek-V4*) SYNC_BARRIER_TIMEOUT=3000 ;;
        *) SYNC_BARRIER_TIMEOUT=1800 ;;
    esac
fi
echo "SYNC_BARRIER_TIMEOUT=${SYNC_BARRIER_TIMEOUT}s (model=${MODEL_NAME:-unset})"

# =============================================================================
# Optional KV cache offloading (HiCache) — enabled when
# KV_OFFLOADING != none AND KV_OFFLOAD_BACKEND == hicache.
# HiCache extends RadixAttention, so radix cache MUST stay on (drop
# --disable-radix-cache). The --hicache-* flags are appended to BOTH the
# prefill and decode server configs.
# =============================================================================
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

    # Layout / IO backend / write policy are backend-specific:
    #   mooncake L3: page_first_direct + the "direct" IO backend (the Mooncake
    #     store maps a page-contiguous segment for RDMA/zero-copy).  This layout
    #     asserts host_pool > device_pool, so it needs a large CPU-DRAM budget.
    #   L2-only (CPU DRAM): layer_first + the "kernel" IO backend.  layer_first
    #     has no host>device constraint (the "direct" IO backend REQUIRES a
    #     page_first layout, so it cannot be paired with layer_first).
    if [[ "$HICACHE_STORAGE_BACKEND" == "mooncake" ]]; then
        HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first}"
        HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
        HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
    else
        HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first_direct}"
        HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
        HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
    fi

    # Mooncake master/connection settings (used only when storage=mooncake).
    # The master runs once on node 0; every prefill/decode server connects to
    # it via NODE0_ADDR so it is reachable across nodes.
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

    # Emit the --hicache-storage-backend flags (empty unless mooncake).  The
    # extra-config JSON is single-quoted so it survives the later `eval` of the
    # launch command as a single argument.
    build_storage_flags() {
        [[ "$HICACHE_STORAGE_BACKEND" != "mooncake" ]] && return 0
        local extra="{\"master_server_address\": \"${MC_MASTER_ADDR}\", \"protocol\": \"${MC_PROTOCOL}\", \"device_name\": \"${MC_DEVICE}\", \"local_hostname\": \"${host_ip}\", \"global_segment_size\": \"${MC_GLOBAL_SEG}\", \"metadata_server\": \"${MC_METADATA_SERVER}\", \"check_server\": false}"
        echo "--hicache-storage-backend mooncake --hicache-storage-backend-extra-config '${extra}' --enable-metrics --enable-cache-report"
    }

    # HiCache capacity. Prefer an absolute per-rank pool derived from the
    # per-node DRAM budget computed by the sweep generator (enforcement); fall
    # back to --hicache-ratio (relative to the GPU KV pool) when no budget is
    # provided, keeping configs that predate the budget unchanged.
    # FORCE_HICACHE_RATIO lets a recipe opt into ratio-based sizing without
    # unsetting TOTAL_CPU_DRAM_GB — that var is also the shared client-side
    # gate (benchmark_lib.sh requires it whenever KV_OFFLOADING=dram) and is
    # forwarded verbatim into client.env below, so unsetting it here would
    # make the aiperf client container fail its own env validation before
    # ever sending a request.
    HICACHE_RATIO="${HICACHE_RATIO:-5}"
    HICACHE_SIZING_FLAGS="--hicache-ratio ${HICACHE_RATIO}"
    # DeepSeek V4's hybrid HiCache pool rejects --hicache-size (requires
    # --hicache-ratio), so the absolute per-node budget cannot be applied to it.
    # See sglang _deepseek_v4_num_host_pages() (raises ValueError when
    # server_args.hicache_size > 0):
    # https://github.com/sgl-project/sglang/blob/9dd57ef8c48e2cd82292d849f01e2130c5203e67/python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py#L262-L266
    # FORCE_HICACHE_RATIO additionally lets a recipe opt into ratio-based sizing
    # for any other model without unsetting TOTAL_CPU_DRAM_GB (see comment above).
    if [[ "${FORCE_HICACHE_RATIO:-0}" != "1" && -n "${TOTAL_CPU_DRAM_GB:-}" && "${TOTAL_CPU_DRAM_GB}" -gt 0 && "${MODEL_NAME}" != *DeepSeek-V4* ]]; then
        # TOTAL_CPU_DRAM_GB is the prefill worker's per-node budget (only prefill
        # offloads KV to CPU DRAM today); --hicache-size is per rank per host
        # pool. A prefill server may span nodes (PREFILL_TP_SIZE is its total
        # ranks), so divide by the ranks that land on one node.
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

    # Prefill always gets HiCache.
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

# =============================================================================
# Container Synchronization
# =============================================================================

# sync.py barrier/health-barrier exits 1 on timeout (and prints which
# node/port never became ready), but without an explicit check here the
# script would silently continue past a timed-out barrier -- printing a
# misleading "success" message and launching the next stage against
# servers/routers that never actually came up, instead of failing fast.
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


# =============================================================================
# Node Role Assignment and Server Launch
# =============================================================================

# Run a blocking command while watching the local server PID. If the server dies
# (crash / OOM / killed) the blocking command is aborted and we return non-zero,
# so the srun task exits non-zero and SLURM's --kill-on-bad-exit tears the whole
# job down in seconds instead of waiting out the ~1800s barrier timeout.
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

    # Dump all resolved commands to a text file for debugging / reproducibility.
    CMD_DUMP="/run_logs/slurm_job-${SLURM_JOB_ID}/commands_${host_name}.txt"
    dump_cmd() { echo -e "\n# ── $1 ──\n$2" >> "$CMD_DUMP"; }
    echo "# Commands dump — $(date -u '+%Y-%m-%d %H:%M:%S UTC')" > "$CMD_DUMP"
    echo "# Host: ${host_name} (${host_ip})  Node rank: ${NODE_RANK}" >> "$CMD_DUMP"
    echo "# Model: ${MODEL_NAME}  Image: ${DOCKER_IMAGE_NAME:-unknown}" >> "$CMD_DUMP"

    # Start the Mooncake store master (L3 HiCache backend) on node 0 only.
    # All prefill/decode servers connect to it via NODE0_ADDR:MC_MASTER_PORT.
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
            # Fail loudly on a port collision. On shared nodes the Mooncake RPC
            # port may already be taken by another user's master; in that case the
            # metrics-port health check below can still pass against the foreign
            # master while our RPC port is dead, and the prefill then hangs.
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

    # start the head prefill server
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
        # Launch under `setsid` so the server (python + its TP-scheduler
        # children) sits in a dedicated process group; teardown can then
        # `kill -- -$pgid` the WHOLE tree. Killing $prefill0_pid alone leaves
        # children holding the process-sub tee's pipe, so the container's outer
        # `| tee` never gets EOF and the container never exits (srun/CI hangs).
        # Process substitution (not `| tee`) keeps $! as the setsid group leader,
        # not tee's. Mirrors the router launch below.
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
        # Agentic router config (main): long-context prefills can look unhealthy to
        # the default circuit breaker during a concurrent burst. Disable the breaker
        # and relax health-check sensitivity so a busy-but-alive worker is not
        # ejected. cache_aware prefill routing exploits HiCache/radix prefix reuse
        # across the agentic trace; round_robin decode keeps the single decode worker
        # fed evenly. Override via ROUTER_RESILIENCE_FLAGS / ROUTER_POLICY_FLAGS.
        ROUTER_RESILIENCE_FLAGS="${ROUTER_RESILIENCE_FLAGS:---disable-circuit-breaker --health-failure-threshold 100 --health-check-timeout-secs 600 --health-check-interval-secs 30}"
        # server_sglang.sh previously read ROUTER_PREFILL_POLICY, but the recipe
        # scripts export PREFILL_ROUTER_POLICY, so the recipe's policy override was
        # silently ignored and the router always fell back to this hardcoded
        # default. Also comment out ROUTER_DECODE_POLICY for now (superseded by
        # --dp-aware below).
        ROUTER_PREFILL_POLICY="${PREFILL_ROUTER_POLICY:-consistent_hashing}"
        # ROUTER_DECODE_POLICY="${ROUTER_DECODE_POLICY:-round_robin}"
        ROUTER_CACHE_THRESHOLD="${ROUTER_CACHE_THRESHOLD:-0.3}"
        ROUTER_BALANCE_ABS_THRESHOLD="${ROUTER_BALANCE_ABS_THRESHOLD:-2}"
        ROUTER_BALANCE_REL_THRESHOLD="${ROUTER_BALANCE_REL_THRESHOLD:-1.1}"
        ROUTER_POLICY_FLAGS="${ROUTER_POLICY_FLAGS:---policy ${ROUTER_PREFILL_POLICY} --dp-aware --cache-threshold ${ROUTER_CACHE_THRESHOLD} --balance-abs-threshold ${ROUTER_BALANCE_ABS_THRESHOLD} --balance-rel-threshold ${ROUTER_BALANCE_REL_THRESHOLD}}"
    else
        # DI router config (8k1k branch, run 28696443568): with defaults the per-worker
        # circuit stays OPEN for cb-timeout-duration-secs=60 before a half-open retrial.
        # In that run every request 503'd from request #1 ("all circuits open or
        # unhealthy") for ~31s and lm_eval (max_retries=5) then gave up -- i.e. the
        # circuit was still open when the client budget ran out, so 0 result files were
        # produced. Shortening the open->half-open window (and letting the router itself
        # retry a failed worker selection) lets a transient trip re-close INSIDE the
        # client retry budget instead of nuking the whole eval. The breaker stays fully
        # ENABLED (thresholds unchanged); this only speeds recovery. Override via
        # ROUTER_CB_ARGS / ROUTER_POLICY_FLAGS.
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
        # sgl-router (Rust/tracing) emits ANSI color codes. NO_COLOR asks it to
        # skip them at the source; the sed strip guarantees a clean file even if
        # it doesn't honor NO_COLOR. Both branches use process substitution so
        # $! stays the router pid, not sed's/tee's pid.
        #
        # Newer sglang-router (>=0.5.14) spawns the actual Rust worker
        # (`sglang::router`, which binds :30000) as a child and lets the python
        # launcher exit, so the worker reparents to init. It KEEPS its process
        # group, though. We therefore launch under `setsid` to isolate the
        # launcher+worker in a dedicated process group and record that pgid, so
        # teardown can `kill -- -$proxy_pgid` the whole group even after the
        # launcher is gone. `kill $proxy_pid` alone would miss the worker.
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

        # Wait for router to be ready via health endpoint
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

        # ---- End-to-end router readiness canary (run 28696443568) ----
        # The /readiness barrier above only proves the router PROCESS is up; it does
        # NOT prove the router can reach a prefill worker and complete a generation.
        # In that run the eval fired the instant /readiness passed and EVERY request
        # 503'd ("No available prefill workers (all circuits open or unhealthy)") from
        # request #1 -> lm_eval gave up -> 0 result files -> "Verify eval scores" failed.
        # Gate the benchmark on ONE successful generation THROUGH the router so the eval
        # never starts against a router whose prefill path is not yet actually serving.
        #
        # Run the poll loop under wait_or_die (like the barriers above) rather than
        # inline: the canary is the FIRST real generation through prefill, so a
        # prefill crash right after /readiness passes is plausible. Without
        # wait_or_die watching prefill0_pid, that just looks like repeated 503s and
        # the loop burns the full ROUTER_CANARY_TIMEOUT (default 600s) instead of
        # detecting the dead pid and aborting in ~5s.
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

    # Export IS_MTP based on whether MTP is enabled
    if [ "$DECODE_MTP_SIZE" -gt 0 ]; then
        export IS_MTP=true
    else
        export IS_MTP=false
    fi

    # Select the benchmark runner.
    # IS_AGENTIC=1/true  → agentic trace replay (trace_replay.sh)
    # IS_AGENTIC unset/0 → fixed-seq-len throughput benchmark (bench.sh)
    if [[ "${IS_AGENTIC:-0}" == "1" || "${IS_AGENTIC:-}" == "true" ]]; then
        # Point aiperf's server-metrics scrape at the per-worker Prometheus
        # /metrics endpoints. The router (:30000) that aiperf auto-detects from
        # --url does not expose Prometheus, so without this the scrape finds no
        # reachable endpoint and all server-side cache/KV fields come out null.
        # Only set it when the workers were actually started with --enable-metrics.
        if [[ "${ENABLE_METRICS:-0}" == "1" && "${#SERVER_METRICS_URLS[@]}" -gt 0 ]]; then
            AIPERF_SERVER_METRICS_URLS=$(IFS=,; echo "${SERVER_METRICS_URLS[*]}")
            export AIPERF_SERVER_METRICS_URLS
            echo "AIPERF_SERVER_METRICS_URLS=${AIPERF_SERVER_METRICS_URLS}"
        fi
        # Per-worker base URLs for cache flushing between concurrency points.
        # trace_replay.sh consults these when CLEAR_CACHE_BETWEEN_CONC=1.
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
        # Separate client image (node-0 sibling container): run the aiperf trace
        # replay in its own sibling container built from CLIENT_IMAGE (which ships
        # a pre-baked aiperf + deps) instead of rebuilding the aiperf venv inside
        # this server container. The server/router stay up in this container while
        # the client container drives the benchmark against the router on
        # localhost (--network host). job.slurm mounts the host docker socket + CLI
        # into this container and forwards HOST_REPO_DIR / HOST_MODEL_DIR /
        # HOST_BENCH_LOGS / CLIENT_CONT_NAME so the sibling can be launched here.
        CLIENT_ENV_FILE="/run_logs/slurm_job-${SLURM_JOB_ID}/client.env"
        mkdir -p "/run_logs/slurm_job-${SLURM_JOB_ID}"
        # Forward the benchmark-relevant env (incl. runtime-computed metrics/flush
        # URLs) to the client container; override the few paths/flags that differ
        # inside the pre-baked image. Unset vars are skipped, so the client keeps
        # its own defaults for anything not exported here.
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
                      AIPERF_AGENTIC_CACHE_WARMUP_DURATION AIPERF_UNSAFE_OVERRIDE \
                      AIPERF_TRAJECTORY_START_MIN_RATIO AIPERF_TRAJECTORY_START_MAX_RATIO \
                      AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES ROUTER_PORT TQDM_MININTERVAL; do
                if [[ -n "${!_v+x}" ]]; then
                    _val="${!_v}"
                    # docker --env-file requires one KEY=VALUE per line with no
                    # embedded newlines; KV_OFFLOAD_BACKEND_METADATA carries
                    # pretty-printed multi-line JSON, which otherwise splits
                    # into unparseable lines (e.g. '"name": "hicache",') and
                    # aborts the client container launch. Re-serialize it to
                    # compact single-line JSON (round-tripping through
                    # json.loads/json.dumps) instead of naively stripping
                    # newlines, so this stays correct even if a value ever
                    # contained a literal newline inside a string. Empty/
                    # "none"/"null" is the normal case when KV offloading is
                    # disabled (job.slurm always sets this var, even to ""),
                    # and must pass through untouched -- matching how
                    # optional_kv_offload_backend_metadata() in
                    # process_agentic_result.py treats those as "no metadata"
                    # rather than invalid JSON.
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
            # Do NOT pin AGENTIC_OUTPUT_DIR: it must default to /workspace (the
            # host repo mount == GITHUB_WORKSPACE) so the aggregated
            # ${RESULT_FILENAME}_conc<N>.json lands where the workflow guard globs
            # it. /workspace is bind-mounted writable, same as the co-located path.
            echo "HF_HOME=/run_logs/hf_cache"
            echo "MODEL_DIR=/models"
            # A pre-baked client image ships aiperf at CLIENT_AIPERF_VENV; when
            # unset (e.g. reusing the server image, which carries no pre-baked
            # venv), trace_replay builds aiperf on the fly from
            # /workspace/utils/aiperf — same as the co-located path.
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

    # Run evaluation if requested (before killing router)
    if [[ "${RUN_EVAL:-false}" == "true" ]]; then
        echo "Running lm-eval evaluation on Node 0..."

        # Health check: verify the router is still serving before running eval.
        # The throughput benchmark may have crashed/exhausted decode workers.
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
            # Must run from repo root so utils/evals/${task}.yaml resolves
            pushd /workspace

            # Source eval functions from benchmark_lib.sh
            source /workspace/benchmarks/benchmark_lib.sh

            # Use EVAL_CONC from workflow if set, otherwise fall back to max of conc list
            if [[ -n "${EVAL_CONC:-}" ]]; then
                export EVAL_CONCURRENT_REQUESTS="${EVAL_CONC}"
            else
                export EVAL_CONCURRENT_REQUESTS=$(echo "$BENCH_MAX_CONCURRENCY" | tr 'x' '\n' | sort -n | tail -1)
            fi

            # Override eval context length with model's configured context_length
            if [[ -n "$prefill_context_length" ]]; then
                export EVAL_MAX_MODEL_LEN="$prefill_context_length"
            fi

            if [[ "$DRY_RUN" -eq 1 ]]; then
                echo "DRY RUN: run_eval --framework lm-eval --port 30000 (conc=${EVAL_CONCURRENT_REQUESTS}, ctx=${EVAL_MAX_MODEL_LEN:-auto})"
            else
                # Run lm-eval against the router on port 30000
                run_eval --framework lm-eval --port 30000
                eval_rc=$?

                if [[ $eval_rc -ne 0 ]]; then
                    echo "ERROR: run_eval exited rc=$eval_rc; skipping metadata write and eval artifact staging" >&2
                    EVAL_FAILED=1
                else
                    # Set metadata env vars for append_lm_eval_summary
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
                    # IS_MULTINODE, FRAMEWORK, PRECISION, MODEL_PREFIX, RUNNER_TYPE,
                    # RESULT_FILENAME are already set via Docker -e flags from job.slurm

                    append_lm_eval_summary
                    # Files (meta_env.json, results*.json, sample*.jsonl) are now in /workspace

                    # Copy eval artifacts to run_logs for NFS extraction by runner
                    EVAL_COPY_DIR="/run_logs/slurm_job-${SLURM_JOB_ID}/eval_results"
                    mkdir -p "$EVAL_COPY_DIR"
                    for f in meta_env.json; do
                        [ -e "/workspace/$f" ] && cp -f "/workspace/$f" "$EVAL_COPY_DIR/"
                    done
                    # Use find for glob patterns to avoid "no match" errors
                    find /workspace -maxdepth 1 -name 'results*.json' -exec cp -f {} "$EVAL_COPY_DIR/" \;
                    find /workspace -maxdepth 1 -name 'sample*.jsonl' -exec cp -f {} "$EVAL_COPY_DIR/" \;

                    echo "Eval completed. Artifacts staged in $EVAL_COPY_DIR"
                fi
            fi

            popd
        fi
    fi

    # Copy benchmark results to BENCHMARK_LOGS_DIR (mounted from host)
    LOGS_OUTPUT="${BENCHMARK_LOGS_DIR:-/run_logs}/logs"
    mkdir -p "$LOGS_OUTPUT"

    if [[ "$DRY_RUN" -eq 0 ]]; then
        cp -r /run_logs/slurm_job-${SLURM_JOB_ID} "$LOGS_OUTPUT/"
        echo "Copied results to $LOGS_OUTPUT/slurm_job-${SLURM_JOB_ID}"
    fi

    echo "Killing the proxy server and prefill server"

    if [[ "$DRY_RUN" -eq 0 ]]; then
        # Kill the router's entire process group (isolated via setsid at launch).
        # The python launcher (proxy_pid) has usually already exited after
        # spawning the detached Rust worker; the worker reparents to init but
        # stays in this process group, so a group-kill reliably closes :30000.
        # `kill $proxy_pid` alone misses the worker and hangs decode/prefill.
        kill -TERM -"${proxy_pgid:-$proxy_pid}" 2>/dev/null || true
        # Group-kill the prefill server tree (setsid at launch) so its
        # TP-scheduler children die too and release the process-sub tee ->
        # the container's outer `| tee` gets EOF and the container can exit.
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
        # setsid isolates the server tree in its own process group so teardown
        # can group-kill it (python + TP-scheduler children); otherwise the
        # children hold the process-sub tee's pipe and the container never exits.
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
        # Group-kill the whole server tree (setsid at launch) so TP-scheduler
        # children die and the process-sub tee gets EOF -> container can exit.
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
    DECODE_CMD="SGLANG_MORI_COMBINE_DTYPE=${MORI_COMBINE_DTYPE_DECODE} ${DECODE_MORI_MOE_ENV} SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK_DECODE:-${MORI_MAX_DISPATCH_TOKENS_DECODE}} MORI_IO_SQ_BACKOFF_TIMEOUT_US=${MORI_IO_SQ_BACKOFF_TIMEOUT_US} MORI_IO_QP_MAX_SEND_WR=${MORI_IO_QP_MAX_SEND_WR} ${LAUNCH_PREFIX:-} python3 -m sglang.launch_server \
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
        # setsid isolates the server tree in its own process group so teardown
        # can group-kill it (python + TP-scheduler children); otherwise the
        # children hold the process-sub tee's pipe and the container never exits.
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
        # Group-kill the whole server tree (setsid at launch) so TP-scheduler
        # children die and the process-sub tee gets EOF -> container can exit.
        kill -TERM -"${decode_pgid:-$decode_pid}" 2>/dev/null || true
    fi

fi

echo "Script completed successfully"
exit 0
