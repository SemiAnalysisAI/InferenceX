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

# Dry Run for debugging purpose
DRY_RUN="${DRY_RUN:-0}"

# GPU count (expandable for different hardware)
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"


# =============================================================================
# Dependencies and Environment Setup
# =============================================================================
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

with open(config_path) as f:
    models = yaml.safe_load(f)

if model_name not in models:
    print(f'echo \"ERROR: Model {model_name} not in models.yaml\"; exit 1')
    sys.exit(0)

m = models[model_name]

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

prefill = m.get('prefill', {})
decode = m.get('decode', {})

print(f'PREFILL_MEM_FRACTION_STATIC=\"{prefill.get(\"mem_fraction_static\", 0.8)}\"')
print(f'PREFILL_DISABLE_RADIX_CACHE=\"{prefill.get(\"disable_radix_cache\", True)}\"')

dp = prefill.get('dp', {})
no_dp = prefill.get('no_dp', {})
print(f'PREFILL_MAX_RUNNING_REQUESTS_DP=\"{dp.get(\"max_running_requests\", 24)}\"')
print(f'PREFILL_CHUNKED_PREFILL_SIZE_DP=\"{eval_formula(dp.get(\"chunked_prefill_size\", 262144))}\"')
print(f'PREFILL_CUDA_GRAPH_BS_DP=\"{dp.get(\"cuda_graph_bs\", \"1 2 3\")}\"')
print(f'PREFILL_MAX_RUNNING_REQUESTS_NO_DP=\"{no_dp.get(\"max_running_requests\", 128)}\"')
print(f'PREFILL_CHUNKED_PREFILL_SIZE_NO_DP=\"{eval_formula(no_dp.get(\"chunked_prefill_size\", 262144))}\"')
s, e = parse_range(no_dp.get('cuda_graph_bs_range', '1-128'), 1, 128)
print(f'PREFILL_CUDA_GRAPH_BS_NO_DP_START=\"{s}\"')
print(f'PREFILL_CUDA_GRAPH_BS_NO_DP_END=\"{e}\"')

print(f'DECODE_MEM_FRACTION_STATIC=\"{decode.get(\"mem_fraction_static\", 0.85)}\"')
print(f'DECODE_PREFILL_ROUND_ROBIN_BALANCE=\"{decode.get(\"prefill_round_robin_balance\", True)}\"')

dp = decode.get('dp', {})
ep_only = decode.get('ep_only', {})
no_dp = decode.get('no_dp', {})

# Decode DP config
print(f'DECODE_MAX_RUNNING_REQUESTS_DP=\"{dp.get(\"max_running_requests\", 4096)}\"')
print(f'DECODE_CHUNKED_PREFILL_SIZE_DP=\"{eval_formula(dp.get(\"chunked_prefill_size\", 262144))}\"')
s, e = parse_range(dp.get('cuda_graph_bs_range', '1-160'), 1, 160)
print(f'DECODE_CUDA_GRAPH_BS_DP_START=\"{s}\"')
print(f'DECODE_CUDA_GRAPH_BS_DP_END=\"{e}\"')

# Decode EP-only config (EP enabled but DP disabled)
print(f'DECODE_MAX_RUNNING_REQUESTS_EP_ONLY=\"{ep_only.get(\"max_running_requests\", 256)}\"')
print(f'DECODE_CHUNKED_PREFILL_SIZE_EP_ONLY=\"{eval_formula(ep_only.get(\"chunked_prefill_size\", 262144))}\"')
s, e = parse_range(ep_only.get('cuda_graph_bs_range', '1-256'), 1, 256)
print(f'DECODE_CUDA_GRAPH_BS_EP_ONLY_START=\"{s}\"')
print(f'DECODE_CUDA_GRAPH_BS_EP_ONLY_END=\"{e}\"')

# Decode no-DP config
print(f'DECODE_MAX_RUNNING_REQUESTS_NO_DP=\"{no_dp.get(\"max_running_requests\", 128)}\"')
print(f'DECODE_CHUNKED_PREFILL_SIZE_NO_DP=\"{eval_formula(no_dp.get(\"chunked_prefill_size\", 262144))}\"')
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
else
    prefill_cuda_graph_bs=($(seq $PREFILL_CUDA_GRAPH_BS_NO_DP_START $PREFILL_CUDA_GRAPH_BS_NO_DP_END))
    prefill_max_running_requests=$PREFILL_MAX_RUNNING_REQUESTS_NO_DP
    prefill_chunked_prefill_size=$PREFILL_CHUNKED_PREFILL_SIZE_NO_DP
fi

# Compute DP-dependent decode parameters (3-way: DP > EP-only > no_dp)
if [[ "$DECODE_ENABLE_DP" == "true" ]]; then
    decode_cuda_graph_bs=($(seq $DECODE_CUDA_GRAPH_BS_DP_START $DECODE_CUDA_GRAPH_BS_DP_END))
    decode_max_running_requests=$((DECODE_CUDA_GRAPH_BS_DP_END * DECODE_TP_SIZE))
    decode_chunked_prefill_size=$DECODE_CHUNKED_PREFILL_SIZE_DP
elif [[ "$DECODE_ENABLE_EP" == "true" ]]; then
    decode_cuda_graph_bs=($(seq $DECODE_CUDA_GRAPH_BS_EP_ONLY_START $DECODE_CUDA_GRAPH_BS_EP_ONLY_END))
    decode_max_running_requests=$DECODE_MAX_RUNNING_REQUESTS_EP_ONLY
    decode_chunked_prefill_size=$DECODE_CHUNKED_PREFILL_SIZE_EP_ONLY
else
    decode_cuda_graph_bs=($(seq $DECODE_CUDA_GRAPH_BS_NO_DP_START $DECODE_CUDA_GRAPH_BS_NO_DP_END))
    decode_max_running_requests=$DECODE_MAX_RUNNING_REQUESTS_NO_DP
    decode_chunked_prefill_size=$DECODE_CHUNKED_PREFILL_SIZE_NO_DP
fi

# Use Decode configuration to configure different TP/DP size between P and D
PREFILL_DECODE_DIFFERENT_TP=""
if [[ "$PREFILL_ENABLE_DP" != "$DECODE_ENABLE_DP" ]]; then
    if [[ "$DECODE_ENABLE_DP" == "true" ]]; then
        PREFILL_DECODE_DIFFERENT_TP="--disaggregation-decode-tp ${DECODE_TP_SIZE} --disaggregation-decode-dp ${DECODE_TP_SIZE}"
    else
        PREFILL_DECODE_DIFFERENT_TP="--disaggregation-decode-tp ${DECODE_TP_SIZE} --disaggregation-decode-dp 1"
    fi
fi

# Build the composed config strings (equivalent to the old MODEL_PREFILL_CONFIGS / MODEL_DECODE_CONFIGS)
PREFILL_MODE_FLAGS="--mem-fraction-static ${PREFILL_MEM_FRACTION_STATIC} --max-running-requests ${prefill_max_running_requests} --chunked-prefill-size ${prefill_chunked_prefill_size} --cuda-graph-bs ${prefill_cuda_graph_bs[*]} ${PREFILL_DECODE_DIFFERENT_TP}"
if [[ "$PREFILL_DISABLE_RADIX_CACHE" == "True" ]] || [[ "$PREFILL_DISABLE_RADIX_CACHE" == "true" ]]; then
    PREFILL_MODE_FLAGS="$PREFILL_MODE_FLAGS --disable-radix-cache"
fi

DECODE_MODE_FLAGS="--mem-fraction-static ${DECODE_MEM_FRACTION_STATIC} --max-running-requests ${decode_max_running_requests} --chunked-prefill-size ${decode_chunked_prefill_size} --cuda-graph-bs ${decode_cuda_graph_bs[*]}"
if [[ "$DECODE_PREFILL_ROUND_ROBIN_BALANCE" == "True" ]] || [[ "$DECODE_PREFILL_ROUND_ROBIN_BALANCE" == "true" ]]; then
    DECODE_MODE_FLAGS="$DECODE_MODE_FLAGS --prefill-round-robin-balance"
fi

if [[ "$DECODE_MTP_SIZE" -gt 0 ]]; then
    MORI_MAX_DISPATCH_TOKENS_DECODE=$((MORI_MAX_DISPATCH_TOKENS_DECODE * (DECODE_MTP_SIZE + 1)))
fi

# =============================================================================
# Cluster Topology Configuration
# =============================================================================
IFS=',' read -ra IP_ARRAY <<< "$IPADDRS"

PREFILL_NODES_PER_WORKER=$((PREFILL_TP_SIZE / GPUS_PER_NODE))
DECODE_NODES_PER_WORKER=$((DECODE_TP_SIZE / GPUS_PER_NODE))
NODE_OFFSET=$((PREFILL_NODES_PER_WORKER * xP))

# Build prefill arguments dynamically based on xP
PREFILL_HEADNODE_URLS=()
PREFILL_ARGS=""
for i in $(seq 0 $((xP - 1))); do
    prefill_idx=$((i * PREFILL_NODES_PER_WORKER))
    PREFILL_HEADNODE_URLS[$i]="${IP_ARRAY[$prefill_idx]}:${HEADNODE_PORT}"
    PREFILL_ARGS="$PREFILL_ARGS --prefill http://${IP_ARRAY[$prefill_idx]}:8000"
done

# Build decode arguments dynamically based on yD
DECODE_HEADNODE_URLS=()
DECODE_ARGS=""
for i in $(seq 0 $((yD - 1))); do
    decode_idx=$((i * DECODE_NODES_PER_WORKER + NODE_OFFSET))
    DECODE_HEADNODE_URLS[$i]="${IP_ARRAY[$decode_idx]}:${HEADNODE_PORT}"
    DECODE_ARGS="$DECODE_ARGS --decode http://${IP_ARRAY[$decode_idx]}:8000"
done

echo "Prefill worker headnode list: ${PREFILL_HEADNODE_URLS[@]}"
echo "Decode  worker headnode list: ${DECODE_HEADNODE_URLS[@]}"

# =============================================================================
# Configuration Builder Functions
# =============================================================================

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
    local base_config="$MODEL_BASE_FLAGS"
    local mtp_config=""
    local dp_config=""
    local specific_config=""

    # MTP config (only if MTP is enabled and mode is decode)
    if [ "$decode_mtp_size" -gt 0 ]; then
        mtp_config="${MODEL_MTP_FLAGS} --speculative-num-steps ${decode_mtp_size} --speculative-num-draft-tokens $((decode_mtp_size + 1))"
    fi

    # DP config (only if DP is enabled)
    if [[ "$enable_dp" == "true" ]]; then
        dp_config="$MODEL_DP_FLAGS"
    fi

    # Mode-specific config
    if [[ "$mode" == "prefill" ]]; then
        specific_config="$PREFILL_MODE_FLAGS"
    elif [[ "$mode" == "decode" ]]; then
        specific_config="$DECODE_MODE_FLAGS"
    fi

    # Combine: parallel args + base config + mtp config (decode only) + dp config + specific config
    local full_config="$parallel_args"
    if [[ -n "$base_config" ]]; then
        full_config="$full_config $base_config"
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

if [[ -n "$MODEL_NAME" ]]; then
    echo "Using model-specific configuration for: $MODEL_NAME"
fi

# =============================================================================
# Container Synchronization
# =============================================================================

echo "Waiting at the container creation barrier on $host_name"
python3 $SGLANG_WS_PATH/sync.py barrier \
    --local-ip ${host_ip} \
    --local-port 5000 \
    --enable-port \
    --node-ips ${IPADDRS} \
    --node-ports 5000 \
    --wait-for-all-ports \
    --timeout 300


# =============================================================================
# Node Role Assignment and Server Launch
# =============================================================================

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
    echo "Prefill env: SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK: ${MORI_MAX_DISPATCH_TOKENS_PREFILL}"
    echo "Decode env: SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_MAX_DISPATCH_TOKENS_DECODE}"
    echo "================================================"

    # start the head prefill server
    PREFILL_CMD="SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_MAX_DISPATCH_TOKENS_PREFILL} python3 -m sglang.launch_server \
        --model-path $MODEL_DIR/$MODEL_NAME \
        --disaggregation-mode prefill \
        --disaggregation-ib-device ${IBDEVICES} \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code \
        ${PREFILL_SERVER_CONFIG} \
        --log-level-http warning"

    if [ "$PREFILL_NODES_PER_WORKER" -gt 1 ]; then
        PREFILL_CMD="$PREFILL_CMD --dist-init-addr ${PREFILL_HEADNODE_URLS[0]} --nnodes ${PREFILL_NODES_PER_WORKER} --node-rank 0"
    fi


    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $PREFILL_CMD"
    else
        set -x
        eval "$PREFILL_CMD" \
            2>&1 | tee /run_logs/slurm_job-${SLURM_JOB_ID}/prefill_${host_name}.log &
        set +x
        prefill0_pid=$!
    fi


    echo "Waiting for all prefill and decode servers to be up . . ."


    BARRIER_CMD="python3 $SGLANG_WS_PATH/sync.py barrier \
        --node-ips ${IPADDRS} \
        --node-ports 8000 \
        --wait-for-all-ports \
        --timeout 1800"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BARRIER_CMD"
    else
        eval "$BARRIER_CMD"
    fi
    echo "Congratulations!!! All prefill and decode servers are up . . ."

    ROUTER_CMD="python -m sglang_router.launch_router \
        --pd-disaggregation \
        --port 30000 \
        --policy random \
        --prefill-policy random \
        --decode-policy random \
        ${PREFILL_ARGS} \
        ${DECODE_ARGS}"


    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $ROUTER_CMD"
    else
        ROUTER_LOG_FILE="/tmp/slurm_job-${SLURM_JOB_ID}_proxy_${host_name}.log"
        set -x
        if [[ "${SGLANG_ROUTER_STDOUT_LOGS:-0}" == "1" ]]; then
            eval "$ROUTER_CMD" 2>&1 | tee "$ROUTER_LOG_FILE" &
        else
            eval "$ROUTER_CMD" >"$ROUTER_LOG_FILE" 2>&1 &
        fi
        set +x
        proxy_pid=$!

        # Wait for router to be ready via health endpoint
        HEALTH_BARRIER_CMD="python3 $SGLANG_WS_PATH/sync.py barrier \
            --node-ips ${NODE0_ADDR} \
            --node-ports 30000 \
            --wait-for-all-health \
            --health-endpoint /readiness \
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
    cd $SGLANG_WS_PATH

    # Export IS_MTP based on whether MTP is enabled
    if [ "$DECODE_MTP_SIZE" -gt 0 ]; then
        export IS_MTP=true
    else
        export IS_MTP=false
    fi

    # n_prefill n_decode prefill_gpus decode_gpus model_dir model_name log_path isl osl concurrency_list req_rate random_range_ratio num_prompts_multiplier
    BENCH_CMD="bash $SGLANG_WS_PATH/bench.sh ${xP} ${yD} $((PREFILL_TP_SIZE*xP)) $((DECODE_TP_SIZE*yD)) \
        $MODEL_DIR $MODEL_NAME /run_logs/slurm_job-${SLURM_JOB_ID} ${BENCH_INPUT_LEN} \
        ${BENCH_OUTPUT_LEN} "${BENCH_MAX_CONCURRENCY}" ${BENCH_REQUEST_RATE} \
        ${BENCH_RANDOM_RANGE_RATIO} ${BENCH_NUM_PROMPTS_MULTIPLIER}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BENCH_CMD"
    else
        set -x
        eval "$BENCH_CMD"
        set +x
    fi

    # Run evaluation if requested (before killing router)
    if [[ "${RUN_EVAL:-false}" == "true" ]]; then
        echo "Running lm-eval evaluation on Node 0..."

        # Must run from repo root so utils/evals/${task}.yaml resolves
        pushd /workspace

        # Source eval functions from benchmark_lib.sh
        source /workspace/benchmarks/benchmark_lib.sh

        # Determine eval concurrency (cap at 64 for eval stability)
        IFS='x' read -r -a _conc_arr <<< "${BENCH_MAX_CONCURRENCY}"
        EVAL_CONC="${_conc_arr[0]:-64}"

        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "DRY RUN: run_eval --framework lm-eval --port 30000 --concurrent-requests $EVAL_CONC"
        else
            # Run lm-eval against the router on port 30000
            run_eval --framework lm-eval --port 30000 --concurrent-requests "$EVAL_CONC"

            # Set metadata env vars for append_lm_eval_summary
            export TP="${PREFILL_TP_SIZE}"
            export CONC="${EVAL_CONC}"
            export EP_SIZE=1
            [[ "${PREFILL_ENABLE_EP}" == "true" ]] && EP_SIZE="${PREFILL_TP_SIZE}"
            export DP_ATTENTION="${PREFILL_ENABLE_DP}"
            export ISL="${BENCH_INPUT_LEN}"
            export OSL="${BENCH_OUTPUT_LEN}"
            # FRAMEWORK, PRECISION, MODEL_PREFIX, RUNNER_TYPE, RESULT_FILENAME
            # are already set via Docker -e flags from job.slurm

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

        popd
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
        kill $proxy_pid
        kill $prefill0_pid
    fi

elif [ "$NODE_RANK" -gt 0 ] && [ "$NODE_RANK" -lt "$NODE_OFFSET" ]; then
    echo "${host_name}:${host_ip} is Prefill Node (Model: ${MODEL_NAME:-'default'})"
    echo "Using prefill config: $PREFILL_SERVER_CONFIG"
    echo "Prefill parallelism: TP=${PREFILL_TP_SIZE}, EP enabled: ${PREFILL_ENABLE_EP}, DP enabled: ${PREFILL_ENABLE_DP}"

    PREFILL_CMD="SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_MAX_DISPATCH_TOKENS_PREFILL} python3 -m sglang.launch_server \
        --model-path $MODEL_DIR/${MODEL_NAME} \
        --disaggregation-mode prefill \
        --disaggregation-ib-device ${IBDEVICES} \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code \
        ${PREFILL_SERVER_CONFIG} \
        --log-level-http warning"

    if [ "$PREFILL_NODES_PER_WORKER" -gt 1 ]; then
        rank=$((NODE_RANK % PREFILL_NODES_PER_WORKER))
        prefill_idx=$((NODE_RANK / PREFILL_NODES_PER_WORKER))
        PREFILL_CMD="$PREFILL_CMD --dist-init-addr ${PREFILL_HEADNODE_URLS[$prefill_idx]} --nnodes ${PREFILL_NODES_PER_WORKER} --node-rank $rank"
    fi

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
    BARRIER_CMD="python3 $SGLANG_WS_PATH/sync.py barrier \
        --node-ips ${NODE0_ADDR} \
        --node-ports 30000 \
        --wait-for-all-ports \
        --timeout 1800"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BARRIER_CMD"
    else
        eval "$BARRIER_CMD"
    fi

    echo "Waiting until proxy server closes..."
    WAIT_CMD="python3 $SGLANG_WS_PATH/sync.py wait \
        --remote-ip ${NODE0_ADDR} \
        --remote-port 30000"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $WAIT_CMD"
    else
        eval "$WAIT_CMD"
    fi

    echo "Killing the rank $NODE_RANK prefill server"

    if [[ "$DRY_RUN" -eq 0 ]]; then
        kill $prefill_pid
    fi

else
    RANK=$((NODE_RANK - xP * PREFILL_NODES_PER_WORKER))
    echo "${host_name}:${host_ip} is Decode Node (Model: ${MODEL_NAME:-'default'})"
    echo "Using decode config: $DECODE_SERVER_CONFIG"
    echo "Decode node rank: $RANK"
    echo "Decode parallelism: TP=${DECODE_TP_SIZE}, EP enabled: ${DECODE_ENABLE_EP}, DP enabled: ${DECODE_ENABLE_DP}"

    DECODE_CMD="SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MORI_MAX_DISPATCH_TOKENS_DECODE} python3 -m sglang.launch_server \
        --model-path ${MODEL_DIR}/${MODEL_NAME} \
        --disaggregation-mode decode \
        --disaggregation-ib-device ${IBDEVICES} \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code \
        ${DECODE_SERVER_CONFIG} \
        --log-level-http warning"

    if [ "$DECODE_NODES_PER_WORKER" -gt 1 ]; then
        rank=$((RANK % DECODE_NODES_PER_WORKER))
        decode_idx=$((RANK / DECODE_NODES_PER_WORKER))
        DECODE_CMD="$DECODE_CMD --dist-init-addr ${DECODE_HEADNODE_URLS[$decode_idx]} --nnodes ${DECODE_NODES_PER_WORKER} --node-rank $rank"
    fi

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
    BARRIER_CMD="python3 $SGLANG_WS_PATH/sync.py barrier \
        --node-ips ${NODE0_ADDR} \
        --node-ports 30000 \
        --wait-for-all-ports \
        --timeout 1800"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $BARRIER_CMD"
    else
        eval "$BARRIER_CMD"
    fi


    echo "Waiting until proxy server closes..."
    WAIT_CMD="python3 $SGLANG_WS_PATH/sync.py wait \
        --remote-ip ${NODE0_ADDR} \
        --remote-port 30000"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY RUN: $WAIT_CMD"
    else
        eval "$WAIT_CMD"
    fi

    echo "Killing the rank $RANK decode server"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        kill $decode_pid
    fi

fi

echo "Script completed successfully"
exit 0
