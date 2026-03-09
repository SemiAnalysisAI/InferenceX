#!/usr/bin/env bash
set -euo pipefail

# Change to the vllm_benchmark directory (required for bench module imports)
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

# Realistic sweep experiment for multi-turn benchmark
# Uses the 20k realistic dataset.
#
# Key differences from sweep_experiment.sh:
#   - Uses realistic turn distribution (40% single-turn, long tail)
#   - -p = number of concurrent users (not constant batch size)
#   - Sweeps: TP x num_users x mode (on/off/noprefix)
#
# Usage:
#   ./sweep_realistic.sh                        # Start fresh
#   ./sweep_realistic.sh sweep_realistic_XXX    # Resume from existing directory

MODEL="nvidia/Llama-3.3-70B-Instruct-FP4"
INPUT_FILE="sample_20k_realistic.json"
PORT=8888
TOTAL_CPU_DRAM_GB=300
REQUEST_TIMEOUT=3600
MAX_RETRIES=3
DURATION=180  # 3 minutes per experiment


# Output directory - use provided arg or create new
if [ $# -ge 1 ] && [ -d "$1" ]; then
    RESULTS_DIR="$1"
    echo "Resuming from existing directory: $RESULTS_DIR"
else
    RESULTS_DIR="sweep_realistic_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RESULTS_DIR"
    echo "Created new results directory: $RESULTS_DIR"
fi

# Log file
LOG_FILE="$RESULTS_DIR/sweep.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo "Starting realistic sweep at $(date)"
echo "Results directory: $RESULTS_DIR"
echo "========================================"

# Arrays for sweep
TP_VALUES=(1 2 4 8)
# Number of concurrent users
USER_VALUES=(2048 1024 512 256 128 64 32 16 8)
# on=prefix caching + CPU offload, off=prefix caching only, noprefix=no prefix caching
OFFLOAD_VALUES=(on off noprefix)

# Function to wait for server to be ready
wait_for_server() {
    local max_wait=600
    local waited=0
    echo "Waiting for server to be ready..."
    while ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if [ $waited -ge $max_wait ]; then
            echo "ERROR: Server did not start within ${max_wait}s"
            return 1
        fi
        echo "  ...waited ${waited}s"
    done
    echo "Server is ready!"
    sleep 5  # Extra buffer for full initialization
}

# Function to stop server
stop_server() {
    echo "Stopping vllm server..."

    # Graceful kill first
    pkill -f "vllm serve" || true
    sleep 3

    # Kill any remaining vllm/multiprocessing workers
    pkill -9 -f "vllm" || true
    pkill -9 -f "multiproc_executor" || true
    pkill -9 -f "from vllm" || true
    sleep 3

    # Kill any python processes using GPU
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read pid; do
        if [ -n "$pid" ]; then
            echo "Killing GPU process: $pid"
            kill -9 "$pid" 2>/dev/null || sudo kill -9 "$pid" 2>/dev/null || true
        fi
    done
    sleep 3

    # Super aggressive: sudo kill anything vllm/python related
    sudo pkill -9 -f "vllm" 2>/dev/null || true
    sudo pkill -9 -f "multiproc_executor" 2>/dev/null || true
    sleep 3

    # Clean up shared memory (NCCL/torch uses this)
    rm -rf /dev/shm/*nccl* /dev/shm/*torch* /dev/shm/*python* 2>/dev/null || true
    rm -rf /tmp/pymp-* /tmp/torch_* 2>/dev/null || true

    # Final check - kill any remaining GPU processes
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read pid; do
        if [ -n "$pid" ]; then
            echo "Force killing remaining GPU process: $pid"
            sudo kill -9 "$pid" 2>/dev/null || true
        fi
    done

    # Wait for cleanup
    sleep 10

    echo "Server stopped"
}

# Function to check if experiment already completed successfully
is_completed() {
    local exp_dir=$1
    if [ -f "$exp_dir/status.txt" ]; then
        local status=$(cat "$exp_dir/status.txt")
        if [ "$status" = "SUCCESS" ]; then
            return 0
        fi
    fi
    return 1
}

# Function to run a single experiment (with retries)
run_experiment() {
    local tp=$1
    local users=$2
    local offload=$3
    local max_retries=3

    # Calculate CPU offload size per GPU
    local offload_size=$((TOTAL_CPU_DRAM_GB / tp))

    # Experiment name
    local exp_name="tp${tp}_users${users}_offload${offload}"
    local exp_dir="$RESULTS_DIR/$exp_name"

    # Check if already completed
    if is_completed "$exp_dir"; then
        echo ""
        echo "========================================"
        echo "SKIPPING $exp_name (already completed successfully)"
        echo "========================================"
        return 0
    fi

    mkdir -p "$exp_dir"

    # Retry loop
    for attempt in $(seq 1 $max_retries); do
        echo ""
        echo "========================================"
        echo "Running experiment: $exp_name (attempt $attempt/$max_retries)"
        echo "  TP=$tp, Users=$users, Mode=$offload"
if [ "$offload" = "on" ]; then
            echo "  Prefix caching: ON, CPU offload: ON (${offload_size}GB per GPU)"
        elif [ "$offload" = "off" ]; then
            echo "  Prefix caching: ON, CPU offload: OFF"
        else
            echo "  Prefix caching: OFF, CPU offload: OFF"
        fi
        echo "  Started at $(date)"
        echo "========================================"

        # Create config file
        cat > "$exp_dir/config.yaml" << EOF
kv-cache-dtype: fp8
compilation-config: '{"pass_config":{"fuse_allreduce_rms":true,"eliminate_noops":true},"custom_ops":["+quant_fp8","+rms_norm"],"cudagraph_mode":"FULL_DECODE_ONLY","splitting_ops":[]}'
async-scheduling: true
max-num-batched-tokens: 8192
EOF

        # Build vllm command
        local max_seqs=$users

        local vllm_cmd="vllm serve $MODEL --host 0.0.0.0 --port $PORT"
        vllm_cmd+=" --config $exp_dir/config.yaml"
        vllm_cmd+=" --max-num-seqs $max_seqs"
        vllm_cmd+=" --gpu-memory-utilization 0.9"
        vllm_cmd+=" --tensor-parallel-size $tp"
        vllm_cmd+=" --attention-config.use_trtllm_attention=0"

        if [ "$offload" = "on" ]; then
            # Prefix caching ON + KV offload to CPU ON
            vllm_cmd+=" --kv_offloading_backend native"
            vllm_cmd+=" --kv_offloading_size $offload_size"
            vllm_cmd+=" --disable-hybrid-kv-cache-manager"
        elif [ "$offload" = "noprefix" ]; then
            # Prefix caching OFF entirely
            vllm_cmd+=" --no-enable-prefix-caching"
        fi
        # offload=off: prefix caching ON, no CPU offload (default behavior)

        # Save the command for reference
        echo "$vllm_cmd" > "$exp_dir/vllm_command.txt"

        # Start server in background
        echo "Starting vllm server..."
        export TORCH_CUDA_ARCH_LIST="10.0"
        export PYTHONNOUSERSITE=1
        export VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB='{"2":32,"4":32,"8":8}'

        $vllm_cmd > "$exp_dir/server_attempt${attempt}.log" 2>&1 &
        local server_pid=$!
        echo "Server PID: $server_pid"

        # Wait for server
        if ! wait_for_server; then
            echo "ERROR: Server failed to start for $exp_name (attempt $attempt/$max_retries)"
            stop_server
            if [ $attempt -eq $max_retries ]; then
                echo "FAILED" > "$exp_dir/status.txt"
                echo "All $max_retries attempts failed for $exp_name"
                return 0
            fi
            echo "Retrying in 30 seconds..."
            sleep 30
            continue
        fi

        # Run benchmark
        echo "Running benchmark (users=$users, duration=${DURATION}s)..."
        local benchmark_cmd="python3 benchmark/benchmark_serving_multi_turn.py"
        benchmark_cmd+=" -i $INPUT_FILE"
        benchmark_cmd+=" -m $MODEL"
        benchmark_cmd+=" -u http://localhost:$PORT"
        benchmark_cmd+=" -p $users"
        benchmark_cmd+=" --duration $DURATION"
        benchmark_cmd+=" --max-retries $MAX_RETRIES"
        benchmark_cmd+=" --request-timeout $REQUEST_TIMEOUT"
        benchmark_cmd+=" --metrics-output $exp_dir/metrics"
        benchmark_cmd+=" --metrics-csv"
        benchmark_cmd+=" --responses-file $exp_dir/responses.json"

        echo "$benchmark_cmd" > "$exp_dir/benchmark_command.txt"

        if $benchmark_cmd > "$exp_dir/benchmark_attempt${attempt}.log" 2>&1; then
            echo "SUCCESS" > "$exp_dir/status.txt"
            echo "Benchmark completed successfully"
            stop_server
            echo "Experiment $exp_name finished at $(date)"
            return 0
        else
            echo "ERROR: Benchmark failed for $exp_name (attempt $attempt/$max_retries)"
            stop_server
            if [ $attempt -eq $max_retries ]; then
                echo "FAILED" > "$exp_dir/status.txt"
                echo "All $max_retries attempts failed for $exp_name"
                return 0
            fi
            echo "Retrying in 30 seconds..."
            sleep 30
        fi
    done
}

# Main sweep loop
total_experiments=$(( ${#TP_VALUES[@]} * ${#USER_VALUES[@]} * ${#OFFLOAD_VALUES[@]} ))
current=0

for tp in "${TP_VALUES[@]}"; do
    for users in "${USER_VALUES[@]}"; do
        for offload in "${OFFLOAD_VALUES[@]}"; do
            current=$((current + 1))
            echo ""
            echo "========================================"
            echo "Experiment $current / $total_experiments"
            echo "========================================"

            run_experiment "$tp" "$users" "$offload"
        done
    done
done

echo ""
echo "========================================"
echo "Sweep completed at $(date)"
echo "Results saved in: $RESULTS_DIR"
echo "========================================"

# Generate summary
echo ""
echo "Generating summary..."
{
    echo "experiment,status,tp,users,offload"
    for exp_dir in "$RESULTS_DIR"/tp*; do
        if [ -d "$exp_dir" ]; then
            exp_name=$(basename "$exp_dir")
            status=$(cat "$exp_dir/status.txt" 2>/dev/null || echo "UNKNOWN")
            # Parse tp, users, offload from exp_name (tp1_users32_offloadon)
            tp=$(echo "$exp_name" | sed 's/tp\([0-9]*\)_.*/\1/')
            users=$(echo "$exp_name" | sed 's/.*users\([0-9]*\)_.*/\1/')
            offload=$(echo "$exp_name" | sed 's/.*offload\(.*\)/\1/')
            echo "$exp_name,$status,$tp,$users,$offload"
        fi
    done
} > "$RESULTS_DIR/summary.csv"

echo "Summary saved to $RESULTS_DIR/summary.csv"
