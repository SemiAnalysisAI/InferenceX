#!/usr/bin/env bash
set -euo pipefail

# Sweep experiment for multi-turn benchmark
# Sweeps: TP (1,2,4,8) x BS (16,32,64,128,256,512) x CPU offload (on/off)
#
# Usage:
#   ./sweep_experiment.sh                    # Start fresh
#   ./sweep_experiment.sh sweep_results_XXX  # Resume from existing directory

MODEL="nvidia/Llama-3.3-70B-Instruct-FP4"
INPUT_FILE="sample_5k.json"
PORT=8888
TOTAL_CPU_DRAM_GB=300
NUM_REQUESTS=5000
REQUEST_TIMEOUT=3600
MAX_RETRIES=3

# Output directory - use provided arg or create new
if [ $# -ge 1 ] && [ -d "$1" ]; then
    RESULTS_DIR="$1"
    echo "Resuming from existing directory: $RESULTS_DIR"
else
    RESULTS_DIR="sweep_results_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RESULTS_DIR"
    echo "Created new results directory: $RESULTS_DIR"
fi

# Log file
LOG_FILE="$RESULTS_DIR/sweep.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo "Starting sweep experiment at $(date)"
echo "Results directory: $RESULTS_DIR"
echo "========================================"

# Arrays for sweep
TP_VALUES=(1 2 4 8)
BS_VALUES=(16 32 64 128 256 512)
OFFLOAD_VALUES=(on off)

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
    pkill -f "vllm serve" || true
    sleep 60
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

# Function to run a single experiment
run_experiment() {
    local tp=$1
    local bs=$2
    local offload=$3

    # Calculate CPU offload size per GPU
    local offload_size=$((TOTAL_CPU_DRAM_GB / tp))

    # Experiment name
    local exp_name="tp${tp}_bs${bs}_offload${offload}"
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

    echo ""
    echo "========================================"
    echo "Running experiment: $exp_name"
    echo "  TP=$tp, BS=$bs, Offload=$offload"
    if [ "$offload" = "on" ]; then
        echo "  CPU offload size per GPU: ${offload_size}GB"
    fi
    echo "  Started at $(date)"
    echo "========================================"

    # Create config file
    cat > "$exp_dir/config.yaml" << EOF
kv-cache-dtype: fp8
compilation-config: '{"pass_config":{"fuse_allreduce_rms":true,"eliminate_noops":true},"custom_ops":["+quant_fp8","+rms_norm"],"cudagraph_mode":"FULL_DECODE_ONLY","splitting_ops":[]}'
async-scheduling: true
max-cudagraph-capture-size: 2048
max-num-batched-tokens: 8192
EOF

    # Build vllm command
    local vllm_cmd="vllm serve $MODEL --host 0.0.0.0 --port $PORT"
    vllm_cmd+=" --config $exp_dir/config.yaml"
    vllm_cmd+=" --max-num-seqs $bs"
    vllm_cmd+=" --gpu-memory-utilization 0.9"
    vllm_cmd+=" --tensor-parallel-size $tp"
    vllm_cmd+=" --attention-config.use_trtllm_attention=0"

    if [ "$offload" = "on" ]; then
        vllm_cmd+=" --kv_offloading_backend native"
        vllm_cmd+=" --kv_offloading_size $offload_size"
        vllm_cmd+=" --disable-hybrid-kv-cache-manager"
    fi

    # Save the command for reference
    echo "$vllm_cmd" > "$exp_dir/vllm_command.txt"

    # Start server in background
    echo "Starting vllm server..."
    export TORCH_CUDA_ARCH_LIST="10.0"
    export PYTHONNOUSERSITE=1
    export VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB='{"2":32,"4":32,"8":8}'

    $vllm_cmd > "$exp_dir/server.log" 2>&1 &
    local server_pid=$!
    echo "Server PID: $server_pid"

    # Wait for server
    if ! wait_for_server; then
        echo "ERROR: Server failed to start for $exp_name"
        stop_server
        echo "FAILED" > "$exp_dir/status.txt"
        return 0  # Don't exit script, continue to next experiment
    fi

    # Run benchmark
    echo "Running benchmark..."
    local benchmark_cmd="python3 benchmark_serving_multi_turn.py"
    benchmark_cmd+=" -i $INPUT_FILE"
    benchmark_cmd+=" -m $MODEL"
    benchmark_cmd+=" -u http://localhost:$PORT"
    benchmark_cmd+=" -p $bs"
    benchmark_cmd+=" -n $NUM_REQUESTS"
    benchmark_cmd+=" --max-retries $MAX_RETRIES"
    benchmark_cmd+=" --request-timeout $REQUEST_TIMEOUT"
    benchmark_cmd+=" --metrics-output $exp_dir/metrics"
    benchmark_cmd+=" --metrics-csv"
    benchmark_cmd+=" --responses-file $exp_dir/responses.json"

    echo "$benchmark_cmd" > "$exp_dir/benchmark_command.txt"

    if $benchmark_cmd > "$exp_dir/benchmark.log" 2>&1; then
        echo "SUCCESS" > "$exp_dir/status.txt"
        echo "Benchmark completed successfully"
    else
        echo "FAILED" > "$exp_dir/status.txt"
        echo "ERROR: Benchmark failed for $exp_name"
    fi

    # Stop server
    stop_server

    echo "Experiment $exp_name finished at $(date)"
}

# Main sweep loop
total_experiments=$((${#TP_VALUES[@]} * ${#BS_VALUES[@]} * ${#OFFLOAD_VALUES[@]}))
current=0

for tp in "${TP_VALUES[@]}"; do
    for bs in "${BS_VALUES[@]}"; do
        for offload in "${OFFLOAD_VALUES[@]}"; do
            current=$((current + 1))
            echo ""
            echo "========================================"
            echo "Experiment $current / $total_experiments"
            echo "========================================"

            run_experiment "$tp" "$bs" "$offload"

            # Brief pause between experiments
            sleep 5
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
    echo "experiment,status,tp,bs,offload"
    for exp_dir in "$RESULTS_DIR"/tp*; do
        if [ -d "$exp_dir" ]; then
            exp_name=$(basename "$exp_dir")
            status=$(cat "$exp_dir/status.txt" 2>/dev/null || echo "UNKNOWN")
            # Parse tp, bs, offload from exp_name (tp1_bs16_offloadon)
            tp=$(echo "$exp_name" | sed 's/tp\([0-9]*\)_.*/\1/')
            bs=$(echo "$exp_name" | sed 's/.*bs\([0-9]*\)_.*/\1/')
            offload=$(echo "$exp_name" | sed 's/.*offload\(.*\)/\1/')
            echo "$exp_name,$status,$tp,$bs,$offload"
        fi
    done
} > "$RESULTS_DIR/summary.csv"

echo "Summary saved to $RESULTS_DIR/summary.csv"
