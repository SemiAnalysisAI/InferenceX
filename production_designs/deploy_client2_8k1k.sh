#!/bin/bash
# deploy_client2_8k1k.sh
# Production deployment for Client 2: 8k input / 1k output pattern
# Optimized for: Long context, short responses (document Q&A, summarization)

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model and framework settings
export IMAGE="${IMAGE:-nvidia/deepseek-r1-fp4:latest}"
export MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-0528}"
export FRAMEWORK="${FRAMEWORK:-dynamo-trtllm}"
export PRECISION="${PRECISION:-fp4}"
export MODEL_PATH="${MODEL_PATH:-/models/deepseek-r1-0528-fp4}"

# Workload pattern
export ISL=8192   # Input sequence length
export OSL=1024   # Output sequence length
export MAX_MODEL_LEN=9416  # ISL + OSL + 200 buffer

# Disaggregation settings for 8k/1k
# Pattern: Heavy prefill, lightweight decode
export CACHE_TRANSCEIVER_MAX_NUM_TOKENS=8448

# Client identification
export CLIENT_NAME="client2_8k1k"
export PORT_OFFSET=${PORT_OFFSET:-100}
export RESULT_FILENAME="client2_8k1k_results"

# Repository paths
DYNAMO_PATH="${DYNAMO_PATH:-/opt/benchmarks/dynamo}"
PERFORMANCE_SWEEPS_PATH="$DYNAMO_PATH/components/backends/trtllm/performance_sweeps"

# Logging
LOG_DIR="/var/log/nvl72_inference"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/client2_8k1k_$(date +%Y%m%d_%H%M%S).log"

# ============================================================================
# LOGGING FUNCTION
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# ============================================================================
# SETUP DYNAMO
# ============================================================================

setup_dynamo() {
    log "Setting up Dynamo framework..."
    
    if [ ! -d "$DYNAMO_PATH" ]; then
        log "Cloning Dynamo repository..."
        git clone https://github.com/ai-dynamo/dynamo.git "$DYNAMO_PATH"
        cd "$DYNAMO_PATH"
        git checkout release/0.5.1-rc0.20251105
        git submodule update --init --recursive
        log "Dynamo repository cloned successfully"
    else
        log "Dynamo repository already exists at $DYNAMO_PATH"
    fi
    
    cd "$PERFORMANCE_SWEEPS_PATH"
}

# ============================================================================
# DEPLOYMENT CONFIGURATIONS
# ============================================================================

deploy_low_concurrency() {
    log "========================================"
    log "Client 2 - Low Concurrency Configuration"
    log "========================================"
    log "Pattern: 8k input / 1k output"
    log "Configuration: 1 ctx node, 3 gen nodes, TP=8"
    log "Target concurrency: 1-18"
    
    # Configuration from repository (runners/launch_gb200-nv.sh line 127):
    # This is a TEP (Tensor-Expert Parallel) mode for lower concurrency
    # - ctx_num=1: 1 context/prefill node (8 GPUs with TP=8)
    # - gen_num=3: 3 generation/decode nodes (24 GPUs total, TP=8 each)
    # - gen_tp_size=8: Tensor parallel size
    # - gen_batch_size=16: Small batches for low latency
    # - gen_max_num_tokens=64: Short generation (1k output)
    # - gen_gpu_memory_fraction=0.9: Can use more memory with lower concurrency
    # - gen_eplb_num_slots=3: Expert load balancing
    # - gen_mtp_size=0: MTP enabled
    # - concurrency: 1-18 requests
    
    ./submit_disagg.sh "mtp=on" "tep" \
        1 3 8 16 64 \
        "0.9" 3 0 "1 2 4 8 18" \
        2>&1 | tee -a "$LOG_FILE"
    
    log "Low concurrency configuration deployed"
}

deploy_medium_concurrency() {
    log "========================================="
    log "Client 2 - Medium Concurrency Configuration"
    log "========================================="
    log "Pattern: 8k input / 1k output"
    log "Configuration: 5 ctx nodes, 1 gen node, TP=32"
    log "Target concurrency: 128-269"
    
    # Configuration from repository (line 129):
    # DEP (Data-Expert Parallel) mode for medium concurrency
    # - ctx_num=5: 5 context/prefill nodes (40 GPUs total, TP=8 each)
    # - gen_num=1: 1 generation/decode node (32 GPUs with TP=32)
    # - gen_tp_size=32: High TP for generation
    # - gen_batch_size=8: Small batches due to large context
    # - gen_max_num_tokens=32: Small chunks for 1k output
    # - gen_gpu_memory_fraction=0.7: Conservative due to large context
    # - concurrency: 128-269
    
    ./submit_disagg.sh "mtp=on" "dep" \
        5 1 32 8 32 \
        "0.7" 3 0 "128 269" \
        2>&1 | tee -a "$LOG_FILE"
    
    log "Medium concurrency configuration deployed"
}

deploy_high_concurrency() {
    log "========================================="
    log "Client 2 - High Concurrency Configuration"
    log "========================================="
    log "Pattern: 8k input / 1k output"
    log "Configuration: 8 ctx nodes, 1 gen node, TP=32"
    log "Target concurrency: 538-1075"
    
    # Configuration from repository (line 131):
    # Maximum prefill throughput for high concurrency
    # - ctx_num=8: 8 context/prefill nodes (64 GPUs total, TP=8 each)
    # - gen_num=1: 1 generation/decode node (32 GPUs with TP=32)
    # - gen_tp_size=32: High TP
    # - gen_batch_size=16: Moderate batches
    # - gen_max_num_tokens=64: Moderate chunks
    # - gen_gpu_memory_fraction=0.7
    # - concurrency: 538-1075 requests
    
    ./submit_disagg.sh "mtp=on" "dep" \
        8 1 32 16 64 \
        "0.7" 3 0 "538 1075" \
        2>&1 | tee -a "$LOG_FILE"
    
    log "High concurrency configuration deployed"
}

deploy_very_high_concurrency() {
    log "========================================="
    log "Client 2 - Very High Concurrency Configuration"
    log "========================================="
    log "Pattern: 8k input / 1k output"
    log "Configuration: 8 ctx nodes, 1 gen node, TP=16"
    log "Target concurrency: 1075-2150"
    
    # Configuration from repository (line 133):
    # Alternative high concurrency with different TP strategy
    # - ctx_num=8: 8 context/prefill nodes
    # - gen_num=1: 1 generation/decode node
    # - gen_tp_size=16: Lower TP, more replicas
    # - gen_batch_size=64: Larger batches
    # - gen_max_num_tokens=256: Larger chunks
    # - gen_gpu_memory_fraction=0.75
    # - concurrency: 1075-2150 requests
    
    ./submit_disagg.sh "mtp=on" "dep" \
        8 1 16 64 256 \
        "0.75" 2 0 "1075 2150" \
        2>&1 | tee -a "$LOG_FILE"
    
    log "Very high concurrency configuration deployed"
}

# ============================================================================
# MONITORING
# ============================================================================

start_monitoring() {
    log "Starting monitoring for Client 2..."
    
    # GPU utilization monitoring
    nvidia-smi dmon -s pucvmet -d 10 > "$LOG_DIR/client2_gpu_metrics.log" 2>&1 &
    GPU_MON_PID=$!
    log "GPU monitoring started (PID: $GPU_MON_PID)"
    
    # NVLink monitoring
    nvidia-smi nvlink -g 0 -c > "$LOG_DIR/client2_nvlink_metrics.log" 2>&1 &
    NVLINK_MON_PID=$!
    log "NVLink monitoring started (PID: $NVLINK_MON_PID)"
    
    # Save monitoring PIDs
    echo "$GPU_MON_PID" > "$LOG_DIR/client2_monitoring.pids"
    echo "$NVLINK_MON_PID" >> "$LOG_DIR/client2_monitoring.pids"
}

# ============================================================================
# HEALTH CHECK
# ============================================================================

health_check() {
    log "Running health checks..."
    
    # Check GPU availability
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    log "Available GPUs: $GPU_COUNT"
    
    if [ "$GPU_COUNT" -lt 36 ]; then
        log "WARNING: Expected at least 36 GPUs for Client 2, found $GPU_COUNT"
    fi
    
    # Check NVLink connectivity
    log "Checking NVLink topology..."
    nvidia-smi topo -m > "$LOG_DIR/nvlink_topology.txt"
    log "NVLink topology saved to $LOG_DIR/nvlink_topology.txt"
    
    # Check model path
    if [ ! -d "$MODEL_PATH" ]; then
        log "WARNING: Model path $MODEL_PATH does not exist"
    else
        log "Model path verified: $MODEL_PATH"
    fi
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    log "========================================="
    log "Client 2 (8k/1k) Deployment Starting"
    log "========================================="
    log "Client: $CLIENT_NAME"
    log "Model: $MODEL"
    log "Framework: $FRAMEWORK"
    log "Pattern: ${ISL} input / ${OSL} output"
    log "Log file: $LOG_FILE"
    log "========================================="
    
    # Pre-flight checks
    health_check
    
    # Setup environment
    setup_dynamo
    
    # Start monitoring
    start_monitoring
    
    # Deploy configurations based on target load
    DEPLOYMENT_MODE="${DEPLOYMENT_MODE:-medium}"
    
    case "$DEPLOYMENT_MODE" in
        low)
            log "Deploying LOW concurrency configuration..."
            deploy_low_concurrency
            ;;
        medium)
            log "Deploying MEDIUM concurrency configuration..."
            deploy_medium_concurrency
            ;;
        high)
            log "Deploying HIGH concurrency configuration..."
            deploy_high_concurrency
            ;;
        very-high)
            log "Deploying VERY HIGH concurrency configuration..."
            deploy_very_high_concurrency
            ;;
        all)
            log "Deploying ALL configurations..."
            deploy_low_concurrency
            sleep 60  # Stagger deployments
            deploy_medium_concurrency
            sleep 60
            deploy_high_concurrency
            sleep 60
            deploy_very_high_concurrency
            ;;
        *)
            log "ERROR: Invalid DEPLOYMENT_MODE: $DEPLOYMENT_MODE"
            log "Valid options: low, medium, high, very-high, all"
            exit 1
            ;;
    esac
    
    log "========================================="
    log "Client 2 (8k/1k) Deployment Complete"
    log "========================================="
    log "Configuration Summary:"
    log "  - Input Length: ${ISL} tokens"
    log "  - Output Length: ${OSL} tokens"
    log "  - Prefill GPUs: 8-64 (heavy, 1-8 nodes)"
    log "  - Decode GPUs: 8-32 (lightweight, 1-3 nodes)"
    log "  - Total GPUs: 16-96 (scalable based on load)"
    log ""
    log "Monitoring logs:"
    log "  - Main log: $LOG_FILE"
    log "  - GPU metrics: $LOG_DIR/client2_gpu_metrics.log"
    log "  - NVLink metrics: $LOG_DIR/client2_nvlink_metrics.log"
    log ""
    log "Next steps:"
    log "  1. Monitor TTFT (critical for 8k context)"
    log "  2. Adjust prefill pool based on queue depth"
    log "  3. Scale up ctx_num if TTFT > 150ms"
    log "========================================="
}

# ============================================================================
# CLEANUP HANDLER
# ============================================================================

cleanup() {
    log "Received interrupt signal, cleaning up..."
    
    if [ -f "$LOG_DIR/client2_monitoring.pids" ]; then
        while read pid; do
            kill "$pid" 2>/dev/null || true
        done < "$LOG_DIR/client2_monitoring.pids"
        rm "$LOG_DIR/client2_monitoring.pids"
    fi
    
    log "Cleanup complete"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Run main function
main "$@"

