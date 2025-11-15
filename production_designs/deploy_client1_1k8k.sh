#!/bin/bash
# deploy_client1_1k8k.sh
# Production deployment for Client 1: 1k input / 8k output pattern
# Optimized for: Short prompts, long generation (creative writing, code generation)

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
export ISL=1024   # Input sequence length
export OSL=8192   # Output sequence length
export MAX_MODEL_LEN=9416  # ISL + OSL + 200 buffer

# Disaggregation settings for 1k/8k
# Pattern: Lightweight prefill, heavy decode
export CACHE_TRANSCEIVER_MAX_NUM_TOKENS=8448

# Client identification
export CLIENT_NAME="client1_1k8k"
export PORT_OFFSET=${PORT_OFFSET:-0}
export RESULT_FILENAME="client1_1k8k_results"

# Repository paths
DYNAMO_PATH="${DYNAMO_PATH:-/opt/benchmarks/dynamo}"
PERFORMANCE_SWEEPS_PATH="$DYNAMO_PATH/components/backends/trtllm/performance_sweeps"

# Logging
LOG_DIR="/var/log/nvl72_inference"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/client1_1k8k_$(date +%Y%m%d_%H%M%S).log"

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
    log "Client 1 - Low Concurrency Configuration"
    log "========================================"
    log "Pattern: 1k input / 8k output"
    log "Configuration: 1 ctx node, 4 gen nodes, TP=8"
    log "Target concurrency: 256-512"
    
    # Configuration:
    # - ctx_num=1: 1 context/prefill node (8 GPUs with TP=8)
    # - gen_num=4: 4 generation/decode nodes (32 GPUs total, TP=8 each)
    # - gen_tp_size=8: Tensor parallel size for generation
    # - gen_batch_size=64: Batch size (moderate for stability)
    # - gen_max_num_tokens=256: Max tokens per forward pass
    # - gen_gpu_memory_fraction=0.75: Use 75% of GPU memory
    # - gen_eplb_num_slots=2: Expert load balancing slots
    # - gen_mtp_size=0: MTP disabled initially
    # - concurrency: 256, 512
    
    ./submit_disagg.sh "mtp=off" "dep" \
        1 4 8 64 256 \
        "0.75" 2 0 "256 512" \
        2>&1 | tee -a "$LOG_FILE"
    
    log "Low concurrency configuration deployed"
}

deploy_medium_concurrency() {
    log "========================================="
    log "Client 1 - Medium Concurrency Configuration"
    log "========================================="
    log "Pattern: 1k input / 8k output"
    log "Configuration: 1 ctx node, 6 gen nodes, TP=8"
    log "Target concurrency: 1024-2048"
    
    # Scaled up decode pool for higher throughput
    # - gen_num=6: More decode nodes (48 GPUs)
    # - gen_batch_size=128: Larger batches
    # - concurrency: 1024-2048
    
    ./submit_disagg.sh "mtp=on" "dep" \
        1 6 8 128 256 \
        "0.75" 2 0 "1024 2048" \
        2>&1 | tee -a "$LOG_FILE"
    
    log "Medium concurrency configuration deployed"
}

deploy_high_concurrency() {
    log "========================================="
    log "Client 1 - High Concurrency Configuration"
    log "========================================="
    log "Pattern: 1k input / 8k output"
    log "Configuration: 2 ctx nodes, 8 gen nodes, TP=16"
    log "Target concurrency: 4096-8192"
    
    # Maximum throughput configuration
    # - ctx_num=2: Redundancy for prefill (16 GPUs, TP=8 each)
    # - gen_num=8: Maximum decode nodes (64 GPUs total, TP=8 each)
    # - gen_batch_size=256: Large batches for max throughput
    # - gen_max_num_tokens=512: Larger forward passes
    # - gen_gpu_memory_fraction=0.7: Lower to prevent OOM
    # - concurrency: 4096-8192
    
    ./submit_disagg.sh "mtp=on" "dep" \
        2 8 16 256 512 \
        "0.7" 1 0 "4096 8192" \
        2>&1 | tee -a "$LOG_FILE"
    
    log "High concurrency configuration deployed"
}

# ============================================================================
# MONITORING
# ============================================================================

start_monitoring() {
    log "Starting monitoring for Client 1..."
    
    # GPU utilization monitoring
    nvidia-smi dmon -s pucvmet -d 10 > "$LOG_DIR/client1_gpu_metrics.log" 2>&1 &
    GPU_MON_PID=$!
    log "GPU monitoring started (PID: $GPU_MON_PID)"
    
    # NVLink monitoring
    nvidia-smi nvlink -g 0 -c > "$LOG_DIR/client1_nvlink_metrics.log" 2>&1 &
    NVLINK_MON_PID=$!
    log "NVLink monitoring started (PID: $NVLINK_MON_PID)"
    
    # Save monitoring PIDs
    echo "$GPU_MON_PID" > "$LOG_DIR/client1_monitoring.pids"
    echo "$NVLINK_MON_PID" >> "$LOG_DIR/client1_monitoring.pids"
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
        log "WARNING: Expected at least 36 GPUs for Client 1, found $GPU_COUNT"
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
    log "Client 1 (1k/8k) Deployment Starting"
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
        all)
            log "Deploying ALL configurations..."
            deploy_low_concurrency
            sleep 60  # Stagger deployments
            deploy_medium_concurrency
            sleep 60
            deploy_high_concurrency
            ;;
        *)
            log "ERROR: Invalid DEPLOYMENT_MODE: $DEPLOYMENT_MODE"
            log "Valid options: low, medium, high, all"
            exit 1
            ;;
    esac
    
    log "========================================="
    log "Client 1 (1k/8k) Deployment Complete"
    log "========================================="
    log "Configuration Summary:"
    log "  - Input Length: ${ISL} tokens"
    log "  - Output Length: ${OSL} tokens"
    log "  - Prefill GPUs: 8-16 (lightweight, 1-2 nodes)"
    log "  - Decode GPUs: 32-64 (heavy, 4-8 nodes)"
    log "  - Total GPUs: 40-80 (scalable based on load)"
    log ""
    log "Monitoring logs:"
    log "  - Main log: $LOG_FILE"
    log "  - GPU metrics: $LOG_DIR/client1_gpu_metrics.log"
    log "  - NVLink metrics: $LOG_DIR/client1_nvlink_metrics.log"
    log ""
    log "Next steps:"
    log "  1. Monitor TTFT and decode throughput"
    log "  2. Adjust concurrency based on load"
    log "  3. Scale decode pool if utilization > 85%"
    log "========================================="
}

# ============================================================================
# CLEANUP HANDLER
# ============================================================================

cleanup() {
    log "Received interrupt signal, cleaning up..."
    
    if [ -f "$LOG_DIR/client1_monitoring.pids" ]; then
        while read pid; do
            kill "$pid" 2>/dev/null || true
        done < "$LOG_DIR/client1_monitoring.pids"
        rm "$LOG_DIR/client1_monitoring.pids"
    fi
    
    log "Cleanup complete"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Run main function
main "$@"

