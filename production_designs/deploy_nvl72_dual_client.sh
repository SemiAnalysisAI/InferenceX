#!/bin/bash
# deploy_nvl72_dual_client.sh
# Master deployment script for both clients on NVL72 rack
# Orchestrates deployment of 1k/8k and 8k/1k patterns with optimal GPU allocation

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/var/log/nvl72_inference"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master_deployment_$(date +%Y%m%d_%H%M%S).log"

# ============================================================================
# LOGGING
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [MASTER] $*" | tee -a "$MASTER_LOG"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" | tee -a "$MASTER_LOG" >&2
}

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

preflight_checks() {
    log "========================================="
    log "Running Pre-Flight Checks"
    log "========================================="
    
    # Check GPU count
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    log "Total GPUs detected: $GPU_COUNT"
    
    if [ "$GPU_COUNT" -lt 72 ]; then
        log_error "ERROR: Expected 72 GPUs (NVL72), found $GPU_COUNT"
        log_error "This script is designed for GB200 NVL72 racks"
        exit 1
    fi
    
    # Check NVIDIA driver
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    log "NVIDIA Driver version: $DRIVER_VERSION"
    
    # Check NVLink
    log "Checking NVLink topology..."
    NVLINK_STATUS=$(nvidia-smi nvlink --status 2>&1 | grep -c "Active" || echo 0)
    log "Active NVLink connections: $NVLINK_STATUS"
    
    # Check required environment variables
    if [ -z "$HF_TOKEN" ]; then
        log_error "ERROR: HF_TOKEN environment variable not set"
        log_error "Please set: export HF_TOKEN='your_token_here'"
        exit 1
    fi
    
    # Check disk space
    DISK_SPACE=$(df -h /models 2>/dev/null | awk 'NR==2 {print $4}')
    log "Available disk space for models: $DISK_SPACE"
    
    log "Pre-flight checks passed ✓"
}

# ============================================================================
# GPU ALLOCATION STRATEGY
# ============================================================================

display_gpu_allocation() {
    log "========================================="
    log "GPU Allocation Strategy"
    log "========================================="
    log ""
    log "NVL72 Rack: 72 GPUs (Blackwell B200)"
    log ""
    log "┌─────────────────────────────────────────────────────────────┐"
    log "│                Client 1: 1k/8k Pattern                       │"
    log "│  (Short prompts, long generation)                           │"
    log "│  - Prefill Pool: 8 GPUs (1 node, TP=8)                     │"
    log "│  - Decode Pool: 24-32 GPUs (3-4 nodes, TP=8 each)          │"
    log "│  - Total: 32-40 GPUs                                        │"
    log "└─────────────────────────────────────────────────────────────┘"
    log ""
    log "┌─────────────────────────────────────────────────────────────┐"
    log "│                Client 2: 8k/1k Pattern                       │"
    log "│  (Long context, short responses)                            │"
    log "│  - Prefill Pool: 24-40 GPUs (3-5 nodes, TP=8 each)         │"
    log "│  - Decode Pool: 8-24 GPUs (1-3 nodes, TP=8 each)           │"
    log "│  - Total: 32-64 GPUs                                        │"
    log "└─────────────────────────────────────────────────────────────┘"
    log ""
    log "Load balancing: 36 GPUs base each, with 8-16 GPU flex pool"
    log ""
}

# ============================================================================
# DEPLOYMENT MODES
# ============================================================================

deploy_balanced() {
    log "========================================="
    log "Deploying BALANCED mode"
    log "========================================="
    log "Equal resources for both clients"
    
    # Client 1: Medium load (36 GPUs)
    log "Starting Client 1 (1k/8k) - Medium concurrency..."
    DEPLOYMENT_MODE=medium bash "$SCRIPT_DIR/deploy_client1_1k8k.sh" &
    CLIENT1_PID=$!
    
    # Client 2: Medium load (36 GPUs)
    log "Starting Client 2 (8k/1k) - Medium concurrency..."
    DEPLOYMENT_MODE=medium bash "$SCRIPT_DIR/deploy_client2_8k1k.sh" &
    CLIENT2_PID=$!
    
    log "Both clients deploying in parallel..."
    log "Client 1 PID: $CLIENT1_PID"
    log "Client 2 PID: $CLIENT2_PID"
    
    # Wait for deployments
    wait $CLIENT1_PID
    CLIENT1_STATUS=$?
    wait $CLIENT2_PID
    CLIENT2_STATUS=$?
    
    if [ $CLIENT1_STATUS -eq 0 ] && [ $CLIENT2_STATUS -eq 0 ]; then
        log "BALANCED deployment completed successfully ✓"
    else
        log_error "Deployment failed - Client1: $CLIENT1_STATUS, Client2: $CLIENT2_STATUS"
        return 1
    fi
}

deploy_client1_heavy() {
    log "========================================="
    log "Deploying CLIENT1-HEAVY mode"
    log "========================================="
    log "Prioritize Client 1 (1k/8k) for high throughput"
    
    # Client 1: High load (48 GPUs)
    log "Starting Client 1 (1k/8k) - High concurrency..."
    DEPLOYMENT_MODE=high bash "$SCRIPT_DIR/deploy_client1_1k8k.sh" &
    CLIENT1_PID=$!
    
    # Client 2: Low load (24 GPUs)
    log "Starting Client 2 (8k/1k) - Low concurrency..."
    DEPLOYMENT_MODE=low bash "$SCRIPT_DIR/deploy_client2_8k1k.sh" &
    CLIENT2_PID=$!
    
    wait $CLIENT1_PID
    wait $CLIENT2_PID
    
    log "CLIENT1-HEAVY deployment completed ✓"
}

deploy_client2_heavy() {
    log "========================================="
    log "Deploying CLIENT2-HEAVY mode"
    log "========================================="
    log "Prioritize Client 2 (8k/1k) for high context processing"
    
    # Client 1: Low load (24 GPUs)
    log "Starting Client 1 (1k/8k) - Low concurrency..."
    DEPLOYMENT_MODE=low bash "$SCRIPT_DIR/deploy_client1_1k8k.sh" &
    CLIENT1_PID=$!
    
    # Client 2: High load (48 GPUs)
    log "Starting Client 2 (8k/1k) - High concurrency..."
    DEPLOYMENT_MODE=high bash "$SCRIPT_DIR/deploy_client2_8k1k.sh" &
    CLIENT2_PID=$!
    
    wait $CLIENT1_PID
    wait $CLIENT2_PID
    
    log "CLIENT2-HEAVY deployment completed ✓"
}

# ============================================================================
# MONITORING DASHBOARD
# ============================================================================

start_monitoring_dashboard() {
    log "========================================="
    log "Starting Monitoring Dashboard"
    log "========================================="
    
    # Create monitoring script
    cat > "$LOG_DIR/monitor_dashboard.sh" << 'EOF'
#!/bin/bash
watch -n 5 '
echo "=== NVL72 Dual-Client Monitoring Dashboard ==="
echo "Timestamp: $(date)"
echo ""
echo "=== GPU Utilization ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -20
echo ""
echo "=== Active Processes ==="
ps aux | grep -E "submit_disagg|trtllm-serve" | grep -v grep | head -10
echo ""
echo "=== Recent Logs ==="
tail -5 /var/log/nvl72_inference/client1_*.log 2>/dev/null
tail -5 /var/log/nvl72_inference/client2_*.log 2>/dev/null
echo ""
echo "Press Ctrl+C to exit"
'
EOF
    chmod +x "$LOG_DIR/monitor_dashboard.sh"
    
    log "Monitoring dashboard created at: $LOG_DIR/monitor_dashboard.sh"
    log "Run it with: bash $LOG_DIR/monitor_dashboard.sh"
}

# ============================================================================
# STATUS CHECK
# ============================================================================

check_deployment_status() {
    log "========================================="
    log "Deployment Status Check"
    log "========================================="
    
    # Check for running processes
    CLIENT1_PROCS=$(ps aux | grep -c "client1_1k8k" | grep -v grep || echo 0)
    CLIENT2_PROCS=$(ps aux | grep -c "client2_8k1k" | grep -v grep || echo 0)
    
    log "Client 1 (1k/8k) processes: $CLIENT1_PROCS"
    log "Client 2 (8k/1k) processes: $CLIENT2_PROCS"
    
    # Check GPU utilization
    log ""
    log "GPU Utilization Summary:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | \
        awk -F',' '{total+=$2; mem+=$3; count++} END {
            printf "  Average GPU utilization: %.1f%%\n", total/count;
            printf "  Average memory used: %.1f GB\n", mem/count;
        }'
    
    # Check log files
    log ""
    log "Recent log files:"
    ls -lht "$LOG_DIR"/*.log 2>/dev/null | head -5
}

# ============================================================================
# MAIN
# ============================================================================

show_usage() {
    cat << EOF
Usage: $0 [MODE] [OPTIONS]

Deployment Modes:
    balanced         Deploy both clients with equal resources (default)
    client1-heavy    Prioritize Client 1 (1k/8k) - more decode GPUs
    client2-heavy    Prioritize Client 2 (8k/1k) - more prefill GPUs
    status           Check deployment status
    stop             Stop all deployments
    monitor          Start monitoring dashboard

Options:
    --dry-run        Show configuration without deploying
    --help           Show this help message

Environment Variables:
    HF_TOKEN         HuggingFace API token (required)
    MODEL_PATH       Path to model files (optional)
    IMAGE            Docker/container image (optional)
    DYNAMO_PATH      Path to Dynamo repository (optional)

Examples:
    # Deploy with balanced allocation
    $0 balanced

    # Prioritize Client 1 for high generation throughput
    $0 client1-heavy

    # Check status
    $0 status

    # Start monitoring
    $0 monitor

EOF
}

main() {
    MODE="${1:-balanced}"
    
    case "$MODE" in
        balanced)
            log "========================================="
            log "NVL72 Dual-Client Deployment"
            log "Mode: BALANCED"
            log "========================================="
            preflight_checks
            display_gpu_allocation
            deploy_balanced
            start_monitoring_dashboard
            check_deployment_status
            ;;
        client1-heavy)
            log "Mode: CLIENT1-HEAVY (1k/8k priority)"
            preflight_checks
            display_gpu_allocation
            deploy_client1_heavy
            start_monitoring_dashboard
            check_deployment_status
            ;;
        client2-heavy)
            log "Mode: CLIENT2-HEAVY (8k/1k priority)"
            preflight_checks
            display_gpu_allocation
            deploy_client2_heavy
            start_monitoring_dashboard
            check_deployment_status
            ;;
        status)
            check_deployment_status
            ;;
        stop)
            log "Stopping all deployments..."
            pkill -f "deploy_client1_1k8k.sh" || true
            pkill -f "deploy_client2_8k1k.sh" || true
            pkill -f "submit_disagg.sh" || true
            log "Stop signal sent to all processes"
            ;;
        monitor)
            start_monitoring_dashboard
            bash "$LOG_DIR/monitor_dashboard.sh"
            ;;
        --help|-h|help)
            show_usage
            ;;
        *)
            log_error "Unknown mode: $MODE"
            show_usage
            exit 1
            ;;
    esac
}

# Run main
main "$@"

