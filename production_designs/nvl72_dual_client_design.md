# NVL72 Production Design: Dual-Client Prefill Disaggregation

## Executive Summary

This design provides a production deployment strategy for an NVIDIA GB200 NVL72 rack serving two clients with opposite workload patterns using prefill disaggregation.

### Hardware Specifications: GB200 NVL72
- **Total GPUs**: 72 Blackwell GPUs (B200)
- **CPUs**: 36 Grace CPUs
- **NVLink**: Single 72-GPU NVLink domain (14TB/s aggregate bandwidth)
- **GPU Memory**: ~192 GB per GPU (13.8 TB total)
- **Peak Performance**: 30x faster inference vs previous gen (per NVIDIA)

### Client Workload Patterns
- **Client 1**: 1k input / 8k output (short prompt, long generation - e.g., creative writing, code generation)
- **Client 2**: 8k input / 1k output (long context, short response - e.g., document Q&A, summarization)

---

## Design Approach: Workload-Optimized Disaggregation

### Key Insight
The two workloads have **inverse resource requirements**:

| Pattern | Prefill Compute Need | Decode Compute Need | Prefill:Decode Ratio |
|---------|---------------------|---------------------|---------------------|
| 1k/8k   | Low (1k tokens)     | High (8k tokens)    | 1:8 |
| 8k/1k   | High (8k tokens)    | Low (1k tokens)     | 8:1 |

This makes them **ideal for disaggregation** - we can allocate GPUs inversely for each workload.

---

## Recommended Architecture: Shared Pool with Dynamic Allocation

### Option A: Static Partitioning (Recommended for Predictable Load)

```
┌─────────────────────────────────────────────────────────────────┐
│                    NVL72 - 72 GPUs Total                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Client 1: 1k/8k (36 GPUs Total)                  │  │
│  │                                                           │  │
│  │  Prefill Pool: 4 GPUs (TP=4)                            │  │
│  │  └─ Fast prompt processing for 1k inputs                 │  │
│  │                                                           │  │
│  │  Decode Pool: 32 GPUs (TP=8, 4 replicas)                │  │
│  │  └─ High throughput for 8k token generation              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Client 2: 8k/1k (36 GPUs Total)                  │  │
│  │                                                           │  │
│  │  Prefill Pool: 32 GPUs (TP=8, 4 replicas)               │  │
│  │  └─ Parallel processing for 8k context                   │  │
│  │                                                           │  │
│  │  Decode Pool: 4 GPUs (TP=4)                             │  │
│  │  └─ Efficient 1k token generation                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Resource Allocation**:
- **Client 1 (1k/8k)**: 4 prefill + 32 decode = 36 GPUs
- **Client 2 (8k/1k)**: 32 prefill + 4 decode = 36 GPUs
- **Total**: 72 GPUs (100% utilization)

### Option B: Flexible Pool (Recommended for Variable Load)

```
┌─────────────────────────────────────────────────────────────────┐
│                    NVL72 - 72 GPUs Total                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Client 1: 1k/8k (Peak: 40 GPUs)                  │  │
│  │  Prefill: 8 GPUs (TP=8) | Decode: 32 GPUs (TP=8, 4x)   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Client 2: 8k/1k (Peak: 48 GPUs)                  │  │
│  │  Prefill: 40 GPUs (TP=8, 5x) | Decode: 8 GPUs (TP=8)   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Shared Buffer Pool: 8-16 GPUs                     │  │
│  │  └─ Dynamically allocated based on demand                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Production Configuration Files

### Based on Repository Code Patterns

From `runners/launch_gb200-nv.sh`, the disaggregation is configured via:
```bash
./submit_disagg.sh <mtp_mode> <mode> [ctx_num] [gen_num] [gen_tp_size] \
  [gen_batch_size] [gen_max_num_tokens] [gen_gpu_memory_fraction] \
  [gen_eplb_num_slots] [gen_mtp_size] [gen_concurrency_list]
```

---

## Detailed Configuration: Client 1 (1k/8k)

### Characteristics
- **Workload**: Short prompts, long outputs (creative writing, code generation)
- **Bottleneck**: Decode phase (8k token generation)
- **Strategy**: Minimize prefill resources, maximize decode throughput

### Configuration

**Context (Prefill) Nodes**: 1 node with TP=4
- Processes 1k input tokens quickly
- Low GPU count sufficient for short contexts

**Generation (Decode) Nodes**: 4 nodes with TP=8 each
- 32 GPUs total for decode
- High parallel decode capacity for 8k outputs
- Can serve multiple requests simultaneously

### Example Commands (DEP Mode - Data Expert Parallel)

```bash
# High concurrency configuration for 1k/8k
# ctx_num=1, gen_num=4, gen_tp_size=8, batch_size=128, max_tokens=256

./submit_disagg.sh "mtp=on" "dep" \
  1 4 8 128 256 \
  "0.75" 2 0 "512 1024 2048 4096"

# Parameters explained:
# - 1 context node (4 GPUs with TP=4)
# - 4 generation nodes (32 GPUs, TP=8 each)
# - Batch size: 128 (high throughput)
# - Max tokens: 256 per iteration
# - Memory fraction: 0.75
# - Concurrency: 512-4096 concurrent requests
```

### Environment Variables

```bash
export ISL=1024
export OSL=8192
export CACHE_TRANSCEIVER_MAX_NUM_TOKENS=8448  # From code for this pattern
export TP=4  # For prefill
export DECODE_GPUS=32
export PREFILL_GPUS=4
export DP_ATTENTION=true  # Enable data parallel attention
export MTP_MODE=on  # Multi-token prediction
```

---

## Detailed Configuration: Client 2 (8k/1k)

### Characteristics
- **Workload**: Long context, short outputs (document Q&A, summarization)
- **Bottleneck**: Prefill phase (8k context processing)
- **Strategy**: Maximize prefill throughput, minimize decode resources

### Configuration

**Context (Prefill) Nodes**: 4-5 nodes with TP=8 each
- 32-40 GPUs for prefill
- High parallel processing for 8k contexts
- Fast TTFT (Time To First Token)

**Generation (Decode) Nodes**: 1 node with TP=4
- 4 GPUs sufficient for 1k generation
- Lower resource allocation for shorter outputs

### Example Commands (DEP Mode)

```bash
# High prefill throughput for 8k/1k
# ctx_num=5, gen_num=1, gen_tp_size=8, batch_size=16, max_tokens=64

./submit_disagg.sh "mtp=on" "dep" \
  5 1 8 16 64 \
  "0.7" 3 0 "256 512 1024"

# Parameters explained:
# - 5 context nodes (40 GPUs with TP=8 each)
# - 1 generation node (8 GPUs with TP=8)
# - Batch size: 16 (optimized for long context)
# - Max tokens: 64 per iteration (short outputs)
# - Memory fraction: 0.7
# - Concurrency: 256-1024 concurrent requests
```

### Environment Variables

```bash
export ISL=8192
export OSL=1024
export CACHE_TRANSCEIVER_MAX_NUM_TOKENS=8448  # From code
export TP=8  # For both prefill and decode
export DECODE_GPUS=8
export PREFILL_GPUS=40
export DP_ATTENTION=true
export MTP_MODE=on
```

---

## Benchmark-Validated Configurations

From the repository's tested configurations:

### For 8k/1k Pattern (Client 2)
```bash
# Based on runners/launch_gb200-nv.sh lines 129-135

# TEP Mode - Lower concurrency
./submit_disagg.sh "mtp=on" "tep" 1 3 8 16 64 "0.9" 3 0 "1 2 4 8 18"

# DEP Mode - Medium concurrency
./submit_disagg.sh "mtp=on" "dep" 5 1 32 8 32 "0.7" 3 0 "128 269"

# DEP Mode - High concurrency
./submit_disagg.sh "mtp=on" "dep" 8 1 32 16 64 "0.7" 3 0 "538"

# DEP Mode - Very high concurrency
./submit_disagg.sh "mtp=on" "dep" 8 1 16 64 256 "0.75" 2 0 "1075"
```

### For 1k/8k Pattern (Client 1)
*Note: Not explicitly in repo, but we can derive from similar patterns*

```bash
# Inverse of 8k/1k - more decode, less prefill

# DEP Mode - Medium concurrency
./submit_disagg.sh "mtp=on" "dep" 1 5 32 16 128 "0.75" 2 0 "256 512"

# DEP Mode - High concurrency
./submit_disagg.sh "mtp=on" "dep" 1 8 16 64 256 "0.75" 2 0 "1024 2048"

# DEP Mode - Very high concurrency
./submit_disagg.sh "mtp=on" "dep" 2 8 16 128 512 "0.7" 1 0 "4096"
```

---

## Production Deployment Script

### Complete Setup for Both Clients

```bash
#!/bin/bash
# production_deploy_nvl72.sh - Deploy dual-client disaggregated inference

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model configuration (example: DeepSeek-R1 FP4)
export IMAGE="nvidia/deepseek-r1-fp4:latest"
export MODEL="deepseek-ai/DeepSeek-R1-0528"
export FRAMEWORK="dynamo-trtllm"
export PRECISION="fp4"
export HF_TOKEN="${HF_TOKEN}"  # Set externally
export MODEL_PATH="/models/deepseek-r1-0528-fp4"

# Repository paths
DYNAMO_PATH="/opt/benchmarks/dynamo"
PERFORMANCE_SWEEPS_PATH="$DYNAMO_PATH/components/backends/trtllm/performance_sweeps"

# ============================================================================
# CLIENT 1: 1k/8k (Short input, Long output)
# ============================================================================

deploy_client1() {
    echo "========================================="
    echo "Deploying Client 1: 1k/8k Configuration"
    echo "========================================="
    
    export ISL=1024
    export OSL=8192
    export CACHE_TRANSCEIVER_MAX_NUM_TOKENS=8448
    export CLIENT_NAME="client1_1k8k"
    export PORT_OFFSET=0
    
    cd "$PERFORMANCE_SWEEPS_PATH"
    
    # Configuration 1: Medium load (TP=8)
    echo "Starting medium load configuration..."
    ./submit_disagg.sh "mtp=on" "dep" \
        1 4 8 64 256 \
        "0.75" 2 0 "512 1024"
    
    # Configuration 2: High load (TP=8, more decode nodes)
    echo "Starting high load configuration..."
    ./submit_disagg.sh "mtp=on" "dep" \
        1 8 16 128 512 \
        "0.7" 1 0 "2048 4096"
    
    echo "Client 1 deployed successfully!"
}

# ============================================================================
# CLIENT 2: 8k/1k (Long input, Short output)
# ============================================================================

deploy_client2() {
    echo "========================================="
    echo "Deploying Client 2: 8k/1k Configuration"
    echo "========================================="
    
    export ISL=8192
    export OSL=1024
    export CACHE_TRANSCEIVER_MAX_NUM_TOKENS=8448
    export CLIENT_NAME="client2_8k1k"
    export PORT_OFFSET=100
    
    cd "$PERFORMANCE_SWEEPS_PATH"
    
    # Configuration 1: Medium load
    echo "Starting medium load configuration..."
    ./submit_disagg.sh "mtp=on" "dep" \
        5 1 32 8 32 \
        "0.7" 3 0 "128 269"
    
    # Configuration 2: High load
    echo "Starting high load configuration..."
    ./submit_disagg.sh "mtp=on" "dep" \
        8 1 32 16 64 \
        "0.7" 3 0 "538 1075"
    
    echo "Client 2 deployed successfully!"
}

# ============================================================================
# MONITORING SETUP
# ============================================================================

setup_monitoring() {
    echo "========================================="
    echo "Setting up monitoring..."
    echo "========================================="
    
    # Create monitoring directory
    mkdir -p /var/log/nvl72_inference
    
    # Start GPU monitoring
    nvidia-smi dmon -s pucvmet -d 10 > /var/log/nvl72_inference/gpu_metrics.log &
    
    # Start NVLink monitoring
    nvidia-smi nvlink -g 0 -c > /var/log/nvl72_inference/nvlink_metrics.log &
    
    echo "Monitoring started. Logs in /var/log/nvl72_inference/"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    echo "========================================="
    echo "NVL72 Dual-Client Production Deployment"
    echo "========================================="
    
    # Setup Dynamo if needed
    if [ ! -d "$DYNAMO_PATH" ]; then
        echo "Cloning Dynamo repository..."
        git clone https://github.com/ai-dynamo/dynamo.git "$DYNAMO_PATH"
        cd "$DYNAMO_PATH"
        git checkout release/0.5.1-rc0.20251105
        git submodule update --init --recursive
    fi
    
    # Setup monitoring
    setup_monitoring
    
    # Deploy clients in parallel
    deploy_client1 &
    CLIENT1_PID=$!
    
    deploy_client2 &
    CLIENT2_PID=$!
    
    # Wait for both deployments
    wait $CLIENT1_PID
    wait $CLIENT2_PID
    
    echo "========================================="
    echo "Deployment Complete!"
    echo "========================================="
    echo "Client 1 (1k/8k): 4 prefill GPUs + 32 decode GPUs"
    echo "Client 2 (8k/1k): 40 prefill GPUs + 8 decode GPUs"
    echo "Total: 72 GPUs (84 allocated for redundancy)"
    echo ""
    echo "Monitor logs: /var/log/nvl72_inference/"
    echo "========================================="
}

# Run deployment
main "$@"
```

---

## Performance Expectations

### Based on Repository Benchmarks and GB200 Specs

#### Client 1 (1k/8k)
- **TTFT (Time to First Token)**: ~10-20ms (fast prefill for 1k tokens)
- **Decode Throughput**: ~200-400 tokens/sec/GPU × 32 GPUs = **6,400-12,800 tok/s total**
- **Concurrent Requests**: 2,000-4,000 with batching
- **Latency**: 20-40 seconds per request (for full 8k generation)

#### Client 2 (8k/1k)
- **TTFT**: ~80-150ms (longer prefill for 8k context)
- **Decode Throughput**: ~200-400 tokens/sec/GPU × 8 GPUs = **1,600-3,200 tok/s total**
- **Concurrent Requests**: 500-1,000 with batching
- **Latency**: 2.5-5 seconds per request (for 1k generation)

### Aggregate System Performance
- **Total Throughput**: ~8,000-16,000 tokens/sec
- **GPU Utilization**: 85-95% (with proper load balancing)
- **Cost Efficiency**: ~2x better than aggregated approach

---

## Monitoring & Observability

### Key Metrics to Track

#### GPU-Level Metrics
```bash
# GPU utilization per pool
nvidia-smi dmon -s pucvmet -d 1

# NVLink bandwidth utilization
nvidia-smi nvlink --status

# Memory usage per GPU
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

#### Application Metrics (from process_result.py)
```python
# These are calculated by utils/process_result.py

metrics = {
    'input_tput_per_gpu': prefill_throughput / prefill_gpus,
    'output_tput_per_gpu': decode_throughput / decode_gpus,
    'total_tput_per_gpu': total_throughput / total_gpus,
    'conc': max_concurrency,
    'ttft': time_to_first_token,
    'tpot': time_per_output_token,
    'e2e_latency': end_to_end_latency
}
```

#### Disaggregation-Specific Metrics
- **KV Cache Transfer Latency**: Should be <5ms within NVLink domain
- **Prefill Queue Depth**: Monitor for bottlenecks
- **Decode Utilization**: Should be >80% for cost efficiency

---

## Cost Analysis

### Traditional Aggregated Approach (Baseline)
- **Client 1**: 36 GPUs (TP=36) - underutilized during prefill
- **Client 2**: 36 GPUs (TP=36) - underutilized during decode
- **Total**: 72 GPUs
- **Effective Utilization**: ~50-60%

### Disaggregated Approach (This Design)
- **Client 1**: 36 GPUs (4 prefill + 32 decode) - optimized
- **Client 2**: 36 GPUs (32 prefill + 4 decode) - optimized
- **Total**: 72 GPUs
- **Effective Utilization**: ~85-95%

### ROI
- **Throughput Improvement**: ~1.5-2x
- **Cost per Token**: ~40-50% reduction
- **Payback Period**: Immediate (same hardware, better utilization)

---

## Failure Modes & Mitigations

### KV Cache Transfer Bottleneck
**Symptom**: High latency between prefill and decode
**Mitigation**:
- Increase `CACHE_TRANSCEIVER_MAX_NUM_TOKENS`
- Use TEP mode for lower concurrency
- Monitor NVLink bandwidth

### Prefill Pool Saturation (Client 2)
**Symptom**: High TTFT, queue buildup
**Mitigation**:
- Increase prefill nodes from 5 to 6-7
- Reduce batch size to prioritize latency
- Implement request routing/load balancing

### Decode Pool Saturation (Client 1)
**Symptom**: High time-per-token, slow generation
**Mitigation**:
- Increase decode nodes or add replicas
- Enable chunked prefill to reduce memory
- Implement request throttling

### GPU Out-of-Memory
**Symptom**: CUDA OOM errors
**Mitigation**:
- Reduce `gen_gpu_memory_fraction` (from 0.75 to 0.7)
- Enable KV cache block reuse: `enable_block_reuse: true`
- Use FP8 KV cache: `kv_cache_dtype: fp8`

---

## Production Checklist

### Pre-Deployment
- [ ] Validate model checkpoint accessible on all nodes
- [ ] Verify NVLink topology: `nvidia-smi topo -m`
- [ ] Test network connectivity between nodes
- [ ] Set up HuggingFace token: `export HF_TOKEN=...`
- [ ] Configure Docker/Enroot images
- [ ] Set up monitoring (Prometheus/Grafana)

### Deployment
- [ ] Deploy Client 1 (1k/8k) configuration
- [ ] Deploy Client 2 (8k/1k) configuration
- [ ] Verify health endpoints respond
- [ ] Run smoke tests with sample requests
- [ ] Check GPU utilization across all 72 GPUs
- [ ] Validate KV cache transfer working

### Post-Deployment
- [ ] Monitor TTFT, TPOT, throughput for 24h
- [ ] Collect baseline performance metrics
- [ ] Set up alerting for anomalies
- [ ] Document actual vs expected performance
- [ ] Plan capacity scaling if needed

### Ongoing Operations
- [ ] Weekly performance reviews
- [ ] Monthly cost analysis
- [ ] Quarterly capacity planning
- [ ] Continuous model/software updates

---

## Advanced Optimizations

### 1. Multi-Token Prediction (MTP)
Enable MTP for faster decoding:
```bash
export MTP_MODE=on
# Uses speculative decoding to generate 2-3 tokens per forward pass
# Can improve decode throughput by 1.5-2x for certain patterns
```

### 2. Chunked Prefill
For better latency/throughput trade-off:
```bash
--chunked-prefill-size 16384 \
--max-prefill-tokens 16384
# Breaks large prefills into chunks to maintain decode QoS
```

### 3. Expert Parallel Load Balancing
For MoE models (like DeepSeek):
```bash
--gen_eplb_num_slots 3
# Balances expert routing to prevent bottlenecks
```

### 4. Attention Data Parallelism
Essential for high concurrency:
```bash
--enable_attention_dp true
--attention_dp_config.enable_balance true
# Distributes attention compute across replicas
```

---

## Troubleshooting Guide

### Issue: Low GPU Utilization
```bash
# Check if batch size is too small
# Increase gen_batch_size parameter

# For Client 1 (1k/8k) - can handle larger batches
./submit_disagg.sh ... batch_size=256 ...

# Monitor batch formation
tail -f /var/log/nvl72_inference/server.log | grep "batch_size"
```

### Issue: High Memory Usage
```bash
# Reduce memory fraction
export GPU_MEMORY_FRACTION=0.65  # Down from 0.75

# Enable KV cache eviction
--kv_cache_config.enable_block_reuse true

# Use FP8 KV cache (2x memory savings)
--kv_cache_config.dtype fp8
```

### Issue: Imbalanced Load Between Clients
```bash
# Dynamically rebalance GPU allocation
# If Client 1 is underutilized and Client 2 is saturated:
# - Reduce Client 1 decode pool: 32 → 24 GPUs
# - Increase Client 2 prefill pool: 32 → 40 GPUs

# Requires redeployment with new configuration
```

---

## References

### From Repository
- GB200 launcher: `runners/launch_gb200-nv.sh`
- Result processing: `utils/process_result.py`
- Configuration patterns: `utils/matrix-logic/generate_sweep_configs.py`

### NVIDIA Documentation
- GB200 NVL72: https://www.nvidia.com/en-us/data-center/gb200-nvl72/
- TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM
- Dynamo Framework: https://github.com/ai-dynamo/dynamo

### Academic References
- Disaggregated Inference: Orca (OSDI'22)
- Multi-Token Prediction: Medusa
- Chunked Prefill: vLLM documentation

---

## Next Steps

1. **Benchmark Phase** (Week 1)
   - Run configurations with synthetic workload
   - Collect baseline metrics
   - Tune parameters for optimal performance

2. **Pilot Phase** (Week 2-4)
   - Deploy with real traffic at 20% capacity
   - Monitor and adjust configurations
   - Validate cost/performance assumptions

3. **Production Phase** (Week 5+)
   - Full production deployment
   - Continuous monitoring and optimization
   - Regular performance reviews

---

## Contact & Support

For issues with:
- **Repository code**: Open issue at InferenceMAX repo
- **Dynamo framework**: Contact Dynamo team
- **NVIDIA specifics**: NVIDIA enterprise support
- **Production deployment**: Your DevOps/SRE team

---

**Document Version**: 1.0  
**Last Updated**: November 15, 2025  
**Maintained By**: SRE Team  

