# Configuration Reference Guide

Quick reference for all production configurations on NVL72.

## TL;DR Command Reference

```bash
# Production deployment (most common)
export HF_TOKEN="your_token"
./deploy_nvl72_dual_client.sh balanced

# Individual clients
DEPLOYMENT_MODE=medium ./deploy_client1_1k8k.sh  # 1k→8k pattern
DEPLOYMENT_MODE=medium ./deploy_client2_8k1k.sh  # 8k→1k pattern

# Monitoring
./deploy_nvl72_dual_client.sh monitor
```

## Configuration Comparison Matrix

### Client 1: 1k/8k (Short Input → Long Output)

| Mode | Prefill GPUs | Decode GPUs | Total | Concurrency | Use Case |
|------|--------------|-------------|-------|-------------|----------|
| **Low** | 8 (1×TP8) | 32 (4×TP8) | 40 | 256-512 | Testing, dev |
| **Medium** | 8 (1×TP8) | 48 (6×TP8) | 56 | 1024-2048 | **Production** |
| **High** | 16 (2×TP8) | 64 (8×TP8) | 80 | 4096-8192 | Peak load |

### Client 2: 8k/1k (Long Input → Short Output)

| Mode | Prefill GPUs | Decode GPUs | Total | Concurrency | Use Case |
|------|--------------|-------------|-------|-------------|----------|
| **Low** | 8 (1×TP8) | 24 (3×TP8) | 32 | 1-18 | Testing, dev |
| **Medium** | 40 (5×TP32) | 32 (1×TP32) | 72 | 128-269 | **Production** |
| **High** | 64 (8×TP32) | 32 (1×TP32) | 96 | 538-1075 | Peak load |
| **Very High** | 64 (8×TP16) | 16 (1×TP16) | 80 | 1075-2150 | Maximum |

## Dual-Client Allocation Strategies

### Option 1: Balanced (Default)
```
Client 1: 36 GPUs (8 prefill + 28 decode)
Client 2: 36 GPUs (28 prefill + 8 decode)
Total:    72 GPUs
```

**When to use**: Equal traffic volume from both clients

### Option 2: Client1-Heavy
```
Client 1: 48 GPUs (8 prefill + 40 decode)
Client 2: 24 GPUs (16 prefill + 8 decode)
Total:    72 GPUs
```

**When to use**: Heavy creative writing, code generation workloads

### Option 3: Client2-Heavy
```
Client 1: 24 GPUs (8 prefill + 16 decode)
Client 2: 48 GPUs (40 prefill + 8 decode)
Total:    72 GPUs
```

**When to use**: Heavy RAG, document Q&A, summarization workloads

## Parameter Reference

### submit_disagg.sh Parameters

```bash
./submit_disagg.sh <mtp_mode> <mode> [ctx_num] [gen_num] [gen_tp_size] \
  [gen_batch_size] [gen_max_num_tokens] [gen_gpu_memory_fraction] \
  [gen_eplb_num_slots] [gen_mtp_size] [gen_concurrency_list]
```

| Parameter | Description | Typical Values | Impact |
|-----------|-------------|----------------|--------|
| `mtp_mode` | Multi-token prediction | `mtp=on`, `mtp=off` | 1.5-2x decode speedup |
| `mode` | Parallelism strategy | `tep`, `dep` | TEP=low conc, DEP=high conc |
| `ctx_num` | Context/prefill nodes | 1-8 | Higher = faster prefill |
| `gen_num` | Generation/decode nodes | 1-8 | Higher = more decode throughput |
| `gen_tp_size` | Tensor parallel size | 8, 16, 32 | Higher = more memory |
| `gen_batch_size` | Batch size | 8-256 | Higher = better throughput |
| `gen_max_num_tokens` | Max tokens/forward | 32-512 | Higher = faster but more memory |
| `gen_gpu_memory_fraction` | Memory usage % | 0.6-0.9 | Lower = safer, higher = more capacity |
| `gen_eplb_num_slots` | Expert load balance | 0, 1, 2, 3 | For MoE models only |
| `gen_mtp_size` | MTP config | 0 (off), 1-3 | MTP speculation depth |
| `gen_concurrency_list` | Target concurrency | Space-separated | Test multiple values |

## Execution Modes Explained

### TEP (Tensor-Expert Parallel)
- **Use for**: Low-to-medium concurrency
- **Characteristics**: Lower batch sizes, faster iteration
- **Memory**: More efficient
- **Example**: 1-50 concurrent requests

### DEP (Data-Expert Parallel)
- **Use for**: Medium-to-high concurrency
- **Characteristics**: Larger batches, higher throughput
- **Memory**: Higher usage
- **Example**: 100+ concurrent requests

## Memory Configuration Guide

### Conservative (Safest)
```bash
gen_gpu_memory_fraction=0.65
```
- Use for: First deployment, unknown load patterns
- Headroom: 35% for overhead
- Risk: Low OOM risk

### Balanced (Recommended)
```bash
gen_gpu_memory_fraction=0.75
```
- Use for: Production with known patterns
- Headroom: 25% for overhead
- Risk: Moderate, well-tested

### Aggressive (Maximum Throughput)
```bash
gen_gpu_memory_fraction=0.85
```
- Use for: Benchmarking, peak performance
- Headroom: 15% only
- Risk: Higher OOM risk under load spikes

## Batch Size Tuning

### Small Batches (Latency-Optimized)
```
gen_batch_size=8-16
```
- **TTFT**: Excellent (fastest)
- **Throughput**: Lower
- **Use for**: Interactive applications, real-time responses

### Medium Batches (Balanced)
```
gen_batch_size=32-64
```
- **TTFT**: Good
- **Throughput**: Good
- **Use for**: Production with mixed requirements

### Large Batches (Throughput-Optimized)
```
gen_batch_size=128-256
```
- **TTFT**: Higher latency
- **Throughput**: Excellent
- **Use for**: Batch processing, offline workloads

## Concurrency Planning

### Estimating Required Concurrency

**Client 1 (1k→8k):**
```
Requests/sec × 30 seconds (generation time) = Required concurrency
Example: 100 req/s × 30s = 3000 concurrent requests
→ Use HIGH mode (4096-8192 capacity)
```

**Client 2 (8k→1k):**
```
Requests/sec × 4 seconds (total time) = Required concurrency
Example: 200 req/s × 4s = 800 concurrent requests
→ Use HIGH mode (538-1075 capacity)
```

## Performance Tuning Checklist

### For Maximum Throughput
- [x] Use DEP mode
- [x] Enable MTP: `mtp=on`
- [x] Large batch size: 128-256
- [x] High memory fraction: 0.80-0.85
- [x] More decode nodes for 1k→8k
- [x] More prefill nodes for 8k→1k

### For Minimum Latency
- [x] Use TEP mode
- [x] Small batch size: 8-16
- [x] Lower memory fraction: 0.70-0.75
- [x] Fewer, faster nodes
- [x] Disable MTP initially
- [x] Monitor TTFT closely

### For Maximum Stability
- [x] Start with balanced mode
- [x] Conservative memory: 0.65-0.70
- [x] Medium batch size: 32-64
- [x] Monitor for 24h before scaling
- [x] Enable all monitoring
- [x] Test failover scenarios

## Common Configuration Patterns

### Pattern 1: Creative Writing Service
```bash
# Client: 1k→8k (prompts → stories)
# Priority: High decode throughput
DEPLOYMENT_MODE=high ./deploy_client1_1k8k.sh
```
**Why**: Long generation (8k tokens) needs many decode GPUs

### Pattern 2: Document Q&A Service
```bash
# Client: 8k→1k (documents → answers)
# Priority: Fast prefill of long context
DEPLOYMENT_MODE=high ./deploy_client2_8k1k.sh
```
**Why**: Long context (8k tokens) needs many prefill GPUs

### Pattern 3: Mixed Production
```bash
# Both clients active, balanced load
./deploy_nvl72_dual_client.sh balanced
```
**Why**: General-purpose deployment

### Pattern 4: Code Generation Platform
```bash
# Heavy code generation (1k→8k)
# Occasional long context (8k→1k)
./deploy_nvl72_dual_client.sh client1-heavy
```
**Why**: Most traffic is code generation with long outputs

### Pattern 5: RAG + Summarization Platform
```bash
# Heavy RAG with long context (8k→1k)
# Occasional creative generation (1k→8k)
./deploy_nvl72_dual_client.sh client2-heavy
```
**Why**: Most traffic is context-heavy retrieval

## Monitoring Metrics

### Key Metrics to Track

| Metric | Target | Alert Threshold | Action |
|--------|--------|-----------------|--------|
| GPU Utilization | 75-90% | <60% or >95% | Scale or rebalance |
| TTFT (1k→8k) | <20ms | >50ms | Check prefill pool |
| TTFT (8k→1k) | <150ms | >300ms | Increase ctx_num |
| Decode tok/s | >200/GPU | <150/GPU | Check batch size |
| Memory usage | 70-85% | >90% | Reduce memory fraction |
| Queue depth | <10 | >50 | Scale up resources |
| Request drops | 0% | >1% | Investigate bottleneck |

### Monitoring Commands

```bash
# Real-time GPU utilization
nvidia-smi dmon -s pucvmet -d 1

# Memory usage across all GPUs
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# NVLink bandwidth
nvidia-smi nvlink --status

# Process monitoring
watch -n 1 'ps aux | grep -E "trtllm|disagg"'

# Log tail
tail -f /var/log/nvl72_inference/*.log
```

## Troubleshooting Decision Tree

```
Is GPU utilization < 70%?
├─ YES: Increase batch size or concurrency
└─ NO: Continue

Is TTFT high for Client 1 (>50ms)?
├─ YES: Increase prefill nodes (unlikely for 1k input)
└─ NO: Continue

Is TTFT high for Client 2 (>300ms)?
├─ YES: Increase ctx_num (5 → 8 nodes)
└─ NO: Continue

Is decode throughput low (<150 tok/s/GPU)?
├─ YES: 
│   ├─ Check batch formation
│   ├─ Increase gen_batch_size
│   └─ Enable MTP if disabled
└─ NO: Continue

Are there OOM errors?
├─ YES:
│   ├─ Reduce gen_gpu_memory_fraction (0.75 → 0.65)
│   ├─ Reduce batch size
│   ├─ Enable KV cache FP8
│   └─ Reduce max_num_tokens
└─ NO: System healthy ✓
```

## Quick Reference: Tested Configurations

### From Repository (Validated on GB200)

**8k/1k Patterns** (Client 2):
```bash
# Line 127: TEP low concurrency
./submit_disagg.sh "mtp=on" "tep" 1 3 8 16 64 "0.9" 3 0 "1 2 4 8 18"

# Line 129: DEP medium concurrency
./submit_disagg.sh "mtp=on" "dep" 5 1 32 8 32 "0.7" 3 0 "128 269"

# Line 131: DEP high concurrency
./submit_disagg.sh "mtp=on" "dep" 8 1 32 16 64 "0.7" 3 0 "538"

# Line 133: DEP very high concurrency
./submit_disagg.sh "mtp=on" "dep" 8 1 16 64 256 "0.75" 2 0 "1075"
```

**1k/1k Pattern** (Reference):
```bash
# Line 101: TEP configuration
./submit_disagg.sh "mtp=on" "tep" 1 4 8 32 128 "0.9" 3 0 "1 2 4 8 16 36"

# Line 103: DEP medium concurrency
./submit_disagg.sh "mtp=on" "dep" 1 1 16 64 256 "0.7" 3 0 "512 1075"
```

## Environment Variable Quick Reference

```bash
# Essential
export HF_TOKEN="hf_..."                                    # Required
export MODEL_PATH="/models/deepseek-r1-0528-fp4"          # Model location

# Common customizations
export DEPLOYMENT_MODE="medium"                             # low|medium|high
export IMAGE="nvidia/deepseek-r1-fp4:latest"               # Container image
export MODEL="deepseek-ai/DeepSeek-R1-0528"                # Model ID
export FRAMEWORK="dynamo-trtllm"                           # Framework
export PRECISION="fp4"                                      # Precision

# Advanced
export DYNAMO_PATH="/opt/benchmarks/dynamo"                # Repo path
export PORT_OFFSET="0"                                      # Port offset
export CACHE_TRANSCEIVER_MAX_NUM_TOKENS="8448"            # KV cache buffer
```

## File Structure

```
production_designs/
├── README.md                           # Main usage guide
├── CONFIGURATION_REFERENCE.md          # This file
├── nvl72_dual_client_design.md         # Full design document
├── deploy_nvl72_dual_client.sh         # Master orchestrator
├── deploy_client1_1k8k.sh              # Client 1 deployment
└── deploy_client2_8k1k.sh              # Client 2 deployment
```

## Support Resources

- **Design Document**: `nvl72_dual_client_design.md` - Full architecture and rationale
- **Usage Guide**: `README.md` - How to use the scripts
- **This File**: Quick reference and configuration lookup
- **Repository**: `../runners/launch_gb200-nv.sh` - Source of truth for configs

---

**Quick Start**: 
```bash
export HF_TOKEN="your_token" && ./deploy_nvl72_dual_client.sh balanced
```

**Version**: 1.0  
**Last Updated**: November 15, 2025

