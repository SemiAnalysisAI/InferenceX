# ISB1 KV Cache Benchmark — GMI Cloud Execution Plan

## Available Hardware

| GPU | HBM | Available | Max Context Before Offload |
|-----|-----|-----------|---------------------------|
| **GB200** | 192GB HBM3e | ✅ | ~384K tokens (FP8 KV) |
| **H100** | 80GB HBM3 | ✅ | ~128K tokens (FP8 KV) |

## Execution Order

Run benchmarks in this order — cheapest/fastest first to validate the setup works.

### Phase 1: Validation Run (1 hour)

Prove the pipeline works end-to-end before burning GPU hours.

```bash
# On H100 — single model, single concurrency, 5 min duration
export MODEL=deepseek-ai/DeepSeek-R1-0528
export TP=8
export EXPORT_FILE=datasets/isb1/exports/extension_131k/vllm/code_131k1k.json

# Launch server
bash benchmarks/single_node/dsr1_fp8_h100_vllm.sh

# Run ONE cell: 2 users, offload=off, 300s
python utils/bench_serving/benchmark_export_replay.py \
  --export-file $EXPORT_FILE \
  --max-concurrency 2 \
  --duration 300 \
  --request-mode multi-turn

# Verify result has actual_context_len > 0
python utils/process_result_isb1.py --result-file results/*.json
```

**Pass criteria:** TTFT and throughput numbers appear. `actual_context_len` > 100K.

### Phase 2: H100 KV Stress Sweep (8 hours)

H100 80GB is the interesting GPU — KV cache fills up first.

```bash
# Models to test:
#   1. DeepSeek-R1 FP8 (TP8)
#   2. GPT-OSS FP4 (TP8)

# Sweep per model:
#   users: [2, 4, 8, 16, 32, 64]        # H100 can't do 128+ at 131K
#   offload-modes: [on, off, noprefix]
#   duration: 1800s (30 min)
#   export: extension_131k/vllm/code_131k1k.json

# Total cells: 2 models × 6 concurrency × 3 offload = 36 cells
# Time: 36 × 30min = 18 hours → with 2 models sequential = ~9 hours
```

**What to look for:**
- Offload cliff: at what concurrency does offload=on start helping?
- Prefix cache hit rate: does it stay >50% under load?
- Preemption count: how many requests get evicted?
- TTFT degradation: when does p99 TTFT exceed 10s?

### Phase 3: GB200 KV Stress Sweep (8 hours)

GB200 192GB has 2.4x more HBM — the cliff comes later.

```bash
# Same sweep but higher concurrency (more HBM room):
#   users: [2, 4, 8, 16, 32, 64, 128, 256]
#   offload-modes: [on, off, noprefix]
#   duration: 1800s

# Add Qwen 3.5 (needs more memory for MoE):
#   3 models × 8 concurrency × 3 offload = 72 cells
#   Time: 72 × 30min = 36 hours → might need to cut duration to 900s
```

**What to look for:**
- At what concurrency does GB200 hit its offload cliff?
- Is the cliff at ~3x H100's cliff (proportional to HBM)?
- Does 192GB allow prefix caching to stay effective longer?

### Phase 4: Long Context Preview (4 hours, GB200 only)

500K and 1M token traces — only GB200 has enough memory.

```bash
# 500K preview (Qwen 3.5 only):
export EXPORT_FILE=datasets/isb1/exports/preview/long_context_500k/\
inferencex_trace_replay__coding_qwen3.5_xlc2_500k_preview_v1__vllm.json

# 1M preview (Qwen 3.5 only):
export EXPORT_FILE=datasets/isb1/exports/preview/long_context_1m/\
inferencex_trace_replay__coding_qwen3.5_ulc2_1m_preview_v1__vllm.json

# Low concurrency (these are HUGE contexts):
#   users: [1, 2, 4]
#   offload-modes: [on, off]
#   duration: 900s
```

**What to look for:**
- Can GB200 serve 1M context at all?
- What's the TTFT for a 1M token prefill?
- Does KV offload work at this scale?

## Estimated GPU Time

| Phase | GPU | Duration | Cost (est) |
|-------|-----|----------|------------|
| 1. Validation | H100 | 1 hour | ~$3 |
| 2. H100 sweep | H100 | 9 hours | ~$27 |
| 3. GB200 sweep | GB200 | 18 hours | ~$90 |
| 4. Long context | GB200 | 4 hours | ~$20 |
| **Total** | | **32 hours** | **~$140** |

## Portable Run Script

Use `gmi_portable_benchmark.sh` for manual runs without GitHub Actions:

```bash
# Set GMI-specific env vars
export GMI_API_KEY="..."
export HF_TOKEN="..."
export MODEL=deepseek-ai/DeepSeek-R1-0528
export GPU_TYPE=h100  # or gb200

# Run the portable benchmark
bash datasets/isb1/scripts/gmi_portable_benchmark.sh \
  --model $MODEL \
  --gpu $GPU_TYPE \
  --export-file datasets/isb1/exports/extension_131k/vllm/code_131k1k.json \
  --users 2,4,8,16,32,64 \
  --offload-modes on,off,noprefix \
  --duration 1800
```

## Result Collection

After each phase, results go to:
```
results/
├── h100_dsr1_fp8_kv_stress/
│   ├── users_2_offload_on.json
│   ├── users_2_offload_off.json
│   └── ...
└── gb200_dsr1_fp8_kv_stress/
    └── ...
```

Process and visualize:
```bash
# Aggregate results
python datasets/isb1/scripts/collect_sweep_results.py \
  --results-dir results/ \
  --output results/sweep_summary.json

# Generate Pareto frontier chart
python datasets/isb1/scripts/plot_pareto.py \
  --summary results/sweep_summary.json \
  --output results/pareto_frontier.png
```

## What Success Looks Like

After all phases, we have:
1. **Pareto frontier chart:** throughput vs p99 TTFT for H100 and GB200
2. **Offload cliff identification:** exact concurrency where offload starts helping
3. **Prefix cache benefit:** measured hit rate under realistic multi-turn load
4. **HBM scaling evidence:** does 2.4x more HBM give 2.4x more capacity?
5. **Long context feasibility:** can GB200 serve 500K/1M context at all?

These results go into the InferenceX PR as evidence that the benchmark works.
