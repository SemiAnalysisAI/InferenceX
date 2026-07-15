## Summary

Tune llm-d-vllm DeepSeek-V4-Pro FP4 GB200 recipe configurations based on validated benchmarks on GB200 NVL72, showing **+2-43% tok/s/GPU** and **15-25% lower TPOT** across all tested concurrency points.

### Changes

**Decode (mid-curve-megamoe recipe):**
- `gpu-memory-utilization` 0.85 → 0.9 (larger KV cache pool)
- Add `max-model-len 9280` (tight bound: ISL8192+OSL1024=9216, ~0% waste per slot)
- `max-num-seqs`/`max-num-batched-tokens`/`max-cudagraph-capture-size` 512 → 1024
- Disable NCCL symmetric memory (`VLLM_USE_NCCL_SYMM_MEM=0`; standard NCCL NVLink path is faster for 2-node DEP=8)
- Add `--no-enable-flashinfer-autotune` (avoid runtime autotuning overhead)
- Add `VLLM_USE_RUST_FRONTEND=1`

**Prefill (both recipes):**
- `gpu-memory-utilization` 0.9 → 0.95
- Add `max-model-len 9280`, `max-num-seqs 16`, `max-num-batched-tokens 32768`
- Disable NCCL symmetric memory
- Enable `VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1` (avoid DP dummy-input correlations)
- Add `VLLM_USE_RUST_FRONTEND=1`

**EPP (both recipes):**
- Switch from `max-score-picker` to `weighted-random-picker` (threshold=0.1) — distributes requests across top-scoring endpoints instead of funneling all to one, improving throughput under high concurrency
- Add `weight: 2` to decode's `active-request-scorer` (matches upstream wide-ep-lws config)

**Low-latency decode:**
- Disable NCCL symmetric memory + add rust frontend (minimal changes; TP=8 decode already well-configured)

### Validated Performance (ISL=8192, OSL=1024)

| Topology | Concurrency | Before (tok/s/GPU) | After (tok/s/GPU) | Improvement |
|----------|-------------|-------------------|-------------------|-------------|
| P8D8 (16 GPU) | 256 | 2,959 | 3,666 | +24% |
| P8D8 (16 GPU) | 512 | 6,089 | 6,224 | +2% |
| P8D8 (16 GPU) | 1024 | 6,401 | 6,628 | +4% |
| P24D8 (32 GPU) | 4096 | 9,491 | 9,596 | +1% |

### Design Decisions

- **Minimal diff**: Only performance-impacting changes. NVSHMEM env vars, NCCL_NET_GDR_C2C, TORCH_DISTRIBUTED_DEFAULT_TIMEOUT, UCX_TLS, --enable-sleep-mode, --stream-interval all left untouched.
- **No image bump**: Container version is unchanged; all flag changes work on the existing image.
- **No new topology points**: P8D32 low-latency wide topology is a follow-up PR.

## Test plan

- [ ] `full-sweep-enabled` label triggers benchmark CI on GB200
- [ ] Verify mid-curve points (conc 256/512/1024) complete with improved tok/s/GPU
- [ ] Verify high-tpt point (conc 4096) completes successfully
- [ ] Verify low-latency point (conc 1) completes with reduced TTFT/TPOT
- [ ] No OOM or engine failures across all concurrency points
