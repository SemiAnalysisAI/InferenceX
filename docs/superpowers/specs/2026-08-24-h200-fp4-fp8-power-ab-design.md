[English](2026-08-24-h200-fp4-fp8-power-ab-design.md) | [中文](2026-08-24-h200-fp4-fp8-power-ab-design_zh.md)

# H200 FP4 vs FP8 8K/1K power A/B design

## Objective

Measure how a Qwen3.5 NVFP4 checkpoint changes interactivity and energy on an existing H200 fleet relative to the matching FP8 checkpoint. Keep the workload fixed at 8,192 input tokens and 1,024 output tokens and use SGLang single-node serving without speculative decoding.

This is a Hopper deployment study. H200 does not execute native NVFP4 Tensor Core math. SGLang serves the NVFP4 checkpoint through its Marlin weight-only W4A16 fallback, while the FP8 arm uses Hopper's native FP8 path. Results must be labelled accordingly and must not be presented as native FP4-versus-FP8 hardware efficiency.

## Frozen comparison contract

- Model architecture: Qwen3.5-397B-A17B.
- Checkpoints: `nvidia/Qwen3.5-397B-A17B-NVFP4-V2` and `Qwen/Qwen3.5-397B-A17B-FP8`.
- Hardware: one H200 node; paired arms use the same runner cohort.
- Engine: `lmsysorg/sglang:v0.5.14-cu130`.
- Workload: fixed sequence length, ISL 8192, OSL 1024, random-range ratio unchanged from the standard InferenceX fixed-sequence harness.
- Serving mode: STP only, radix cache disabled, FP8 E4M3 KV cache, BF16 Mamba state.
- Primary paired topology: TP8/EP8.
- Primary concurrency points: 1, 2, 4, 8, 16, 32, and 64.
- Replication: three independent repetitions per precision and concurrency after the canary gate.

## Runtime implementation

Add `qwen3.5-fp4-h200-sglang` to `configs/nvidia-master.yaml`. Its initial search space matches the existing H200 FP8 TP8/EP8 topology and the primary concurrency list.

Add `benchmarks/single_node/fixed_seq_len/qwen3.5_fp4_h200.sh`. It follows the existing H200 FP8 harness but selects the Hopper-compatible NVFP4 fallback explicitly:

```text
--quantization modelopt_fp4
--fp4-gemm-backend marlin
--moe-runner-backend marlin
--attention-backend flashinfer
--kv-cache-dtype fp8_e4m3
--disable-radix-cache
```

The Blackwell-only `trtllm_mha` attention and `flashinfer_trtllm` MoE settings in the B200 FP4 script must not be copied to H200.

## Staged execution and gates

1. Generate and inspect an FP4 c4 TP8/EP8 job and an FP8 c4 TP8/EP8 control from the same branch.
2. Dispatch only those two canaries.
3. Accept a canary only when all requests complete, workload metrics are valid, power telemetry is valid, both Marlin backends are confirmed in the FP4 server log, and the server shows no sustained progress stall or idle-power collapse.
4. After both canaries pass, dispatch the TP8/EP8 concurrency sweep with three repetitions per point.
5. Then probe TP4/EP1 for both precisions from low to high concurrency. Stop each arm at its first reproducible memory or runtime cliff. Probe FP4 TP2/EP1 separately as a deployment-consolidation arm.
6. Attempt c96 and c128 only after the c64 TP8/EP8 points are healthy and have adequate memory headroom.

MTP is excluded. Previous FP4 plus MTP runs stalled inside SGLang, and adding MTP would confound the precision comparison.

## Measurements and interpretation

Primary article curves:

- measured average board W/GPU versus per-user interactivity (`1 / mean TPOT`);
- measured J/output-token versus per-user interactivity.

Supporting metrics are total board J/query, total and output throughput per GPU, mean and tail TTFT, mean TPOT, and end-to-end latency. Power integration uses only the accepted workload window.

Only same-topology FP4/FP8 pairs support a precision-path comparison. TP2, TP4, and TP8 results may be compared for deployment economics using total energy, throughput, and GPU count, but a different-GPU-count comparison must not be described as a pure precision effect.

## Validation

- Add a matrix-generation test asserting balanced H200 FP4/FP8 TP8/EP8 8K/1K rows.
- Run the full `utils/matrix_logic/` test suite.
- Generate the exact c4 commands locally and inspect the emitted JSON instead of relying on filter assumptions.
- Run shell syntax checking on the new benchmark script.
- Before a remote dispatch, verify the pushed SHA and generated matrix count.
- Accept benchmark data only from successful artifacts with valid workload and power windows.

## Deliverables

- An isolated branch containing the config, Hopper FP4 launcher, test, and bilingual design documentation.
- Two c4 canary jobs and their artifact-backed acceptance result.
- If the canaries pass, a complete TP8/EP8 repeated sweep followed by staged TP4/TP2 and c96/c128 probes.
- A result ledger that separates native FP8, NVFP4 Marlin fallback, same-topology causal pairs, and deployment-consolidation comparisons.
