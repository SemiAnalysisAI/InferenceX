# Offline TRT Agent Notes

Read `README.md` and `/TRT_BENCH_NOTES.md` before changing or debugging this
benchmark.

- This branch is disposable and must remain isolated from the normal serving
  sweep. Never edit `nvidia-master.yaml` or `perf-changelog.yaml`.
- The benchmark has one measurement contract and two hardware profiles:
  validated B300 DEP8 and current GB300 NVL16 DEP16. Both use DeepSeek-V4 Pro,
  exact global batches `16`, `64`, and `128`, 8192 input tokens, MTP3, two
  warmup decode rounds, and 256 measured decode rounds.
- `global_batch_size` is authoritative. TRT `max_batch_size` and CUDA graph
  size are exactly `global_batch_size / active_gpu_count`; `max_num_tokens`
  is exactly `local_batch_size * 8192` so every local prompt can prefill
  together.

## GB300 Profile

- Select with `TRT_BENCH_HARDWARE_PROFILE=gb300`. It is four physical nodes,
  four GPUs/tasks per node, and one external 16-rank
  `trtllm-llmapi-launch` world. The ranks must be exactly `0..15`.
- Use image
  `nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1`
  and TRT source `34a563ac6d8cc0ca7068c7f619e869fb8a625333`. The working
  topology/config reference is InferenceX PR #1689.
- Keep TP16, EP16, attention DP, LM-head TP, MoE TP1,
  `MEGAMOE_DEEPGEMM`, the pinned EP16/384-slot load-balancer file, KV
  fraction `0.70`, PDL, and ConfigurableMoE enabled. The pinned TRT source
  defaults `ENABLE_CONFIGURABLE_MOE` to `1`; forcing it to `0` constructs the
  MegaMoE base backend without the wrapper's concrete `forward_impl`.
- GB300 local batches are `1`, `4`, and `8`; max token shapes are `8192`,
  `32768`, and `65536`. The B300-only attention workspace, 12 GiB KV reserve,
  FP8 quantizer guard, and oversized DeepGemm chunker must remain disabled.
- The barrier arm file, rank marker, corpus, and worker artifacts must be on
  shared `/workspace`, never node-local `/tmp`.
- External MPI management workers start before rank 0 launches `run.py`.
  Therefore `dsv4_fp4_gb300_trt.sh` must export the complete output of
  `emit_rank_environment.py` before `trtllm-llmapi-launch`; setting dynamic
  `TRTLLM_BENCH_*` values only in `run.py` leaves ranks 1-15 without the
  benchmark contract. Preserve rank 0's exact preseed validation.
- Before engine launch, require four nodes with four ranks each and four
  `Completed`/`Success` NVLink Fabric records per node. Parse full
  `nvidia-smi -q` output, not the unsupported `-d FABRIC` selector, and
  require one shared non-empty `ClusterUUID` and `CliqueId` across all 16
  GPUs. Preserve `offline_rank_map_gbsN.tsv`,
  `offline_topology_gbsN.log`, `offline_allocation_gbsN.log`, and per-node
  GPU telemetry. The host launcher must stream `salloc` output and emit a
  one-minute pending-allocation heartbeat so queue time is not mistaken for
  TRT initialization. After rank 0 has copied the result and archived debug
  files, it must atomically publish `offline_completion_gbsN.json` with the
  controller return code and result status; the host verifies both before
  canceling the allocation so external MPI management ranks cannot hold the
  job until its Slurm time limit.
- The dispatchable workflow is `.github/workflows/e2e-tests.yml` with
  `inputs[hardware-profile]=gb300`; matrix concurrency is intentionally one
  because every row consumes 16 GPUs.

## B300 Baseline Constraints

- Start KV capacity from
  `kv_cache_config.free_gpu_memory_fraction=0.60`. An explicit KV token cap
  underprovisions the pinned DeepSeek-V4 multi-pool cache and staggered half
  of each local batch in run `27486168511`. For GBS128, subtract exactly
  12 GiB from TRT's calibrated `max_gpu_total_bytes` after capacity
  estimation and before final manager construction. Preserve at least TRT's
  aggregate `CacheCost.bytes_for_tokens(16 * 9344)`; fail rather than reducing
  that minimum. GBS16/64 use no reserve. Run `27491999545` proved why the
  reserve is required: final KV allocation left 1.17-2.35 GiB free before the
  real prefill requested an 8 GiB FP8 Q buffer and a following roughly 2 GiB
  BF16 RoPE buffer. The fixed
  `moe_config.max_num_tokens=65536` cap chunks only oversized fused-MoE
  tensors inside that one executor iteration; measured decode never reaches
  the cap.
- Keep the synthetic pure-context warmup request cap at 65536 tokens by
  wrapping `PyTorchModelEngine._create_warmup_request`. Never mutate
  `engine.max_num_tokens`: DeepSeek-V4 attention metadata allocates lazily
  from it, and run `27490833024` created undersized 65536-token runtime buffers
  that failed on the 84087-token KV-capacity probe. The runtime field must stay
  131072 for GBS128. Never treat the synthetic request cap as permission to
  split or shrink the validated full-batch prefill.
- Keep the GBS128 eager attention workspace reservation at
  `200 KiB * 131072 + 256 MiB = 27111981056` bytes. Run `27491160719` showed
  the capped warmup allocate 12953234944 bytes, then TRT attempted an in-place
  resize to 16600658432 bytes for its 84087-token capacity probe and hit an
  illegal memory access on every rank. Reserve only
  `TrtllmAttentionMetadata.workspace` when the cached runtime metadata is
  created. Do not preallocate or replace `cuda_graph_workspace`.
- GBS128 initialization builds two engine phases and resets attention
  metadata between them. Duplicate workspace events are expected. The KV
  reserve event occurs after phase-one calibration and must cover ranks
  `0..7` exactly.
- Keep the rank-local packed-FP8 guard at 32768 rows. It selects TRT's Triton
  quantizer only for large GBS64/128 prefill projections; measured decode has
  at most 16 rows and stays on the original fused path.
- Keep `fp8SwapABGemmRunner` chunking at 65536 rows. Run `27492438399`
  completed initialization and reached real GBS128 prefill, then failed after
  launching the Triton quantizer on all 131072 rows. The hook allocates one
  final output and writes two row chunks through the existing transformed
  DeepGemm weights. Synchronize oversized chunks for precise failures.
  Decode has at most 16 rows and must call the pinned runner unchanged.

## Shared Measurement Invariants

- `LLM.generate()` submits requests individually. The MPI entry shim patches
  TRT's idle request fetch so each warmup/measured pass waits for exactly one
  complete GBS before routing. Do not remove that barrier while claiming a
  fixed-batch result.
- Keep the request barrier disarmed throughout `LLM` construction. TRT's
  GBS128 KV-capacity calibration submits 120 internal dummy requests, which
  must pass through normally. The parent atomically creates the shared arm
  file only after initialization returns and before the first real warmup
  `generate()` call. Do not arm at MPI entry or weaken the post-arm gate.
- A successful result must prove one full-local-batch prefill iteration,
  followed by decode at the same exact local batch for 256 consecutive
  iterations, with no queued or paused requests. Never weaken this validation
  to make a run pass.
- Keep `max_stats_len=2048`. It bounds the stats history while covering the
  zero-acceptance worst case of 1024 decode iterations.
- Timing uses TRT `iterLatencyMS` with overlap scheduling disabled. It covers
  a complete TRT executor iteration, while Huawei sums internal main/MTP CANN
  timing regions. Match Huawei's aggregation: skip the first measured round,
  calculate the 25th/75th percentiles, and discard only values above
  `Q3 + 1.5 * IQR`.
- The headline metric is raw decode-round throughput:
  `GBS / decode_round_TPOT / active_gpu_count`. MTP output yield and
  acceptance are separate.
- The 1025-token measured output cap guarantees at least 256 MTP3 decode
  iterations even if every round emits four tokens. Only the first 256 valid
  full-batch rounds are measured.
- Preserve perfect routing, exact 8192-token real prompts, temperature 1,
  engine-global seed 42, LM-head TP, heuristic sparse top-k, and rank
  environment validation. ConfigurableMoE remains enabled on both profiles;
  it is required by the GB300 `MEGAMOE_DEEPGEMM` path.
- Do not enable `TLLM_METRICS_ALL_RANKS`; its per-iteration collective changes
  the timing path. Rank 0 stats are valid only after exact equal-length prompt
  routing and full-local-batch validation.
- The comparison is methodological, not identical hardware: this branch uses
  B300 or GB300 FP4 GPUs, while Huawei publishes 950DT chips with hybrid
  MXFP8/MXFP4.
- Final known-good full sweep is Actions run `27493336994` at
  `9796f5d17c96ab56136b8b9b1e196b6e6db84426`. Its GBS16/64/128 raw
  decode-step rates are `90.621203`, `248.910481`, and `434.410801`
  steps/s/GPU. Use it as the regression baseline when debugging later runs.
- `offline_aggregate.json` is authoritative. `results_bmk/agg_bmk.json` uses
  acceptance-adjusted output-token throughput and equivalent output-token
  TPOT for standard renderer fields, while retaining custom decode-round
  fields.
- Keep launcher/controller/worker phase logs and 60-second heartbeats.
  Heartbeats must retain the per-rank marker summary for warmup, clock sync,
  and executor worker start; canceled Actions jobs may not preserve the
  node-local worker log.
- Treat TRT's `Fatal error detected, initiating shutdown` line as terminal.
  The MPI parent can remain alive after all ranks fail, so waiting for the
  full controller timeout only hides the real error and wastes the node.
- Treat rank `entry_failed` and `*_error` marker events as terminal for the
  same reason. Run `27504087069` otherwise waited 1800 seconds after all 16
  external ranks failed immediately on a missing warmup-cap variable.
- For a readiness hang, dispatch with `worker-stack-period=120` and a short
  controller timeout, then let the controller exit on its own so the debug
  archive contains TRT's all-thread stack dumps. Keep the normal default at
  `-1`; stack dumping changes logging overhead and is diagnostic only.
- Verify with `python -m pytest utils/bench_offline -v`,
  `python -m compileall utils/bench_offline`, `bash -n` and `shellcheck` on
  all three launchers, and `actionlint .github/workflows/e2e-tests.yml`.
- A GPU benchmark is not valid until its Actions artifact proves the schedule
  validation, timing filter, exact rank set, and result values.
