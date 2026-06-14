# Offline TRT Agent Notes

Read `README.md` and `/TRT_BENCH_NOTES.md` before changing or debugging this
benchmark.

- This branch is disposable and must remain isolated from the normal serving
  sweep. Never edit `nvidia-master.yaml` or `perf-changelog.yaml`.
- The benchmark has one contract only: DeepSeek-V4 Pro, B300 DEP8, exact
  global batches `16`, `64`, and `128`, 8192 input tokens, MTP3, two warmup
  decode rounds, and 256 measured decode rounds.
- `global_batch_size` is authoritative. TRT `max_batch_size` and CUDA graph
  size are exactly `global_batch_size / 8`; `max_num_tokens` is exactly
  `local_batch_size * 8192` so every local prompt can prefill together.
- Leave KV capacity memory-derived with
  `kv_cache_config.free_gpu_memory_fraction=0.60`. An explicit KV token cap
  underprovisions the pinned DeepSeek-V4 multi-pool cache and staggered half
  of each local batch in run `27486168511`. The fixed
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
- Keep the rank-local packed-FP8 guard at 32768 rows. It selects TRT's Triton
  quantizer only for large GBS64/128 prefill projections; measured decode has
  at most 16 rows and stays on the original fused path.
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
  `GBS / decode_round_TPOT / 8`. MTP output yield and acceptance are separate.
- The 1025-token measured output cap guarantees at least 256 MTP3 decode
  iterations even if every round emits four tokens. Only the first 256 valid
  full-batch rounds are measured.
- Preserve perfect routing, exact 8192-token real prompts, temperature 1,
  engine-global seed 42, LM-head TP, heuristic sparse top-k, ConfigurableMoE,
  and rank environment validation.
- Do not enable `TLLM_METRICS_ALL_RANKS`; its per-iteration collective changes
  the timing path. Rank 0 stats are valid only after exact equal-length prompt
  routing and full-local-batch validation.
- The comparison is methodological, not identical hardware: this branch uses
  eight B300 GPUs and FP4, while Huawei publishes sixteen 950DT chips with
  hybrid MXFP8/MXFP4.
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
- For a readiness hang, dispatch with `worker-stack-period=120` and a short
  controller timeout, then let the controller exit on its own so the debug
  archive contains TRT's all-thread stack dumps. Keep the normal default at
  `-1`; stack dumping changes logging overhead and is diagnostic only.
- Verify with `python -m pytest utils/bench_offline -v`,
  `python -m compileall utils/bench_offline`, `bash -n` on both launchers, and
  YAML parsing of `.github/workflows/e2e-tests.yml`.
- A GPU benchmark is not valid until its Actions artifact proves the schedule
  validation, timing filter, exact rank set, and result values.
