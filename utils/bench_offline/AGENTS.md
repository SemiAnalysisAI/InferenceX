# Offline TRT Agent Notes

Read `README.md` and `/TRT_BENCH_NOTES.md` before changing or debugging this
benchmark.

- This branch is disposable and must remain isolated from the normal serving
  sweep. Do not add entries to `nvidia-master.yaml` or `perf-changelog.yaml`.
- Preserve the Huawei-comparable workload: 8192 prompt tokens, 625 generated
  tokens, MTP3, temperature 1, one full-shape warmup, and exactly one measured
  pass. A single-candidate experiment uses one fresh engine; legacy serial
  tuning also uses one pass per candidate plus a fresh one-pass final engine.
  TP4/DEP4 rows must record their four active GPUs and topology explicitly.
- The pinned TRT PyTorch sampler ignores request-level seeds and advances one
  engine-global seed-42 generator. Do not claim per-request determinism.
- Preserve the token-based headline metric. Huawei conversion uses observed
  output tokens per TRT decode iteration, not raw draft acceptance.
- The pinned TRT PyTorch MTP path may return zero proposed/accepted draft
  counters. Record raw acceptance as unavailable and use
  `(tokens_per_step - 1) / 3` for effective MTP3 acceptance.
- Preserve output-sequence digests; they expose output-dependent MTP
  variation between tuning candidates and measured passes.
- Keep legacy tuning limited to six attempts and preserve scheduler-first
  order. Prefer the single-candidate experiment mode for optimization sweeps.
- Every tuning and final measurement must use a fresh `LLM` instance.
- Preserve the image-keyed CuTe DSL cache-path propagation while debugging,
  but do not claim it reduces startup: cache-prime run `27475868179` created
  zero files there. Verify every active MPI marker records the same path and
  that the exact active rank set is `0..N-1`.
- For attention DP, TRT defines `max_batch_size` as per local rank. Keep the
  legacy global mode only as an A/B control. Local-rank experiments must set
  engine and CUDA graph capacity to
  `ceil(global_concurrency / attention_dp_ranks)` and record the resolved
  values in the result.
- The pinned TRT source preserves checkpoint-derived DeepSeek-V4 sparse
  settings when a partial sparse-attention override is supplied. Validate
  `use_cute_dsl_topk`, `use_cute_dsl_paged_mqa_logits`,
  `enable_heuristic_topk`, and explicit `indexer_k_dtype` after engine start.
  CuTE DSL requests must fail unless `IS_CUTLASS_DSL_AVAILABLE` is true.
  Run `27477088665` proved FP4 indexer K is incompatible with the paged-MQA
  DSL kernel: its query is `torch.int8`, while the kernel requires
  `float8_e4m3fn`. Only exercise that kernel with explicit
  `indexer_k_dtype=fp8`.
- `max_seq_len=8832` is the block-aligned tight capacity for the fixed
  8192-input/625-output/MTP3 shape. Keep `9216` as the control.
- `print_iter_log=false` only disables native TRT iteration output. Preserve
  all launcher/controller/worker phase and heartbeat logs.
- Do not silently pad prompts, reduce MTP depth, change the MoE backend, or
  switch to HTTP serving to make a run pass. LM-head TP is an explicit
  candidate field and must be recorded in the result.
- The current Huawei gate uses exact global batches 16, 64, and 128 with
  MTP3. It compares B300 steps per active GPU with Huawei steps per chip and
  records active GPU count/topology. Never describe this as equal topology or
  total-system throughput; Huawei uses 16 chips while TP4/DEP4 use four B300s.
- Exact prompt construction may use the recorded context/suffix whitespace
  adjustment or context-tail trim. It must never insert pad or synthetic
  token IDs.
- A high-concurrency capacity failure is a valid result row. Unexpected
  low-concurrency failures remain job failures.
- Preserve the timestamped launcher/controller/worker progress logs. The
  controller heartbeat should remain low-frequency and must not stream or
  parse native TRT iteration logs into the Actions log.
- Profile candidates must use one increasing `profile_iterations` range.
  `50-51` captures one decode iteration in the full-shape warmup without
  adding a measured pass. Require one non-empty trace per active rank and
  keep traces in `offline_profiles_*.tar.gz`, outside the debug archive.
- Candidate-controlled `TRTLLM_*` and `TLLM_*` settings must be present in
  every MPI rank marker. Do not accept a labeled backend or communication
  result if rank propagation validation fails.
- `moe_autotune_dummy_distribution` is a source-backed candidate. The pinned
  default is `random`; use `balanced` only as a labeled comparison against
  the forced balanced router and validate it in every rank marker.
- The DeepSeek-V4 redundant-allreduce backport is TP4-only. Check pinned and
  patched source hashes before TRT import, run control and optimized rows
  from the same patched source, and validate that hash on every active rank.
- The checked-in B300 production recipe caps the fixed 8K/MTP3 TP4 shape at
  concurrency 32. Do not schedule TP4 above c32; use DEP4 at c64/c128.
  Run `27480420625` proved c128 TP4 spends about 8.3 minutes initializing,
  then OOMs late in warmup after about 35 minutes total.
- `ENABLE_CONFIGURABLE_MOE` is an explicit branch-only candidate. Propagate
  it through `TRTLLM_BENCH_ENABLE_CONFIGURABLE_MOE`, restore it in
  `trt_mpi_entry.py`, and validate both names in every active-rank marker.
- On runtime errors, retain the result JSON, worker JSON/log, candidate config,
  corpus manifest, perfect-router marker, and GPU metrics.
- Verify changes with `python -m pytest utils/bench_offline -v`,
  `python -m compileall utils/bench_offline`, `bash -n` on both launcher
  scripts, and YAML parsing of `.github/workflows/e2e-tests.yml`.
- The GPU benchmark has not run until an Actions artifact proves it. Local
  unit tests validate accounting and orchestration only.
