# Offline TRT Agent Notes

Read `README.md` and `/TRT_BENCH_NOTES.md` before changing or debugging this
benchmark.

- This branch is disposable and must remain isolated from the normal serving
  sweep. Do not add entries to `nvidia-master.yaml` or `perf-changelog.yaml`.
- Preserve the Huawei-comparable workload: 8192 prompt tokens, 625 generated
  tokens, MTP3, DEP8, temperature 1, one full-shape warmup, and one measured
  pass. TP4/DEP4 tests must be separately labeled and must not receive Huawei
  ratios. A single-candidate experiment uses one fresh engine; legacy serial
  tuning also uses one pass per candidate plus a fresh one-pass final engine.
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
  settings when `use_cute_dsl_paged_mqa_logits` is explicitly enabled.
  Treat it as an isolated kernel experiment, validate the resolved flag, and
  fail unless `IS_CUTLASS_DSL_AVAILABLE` is true. Run `27477088665` proved
  the current FP4 checkpoint is incompatible with this kernel: its query is
  `torch.int8`, while `cute_dsl_fp8_paged_mqa_logits` requires
  `float8_e4m3fn`. Do not rerun that path unless the checkpoint or TRT dtype
  handling changes.
- `max_seq_len=8832` is the block-aligned tight capacity for the fixed
  8192-input/625-output/MTP3 shape. Keep `9216` as the control.
- `print_iter_log=false` only disables native TRT iteration output. Preserve
  all launcher/controller/worker phase and heartbeat logs.
- Do not silently pad prompts, reduce MTP depth, change the MoE backend, or
  switch to HTTP serving to make a run pass. LM-head TP is an explicit
  candidate field and must be recorded in the result.
- Huawei comparison claims require DEP8, MTP3, and matching batch per chip.
  Non-matching parallelism or MTP-depth experiments may be useful
  optimizations, but their Huawei ratios must remain unavailable.
- Exact prompt construction may use the recorded context/suffix whitespace
  adjustment or context-tail trim. It must never insert pad or synthetic
  token IDs.
- A high-concurrency capacity failure is a valid result row. Unexpected
  low-concurrency failures remain job failures.
- Preserve the timestamped launcher/controller/worker progress logs. The
  controller heartbeat should remain low-frequency and must not stream or
  parse native TRT iteration logs into the Actions log.
- On runtime errors, retain the result JSON, worker JSON/log, candidate config,
  corpus manifest, perfect-router marker, and GPU metrics.
- Verify changes with `python -m pytest utils/bench_offline -v`,
  `python -m compileall utils/bench_offline`, `bash -n` on both launcher
  scripts, and YAML parsing of `.github/workflows/e2e-tests.yml`.
- The GPU benchmark has not run until an Actions artifact proves it. Local
  unit tests validate accounting and orchestration only.
