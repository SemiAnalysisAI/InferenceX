# Offline TRT Agent Notes

Read `README.md` and `/TRT_BENCH_NOTES.md` before changing or debugging this
benchmark.

- This branch is disposable and must remain isolated from the normal serving
  sweep. Do not add entries to `nvidia-master.yaml` or `perf-changelog.yaml`.
- Preserve the fixed workload: 8192 prompt tokens, 625 generated tokens,
  MTP3, DEP8, exact seeds, one full-shape warmup, and three final passes.
- Preserve the token-based headline metric. Huawei conversion uses observed
  output tokens per TRT decode iteration, not raw draft acceptance.
- The pinned TRT PyTorch MTP path may return zero proposed/accepted draft
  counters. Record raw acceptance as unavailable and use
  `(tokens_per_step - 1) / 3` for effective MTP3 acceptance.
- Preserve output-sequence digests; they are the check that fresh-engine
  tuning candidates sampled the same token streams.
- Keep tuning limited to six attempts and preserve scheduler-first order.
- Every tuning and final measurement must use a fresh `LLM` instance.
- Do not silently pad prompts, reduce MTP depth, enable LM-head TP, change the
  MoE backend, or switch to HTTP serving to make a run pass.
- A high-concurrency capacity failure is a valid result row. Unexpected
  low-concurrency failures remain job failures.
- On runtime errors, retain the result JSON, worker JSON/log, candidate config,
  corpus manifest, perfect-router marker, and GPU metrics.
- Verify changes with `python -m pytest utils/bench_offline -v`,
  `python -m compileall utils/bench_offline`, `bash -n` on both launcher
  scripts, and YAML parsing of `.github/workflows/e2e-tests.yml`.
- The GPU benchmark has not run until an Actions artifact proves it. Local
  unit tests validate accounting and orchestration only.
