# Hugepage capacity probe

This host-only diagnostic checks an exact reservation using the existing MI355X `hugepage_sanity.sh` hook. It does not start a model, register RDMA memory or produce benchmark results.

Use an otherwise idle host or a dedicated single-node Slurm allocation. The probe refuses existing hugepage reservations and GPU processes, requires at least 512 GiB ordinary-memory headroom, first reserves 512 pages, then requests the caller's target. It verifies the kernel's total and free page counts and restores the initial zero-page reservation on exit. Do not run concurrently with other host-memory experiments.

For a 2399 GB pool plus 2% reservation slack, pass `--target-pages 1166811 --min-headroom-gib 512`, the absolute path to the upstream hook through `--hook`, and either `--expected-host HOST` or `--expected-job JOB_ID`. Run with unbuffered Python and retain stdout/stderr as the evidence log.

A successful reservation establishes a lower bound on supported capacity at that moment, not an absolute limit, full-pool RDMA registration, or model runtime stability. Administrative policy, ordinary-memory headroom and fragmentation are separate constraints.
