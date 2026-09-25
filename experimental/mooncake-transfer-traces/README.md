# Mooncake two-node H100 transfer tracing

**English** | [中文](README_zh.md)

An experimental snapshot of Mooncake transfer-engine instrumentation, Slurm benchmark launchers,
and the two-node H100 measurements from 24 September 2026. This is a standalone Mooncake
experiment, not an official InferenceX serving benchmark or a registered InferenceX runner/recipe.

## Results

Job **1695** completed all **90 cases** (15 message sizes × read/write × 3 repeats) in **9 min 32 s**.
At **64 MiB**, average payload bandwidth was **41.751 GB/s read** and **40.152 GB/s write**.
The traces are **incomplete**: **2,483,420 event records were dropped (4.50%)** by the recorder queue.
Do not use them as a lossless replay or an exact outstanding-request timeline.

![Bandwidth comparison](results/tebench-1695.6UKR6R/bandwidth-comparison.png)

| Message size | Read GB/s | Write GB/s | Read vs untraced | Write vs untraced |
| --- | ---: | ---: | ---: | ---: |
| 4 KiB | 0.465 | 0.468 | -7.04% | -6.77% |
| 8 KiB | 0.930 | 0.940 | -5.24% | -5.07% |
| 16 KiB | 1.778 | 1.826 | -4.87% | -5.17% |
| 32 KiB | 3.387 | 3.451 | -5.40% | -5.63% |
| 64 KiB | 6.195 | 6.315 | -6.28% | -6.15% |
| 128 KiB | 10.190 | 10.183 | -7.73% | -8.29% |
| 256 KiB | 15.579 | 15.362 | -10.99% | -10.03% |
| 512 KiB | 20.957 | 21.272 | -8.74% | -7.57% |
| 1 MiB | 28.261 | 28.181 | -6.84% | -5.32% |
| 2 MiB | 29.451 | 29.845 | -3.52% | -3.63% |
| 4 MiB | 34.996 | 34.617 | -2.25% | -2.29% |
| 8 MiB | 38.708 | 37.683 | -1.24% | -1.43% |
| 16 MiB | 40.931 | 39.432 | -0.70% | -0.88% |
| 32 MiB | 41.468 | 40.072 | -0.20% | -0.75% |
| 64 MiB | 41.751 | 40.152 | -0.17% | -0.17% |

Numbers are means of three repeats; plot whiskers show repeat minima/maxima, not confidence intervals.
The reference is untraced job **1612**, using the same message sizes and concurrency. This is an
observational comparison between separate runs and binaries, not an isolated measurement of tracing
cost. Small messages were roughly **4–11% slower**; at 64 MiB the decrease was about **0.17%**.
At 4 KiB, mean wall time was **8.81 µs read / 8.76 µs write**, versus **8.19 / 8.17 µs** previously.

The workload uses registered **GPU memory**, one H100 80 GB per node, **one thread**, **batch size 1**,
and a 256 MiB buffer. Each case has 1 s warmup and 5 s measurement. Recording includes both phases.

- Initiator: `slurm-h100-206-025` (`10.0.1.136`).
- Target: `slurm-h100-206-035` (`10.0.3.49:15309`).
- Read: payload moves **035 → 025**; write: payload moves **025 → 035**.
- Backend: TENT; transport: one-sided RDMA; discovery: p2p metadata.
- GB/s denotes decimal payload bytes/s; KiB and MiB denote binary message sizes.

Large messages amortize per-request overhead and approach **40 GB/s**. Individual 64 KiB requests
achieved only **6.2–6.3 GB/s** at this concurrency. These are transport-level results, excluding
KV lookup, placement, packing, session affinity and prefill/decode compute. They do not establish
multi-NIC aggregate capacity, simultaneous bidirectional bandwidth, or actual PD serving performance.

## Trace integrity and limitations

![Trace loss](results/tebench-1695.6UKR6R/trace-loss.png)

The full scan of six initiator files found:

| Quantity | Count |
| --- | ---: |
| Retained event records | 52,712,290 |
| Dropped event records | 2,483,420 |
| Matched submit/completion pairs | 26,355,861 |
| Submit-only request IDs | 285 |
| Complete-only request IDs | 283 |
| Request IDs absent entirely | 1,241,426 |
| Attempted logical requests, including warmup | 27,597,855 |

All retained completions report successful RDMA completion; no duplicate events were found.
Every initiator footer reconciles with its actual record count. The logs report `write_failed=0`:
loss came from the bounded recording queue. This does not prove the outcome of missing events.
The passive target contains only a start record and is killed during normal launcher cleanup.

The recorder preserves original descriptors across merging, records completion once across retries,
and tags internal staging requests separately. It is disabled by default and works with aggregate
metrics compiled out. Enable it with `TENT_TRACE_DIR`; the Slurm launcher exposes `TRACE=1`.

Timestamps describe admission and **observed** completion, including polling delay. They are not NIC
hardware timestamps, packets, CPU instructions, or target-side receive calls. IDs are local to a
trace file, not application/session IDs. Loss is not known to be random, so retained latency and
rate distributions may be biased. Exact replay needs zero-drop recording plus driver-level IDs and
phase markers. A larger buffer absorbs bursts but cannot fix sustained writer overload.

## Package contents

| Path | Contents |
| --- | --- |
| [patches/0001-tent-transfer-event-recorder.patch](patches/0001-tent-transfer-event-recorder.patch) | Recorder implementation, engine/peer lookup integration, nine tests, CMake registration, schema documentation |
| [patches/0002-tebench-slurm.patch](patches/0002-tebench-slurm.patch) | Two-node H100 launcher, optional tracing, compute-node Pyxis build, usage notes |
| [patches/0003-store-benchmark-slurm.patch](patches/0003-store-benchmark-slurm.patch) | Earlier Store KV/microbenchmark/local-storage Slurm adaptations; these are not the source of the reported RDMA measurements |
| [patches/series](patches/series) | Patch order |
| [results/tebench-1695.6UKR6R](results/tebench-1695.6UKR6R) | Traced summaries, logs, topology, plots, integrity analysis and 1,440 sampled request spans across 90 windows |
| [results/tebench-1612.XD95we](results/tebench-1612.XD95we) | Untraced baseline summaries and topology |
| [scripts/analyze_traces.py](scripts/analyze_traces.py) | Streaming full-file trace audit; explicit result/trace directories and worker count |
| [scripts/extract_trace_windows.py](scripts/extract_trace_windows.py) | Extract 16 matched request spans near each phase midpoint; assumes these sequential, single-thread traces |
| [validation/ctest.log](validation/ctest.log) | Original 64-test validation output |
| [manifest.json](manifest.json) | Base revision, patched-file hashes, provenance and external raw trace inventory |
| [SHA256SUMS](SHA256SUMS) | Checksums of packaged files, excluding this checksum file |
| [LICENSE-Mooncake](LICENSE-Mooncake) | Upstream Apache-2.0 license |

The **16.39 GiB raw event files**, built binaries and container image are not included. On the
original cluster, raw traces remain at:

```text
/mnt/home/kimbo/networkx/Mooncake/benchmark-results/tebench-1695.6UKR6R/events
```

The package is usable for reading results without that directory. Re-running the full audit needs
the external traces. Historical command/log paths are preserved as provenance, not portable defaults.
The interactive conversation viewer is not bundled; its sampled data is in `trace-windows.json`.

## Apply and reproduce

The patches target Mooncake commit **`fe0d23e332f41c7d54b5c217194dce5f8e751b21`** and capture the
previously uncommitted changes. Use a fresh checkout; do not apply them again to the already-modified
original workspace. Replace the first two paths below:

```bash
export PACKAGE=/path/to/InferenceX/experimental/mooncake-transfer-traces
export MOONCAKE_ROOT=/path/to/Mooncake-repro
git clone https://github.com/kvcache-ai/Mooncake.git "$MOONCAKE_ROOT"
git -C "$MOONCAKE_ROOT" checkout --detach fe0d23e332f41c7d54b5c217194dce5f8e751b21
git -C "$MOONCAKE_ROOT" apply --check "$PACKAGE"/patches/*.patch
git -C "$MOONCAKE_ROOT" apply "$PACKAGE"/patches/*.patch
```

The archived Slurm scripts run inside Mooncake and retain its original configuration conventions.
They are not new InferenceX runner entrypoints. Patch 3 is optional for the transfer experiment;
patches 1 and 2 suffice for tebench. After application, the detailed recorder reference is
`mooncake-transfer-engine/tent/TRANSFER_EVENTS.md` in that checkout.

On the original cluster, reproduce the full traced sweep with:

```bash
export BUILD=1 TRACE=1 BUILD_ONLY=0
export MIN_BYTES=4096 MAX_BYTES=67108864 BUFFER_BYTES=268435456
export BATCH_SIZE=1 THREADS=1 DURATION=5 REPEATS=3
export SEG_TYPE=VRAM OPS="read write" TENT_TRACE_BUFFER_RECORDS=65536
sbatch --partition=h100 --account=cw-sup --nodes=2 \
  --nodelist=slurm-h100-206-025,slurm-h100-206-035 --gpus-per-node=1 \
  "$MOONCAKE_ROOT/mooncake-transfer-engine/benchmark/slurm/tebench.sbatch"
```

The launcher builds in a lightweight writable `ubuntu:22.04` Pyxis container on a compute node,
using host CUDA at `/usr/local/cuda-13.0`; the login node does not need CMake. It installs build
packages in the container, caches `build/tebench-pyxis/runtime.sqsh`, and runs both peers in that
image. Prerequisites are Slurm with the named partition/account/nodes, Pyxis/Enroot, shared writable
storage, CUDA, RDMA devices and package-download access. Adapt these paths/resources on another
cluster. The image tag and apt packages are not digest/version pinned, so rebuilds are not
bit-for-bit reproducible. `BUILD=0` is valid only after a successful build with the recorder.

Results appear under `$MOONCAKE_ROOT/benchmark-results/tebench-JOBID.XXXXXX/`; traces are in `events/`.
The script's outer log is `tebench-JOBID.out` in the submission directory unless `--output` overrides
it. The queue size above reproduces the observed overflow; it does **not** guarantee a complete trace.
For a controlled overhead comparison, run `TRACE=0` and `TRACE=1` back-to-back with the same binary
and settings. Avoid drawing lossless-replay conclusions until all initiator footers report zero drops.

## Re-analyze and validation

The streaming audit requires Python 3.10+ and `orjson`; run it on a compute node for the full dataset.
Change `EVENTS_DIR` if the raw files have moved. These commands replace the derived JSON files in
`RUN_DIR`, so use a copied run directory if you want to retain the packaged versions unchanged.

```bash
export RUN_DIR="$PACKAGE/results/tebench-1695.6UKR6R"
export EVENTS_DIR=/mnt/home/kimbo/networkx/Mooncake/benchmark-results/tebench-1695.6UKR6R/events
python3 -m pip install -r "$PACKAGE/scripts/requirements.txt"
python3 "$PACKAGE/scripts/analyze_traces.py" \
  --run-dir "$RUN_DIR" --events-dir "$EVENTS_DIR" --workers 3
python3 "$PACKAGE/scripts/extract_trace_windows.py" \
  --run-dir "$RUN_DIR" --events-dir "$EVENTS_DIR"
```

The full audit joins file-local request IDs and checks footer counts. The window extractor uses
monotonic event order for seeking; it is not a general extractor for arbitrarily interleaved
multi-thread traces. The scripts do not regenerate the Markdown or plots.

Recorded runtime evidence:

- Job **1694**: **64 tests passed**: 9 recorder, 8 merge, 30 failover, 14 queue-dispatch, 3 causal-chain tests; metrics compiled out.
- Job **1693**: two-node H100 smoke test, **16,442 matched pairs**, zero drops.
- Job **1695**: all 90 full-sweep cases completed, with **4.50% event loss**.
- Job **1696**: full scan of all six initiator traces completed.

Packaging validation applies all patches to the pinned base files, compares the resulting file
hashes with the tested working tree, and runs the packaged analysis tools on controlled trace
fixtures. This packaging step does not claim another GPU performance run.
