<div align="center">

**English** | [中文](./TPU_BENCHMARK_METHODOLOGY_zh.md)

</div>

# TPU operator benchmark methodology

## Scope and backend policy

OperatorX should exercise the highest-level production API that implements an
operation. A backend should not replace an available library operation with a
hand-written lower-level kernel merely to improve a benchmark number. For the plain
Linear/GEMM pilot, the JAX backend calls compiled `jnp.dot`.

The pilot uses two BF16 shapes:

| Name | M | N | K | Regime |
|---|---:|---:|---:|---|
| `tpu-gemm-skinny` | 16 | 8192 | 8192 | Decode-like, memory/dispatch exposed |
| `tpu-gemm-square` | 8192 | 8192 | 8192 | Compute-bound |

They are encoded in `testlists/tpu_gemm_pilot.json`.

## Timing contracts

Do not combine unlike measurements with `min()`. Report these contracts separately:

1. **`device_execution_us` (canonical calibration):** collect an XProf
   `TRACE_ONLY_XLA` profile after compilation and warmup. Use a unique
   `jax.profiler.StepTraceAnnotation` and measure the TPU-side XLA module's critical
   path. A composite operation can lower to overlapping HLO events, so summing event
   durations is not generally correct.
2. **`queued_throughput_us` (fast sweep proxy):** enqueue a batch of identical
   precompiled operations, retain every output, block on the whole output collection,
   and divide total completion time by the batch size. Report several batches and
   their p10/p50/p90. Also report enqueue time per operation so a host-feed bottleneck
   is visible. Promote this proxy to the standard `latency_us` only after calibration
   against XProf for the relevant hardware/backend/op regime.
3. **`dispatch_to_ready_us` (application-visible latency):** time one precompiled
   dispatch followed by `block_until_ready()`. This includes Python/PJRT dispatch and
   synchronization and must not be compared directly with a device-only cost model.

Compile and warm up outside every timed region. Inputs must already be device-resident.
Record the sample count, JAX/jaxlib/libtpu versions, device topology, and residency
mode.

## Residency

Preparing a new JAX array before every sample does not prove cold on-chip state.
Report one of:

- `steady_state`: stable HBM buffers reused across samples;
- `cold_hbm`: a documented rotating-buffer working set larger than on-chip VMEM,
  preferably verified with profiler counters;
- `residency_unknown`: use when HBM residency cannot be established.

The pilot script uses `steady_state`. Cold-HBM validation is follow-up work.

## Ironwood accounting

An Ironwood physical chip contains two independently exposed chiplets. JAX shows one
device per chiplet. Google's published 2307 BF16 TFLOP/s and 7380 GB/s HBM bandwidth
are per physical chip; a single-device GEMM therefore uses 1153.5 BF16 TFLOP/s as its
peak normalization. Each chiplet has its own 96 GiB memory space. Cross-check the
bandwidth normalization before using `% of bandwidth` because the published bandwidth
is chip-level.

Do not compare a one-chiplet JAX result directly with a SimulatorX physical-chip
device model. Either model one chiplet in SimulatorX or shard the hardware GEMM across
both chiplets.

## Ironwood pilot evidence

JAX 0.11.0 on a `tpu7x-8` host produced:

| Shape | Current runner | XProf device | Corrected queued proxy | Proxy error |
|---|---:|---:|---:|---:|
| `16×8192×8192` | 51.2 µs | 44.35 µs | 46.67 µs | +5.2% |
| `8192×8192×8192` | 1382.2 µs | 1244.75 µs | 1246.38 µs | +0.13% |

For the skinny shape, a synchronized steady-state call was about 155 µs, roughly
250% above XProf device duration. For the square shape, XProf reports about
883 TFLOP/s, or 76.6% of the single-chiplet BF16 peak.

These measurements justify the queued proxy for a fast pilot, but they do not yet
justify replacing the canonical device measurement for every TPU operation.

## Validation gates

The pilot can produce a self-contained validation bundle. A shape is accepted only
when all of these checks pass:

1. **Semantics:** sample nine output cells and compare them with an independent NumPy
   FP32 dot over the corresponding BF16 input rows and columns.
2. **Placement:** assert A, B, and the output are all on the selected single JAX
   device, and record the compiled input/output shardings.
3. **Logical program:** save StableHLO and require `stablehlo.dot_general` with the
   requested input and output dimensions.
4. **Compiled program:** save the exact compiled executable's optimized HLO and
   require the `jit_dot` module. Record the TPU emitter, window configuration, scoped
   memory allocation, and cross-program prefetch decision.
5. **Device timing:** capture one profile per shape with `TRACE_ONLY_XLA`, one chip
   profiled, host tracing limited to the named annotation, and Python tracing off.
   Parse the module-level `jit_dot(<fingerprint>)` events and require exactly the
   requested sample count.

The parser treats the module event as the executable critical path. It records child
HLO events for diagnostics but never adds their durations because they can overlap.
XProf's `raw_bytes_accessed` is a compiler/profiler access count; matching logical
tensor bytes does not prove that all accesses reached HBM.

On the validation host, both shapes passed all gates on
`TPU_0(process=0,(0,0,0,0))`, which is `TPU7x` chiplet 0:

| Shape | Sampled max relative error | XProf p50 | Queued p50 | Proxy error |
|---|---:|---:|---:|---:|
| `16×8192×8192` | 0.289% | 43.34 µs | 46.99 µs | +8.42% |
| `8192×8192×8192` | 0.279% | 1243.34 µs | 1246.82 µs | +0.28% |

StableHLO preserves the requested BF16 dot and contracting dimensions `[1] × [0]`.
The optimized TPU HLO represents both GEMMs as convolution fusions using
`EmitAllBatchInSublanes`, but chooses different window configurations. The skinny
shape uses cross-program prefetch while the square shape does not. XProf reports the
expected logical work:

| Shape | Model FLOPs | Raw bytes accessed |
|---|---:|---:|
| `16×8192×8192` | 2,147,483,648 | 134,742,016 |
| `8192×8192×8192` | 1,099,511,627,776 | 402,653,184 |

## Reproduce the pilot

From the OperatorX directory in the InferenceX worktree:

```bash
cd ~/InferenceX/.worktrees/tpu-exploration/experimental/operatorx

PYTHONPATH=.. \
~/SimulatorX/.venv/bin/python scripts/tpu_gemm_pilot.py \
  --json-out /tmp/operatorx-tpu-gemm-pilot.json
```

Inspect the machine-readable rows:

```bash
jq '.rows[] | {
  name,
  shape,
  dispatch_to_ready_us,
  queued_throughput_us,
  enqueue_us_per_op,
  achieved_tflops,
  pct_of_chiplet_peak
}' /tmp/operatorx-tpu-gemm-pilot.json
```

Run the complete correctness, placement, HLO, and XProf validation:

```bash
cd ~/InferenceX/.worktrees/tpu-exploration/experimental/operatorx

VALIDATION_DIR="$(mktemp -d /tmp/operatorx-tpu-validation.XXXXXX)"

PYTHONPATH=.. \
python scripts/tpu_gemm_pilot.py \
  --artifacts-dir "$VALIDATION_DIR" \
  --profile \
  --profile-iters 5 \
  --json-out "$VALIDATION_DIR/results.json"

echo "$VALIDATION_DIR"
```

Inspect the validation summary:

```bash
jq '.rows[] | {
  name,
  device,
  correctness,
  placement,
  compiler_algorithm: .hlo.compiler_algorithm,
  device_execution_us,
  queued_proxy_error_pct,
  device_achieved_tflops,
  device_pct_of_chiplet_peak
}' "$VALIDATION_DIR/results.json"
```

Reparse either saved Perfetto trace independently:

```bash
PYTHONPATH=.. \
python -m operatorx.runners.tpu.xprof \
  "$VALIDATION_DIR/profiles/tpu-gemm-skinny" \
  --module-name jit_dot \
  --expected-samples 5 \
  --annotation-name operatorx_tpu-gemm-skinny
```

To run the same testlist through the current standard OperatorX entrypoint:

```bash
cd ~/InferenceX/.worktrees/tpu-exploration/experimental/operatorx

PYTHONPATH=.. \
WORLD_SIZE=1 \
OPERATORX_CLUSTER=v7x_4x \
OPERATORX_BACKENDS=jax \
~/SimulatorX/.venv/bin/python -m operatorx \
  --platform tpu \
  --testlists tpu_gemm_pilot
```

The standard entrypoint writes a run JSON under `results/tpu/v7x_4x/`. Its current
`latency_us` still uses the legacy `min(single_shot, pipelined)` methodology; use the
pilot script's separated metrics for methodology decisions until the runner refactor
lands.

Official references:

- [JAX benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html)
- [JAX asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html)
- [Profiling JAX computations with XProf](https://openxla.org/xprof/jax_profiling)
- [TPU7x architecture](https://docs.cloud.google.com/tpu/docs/tpu7x)
