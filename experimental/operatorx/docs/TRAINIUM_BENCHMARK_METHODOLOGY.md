<div align="center">

**English** | [中文](./TRAINIUM_BENCHMARK_METHODOLOGY_zh.md)

</div>

# Trainium3 NKI GEMM benchmark methodology

## Measurement contract

The canonical calibration metric is device execution time from NKI's native
standalone profiling path:

1. compile the `@nki.jit` BF16 GEMM for `trn3`, grid/LNC degree two;
2. load the NEFF with NKI's Spike runtime;
3. execute ten unprofiled warmups with stable device buffers;
4. capture 21 independent NTFF device profiles; and
5. parse each profile with `neuron-profile view --output-format summary-json`.

The retained `device_execution_us` distribution comes from profile `total_time`.
It excludes framework dispatch and compilation. NKI `benchmark` latency is not the
calibration metric because that path includes host/device transfer in addition to
NEFF execution. The method follows the native NKI profiling model documented by
[AWS `nki.profile`](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.profile.html),
using the installed NKI 0.3 standalone compiler and Spike runtime directly so the
existing OperatorX `@nki.jit` kernel can be profiled without PyTorch/XLA.

## Corpus and NKI padding

The testlist is exactly `testlists/tpu_gemm_sweep.json`, the 20-shape BF16 corpus
used by the TPU exploration. It covers decode/prefill M values, square GEMMs, and
8192↔28672 feed-forward projections.

The current OperatorX NKI kernel pads BF16 GEMMs to:

- M: multiple of 2048;
- N: multiple of 2048 for LNC=2; and
- K: multiple of 1024.

Ten distinct padded executables cover the 20 requests. Requests that resolve to the
same executable intentionally share its 21 native device samples and NEFF hash. Each
row records both requested and executed shapes, logical and executed FLOPs/bytes,
and the padding FLOPs ratio. SimulatorX projects the requested logical shape, so NKI
padding remains visible as implementation overhead.

The follow-on `testlists/trainium_gemm_tile_boundaries.json` corpus contains 20
requests around the 2048-row, 2048-column, and 1024-reduction padding boundaries.
It resolves to 11 executables and includes eight exact controls. Requests immediately
below, at, and above a boundary deliberately form shared-executable latency plateaus;
the exact controls separate native component behavior from padding effects.

The model-development follow-on `testlists/trainium_gemm_exact_holdout.json`
contains 24 exact executables over M/N in {2048, 4096, 6144, 8192} and K in
{1024, 2048, 3072, 4096}. Every K occurs six times. Sixteen rows are labeled
`train`; eight rows hold out four complete M/N pairs, with two K values per pair.
The frozen split prevents an output-wave tail from being evaluated on an M/N
combination it saw during fitting.

## Acceptance gates

An executable is accepted only when:

1. **Semantics:** deterministic BF16 inputs of one produce the expected output K at
   sampled cells, including the last padded row and column.
2. **Placement:** the NKI target is `trn3`; compiler metadata records LNC=2; the
   launch grid is two physical NeuronCore-v4 cores.
3. **Compiled program:** the NEFF and compiler-info SHA-256 values are retained;
   each profile contains matmul instructions.
4. **Work identity:** every profile's `hardware_flops` matches `2*M*N*K` for the
   executed padded shape within 0.01%, or differs by no more than the explicit
   `2*N*K` FLOPs of one M-row plane. The latter bound covers an observed derived-
   counter boundary omission; compiled shape and output correctness remain
   independently gated.
5. **Timing:** exactly 21 positive `total_time` samples are retained, with
   p10/p50/p90/min/max.
6. **Traffic evidence:** every profile contains positive HBM read and write counts.
7. **Residency:** inputs and outputs are allocated once and reused across warmup and
   profiles, so the result is labeled `steady_state`.

The native profile does not use actual input values when only shape descriptors are
passed through the legacy `nki.profile` decorator. This harness instead executes the
compiled NKI 0.3 NEFF with real NumPy BF16 buffers through Spike, allowing a separate
correctness run while preserving the same native NTFF timing mechanism.

## Compact evidence and cleanup

`scripts/trainium_gemm_sweep.py` runs the complete corpus in one invocation. It
temporarily stores NEFFs, NTFFs, MLIR/BIR, compiler output, and logs under a fresh
directory. After every trace has been parsed and the compact JSON has been written,
the full directory is deleted by default. `--keep-artifacts` is an explicit debug
escape hatch and is not used for the checked-in run.

The compact dataset retains:

- all 21 device times and selected device counters per distinct executable;
- requested and executed shapes;
- validation outcomes and residency label;
- NEFF/compiler-info hashes and NEFF size;
- Neuron device inventory and NKI/compiler/runtime/profiler versions; and
- the exact measurement and testlist contract.

No raw NTFF, NEFF, compiler dump, or run log is checked in after the official sweep.

## Reproduce

From `experimental/operatorx` on this Trainium3 partition:

```bash
export PATH=/opt/aws/neuron/bin:/opt/aws_neuronx_venv_pytorch_2_9/bin:$PATH
export PYTHONPATH="$PWD/.."

python scripts/trainium_gemm_sweep.py \
  --json-out data/trn3_lnc2_gemm_sweep_20260731.json \
  --samples 21 \
  --warmup 10
```

The command prints each distinct executable as it runs, writes the compact result,
and confirms deletion of its temporary profile root.

The tile-boundary follow-on uses the same contract:

```bash
python scripts/trainium_gemm_sweep.py \
  --testlist testlists/trainium_gemm_tile_boundaries.json \
  --json-out data/trn3_lnc2_gemm_tile_boundaries_20260731.json \
  --samples 21 \
  --warmup 10
```

The exact train/holdout grid also uses the same one-shot contract:

```bash
python scripts/trainium_gemm_sweep.py \
  --testlist testlists/trainium_gemm_exact_holdout.json \
  --json-out data/trn3_lnc2_gemm_exact_holdout_20260731.json \
  --samples 21 \
  --warmup 10
```
