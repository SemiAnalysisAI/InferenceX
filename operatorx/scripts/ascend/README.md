# Ascend GEMM reproduction

This standalone harness measures single-device native `torch_npu` GEMMs with queued events and numerical checks. It accepts OperatorX testlists and stores raw samples, validation results, runtime/source fingerprints and per-case telemetry in a resumable SQLite journal. It does not register an OperatorX platform, add a CI pool, or upload results.

The measurement path supports unquantized BF16/FP16/FP32 contracts and E4M3 GEMM with dynamic FP32 activation scales `[1,128]`, static FP32 weight scales `[128,128]`, and BF16 output. Activation quantization is inside the FP8 timer; static weight preparation is outside. Unsupported arguments, K tails, and cases above the explicit memory budget remain visible in the plan rather than being silently changed or padded. Parallel/distributed cases are not implemented.

## Environment

The hardware-tested stack is Ascend950PR_9579 (28 Cube cores, 128 GiB HBM), CANN 9.1.0, PyTorch 2.10.0 and torch_npu 2.10.0.post4 on Linux. Load the installed CANN environment and make the driver's libraries available before running. The host also needs `npu-smi`; FP8 preflight needs a C++17 compiler and `libdl`. No GPU libraries are imported for CPU-only planning.

Run commands from the repository root, in the Python environment containing `torch_npu`. Choose board/chip/logical-device IDs from the actual SMI/ACL mapping; the example below uses board 0, chip 0 and logical device 0. The controller checks that mapping and requires an idle, healthy board with clean fault/ECC/PCIe diagnostics. Identity probing occurs in a disposable process before Torch initializes ACL.

```bash
ASCEND_SCRIPTS="$PWD/operatorx/scripts/ascend"
ASCEND_RESULTS="/tmp/ascend-results"
mkdir -p "$ASCEND_RESULTS"
```

Use a fresh result directory for a changed plan, source, hardware identity or timing configuration. An existing directory cannot silently combine measurements from different fingerprints. The health parser is strict: an unrecognized driver table fails rather than guessing the device identity.

## Unquantized GEMM

The bundled examples include the large BF16 shape `M=49152, N=7168, K=6144`. A normal testlist such as `operatorx/testlists/gemm_deepseek_v4.json` can replace the example input; multiple files are accepted. Case identity includes every argument, including output precision and scale descriptors.

```bash
python "$ASCEND_SCRIPTS/prepare.py" \
  --input "$ASCEND_SCRIPTS/examples/bf16.json" --kind unquantized \
  --memory-budget-bytes 103079215104 --output "$ASCEND_RESULTS/bf16-plan.json"

python "$ASCEND_SCRIPTS/operatorx_batch_campaign.py" run \
  --plan "$ASCEND_RESULTS/bf16-plan.json" --run-dir "$ASCEND_RESULTS/bf16" \
  --phase smoke --device 0 --board-id 0 --chip-id 0 --expected-name 950PR \
  --warmups 5 --warmup-ms 25 --samples 10 --seed 20261023 \
  --timeout-seconds 300 --cpu-threads 4
```

Then repeat the controller command with `--phase all-ready`, the same plan and run directory. Already successful smoke cases are reused; every other selected case receives its own before/after numerical gates. Execution stops on an unexpected kernel, numerical, health or timeout failure. Inspect failures before resuming in a fresh process. The 300-second per-case watchdog preserves a running marker if execution hangs.

## FP8: inspect the installed tile first

The tested CANN kernel can select a local A tile larger than L0A. For `M=264, N=2304, K=7168`, the installed tile is `272×128`: two FP8 A buffers require 69,632 bytes, exceeding the 65,536-byte capacity. This is a **local tile** limit, not a limit on whole matrix size or a blanket `M <= 256` rule.

The preload hook reads the actual host tiling before GEMM dispatch. Preflight always rejects the GEMM with error `100000`, including tiles that fit; it produces no timing result. The decoder supports the validated 152-byte ABI and dispatch key 5124 for the named FP8 kernel. Other kernels/ABIs are skipped as unverified. The double-buffer bound is conservative and source-derived; this is not a general CANN tiling decoder.

```bash
c++ -std=c++17 -O2 -shared -fPIC \
  "$ASCEND_SCRIPTS/operatorx_fp8_l0_guard.cpp" -ldl \
  -o "$ASCEND_RESULTS/liboperatorx_fp8_l0_guard.so"

python "$ASCEND_SCRIPTS/prepare.py" \
  --input "$ASCEND_SCRIPTS/examples/fp8.json" --kind fp8 \
  --memory-budget-bytes 103079215104 --output "$ASCEND_RESULTS/fp8-plan.json"

python "$ASCEND_SCRIPTS/operatorx_batch_campaign.py" _preflight-child \
  "$ASCEND_RESULTS/health.json" \
  '{"board_id":0,"chip_id":0,"device":0,"expected_name":"950PR","expect_pci":null}'

TASK_QUEUE_ENABLE=0 LD_PRELOAD="$ASCEND_RESULTS/liboperatorx_fp8_l0_guard.so" \
python "$ASCEND_SCRIPTS/operatorx_fp8_l0_preflight.py" \
  --shape 240 2304 7168 --output "$ASCEND_RESULTS/guard-smoke.jsonl" \
  --guard-so "$ASCEND_RESULTS/liboperatorx_fp8_l0_guard.so" \
  --worker-dir "$ASCEND_SCRIPTS" --health-report "$ASCEND_RESULTS/health.json" \
  --device 0 --memory-budget-bytes 103079215104
```

First verify that this known-safe smoke records `status=inspected`, `observed=true`, `decision=within_capacity`, `mode=1`, `forward_allowed=false`, and 61,440 L0A bytes. Do not check a potentially unsafe shape until actual interception is established on the installed stack.

```bash
TASK_QUEUE_ENABLE=0 LD_PRELOAD="$ASCEND_RESULTS/liboperatorx_fp8_l0_guard.so" \
python "$ASCEND_SCRIPTS/operatorx_fp8_l0_preflight.py" \
  --plan "$ASCEND_RESULTS/fp8-plan.json" --output "$ASCEND_RESULTS/guard.jsonl" \
  --guard-so "$ASCEND_RESULTS/liboperatorx_fp8_l0_guard.so" \
  --worker-dir "$ASCEND_SCRIPTS" --health-report "$ASCEND_RESULTS/health.json" \
  --device 0 --memory-budget-bytes 103079215104

python "$ASCEND_SCRIPTS/fp8_plan.py" \
  --plan "$ASCEND_RESULTS/fp8-plan.json" --observations "$ASCEND_RESULTS/guard.jsonl" \
  --output "$ASCEND_RESULTS/fp8-approved.json"

python "$ASCEND_SCRIPTS/operatorx_batch_campaign.py" run \
  --plan "$ASCEND_RESULTS/fp8-approved.json" --run-dir "$ASCEND_RESULTS/fp8" \
  --worker-script "$ASCEND_SCRIPTS/operatorx_fp8_block_worker.py" \
  --phase smoke --device 0 --board-id 0 --chip-id 0 --expected-name 950PR \
  --warmups 5 --warmup-ms 25 --samples 10 --seed 20261023 \
  --timeout-seconds 300 --cpu-threads 4
```

Run `all-ready` afterward as above. `LD_PRELOAD` and `TASK_QUEUE_ENABLE=0` are scoped to the inspection commands and must not be set for timing. The controller requires complete observations for the selected FP8 plan and rechecks the device identity before execution. Over-capacity and unverified cases remain explicit skips, with no fallback implementation or accepted latency.

## Timing and validation contract

- Five warmups, followed by a 25 ms ramp; the first timing call in each process has a 1-second ramp floor.
- Ten preallocated event pairs, a 256 MiB write before each operation, all ten calls queued, then one synchronization. The reported latency is the sixth sorted sample.
- Four queued cache writes precede the timed batch as a launch shield. They are **not equivalent to CUDA `_sleep`**. Cache eviction is not proven. NPU graphs and GPU throttle-retry selection are not emulated.
- BF16 uses the serving-style contiguous `[N,K]` weight layout. FP8 times dynamic quantization and native GEMM together. FP8 uses one Torch CPU thread, while the unquantized worker uses the controller's configured count.
- Sampled CPU FP64 full-K dot products and a full output-finiteness scan run before and after timing. FP8 additionally checks every activation's quantization error and scale validity. This is not an exhaustive CPU output reference or a claim of CUDA-quantizer byte equivalence.
- Queued batch wall time includes shield/cache writes and final synchronization. It is recorded separately and is not represented as individual synchronized wall samples.
- Journals keep all attempts and raw event samples. The last attempt determines resume status; no fastest-of-retries selection is performed. Nothing is submitted automatically.

The measurement routines were used for the queued 950PR campaign; the large BF16 shape reproduced approximately 377 TFLOPS against a 378-TFLOPS nominal Cube reference. The entry-point packaging and plan approval checks are separate from the external campaign's frozen source hashes. Results from a per-sample synchronized loop are a different protocol and must not be mixed into these curves.

## Files and checks

`results.sqlite3` contains the runtime/source fingerprint and append-only attempts. `results.json` is a latest-attempt summary. Each attempt directory contains the requested case, raw worker output, stdout/stderr, process checks and sampled SMI telemetry. These standalone artifacts are not the normal `operatorx.core.Result` ingestion format.

```bash
python -m pytest operatorx/tests/test_ascend_plan.py \
  operatorx/tests/test_ascend_queued_timing.py operatorx/tests/test_ascend_l0_guard.py
```

The guard fixture is captured host tiling data, not a vendor device binary. Tests compile the guard on the CPU and exercise real decoder and launch-hook behavior against a mock runtime. Native preflight is still required to establish Linux symbol interception on each installed stack.
