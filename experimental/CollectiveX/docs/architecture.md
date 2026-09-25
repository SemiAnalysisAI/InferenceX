**English** | [中文](architecture_zh.md)

# CollectiveX implementation map

CollectiveX has two execution paths: distributed expert-parallel (EP) communication and
single-GPU vLLM block copies. The workflow and launchers orchestrate processes; the EP
adapters implement a common Python interface. Configuration, measurement semantics,
artifact names, and exit statuses form the contracts between those layers.

## From dispatch to a result

1. [collectivex-sweep.yml](../../../.github/workflows/collectivex-sweep.yml) accepts a manual
   dispatch. [sweep_matrix.py](../sweep_matrix.py) expands the workload and platform registries.
   A shard groups cases by GPU pool, backend, mode, node count, and precision; each case
   contains its complete token ladder. Unsupported requested cases remain recorded but
   are excluded from execution.
2. A [launcher](../launchers/) allocates the hardware and stages an isolated source tree.
   Slurm pools execute one `run_ep.py` process per GPU; the Taiwan Docker pools use
   `torchrun`. Cases execute sequentially within the shard.
3. [run_ep.py](../bench/run_ep.py) initializes the GPU, lazily imports the adapter selected
   by `BACKENDS`, forms the process group, constructs the adapter, and calls
   [ep_harness.run_sweep](../bench/ep_harness.py).
4. The harness prepares inputs, checks correctness, measures isolated components and
   free-running pairs, then checks correctness again. [ep_results.py](../bench/ep_results.py)
   reduces the samples, writes the rank-zero case-attempt JSON, and agrees the exit status.
5. The launcher collects the JSON files. The workflow renders latency and bandwidth
   summaries and uploads `cxshard-*` artifacts. Cleanup releases the allocation, including
   after launcher interruption. Failed cases still fail the shard; result upload also runs
   for partial or failed executions.

## Python module responsibilities

| Module | Responsibility |
| --- | --- |
| [ep_case.py](../bench/ep_case.py) | Case IDs, CLI inputs, token ladders, runtime version formatting; no vendor imports |
| [ep_backend.py](../bench/ep_backend.py) | Abstract transport contract, `RankInputs`, `WorkloadSpec`, deterministic inputs, FP8 invariants |
| [ep_timing.py](../bench/ep_timing.py) | `EPTiming`: warmup, fresh-pair rules, isolated windows, and sibling timing chains |
| [ep_measurement.py](../bench/ep_measurement.py) | CUDA-event timing, rank reductions, percentiles, and `PointSamples` |
| [ep_oracle.py](../bench/ep_oracle.py) | Independent reference arithmetic, layout-specific receive checks, shared cleanup and combine verification |
| [ep_results.py](../bench/ep_results.py) | Byte accounting, artifact schema, atomic writes, and result logging |
| [ep_harness.py](../bench/ep_harness.py) | The ordered correctness and measurement passes for one case |
| [ep_legacy.py](../bench/ep_legacy.py) | Operations shared by DeepEP and UCCL's compatible legacy Buffer APIs |
| `ep_deepep_v2.py`, `ep_uccl.py`, `ep_mori.py`, `ep_nccl.py`, `ep_flashinfer.py` | Vendor construction, transport, receive views, and teardown differences |
| [routing.py](../bench/routing.py) | Deterministic routing, activations, source identity, and locality statistics |

`EPBackend` inherits the timing implementation from `EPTiming`. DeepEP and UCCL additionally
inherit `LegacyBufferOperations`; their normal transports and vendor quantizers remain in
their adapters. Oracle-transformed inputs use the adapter's ordinary combine operation
where the API is identical. Backend imports remain lazy and occur after GPU initialization.

## Runtime module responsibilities

[runtime/common.sh](../runtime/common.sh) is the supported sourcing entry point. It owns
logging and operator configuration, then loads these modules into the same shell:

| Module | Responsibility |
| --- | --- |
| [network.sh](../runtime/network.sh) | Fabric selectors, link-layer rules, and network validation |
| [slurm.sh](../runtime/slurm.sh) | Allocations, rendezvous, rank identity, health checks, and allocation cleanup |
| [images.sh](../runtime/images.sh) | Image identity, import locks, cache reuse, and import retries |
| [sources.sh](../runtime/sources.sh) | Backend source/version pins, exact-commit staging, and cache mounts |
| [staging.sh](../runtime/staging.sh) | Compute-visible source isolation, result collection, and stage cleanup |
| [execution.sh](../runtime/execution.sh) | Case execution, backend preparation, and launcher cleanup traps |

[prepare_backend.sh](../runtime/prepare_backend.sh) validates the container network and
writes the rank environment. It loads [build_common.sh](../runtime/build_common.sh) for
toolchain discovery and guarded cache installation, plus the individual
[backend build modules](../runtime/backends/). The launcher stages pinned source trees before
allocation; each node prepares its backend before any GPU rank starts. Keep cache identity,
readiness checks, and error handling together when changing an installer.

## Separate block-copy path

`backend=swap-blocks` selects [swap_matrix.py](../swap_matrix.py),
[launch_swap-blocks.sh](../launchers/launch_swap-blocks.sh), and
[run_swap_blocks.py](../bench/run_swap_blocks.py). This path uses functions and callbacks,
one GPU, its own wall-clock timing, and the `collectivex-swap-blocks-v1` schema. Its
measurement contract is documented in [swap-blocks.md](swap-blocks.md).

## Validation boundaries

The [tests](../tests/) exercise matrix generation, CLI argument forwarding, correctness
models, event placement, complete artifact publication, cache behavior, source staging,
and runtime failure handling. Include `experimental/operatorx/tests/` when changing runtime
helpers: OperatorX also consumes the platform registry and copies the runtime directory.
CPU tests do not establish GPU transport performance; real allocations remain necessary
for that evidence. The [methodology](methodology.md) defines the measurement semantics.
