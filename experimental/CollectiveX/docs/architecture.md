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
2. [ci.py](../ci.py) calls the Python [execution controller](../runtime/execution.py), which
   allocates the hardware and stages an isolated source tree.
   Slurm pools execute one `run_ep.py` process per GPU; the Taiwan Docker pools use
   `torchrun`. Cases execute sequentially within the shard.
3. [run_ep.py](../bench/run_ep.py) initializes the GPU, lazily imports the adapter selected
   by `BACKENDS`, forms the process group, constructs the adapter, and calls
   [ep_harness.run_sweep](../bench/ep_harness.py).
4. The harness prepares inputs, checks correctness, measures graph replay for supported modes (eager components and
   free-running pairs otherwise), then checks correctness again. [ep_results.py](../bench/ep_results.py)
   reduces the samples, writes the rank-zero case-attempt JSON, and agrees the exit status.
5. The launcher collects the JSON files. The workflow renders latency and bandwidth
   summaries and uploads `cxshard-*` artifacts. Cleanup releases the allocation, including
   after launcher interruption. Failed cases still fail the shard; result upload also runs
   for partial or failed executions.

## Python module responsibilities

| Module | Responsibility |
| --- | --- |
| [run_ep.py](../bench/run_ep.py) | CLI inputs, lazy backend dispatch, runtime initialization and version reporting |
| [ep_backend.py](../bench/ep_backend.py) | Abstract transport contract, `RankInputs`, `WorkloadSpec`, deterministic inputs, FP8 invariants |
| [ep_measurement.py](../bench/ep_measurement.py) | `EPTiming` warmup, graph/eager timing templates, CUDA events, rank reductions, percentiles, and `PointSamples` |
| [ep_oracle.py](../bench/ep_oracle.py) | Independent reference arithmetic, layout-specific receive checks, shared cleanup and combine verification |
| [ep_results.py](../bench/ep_results.py) | Case identity, byte accounting, artifact schema, atomic writes, and result logging |
| [ep_harness.py](../bench/ep_harness.py) | The ordered correctness and measurement passes for one case |
| [ep_legacy.py](../bench/ep_legacy.py) | Operations shared by DeepEP and UCCL's compatible legacy Buffer APIs |
| `ep_deepep_v2.py`, `ep_uccl.py`, `ep_mori.py`, `ep_nccl.py`, `ep_flashinfer.py` | Vendor construction, transport, receive views, and teardown differences |
| [routing.py](../bench/routing.py) | Deterministic routing, activations, source identity, and locality statistics |

`EPBackend` inherits the timing implementation from `EPTiming`. DeepEP and UCCL additionally
inherit `LegacyBufferOperations`; their normal transports and vendor quantizers remain in
their adapters. Oracle-transformed inputs use the adapter's ordinary combine operation
where the API is identical. Backend imports remain lazy and occur after GPU initialization.

## Runtime module responsibilities

[ci.py](../ci.py) is the workflow and manual entry point. `matrix`, `extract`, `execute`,
`finalize`, and `cleanup` keep Actions responsible for scheduling and artifact upload while
Python owns the execution lifecycle. Workflow steps invoke the host venv interpreter directly,
so its `PATH` and `VIRTUAL_ENV` do not replace the pinned image's Python in worker containers.

| Module | Responsibility |
| --- | --- |
| [config.py](../runtime/config.py) | Pool resource requests, settings precedence, container options, and case/block-copy arguments |
| [scheduler.py](../runtime/scheduler.py) | `simple-slurm` steps, allocation lifecycle, subprocesses, private logs, locks, and signals |
| [probe.py](../runtime/probe.py) | Hardware/network probes, fabric selectors, link-layer rules, and image digest lookup |
| [storage.py](../runtime/storage.py) | Image-cache identity/imports and isolated source staging/collection |
| [build.py](../runtime/build.py) | Exact source/dependency pins, preparation, guarded build caches, and backend environments |
| [node.py](../runtime/node.py) | Stdlib host utilities, per-node setup, and rank environment loading before `exec` |
| [execution.py](../runtime/execution.py) | Allocation retries, preparation, sequential cases, and recoverable cleanup |
| [docker.py](../runtime/docker.py) | Slurm-less pool execution, Docker build caches, and owned-container cleanup |

The host uses [simple-slurm](https://github.com/amq92/simple_slurm), pinned in the `collectivex`
extra. `salloc --no-shell` and targeted `squeue`/`scancel` calls preserve the existing allocation
lifecycle. The library executes `srun`; its shell-string API receives quoted arguments, with
bare flags and Pyxis options passed explicitly. No repository-owned Bash launcher remains.

Compute-host utilities are streamed as a stdlib zipapp before shared storage or a container is
available. Backend setup runs once per node and writes a private, allowlisted JSON environment.
Rank bootstrap applies it, derives rank identity from Slurm, and executes the unchanged benchmark.
Case arguments are ordinary Python lists: there are no NUL-delimited argument files or generated
shell environment scripts.

`execution.json` and `jobid` retain resource ownership for the independent workflow finalizer.
Cleanup stops the allocation before collecting or deleting staging; an unconfirmed stop retains
its recovery record. Signal exits preserve `128 + signal`. Partial results are collected after
failure, and only containers recorded for the current execution enter its final cleanup.

## Separate block-copy path

`backend=swap-blocks` selects [swap_matrix.py](../swap_matrix.py),
the execution controller (Slurm) or Docker executor, and
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
