# TRT Bench Working Notes

These are operational notes for running and debugging `trt-bench`.

## Current Contract

- Branch-only; never merge to `main`.
- Exact 8192-token prompts.
- 256 measured full-batch decode rounds.
- Headline throughput is
  `GBS / decode-round TPOT / active_gpu_count`.
- MTP yield is separate.

Three benchmark-profile families share that outer contract:

- `huawei`: B300 DEP8 or GB300 DEP16, GBS16/64/128, MTP3, one exact
  full-batch prefill, non-overlap `iterLatencyMS`, two request-warmup rounds,
  one measured round skipped, and heuristic sparse top-k.
- PR-max: GB300 profiles `pr-tp32-mtp3`, `pr-tp16-mtp3`, and
  `pr-tp8-mtp1`; PR-active/capacity GBS pairs 192/256, 400/512, and
  3440/4096; overlap scheduler; staged prefill setup; no separate request
  warmup; rank-0 TRT `host_step_time`; eight measured rounds skipped; no
  sparse-attention config; learned model router; default
  `attention_dp_config=None`.
- Rack: `rack-tp8x9-mtp1`; 18 GB300 nodes and 72 GPUs; nine synchronized
  TP8/EP8 MTP1 engines; GBS72/288/576 for Huawei local-batch matching and
  GBS30960/36864 for rack saturation; slowest-replica rank-0
  `host_step_time` per logical round.

The validated baselines are B300 DEP8 at TRT `c185066` and GB300 NVL16
DEP16 using the DeepSeek-V4 development image from InferenceX PR #1689.
The final GB300 sweep is Actions run `27517035480` at
`c0a845521b51e5fb5eca5f9bb4ac2e3a6c60b43d`.

## PR #1689 Max-Throughput Profiles

Source:

```text
PR: https://github.com/SemiAnalysisAI/InferenceX/pull/1689
run: https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27164980476/attempts/14
srt-slurm branch: sa-submission-q2-2026
image: nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1
```

| Profile | Decode topology | Nodes | Offline GBS/local (active, capacity) | MTP | KV | PR output/decode-GPU | PR output/all-GPU |
|---|---|---:|---:|---:|---:|---:|---:|
| `pr-tp32-mtp3` | TP32/EP32 | 8 | 192/6, 256/8 | 3 | 0.70 | 676.763 | 451.175 |
| `pr-tp16-mtp3` | TP16/EP16 | 4 | 400/25, 512/32 | 3 | 0.70 | 2072.1585 | 828.8634 |
| `pr-tp8-mtp1` | TP8/EP8 | 2 | 3440/430, 4096/512 | 1 | 0.80 | 9686.735 | 1383.8193 |

The launcher dynamically allocates 32, 16, or 8 decode GPUs and downloads the
matching EP32, EP16, or EP8 384-slot EPLB map. The CUDA graph lists are copied
exactly from the recipes. Attempt 14's output throughput times mean TPOT gives
active decode populations 189.65, 401.41, and 3437.66; the first point in
each pair is the nearest rank-divisible fixed batch. The second is the copied
engine capacity.

The exact PR engine logs show `attention_dp_config=None`,
`disable_overlap_scheduler=False`, `return_perf_metrics=False`, and
`enable_iter_perf_stats=False`. PR-max preserves the first three and uses the
learned model router, as the recipes do. It enables
`enable_iter_perf_stats=True` with `max_stats_len=2048` only to prove the
offline fixed-batch window and derive MTP yield. Timing still comes from
rank-0 `host_step_time`.

PR decode workers receive KV from dedicated prefill workers, so their
`max_num_tokens` values are 32, 128, and 1024. The monolithic offline worker
must admit 8192-token prompts itself. It records those reference values but
uses `max_num_tokens=32768`, which takes 2, 8, or 128 conservative setup
iterations. The output cap is extended when needed so the earliest request
cannot finish before 256 exact full-batch decode rounds.

Do not use overlap `iterLatencyMS` as the result. The controller aligns
selected iteration IDs with rank-0 `print_iter_log`, filters
`host_step_time` from `offline_timing_ID.log`, and stores the original stats
timing only as a diagnostic. `offline_world_ID.log` is the mirrored all-rank
console stream. TRT resets iteration IDs after engine warmup, so the final
occurrence of each selected ID is the measured executor pass.
The result's `pr_reference` object reports:

- output tok/s per decode GPU, matching InferenceX's serving denominator
- output tok/s normalized by the PR's decode plus prefill GPUs
- ratios to attempt 14 under both denominators

The total-fleet number is a comparison normalization. The offline allocation
contains only the decode GPUs and performs prefill on them before timing.

## GB300 NVL72 Rack Implementation

Source recipe:

```text
PR: https://github.com/SemiAnalysisAI/InferenceX/pull/1689
run: https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27164980476/attempts/14
commit: e2dd5ca5b334e33b6692f81d0dbf9ce59b860d96
TRT source: 34a563ac6d8cc0ca7068c7f619e869fb8a625333
image: nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1
srt-slurm branch: sa-submission-q2-2026
```

The fastest final recipe is TP8/EP8 MTP1. TRT's model-parallel world size is
`tp * pp * cp`, so this remains an eight-rank engine. MoE cluster parallelism
does not turn it into a 72-rank engine. The rack implementation therefore
runs nine independent engines:

```text
18 physical nodes * 4 GPUs = 72 GPUs
2 adjacent nodes * 4 GPUs = 8 ranks per engine
9 disjoint node pairs = 9 TP8/EP8 engines
reported parallelism = 9xDEP8
```

The parent launcher owns one 18-node Slurm allocation. Before engine launch it
requires:

- one exact 72-rank map, ranks `0..71`
- four local ranks on each of exactly 18 hosts
- Fabric `Completed`/`Success` for all 72 GPUs
- one shared non-empty `ClusterUUID` and `CliqueId`

Each child launcher owns one assigned node pair, validates ranks `0..7`,
preseeds the complete external-MPI environment, and starts one
`trtllm-llmapi-launch`. The nine NCCL worlds do not communicate with each
other. All GPUs still belong to the same proven NVLink domain, but the result
must never be labeled TP72.

Copied TP8 engine settings:

- max batch size 512
- CUDA graph batches `1,2,4,8,16,24,32,40..512`
- KV cache fraction 0.80
- overlap scheduler and rank-0 `print_iter_log`
- learned model router; no perfect router
- EP8/384-slot load-balancer map
- `MEGAMOE_DEEPGEMM`, low-precision combine, and PDL
- default parallel weight loading
- `MIMALLOC_PURGE_DELAY=0`
- `UCX_TLS=cuda_ipc,cuda_copy,sm,self,tcp`
- full `/dev/shm` tmpfs and unique EPLB shared-memory name per child

Do not copy the recipe's prefill-only
`TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=1`,
`MIMALLOC_PURGE_DELAY=-1`, or
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` into TP8 decode engines.
Run `27531206092` did that and left eight of nine rack replicas idle during
model finalization at 0% GPU utilization; only replica 3 reached the measured
barrier.

Run `27533885582`, source
`7fd1bdb0bf6a6c9e23d52e38a4fb3db78cb6d99d`, corrected the decode
environment. Replicas `r01` through `r08` all completed initialization and
reached the shared measured-pass barrier in about 17 minutes. Replica `r00`
did not: per-rank diagnostics showed ranks 0-3 at KV-cache construction,
ranks 4/6/7 in MoE finalization, and rank 5 stationary since 08:34 UTC while
opening `model-00043-of-00064.safetensors`. There was no TensorRT fatal
record. The run was canceled after all other replicas proved ready.

This isolates the remaining fault to simultaneous 72-rank model loading, not
the copied TP8 decode configuration. Rack initialization now runs in three
waves of three engines. Every wave must reach the pre-measurement barrier
within 1500 seconds before the next wave starts, limiting concurrent model
loading to 24 ranks. The measured pass still releases all nine engines
together and therefore preserves the 72-GPU fixed-global-batch contract.

The direct offline engine keeps the documented `max_num_tokens=32768`
adaptation because it must admit its own 8K prompts. The serving decode worker
in the PR receives transferred KV and uses `max_num_tokens=1024`.

Rack batches:

| Rack GBS | Engine GBS | Local/GPU | Meaning |
|---:|---:|---:|---|
| 72 | 8 | 1 | same local batch as Huawei GBS16 |
| 288 | 32 | 4 | same local batch as Huawei GBS64 |
| 576 | 64 | 8 | same local batch as Huawei GBS128 |
| 30960 | 3440 | 430 | nine times attempt-14 active decode population |
| 36864 | 4096 | 512 | nine times copied engine capacity |

Every child independently proves 256 consecutive exact fixed-batch decode
iterations. Immediately before its measured `generate()` call, each child
writes `replica_NN.ready.json` and waits. Replica 0 publishes the release only
after all nine ready files are visible. This barrier synchronizes the
measured passes, not engine initialization. Children may wait up to one hour
for the slowest engine to finish initialization. Aggregation rejects a
measured-pass start skew above 10 seconds.

For logical rack round `i`, aggregation takes:

```text
rack_round_latency[i] =
    max(replica_0_host_step[i], ..., replica_8_host_step[i])
```

It then discards the first eight rack rounds, applies only the upper-IQR
filter, and calculates:

```text
decode_step_tput_per_gpu =
    rack_global_batch_size / rack_round_tpot_seconds / 72
```

Speculative proposed/accepted counts are summed across all nine child windows.
Huawei uses MTP3 while this copied engine uses MTP1. Compare raw step
throughput first; apply the measured MTP yield only when discussing output
tokens.

Only `offline_result_rack-tp8x9-mtp1-gbsN.json` is renderer-visible. Child
results stay under:

```text
.offline_rack_ID_JOB/
  barrier/
  replicas/
    r00/ ... r08/
```

That tree is included in `offline_debug_ID.tar.gz` and excluded from
`collect-results`. A rack result is invalid if any child fails, a barrier
index is missing, fabric provenance differs, or fewer than nine timing vectors
reach aggregation.

## GB300 NVL16 Implementation

Reference:

```text
PR: https://github.com/SemiAnalysisAI/InferenceX/pull/1689
image: nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1
TRT source: 34a563ac6d8cc0ca7068c7f619e869fb8a625333
runner: gb300-nv
```

The successful PR launch uses four physical GB300 nodes, four tasks per node,
and one 16-rank `trtllm-llmapi-launch` decode worker. The offline launcher
reuses that execution model directly and does not start Dynamo, HTTP, NATS,
etcd, or srt-slurm.

Fixed GB300 engine profile:

- TP16, EP16, attention DP, LM-head TP, MoE TP1.
- `MEGAMOE_DEEPGEMM`.
- KV FP8, 128-token blocks, no block reuse, memory fraction `0.70`.
- Pinned load balancer
  `/dsv4-eplb-configs/moe_load_balancer_gen_ep16_slots384.yaml`.
- Load-balancer source is pinned to NVIDIA/srt-slurm
  `sa-submission-q2-2026`, SHA256
  `278da78f94be418d189015b18625ba2dbdfe03ee4be09e1a685f0e93708f681b`.
- PDL and ConfigurableMoE enabled. The pinned TRT source defaults
  `ENABLE_CONFIGURABLE_MOE` to `1`, and the PR recipe leaves that default
  intact. `MEGAMOE_DEEPGEMM` needs the wrapper's concrete forward scheduler.
- CUDA graph contains only the exact fixed local batch for each row.
- Overlap remains disabled because this benchmark times synchronized decode
  rounds rather than the serving recipe's overlapped scheduler.

Capacity:

| GBS | Local/rank | `max_batch_size` | CUDA graph | `max_num_tokens` | Synthetic warmup | Minimum KV tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 1 | 1 | 8192 | 8192 | 9344 |
| 64 | 4 | 4 | 4 | 32768 | 32768 | 37376 |
| 128 | 8 | 8 | 8 | 65536 | 32768 | 74752 |

GB300 GBS128 keeps its 65536-token runtime and real full-batch prefill, but
uses the GBS64-proven 32768-token shape for TRT's synthetic context-only
tuning request. The B300-only eager attention workspace, 12 GiB KV reserve,
and 131072-row DeepGemm chunker are disabled. The packed-FP8 tactic guard is
enabled above 32768 rows, so the real GBS128 prefill uses TRT's Triton
quantizer; all measured decode shapes stay on the fused path.

Launch chain:

1. Import the ARM64 image on a GB300 compute node into the shared squash
   cache. Do not import it on the x86 runner/login node.
2. Allocate four nodes with `--gpus-per-node=4` and
   `--ntasks-per-node=4`.
3. Run a 16-task probe and require global ranks `0..15`, local ranks `0..3`
   on each of exactly four hosts.
4. Run a one-task-per-node full `nvidia-smi -q` Fabric probe. Require four
   GPUs plus four `State: Completed` and four `Status: Success` records per
   node, then require one shared non-empty `ClusterUUID` and `CliqueId`
   across all 16 GPUs.
5. Start one telemetry task per physical node.
6. Start the engine with:

```text
srun --overlap --mpi=pmix --oversubscribe --cpu-bind=verbose,none \
  --nodes=4 --ntasks=16 --ntasks-per-node=4 \
  ... \
  numactl -m 0,1 trtllm-llmapi-launch \
  bash /workspace/benchmarks/single_node/offline/run_dsv4_trt_container.sh
```

The rank marker and fixed-batch arm file must be under shared `/workspace`.
External MPI rank processes also need
`PYTHONPATH=/workspace/utils/bench_offline` before
`trtllm-llmapi-launch`, otherwise they cannot import the monkeypatched worker
entry submitted by the controller.

External MPI has an additional launch-order requirement. Its 16 management
workers are created before rank 0 starts `run.py`, so environment variables
added only to the later `trt_worker.py` subprocess do not reach the other
ranks. The host launcher must call `emit_rank_environment.py` and export the
complete fixed-batch contract before `trtllm-llmapi-launch`. This includes the
warmup cap, fixed GBS, arm file, rank-contract JSON, perfect-router aliases,
CuTe cache alias, MoE controls, KV controls, and PDL. `run.py` recomputes the
contract and fails before engine construction if any preseeded value differs.
Do not move this setup into `run_dsv4_trt_container.sh`; that command starts
too late for external ranks.

Artifacts unique to GB300:

- `offline_allocation_ID.log`
- `offline_rank_map_ID.tsv`
- `offline_topology_ID.log`
- `offline_gpu_metrics_ID_HOST.csv` for every allocated host
- `offline_completion_ID.json`
- `offline_world_ID.log`
- `offline_timing_ID.log`

The topology log proves that the expected profile-sized rank set entered the
GB300 NVLink Fabric before the measured engine was started. The result itself
records the Slurm node list, Fabric `ClusterUUID`, `CliqueId`, and artifact
names.

The final GBS16/64/128 sweep completed end to end in run `27517035480`.
Every row has an Actions artifact proving the exact 16-rank set, fabric
checks, fixed-batch schedule, 256-round window, matching completion record,
and flat renderer row. See `Final Validated GB300 Run`.

### GB300 Canary History

Run `27529535864`, source
`2976c179449641fa2b09b6f470bdd3cf86c1288a`, was the first 72-GPU rack
canary. Slurm job `8799` allocated all 18 nodes, validated all 72 GPUs under
Fabric `ClusterUUID 8fe56262-d2bb-4602-b338-8898d34c4731` and
`CliqueId 32766`, launched nine disjoint TP8/EP8 engines, and completed both
TRT engine warmup phases on replica `r00`. The run then failed before measured
generation because the parent passed `TRT_BENCH_REPLICA_INDEX` while
`trt_worker.py` consumes `TRT_BENCH_RACK_REPLICA_INDEX`; the child exported
the latter without assigning it. The rack launch now uses the exact worker
variable name end to end and validates/logs replica count, index, timeout, and
barrier path in the controller before TensorRT initialization.

Run `27520024692`, source
`9d8c85b4093a7754eb87f20819ec0d7f845e15d5`, was the first copied PR-max
TP32 run. Its GBS192 worker completed successfully on 32 ranks, selected
iterations `3..258`, and observed `3.202474` output tokens per decode step.
The controller failed only because TRT emitted rank-0 `print_iter_log` rows
to the outer `srun` stream rather than `worker.log`; the remaining matrix was
canceled before spending GPU time on the same postprocessing failure.

Reconstructing the final occurrence of each selected iteration from the
Actions log gives a provisional `23.579741 ms` round TPOT,
`254.455724` decode steps/s/GPU, and `814.887830` output tok/s/GPU after the
normal startup skip and upper-IQR filter (`233/255` rows retained). This is a
plumbing validation, not an official benchmark row. The follow-up captures
the same outer stream in `offline_world_ID.log` and requires the controller
to publish the successful result before collection.

Run `27524518166`, source
`edd91e444f581129771dbe0184e4d6c65baa6dc0`, proved that host-side capture
was still not a stable controller input for larger logs. TP16 GBS400's worker
completed successfully, selected exact local-batch-25 iterations `9..264`,
and the final artifact contained rank-0 timing rows through iteration `494`.
During postprocessing, however, the rank-0 container's NFS view of the
host-appended file stopped at iteration `131`; the controller timed out after
10 seconds and GBS512 was canceled. The follow-up writes
`offline_timing_ID.log` directly from Slurm rank 0 on the same compute node
as the controller, while a separate follower mirrors that file to the
all-rank console stream.

Run `27525200755`, source
`d8abc002a59d930987870c2ca0682f77b2fbb589`, validated the direct rank-0
timing path at TP16 GBS400. GBS512 then exposed an independent startup race:
MPI rank-entry hooks created marker files in the shared work directory while
the rank-0 container script was deleting that directory. Rank 0 exited on
`rm: Directory not empty` before model loading; GPU telemetry remained idle.
The host now removes and creates the unique work directory before `srun`,
and external-MPI rank 0 only reuses it.

Run `27502789238`, source
`3ca9a9febe452918641fc842e5e425553cf035a2`, reached Slurm allocation
`8707` on `im-gb300-r01-c011` through `c014`. Its rank artifact proves global
ranks `0..15`, four ranks per host, and local ranks `0..3`. The run stopped
before TRT because this driver rejects `nvidia-smi -q -d FABRIC` with exit
code `2`. Fabric data is present in full `nvidia-smi -q` output. The follow-up
uses the tested `gb300_fabric.py` parser and additionally requires one shared
non-empty `ClusterUUID` and one shared non-empty `CliqueId` across all 16
GPUs.

Run `27504087069`, source
`eec84a369a90148aef7d2bc7531bbcaede3529b7`, reached Slurm allocation
`8712` on `im-gb300-r01-c015` through `c018`. It proved:

- Global ranks `0..15`, local ranks `0..3`, four ranks on each node.
- Fabric `State: Completed` and `Status: Success` for all 16 GPUs.
- Shared Fabric `ClusterUUID`
  `8fe56262-d2bb-4602-b338-8898d34c4731`.
- Shared Fabric `CliqueId` `32766`.

The engine never loaded. All 16 external workers immediately raised
`KeyError: 'TRTLLM_BENCH_ENGINE_WARMUP_MAX_TOKENS'` in
`trt_mpi_entry.py`. Maximum observed memory was only 4464 MiB on one node and
2930 MiB on the other three, with at most 2% GPU utilization. The outer MPI
server reported `16/16 MPI worker(s) failed`, but rank 0 remained alive until
the 1800-second controller timeout. This established that the external MPI
management workers did not inherit the controller subprocess's dynamic
environment.

The follow-up preseeds the rank environment on the host before
`trtllm-llmapi-launch`, validates it again in `run.py`, and writes
`entry_start`/`entry_failed` markers around rank shim installation. The
controller now treats an `entry_failed` or `*_error` marker as terminal, so
the same class of failure should stop on the next 60-second heartbeat instead
of consuming the full timeout.

Run `27508071804`, source
`8ac00f61f2d48bf97b5c10c217d92d463da35e32`, reached Slurm allocation
`8714` on `im-gb300-r01-c015` through `c018`. It proved the host-side
environment propagation fix: all 16 ranks emitted `entry_ready` and reached
engine warmup. About 24 seconds into warmup, every rank raised
`NotImplementedError` from the abstract `MoE.forward_impl`. The profile had
incorrectly forced `ENABLE_CONFIGURABLE_MOE=0`. At TRT source
`34a563ac6d8cc0ca7068c7f619e869fb8a625333`, `create_moe()` defaults that
variable to `1` and wraps `MegaMoEDeepGemm` in `ConfigurableMoE`; the
successful PR #1689 run uses the same default. The controller wrote its failed
result after roughly 189 seconds, but the outer `trtllm-llmapi-launch` rank
world stayed alive until the Slurm allocation expired.

The next canary enables ConfigurableMoE and adds an atomic completion record.
Rank 0 publishes it only after copying `result.json` and creating the debug
archive. The record contains the controller return code and copied result
status. The host watches it, logs one-minute world progress, verifies that the
statuses agree, and cancels the Slurm allocation immediately. Benchmark
success comes from the controller result rather than the transport's expected
post-cancel exit code.

Run `27511242827`, source
`dc671ab6098f0b7176d65377f8b80fbb176d1f07`, completed GBS16 on Slurm
allocation `8717`, nodes `im-gb300-r01-c012` through `c015`. It proved:

- Exact global ranks `0..15`, four ranks per node, and local ranks `0..3`.
- Fabric `ClusterUUID`
  `8fe56262-d2bb-4602-b338-8898d34c4731` and `CliqueId` `32766`.
- One context-only full-batch prefill, zero mixed context/decode iterations,
  and 256 consecutive full-batch decode rounds.
- Controller completion canceled the external MPI world immediately instead
  of waiting for the Slurm time limit.
- A renderer-compatible one-row `agg_bmk.json`.

GBS16 measured `25.943764 ms` round TPOT, `38.544908` decode steps/s/GPU,
`3.140625` output tokens/step, `121.055103` output tok/s/GPU, and
`71.716572` wall output tok/s/GPU. The filter retained `247/255` candidate
rounds. The raw step rate was `0.679804x` Huawei and the acceptance-adjusted
output rate was `0.875004x` Huawei.

The result is valid, but its diagnostic marker exposed a shared-filesystem
write bug: `perfect_router.jsonl` contained 188 physical lines, of which 12
were malformed by concurrent rank appends. Enough later records survived to
prove all required ranks and schedule events, but the final sweep must not
silently tolerate that. The follow-up serializes every JSONL append with
`flock`, takes the same lock for parser snapshots, and fails rank propagation
if any malformed record remains. It also makes the host heartbeat read the
host-visible `${GITHUB_WORKSPACE}` marker path; the canary incorrectly logged
`rank_events=0` while checking its `/workspace` container alias.

Run `27511740130`, source
`782fb58191c7887ab7b3cfeea44917a2e7376419`, proved the marker fix at GBS16:
all 248 physical JSONL lines parsed, all required lifecycle events covered
ranks `0..15`, and the host heartbeat reported increasing event counts.
GBS16 completed successfully, but the sequential GBS64 job inherited
`gb300-nv_0` immediately after srt-slurm run `27164980476`. Root-level
`actions/checkout` ran `git clean -ffdx` over that job's open NFS log files,
retried for ten minutes, and failed on `.nfs*` with `EBUSY`. GBS128 was then
canceled by fail-fast; no GBS64 benchmark process or GPU allocation started.

The follow-up checks out this workflow under
`${GITHUB_WORKSPACE}/offline-bench` and sets `TRT_BENCH_WORKSPACE` to that
isolated tree. Benchmark source, output, container mounts, and artifact paths
all use the isolated directory. Never restore a root-level clean checkout on
these shared runners: unrelated rack workflows intentionally leave large
untracked trees and may still have open NFS handles during job handoff.

Run `27513364142`, source
`a6cb0c4240d822d9c982e3b663cf02e33a9947a0`, proved the isolated checkout:
GBS16 completed and GBS64 reached a valid controller result. GBS16 measured
`29.196502 ms` round TPOT, `34.250679` decode steps/s/GPU,
`3.019531` output tokens/step, and `103.420994` output tok/s/GPU. GBS64
measured `32.447992 ms`, `123.274191` decode steps/s/GPU,
`3.482422` output tokens/step, and `429.292739` output tok/s/GPU. Its schedule
had one full-batch prefill, zero mixed iterations, and 256 consecutive
local-batch-4 decode rounds; the filter retained `203/255` candidate rounds.

The workflow still marked GBS64 failed because the external MPI world exited
cleanly about 45 seconds after rank 0 atomically published its success files.
The host's repeated negative NFS lookup did not expose those files until
roughly 50 seconds after publication, just after the old five-second
post-exit grace expired. Artifact upload then found the successful completion
and debug archive, while the host failure fallback had overwritten only the
top-level result. GBS128 was canceled by fail-fast.

The follow-up requires both the result and completion file, waits up to
`TRT_BENCH_COMPLETION_VISIBILITY_TIMEOUT=120` seconds after a clean transport
exit, logs visibility every ten seconds, and only then compares statuses.
Do not reduce this to a short fixed sleep or accept the MPI return code as
benchmark success.

Run `27514818464`, source
`e173082762cf949819d6220e1b44f54967c87a26`, proved the handoff fix: the host
read matching failed result/completion files and canceled the allocation
without the earlier visibility fallback. GBS16 itself failed schedule
validation because its measured stats began with inactive warmup tail
iteration `4`, followed by the real measured prefill at `5` and 279
consecutive full-batch decode rounds at `6..284`. The prior successful GBS16
had drained three such tails into the warmup read, so its measured history
started directly at prefill.

This is a pinned TRT `IterationResult` boundary race. `get_results()` marks
the queue done when it appears empty; a late stat can remain queued until the
next prompt submission calls `mark_undone()`. The selector now ignores only
leading rows that prove they are an inactive prior-pass tail:
`context=0`, `generation=scheduled=local_batch`, and
`active=queued=paused=0`. It records the ignored count and iteration range.
Any active, queued, partial, or mixed work before prefill still fails.

Run `27515257151`, source
`4a003c6a839a66eeeb31be733ea693b696479dd3`, completed GBS16 and GBS64.
GBS16 measured `25.939235 ms`, `38.551639` decode steps/s/GPU,
`2.957031` output tokens/step, and `113.998400` output tok/s/GPU. GBS64
measured `32.456790 ms`, `123.240778` decode steps/s/GPU,
`3.477539` output tokens/step, and `428.574620` output tok/s/GPU. Both had one
full-batch prefill, zero mixed iterations, and 256 consecutive full-batch
decode rounds. Their result and completion statuses matched.

GBS128 failed during TRT engine warmup on rank 11. The runtime shape was
`max_batch_size=8`, `max_num_tokens=65536`; the fused
`fp8_quantize_1x128_packed_ue8m0` kernel rejected its 65536-row launch with
CUDA `invalid argument`. Telemetry still showed about 10-12 GiB free per GPU,
so this was not an OOM. The profile had disabled the packed-FP8 guard because
the shape fits the warmup ceiling, but that ceiling is exactly the failing
kernel shape. The follow-up enables the existing 32768-row Triton fallback
for GB300 while leaving DeepGemm chunking and B300 memory workarounds off.

Run `27516149323`, source
`5245f9b221939bee948058146379dc8fb5fd3053`, proved the packed-FP8 guard:
GBS128 passed the previous 65536-row CUDA `invalid argument` and completed
about 325 seconds of synthetic engine warmup/autotuning. Initialization then
failed with TRT's explicit insufficient-memory report while creating
temporary KV-estimation resources. Per-rank model resources had reached about
272 GiB, leaving 66.67 GiB free where the temporary executor needed
97.33 GiB. Peak telemetry was 271-278 GiB used per GPU.

The follow-up retains `max_num_tokens=65536`, local batch 8, minimum KV
capacity 74752, and the real one-iteration 65536-token prefill. It changes
only `_create_warmup_request()`'s pure-context tuning shape to 32768 tokens,
which is already proven by the successful GBS64 row on the same image.

Run `27516584571`, source
`c0a845521b51e5fb5eca5f9bb4ac2e3a6c60b43d`, was the successful GBS128
canary for that change. It kept runtime `max_num_tokens=65536`, capped only
the synthetic tuning request at 32768 tokens, initialized in about 625
seconds, and completed the exact local-batch-8 schedule. It measured
`34.319116 ms` round TPOT, `233.106236` decode steps/s/GPU,
`3.310059` output tokens/step, and `771.595299` output tok/s/GPU. All 339
marker records parsed, all 16 ranks reached warmup and executor start, and
the minimum observed free memory was 7932 MiB.

Run `27517035480`, at the same source revision, is the final sequential
GBS16/64/128 sweep. All three jobs and `collect-results` succeeded. The
aggregate and `results_bmk` copies of `agg_bmk.json` are byte-identical,
contain exactly three flat rows, and use `conc=global_batch_size` only as the
renderer compatibility alias. See `Final Validated GB300 Run` for the exact
rows and validation details.

## Why The Old Result Was Too High

Run `27483465692` used a request pool and mean per-request decode windows. At
GBS 1024 it reported:

```text
derived output: 1582.72 tok/s/GPU
wall output:     731.58 tok/s/GPU
```

Those are not two measurements of the same schedule. The derived formula
multiplied each request's active-window speed by all 1024 requests even though
TRT queued and staggered them. The implied average active decode population
was about 473 requests, not 1024.

That run and all deleted experiment matrices are historical optimization data,
not Huawei-comparable benchmark rows.

## Source Audit

Huawei:

```text
/Users/bshan/Documents/cann-recipes-infer-master/
```

Relevant implementation:

- `models/deepseek-v4/infer.py`: one warmup generation, then one measured
  generation.
- `models/deepseek-v4/models/model_infer.py`: warmup stops after two decode
  rounds; measured decode runs `max_new_tokens=256` rounds; each recorded time
  is main-model time plus all MTP model times.
- `executor/utils/common_utils.py`: skips the first recorded decode round,
  removes values above `Q3 + 1.5 * IQR`, and averages retained values.
- The guide's throughput is exactly `GBS / TPOT / chips`.

Pinned TRT:

- `LLM.get_stats()` returns per-iteration `iterLatencyMS` and inflight batch
  counts when `enable_iter_perf_stats=true`.
- Set `max_stats_len=2048`. It keeps the history bounded while retaining
  prefill plus the worst-case 1024 decode iterations. The pinned executor
  treats `-1` as unbounded.
- `max_batch_size` is per attention-DP rank.
- `TLLM_METRICS_ALL_RANKS=1` adds a per-iteration collective and must remain
  disabled for timing.
- `iterLatencyMS` measures a complete TRT executor iteration. Huawei sums
  main-model and MTP timing regions inside CANN. The workload, decode window,
  and filtering match; the runtime instrumentation point does not.

## B300 Capacity And Workaround History

| GBS | Local/rank | `max_batch_size` | CUDA graph | `max_num_tokens` | Minimum KV tokens | KV reserve |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 2 | 2 | 2 | 16384 | 18688 | 0 |
| 64 | 8 | 8 | 8 | 65536 | 74752 | 0 |
| 128 | 16 | 16 | 16 | 131072 | 149504 | 12 GiB |

`max_num_tokens` is intentionally much larger than the old recipe. It permits
all local 8192-token prompts to prefill in the same iteration. KV capacity
starts from TRT's memory-derived calibration with a fixed 0.60 free-memory
fraction. Do not set an exact `kv_cache.max_tokens`: the pinned DeepSeek-V4
multi-pool cache manager did not turn the exact-sequence quota into sufficient
physical capacity, and the cap used in run `27486168511` admitted only half of
each local batch into prefill.

At GBS128, subtract 12 GiB from the calibrated final
`max_gpu_total_bytes`. The hook runs immediately after
`KvCacheCreator.configure_kv_cache_capacity()` and before
`build_managers(..., False)`. DeepSeek-V4 uses a `KVCacheManagerV2` subclass,
so this byte budget directly controls the final multi-pool allocation. Before
changing it, the hook uses TRT's aggregate cache cost to require capacity for
`16 * 9344 = 149504` tokens. It fails initialization if the reserve would
violate that bound. GBS16 and GBS64 leave calibration unchanged.

TRT fused MoE is capped at 65536 tokens per internal invocation so its
synthetic autotune and prefill tensors are chunked without splitting the
executor-level prefill iteration. Measured decode is far below the cap.
The FP8 block-scaled linear runner likewise keeps each oversized activation
quantization and DeepGemm call at no more than 65536 rows while writing into
one final output tensor.

The pinned engine also uses `max_num_tokens` to build synthetic shapes during
`PyTorchModelEngine.warmup()`. GBS128 run `27487131935` completed GBS16 and
GBS64,
but spent 62 minutes in GBS128 initialization before cancellation because it
was tuning a synthetic 131072-token prefill shape. The first mitigation
temporarily changed `engine.max_num_tokens` to 65536 during warmup and restored
it afterward.

Run `27488380128` exposed an implementation mistake in the first cap: it
patched the abstract `ModelEngine.warmup`, while the runtime calls the
overridden `PyTorchModelEngine.warmup`. GPU activity stopped after initial
setup and the job remained in engine initialization for 52 minutes. That
revision and its unit test then targeted the concrete override explicitly.

Run `27489466718` proved the concrete cap executes: rank output reported
`runtime_max_tokens=131072`, `tuned_max_tokens=65536`, then completion after
334.667 seconds with `restored_max_tokens=131072`. Engine initialization still
did not return and all GPUs were idle afterward. Per-rank marker events now
cover engine warmup, global clock synchronization, and executor worker start
so a canceled canary identifies the exact rank lifecycle boundary.

Run `27490077837` showed every rank completed executor worker startup. Run
`27490378501` then preserved all-thread dumps and exposed the actual blocker:
rank 0's event loop raised `Fixed-batch barrier timed out after 120.0s with
120/128 requests` inside TRT's KV-cache capacity calibration. The other ranks
were waiting for rank 0's broadcast, so the parent remained blocked in
`configure_kv_cache_capacity()`.

Those 120 requests are TRT-generated calibration dummies, not the benchmark
batch. The MPI request shim is now installed but disarmed while `LLM(...)`
initializes. The controller clears a unique arm file before launch; the parent
worker atomically creates it only after `LLM(...)` returns and immediately
before real warmup generation. Rank 0 then latches the gate on for both the
warmup and measured passes. A stale or early arm file is a hard error.

Run `27490833024` proved that arm lifecycle works: calibration was no longer
blocked at `120/128`. It then exposed a flaw in the old warmup mitigation.
DeepSeek-V4 attention metadata is allocated lazily during the first warmup
forward, so lowering `engine.max_num_tokens` created 65536-token buffers.
Calibration later scheduled 84087 tokens and failed in
`host_req_idx_per_token` with target size 65536.

The current shim never changes runtime `max_num_tokens`. It wraps only
`PyTorchModelEngine._create_warmup_request()` and clamps pure-context
synthetic requests to 65536 tokens. The first forward therefore allocates
attention/spec metadata for the full 131072-token runtime capacity while
expensive general/autotuner warmup shapes remain bounded. The run remains
invalid unless schedule validation proves one local-batch-16 prefill and 256
full-batch decode rounds.

Run `27491160719` verified that correction: all eight ranks retained
`runtime_max_tokens=131072`, capped only the synthetic context request, and
completed engine warmup in about 335 seconds. After clock synchronization and
executor-worker startup, TRT's 84087-token capacity probe tried to grow the
non-graph MLA attention workspace from 12,953,234,944 to 16,600,658,432
bytes. The pinned C++ op calls `resize_()` on that live CUDA tensor, and every
rank then reported an illegal memory access. MPI stayed alive, so the old
controller mislabeled the run as an 1800-second timeout.

GBS128 now preallocates only the cached eager
`TrtllmAttentionMetadata.workspace` before its first forward:

```text
200 KiB * 131072 runtime tokens + 256 MiB = 27111981056 bytes
```

That is above the observed approximately 192 KiB/token growth curve and
includes fixed headroom. `cuda_graph_workspace` remains untouched because it
is a separate small decode buffer. GBS16 and GBS64 set the reservation to
zero. Every GBS128 rank must emit `attention_workspace_preallocated` with the
exact target, an allocation at least that large, and zero CUDA-graph
workspace bytes at hook time. The controller also treats TRT's native
`Fatal error detected, initiating shutdown` line as terminal instead of
waiting for a stuck MPI parent.

Run `27491999545` proved the eager workspace fix. All eight ranks allocated
the exact 27,111,981,056-byte target in both engine phases. The first engine
warmup completed in about 337.5 seconds, final warmup completed in about
42.4 seconds, and `LLM(...)` returned after 580.9 seconds without the old
workspace resize or illegal memory access.

The first real GBS128 warmup then failed in
`_deepseek_v4_q_b_layernorm_fused_fp8`. Final KV pools had left only
1.17-2.35 GiB free per GPU, but the exact full-batch prefill needed an 8 GiB
FP8 Q projection buffer:

```text
131072 tokens * 128 heads * 512 qk_head_dim * 1 byte = 8 GiB
```

The next BF16 RoPE projection is roughly another 2 GiB:

```text
131072 tokens * 128 heads * 64 rope_dim * 2 bytes = 2 GiB
```

The new 12 GiB final-KV reserve covers both tensors and allocator margin. This
does not lower runtime `max_num_tokens`, split prefill, reduce MTP3, or alter
the decode kernel path.

Run `27492438399` proved the reserve hook on ranks `0..7`. Both engine warmups
completed, `LLM(...)` returned after 575.4 seconds, and the real full-batch
prefill ran for about 118 seconds before an illegal memory access. The first
useful stack was `q_b_proj -> fp8_swap_ab_gemm`, with the error surfacing at
the output allocation immediately after the 131072-row Triton activation
quantizer launch. Because CUDA reported it asynchronously, that stack does
not distinguish the quantizer from the following DeepGemm operation.

The rank shim now wraps `fp8SwapABGemmRunner.forward()` only above 65536 rows.
It allocates the final output once, quantizes contiguous row chunks, and calls
DeepGemm with the pinned transformed weight/scale pair while writing directly
to output row views. Every oversized chunk is synchronized. This affects
prefill and calibration only; decode rows remain 2, 8, or 16 and execute the
original runner without the wrapper body.

Run `27486396235` proved memory-derived KV restores a full GBS16 prefill, but
GBS64 then exposed a separate pinned-kernel limit: the packed-FP8 CUDA
quantizer rejected the 65536-row MTP `h_proj` launch during engine warmup.
Each MPI rank now restricts that fused tactic to at most 32768 rows and uses
TRT's existing Triton quantizer above the limit. Decode matrices are only
local batch 2/8/16, so headline timing remains on the original fused path.

## Schedule Gate

After engine initialization, the branch-local MPI entry shim waits at each
idle pass boundary until exactly one complete GBS has been enqueued. Before
the arm file exists, it passes TRT's internal initialization requests through
unchanged. This prevents the live executor from starting benchmark prefill
after only the first few `generate_async()` submissions without intercepting
TRT's own capacity probes.

There must be exactly one prefill iteration with:

```text
context == local_batch
generation == 0
scheduled == local_batch
active == local_batch
queued == 0
paused == 0
```

The first decode iteration and the next 255 must all show:

```text
context == 0
generation == local_batch
scheduled == local_batch
active == local_batch
queued == 0
paused == 0
```

Iteration IDs must be consecutive. Any mixed context/decode iteration or
partial first decode batch is `fixed_batch_validation`.

## Output Cap

TRT stops by emitted tokens, while Huawei stops by decode rounds. The measured
cap is:

```text
1 + 256 * (3 + 1) = 1025 tokens
```

Therefore no request can finish before 256 MTP3 rounds. The first 256 valid
rounds are measured; later rounds only let lower-acceptance requests reach the
common output cap.

The warmup cap is six tokens, which guarantees at least two MTP3 rounds. Only
the first two valid warmup rounds are required.

## Dispatch

GB300 NVL72 GBS72 canary:

```bash
BENCH_REF="$(git rev-parse HEAD)"
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='trt-bench' \
  -f "inputs[ref]=$BENCH_REF" \
  -f 'inputs[hardware-profile]=gb300' \
  -f 'inputs[benchmark-profile]=rack-huawei-sweep' \
  -f 'inputs[test-name]=DSV4 GB300 NVL72 TRT rack GBS72 canary' \
  -f 'inputs[global_batch_sizes]=72' \
  -f 'inputs[salloc-time]=360' \
  -f 'inputs[worker-timeout]=18000' \
  -f 'inputs[worker-stack-period]=-1'
```

GB300 NVL72 Huawei-local-batch sweep:

```bash
BENCH_REF="$(git rev-parse HEAD)"
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='trt-bench' \
  -f "inputs[ref]=$BENCH_REF" \
  -f 'inputs[hardware-profile]=gb300' \
  -f 'inputs[benchmark-profile]=rack-huawei-sweep' \
  -f 'inputs[test-name]=DSV4 GB300 NVL72 TRT Huawei-local-batch sweep' \
  -f 'inputs[global_batch_sizes]=auto' \
  -f 'inputs[salloc-time]=360' \
  -f 'inputs[worker-timeout]=18000' \
  -f 'inputs[worker-stack-period]=-1'
```

GB300 NVL72 saturation sweep:

```bash
BENCH_REF="$(git rev-parse HEAD)"
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='trt-bench' \
  -f "inputs[ref]=$BENCH_REF" \
  -f 'inputs[hardware-profile]=gb300' \
  -f 'inputs[benchmark-profile]=rack-max-sweep' \
  -f 'inputs[test-name]=DSV4 GB300 NVL72 TRT max offline' \
  -f 'inputs[global_batch_sizes]=auto' \
  -f 'inputs[salloc-time]=360' \
  -f 'inputs[worker-timeout]=18000' \
  -f 'inputs[worker-stack-period]=-1'
```

The workflow matrix is sequential. Start with GBS72. Once it succeeds, run
`rack-huawei-sweep` for the local-batch comparison and `rack-max-sweep` for
the actual maximum-throughput points. All rack rows request the complete
18-node allocation.

GB300 PR-max sweep:

```bash
BENCH_REF="$(git rev-parse HEAD)"
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='trt-bench' \
  -f "inputs[ref]=$BENCH_REF" \
  -f 'inputs[hardware-profile]=gb300' \
  -f 'inputs[benchmark-profile]=pr-max-sweep' \
  -f 'inputs[test-name]=DSV4 GB300 TRT PR max offline' \
  -f 'inputs[global_batch_sizes]=auto' \
  -f 'inputs[salloc-time]=300' \
  -f 'inputs[worker-timeout]=18000' \
  -f 'inputs[worker-stack-period]=-1'
```

Single PR profile: replace `pr-max-sweep` with `pr-tp32-mtp3`,
`pr-tp16-mtp3`, or `pr-tp8-mtp1`. Keep
`inputs[global_batch_sizes]=auto` for both copied points, or pass a validated
subset such as `512` for a targeted capacity rerun. `pr-max-sweep` always
requires `auto`.

Attempt 14's final copied jobs took about 38 minutes for TP32, 80 minutes for
TP16, and 216 minutes for TP8. The TP8 cost is dominated by initializing the
complete CUDA graph list through batch 512, so reducing the benchmark to one
request pass does not remove it. Keep the PR-max 300-minute Slurm limit,
18000-second worker timeout, and 420-minute Actions job timeout unless a later
run proves a tighter bound.

GB300 Huawei full sweep:

```bash
BENCH_REF="$(git rev-parse HEAD)"
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='trt-bench' \
  -f "inputs[ref]=$BENCH_REF" \
  -f 'inputs[hardware-profile]=gb300' \
  -f 'inputs[benchmark-profile]=huawei' \
  -f 'inputs[test-name]=DSV4 GB300 TRT Huawei fixed GBS' \
  -f 'inputs[global_batch_sizes]=16,64,128' \
  -f 'inputs[salloc-time]=180' \
  -f 'inputs[worker-timeout]=7200' \
  -f 'inputs[worker-stack-period]=-1'
```

GB300 GBS16 canary:

```bash
BENCH_REF="$(git rev-parse HEAD)"
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='trt-bench' \
  -f "inputs[ref]=$BENCH_REF" \
  -f 'inputs[hardware-profile]=gb300' \
  -f 'inputs[benchmark-profile]=huawei' \
  -f 'inputs[test-name]=DSV4 GB300 TRT fixed GBS16 canary' \
  -f 'inputs[global_batch_sizes]=16' \
  -f 'inputs[salloc-time]=120' \
  -f 'inputs[worker-timeout]=7200' \
  -f 'inputs[worker-stack-period]=-1'
```

B300 reruns use the same command with
`-f 'inputs[hardware-profile]=b300'`.

Find and watch:

```bash
RUN_ID=$(gh run list --repo SemiAnalysisAI/InferenceX \
  --workflow e2e-tests.yml --event workflow_dispatch --limit 1 \
  --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" --repo SemiAnalysisAI/InferenceX --exit-status
```

When a benchmark is silent inside `salloc`, run a read-only queue snapshot on
an available GB300 runner:

```bash
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='trt-bench' \
  -f 'inputs[diagnostic-only]=true' \
  -f 'inputs[diagnostic-job-name]=gb300-nv_1' \
  -f 'inputs[test-name]=GB300 Slurm status'
```

Normal launches stream `salloc` output, print one queue heartbeat per minute,
and preserve `offline_allocation_ID.log`. The external `srun`/MPI stream is
shown live and preserved as `offline_world_ID.log`. Slurm rank 0 writes
`offline_timing_ID.log` directly on its compute node and mirrors it into the
world stream; PR-max headline timing is parsed from the direct file. A
`PENDING` heartbeat means TRT has not started and the worker timeout has not
begun. The matrix is sequential. It does not fail fast, so one copied PR
profile failing does not suppress the other topology results. It runs TP32,
then TP16, then TP8 so launch/config errors surface on the shortest copied
engine before the multi-hour TP8 graph initialization.

Set `diagnostic-job-name` to the runner name shown on the target Actions job.
The diagnostic prints the matching Slurm allocation and steps, then tails the
top-level rack controller/world logs and all visible child
`host_launcher.log` files without modifying the allocation.

For an initialization hang, let the controller time out so the container
finalizer preserves `worker.log`; do not cancel the Actions run externally.
Enable TRT's built-in all-thread dumps:

```bash
-f 'inputs[worker-timeout]=900' \
-f 'inputs[worker-stack-period]=120'
```

The normal benchmark default is `-1`, which disables stack dumps.

## Result Interpretation

Authoritative offline row:

- `decode_round_tpot_ms`: filtered full-batch timing from `timing_source`
- `timing_source`: Huawei `iter_latency_ms` or PR-max
  `trt_print_iter_log_host_step_time`; rack uses
  `slowest_replica_trt_print_iter_log_host_step_time`
- `decode_step_tput_per_gpu`: full-batch step rate per active decode GPU
- `observed_tokens_per_step`: MTP output multiplier
- `output_tput_per_gpu`: step rate times output multiplier
- `wall_output_tput_per_gpu`: whole generation diagnostic
- `filter`: first-round skip and retained/outlier counts
- `schedule_validation`: exact selected iteration and local-batch proof
- `stats_iter_latency_diagnostic`: overlap iteration timing retained only for
  PR-max diagnosis
- `pr_reference`: attempt-14 reference values and decode/fleet ratios
- `rack`: all nine child sources, node pairs, child timing values, and the
  same-index slowest-replica aggregation rule
- `measured_start_skew_seconds`: spread between the nine child measured-pass
  start timestamps; useful for verifying the shared barrier

Flat renderer row:

- `tput_per_gpu` and `output_tput_per_gpu`: acceptance-adjusted output rate
- `mean_tpot`: equivalent output-token TPOT
- `decode_round_tpot_ms`: custom raw-round field
- `decode_step_tput_per_gpu`: custom raw-step field
- `global_batch_size`, `local_batch_size`, `measured_decode_rounds`: custom
  workload proof
- `conc`: renderer compatibility alias for `global_batch_size`; it is not
  HTTP serving concurrency and did not control scheduling
- `benchmark_profile` and `timing_source`: required to distinguish Huawei
  rows from PR-max and rack rows
- `replica_count=9`, `decode_tp=8`, `decode_ep=8`, and
  `num_decode_gpu=72` identify the rack as nine active DEP8 engines rather
  than a nonexistent TP72 engine

Do not put the custom aggregate wrapper in `results_bmk`; the unofficial-run
API expects every JSON object there to be a flat benchmark row.

## Final Validated GB300 Run

Workflow:

```text
run: 27517035480
url: https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27517035480
git: c0a845521b51e5fb5eca5f9bb4ac2e3a6c60b43d
TRT source: 34a563ac6d8cc0ca7068c7f619e869fb8a625333
image: nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1
```

Renderer:

```text
https://inferencemax-r4i4xgna4-semianalysisai.vercel.app/inference?unofficialrun=27517035480
```

| GBS | Local/rank | Round TPOT ms | Steps/s/GPU | Tok/step | Output tok/s/GPU | Wall tok/s/GPU | Retained | Step/Huawei | Output/Huawei |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 25.921265 | 38.578364 | 3.003906 | 115.885788 | 90.025851 | 213/255 | 0.680394 | 0.837640 |
| 64 | 4 | 32.375174 | 123.551461 | 3.509766 | 433.636669 | 212.253702 | 237/255 | 0.587892 | 0.845641 |
| 128 | 8 | 34.231453 | 233.703196 | 3.450195 | 806.321670 | 367.156113 | 223/255 | 0.601971 | 0.851196 |

The jobs ran sequentially in `13m42s`, `13m58s`, and `15m22s`. Engine
initialization took `614.525s`, `612.818s`, and `642.487s`. Every row proved:

- Global ranks exactly `0..15`, four local ranks on each of four nodes.
- Fabric `ClusterUUID`
  `8fe56262-d2bb-4602-b338-8898d34c4731`, `CliqueId` `32766`, and
  `Completed`/`Success` on all 16 GPUs.
- One context-only full-local-batch prefill, zero mixed iterations, and 256
  consecutive full-local-batch decode rounds.
- Matching successful controller result and atomic completion record.
- Strict marker parsing with zero malformed records. GBS16/64/128 contained
  256, 244, and 352 valid marker records, and all required lifecycle events
  covered ranks `0..15`.
- Runtime `max_num_tokens` of 8192, 32768, and 65536. GBS128 alone capped
  TRT's synthetic context tuning request to 32768 while preserving its real
  local-batch-8, 65536-token prefill.

Minimum observed free memory was 51980 MiB at GBS16, 34114 MiB at GBS64, and
7932 MiB at GBS128. Cleanup-time TCPStore and Slurm cancellation warnings
occur after the successful result/completion pair is published and the host
intentionally terminates the remaining external MPI management world; they
are not benchmark failures.

### Why PR #1689 Reports 2072 tok/s/GPU

Reference:

```text
PR: https://github.com/SemiAnalysisAI/InferenceX/pull/1689
run: https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27164980476/attempts/14
row: D(tp16/ep16/dptrue/nw1), P(tp4/ep4/dptrue/nw6), conc=666
recipe: ctx6dep4_gen1dep16_batch32_eplb384_mtp3.yaml
```

The PR row and fixed GBS128 row use the same image, TP16/EP16 decode topology,
MTP3, load balancer, and GB300 fabric, but they do not measure the same
operating point:

| Property | Offline GBS128 | PR #1689 TP16 serving |
|---|---:|---:|
| Prefill GPUs | 0 separate | 24 |
| Decode GPUs | 16 | 16 |
| Local decode batch | exactly 8 | mostly 24-32 |
| Global decode population | exactly 128 | about 401 active, capacity 512 |
| Scheduler | non-overlap, fixed batch | overlap, continuous batch |
| Input/output lengths | exactly 8192 / 256 measured rounds | average 7378 / 922 tokens |
| Reported output tok/s/decode-GPU | 806.32 | 2072.16 |
| Output tok/s across all allocated GPUs | 806.32 | 828.86 |

The PR produced 33154.54 output tok/s. `utils/process_result.py` divides
disaggregated output throughput by `decode_gpus`, so its public 2072.16 value
does not charge the 24 prefill GPUs. Charging all 40 GPUs gives 828.86
tok/s/GPU, 1.02796x the offline GBS128 value.

The other final 8k/1k PR rows are not closer comparisons:

| Decode topology | Prefill + decode GPUs | Concurrency | Output/decode-GPU | Output/all-GPU |
|---|---:|---:|---:|---:|
| TP32/EP32, MTP3 | 16 + 32 | 333 | 676.76 | 451.18 |
| TP16/EP16, MTP3 | 24 + 16 | 666 | 2072.16 | 828.86 |
| TP8/EP8, MTP1 | 48 + 8 | 4301 | 9686.74 | 1383.82 |

The TP8 row's resolved config has `max_batch_size=512` per attention-DP rank,
CUDA graphs through batch 512, and `max_draft_len=1`. Its client TPOT implies
about 3438 active decode requests globally, or 430/rank. Do not compare its
9686.74 output tok/s/decode-GPU with local-batch-8 MTP3.

The TP16 row's client mean TPOT was 12.107 ms/output token. Little's law
therefore gives about 401 active decode requests, or 25.1 per attention-DP
rank, which agrees with the engine log spending most measured iterations in
its batch-32 CUDA graph. The offline equivalent output TPOT is lower at 9.922
ms/output token, but it has only 128 active requests. At the offline round
latency and MTP yield, matching 2072.16 output tok/s/decode-GPU requires about
GBS329.

The steady engine rows with 24-30 actual scheduled requests averaged 27.73
requests/rank and 32.49 ms device step time. They reconstruct the serving
headline:

```text
27.73 / 0.03249 = 853.3 request-rounds/s/decode-GPU
2072.16 / 853.3 = 2.428 output tokens/request-round
```

The offline row is `8 / 0.03423 = 233.7` request-rounds/s/GPU and then
multiplies by its observed 3.450 tokens/round to produce 806.32. The PR's
2.57x output-throughput lead is therefore primarily its 3.47x larger local
decode population, partly offset by lower output yield.

The engine log exposes one smaller timing clue that deserves a separate
diagnostic, but not an apples-to-apples result. Iterations selecting the
batch-8 CUDA graph averaged 27.67 ms of device step time, but graph padding was
enabled and most had only 6-8 real scheduled requests. The 13 measured rows
with exactly 8 scheduled requests and that graph averaged 28.29 ms. The
offline row reports 34.23 ms `iterLatencyMS`. Overlap-mode device step time is
GPU forward timing, while non-overlap `iterLatencyMS` covers the complete
executor loop, so these values cannot establish a fixed-batch performance
gap. Huawei synchronizes and times only its model decode calls. If the pinned
TRT image can expose per-iteration GPU-forward timing without changing the
schedule, record it as a secondary `gpu_forward_round_tpot_ms`; do not replace
the validated full-loop headline or enable overlap in the Huawei-style path.

Exact flat renderer rows, sorted by GBS for readability:

```json
[
  {
    "conc": 16,
    "decode_dp_attention": true,
    "decode_ep": 16,
    "decode_num_workers": 0,
    "decode_round_tpot_ms": 25.92126528422038,
    "decode_step_tput_per_gpu": 38.57836371161835,
    "decode_tp": 16,
    "disagg": false,
    "framework": "trt",
    "global_batch_size": 16,
    "hw": "gb300-nv",
    "image": "nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1",
    "infmax_model_prefix": "dsv4",
    "is_multinode": true,
    "isl": 8192,
    "local_batch_size": 1,
    "mean_e2el": 9.530830375020741,
    "mean_intvty": 115.88578786810356,
    "mean_tpot": 0.008629185842341245,
    "mean_ttft": 0.5109100000117905,
    "measured_decode_rounds": 256,
    "median_e2el": 9.67358000006061,
    "median_tpot": 0.008602623511662874,
    "median_ttft": 0.5085340000223368,
    "model": "deepseek-ai/DeepSeek-V4-Pro",
    "num_decode_gpu": 16,
    "num_prefill_gpu": 16,
    "observed_tokens_per_step": 3.00390625,
    "osl": 1025,
    "output_tput_per_gpu": 115.88578786810356,
    "p90_e2el": 11.246076000039466,
    "p90_tpot": 0.008787840209484721,
    "p90_ttft": 0.5087775000138208,
    "p99_e2el": 11.329238049976993,
    "p99_tpot": 0.00907306566970091,
    "p99_ttft": 0.5418207500479184,
    "precision": "fp4",
    "prefill_dp_attention": true,
    "prefill_ep": 16,
    "prefill_num_workers": 0,
    "prefill_tp": 16,
    "spec_decoding": "mtp",
    "tput_per_gpu": 115.88578786810356
  },
  {
    "conc": 64,
    "decode_dp_attention": true,
    "decode_ep": 16,
    "decode_num_workers": 0,
    "decode_round_tpot_ms": 32.375173729683276,
    "decode_step_tput_per_gpu": 123.5514605542514,
    "decode_tp": 16,
    "disagg": false,
    "framework": "trt",
    "global_batch_size": 64,
    "hw": "gb300-nv",
    "image": "nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1",
    "infmax_model_prefix": "dsv4",
    "is_multinode": true,
    "isl": 8192,
    "local_batch_size": 4,
    "mean_e2el": 12.287708937481511,
    "mean_intvty": 108.40916729296376,
    "mean_tpot": 0.00922431215893035,
    "mean_ttft": 1.9545068906263623,
    "measured_decode_rounds": 256,
    "median_e2el": 11.869209499913268,
    "median_tpot": 0.009235091254521426,
    "median_ttft": 1.9454935000976548,
    "model": "deepseek-ai/DeepSeek-V4-Pro",
    "num_decode_gpu": 16,
    "num_prefill_gpu": 16,
    "observed_tokens_per_step": 3.509765625,
    "osl": 1025,
    "output_tput_per_gpu": 433.63666917185503,
    "p90_e2el": 14.759949199971743,
    "p90_tpot": 0.009364104018850863,
    "p90_ttft": 1.9460564998677001,
    "p99_e2el": 18.62660663995193,
    "p99_tpot": 0.009456872021946299,
    "p99_ttft": 2.092686970019713,
    "precision": "fp4",
    "prefill_dp_attention": true,
    "prefill_ep": 16,
    "prefill_num_workers": 0,
    "prefill_tp": 16,
    "spec_decoding": "mtp",
    "tput_per_gpu": 433.63666917185503
  },
  {
    "conc": 128,
    "decode_dp_attention": true,
    "decode_ep": 16,
    "decode_num_workers": 0,
    "decode_round_tpot_ms": 34.231453198488516,
    "decode_step_tput_per_gpu": 233.70319552642417,
    "decode_tp": 16,
    "disagg": false,
    "framework": "trt",
    "global_batch_size": 128,
    "hw": "gb300-nv",
    "image": "nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1",
    "infmax_model_prefix": "dsv4",
    "is_multinode": true,
    "isl": 8192,
    "local_batch_size": 8,
    "mean_e2el": 14.699438031249883,
    "mean_intvty": 100.79020871519246,
    "mean_tpot": 0.009921598662681075,
    "mean_ttft": 4.08818702345161,
    "measured_decode_rounds": 256,
    "median_e2el": 14.431192999938503,
    "median_tpot": 0.009928362382359184,
    "median_ttft": 4.070200000074692,
    "model": "deepseek-ai/DeepSeek-V4-Pro",
    "num_decode_gpu": 16,
    "num_prefill_gpu": 16,
    "observed_tokens_per_step": 3.4501953125,
    "osl": 1025,
    "output_tput_per_gpu": 806.3216697215396,
    "p90_e2el": 16.88335309997201,
    "p90_tpot": 0.01001901159602321,
    "p90_ttft": 4.072401500120759,
    "p99_e2el": 17.906867700079456,
    "p99_tpot": 0.010085205285433767,
    "p99_ttft": 4.3624957999400795,
    "precision": "fp4",
    "prefill_dp_attention": true,
    "prefill_ep": 16,
    "prefill_num_workers": 0,
    "prefill_tp": 16,
    "spec_decoding": "mtp",
    "tput_per_gpu": 806.3216697215396
  }
]
```

## Final Validated B300 Run

Workflow:

```text
run: 27493336994
url: https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27493336994
git: 9796f5d17c96ab56136b8b9b1e196b6e6db84426
TRT source: c185066
image: ghcr.io#semianalysisai/trtllm-deepseek-v4:feat-deepseek_v4-c185066
```

Renderer:

```text
https://inferencemax-r4i4xgna4-semianalysisai.vercel.app/inference?unofficialrun=27493336994
```

An unauthenticated `curl` to the renderer returns HTTP 401. Open it in an
authenticated browser session. The run has a `results_bmk` artifact whose
`agg_bmk.json` is a top-level array of the flat rows below.

| GBS | Local/rank | Round TPOT ms | Steps/s/GPU | Tok/step | Output tok/s/GPU | Wall tok/s/GPU | Retained |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 2 | 22.069890 | 90.621203 | 3.001953 | 272.040604 | 198.178843 | 251/255 |
| 64 | 8 | 32.140069 | 248.910481 | 3.507324 | 873.009760 | 378.422175 | 251/255 |
| 128 | 16 | 36.831497 | 434.410801 | 3.298096 | 1432.728396 | 143.915720 | 253/255 |

Every row proved one context-only full-local-batch prefill, zero mixed
context/generation iterations, and 256 consecutive full-local-batch decode
rounds. GBS128 additionally proved attention-workspace reservation, the
12 GiB KV reserve, and 131072-row FP8 chunk completion on ranks `0..7` with
no chunk errors. Final GPU telemetry observed minimum free memory of
42714 MiB at GBS16, 1742 MiB at GBS64, and 268 MiB at GBS128.

Exact flat renderer rows, sorted by GBS for readability:

```json
[
  {
    "conc": 16,
    "decode_dp_attention": true,
    "decode_ep": 8,
    "decode_num_workers": 0,
    "decode_round_tpot_ms": 22.069890185656302,
    "decode_step_tput_per_gpu": 90.62120305881011,
    "decode_tp": 8,
    "disagg": false,
    "framework": "trt",
    "global_batch_size": 16,
    "hw": "b300",
    "image": "ghcr.io#semianalysisai/trtllm-deepseek-v4:feat-deepseek_v4-c185066",
    "infmax_model_prefix": "dsv4",
    "is_multinode": false,
    "isl": 8192,
    "local_batch_size": 2,
    "mean_e2el": 8.506522687501274,
    "mean_intvty": 136.02030185682727,
    "mean_tpot": 0.0073518437053064585,
    "mean_ttft": 1.1691594375006389,
    "measured_decode_rounds": 256,
    "median_e2el": 8.623861999993096,
    "median_tpot": 0.00730873162512199,
    "median_ttft": 1.1679470000017318,
    "model": "deepseek-ai/DeepSeek-V4-Pro",
    "num_decode_gpu": 8,
    "num_prefill_gpu": 8,
    "observed_tokens_per_step": 3.001953125,
    "osl": 1025,
    "output_tput_per_gpu": 272.04060371365455,
    "p90_e2el": 9.453417500000796,
    "p90_tpot": 0.007495053650577423,
    "p90_ttft": 1.1728185000029043,
    "p99_e2el": 10.272390650002489,
    "p99_tpot": 0.007593575591147528,
    "p99_ttft": 1.1783019500035152,
    "precision": "fp4",
    "prefill_dp_attention": true,
    "prefill_ep": 8,
    "prefill_num_workers": 0,
    "prefill_tp": 8,
    "spec_decoding": "mtp",
    "tput_per_gpu": 272.04060371365455
  },
  {
    "conc": 64,
    "decode_dp_attention": true,
    "decode_ep": 8,
    "decode_num_workers": 0,
    "decode_round_tpot_ms": 32.140068798901076,
    "decode_step_tput_per_gpu": 248.91048149447437,
    "decode_tp": 8,
    "disagg": false,
    "framework": "trt",
    "global_batch_size": 64,
    "hw": "b300",
    "image": "ghcr.io#semianalysisai/trtllm-deepseek-v4:feat-deepseek_v4-c185066",
    "infmax_model_prefix": "dsv4",
    "is_multinode": false,
    "isl": 8192,
    "local_batch_size": 8,
    "mean_e2el": 15.425251343747732,
    "mean_intvty": 109.1262200057867,
    "mean_tpot": 0.009163700529047669,
    "mean_ttft": 4.942900421874583,
    "measured_decode_rounds": 256,
    "median_e2el": 15.344337999995332,
    "median_tpot": 0.009183670972260893,
    "median_ttft": 4.937785499962047,
    "model": "deepseek-ai/DeepSeek-V4-Pro",
    "num_decode_gpu": 8,
    "num_prefill_gpu": 8,
    "observed_tokens_per_step": 3.50732421875,
    "osl": 1025,
    "output_tput_per_gpu": 873.0097600462936,
    "p90_e2el": 17.614075799973214,
    "p90_tpot": 0.009266875239280244,
    "p90_ttft": 4.976919700019062,
    "p99_e2el": 21.29510529000894,
    "p99_tpot": 0.009333493034769593,
    "p99_ttft": 4.982044999995269,
    "precision": "fp4",
    "prefill_dp_attention": true,
    "prefill_ep": 8,
    "prefill_num_workers": 0,
    "prefill_tp": 8,
    "spec_decoding": "mtp",
    "tput_per_gpu": 873.0097600462936
  },
  {
    "conc": 128,
    "decode_dp_attention": true,
    "decode_ep": 8,
    "decode_num_workers": 0,
    "decode_round_tpot_ms": 36.83149673250824,
    "decode_step_tput_per_gpu": 434.4108010652217,
    "decode_tp": 8,
    "disagg": false,
    "framework": "trt",
    "global_batch_size": 128,
    "hw": "b300",
    "image": "ghcr.io#semianalysisai/trtllm-deepseek-v4:feat-deepseek_v4-c185066",
    "infmax_model_prefix": "dsv4",
    "is_multinode": false,
    "isl": 8192,
    "local_batch_size": 16,
    "mean_e2el": 105.22999424999762,
    "mean_intvty": 89.54552477401855,
    "mean_tpot": 0.011167503931923442,
    "mean_ttft": 93.36035996875306,
    "measured_decode_rounds": 256,
    "median_e2el": 105.04664299997967,
    "median_tpot": 0.011193058794137243,
    "median_ttft": 93.350035499956,
    "model": "deepseek-ai/DeepSeek-V4-Pro",
    "num_decode_gpu": 8,
    "num_prefill_gpu": 8,
    "observed_tokens_per_step": 3.298095703125,
    "osl": 1025,
    "output_tput_per_gpu": 1432.7283963842967,
    "p90_e2el": 108.02184229998383,
    "p90_tpot": 0.011256818371085943,
    "p90_ttft": 93.43148689994122,
    "p99_e2el": 112.4452570800099,
    "p99_tpot": 0.01132884211451625,
    "p99_ttft": 93.4409581200045,
    "precision": "fp4",
    "prefill_dp_attention": true,
    "prefill_ep": 8,
    "prefill_num_workers": 0,
    "prefill_tp": 8,
    "spec_decoding": "mtp",
    "tput_per_gpu": 1432.7283963842967
  }
]
```

## Debug Checklist

1. Inspect `Show result headline`.
2. Download `offline-trt-job-ID`.
3. Read `offline_controller_ID.log`.
4. For PR-max, read `offline_timing_ID.log` and confirm the selected rank-0
   host-step rows are present. Use `offline_world_ID.log` for all-rank
   launch/runtime context.
5. Extract `offline_debug_ID.tar.gz`, then inspect `worker_result.json` and
   `worker.log`. For rack runs, the archive instead contains
   `.offline_rack_ID_JOB/replicas/r00` through `r08`; begin with each
   `host_launcher.log`, then the failed child's world/timing/completion files.
   Extract that child's nested `offline_debug_*.tar.gz` for worker files.
6. For rack runs, require nine child result files, barrier-ready indices
   `0..8`, one release record, matching child fabric UUID/clique values, and
   top-level timing source
   `slowest_replica_trt_print_iter_log_host_step_time`.
   Each child must log `validated rack synchronization environment` before
   engine start, and each controller must log the same validation before
   spawning `trt_worker.py`.
7. Query iteration stats:

   ```bash
   jq -r '
     .[] |
     [
       .iter,
       .iterLatencyMS,
       .numActiveRequests,
       .numQueuedRequests,
       .inflightBatchingStats.numContextRequests,
       .inflightBatchingStats.numGenRequests,
       .inflightBatchingStats.numScheduledRequests
     ] | @tsv
   ' measured_iteration_stats.json | head -40
   ```

8. Confirm rank markers are exactly `0..world_size-1`.
9. On GB300, inspect `offline_rank_map_ID.tsv` and
   `offline_topology_ID.log` before debugging TRT. PR-max expects 8, 16, or
   32 ranks according to profile. Rack expects 72 top-level ranks and nine
   independent child maps of eight ranks. A rank-placement or fabric failure
   is infrastructure, not a benchmark result.
10. For PR-max, require
   `timing_source=trt_print_iter_log_host_step_time`, inspect setup counts,
   and verify the selected range is 256 consecutive full-batch rows. Also
   require `perfect_router.enabled=false`, no perfect-router alias on any
   rank, and `resolved_parallelism.attention_dp_configured=false`.
11. On B300 GBS128, require `fp8_prefill_gemm_chunked` on ranks `0..7` for
   131072 rows, two 65536-row chunks, and synchronized chunks.
12. Check GPU telemetry, including `memory.used` and `memory.free`, for OOM,
   idle prefill gaps, or a hung rank.

Do not weaken the fixed-batch gate. Infrastructure errors may be retried on a
fresh node; capacity or schedule failures require an implementation fix.
