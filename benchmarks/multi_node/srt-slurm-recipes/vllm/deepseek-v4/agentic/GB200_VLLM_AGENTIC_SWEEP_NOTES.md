# GB200 DeepSeek V4 vLLM Disaggregated Agentic Sweep Notes

This is the engineering log for the GB200 DeepSeek V4 Pro NVFP4 vLLM
disaggregated agentic sweep. It records configuration changes, official runs,
observations, and the evidence behind each decision. It must be updated as the
sweep progresses.

## Target

- Model: `deepseek-ai/DeepSeek-V4-Pro` (NVFP4 checkpoint)
- Hardware: GB200, four GPUs exposed per Slurm node
- Runtime: vLLM behind Dynamo disaggregated serving
- Workload: `semianalysis_cc_traces_weka_061526`
- Required result: successful official GitHub Actions artifacts covering
  multiple concurrency points and multiple prefill/decode topologies
- Baseline: single-node B200 aggregate vLLM results from the InferenceX results
  database

The workload has very long contexts and approximately 96% theoretical prefix
reuse. Fixed-sequence recipes are not valid starting points because they use
much shorter prompts and usually disable prefix caching.

## Methodology

1. Bring up a low-cost topology and prove model loading and cross-node NIXL
   transfer.
2. Validate KV event ingestion and request transport independently of
   performance.
3. Compare P/D balance at a constant 40-inference-GPU budget:
   `4P/1D`, `3P/2D`, `2P/3D`, and `1P/4D`.
4. Use c64 as a topology gate. Retain c32/c64 for the decode-starved 4P/1D
   comparison, and run the dense c16/c24/c32/c40/c48/c56/c64/c72/c80/c96
   curve on the rate-matched 3P/2D topology. An official c192 point supplies
   the hard-overload check outside the final dense grid.
5. Compare total and per-GPU throughput, TTFT, TPOT, E2E latency, request
   errors, and measured cache reuse against the B200 aggregate baseline.

Each P or D worker is TP8/TEP8 and spans two four-GPU Slurm nodes. Every
constant-budget topology therefore uses ten inference nodes plus one dedicated
NATS/etcd node.

## Configuration Changes

### Initial TP8 disaggregation (`444371fb`)

- Added a GB200 agentic vLLM configuration using vLLM `v0.23.0`.
- Used TP8/TEP8 workers because the current NVFP4 checkpoint is not supported
  by the older vLLM images used by the fixed-sequence recipes.
- Kept prefix caching enabled and used the full 061526 agentic trace.

### KV-aware 4P/1D topology (`76ad0903`)

- Added four cache-affinitized prefill replicas and one decode replica.
- Enabled Dynamo KV routing and explicit vLLM KV-event publication.
- Ensured each two-node TP8 prefill replica had one internally consistent and
  externally distinct NIXL engine ID.

### TEP8 correction (`b56d0d9e`)

- Changed prefill model sharding from TEP4 to TEP8.
- Evidence: official run `27732316539` reached 182.54 GiB per GPU and failed
  while requesting another 1.97 GiB during MoE weight construction.
- Conclusion: TEP4 cannot load this checkpoint on the available GB200 nodes.

### Dynamo/vLLM compatibility and request transport (`1e29f559`)

- Updated Dynamo from `1.2.0.dev20260426` to
  `1.2.0.dev20260526`.
- Set `DYN_REQUEST_PLANE=tcp` for frontend, prefill, and decode workers.
- Kept KV events on their separate ZMQ-to-NATS event path.
- Evidence for the Dynamo change: every vLLM v0.23 `BlockStored` event failed
  to decode with the April 26 wheel. The compatible trailing-field parser was
  added upstream on April 29 and is present in the May 26 wheel.
- Evidence for TCP: the earlier c64 canary returned HTTP 503 with
  `Rejecting request: all workers are busy` from the NATS request plane.
  Dynamo documents TCP as its fastest request plane.
- The apparent `DYN_VLLM_KV_EVENT_PORT=5200+` versus recipe port `20080`
  mismatch was investigated and disproved: vLLM and its colocated Dynamo
  worker correctly communicate over local `tcp://127.0.0.1:20080`.

### P/D topology grid (`74dfa0bb`)

- Added constant-40-GPU recipes for `3P/2D`, `2P/3D`, and `1P/4D` alongside
  `4P/1D`.
- Added c32/c64/c128/c192 to every topology in
  `.github/configs/nvidia-master.yaml`.
- Multi-decode recipes use generated vLLM v0.23 engine IDs. This avoids ID
  collisions across decode replicas while vLLM synchronizes the generated ID
  across the two nodes of each TP8 replica.

### GB200 NIXL VMM registration (`84be7898`)

- Enabled vLLM's CuMem allocator on every prefill and decode worker in all
  four topology recipes.
- Evidence: the corrected 4P/1D run transferred hundreds of MiB per request,
  but NIXL throughput degraded to roughly 14--25 MiB/s under load. P90 transfer
  latency reached 50--92 seconds, and prefill workers released hundreds of
  expired transfer leases; 495 expiration messages reported that zero remote
  workers had retrieved the blocks.
- vLLM 0.23 documents that GB-series multi-node NVLink requires VMM-backed KV
  cache registration via `--enable-cumem-allocator` or sleep mode together
  with `UCX_CUDA_IPC_ENABLE_MNNVL=y`. The environment variable was present,
  but the allocator flag was missing from the four new topology recipes.
- The existing fixed-sequence DeepSeek V4 GB200 recipes enable sleep mode,
  which implicitly enables the same CuMem allocator. The explicit allocator
  flag is used here because the benchmark does not need sleep/wake behavior.
- `kv_buffer_size` was not changed: in vLLM 0.23 it does not size the
  `NixlConnector` transfer path, so changing it would not address this fault.

### CuMem load failure and RDMA transport

- Official run `27738234911` tested all four c64 topologies with the CuMem
  allocator enabled and a scenario-compliant 900-second duration.
- Every topology failed during TP8 decode model initialization before serving
  requests. FlashInfer NVFP4 MoE weight conversion requested 20 MiB with only
  11.5--11.6 MiB free on most ranks. vLLM reported 183.98 GiB in use per GPU,
  including 68.79 GiB in the CuMem private pool.
- The successful non-CuMem decode from `27734909066` loaded the same model in
  107.98 GiB/GPU and subsequently allocated a 9.16-million-token KV cache.
- CuMem is therefore not viable for this TP8 NVFP4 topology in vLLM 0.23. The
  recipes now use the cluster's UCX reliable-connection RDMA transport
  (`UCX_TLS=cuda_copy,rc`) for NIXL instead. This transport is already used by
  the repository's GB200 vLLM disaggregated recipes for other models and does
  not require VMM-backed allocations.
- TCP is intentionally omitted so an unavailable RDMA transport fails visibly
  instead of silently reproducing the 14--25 MiB/s data path.

### Direct-cluster RDMA canary and registration-cache limit

- Manual Slurm job `19243` launched the 2P/3D c64 topology for 900 seconds
  from commit `7979c6b1` on eleven Watchtower nodes. A two-node host probe
  first confirmed six `mlx5` RDMA devices per node. The worker launch logs
  confirm `UCX_TLS=cuda_copy,rc` with no TCP fallback. This job is a transport
  canary only; official results still require GitHub Actions artifacts.
- Job `19243` loaded all five model replicas, instantiated UCX on every NIXL
  worker, registered the GPU KV caches, and reached healthy status without an
  OOM or transport error. It then exited before sending requests because the
  dedicated manual checkout had not initialized the `utils/aiperf` submodule;
  `uv` correctly rejected the empty directory as not being a Python project.
  GitHub Actions checks out submodules recursively, so this was a manual-canary
  harness error rather than a recipe or production workflow defect. The rerun
  initializes submodules explicitly.
- During initialization, vLLM 0.23 warned on every decode process that Dynamo
  had imported NIXL before vLLM could set `UCX_RCACHE_MAX_UNRELEASED=1024`.
  The repository's existing vLLM disaggregated recipes set this variable on
  both prefill and decode workers for the same reason. All four GB200 agentic
  topology recipes now set it explicitly so UCX does not retain an unbounded
  number of released memory registrations during long, high-churn KV runs.
- Message capacity is already separated by plane: the recipes raise NATS
  `max_payload` to 32 MiB for long agentic control messages, vLLM's KV-event
  publisher reports a 100,000-event high-water mark and queue, and bulk KV
  bytes travel through NIXL/UCX rather than NATS. No truncation or payload-too-
  large evidence has appeared in job `19243`; additional buffer tuning will be
  evidence-driven rather than applied speculatively.

### Sliding-window prefix retention

- The first successful router-only affinity gate (`27794559671`, 3P/2D job
  `19266`) proved that `bind` worked: 1,244 bindings and 1,012 sticky-session
  hits were logged, with no request errors. However, final per-prefill vLLM
  prefix-hit rates remained only 10.7--12.5%, well below the 93.5% B200
  reference and the trace's roughly 96% theoretical reuse.
- The validated B200 and B300 DSv4 vLLM agentic launchers both set
  `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=32768`. In vLLM v0.23 this keeps
  sparse long-lived replay boundaries for the model's sliding-window KV group
  instead of allowing dense transient SWA entries to evict useful prefixes;
  the original single-node tuning moved measured hits from approximately 0%
  to 74% before later improvements.
- All GB200 disaggregated recipes now set the same 32k retention interval on
  both prefill and decode workers. The queued 4P/1D sibling in `27794559671`
  was cancelled before allocation so it would not repeat the known-incomplete
  server configuration.

#### Successful direct canary (`19244`)

- The corrected manual checkout initialized submodules and ran 2P/3D c64 for
  900 profiling seconds from commit `193692e7`. Slurm completed successfully.
- AIPerf completed 489/489 profiled requests with no recorded request errors.
  The 489 records covered 239 conversations; 112 conversations completed at
  least two consecutive turns and the longest completed five, so this run
  reached genuine multi-turn behavior rather than the all-first-turn state of
  the earlier 300-second run.
- NIXL/UCX was healthy throughout. Representative 146 MiB--1.4 GiB transfers
  sustained roughly 13--43 GB/s with typical average transfer latency of
  8--35 ms. There were no expired producer leases, failed notifications,
  invalid-block reports, HTTP 503s, NIXL/UCX errors, or payload truncations
  during profiling. The one frontend prefill-disconnect error occurred during
  the post-duration drain/worker teardown and is not present in the 489
  profiled records.
- Performance was 48,811 total tok/s (48,323 input + 488 output), 1,220
  tok/s/GPU, 187.8 s mean TTFT, 13.34 ms mean TPOT, and 200.1 s mean E2E.
  Transport is no longer the bottleneck, but this 2P/3D point is still below
  the B200 aggregate c64 baseline of 81,863 tok/s and 13.85 s mean TTFT.
- Final vLLM prefix-hit counters were only about 5.6% and 2.1% on the two
  prefills despite 96.9% theoretical reuse. Dynamo did route some requests
  with nonzero effective cached blocks, while many selections still reported
  zero. The official c64 topology gate is therefore required to determine
  whether more prefill replicas recover throughput; a green transport canary
  alone is not treated as the final performance result.

### Slurm job-name prefix (branch-only historical workaround)

- This branch prefixes GB200 Slurm jobs with `ifx-` in
  `runners/launch_gb200-nv.sh`.
- It is not standard on `main`.
- It was introduced after jobs `18593` and `18599` were reportedly cancelled
  by another Watchtower runner fleet that reused the `gb200-nv_N` names.
- This naming change does not alter topology or benchmark behavior.

## Validation

- `python -m pytest utils/matrix_logic/ -q`: 158 passed.
- All four topology recipes parse as YAML and were checked for:
  - 40 inference GPUs plus one infrastructure node;
  - matching vLLM image and Dynamo wheel;
  - KV routing enabled;
  - TCP request plane on all components;
  - collision-free generated engine IDs for multi-decode topologies.
- The final sweep generator emits 12 points: 4P/1D at c32/c64, plus 3P/2D at
  c16/c24/c32/c40/c48/c56/c64/c72/c80/c96. The 2P/3D and 1P/4D recipes remain
  checked in as topology-gate evidence but are excluded because they were
  strongly prefill-starved at c64. High-concurrency 4P/1D points are excluded
  because the affinity-correct c64 gate proved that its single decode replica
  is already saturated. The successful c192 artifact is retained as overload
  evidence rather than spending more runs on dominated c128+ points.

## Official Runs

| Run | Configuration | Outcome | Key evidence |
| --- | --- | --- | --- |
| `27728896563` | 1P/1D, c32/c64/c128/c192 | Partially green, later cancelled | c64: about 26.7k tok/s total, 423s mean TTFT, about 5% cache hit; not competitive |
| `27732316539` | TEP4 prefill attempt | Failed | Model-load OOM at 182.54 GiB/GPU plus 1.97 GiB allocation |
| `27732541012` | 4P/1D c64, old Dynamo/NATS | Failed warmup | 45/85 errors; empty router indexes; KV-event decode errors; HTTP 503 overloads |
| `27734909066` | 4P/1D c64, new Dynamo/TCP | Success | Official artifact; clean 85/85 warmup and 99/99 profiled requests |
| `27737167704` | c64 topology sweep | Cancelled before allocation | Server-log audit found the four recipes lacked VMM-backed KV registration required for GB200 multi-node NVLink |
| `27738234911` | c64 topology sweep, VMM enabled | Failed | All four decode workers OOM during NVFP4 weight conversion before serving; no benchmark results |
| `27770234988` | c64 topology sweep, RDMA + bounded registration cache | Success | All four 900-second topology jobs and aggregate/raw/server-log artifacts completed successfully |
| `27785852838` | selected 4P/1D + 3P/2D curves, c32 | Cancelled | 4P/1D completed, but the run was stopped after proving AIPerf omitted Dynamo `nvext.session_control` |
| `27785854604` | selected 4P/1D + 3P/2D curves, c128/c192 | Cancelled | Stopped for the same missing conversation-binding metadata; three matrix jobs had separately failed checkout before using GPUs |
| `27790985904` | attempted affinity-enabled 4P/1D + 3P/2D c64 gate | Cancelled / invalid | The May 26 Dynamo wheel rejected all AIPerf `session_control.action=bind` warmup requests; 4P/1D Slurm job `19259` had 85/85 errors and 3P/2D job `19260` was cancelled pending |
| `27794559671` | corrected-Dynamo 3P/2D c64 gate | 3P/2D success; run cancelled before 4P/1D | Job `19266` completed 657 profiled requests with zero errors and 10--38 GB/s KV transfers; affinity worked, but missing SWA retention limited final local hits to 10.7--12.5% |
| `27798151112` | retention-corrected 4P/1D + 3P/2D c64 gate | Cancelled externally after allocation | The 4P/1D job `19272` started with `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=32768`; 3P/2D job `19273` was pending. GitHub cancelled both matrix jobs at 03:30 UTC, after 97 and 68 minutes respectively. Neither the workflow's 480-minute job timeout nor Slurm's eight-hour limit was reached, so this run produced no valid performance artifact. Both orphaned Slurm jobs were explicitly cancelled. |
| `27804547383` | incorrectly broad c64 dispatch | Cancelled before agentic allocation | The broad model/framework/runner filter also selected seven fixed-sequence GB200 configurations. The dispatch was cancelled immediately; no agentic Slurm job started and no benchmark result from this run is used. |
| `27804604959` | exact-key retention-corrected c64 gate | 3P/2D success; 4P/1D cancelled externally | Slurm job `19278` completed successfully with official aggregate, raw, and server-log artifacts. GitHub cancelled the run while 4P/1D job `19279` was loading; that orphan was cancelled. |
| `27809946853` | isolated retention-corrected 4P/1D c64 gate | Success | Job `19392` completed with official aggregate, raw, server-log, and collected-result artifacts. Cache affinity worked, but the single decode replica saturated badly. |
| `27815458708` | initial 10-point wide frontier | Intentionally stopped after four successful points | Official successes: 3P/2D c64, c96, c192 and 4P/1D c32. c96 and c192 were already deeply overloaded, so pending c128/c160/c256/c384 and the duplicate 4P/1D c64 were cancelled before spending additional GPU time. |
| `27834000342` | final 12-point dense sweep | Success | Attempt 2 completed all 12 points with aggregate, raw, and server-log artifacts. The workflow concluded successfully with no failed or cancelled jobs. |

### Unexpected cancellation of the first retention-corrected gate

- Run `27798151112` was cancelled at the workflow level while its two matrix
  jobs were in `Launch multi-node job script`. GitHub records `cquil11` as the
  dispatching and triggering actor but does not expose a separate cancellation
  actor in the run API.
- This was not a benchmark timeout: the 4P/1D matrix job had run for about 97
  minutes, the 3P/2D matrix job for about 68 minutes, and the reusable
  multi-node workflow allows 480 minutes.
- Slurm job `19272` had allocated its eleven requested nodes and its launch log
  confirmed the 32k retention environment. Job `19273` was still waiting for
  resources. GitHub's cancellation left the Slurm allocations orphaned, so
  both were cancelled manually before the gate was resubmitted.
- The cancelled self-hosted jobs later resurfaced from the runner work queue as
  Slurm jobs `19280` and `19281`, despite the GitHub run already being marked
  completed/cancelled. Their process environments identified
  `GITHUB_RUN_ID=27798151112`. Both were cancelled before allocation, and the
  stale runner workers then exited; no jobs from the active gate were touched.

### Retention-corrected 3P/2D c64 result (`27804604959`, job `19278`)

- Official aggregate: 4,145/4,145 successful requests, 269,251 total tok/s
  (267,172 input + 2,078 output), 6,731 tok/s per inference GPU, 3.27s mean
  TTFT, 20.95ms mean TPOT, and 22.49s mean E2E.
- The three prefill engines finished at 92.9%, 92.3%, and 94.9% vLLM prefix
  cache hit rate. This is the expected steady-state cache regime and confirms
  that `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=32768` fixed the prior
  10.7--12.5% result.
- Dynamo logged 7,384 affinity bindings and 7,202 sticky-session/cache-aware
  selections. There were no bind failures or unsupported session actions.
- Across 394 sampled decode log windows, NIXL transferred at 7.6--42.8 GB/s
  (23.3 GB/s mean). There were no expired transfer leases, NIXL/UCX errors,
  payload truncations, or NATS payload-too-large errors during the measured
  run.
- The frontend cancelled residual in-flight requests when AIPerf's fixed
  1,800-second profile and 30-second drain ended, and distributed ranks logged
  expected TCPStore disconnects during Slurm teardown. These occurred after
  measurement; the aggregate contains only successful requests.
- Compared with the B200 aggregate c64 no-offload reference (81,863 tok/s,
  13.85s mean TTFT, 82.1ms TPOT, 93.5% cache hit), this point has 3.29x system
  throughput, 4.24x lower mean TTFT, comparable cache reuse, and 3.92x lower
  TPOT. Per-GPU throughput is lower because the disaggregated system uses 40
  inference GPUs instead of eight; system throughput and latency are the
  relevant disaggregation acceptance metrics.

### Per-topology dispatch keys

- Run `27804604959` was explicitly cancelled at the workflow level at 06:34
  UTC, after its 3P/2D artifact succeeded but while 4P/1D Slurm job `19279`
  was still loading. This was not the workflow's 480-minute timeout or the
  recipe's eight-hour Slurm limit. The orphaned Slurm job was cancelled.
- The combined master entry made every exact-key dispatch create two 11-node
  matrix jobs. Because Watchtower has 18 nodes, those jobs cannot execute
  concurrently; Slurm queue time consumes the waiting GitHub job's timeout
  and a whole-run cancellation can discard the sibling still in progress.
- The master config is therefore split into
  `dsv4-fp4-gb200-dynamo-vllm-agentic-4p1d` and
  `dsv4-fp4-gb200-dynamo-vllm-agentic-3p2d`. The recipe, runtime arguments,
  result labels, 40-GPU budget, and six-point concurrency grid are unchanged.
  The split only allows each official dispatch to contain one independently
  schedulable topology point.

### Retention-corrected 4P/1D c64 result (`27809946853`, job `19392`)

- Official aggregate: 1,777/1,777 successful requests, 98,146 total tok/s
  (97,420 input + 726 output), 2,454 tok/s per inference GPU, 40.56s mean
  TTFT, 58.77ms mean TPOT, and 88.07s mean E2E.
- The four prefill engines finished at 87.9%, 85.2%, 89.3%, and 89.0% vLLM
  prefix cache hit rate. Dynamo logged 3,370 affinity bindings and 3,164
  sticky-session/cache-aware selections, with no bind errors.
- NIXL transfer throughput averaged 16.2 GB/s across 101 sampled windows and
  reached 42.1 GB/s. There were no expired leases, NIXL/UCX errors, HTTP 503s,
  payload truncations, or NATS payload-too-large errors during measurement.
- At the same c64 and 40-GPU budget, 3P/2D delivers 2.74x the system throughput,
  12.4x lower mean TTFT, and 2.81x lower mean TPOT. The 4P/1D run also slowed
  continuously as its one decode replica accumulated work. This is direct
  evidence that 4P/1D is decode-starved once cache affinity removes most
  repeated prefill work; it is retained at c32/c64 for comparison but not run
  at higher concurrency.

### Concurrency-knee refinement (`27815458708`)

- 3P/2D c96 completed 1,571 successful requests at 95,705 tok/s, 139.34s mean
  TTFT, 685.77s p95 TTFT, and 16.25ms mean TPOT. 3P/2D c192 completed 1,851
  successful requests at 95,344 tok/s, 189.97s mean TTFT, 1,209.01s p95 TTFT,
  and 15.91ms mean TPOT. Neither point recorded profile request errors.
- Both are dominated by c64 (269,251 tok/s, 3.27s mean TTFT). At high
  concurrency, too many scenario-mandated cold first turns are simultaneously
  resident; conversations progress too slowly to reach their repeated-prefix
  turns. This creates a cache/locality collapse even though Dynamo affinity,
  prefix caching, and 32k retention are enabled.
- c192 is sufficient hard-overload evidence. The pending c128/c160/c256/c384
  points were cancelled because they cannot resolve the knee and would repeat
  a dominated operating regime. The final grid instead adds eight-token steps
  from c16 through c80, plus c96, to resolve the useful frontier around c64.
- The same run's 4P/1D c32 point completed 1,678 requests at 136,675 tok/s,
  2.70s mean TTFT, 12.63s p95 TTFT, and 19.83ms mean TPOT. It is a valid
  low-latency comparison point and remains in the final grid.

### Final dense sweep (`27834000342`)

Attempt 1 completed 3P/2D c16, c32, and c56 before an intentional workflow
cancellation. Attempt 2 used GitHub's rerun-failed path, retained those green
artifacts, and completed the other nine points. Stale attempt-1 Slurm jobs
`19416`, `19420`, and `19421` were cancelled before attempt 2 allocated nodes.

| Topology | Conc. | Successful | Total tok/s | Tok/s/GPU | Mean interactivity | Mean TTFT | P95 TTFT | Mean TPOT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3P/2D | 16 | 1,058/1,058 | 77,064 | 1,927 | 71.2 tok/s | 2.26s | 13.03s | 14.22ms |
| 3P/2D | 24 | 1,515/1,515 | 114,823 | 2,871 | 66.8 tok/s | 2.61s | 12.83s | 15.19ms |
| 3P/2D | 32 | 1,804/1,804 | 147,021 | 3,676 | 60.4 tok/s | 2.64s | 12.06s | 16.71ms |
| 3P/2D | 40 | 2,362/2,362 | 172,412 | 4,310 | 55.6 tok/s | 2.51s | 11.85s | 18.23ms |
| 3P/2D | 48 | 3,039/3,039 | 213,778 | 5,344 | 53.2 tok/s | 2.66s | 12.12s | 19.00ms |
| 3P/2D | 56 | 3,546/3,546 | 237,915 | 5,948 | 50.6 tok/s | 3.23s | 13.02s | 20.09ms |
| 3P/2D | 64 | 4,099/4,099 | 264,609 | 6,615 | 48.5 tok/s | 3.66s | 13.49s | 20.91ms |
| 3P/2D | 72 | 4,052/4,052 | 259,002 | 6,475 | 47.8 tok/s | 6.91s | 32.29s | 21.26ms |
| 3P/2D | 80 | 3,809/3,809 | 268,365 | 6,709 | 47.3 tok/s | 16.19s | 63.37s | 21.49ms |
| 3P/2D | 96 | 1,601/1,601 | 101,197 | 2,530 | 62.7 tok/s | 140.56s | 728.44s | 16.24ms |
| 4P/1D | 32 | 1,676/1,676 | 136,519 | 3,413 | 50.8 tok/s | 2.66s | 12.99s | 19.89ms |
| 4P/1D | 64 | 1,752/1,752 | 96,840 | 2,421 | 30.4 tok/s | 34.48s | 184.37s | 56.31ms |

Conclusions:

- 3P/2D c64 is the practical throughput knee. c72 has lower throughput and
  worse latency; c80 gains only 1.4% throughput over c64 while mean TTFT rises
  4.4x; c96 is a hard overload collapse. The c96 interactivity metric is
  survivor-biased by the smaller set of requests that completed while prompt
  queueing drove mean TTFT to 141 seconds.
- The eight useful throughput/interactivity tradeoff points are 3P/2D
  c16/c24/c32/c40/c48/c56/c64/c80. c72 is dominated and c96 is overload
  evidence rather than an operating point.
- 3P/2D dominates 4P/1D. At c32 it delivers 7.7% more throughput with nearly
  identical TTFT. At c64 it delivers 2.73x throughput, 9.4x lower mean TTFT,
  and 2.69x lower mean TPOT. One decode replica cannot sustain this workload.
- Final local prefill cache hit rates were 91.6--95.0% through 3P/2D c64.
  They fell to 82.8--93.2% at c72/c80 and 54.2--57.4% at c96 as overload
  prevented conversations from progressing through their reusable turns.
- NIXL remained healthy at approximately 16.5--28.5 GB/s mean throughput by
  point, with peaks around 43 GB/s. There were no NIXL/UCX failures, payload
  truncations, NATS payload-too-large errors, HTTP 503s, or profiled request
  errors. Expired producer leases at c64/c72/c96 occurred in the fixed-duration
  post-profile drain/teardown window for requests absent from the successful
  aggregates; they did not indicate a measured transfer-path failure.
- Engine startup was audited directly from the cluster. Long TP8 decode loads
  continued advancing checkpoint-shard, CUDA-graph, DeepGEMM, KV-registration,
  and worker-registration logs; all workers became healthy before profiling.
  No startup was silently accepted based only on Slurm RUNNING state.
- Against the B200 aggregate c64 no-offload reference, 3P/2D c64 has 3.23x
  total system throughput, 3.78x lower mean TTFT, and 3.93x lower mean TPOT.
  It is nevertheless 35.4% lower in throughput per inference GPU: 6,615 versus
  10,233 tok/s/GPU. Forty GB200 inference GPUs deliver 3.23x the throughput of
  eight B200 GPUs, or 64.6% scaling efficiency. Disaggregation is a system
  throughput and latency win here, not a GPU-efficiency win.

### Official RDMA topology gate: completed points

- The 3P/2D c64 job in `27770234988` completed successfully with an official
  aggregate and raw artifact: 682/682 requests, 71,625 total tok/s (70,954
  input + 671 output), 1,791 tok/s/GPU, 115.8 s mean TTFT, 15.34 ms mean
  TPOT, and 129.7 s mean E2E. NIXL remained in the tens-of-GB/s range.
- This is 47% more total throughput and 72 seconds lower mean TTFT than the
  direct 2P/3D c64 canary, confirming that c64 is prefill-limited. It remains
  12.5% below the 81,863 tok/s B200 aggregate c64 baseline and has much higher
  TTFT, so the more prefill-heavy 4P/1D point remains necessary before choosing
  the final curve topologies.
- The full c64 ranking is:
  - 4P/1D: 922/922 requests, 96,043 tok/s, 2,401 tok/s/GPU, 70.9 s mean
    TTFT, 20 ms mean TPOT, and 87.8 s mean E2E;
  - 3P/2D: 682/682 requests, 71,625 tok/s, 1,791 tok/s/GPU, 115.8 s mean
    TTFT, 15 ms mean TPOT, and 129.7 s mean E2E;
  - 2P/3D: 490/490 requests, 48,443 tok/s, 1,211 tok/s/GPU, 189.3 s mean
    TTFT, 13 ms mean TPOT, and 201.5 s mean E2E;
  - 1P/4D: 250/250 requests, 23,775 tok/s, 594 tok/s/GPU, 388.9 s mean
    TTFT, 12 ms mean TPOT, and 400.3 s mean E2E.
- 4P/1D exceeds the B200 aggregate c64 throughput baseline by 17.3%, although
  its TTFT remains higher because the realized prefill cache rate is still far
  below the baseline. 3P/2D is retained as the useful decode-heavier curve.
  The 2P/3D and 1P/4D recipes remain checked in with their gate evidence, but
  their search-space entries were removed from `nvidia-master.yaml` so the
  final sweep does not spend GPU time on proven prefill-starved shapes.

### Dynamo conversation binding

- Recipe-side routing was configured correctly: the Dynamo frontend used
  `router-mode: kv`, every prefill published KV events, and vLLM prefix caching
  was enabled (none of the agentic recipes passed
  `--no-enable-prefix-caching`). However, those settings alone do not make a
  multi-turn conversation sticky.
- The pinned AIPerf branch supports `--use-dynamo-conv-aware-routing`, which
  adds Dynamo's `nvext.session_control` bind/close block to OpenAI request
  bodies using the stable conversation correlation ID. The InferenceX replay
  command originally did not enable the option. `X-Correlation-ID` is tracing
  metadata and does not establish a Dynamo session binding.
- This explains the official 4P/1D c64 evidence: the four prefill engines
  reported only 3.2--4.2% local prefix hits, token-weighted cache-read usage
  was 3.75M / 88.45M = 4.24%, and only 250/2,172 (11.5%) frontend prefill
  selections had nonzero effective cached blocks despite 96.6% theoretical
  trace reuse.
- `build_replay_cmd` now enables AIPerf's Dynamo conversation-aware routing for
  every `dynamo-*` agentic framework. Runs produced before this correction are
  retained as diagnostic transport/topology evidence but are not accepted as
  the final high-cache agentic sweep.
- Corrected high-concurrency runs use a 3,600-second Dynamo session inactivity
  timeout instead of AIPerf's 300-second default. At saturation a single long
  request plus the next inter-turn delay can exceed five minutes; allowing the
  affinity lease to expire would silently destroy the cache locality being
  measured. This setting does not change HTTP error handling or server health
  deadlines.
- Compatibility audit after run `27790985904` found that AIPerf's router-only
  `action: bind` and the pinned `1.2.0.dev20260526` frontend were incompatible:
  that Dynamo build only deserializes `open|close`, and all 85 4P/1D warmup
  requests failed before routing. `open` is not a substitute for vLLM because
  it is an engine-backed SGLang streaming-session operation; without a worker
  `session_control` endpoint it does not create router affinity.
- Dynamo added router-only `SessionAction::Bind` specifically for this case.
  The GB200 recipes now pin `1.3.0.dev20260618`, the first available cluster
  nightly containing that implementation. The same release remains newer than
  the May build already validated for vLLM v0.23 KV-event decoding.
- AIPerf's `AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID` option was also
  audited. Dynamo only added that coding-agent header on June 18, and it maps
  to passive `agent_context`; it does not populate `session_control` or create
  KV affinity. It is therefore intentionally not used as a replacement for
  `bind`.

## Corrected 4P/1D c64 Gate (`27734909066`)

Functional results:

- All four prefills and one decode worker registered.
- Warmup: 85 completed, zero errors.
- Profiling: 99 successful records, zero request errors.
- HTTP 503 responses: zero.
- KV-event decode failures: zero.
- Dynamo made 125 selections with nonzero `effective cached blocks`, proving
  that the KV index was populated and used.

Performance from `results_bmk/agg_bmk.json`:

- Total throughput: 22,573.5 tok/s.
- Per-GPU throughput: 564.3 tok/s across 40 inference GPUs.
- Output throughput: 124.2 tok/s total.
- Mean TTFT: 133.0s; p95 TTFT: 205.5s.
- Mean TPOT: 115.6ms; p95 TPOT: 209.5ms.
- Mean E2E: 176.3s; p95 E2E: 263.8s.
- Theoretical cache hit: 96.5%.
- Final vLLM GPU prefix-hit rates by prefill: approximately 1.3% to 3.5%.

Conclusion: this run proves the Dynamo compatibility and request-plane fixes,
but not a healthy NIXL data plane. It is slower than the earlier 1P/1D c64
point and far below the B200 aggregate baseline. Server logs show multi-second
to multi-minute KV transfers and expired producer leases, consistent with the
missing GB200 VMM registration. The gap between 96.5% theoretical reuse and
low single-digit measured reuse must be re-evaluated after correcting that
transport defect; a green workflow alone is not acceptance.

### Cache-rate interpretation

The apparent 96.5% theoretical versus 1.3--3.5% realized cache-rate gap in
`27734909066` is explained by trajectory progress rather than token/hash
divergence:

- all 99 completed profiling records belonged to 99 distinct AIPerf sessions;
  no session completed a second request;
- the `inferencex-agentx-mvp` scenario requires `first_turn_prefix` cache
  busting, so each completed first request was intentionally cold;
- 131 requests were still in flight and cancelled after the 300-second window;
- `theoretical_cache_hit_rate` in `process_agentic_result.py` walks historical
  trace turns zero through the sampled turn and assumes those prior turns were
  served with an infinite cache. It measures potential reuse, not realized
  reuse when a replay starts midway through a trace and its first prefix is
  deliberately busted.

The scenario requires at least 900 seconds specifically to reach steady state.
The VMM-enabled rerun uses 900 seconds and must show completed multi-turn
sessions before its cache and throughput measurements are accepted.

## Baseline

The June 10 B200 aggregate vLLM agentic run (`27297117163`) provides these
reference points from the results database:

- c64 without offload: 81,863 tok/s total, 10,233 tok/s/GPU, 13.85s mean
  TTFT, 82.1ms mean TPOT, and 93.5% measured server cache hit;
- c64 with offload: 72,981 tok/s total, 9,123 tok/s/GPU, 12.73s mean TTFT,
  101.5ms mean TPOT, and 93.2% measured server cache hit;
- c128 without offload: 83,163 tok/s total and 10,395 tok/s/GPU;
- c128 with offload: 98,908 tok/s total and 12,363 tok/s/GPU.

The GB200 disaggregated sweep does not use KV offloading, as requested. Both
B200 modes remain useful references. The dense 3P/2D c16--c96 grid resolves
the useful region and its first overload transition; the official c192 point
provides the hard-overload boundary.

## Completion

The final official 40-GPU sweep, artifact audit, topology comparison,
concurrency knee analysis, and B200 comparison are complete. No additional
timeout or message-buffer change was needed for that c16--c96 sweep.

## High-Concurrency Per-GPU Optimization

The next phase changes the objective from maximizing the 40-GPU system's
absolute throughput to maximizing throughput per inference GPU at hundreds of
concurrent trajectories. Replica count is therefore a search axis rather than
a fixed budget.

### Initial diagnosis and 1P/1D candidate

- The c64--c96 3P/2D logs show that high concurrency is not a decode-only
  capacity problem. At c96, local prefill cache reuse fell to 54--57%, decode
  generation throughput collapsed, and TTFT reached 141 seconds even though
  both decode workers continued to report healthy external-cache transfers.
- AIPerf did perform its trajectory-aware warmup. At c96 it dispatched 131
  warmup credits and waited 397 seconds for all of them before profiling. The
  high-concurrency collapse is therefore not an accidentally cold measured
  phase, although larger future runs need longer affinity leases because their
  warmup can take substantially longer.
- The existing 1P/1D recipe predated the fixes validated by the final sweep.
  It has been brought to the same runtime contract: Dynamo June 18 nightly,
  TCP request plane, KV router and reset, vLLM KV-event publication, 32k prefix
  retention, UCX RC/NIXL, bounded registration cache, generated synchronized
  engine IDs, and no CuMem/sleep-mode allocator.
- 1P/1D uses only 16 inference GPUs. It must exceed 105,844 total tok/s to beat
  the final 3P/2D c64 result's 6,615 tok/s/GPU. Its initial grid is
  c64/c96/c128/c160/c192/c256/c384.
- The high-concurrency recipe sets the Dynamo affinity inactivity timeout to
  14,400 seconds. This is not a request timeout and does not hide failures; it
  only prevents early warmup bindings from expiring before a long warmup and
  1,800-second profile finish.
- A 1P/2D candidate uses 24 inference GPUs to test whether a second decode can
  raise the saturation ceiling without paying for additional cache-fragmenting
  prefill replicas. It must exceed 158,766 total tok/s to beat 3P/2D c64 on
  throughput per GPU. c144 is its isolated first canary; the wider candidate
  grid spans c96--c384.
- A balanced 2P/2D candidate uses 32 inference GPUs and must exceed 211,687
  total tok/s to beat 3P/2D c64 on throughput per GPU. c176 is its isolated
  canary, with a wider c128--c384 grid available if it clears that gate.

## Acceptance Criteria

- At least two P/D parallelism configurations represented in official
  successful GitHub Actions artifacts.
- Multiple concurrency points for each selected topology.
- No request errors or hidden warmup aborts.
- Artifacts contain aggregated and raw agentic results.
- Server logs demonstrate working KV-event ingestion and routing.
- Performance is reviewed against B200 aggregate results and any remaining
  gap is explained by measured evidence rather than workflow success alone.

## High-Concurrency Canary Results and Revised Baseline

Three minimum-replica TEP canaries completed successfully but were rejected
for performance:

| Run | Topology / concurrency | Total tok/s | Tok/s/GPU | Mean TTFT | Final prefill cache hit |
| --- | --- | ---: | ---: | ---: | ---: |
| `27886872533` | 1P/1D c128 | 23,073 | 1,442 | 989s | 2.7% |
| `27886909223` | 1P/2D c144 | 22,399 | 933 | 993s | 1.5% |
| `27886930163` | 2P/2D c176 | 53,087 | 1,659 | 343s | 17.9--26.5% |

- Warmup took 1,395s, 1,507s, and 932s respectively, confirming that the
  four-hour Dynamo session lease is required at these concurrency levels.
- Prefill workers were the bottleneck (about 97% active for both one-prefill
  cases and 78% for 2P/2D); decode workers were only 46--66% active. Adding a
  second decode did not improve a workload that had already evicted its
  warmed prefix working set.
- Hundreds of trajectories cannot be made efficient merely by raising
  concurrency on one or two TEP8 prefill caches. Future points must increase
  cache domains and/or prefill attention concurrency while reducing excess
  decode GPU cost.

The complete June 10 B200 single-node aggregate offload curve changes the
normalized target. Its strongest completed point was c196 at 113,672 tok/s
total, 14,209 tok/s/GPU, 15.53s mean TTFT, and 279.09ms mean TPOT. c128 was
98,908 tok/s total and 12,363 tok/s/GPU. The current GB200 3P/2D c64 point
(6,615 tok/s/GPU) is therefore only 46.6% of the best B200 normalized
throughput, despite substantially better TPOT. New candidates are compared
against both the GB200 frontier and this stronger B200 offload reference.

### TP4 x DP2 search

The first DEP candidate uses three prefill workers and one decode worker;
every eight-GPU worker is TP4 x DP2 with expert parallelism across all eight
GPUs. Dynamo 1.3.0.dev20260618 routes to `(worker_id, dp_rank)` and forwards
`data_parallel_rank` to vLLM, so cache affinity is evaluated per DP rank
instead of relying on round-robin inside the worker. At c112, six prefill DP
ranks receive roughly 19 active trajectories each while the two decode ranks
avoid paying for a second eight-GPU decode replica. The topology uses 32
inference GPUs and must exceed 211,687 tok/s to beat 3P/2D c64 on normalized
throughput.

Run `27911020056` / Slurm job `19533` rejected the hybrid TP4 x DP2 shape at
launcher expansion, before model loading or benchmarking. The pinned
srt-slurm implementation treats a worker with `data-parallel-size > 1` as
pure DEP and launches one process per assigned GPU. For an eight-GPU worker
declared as TP4 x DP2 it emitted DP ranks 0--7 while every process still had
`--data-parallel-size 2 --tensor-parallel-size 4`; ranks 2--7 are invalid and
the implied allocation is larger than the eight GPUs granted to the worker.
The run was cancelled after 51 seconds instead of consuming nine nodes on an
unsupported topology. Supporting hybrid TP x DP requires a first-class
srt-slurm launcher change and is not emulated with ad-hoc process overrides.

Run `27911088706` / Slurm job `19534` completed the 4P/2D TEP8/TP8 c112
candidate at 176,730 total tok/s, or 3,682 tok/s/GPU. Mean TTFT was 63.34s,
interactivity was 52.31 output tok/s, and all 2,656 measured requests
completed without errors. This is substantially below 3P/2D c80's 6,709
tok/s/GPU, so the 48-GPU topology is rejected despite its clean execution.
The next search axis removes excess decode capacity: 2P/1D c52 and 3P/1D c68.

## Multi-node Server Metrics Plumbing

The shared agentic replay builder now accepts
`AIPERF_SERVER_METRICS_URLS` as a comma-separated list and passes its values
to AIPerf under one `--server-metrics` option. Worker discovery is performed
by srt-slurm's host-side orchestrator, which already owns the resolved process
topology and system ports. Agentic custom benchmarks opt in with
`benchmark.aiperf_server_metrics: true`; srt-slurm injects one URL per logical
worker leader as `AIPERF_SERVER_METRICS_URLS`. Distributed follower processes
are excluded because they do not own separate vLLM engines. Stable
deduplication preserves topology order, so the URLs remain prefill 0, prefill
1, ..., decode 0, ... regardless of allocated hostnames. AIPerf continues to
auto-scrape the frontend metrics endpoint as well.

Run `27915217510` / Slurm job `19535` exposed the flaw in the initial
container-side implementation. All 2P/1D engines became healthy after about
30 minutes, but the benchmark failed before replay because the vLLM image does
not contain `scontrol`. That approach was removed rather than installing
Slurm tooling or parsing compressed hostlists inside the model container.
srt-slurm commit `de59739b172e507e15ebf145bfe305f606e82fbf` adds the explicit
custom-benchmark opt-in and leader-only, topology-ordered discovery.

The replacement was validated without loading DeepSeek weights. GB200 mocker
job `19536` first caught that URL-set sorting scrambled topology order. After
stable ordering was added, mocker job `19537` registered 2P/1D in 32 seconds,
injected `7500`, `7501`, and `7502` into the custom benchmark container, and
successfully fetched all three `/metrics` endpoints. It completed successfully
in 53 seconds. The real two-node-per-worker topology consequently resolves to
leader ports `7500`, `7502`, `7504`, and so on from srt-slurm's actual process
objects, without node-offset or stride assumptions.

AIPerf's iterative realtime block now keeps these worker endpoints separate
instead of collapsing them into one `srv` row. Dynamo's
`dynamo_component=prefill|backend` label produces stable `srv prefill N` and
`srv decode N` rows (with `backend` presented as `decode`); unlabeled endpoints
fall back to their host and port. This exposes per-worker cache hit rate, KV
usage, queue depth, preemptions, and token throughput so cache-affinity and
load-imbalance failures are visible while a sweep is still running. The
existing aggregate snapshot API and per-endpoint export data remain intact.

Official run `27915217510` attempt 2 / Slurm job `19538` validated the complete
path on the real 2P/1D DeepSeek configuration. The generated AIPerf command
contained prefill endpoints `:7500` and `:7502` plus decode endpoint `:7504`.
Iterative output emitted distinct `srv prefill 0`, `srv prefill 1`, and
`srv decode 0` rows with populated cache, KV usage, queue, preemption, and
token-throughput fields. The prefills finished around 91.6% and 92.3% local
prefix hit rate. The official aggregate completed 2,689 requests at 187,624
total tok/s, 7,818 tok/s/GPU, 4.78s mean TTFT, 24.69ms mean TPOT, and 40.96
output tok/s interactivity. This improves normalized throughput by 16.5% over
the previous GB200 best (3P/2D c80 at 6,709 tok/s/GPU). Decode metrics showed
roughly 35--52 running requests and up to 89% KV usage while prefills were
usually lightly queued, so the wider high-throughput search retains one decode
and sweeps concurrency rather than adding decode GPUs preemptively.

## Server-Reuse Concurrency Batches

Multi-node agentic matrix entries now retain each search entry's concurrency
list instead of expanding it into one GitHub/Slurm job per point. The custom
benchmark installs AIPerf once, then runs every concurrency sequentially
against the same healthy server allocation. Each point gets its own result
directory, command log, raw artifacts, and aggregate JSON; the workflow checks
that the number of successful aggregates equals the requested concurrency
count and uploads all of them under the normal official artifact paths.

This does not inherit warmed trajectory prefixes across points. The locked
AgentX scenario injects `first_turn_prefix` cache-bust markers, and their digest
includes AIPerf's automatically generated per-invocation benchmark UUID. Each
concurrency process therefore occupies a disjoint KV keyspace, while that
process's warmup and measured phases intentionally share markers. Conversation
correlation IDs are also unique per invocation and completed trajectories emit
Dynamo session `close`. Old cache blocks are naturally evictable and cannot
match a later point's salted prefix.

The 2P/1D and 3P/1D grids are split into low, mid, and high-concurrency batches
of four points. This caps the number of sequential profiles sharing an engine
allocation and bounds the cost of a late benchmark failure or engine
degradation. Each allocation stays below the eight-hour Actions limit: two
hours of measured profile time plus one engine startup and per-point
warmup/drain. This replaces twelve 30-minute model startups per topology with
three, without weakening per-point artifact or cache-isolation semantics.

The generator also enforces a maximum of four sequential concurrency points
per multi-node agentic allocation. This applies to pre-existing configurations
as well as the new one-decode grids, preventing an unrelated long concurrency
list from silently becoming one oversized server-reuse batch.

Manual GB200 mocker jobs `19542`--`19544` exercised the production custom
benchmark launch and discovered the frontend plus all three worker metrics
endpoints in topology order. The mocker could not complete an AgentX
trajectory: its disaggregated prefill worker advertised the cross-node
bootstrap endpoint as `127.0.1.1:7200`, so decode failed to connect. This is a
mocker-only transport limitation, not the production vLLM/NIXL path; the
grouped shell loop remains covered by a local stub execution and matrix tests,
while the first production 2P/1D batch is the end-to-end validation.

The B200 database baseline was re-queried before the high-throughput launch.
Workflow `2022` c196 with offload remains the normalized target at 113,671.86
tok/s total and 14,208.98 tok/s/GPU (15.526s mean TTFT, 279.09ms mean TPOT).

## High-Concurrency Server-Reuse Run

Official run `27922709353` uses one 2P/1D allocation for c128, c160, c192,
and c256. The first point completed successfully but established that c128 is
already deep overload for this topology: 59,345.29 tok/s total, 2,472.72
tok/s/GPU, 337.97s mean TTFT, and 21.82% aggregate server GPU cache hit rate.
The theoretical trace reuse remained 96.65%. Decode generally had only 6--28
running requests and roughly 15--30% KV usage; both prefills stayed near 20%
KV usage. There were no NIXL errors, and all decode ranks passed compatibility
checks. This rules out decode saturation and KV-capacity thrashing at c128;
the fixed-duration run admitted too many distinct long trajectories to reach
the high-reuse steady state.

Added a separate 1P/1D DEP8-prefill experiment using 16 inference GPUs. It
keeps Dynamo KV routing, prefix caching, the 32K retention interval, KV event
publication, and the validated TP8 decode path. Its prefill follows the repo's
existing vLLM DEP pattern (`TP1 x DP8`, EP8) and raises
the prefill batch-token ceiling from 16K to 32K. This tests whether eight-way
attention parallelism can improve raw prefill throughput enough to offset the
expected per-rank load-balance and cache-affinity penalty.

The initial server-reuse run exposed a missing quiescence boundary after c128.
AIPerf reached its 30-minute profile deadline with 265 requests still in
flight, forced phase completion after its cancellation grace period, and the
shell immediately launched c160. The second warmup stalled while the Dynamo
frontend still reported hundreds of active/pending requests from both
invocations, so c160 and later points were invalid and run `27922709353` was
cancelled.

The grouped runner now polls the Dynamo frontend's active-request gauge plus
every discovered worker's vLLM running/waiting gauges between points. It
requires three consecutive idle polls before continuing, waits up to 30
minutes by default, and fails rather than contaminating the next result if the
system cannot drain. It does not clear KV state.

The first DEP8 bring-up (`27925198626`, Slurm `19547`) failed during model
load with `KeyError: layers.0.ffn.experts.w13_input_scale`. EP filtering had
already selected the expected 48/384 experts per rank. The failure came from
the explicitly inherited `deep_gemm_mega_moe` loader, whose expected scale
layout does not match the current v0.23 NVFP4 checkpoint. The override was
removed so DEP8 uses the same checkpoint-compatible default MoE backend as the
successful TEP8 recipes; no topology or cache setting changed.

The compatible default backend then loaded far enough to establish the actual
DEP8 limit (`27925448660`, Slurm `19548`): every prefill rank used 166.43 GiB
and OOMed requesting another 21.65 GiB with 17.56 GiB free. DEP replicates the
attention weights that TEP8 shards, so the current checkpoint cannot run DEP8
on the 184 GiB GB200 GPUs. The failed DEP recipe and master entry were removed
rather than leaving a known-broken official configuration.

The next TEP experiments raise prefill `max-num-batched-tokens` from 16K to
32K, matching the repo's established GB200 high-throughput prefill setting.
The c128 audit showed low prefill KV occupancy and no decode saturation, so
larger compute batches directly target the uncached-prefill bottleneck without
reducing cache capacity. The selected batches are 1P/1D c32/40/52/64 (16 GPUs)
and 2P/1D c52/64/80/96 (24 GPUs).

### Grouped 32K prefill-batch experiment

- Official runs `27925769616` (1P/1D c32/40/52/64, Slurm `19550`) and
  `27925769598` (2P/1D c52/64/80/96, Slurm `19549`) test up to four
  concurrencies per allocation with a drain boundary between points.
- The first boundaries worked as designed: after c32 and c52, respectively,
  the gate observed zero active frontend requests and zero running/waiting
  worker requests for three consecutive polls before starting the next point.
- Both first-point profiles completed every recorded request (317/317 for
  1P/1D c32 and 1,405/1,405 for 2P/1D c52) with no request errors. The AIPerf
  phase-end cancelled-credit timeout did not remove a successful request.
- 1P/1D c32 delivered 26,566.9 tok/s total, 1,660.43 tok/s/GPU, 193.21 s mean
  TTFT, and 13.46 ms mean TPOT. Its prefill local cache hit was only about 7%
  near profile completion.
- 2P/1D c52 delivered 103,259.7 tok/s total, 4,302.49 tok/s/GPU, 46.48 s mean
  TTFT, and 20.21 ms mean TPOT. Its two prefills ended near 44.7% and 65.5%
  local cache hit and supplied about 44.9K and 57.8K input tok/s. Decode had
  no sustained queue and supplied about 808 output tok/s, so it was not the
  throughput limit.
- The 2P/1D result is a direct regression from the earlier official 16K-batch
  c52 result (`27915217510`): 187,623.56 tok/s total and 7,817.65 tok/s/GPU.
  The earlier run's two prefill logs ended at 91.0% and 90.3% local prefix
  cache hit. Its aggregate JSON's 92.5% external hit value is primarily the
  decode worker's transferred-KV path and is not labeled as prefill hit.
  The fixed-length recipe's 32K batch setting therefore does not transfer to
  this long-context cache-heavy workload. Both TEP8 recipes are restored to
  16K for the next controlled grouped sweep; the running 32K allocations are
  retained to finish their remaining comparison points.
- Commit `63a372c3` restores the validated 16K setting. Official reruns are
  `27928586423` (1P/1D c40/52/64/80) and `27928587096` (2P/1D
  c52/64/80/96). Each dry-run generated exactly one four-point allocation.
  Two earlier dispatches (`27928524345`, `27928524960`) used the workflow from
  `main`; that version lacks the branch's multi-node agentic wiring, so both
  ended without creating a benchmark job or allocating a GPU. The corrected
  dispatches use `chore/agentx-v0.4` as both workflow and benchmark ref.

### DEP8 memory-limit correction

The initial interpretation of DEP8 run `27925448660` / Slurm `19548` was too
broad. The checkpoint loaded and the failure occurred in FlashInfer's TRT-LLM
NVFP4 MoE kernel warmup, not while loading weights. The 32K-token recipe made
the token-proportional MoE intermediate request 21.65 GiB while 17.56 GiB was
free on every rank. The compatible default loader was correct; the batch was
not. The DEP8 recipe and master entry are restored with a 16K prefill batch,
which should approximately halve that intermediate. No allocator setting,
memory-utilization reduction, PP, or custom checkpoint loader is introduced.
The four-point c64/96/128/160 group will determine whether DEP's attention
parallelism can repay its cache-affinity/load-balance cost at high concurrency.

## Final Minimum-Replica Throughput Results

All five grouped workflows completed successfully after the monitoring
connection was interrupted. Each workflow published aggregate, raw replay,
server-log, collected-result, and run-stat artifacts. Every one of the 20
measured points reports all recorded requests successful.

| Run | Topology | Batch | Conc. | Successful | Total tok/s | Tok/s/GPU | Mean TTFT | P95 TTFT | Mean TPOT | Interactivity |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `27925769616` | 1P/1D TEP8 | 32K | 32 | 317/317 | 26,567 | 1,660 | 193.21s | 249.91s | 13.46ms | 74.3 tok/s |
| `27925769616` | 1P/1D TEP8 | 32K | 40 | 347/347 | 27,667 | 1,729 | 222.02s | 303.68s | 13.69ms | 73.1 tok/s |
| `27925769616` | 1P/1D TEP8 | 32K | 52 | 313/313 | 25,597 | 1,600 | 304.37s | 383.61s | 13.84ms | 72.2 tok/s |
| `27925769616` | 1P/1D TEP8 | 32K | 64 | 583/583 | 26,514 | 1,657 | 421.53s | 524.18s | 14.75ms | 67.8 tok/s |
| `27928586423` | 1P/1D TEP8 | 16K | 40 | 451/451 | 36,324 | 2,270 | 158.15s | 297.57s | 15.06ms | 66.4 tok/s |
| `27928586423` | 1P/1D TEP8 | 16K | 52 | 326/326 | 26,748 | 1,672 | 291.26s | 384.70s | 14.13ms | 70.8 tok/s |
| `27928586423` | 1P/1D TEP8 | 16K | 64 | 584/584 | 27,443 | 1,715 | 406.14s | 515.79s | 14.96ms | 66.9 tok/s |
| `27928586423` | 1P/1D TEP8 | 16K | 80 | 465/465 | 24,319 | 1,520 | 644.29s | 856.19s | 14.83ms | 67.4 tok/s |
| `27925769598` | 2P/1D TEP8 | 32K | 52 | 1,405/1,405 | 103,260 | 4,302 | 46.48s | 118.91s | 20.21ms | 49.5 tok/s |
| `27925769598` | 2P/1D TEP8 | 32K | 64 | 1,250/1,250 | 67,753 | 2,823 | 121.56s | 320.44s | 18.27ms | 54.7 tok/s |
| `27925769598` | 2P/1D TEP8 | 32K | 80 | 1,020/1,020 | 61,616 | 2,567 | 207.72s | 589.69s | 17.86ms | 56.0 tok/s |
| `27925769598` | 2P/1D TEP8 | 32K | 96 | 1,133/1,133 | 63,235 | 2,635 | 231.16s | 706.87s | 17.96ms | 55.7 tok/s |
| `27928587096` | 2P/1D TEP8 | 16K | 52 | 2,576/2,576 | 180,055 | 7,502 | 5.39s | 24.26s | 25.13ms | 39.8 tok/s |
| `27928587096` | 2P/1D TEP8 | 16K | 64 | 2,413/2,413 | 142,163 | 5,923 | 31.74s | 208.79s | 24.54ms | 40.7 tok/s |
| `27928587096` | 2P/1D TEP8 | 16K | 80 | 1,037/1,037 | 62,011 | 2,584 | 172.85s | 597.55s | 17.84ms | 56.0 tok/s |
| `27928587096` | 2P/1D TEP8 | 16K | 96 | 1,112/1,112 | 62,598 | 2,608 | 242.54s | 766.37s | 17.82ms | 56.1 tok/s |
| `27928732675` | 1P/1D DEP8 | 16K | 64 | 1,720/1,720 | 89,816 | 5,614 | 71.80s | 284.00s | 19.77ms | 50.6 tok/s |
| `27928732675` | 1P/1D DEP8 | 16K | 96 | 1,710/1,710 | 97,121 | 6,070 | 102.20s | 1,010.47s | 19.83ms | 50.4 tok/s |
| `27928732675` | 1P/1D DEP8 | 16K | 128 | 1,959/1,959 | 110,723 | 6,920 | 105.47s | 1,029.32s | 20.51ms | 48.8 tok/s |
| `27928732675` | 1P/1D DEP8 | 16K | 160 | 1,968/1,968 | 112,308 | 7,019 | 119.14s | 1,014.78s | 20.26ms | 49.4 tok/s |

### Interpretation

- The reproducible moderate-concurrency winner is 2P/1D TEP8 16K c52:
  180,055 tok/s total and 7,502 tok/s/GPU. The independent earlier c52 run
  `27915217510` reached 187,624 tok/s and 7,818 tok/s/GPU, a 4.0% difference.
  Its prefills ended at 90.6% and 92.2% local prefix hit in the grouped rerun.
- c52 is a sharp locality knee. At c64 those same prefills ended near 76%
  local hit and normalized throughput fell 21%. At c80 they ended at
  46.5--50.3% and throughput fell 66% from c52. c96 remained in the same
  collapsed regime. More admitted trajectories delay repeated turns and
  replace reusable prefixes with cold first-turn working sets.
- Raising the TEP prefill scheduler batch from 16,384 to 32,768 tokens is
  harmful for this trace. At 2P/1D c52 it reduced total throughput 42.7%,
  increased mean TTFT from 5.39s to 46.48s, and reduced end-of-profile local
  prefill hit from roughly 91% to 45--66%. The larger active chunk improves
  neither useful batching nor cache residency for long agentic prompts.
- A single TEP8 prefill is not viable for this workload. It remains nearly
  continuously occupied by one long chunk while cold trajectories accumulate;
  changing 32K to 16K only improves the lowest point and does not prevent
  cache collapse. Its high output-token interactivity is survivor-biased:
  very few requests complete and TTFT grows to hundreds of seconds.
- DEP8 16K is valid and materially better than single-prefill TEP8. The
  32K OOM was a batch-sized MoE workspace issue, not a checkpoint-fit limit.
  DEP scales monotonically through c160, reaches 112,308 tok/s total on 16
  inference GPUs, and holds TPOT near 20ms. However, c160 gains only 1.4%
  throughput over c128 while mean TTFT rises 13%; c128 is the practical DEP
  knee and c160 is the maximum-throughput endpoint.
- DEP's c128/c160 local prefill hit was about 72%/56%, respectively. Its eight
  attention DP ranks expose more raw prefill concurrency, but distributing
  requests and cache blocks across ranks loses locality. This matches the
  predicted DEP cache-affinity/load-balance tradeoff.
- The one decode replica was active but not the primary limit in the rejected
  high-concurrency cases. DEP delivered about 0.95K output tok/s at c128/c160;
  the good TEP c52 point delivered about 1.30K output tok/s. Prefill locality
  collapsed before decode reached the demonstrated c52 output rate.

### Baseline and acceptance result

The target remains the database's B200 aggregate vLLM c196 offload record,
workflow `2022` / GHA `27297117163`: 113,671.86 tok/s total, 14,208.98
tok/s/GPU, 15.526s mean TTFT, and 279.09ms mean TPOT on eight inference GPUs.

- Best GB200 normalized result observed: 7,817.65 tok/s/GPU from the earlier
  2P/1D TEP8 16K c52 run, 45.0% below the B200 target.
- Best result in the final grouped reruns: 7,502.28 tok/s/GPU, 47.2% below.
- Best hundreds-concurrency result: DEP8 c160 at 7,019.27 tok/s/GPU, 50.6%
  below, with unacceptable 119s mean and 1,015s p95 TTFT.
- Disaggregation clearly improves total system throughput and TPOT in the
  3P/2D low-latency curve, but no tested GB200 vLLM topology beats the B200
  aggregate result after normalization by all inference GPUs.

### Final infrastructure audit

- The grouped drain gate reached frontend and worker idle state between
  points; every final workflow concluded green with five expected artifact
  groups and no missing aggregate JSON.
- Worker metrics discovery produced distinct frontend, prefill, and decode
  collectors. The TEP 2P/1D runs reported four collectors; 1P/1D reported
  three. DEP's single logical prefill endpoint aggregates its internal DP
  engine metrics.
- No profiled NIXL, UCX, HTTP 503, NATS payload, or request errors were found
  in the final 16K server logs. Expired producer-lease messages correspond to
  fixed-duration cancellation/drain requests absent from the successful
  aggregates, not failed measured transfers.
- No larger message buffer was required. NATS retained its 32 MiB control
  payload limit; bulk KV data used NIXL over UCX RC rather than NATS.
- Every Slurm allocation exited, and no orphaned job remained after workflow
  completion.

## Full-Sweep Pruning Audit

The completed curves already justify removing configurations that cannot
contribute a useful throughput/latency frontier point:

- Remove TEP8 1P/1D. Its tested 16K points deliver only 1,520--2,270
  tok/s/GPU with 158--644 seconds mean TTFT. The apparent high output-token
  interactivity is survivor bias under severe prefill starvation.
- Remove 1P/2D. The one-prefill TEP topology is conclusively prefill/locality
  limited; adding decode capacity cannot repair that bottleneck.
- Retain 2P/1D low-concurrency points and c52, but remove c64/c80/c96 and the
  c128+ batch. c52 has higher throughput and radically lower TTFT; c128 is
  already deep overload and rules out the larger points.
- Remove 2P/2D. The measured 2P collapse occurs before decode saturation, so
  the second decode replica adds GPUs without addressing the limiting tier.
- Remove 4P/2D c112. Its 3,682 tok/s/GPU and 63.34-second mean TTFT are far
  from the established frontier.
- Retain the useful 3P/2D c16/c24/c32/c40/c48/c56/c64/c80 curve, but remove
  dominated c72 and overloaded c96.
- Retain DEP8 c128/c160 temporarily as the maximum-normalized-throughput
  endpoints despite their poor TTFT.
- Remove 3P/1D. Its c56 gate delivered 156,510 tok/s total, 4,891 tok/s/GPU,
  5.14-second mean TTFT, and 38.95 tok/s interactivity. The validated 2P/1D
  c52 point uses eight fewer GPUs while delivering both higher total throughput
  (180,055--187,624 tok/s) and higher normalized throughput
  (7,502--7,818 tok/s/GPU) at essentially the same TTFT/interactivity. The
  single TP8 decoder was queued at 83--97% KV occupancy while all three
  prefills remained mostly unqueued, so c68/c80/c96 were cancelled.
- Remove 4P/1D. Its measured c32/c64 points were already dominated by 3P/2D,
  and the 3P/1D gate now proves that adding still more TEP8 prefill replicas
  behind the same saturated TP8 decoder cannot recover normalized throughput.

The resulting compact full sweep should keep 2P/1D c16/c24/c32/c40/c52,
3P/2D c16/c24/c32/c40/c48/c56/c64/c80, and provisionally DEP8 c128/c160.
The DEP8 endpoints remain provisional until the DEP8-decode experiment shows
whether they contribute information not covered by the replacement topology.
Whole-entry removals should also delete their now-unreferenced recipe files.

## Full-Rack Manual Tuning Loop

The next tuning phase uses direct srt-slurm submissions on Watchtower rather
than paying the GitHub Actions feedback-loop cost for every diagnostic point.
Final accepted frontier points still require an official Actions artifact.

- Watchtower has 18 four-GPU GB200 nodes (72 GPUs). Manual diagnostic recipes
  set `infra.etcd_nats_dedicated_node: false`, colocating NATS/etcd with the
  first worker/head node so three independent 24-GPU 2P/1D stacks can run at
  once. Checked-in recipes remain unchanged pending stability validation.
- Jobs `19581`, `19582`, and `19583` independently test two TEP8 prefills plus
  a TP1 x DP8 / EP8 decoder at c52, c64, and c80. Each uses a 900-second replay
  and has its own frontend, router state, result directory, and logs.
- The comparison holds the validated 16K prefill batch, prefix retention,
  Dynamo KV/session-aware routing, and NIXL/UCX transport constant. It tests
  whether decode DEP moves the c52--c80 knee without conflating prefill
  changes.
- Follow-up manual recipes are staged for two and three DEP8 prefill workers,
  each paired with either TP8 or DEP8 decode. DEP prefill uses the validated
  16K batch, TP1 x DP8 / EP8, and `max-num-seqs: 256`; no loader, allocator,
  offload, or custom-kernel workaround is introduced.
- Promising shapes will use grouped concurrency batches of at most four points
  and push through c128/c160/c192/c256/c320 until normalized throughput
  plateaus or TTFT becomes unreasonable. Every grouped point retains the
  unique cache-bust namespace and idle-drain boundary.

Five-minute interim results from the decode-DEP comparison:

- c52 stabilized near 179.7K total tok/s (7.49K/GPU), 13.7-second p95 TTFT,
  and 95--96% local prefill hits.
- c64 stabilized near 172.5K total tok/s (7.19K/GPU), 24.8-second p95 TTFT,
  and 82--88% local prefill hits. This materially exceeds the earlier TP8
  decode c64 point (~142K total / 5.92K/GPU) while preserving useful latency.
- c80 reproduced the collapsed regime: approximately 60.5K total tok/s
  (2.52K/GPU), 216-second p95 TTFT, and 44--51% prefill hits. Job `19583` was
  stopped after more than five stable profiling minutes and its complete
  server/config/log tree was copied to the local `.gb200-manual-results`
  directory.
- Jobs `19581`/`19582` continue to full 900-second c52/c64 profiles. Freed
  nodes immediately started job `19585`, which runs two DEP8 prefills plus a
  TP8 decoder at grouped c64/c96/c128/c160.

Final 900-second decode-DEP results:

- c52: 177,031 total tok/s, 7,376 tok/s/GPU, 4.00-second mean / 11.74-second
  p90 TTFT, 38.62 tok/s p90 interactivity, and 94.2--94.5% local prefill
  hits. All 1,269 recorded requests succeeded. This is effectively neutral
  versus the validated TP8-decode c52 result rather than a new frontier.
- c64: 152,623 total tok/s, 6,359 tok/s/GPU, 24.33-second mean /
  76.96-second p90 TTFT, 39.62 tok/s p90 interactivity, and 78.7--82.9%
  local prefill hits. All 1,381 recorded requests succeeded. DEP decode is
  approximately 7% faster than the prior grouped TP8-decode c64 point, but
  c52 still dominates it after the longer profile exposes continued locality
  decay.
- The results confirm that decoder execution only moves the overload knee
  modestly; prefill cache locality/capacity is the primary high-throughput
  tuning axis.
- The final JSONs and complete output trees are preserved under
  `.gb200-manual-results/2026-06-22/`. Jobs `19585`--`19587` now occupy the
  full rack: 2xDEP8-prefill/TP8-decode at c64/c96/c128/c160, and two
  2xDEP8-prefill/DEP8-decode stacks covering c64/c96/c128/c160 and
  c192/c256/c320.

## June 21 Ground-Up Retune

The results above remain useful topology priors, but they used the June 15
trace and fixed-sequence-derived vLLM limits. The new frontier must be measured
again with `semianalysis_cc_traces_weka_062126` and the following common
contract:

- vLLM v0.23 and Dynamo `1.3.0.dev20260618`.
- No active `max-num-seqs`, decode `max-num-batched-tokens`, eager execution,
  async-scheduling disable, or explicit CUDA-graph capture-size override.
- 16K prefill batching for TEP8. A vLLM maintainer confirmed this is
  appropriate after the 32K candidate exceeded the available MoE workspace.
  DEP8 requires 8K because each rank holds a full model replica and its 16K
  FP4 MoE intermediate OOMs on the first inference step.
- Model Runner V2 on both tiers. vLLM v0.23 supports it but does not select it
  automatically for a quantized MoE model.
- `FLASHINFER_MLA_SPARSE_DSV4` on both tiers and FP8 prefill-query
  quantization on prefill workers. The v0.23 source specifically recommends
  query quantization for long-context prefill with FP8 KV.
- No `VLLM_USE_RUST_FRONTEND`: srt-slurm launches workers through
  `python -m dynamo.vllm`, behind Dynamo's own frontend, so the standalone
  vLLM Rust HTTP frontend is not on the request path.
- Prefix retention 32768, FP4 indexer cache, Dynamo KV routing/events,
  NIXL/UCX RC, stream interval 10, and no CPU KV offload remain unchanged.
- NATS/etcd are colocated on the first worker node. The June 15 full-rack
  experiments showed no control-plane, lease, NIXL, UCX, frontend, or KV
  transfer failures with this layout. It removes one non-inference node from
  each allocation and lets three 24-GPU stacks fill the 72-GPU rack.

The previously manual-only 2P/1D DEP8-prefill + DEP8-decode topology is now
checked in as
`disagg-gb200-2p1d-dep8-dep8-agentic.yaml`, with synchronized
master metadata. Initial clean baselines will compare 2P/1D TEP8/TP8,
TEP8/DEP8, and DEP8/DEP8 in parallel. PR #1789 Actions run `27993108678`
currently owns all 18 GB200 nodes; it is intentionally left undisturbed.

The first clean-default startup (`19627`) exposed one required exception to
the default-graph policy. Both TEP8 prefills loaded successfully and completed
all 51 piecewise CUDA-graph captures. vLLM then began the FULL portion of its
default `FULL_AND_PIECEWISE` mode. `VLLM_USE_NCCL_SYMM_MEM=1` attempted
`ncclCommWindowRegister` from the tensor-parallel all-reduce while CUDA capture
was active, invalidating the stream; the subsequent NCCL and c10d failures
were teardown effects. No benchmark request ran.

All GB200 recipes now set `cudagraph_mode=PIECEWISE` on both tiers. This keeps
vLLM compilation, CUDA graphs, and NCCL symmetric-memory collectives while
leaving graph-incompatible operations outside the graph, matching vLLM's
documented purpose for PIECEWISE mode. It does not reinstate `mode: 0`, eager
execution, an explicit capture-size list, or any memory-utilization workaround.
Replacement direct-Slurm jobs are `19632` (TEP8/TP8), `19633` (TEP8/DEP8),
and `19634` (DEP8/DEP8).

Job `19634` reached readiness, then every DEP8 prefill rank failed on the first
16K-token inference step. FlashInfer's TRT-LLM FP4 MoE kernel requested a
9.84 GiB intermediate tensor with only 5.71--7.93 GiB physically free, while
PyTorch reported 10.10--12.31 GiB reserved but unallocated. The experimental
DEP8-prefill recipe now tests PyTorch's documented fragmentation mitigation,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, before reducing the
maintainer-approved 16K prefill batch or sacrificing KV-cache capacity.

That allocator setting is intentionally rejected by vLLM when NIXL is active:
expandable segments can remap virtual KV addresses and invalidate NIXL's
pinned/registered memory. Enabling the CuMem allocator is not an acceptable
substitute because the prior v0.23 GB200 tests showed CuMem exhausting HBM
during NVFP4 model conversion. The next experimental DEP8-prefill run instead
uses an 8K prefill batch. This directly halves the failing FP4 MoE intermediate
while retaining the 0.9 HBM allocation target and NIXL's address guarantees.

The first June 21 TEP8/DEP8 results showed a decode-side regression relative
to the historical v0.23 configuration: c52 trends near 112K total tok/s with
~78 ms ITL, versus 177K total and ~27 ms ITL historically, despite healthy
prefill locality and much lower TTFT. A controlled recipe now tests decode
MRV2 off while retaining prefill MRV2, FlashInfer DSV4 attention, PIECEWISE
graphs, topology, dataset, and all other settings. The maintained recipe and
job `19633` are the MRV2-on arm.

The vLLM maintainer clarified the official PD recipe's prefill settings:
`enforce-eager` is intentional because long-context prefill is compute-bound
and already hides kernel-launch latency, so CUDA graphs provide little prefill
benefit. It should not be generalized to decode, where launch overhead remains
important. The same maintainer said `max-num-seqs: 2` is not necessary. After
the active MRV2 A/B, a controlled prefill-eager arm will hold decode and all
other settings constant; the sequence limit will remain unset.

The first controlled MRV2 result rejects the decode-regression hypothesis.
At c52, decode MRV2 off (`19639`) delivered 85,786 tok/s total with 21.29s
p90 TTFT and 13.32 tok/s p90 interactivity, versus MRV2 on (`19633`) at
102,526 tok/s, 3.80s, and 13.01 tok/s. MRV2 therefore remains enabled on both
tiers. Job `19640` reuses c52/c64 for the next isolated arm: prefill eager,
decode MRV2/PIECEWISE unchanged, and every other setting held constant.

The 8K DEP8/DEP8 experiment is materially stronger than the initial TEP
points. Job `19637` completed c64 at 144,445 tok/s total (6,018/GPU), 13.46s
p90 TTFT, and 15.58 tok/s p90 interactivity. Its c80 phase is near 194K tok/s
after nine profiling minutes with ~11.3s p95 TTFT and no errors, making it the
leading June 21 high-throughput candidate pending the final export.

The final 8K DEP8/DEP8 c80 and c96 exports establish the first strong June 21
high-throughput curve. c80 delivered 178,668 tok/s total (7,444/GPU), 7.20s
p90 TTFT, and 15.38 tok/s p90 interactivity. c96 improved to 201,534 tok/s
(8,397/GPU), 4.70s p90 / 9.98s p95 TTFT, and 14.80 tok/s p90 interactivity.
c96 completed 1,323 requests without errors; 65 phase-end cancellations are
4.7% of the measured work and remain below the 10% credibility threshold.
Job `19642` extends this exact 8K configuration through c112/c128/c160.

Two follow-up parameter A/Bs are also conclusive. Reducing DEP8 prefill to a
12K batch while lowering `gpu-memory-utilization` to 0.88 (`19641`) produced
only about 89K tok/s at c64 with roughly 35s p95 TTFT after more than five
profiling minutes. This is clearly worse than the 8K/0.90 c64 result above, so
the run was stopped and 8K/0.90 remains the DEP8 prefill setting. Enabling
prefill `enforce-eager` (`19640`) produced 102,445 tok/s at c52 with 13.61s
p90 TTFT and 13.72 tok/s p90 interactivity. The non-eager control was 102,526
tok/s, 3.80s, and 13.01 tok/s respectively. Eager therefore provides no
measurable throughput gain on this workload and materially regresses TTFT; it
will not be adopted despite being valid for a compute-bound prefill service in
general.

The remaining decode-performance isolation changes only the DEP8 decoder from
PIECEWISE graphs to the historical
`{"cudagraph_mode":"FULL_DECODE_ONLY","mode":0}` while leaving both DEP8
prefills PIECEWISE. Job `19643` tests c80/c96. In parallel, job `19644` tests a
16-GPU 1P/1D DEP8/DEP8 service at c32/c48/c64/c80 to establish the
single-prefill scaling law and determine whether its lower GPU denominator
adds normalized-throughput frontier points. Job `19642` began c112 warmup
after a normal (not hung) DeepGEMM and PIECEWISE graph startup.

Job `19642` finalized c112 at 212,671 tok/s total (8,861/GPU), 28.23s p90 /
38.01s p95 TTFT, and 14.51 tok/s p90 interactivity. All 1,752 recorded
requests succeeded; its 104 phase-end cancellations are 5.6% and below the
10% credibility threshold. This is the new June 21 normalized-throughput
leader, while c96 remains a separate frontier point because it gives up only
5.2% normalized throughput for a much lower 4.70s p90 TTFT.

The first submissions of the decode-graph and 1P/1D tests (`19643` and
`19644`) completed model loading and graph capture but then exited before
traffic. The non-interactive manual shell had omitted the required host
`INFMAX_WORKSPACE`; srt-slurm therefore did not add the repository's
`/infmax-workspace` container mount and the custom benchmark returned 127 for
missing `agentic_srt.sh`. This is a submission-environment error, not an
engine, graph, Dynamo, or NIXL failure. Corrected jobs `19645` and `19646`
explicitly export the normal workspace path, and their generated mount lists
were verified before continuing.

The following c128 phase then moved the frontier materially: 254,766 tok/s
total (10,615/GPU), 10.59s p90 / 13.30s p95 TTFT, and 14.21 tok/s p90
interactivity. All 2,049 records succeeded, theoretical reuse was 97.23%, and
aggregate measured GPU cache hit was 36.25%. It is 19.8% faster than c112
while also lowering TTFT, so c112 is dominated apart from a negligible 0.30
tok/s interactivity difference. The c96 point remains useful because it has
4.70s p90 TTFT and 14.80 tok/s interactivity. The service proceeds to c160 to
locate the normalized-throughput knee.

The c160 phase completed at 332,410 tok/s total (13,850/GPU), 5.11s p90 /
10.95s p95 TTFT, and 13.34 tok/s p90 interactivity. All 2,450 records
succeeded and 148 phase-end cancellations are 5.7%. This is the first cleaned
June 21 GB200 point to approach the B200 aggregate maximum of 14,209/GPU,
while its TTFT and interactivity are far better than that B200 c196 point's
42.70s and 2.11 tok/s.

The corrected FULL_DECODE_ONLY job (`19645`) also completed: c80 delivered
245,470 tok/s total (10,228/GPU), 27.34s p90 TTFT, and 35.36 tok/s p90
interactivity; c96 delivered 316,736 tok/s (13,197/GPU), 9.95s, and 32.48
tok/s. Both had zero request errors and less than 6% phase-end cancellation.
Startup and all full graph captures succeeded, confirming that
FULL_DECODE_ONLY is compatible with DEP8; the earlier NCCL registration
failure was specific to the default FULL_AND_PIECEWISE TP path.

These two graph results are not yet accepted as exact production points. The
manual launcher exported `FRAMEWORK=vllm` rather than `dynamo-vllm`, so AIPerf
did not append `--use-dynamo-conv-aware-routing`. Dynamo itself remained in
KV-router mode, but the request header contract was not identical to the
official AgentX path. The 1P/1D c32 job (`19646`) likewise omitted the header
and aborted correctly when nine warmup traces failed; no result was accepted
and the failure threshold was not weakened. The official master metadata uses
`framework: dynamo-vllm`, and a corrected manual or official run must retain
the conversation-aware routing flag.

## Selected Official Frontier Sweep

The diagnostic phase leaves three production DEP8/DEP8 recipes. Prefill stays
PIECEWISE with an 8K batch; decode uses FULL_DECODE_ONLY. The former avoids
the 16K DEP MoE-workspace OOM, while the latter completed all captures and
restored decode throughput. The master YAML now contains exactly four engine
starts and twelve concurrency points:

- 2P/1D: c32/c48/c64/c80 and c96/c128/c160;
- 3P/1D: c160/c192/c240;
- 4P/1D: c256/c320.

This scales prefill replicas with concurrency while retaining one decode
replica, matching the measured high-throughput bottleneck and avoiding an
unnecessary GPU-normalization penalty. The older TEP/TP, TEP/DEP, 1P/1D, and
multi-decode GB200 recipes are no longer referenced by the master sweep and
were removed. The generated matrix contains four grouped jobs, all recipe
worker counts match master metadata, and all 164 matrix-logic tests pass.

## Official June 23 Sweep and Dynamo Lease Hardening

Official workflow `28026900510` uses the June 21 corpus, Dynamo KV routing,
and AIPerf conversation-aware routing. Its first six completed points are:

| Topology | Conc | Total tok/s | Tok/s/GPU | p90 TTFT | p90 interactivity |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2P/1D | 96 | 303,416 | 12,642 | 20.23s | 32.40 |
| 2P/1D | 128 | 423,208 | 17,634 | 2.76s | 30.03 |
| 2P/1D | 160 | 533,251 | 22,219 | 3.31s | 29.12 |
| 3P/1D | 160 | 496,397 | 15,512 | 19.47s | 29.55 |
| 3P/1D | 192 | 492,366 | 15,386 | 10.20s | 26.55 |
| 3P/1D | 240 | 270,919 | 8,466 | 172.63s | 26.15 |

Every recorded request in these artifacts succeeded. The 2P/1D c160 point is
56.4% above the B200 aggregate curve's 14,209 tok/s/GPU maximum while also
having much lower p90 TTFT and much higher p90 interactivity. The 3P/1D c240
point crossed the overload knee: decode KV usage reached approximately 100%,
TTFT rose sharply, and normalized throughput collapsed.

Two later engine starts exposed an infrastructure timeout rather than a model
failure. During slow, concurrent Lustre loading, a brief etcd stall exceeded
Dynamo's 10-second default discovery lease. Engines continued loading and
captured their graphs, but endpoint and metrics registration then failed
because the original lease had expired. CUDA, NCCL, NIXL, and graph capture
were healthy. Dynamo documents `ETCD_LEASE_TTL` as the supported bare-metal
lease control, so all GB200 production recipes now set it to 120 seconds on
the frontend, prefill ranks, and decode ranks. Slurm's critical-process
monitor still detects hard exits immediately; the longer etcd lease only
tolerates transient control-plane stalls during startup.
