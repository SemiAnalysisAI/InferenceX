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

The final official sweep, artifact audit, topology comparison, concurrency
knee analysis, and B200 comparison are complete. No additional timeout or
message-buffer change is supported by the final logs.

## Acceptance Criteria

- At least two P/D parallelism configurations represented in official
  successful GitHub Actions artifacts.
- Multiple concurrency points for each selected topology.
- No request errors or hidden warmup aborts.
- Artifacts contain aggregated and raw agentic results.
- Server logs demonstrate working KV-event ingestion and routing.
- Performance is reviewed against B200 aggregate results and any remaining
  gap is explained by measured evidence rather than workflow success alone.
