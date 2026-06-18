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
4. Use c64 as a topology gate, then run c32/c64/c128/c192 curves for viable
   topologies.
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
- The sweep generator emits 16 points: four topologies times four
  concurrencies.

## Official Runs

| Run | Configuration | Outcome | Key evidence |
| --- | --- | --- | --- |
| `27728896563` | 1P/1D, c32/c64/c128/c192 | Partially green, later cancelled | c64: about 26.7k tok/s total, 423s mean TTFT, about 5% cache hit; not competitive |
| `27732316539` | TEP4 prefill attempt | Failed | Model-load OOM at 182.54 GiB/GPU plus 1.97 GiB allocation |
| `27732541012` | 4P/1D c64, old Dynamo/NATS | Failed warmup | 45/85 errors; empty router indexes; KV-event decode errors; HTTP 503 overloads |
| `27734909066` | 4P/1D c64, new Dynamo/TCP | Success | Official artifact; clean 85/85 warmup and 99/99 profiled requests |
| `27737167704` | c64 topology sweep | Cancelled before allocation | Server-log audit found the four recipes lacked VMM-backed KV registration required for GB200 multi-node NVLink |
| `27738234911` | c64 topology sweep, VMM enabled | Failed | All four decode workers OOM during NVFP4 weight conversion before serving; no benchmark results |
| `27770234988` | c64 topology sweep, RDMA + bounded registration cache | In progress | Official 900-second 4P/1D, 3P/2D, 2P/3D, and 1P/4D gate dispatched from `96f6346a` after direct canary `19244` validated the data plane |

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
B200 modes remain useful references, and the c32/c64/c128/c192 grid brackets
the aggregate system's useful operating region.

## Open Investigation

1. Re-run the c64 P/D topology comparison with VMM-backed NIXL transfers and
   verify transfer throughput, latency, and lease-expiration counters in the
   prefill/decode logs. The 4P/1D result was strongly decode-limited, so
   D-heavy layouts are expected to improve latency and completed-request rate.
2. Determine whether Dynamo still observes only small cached overlaps even
   though the trace reports about 96% theoretical reuse and the prefills have
   spare KV
   capacity. Candidate causes must be proven from token/hash/routing evidence;
   no cache-hit metric bypass or synthetic workload substitution is acceptable.
3. Select viable topologies and run official c32/c64/c128/c192 curves.
4. Download and review final artifacts against the B200 aggregate baseline.

## Acceptance Criteria

- At least two P/D parallelism configurations represented in official
  successful GitHub Actions artifacts.
- Multiple concurrency points for each selected topology.
- No request errors or hidden warmup aborts.
- Artifacts contain aggregated and raw agentic results.
- Server logs demonstrate working KV-event ingestion and routing.
- Performance is reviewed against B200 aggregate results and any remaining
  gap is explained by measured evidence rather than workflow success alone.
