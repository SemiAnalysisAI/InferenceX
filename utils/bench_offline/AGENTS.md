# Offline TRT Agent Notes

Read `README.md` and `/TRT_BENCH_NOTES.md` before changing or debugging this
benchmark.

- This branch is disposable and must remain isolated from the normal serving
  sweep. Never edit `nvidia-master.yaml` or `perf-changelog.yaml`.
- There are three isolated contracts. `huawei` is the validated B300/GB300
  fixed-GBS comparison. The three `pr-*` profiles copy PR #1689's GB300
  decode recipes for maximum offline decode saturation.
  `rack-tp8x9-mtp1` runs nine synchronized copies of the fastest TP8 recipe
  across one proven 72-GPU GB300 fabric domain.
- `global_batch_size` is authoritative. In `huawei`, `max_num_tokens` is
  `local_batch_size * 8192` and prefill must be one exact iteration. In
  PR-max, fixed GBS is the copied decode capacity and
  `max_num_tokens=32768` permits staged local prefill before the exact
  full-batch decode window.

## Huawei GB300 Profile

- Select with `TRT_BENCH_HARDWARE_PROFILE=gb300`. It is four physical nodes,
  four GPUs/tasks per node, and one external 16-rank
  `trtllm-llmapi-launch` world. The ranks must be exactly `0..15`.
- Use image
  `nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1`
  and TRT source `34a563ac6d8cc0ca7068c7f619e869fb8a625333`. The working
  topology/config reference is InferenceX PR #1689.
- Keep TP16, EP16, attention DP, LM-head TP, MoE TP1,
  `MEGAMOE_DEEPGEMM`, the pinned EP16/384-slot load-balancer file, KV
  fraction `0.70`, PDL, and ConfigurableMoE enabled. The pinned TRT source
  defaults `ENABLE_CONFIGURABLE_MOE` to `1`; forcing it to `0` constructs the
  MegaMoE base backend without the wrapper's concrete `forward_impl`.
- GB300 local batches are `1`, `4`, and `8`; max token shapes are `8192`,
  `32768`, and `65536`. The B300-only attention workspace, 12 GiB KV reserve,
  and oversized DeepGemm chunker must remain disabled. Keep the shared
  packed-FP8 quantizer guard at 32768 rows: run `27515257151` showed the
  GB300 GBS128 65536-row warmup projection fail
  `fp8_quantize_1x128_packed_ue8m0` with a CUDA `invalid argument`.
- Keep GB300 GBS128 runtime `max_num_tokens=65536`, but cap only TRT's
  synthetic pure-context warmup request at 32768 tokens. Run `27516149323`
  proved the packed-FP8 guard, then exhausted memory while creating temporary
  KV-estimation resources after the 65536-token autotune. GBS64 proves the
  same image initializes the 32768-token tuning shape. Do not apply the
  B300-only memory workarounds or shrink the real GBS128 prefill.
- The barrier arm file, rank marker, corpus, and worker artifacts must be on
  shared `/workspace`, never node-local `/tmp`.
- External MPI management workers start before rank 0 launches `run.py`.
  Therefore `dsv4_fp4_gb300_trt.sh` must export the complete output of
  `emit_rank_environment.py` before `trtllm-llmapi-launch`; setting dynamic
  `TRTLLM_BENCH_*` values only in `run.py` leaves ranks 1-15 without the
  benchmark contract. Preserve rank 0's exact preseed validation.
- Before engine launch, require four nodes with four ranks each and four
  `Completed`/`Success` NVLink Fabric records per node. Parse full
  `nvidia-smi -q` output, not the unsupported `-d FABRIC` selector, and
  require one shared non-empty `ClusterUUID` and `CliqueId` across all 16
  GPUs. Preserve `offline_rank_map_gbsN.tsv`,
  `offline_topology_gbsN.log`, `offline_allocation_gbsN.log`, and per-node
  GPU telemetry. The host launcher must stream `salloc` output and emit a
  one-minute pending-allocation heartbeat so queue time is not mistaken for
  TRT initialization. After rank 0 has copied the result and archived debug
  files, it must atomically publish `offline_completion_gbsN.json` with the
  controller return code and result status; the host verifies both before
  canceling the allocation so external MPI management ranks cannot hold the
  job until its Slurm time limit.
- The dispatchable workflow is `.github/workflows/e2e-tests.yml` with
  `inputs[hardware-profile]=gb300` and
  `inputs[benchmark-profile]=huawei`; matrix concurrency is intentionally one.
- Keep the self-hosted checkout isolated under
  `${GITHUB_WORKSPACE}/offline-bench` and use `TRT_BENCH_WORKSPACE` for
  source, output, mounts, and artifacts. Do not clean the runner's repository
  root. Run `27511740130` inherited open `.nfs*` logs from an unrelated
  srt-slurm job; root-level `git clean -ffdx` spent ten minutes deleting that
  tree and then failed with `EBUSY` before GBS64 launched.
- Treat GB300 controller success as ready only when both
  `offline_result_gbsN.json` and `offline_completion_gbsN.json` are visible
  and their statuses agree. The rack workspace has roughly 50-second negative
  NFS lookup caching: run `27513364142` completed GBS64 successfully inside
  the container, but the cleanly exited MPI world exposed the files just
  after the old five-second host grace expired. Preserve the bounded
  `TRT_BENCH_COMPLETION_VISIBILITY_TIMEOUT` wait and its progress logs.
- Pinned TRT can enqueue inactive iteration-stat tails after `get_stats()`
  marks its queue done, then expose them when the next submission calls
  `mark_undone()`. Run `27514818464` returned one warmup tail before the
  measured prefill. The selector may ignore only leading rows with the exact
  fixed local generation/scheduled count and `active=queued=paused=0`; record
  their count/range and reject every active, partial, queued, or mixed row.
- `perfect_router.jsonl` is shared by all 16 ranks over the workspace
  filesystem. Every writer must use `io_utils.append_json_line`, every strict
  reader must take the matching lock, and malformed lines are a validation
  failure. Do not restore plain text append: run `27511242827` produced 12
  corrupted records from concurrent writes. Host-side progress must inspect
  the `${GITHUB_WORKSPACE}` path, not the container-only `/workspace` alias.

## PR-Max GB300 Profiles

- Select `TRT_BENCH_CONFIG_PROFILE=pr-tp32-mtp3`,
  `pr-tp16-mtp3`, or `pr-tp8-mtp1`. Workflow alias `pr-max-sweep` runs all
  three sequentially and does not fail fast.
- Run both PR-active and capacity points: TP32 GBS192/256 (local6/8),
  TP16 GBS400/512 (local25/32), and TP8 GBS3440/4096 (local430/512).
  Engine `max_batch_size` remains 8/32/512 and the full copied graph list is
  preserved at both points.
- Preserve each PR recipe's full CUDA graph list, KV fraction, EPLB map,
  overlap scheduler, `print_iter_log`, PDL, LM-head TP, attention DP,
  `MEGAMOE_DEEPGEMM`, and low-precision combine.
- Preserve the learned model router and omit `attention_dp_config`, matching
  the resolved PR engines. Never enable either perfect-router alias in a
  `pr-*` profile.
- Do not add heuristic sparse attention. The resolved PR decode configs have
  no `sparse_attention_config`.
- PR decode-only `max_num_tokens` is 32/128/1024 because those workers receive
  transferred KV. Direct offline prefill uses the explicit 32768-token
  adaptation. Every result must record both values.
- PR-max skips the separate request warmup. Engine graph warmup runs during
  initialization. The measured request pass permits staged/mixed setup, then
  requires 256 consecutive exact full-batch decode iterations and discards
  the first eight.
- Under overlap, `iterLatencyMS` is diagnostic only. The controller must use
  rank-0 TRT `host_step_time` for the selected IDs. Never publish overlap
  `iterLatencyMS` as PR-max TPOT.
- The PR disables iteration perf stats. Offline PR-max intentionally enables
  a bounded 2048-row history only for exact-window proof and MTP yield; this
  instrumentation must not become the headline timing source.
- Within the selected window, context is zero and active, scheduled, and
  generation counts equal the exact local batch with no queued or paused
  requests.
- Dispatch with `inputs[benchmark-profile]=pr-max-sweep` and
  `inputs[global_batch_sizes]=auto`. Experiment IDs include profile and GBS,
  for example `pr-tp8-mtp1-gbs4096`.

## GB300 NVL72 Rack Profile

- Select workflow profile `rack-huawei-sweep` for rack GBS
  `72/288/576`, or `rack-max-sweep` for `30960/36864`. The internal engine
  profile is `rack-tp8-mtp1-engine`.
- Allocate exactly 18 four-GPU nodes. Require the full rank map `0..71`,
  four local ranks per host, and one non-empty Fabric `ClusterUUID` and
  `CliqueId` across all 72 GPUs before launching any engine.
- Partition the allocation into nine disjoint adjacent node pairs. Each pair
  runs one independent external-MPI TP8/EP8 MTP1 engine with ranks `0..7`.
  Report the result as `9xDEP8`; never describe it as TP72. All 72 GPUs are
  active, but TensorRT/NCCL collectives remain within each pair.
- Preserve PR #1689 run `27164980476` attempt 14's TP8 engine details:
  max batch 512, the complete CUDA graph list through 512, KV fraction 0.80,
  overlap scheduling, learned router, EP8/384 EPLB, PDL,
  `MEGAMOE_DEEPGEMM`, serialized weight loading,
  `MIMALLOC_PURGE_DELAY=-1`, and full `/dev/shm`.
- Rack GBS72/288/576 preserve Huawei local batches 1/4/8 per GPU. Rack
  GBS30960/36864 are nine times the copied TP8 active/capacity points.
  Huawei is MTP3 and this engine is MTP1; compare raw decode-step throughput
  before separately applying measured MTP yield.
- Each child proves its own 256-round exact fixed-batch window. All children
  must reach one shared measured-pass barrier. For logical rack round `i`,
  use the maximum rank-0 `host_step_time` across the nine child round-`i`
  values, then skip eight logical rounds and apply the upper-IQR filter.
  Reject measured-pass start skew above 10 seconds; the initialization
  barrier timeout is one hour.
- Child results and logs must remain under
  `.offline_rack_ID_JOB/replicas/rNN` and enter the uploaded debug archive
  only. Publish exactly one top-level `offline_result_ID.json`, so
  `collect-results` cannot mistake child engines for benchmark rows.
- A rack run is invalid if any child fails, any barrier index is missing, the
  measured start times are not synchronized, child fabric provenance differs,
  or the parent row lacks all nine child result proofs.

## B300 Baseline Constraints

- Start KV capacity from
  `kv_cache_config.free_gpu_memory_fraction=0.60`. An explicit KV token cap
  underprovisions the pinned DeepSeek-V4 multi-pool cache and staggered half
  of each local batch in run `27486168511`. For GBS128, subtract exactly
  12 GiB from TRT's calibrated `max_gpu_total_bytes` after capacity
  estimation and before final manager construction. Preserve at least TRT's
  aggregate `CacheCost.bytes_for_tokens(16 * 9344)`; fail rather than reducing
  that minimum. GBS16/64 use no reserve. Run `27491999545` proved why the
  reserve is required: final KV allocation left 1.17-2.35 GiB free before the
  real prefill requested an 8 GiB FP8 Q buffer and a following roughly 2 GiB
  BF16 RoPE buffer. The fixed
  `moe_config.max_num_tokens=65536` cap chunks only oversized fused-MoE
  tensors inside that one executor iteration; measured decode never reaches
  the cap.
- Keep the synthetic pure-context warmup request cap at 65536 tokens by
  wrapping `PyTorchModelEngine._create_warmup_request`. Never mutate
  `engine.max_num_tokens`: DeepSeek-V4 attention metadata allocates lazily
  from it, and run `27490833024` created undersized 65536-token runtime buffers
  that failed on the 84087-token KV-capacity probe. The runtime field must stay
  131072 for GBS128. Never treat the synthetic request cap as permission to
  split or shrink the validated full-batch prefill.
- Keep the GBS128 eager attention workspace reservation at
  `200 KiB * 131072 + 256 MiB = 27111981056` bytes. Run `27491160719` showed
  the capped warmup allocate 12953234944 bytes, then TRT attempted an in-place
  resize to 16600658432 bytes for its 84087-token capacity probe and hit an
  illegal memory access on every rank. Reserve only
  `TrtllmAttentionMetadata.workspace` when the cached runtime metadata is
  created. Do not preallocate or replace `cuda_graph_workspace`.
- GBS128 initialization builds two engine phases and resets attention
  metadata between them. Duplicate workspace events are expected. The KV
  reserve event occurs after phase-one calibration and must cover ranks
  `0..7` exactly.
- Keep the rank-local packed-FP8 guard at 32768 rows. It selects TRT's Triton
  quantizer only for large GBS64/128 prefill projections; measured decode has
  at most 16 rows and stays on the original fused path.
- Keep `fp8SwapABGemmRunner` chunking at 65536 rows. Run `27492438399`
  completed initialization and reached real GBS128 prefill, then failed after
  launching the Triton quantizer on all 131072 rows. The hook allocates one
  final output and writes two row chunks through the existing transformed
  DeepGemm weights. Synchronize oversized chunks for precise failures.
  Decode has at most 16 rows and must call the pinned runner unchanged.

## Shared Measurement Invariants

- `LLM.generate()` submits requests individually. The MPI entry shim patches
  TRT's idle request fetch so each warmup/measured pass waits for exactly one
  complete GBS before routing. Do not remove that barrier while claiming a
  fixed-batch result.
- Keep the request barrier disarmed throughout `LLM` construction. TRT's
  GBS128 KV-capacity calibration submits 120 internal dummy requests, which
  must pass through normally. The parent atomically creates the shared arm
  file only after initialization returns and before the first real benchmark
  `generate()` call. Do not arm at MPI entry or weaken the post-arm gate.
- Huawei success proves one full-local-batch prefill followed by 256 exact
  decode iterations. PR-max success permits setup prefill staging but proves
  the same 256-iteration exact decode window. Never weaken either proof.
- Keep `max_stats_len=2048`. It bounds the stats history while covering the
  zero-acceptance worst case of 1024 decode iterations.
- Huawei timing uses TRT `iterLatencyMS` with overlap disabled and skips one
  round. PR-max timing uses rank-0 `host_step_time` with overlap enabled and
  skips eight. Rack timing takes the maximum same-index PR-max
  `host_step_time` across all nine engines before skipping eight. All discard
  only values above `Q3 + 1.5 * IQR`.
- The headline metric is raw decode-round throughput:
  `GBS / decode_round_TPOT / active_gpu_count`. MTP output yield and
  acceptance are separate.
- Huawei uses a 1025-token cap. PR-max computes a profile-specific cap that
  keeps the earliest requests alive through staged setup and 256 full-batch
  rounds without exceeding `max_seq_len=9256`.
- Preserve exact 8192-token real prompts, temperature 1, engine-global seed
  42, LM-head TP, and rank environment validation. Huawei uses perfect
  routing; PR-max uses the learned router. Heuristic sparse top-k belongs only
  to Huawei. ConfigurableMoE remains enabled where required by the GB300
  `MEGAMOE_DEEPGEMM` path.
- Do not enable `TLLM_METRICS_ALL_RANKS`; its per-iteration collective changes
  the timing path. Rank 0 stats are valid only after exact equal-length prompt
  routing and full-local-batch validation.
- The comparison is methodological, not identical hardware: this branch uses
  B300 or GB300 FP4 GPUs, while Huawei publishes 950DT chips with hybrid
  MXFP8/MXFP4.
- Final known-good B300 sweep is Actions run `27493336994` at
  `9796f5d17c96ab56136b8b9b1e196b6e6db84426`. Its GBS16/64/128 raw
  decode-step rates are `90.621203`, `248.910481`, and `434.410801`
  steps/s/GPU. Use it as the B300 regression baseline.
- Final known-good GB300 sweep is Actions run `27517035480` at
  `c0a845521b51e5fb5eca5f9bb4ac2e3a6c60b43d`. Its GBS16/64/128 raw
  decode-step rates are `38.578364`, `123.551461`, and `233.703196`
  steps/s/GPU; output rates are `115.885788`, `433.636669`, and
  `806.321670` tok/s/GPU. All rows passed exact rank, fabric, fixed-prefill,
  256-round, marker, completion, aggregate, and renderer validation. Use it
  as the GB300 regression baseline.
- `offline_aggregate.json` is authoritative. `results_bmk/agg_bmk.json` uses
  acceptance-adjusted output-token throughput and equivalent output-token
  TPOT for standard renderer fields, while retaining custom decode-round
  fields. The flat `conc` field aliases `global_batch_size` for renderer
  compatibility; it is not serving concurrency.
- Keep launcher/controller/worker phase logs and 60-second heartbeats.
  Heartbeats must retain the per-rank marker summary for warmup, clock sync,
  and executor worker start; canceled Actions jobs may not preserve the
  node-local worker log.
- Treat TRT's `Fatal error detected, initiating shutdown` line as terminal.
  The MPI parent can remain alive after all ranks fail, so waiting for the
  full controller timeout only hides the real error and wastes the node.
- Treat rank `entry_failed` and `*_error` marker events as terminal for the
  same reason. Run `27504087069` otherwise waited 1800 seconds after all 16
  external ranks failed immediately on a missing warmup-cap variable.
- For a readiness hang, dispatch with `worker-stack-period=120` and a short
  controller timeout, then let the controller exit on its own so the debug
  archive contains TRT's all-thread stack dumps. Keep the normal default at
  `-1`; stack dumping changes logging overhead and is diagnostic only.
- Verify with `python -m pytest utils/bench_offline -v`,
  `python -m compileall utils/bench_offline`, `bash -n` and `shellcheck` on
  every offline launcher, and `actionlint .github/workflows/e2e-tests.yml`.
- A GPU benchmark is not valid until its Actions artifact proves the schedule
  validation, timing filter, exact rank set, and result values.
