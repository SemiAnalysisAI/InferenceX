# Offline TensorRT-LLM Agent Notes

Read `README.md` and `/TRT_BENCH_NOTES.md` before changing or debugging this
benchmark.

## Scope

- This is disposable branch-only code. It must remain isolated from the
  normal serving sweep.
- Never edit `.github/configs/nvidia-master.yaml` or `perf-changelog.yaml`.
- `.github/workflows/e2e-tests.yml` is intentionally replaced on this branch.
- A benchmark is valid only when its Actions artifact proves the exact batch,
  schedule, timing source, rank set, fabric, and completion state.

## Shared Invariants

- `global_batch_size` is authoritative. `conc` is only a renderer alias.
- Prompts contain exactly 8192 real DeepSeek-V4 chat-formatted token IDs.
- The fixed-batch request gate is installed before engine construction but
  remains disarmed during TRT initialization and calibration.
- Rank 0 arms the gate only after `LLM(...)` returns. Every real pass waits
  for the exact complete batch before routing.
- Every headline result requires 256 consecutive full-local-batch decode
  iterations with context, queued, and paused counts equal to zero.
- MTP proposed/accepted counters must cover the same selected iterations.
  Never infer headline yield from whole-request telemetry.
- Raw throughput is
  `GBS / decode_round_tpot_seconds / active_gpu_count`.
- Output throughput is raw throughput times measured tokens per step.
- Huawei timing uses non-overlap `iterLatencyMS` and skips one round.
- PR and rack timing use rank-0 `host_step_time` and skip eight rounds.
- Rack timing takes the maximum same-index host-step value across all nine
  engines before filtering.
- Filtering drops only values above `Q3 + 1.5 * IQR`.
- Keep `max_stats_len=2048`.
- Do not enable `TLLM_METRICS_ALL_RANKS`; its collective changes timing.
- Keep launcher, controller, and worker phase logs plus 60-second heartbeats.
- Treat TRT fatal lines, `entry_failed`, and rank `*_error` markers as
  terminal even when the external MPI parent remains alive.

## Huawei Profiles

- B300 uses one eight-GPU DEP8 engine and TRT source `c185066`.
- GB300 uses one four-node, 16-GPU DEP16 engine, image
  `nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1`,
  and TRT source `34a563ac6d8cc0ca7068c7f619e869fb8a625333`.
- GBS is 16/64/128. GB300 local batch is 1/4/8; B300 is 2/8/16.
- Huawei mode uses MTP3, perfect routing, overlap disabled, two request
  warmup rounds, one exact context-only full-batch prefill, and a 1025-token
  output cap.
- Before GB300 engine launch, require ranks `0..15`, four ranks on each of
  four hosts, Fabric `Completed`/`Success` on all GPUs, and one shared
  non-empty `ClusterUUID` and `CliqueId`.
- External MPI workers start before rank 0 creates the later worker process.
  Preserve the complete environment preseed and rank-0 validation.
- Arm, corpus, marker, result, and completion paths must use shared
  `/workspace`, not node-local `/tmp`.
- Treat success as visible only after result and completion JSON both exist
  and agree. Preserve the bounded NFS visibility wait.
- The iteration selector may ignore only a leading inactive tail with the
  exact fixed generation/scheduled count and
  `active=queued=paused=0`. Reject any other pre-prefill row.

## PR Profiles

- `pr-tp32-mtp3`: GBS192/256, TP32/EP32, max batch 8.
- `pr-tp16-mtp3`: GBS400/512, TP16/EP16, max batch 32.
- `pr-tp8-mtp1`: GBS3440/4096, TP8/EP8, max batch 512.
- Preserve the complete attempt-14 CUDA graph list, KV fraction, EPLB map,
  overlap scheduler, learned router, unset `attention_dp_config`, PDL,
  LM-head TP, MegaMoE backend, and low-precision combine.
- Never enable perfect routing or heuristic sparse attention in a PR profile.
- Direct offline prefill uses the explicit `max_num_tokens=32768`
  adaptation. Do not substitute the serving worker's transferred-KV limit.
- PR profiles skip request warmup and permit staged setup, then require the
  same exact 256-round decode window.
- Overlap `iterLatencyMS` is diagnostic only. Publish rank-0
  `print_iter_log` `host_step_time`.

## Rack Profile

- Workflow `rack-huawei-sweep` runs GBS72/288/576.
- Workflow `rack-max-sweep` runs GBS30960/36864.
- Allocate exactly 18 four-GPU nodes and prove ranks `0..71` plus one shared
  72-GPU Fabric UUID/clique before launching engines.
- Partition into nine disjoint adjacent node pairs. Each pair is one
  independent TP8/EP8 MTP1 world with ranks `0..7`.
- Report `9xDEP8`, never TP72.
- Preserve attempt 14's TP8 engine: max batch 512, CUDA graphs through 512,
  KV fraction 0.80, overlap, learned router, EP8/384 EPLB, PDL,
  MegaMoE, default parallel weight loading, and `MIMALLOC_PURGE_DELAY=0`.
- Do not copy prefill-only serialized loading,
  `MIMALLOC_PURGE_DELAY=-1`, or expandable allocator settings into decode
  engines.
- Admit one engine at a time through model loading. Launch the next only
  after all eight ranks emit `engine_warmup_start`.
- Keep the 600-second model-load deadline, three attempts, and 15-second
  retry delay. Retrying one child must not restart initialized engines.
- All nine children must reach the measured-pass barrier. Replica 0 publishes
  one common start 90 seconds in the future. Reject start skew above
  10 seconds.
- Child results remain inside the rack debug archive. Publish exactly one
  top-level 72-GPU result.
- A rack row is invalid if any child fails, a barrier index is missing,
  child fabric differs, fewer than nine timing vectors exist, or the parent
  result lacks all child proofs.

## B300 Workarounds

These apply only to the pinned B300 image:

- Start KV capacity from memory fraction 0.60. Do not set an explicit KV
  token cap.
- GBS128 keeps runtime `max_num_tokens=131072` while capping only TRT's
  synthetic pure-context warmup request at 65536.
- GBS128 subtracts exactly 12 GiB from calibrated
  `max_gpu_total_bytes`, after proving the adjusted cache still covers
  `16 * 9344` tokens.
- GBS128 reserves exactly 27111981056 bytes on
  `TrtllmAttentionMetadata.workspace`. Do not replace
  `cuda_graph_workspace`.
- Keep the packed-FP8 quantizer guard above 32768 rows.
- Keep `fp8SwapABGemmRunner` prefill chunking at 65536 rows. Decode rows stay
  on the original implementation.
- These workarounds must not split the executor-level full-batch prefill or
  change the selected decode path.

## Huawei Comparison

- Huawei reference raw rates are 56.70, 210.16, and 388.23 decode
  steps/s/chip for GBS16/64/128.
- Rack GBS72/288/576 match Huawei local batches 1/4/8, not Huawei global
  batches.
- Compare raw steps first.
- For an output-token statement with equal MTP yield, multiply Huawei's raw
  rate by GB300's measured `observed_tokens_per_step`.
- The same-yield output ratio must equal the raw step ratio.
- Keep the own-yield comparison separate because Huawei publishes 2.44
  tokens/step while the copied TP8 engine uses MTP1.

## Baselines

- B300 fixed GBS: run `27493336994`, source `9796f5d1`.
  Raw rates for GBS16/64/128 are 90.621203, 248.910481, and
  434.410801 steps/s/GPU.
- GB300 NVL16 fixed GBS: run `27517035480`, source `c0a84552`.
  Raw rates are 38.578364, 123.551461, and 233.703196 steps/s/GPU.
- GB300 rack maximum: run `27545752641`, source `775a1451`.
  GBS30960/36864 output rates are 9468.968467 and 10262.766175
  tok/s/GPU. GBS36864 is the branch maximum.
- GB300 rack Huawei-local-batch: run `27555308252`, source `0ec5b7d4`.
  GBS72/288/576 raw rates are 53.504297, 172.497882, and
  298.775209 steps/s/GPU. Same-yield Huawei ratios are 0.943638,
  0.820793, and 0.769583.
- Exact values, flat rows, retries, and renderer links are in
  `/TRT_BENCH_NOTES.md`.

## Artifact Checks

- `offline_aggregate.json` is authoritative.
- `agg_bmk.json` must be a top-level array of flat rows.
- Standard renderer throughput and TPOT fields are acceptance-adjusted.
- Custom fields retain raw round TPOT, raw step throughput, fixed GBS, local
  batch, token yield, round count, and timing source.
- On GB300, require exact rank map, expected host count, four local ranks per
  host, one Fabric UUID/clique, and matching successful completion record.
- Rack results additionally require all nine child schedule proofs and the
  synchronized barrier record.

## Verification

```bash
python -m pytest utils/bench_offline/ -q
python -m compileall -q utils/bench_offline
bash -n benchmarks/single_node/offline/dsv4_fp4_b300_trt.sh
bash -n benchmarks/single_node/offline/run_dsv4_trt_container.sh
bash -n benchmarks/multi_node/offline/dsv4_fp4_gb300_trt.sh
bash -n benchmarks/multi_node/offline/dsv4_fp4_gb300_rack_trt.sh
bash -n benchmarks/multi_node/offline/dsv4_fp4_gb300_rack_replica.sh
shellcheck benchmarks/single_node/offline/dsv4_fp4_b300_trt.sh
shellcheck benchmarks/single_node/offline/run_dsv4_trt_container.sh
shellcheck benchmarks/multi_node/offline/dsv4_fp4_gb300_trt.sh
shellcheck benchmarks/multi_node/offline/dsv4_fp4_gb300_rack_trt.sh
shellcheck benchmarks/multi_node/offline/dsv4_fp4_gb300_rack_replica.sh
actionlint .github/workflows/e2e-tests.yml
git diff --check
```
