# DeepSeek-V4 TRT Offline Benchmarks

This is a branch-only benchmark. It does not use the normal InferenceX serving
pipeline and is not intended to merge into `main`.

It has three deliberately separate contracts:

- `huawei` preserves the validated Huawei-style fixed-GBS method.
- `pr-tp32-mtp3`, `pr-tp16-mtp3`, and `pr-tp8-mtp1` copy the three final
  GB300 decode recipes from InferenceX PR #1689 and measure maximum saturated
  offline decode throughput.
- `rack-tp8x9-mtp1` runs nine synchronized copies of the PR's fastest TP8
  decode engine across one 72-GPU GB300 NVLink domain. It measures both
  Huawei-equivalent local batches and full-engine saturation.

Do not compare or combine their timing fields without checking
`benchmark_profile` and `aggregate.timing_source`.

## Huawei-Style Contract

The benchmark reproduces the measurement contract implemented by Huawei's
DeepSeek-V4 offline recipe:

- fixed global batch
- full-batch prefill before decode
- two warmup decode rounds
- 256 measured decode rounds
- MTP3
- first measured round skipped
- upper-IQR outlier removal
- raw decode-round TPOT
- throughput calculated as `global batch / TPOT / devices`
- MTP accepted-token yield reported separately

The reference implementation is:

```text
/Users/bshan/Documents/cann-recipes-infer-master/
  docs/models/deepseek-v4/deepseek_v4_inference_guide.md
  models/deepseek-v4/models/model_infer.py
  executor/utils/common_utils.py
```

The hardware is not identical. Huawei's published DeepSeek-V4 Pro table uses
16 950DT chips and hybrid MXFP8/MXFP4. This branch has two FP4 TRT profiles:
the validated one-node B300 DEP8 baseline and the current four-node GB300
NVL16 DEP16 target.

## Fixed Workload

| Setting | Value |
|---|---|
| Model | `/scratch/models/DeepSeek-V4-Pro` |
| Global batch sizes | `16`, `64`, `128` |
| Input length | exactly 8192 real token IDs |
| MTP depth | 3 draft tokens |
| Warmup | at least two full-batch decode rounds |
| Measurement | first 256 consecutive full-batch decode rounds |
| Sampling | temperature 1, top-p 1, top-k 0, EOS ignored |
| Seed | TRT engine-global seed 42 |
| Routing | perfect router |
| Overlap scheduler | disabled |

| Profile | Nodes x GPUs | TRT image/source | Parallelism | MoE/KV |
|---|---:|---|---|---|
| `b300` | `1 x 8` | `ghcr.io#semianalysisai/trtllm-deepseek-v4:feat-deepseek_v4-c185066`, `c185066` | DEP8, TP8, EP8, attention DP | TRTLLM MoE, KV fraction `0.60` |
| `gb300` | `4 x 4` in one NVLink domain | `nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1`, `34a563ac6d8cc0ca7068c7f619e869fb8a625333` | DEP16, TP16, EP16, attention DP | ConfigurableMoE + `MEGAMOE_DEEPGEMM`, EP16/384-slot load balancer, KV fraction `0.70` |

The GB300 runtime/topology reference is
[InferenceX PR #1689](https://github.com/SemiAnalysisAI/InferenceX/pull/1689).
The offline path does not use Dynamo or srt-slurm; it reuses that PR's working
image, four-node task layout, `trtllm-llmapi-launch`, and decode-engine
configuration.

The pinned TRT source defaults `ENABLE_CONFIGURABLE_MOE=1`, and the reference
recipe leaves that default intact. The MegaMoE backend requires the
`ConfigurableMoE` wrapper to provide the concrete forward scheduler.

The prompt is built from pinned InfiniteBench `longbook_qa_eng.jsonl` data and
DeepSeek-V4 chat formatting. Prompt construction fails instead of inserting
pad or synthetic token IDs.

## PR-Max GB300 Profiles

These profiles reproduce the PR's decode topology, CUDA graph set, MTP depth,
KV fraction, MegaMoE backend, and EPLB map. They do not start Dynamo or
dedicated prefill workers. The same decode GPUs first admit the 8K prompts,
then the benchmark selects a saturated decode-only window.

They also match the resolved PR engines by using the learned model router and
leaving `attention_dp_config` unset. Perfect routing is Huawei-profile-only;
using it here would bypass the learned router, invalidate output correctness,
and make MTP acceptance incomparable with the PR.

| Profile | Nodes x GPUs | Offline GBS (PR-active, capacity) | Local batch/rank | MTP | CUDA graphs | KV fraction | PR concurrency |
|---|---:|---:|---:|---:|---|---:|---:|
| `pr-tp32-mtp3` | `8 x 4` | 192, 256 | 6, 8 | MTP3 | `1,2,4,8` | 0.70 | 333 |
| `pr-tp16-mtp3` | `4 x 4` | 400, 512 | 25, 32 | MTP3 | `1,2,4,8,16,24,32` | 0.70 | 666 |
| `pr-tp8-mtp1` | `2 x 4` | 3440, 4096 | 430, 512 | MTP1 | `1,2,4,8,16,24,32,40..512` | 0.80 | 4301 |

Attempt 14's output throughput times mean TPOT implies about 190, 401, and
3438 active decode requests. Those are rounded to rank-divisible fixed GBS
192, 400, and 3440. Each profile also runs its engine-capacity endpoint. The
engine `max_batch_size` and full CUDA graph list remain copied from the PR for
both points. Neither fixed GBS is the PR's HTTP concurrency, which also
includes requests in prefill and queues.

The PR decode workers receive transferred KV and therefore use decode-only
`max_num_tokens` values of 32, 128, and 1024. A direct offline worker must
prefill locally, so all three profiles use `max_num_tokens=32768`. This is the
only intentional workload-capacity adaptation and is recorded in every
result.

The PR engines disable iteration performance stats. Offline PR-max enables a
bounded 2048-row stats history only to prove the exact full-batch window and
measure accepted-token yield. Headline timing comes from the PR's existing
rank-0 `print_iter_log` `host_step_time`, not from iteration-stat latency.

PR-max uses overlap scheduling and one request pass. TRT engine initialization
warms the CUDA graphs; the benchmark then allows staged/mixed prefill setup,
requires 256 consecutive exact full-batch decode iterations, discards the
first eight, and applies the upper-IQR filter.

`iterLatencyMS` spans the wrong scope under overlap. The controller therefore
parses rank-0 `print_iter_log` output and uses `host_step_time` for
`decode_round_tpot_ms`. The original iteration-stat result remains under
`aggregate.stats_iter_latency_diagnostic`. MTP yield still comes from the
same selected iteration-stat window.

Attempt 14 reference output rates are stored with each profile:

| Profile | Output tok/s/decode-GPU | Output tok/s/all PR GPUs |
|---|---:|---:|
| `pr-tp32-mtp3` | 676.763 | 451.175 |
| `pr-tp16-mtp3` | 2072.1585 | 828.8634 |
| `pr-tp8-mtp1` | 9686.735 | 1383.8193 |

The result reports both decode-GPU comparison and a hypothetical PR-fleet
normalization. The latter divides offline total output by the PR's decode plus
prefill GPU count; those prefill GPUs were not allocated by the offline run.

## GB300 NVL72 Rack Profile

`rack-tp8x9-mtp1` scales the fastest resolved recipe from
[PR #1689 attempt 14](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27164980476/attempts/14)
to the complete 18-node, 72-GPU GB300 allocation:

- nine independent TP8/EP8 MTP1 TensorRT engines
- two adjacent four-GPU nodes and eight ranks per engine
- one Slurm allocation spanning all 18 nodes
- one required Fabric `ClusterUUID` and `CliqueId` across all 72 GPUs
- the attempt-14 image, TRT revision, EP8/384 EPLB map, learned router,
  overlap scheduler, KV fraction `0.80`, and CUDA graphs through batch 512
- the recipe's decode environment: default parallel weight loading,
  `MIMALLOC_PURGE_DELAY=0`, and no prefill-only expandable allocator setting

This is intentionally reported as `9xDEP8`, not TP72. Every GPU is active,
but each TensorRT/NCCL world is confined to its assigned two-node pair. TRT's
model-parallel world size is `tp * pp * cp`; the copied TP8 engine cannot be
made into a 72-rank engine by changing MoE cluster settings.

| Rack GBS | Per-engine GBS | Local batch/GPU | Purpose |
|---:|---:|---:|---|
| 72 | 8 | 1 | Huawei GBS16 local-batch match |
| 288 | 32 | 4 | Huawei GBS64 local-batch match |
| 576 | 64 | 8 | Huawei GBS128 local-batch match |
| 30960 | 3440 | 430 | 9x attempt-14 active decode population |
| 36864 | 4096 | 512 | 9x copied engine capacity |

Model loading is admission-controlled to one engine at a time. The next
replica starts after all eight ranks of the current replica emit
`engine_warmup_start`, so CUDA graph capture overlaps across replicas while
safetensor reads remain serialized. Run `27533885582` proved that launching
all 72 ranks together can block one rank; run `27535038325` reproduced the
same shard-43 stall with three concurrent engines. Run `27537092211` then
stalled one serialized engine at shard 31, so each child gets up to three
bounded model-load attempts without restarting engines already at warmup or
the barrier. Replica 0 publishes a common start 90 seconds after the release
file, so NFS metadata visibility cannot stagger the measured calls. The
measured pass remains one synchronized nine-engine release.

Logical rack decode round `i` uses the maximum rank-0 `host_step_time` for
round `i` across the nine engines. The rack result then skips the first eight
logical rounds and applies the upper-IQR filter. This models a fixed global
batch whose next step cannot complete before its slowest replica. Aggregation
rejects measured-pass start skew above 10 seconds.

Speculative proposed/accepted counters are summed across replicas for the
same 256-round window. Huawei uses MTP3 and the copied TP8 engine uses MTP1,
so raw decode-step throughput and observed output-token yield remain separate.
The 72/288/576 rows compare against Huawei GBS16/64/128 by equal local batch,
not equal global batch.

The final maximum-throughput sweep is
[Actions run 27545752641](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27545752641)
at source `775a1451074966b871f1cbd57229894d393f4af0`:

| Rack GBS | Local/GPU | Rack step ms | Steps/s/GPU | Tok/step | Output tok/s/GPU | Versus PR TP8 |
|---:|---:|---:|---:|---:|---:|---:|
| 30960 | 430 | 82.041011 | 5241.281069 | 1.806613 | 9468.968467 | -2.248095% |
| 36864 | 512 | 90.548343 | 5654.438070 | 1.814993 | 10262.766175 | +5.946593% |

GBS36864 is the best validated GB300 result. It fills the copied batch-512
engine and exceeds attempt 14's `9686.735465` output tok/s/decode-GPU by
`5.946593%`, even though the rack metric takes the slowest of nine
synchronized engines for every logical round. The individual TP8 replicas
ranged from `10356.040023` to `10538.418796` output tok/s/GPU.

The renderer-compatible result is available at the
[InferenceMAX unofficial run](https://inferencemax-r4i4xgna4-semianalysisai.vercel.app/inference?unofficialrun=27545752641).
The exact flat rows and retry details are recorded in
[TRT_BENCH_NOTES.md](../../TRT_BENCH_NOTES.md).

## Global And Local Batch

There is one authoritative `global_batch_size`. Every per-rank capacity is
derived from it and the selected profile's attention-DP width. For `huawei`:

```text
local_batch_size = global_batch_size / active_gpu_count
max_batch_size = local_batch_size
cuda_graph_batch_size = local_batch_size
max_num_tokens = local_batch_size * 8192
synthetic pure-context warmup request =
    min(max_num_tokens, profile_and_batch_warmup_cap)
minimum runtime KV tokens = local_batch_size * 9344
```

For PR-max, engine `max_batch_size` remains the copied 8, 32, or 512. The
PR-active points therefore run local batches 6, 25, or 430 below those caps;
the capacity points run 8, 32, or 512. All use `max_num_tokens=32768`,
`max_seq_len=9256`, and may span multiple setup iterations before the exact
full-batch decode window.

### B300 Capacity

| GBS | Local batch/rank | TRT max_num_tokens/rank | Synthetic warmup | Minimum KV tokens/rank | KV reserve |
|---:|---:|---:|---:|---:|---:|
| 16 | 2 | 16384 | 16384 | 18688 | 0 |
| 64 | 8 | 65536 | 65536 | 74752 | 0 |
| 128 | 16 | 131072 | 65536 | 149504 | 12 GiB |

### GB300 Capacity

| GBS | Local batch/rank | TRT max_num_tokens/rank | Synthetic warmup | Minimum KV tokens/rank | KV reserve |
|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 8192 | 8192 | 9344 | 0 |
| 64 | 4 | 32768 | 32768 | 37376 | 0 |
| 128 | 8 | 65536 | 32768 | 74752 | 0 |

GB300 GBS128 retains a 65536-token runtime capacity and real full-batch
prefill, but tunes only a 32768-token synthetic context shape. Run
`27516149323` reached the guarded 65536-token warmup, then exhausted memory
while TRT created temporary KV-estimation resources. GBS64 already proves the
same image and operators initialize at 32768 tokens. The B300-only eager
attention workspace, 12 GiB KV reduction, and oversized DeepGemm chunker
remain disabled.

### B300 Pinned-Image Workarounds

The remaining capacity history in this section applies to the pinned B300
`c185066` image.

KV capacity starts from TRT's memory-derived calibration at a fixed 60%
fraction. Do not set `kv_cache.max_tokens`: the pinned DeepSeek-V4 multi-pool
cache manager did not translate the exact-sequence token quota into enough
physical capacity for the fixed batch, and run `27486168511` admitted only
half of each local batch into prefill. At GBS128 only, the MPI shim subtracts
12 GiB from TRT's calibrated `max_gpu_total_bytes` before final cache
construction. It first uses TRT's own aggregate `CacheCost` to prove the
adjusted budget still covers all 16 sequences through `max_seq_len=9344`,
or initialization fails. GBS16 and GBS64 retain the unmodified calibrated
budget.

The fixed 65536-token MoE cap applies inside a fused-MoE invocation. It lets
TRT internally chunk the very large prefill/autotune tensor while the executor
still schedules the complete local batch in one prefill iteration.

TRT's internal `PyTorchModelEngine.warmup()` otherwise profiles tunable
operators with the full runtime `max_num_tokens`. At GBS128 that means a
synthetic 131072-token prefill shape and can spend hours in initialization.
The branch-local MPI shim leaves `engine.max_num_tokens=131072` unchanged and
caps only pure-context requests created by TRT's
`_create_warmup_request()` at 65536 tokens. The distinction is required:
DeepSeek-V4 sparse-attention metadata is allocated lazily from
`engine.max_num_tokens` on the first forward. Temporarily lowering that field
allocated 65536-token buffers and made the later 84087-token KV-capacity probe
fail in run `27490833024`. Runtime buffers now retain full capacity. A result
is still accepted only if the schedule proof shows one complete
local-batch-16 prefill and 256 consecutive local-batch-16 decode iterations.

Run `27491160719` proved the runtime metadata fix and completed all-rank engine
warmup in about 335 seconds. It then exposed a separate pinned TRT defect:
the non-graph MLA attention workspace had been sized to 12,953,234,944 bytes
by the capped 65536-token warmup. The later 84087-token capacity probe tried
to resize that live CUDA tensor to 16,600,658,432 bytes and all ranks reported
an illegal memory access. GBS128 now reserves 27,111,981,056 bytes on
`TrtllmAttentionMetadata.workspace` immediately after the full-capacity
metadata object is created. This is the 131072-token runtime budget at
200 KiB/token plus 256 MiB headroom. The separate
`cuda_graph_workspace` remains untouched; it is small and decode-specific.
GBS16 and GBS64 reserve nothing because their full runtime shape is already
covered by the 65536-token warmup.

Run `27491999545` proved that workspace reservation: both TRT engine phases
completed without the previous in-place resize or illegal memory access, and
`LLM(...)` returned after about 581 seconds. The first real full-batch prefill
then failed because final KV allocation left only 1.17-2.35 GiB free per GPU.
DeepSeek-V4's fused Q path requested an 8 GiB FP8 buffer for
`[131072, 128 * 512]`, followed by a roughly 2 GiB BF16 RoPE buffer for
`[131072, 128 * 64]`. TRT's capped synthetic calibration does not exercise
that full real-prefill transient. The 12 GiB final-KV reduction covers those
10 GiB of tensors plus allocator headroom without changing runtime
`max_num_tokens`, fixed-batch admission, MTP depth, or the measured decode
path.

Run `27492438399` proved the 12 GiB KV reduction on all eight ranks. Both
engine phases completed, `LLM(...)` returned after about 575 seconds, the
fixed-batch barrier armed, and the real 131072-token prefill ran for about
118 seconds before every rank reported an illegal memory access. The first
useful stack was in `q_b_proj` inside TRT's
`fp8SwapABGemmRunner.forward()`. The error surfaced at the output allocation,
immediately after `_fp8_quantize_1x128_ue8m0()` had launched on all 131072
rows, so the asynchronous root could be the oversized Triton quantizer or the
following DeepGemm path.

The MPI shim now wraps only `fp8SwapABGemmRunner.forward()`. Inputs with at
most 65536 rows use the pinned implementation unchanged. Larger matrices
allocate the final output once, then quantize and execute DeepGemm over
contiguous row slices of at most 65536, writing directly into output views.
The transformed weight and weight-scale layout remains unchanged. Each
oversized chunk is synchronized because this path is prefill-only; any pinned
kernel fault therefore surfaces at the exact chunk instead of a later
allocation or sampler event. This internal GEMM chunking does not split the
executor's required one-iteration full-batch prefill, and measured decode has
only 2, 8, or 16 rows.

The pinned packed-FP8 CUDA quantizer also fails its kernel launch for the
65536-row MTP projection produced by GBS64 prefill. Every rank installs a
guard that selects TRT's existing Triton FP8 quantizer above 32768 rows.
Measured decode has only `local_batch_size` rows, at most 16, so this guard
does not change the measured decode kernel path. Decode also has at most 128
tokens node-wide, so it never reaches the MoE cap.

The old harness used approximately one prompt's prefill token budget even for
large global batches. TRT therefore queued and staggered requests. Dividing
the global request count by mean per-request TPOT assumed requests that were
not actually decoding together, which is why the derived number could be
roughly twice wall throughput.

`LLM.generate()` also enqueues the batch request by request while the executor
is live. The branch-local MPI entry shim therefore holds the idle executor
until exactly one complete global batch is present, then releases all requests
to attention-DP routing together.

The request gate is installed but deliberately disarmed during `LLM`
construction. TRT's KV-capacity calibration sends its own dummy requests
(120 at GBS128 in run `27490378501`); those are engine-initialization work, not
the benchmark batch. The controller removes any stale arm file before launch.
After `LLM(...)` returns, the parent worker atomically creates the shared arm
file and only then calls the real warmup `generate()`. Every rank validates the
same absolute arm path and permanently latches the gate on when it observes
the file. Arming the gate at MPI entry deadlocks GBS128 waiting for
`120/128` calibration requests.

Each real pass must show one prefill iteration:

```text
numContextRequests = local_batch_size
numGenRequests = 0
numScheduledRequests = local_batch_size
numActiveRequests = local_batch_size
numQueuedRequests = 0
numPausedRequests = 0
```

Decode must then begin with:

```text
numContextRequests = 0
numGenRequests = local_batch_size
numScheduledRequests = local_batch_size
numActiveRequests = local_batch_size
numQueuedRequests = 0
numPausedRequests = 0
```

The decode conditions must hold for 256 consecutive TRT iterations.

## Timing

TRT is created with:

```text
enable_iter_perf_stats = true
max_stats_len = 2048
disable_overlap_scheduler = true
```

Those values describe the Huawei profile. PR-max sets
`disable_overlap_scheduler=false` and `print_iter_log=true`; it retains
`enable_iter_perf_stats=true` solely for schedule and MTP-yield validation.

The explicit `2048`-entry history bounds memory while covering prefill plus the
worst-case 1024 decode iterations needed when no MTP drafts are accepted. The
pinned executor also supports `-1` as unbounded, but this benchmark does not
need an unbounded history.

After generation, the worker reads `LLM.get_stats()` and selects the first 256
consecutive full-local-batch decode iterations. It then applies the same
filter as Huawei's `process_infer_time`:

1. Skip the first selected decode round.
2. Calculate Q1 and Q3 from the remaining 255 values.
3. Calculate `upper_fence = Q3 + 1.5 * (Q3 - Q1)`.
4. Drop only latencies above the upper fence.
5. Average retained round latencies.

The pinned TRT stats queue can publish inactive tail iterations after one
`get_stats()` call has marked the queue done. The next prompt submission marks
that queue active again, so those prior-pass rows can appear immediately
before the next pass's prefill. The selector ignores only a leading tail with
the exact fixed local generation/scheduled count and
`active=queued=paused=0`. It records the ignored count and iteration range.
Any active, queued, partial, or mixed row before prefill remains a validation
failure.

The timer scope is the closest TRT runtime equivalent, not the same
instrumentation point. TRT's `iterLatencyMS` covers one complete executor
iteration with overlap disabled. Huawei records and sums the main-model and
MTP model timing regions inside its CANN decode loop. Treat the result as a
matched workload and aggregation contract, with a runtime-specific timer
scope.

The headline fields are:

```text
decode_round_tpot_ms
decode_step_tput_per_gpu =
    global_batch_size / decode_round_tpot_seconds / active_gpu_count
```

The generation output cap is 1025 tokens:

```text
1 first token + 256 rounds * (1 target + 3 drafts)
```

That guarantees no request can finish before 256 MTP3 decode rounds. Generation
may run longer when acceptance is lower, but only the first 256 validated
full-batch rounds enter the headline metric.

## MTP Yield

MTP yield is not folded into Huawei's raw table throughput. The benchmark
reports it separately:

```text
observed_tokens_per_step = 1 + accepted_drafts_per_step
output_tput_per_gpu =
    decode_step_tput_per_gpu * observed_tokens_per_step
equivalent_output_tpot_ms =
    decode_round_tpot_ms / observed_tokens_per_step
```

MTP yield must come from TRT's iteration-level speculative counters for the
same validated 256-round window. A row fails validation if those counters are
unavailable; whole-request telemetry is not substituted into the headline
window.

Huawei publishes 1.44 accepted drafts per MTP3 round, or 2.44 output tokens
per round.

## Reference Rows

| GBS | Huawei chips | TPOT ms | Decode steps/s/chip |
|---:|---:|---:|---:|
| 16 | 16 | 17.64 | 56.70 |
| 64 | 16 | 19.03 | 210.16 |
| 128 | 16 | 20.61 | 388.23 |

The direct comparison is the selected profile's
`decode_step_tput_per_gpu` divided by Huawei
`decode_step_tput_per_chip`. The result also reports an output-token ratio
using each stack's observed or published MTP yield.

The final validated B300 baseline is run `27493336994` at git revision
`9796f5d17c96ab56136b8b9b1e196b6e6db84426`:

| GBS | Round TPOT ms | Decode steps/s/GPU | Tok/step | Output tok/s/GPU |
|---:|---:|---:|---:|---:|
| 16 | 22.069890 | 90.621203 | 3.001953 | 272.040604 |
| 64 | 32.140069 | 248.910481 | 3.507324 | 873.009760 |
| 128 | 36.831497 | 434.410801 | 3.298096 | 1432.728396 |

B300 artifacts:

- [GitHub Actions run](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27493336994)
- [InferenceMAX unofficial run](https://inferencemax-r4i4xgna4-semianalysisai.vercel.app/inference?unofficialrun=27493336994)

The final validated GB300 NVL16 sweep is run `27517035480` at git revision
`c0a845521b51e5fb5eca5f9bb4ac2e3a6c60b43d`:

| GBS | Round TPOT ms | Decode steps/s/GPU | Tok/step | Output tok/s/GPU | Step/Huawei | Output/Huawei |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 25.921265 | 38.578364 | 3.003906 | 115.885788 | 0.680394 | 0.837640 |
| 64 | 32.375174 | 123.551461 | 3.509766 | 433.636669 | 0.587892 | 0.845641 |
| 128 | 34.231453 | 233.703196 | 3.450195 | 806.321670 | 0.601971 | 0.851196 |

All three rows proved ranks `0..15`, one shared 16-GPU NVLink Fabric domain,
one full-batch prefill, zero mixed iterations, and 256 consecutive
full-local-batch decode rounds. The complete flat renderer rows are recorded
in [TRT_BENCH_NOTES.md](../../TRT_BENCH_NOTES.md).

GB300 artifacts:

- [GitHub Actions run](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27517035480)
- [InferenceMAX unofficial run](https://inferencemax-r4i4xgna4-semianalysisai.vercel.app/inference?unofficialrun=27517035480)

### PR #1689 Serving Comparison

[PR #1689 attempt 14](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27164980476/attempts/14)
does not contain an equivalent GBS128 offline result. The same-width TP16
decode row used:

- 24 dedicated prefill GPUs plus 16 decode GPUs
- serving concurrency 666
- decode `max_batch_size=32` per attention-DP rank, or global capacity 512
- overlap scheduling and continuous batching
- random input/output lengths averaging about 7378/922 tokens

It completed 33154.54 output tok/s. InferenceX reports
`output_tput_per_gpu=2072.16` by dividing only by the 16 decode GPUs. Dividing
by all 40 GPUs gives 828.86 output tok/s/GPU, only 2.80% above this benchmark's
GBS128 result of 806.32 output tok/s/GPU.

The attempt's three final 8k/1k rows also use materially different operating
points:

| Decode topology | Prefill + decode GPUs | Concurrency | Output tok/s/decode-GPU | Output tok/s/all-GPU |
|---|---:|---:|---:|---:|
| TP32/EP32, MTP3 | 16 + 32 | 333 | 676.76 | 451.18 |
| TP16/EP16, MTP3 | 24 + 16 | 666 | 2072.16 | 828.86 |
| TP8/EP8, MTP1 | 48 + 8 | 4301 | 9686.74 | 1383.82 |

The TP8 headline uses `max_batch_size=512` per decode rank and
`max_draft_len=1`, not this benchmark's local batch 8 and MTP3. Its client TPOT
implies roughly 430 active requests per decode rank. It is a high-batch serving
saturation point, not evidence that fixed GBS128 should produce 9686.74
tok/s/GPU.

The TP16 serving row also maintained about 401 active decode requests by
Little's law (`33154.54 * 0.012107`), or about 25 per attention-DP rank. This
benchmark holds exactly 8 per rank. At the offline row's measured round latency
and MTP yield, reaching 2072.16 output tok/s/decode-GPU would require
approximately GBS329, not GBS128.

The decoder log independently reconstructs the headline number. Its steady
batch-32-graph rows with 24-30 actual scheduled requests averaged 27.73
requests/rank and 32.49 ms device step time:

```text
27.73 / 0.03249 = 853.3 request-rounds/s/decode-GPU
2072.16 / 853.3 = 2.428 output tokens/request-round
```

The PR gets 2072 mainly by running about 3.47 times the local decode batch,
not by making a local-batch-8 round 2.57 times faster.

There is still a smaller runtime-timing clue, but the PR does not contain an
exact fixed-batch comparison. Iterations selecting its batch-8 CUDA graph,
usually with only 6-8 real scheduled requests because graph padding was
enabled, averaged about 27.67 ms of reported device step time. The 13 rows
with exactly 8 scheduled requests and that graph averaged 28.29 ms. The
offline GBS128 row averages 34.23 ms of complete non-overlapped executor
iteration time. These scopes are not interchangeable: the latter includes TRT
scheduler, sampler, request-state, and stats work, while Huawei times
synchronized model decode calls. A future forward-only fixed-batch diagnostic
would help isolate that difference, but it must remain separate from the
validated headline metric. Increasing GBS or enabling overlap would reproduce
a serving saturation experiment, not correct this offline result.

## Execution Chain

1. Dispatch `.github/workflows/e2e-tests.yml`.
2. Run one requested global batch at a time.
3. Allocate one eight-GPU B300 node, the selected 2/4/8-node GB300 engine, or
   the rack profile's complete 18-node/72-GPU domain. GB300 validates the
   complete dynamic rank map and one shared Fabric `ClusterUUID` and
   `CliqueId` across every allocated GPU.
4. Build the exact 8192-token corpus.
5. Before each GB300 `trtllm-llmapi-launch`, preseed the complete fixed-batch
   environment because external MPI management workers exist before rank 0
   creates its later worker subprocess. Rank 0 recomputes and validates this
   contract before engine construction.
6. Start one fresh TensorRT-LLM engine for Huawei/PR-max, or nine independent
   TP8 engines on disjoint two-node pairs for the rack profile. The rack
   parent owns the allocation and fabric proof; each child owns its MPI world,
   logs, completion record, and hidden child result.
7. Keep runtime capacity at the configured value while applying only the
   documented synthetic-warmup and B300 memory workarounds. On B300 only,
   keep oversized FP8 activation quantization and DeepGemm
   calls bounded while preserving one executor-level prefill iteration.
8. Atomically arm the fixed-batch request gate after engine initialization.
9. In `huawei`, run the short full-batch request warmup. PR-max skips this
   second request pass and uses engine graph warmup plus eight discarded
   measured rounds.
10. For rack runs, wait until all nine engines are ready, then release one
    shared measured-pass barrier.
11. Run one measured generation and validate 256 exact full-batch decode
    iterations per engine. PR-max and rack children use rank-0
    `host_step_time`; Huawei uses non-overlap `iterLatencyMS`.
12. Rack aggregation aligns the nine 256-value timing vectors by round index,
    takes the per-round maximum, then applies the startup skip and upper-IQR
    filter. Only the resulting 72-GPU row is copied to the top-level result
    path.
13. After result and debug files are finalized, atomically publish completion
    records, verify shared-filesystem visibility, and cancel the allocation.
14. Upload per-job result/debug/topology artifacts, then collect
    `offline_aggregate.json`, `offline_summary.md`, and `agg_bmk.json`.

There is no HTTP server, request-rate generator, generic benchmark client,
serial scheduler tuning, or normal InferenceX matrix processing.

## Result Fields

`offline_aggregate.json` is authoritative:

- `decode_round_tpot_ms`: Huawei-style filtered full-batch round latency
- `decode_step_tput_per_gpu`: raw Huawei-comparable step rate
- `observed_tokens_per_step`: measured TRT MTP output yield
- `output_tput_per_gpu`: step rate multiplied by output yield
- `wall_output_tput_per_gpu`: whole `LLM.generate()` diagnostic
- `filter.retained_rounds`: retained values after first-round skip/outliers
- `filter.outlier_rounds`: upper-IQR values removed
- `schedule_validation`: iteration range and exact local-batch proof
- `hardware_to_huawei_decode_step_ratio`: direct raw-step comparison
- `hardware_to_huawei_output_ratio`: output rate using each stack's yield
- `timing_source`: `iter_latency_ms` for Huawei or
  `trt_print_iter_log_host_step_time` for PR-max
- `pr_reference`: copied serving result plus decode-GPU and PR-fleet
  comparison ratios for PR-max rows
- `rack`: child engine provenance and the synchronized round-alignment rule
  for NVL72 rows
- `slowest_replica_trt_print_iter_log_host_step_time`: rack timing source;
  each logical round is the maximum same-index TP8 host-step latency

`results_bmk/agg_bmk.json` remains renderer-compatible. Standard throughput
and TPOT fields use acceptance-adjusted output-token metrics. Custom flat
fields retain raw decode-round TPOT, decode-step throughput, GBS, local batch,
token yield, and measured round count. Its `conc` field is only a compatibility
alias for `global_batch_size`; this offline path does not use serving
concurrency.

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

Huawei-local-batch rack sweep:

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

Maximum-throughput rack sweep:

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

PR-max sweep:

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

Use one of `pr-tp32-mtp3`, `pr-tp16-mtp3`, or `pr-tp8-mtp1` instead of
`pr-max-sweep` for a single profile. PR profiles require
`global_batch_sizes=auto`.

Huawei-style sweep:

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

Monitor:

```bash
RUN_ID=$(gh run list --repo SemiAnalysisAI/InferenceX \
  --workflow e2e-tests.yml --event workflow_dispatch --limit 1 \
  --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" --repo SemiAnalysisAI/InferenceX --exit-status
```

Download:

```bash
gh run download "$RUN_ID" --repo SemiAnalysisAI/InferenceX \
  -n offline-trt-summary -D ./offline-summary
jq '.rows' ./offline-summary/offline_aggregate.json
jq '.' ./offline-summary/agg_bmk.json
```

## Debugging

Each job uploads files keyed by its experiment ID (`gbs64` for old Huawei
dispatches, `pr-tp8-mtp1-gbs4096`, or
`rack-tp8x9-mtp1-gbs36864`):

- `offline_result_ID.json`
- `offline_controller_ID.log`
- `offline_gpu_metrics_ID.csv` on B300
- `offline_gpu_metrics_ID_HOST.csv` on every GB300 node
- `offline_allocation_ID.log` on GB300
- `offline_rank_map_ID.tsv` on GB300
- `offline_topology_ID.log` on GB300
- `offline_completion_ID.json` on GB300
- `offline_world_ID.log` on GB300
- `offline_timing_ID.log` on GB300
- `offline_debug_ID.tar.gz`

The debug archive contains the corpus manifest, worker log/result,
`warmup_iteration_stats.json` when applicable,
`measured_iteration_stats.json`, and the every-rank environment marker.
For rack runs it contains `.offline_rack_ID_JOB/`, including
`replicas/r00` through `replicas/r08`, each child launcher/world/timing log,
completion record, nested child debug archive, and child result. Those child
results remain inside the archive so `collect-results` sees exactly one
72-GPU row.

Debug in this order:

1. Read the result's `failure_kind`, `phase`, and `error`.
2. Read `offline_controller_gbsN.log`.
3. For PR-max, read `offline_timing_ID.log` for the authoritative rank-0
   `print_iter_log` host-step timing. `offline_world_ID.log` is the mirrored
   all-rank console stream. Then extract the debug archive and inspect
   `worker.log` for the worker lifecycle and iteration stats.
4. For rack runs, first inspect the parent world/topology log, then
   `replicas/rNN/host_launcher.log`. Inspect the failed child's world,
   timing, and completion logs, then extract its nested debug archive for
   worker logs. Require all nine barrier-ready files before the shared release
   record.
5. Use `jq` on iteration stats; do not dump the full JSON into the chat.
6. For Huawei, confirm one full context-only iteration precedes decode. For
   PR-max, staged/mixed setup is allowed, but the selected window must contain
   256 consecutive exact full-batch decode iterations.
7. Confirm rank markers are exactly `0..world_size-1` with identical fixed
   environment. Huawei requires perfect-router value `1`; PR-max requires it
   to be absent on every rank.
8. On GB300, confirm 2, 4, 8, or 18 hosts according to profile, four local
   ranks per host, and one shared non-empty Fabric `ClusterUUID` and
   `CliqueId`. Rack children must each contain ranks `0..7` on their assigned
   node pair, and the top-level row must say `9xDEP8`, not TP72. For PR-max,
   also confirm `timing_source` is
   `trt_print_iter_log_host_step_time`.
9. Confirm `offline_completion_gbsN.json` has the same `result_status` as the
   result and a matching return code. Its presence proves rank 0 finished
   copying the result and creating the debug archive before the host canceled
   the external MPI allocation.
10. For initialization stalls, use the controller heartbeat's `rank_progress`
   to identify which ranks reached warmup completion, clock synchronization,
   and executor worker start.
11. If ranks fail during shim installation, inspect `entry_failed` markers.
   If there are only `sitecustomize` rows and no `entry_start`, confirm the
   host logged `preseeded external MPI rank environment` before
   `trtllm-llmapi-launch`.
12. Confirm `fixed_batch_barrier.armed.json` was created only after the worker
   logged `engine initialization complete`.
13. On B300 GBS128, confirm every rank emitted
   `attention_workspace_preallocated` with target and allocated bytes
   `27111981056`, and `cuda_graph_workspace_bytes=0`.
14. On B300 GBS128, confirm every rank emitted `kv_prefill_reserve_applied` with
    reserve `12884901888`, minimum tokens `149504`, an exact
    configured-minus-adjusted delta, and adjusted bytes at least the reported
    minimum runtime KV bytes.
15. On B300 GBS128, confirm every rank emitted `fp8_prefill_gemm_chunked` for
    exactly 131072 rows, two 65536-row chunks, and synchronized chunks.
16. Inspect `memory.used` and `memory.free` in the applicable GPU telemetry
    files around full prefill.

Do not weaken either profile's schedule proof. Huawei must retain one complete
full-batch prefill and non-overlap timing. PR-max must retain its copied
overlap scheduler, learned router, unset `attention_dp_config`, exact decode
capacity, CUDA graph set, MTP depth, and host-step timing. Internal fused-op
chunking is allowed only when it does not change the selected decode window.
Rack results additionally require all nine child proofs, one measured-pass
barrier, same-index slowest-replica timing, and one 72-GPU fabric proof.
