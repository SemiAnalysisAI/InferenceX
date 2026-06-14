# DeepSeek-V4 TRT Huawei-Style Offline Benchmark

This is a branch-only benchmark. It does not use the normal InferenceX serving
pipeline and is not intended to merge into `main`.

## What It Measures

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

## Global And Local Batch

There is one authoritative `global_batch_size`. Every per-rank capacity is
derived from it and the selected profile's attention-DP width:

```text
local_batch_size = global_batch_size / active_gpu_count
max_batch_size = local_batch_size
cuda_graph_batch_size = local_batch_size
max_num_tokens = local_batch_size * 8192
synthetic pure-context warmup request = min(max_num_tokens, 65536)
minimum runtime KV tokens = local_batch_size * 9344
```

### B300 Capacity

| GBS | Local batch/rank | TRT max_num_tokens/rank | Minimum KV tokens/rank | KV reserve |
|---:|---:|---:|---:|---:|
| 16 | 2 | 16384 | 18688 | 0 |
| 64 | 8 | 65536 | 74752 | 0 |
| 128 | 16 | 131072 | 149504 | 12 GiB |

### GB300 Capacity

| GBS | Local batch/rank | TRT max_num_tokens/rank | Minimum KV tokens/rank | KV reserve |
|---:|---:|---:|---:|---:|
| 16 | 1 | 8192 | 9344 | 0 |
| 64 | 4 | 32768 | 37376 | 0 |
| 128 | 8 | 65536 | 74752 | 0 |

All GB300 runtime shapes fit the fixed 65536-token synthetic warmup ceiling.
The B300-only eager attention workspace, 12 GiB KV reduction, FP8 quantizer
guard, and oversized DeepGemm chunker are disabled in the GB300 profile.

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

Artifacts:

- [GitHub Actions run](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27493336994)
- [InferenceMAX unofficial run](https://inferencemax-r4i4xgna4-semianalysisai.vercel.app/inference?unofficialrun=27493336994)

## Execution Chain

1. Dispatch `.github/workflows/e2e-tests.yml`.
2. Run one requested global batch at a time.
3. Allocate either one eight-GPU B300 node or four four-GPU GB300 nodes.
   GB300 captures an exact `0..15` rank map and requires four successful
   NVLink Fabric states on every physical node plus one shared
   `ClusterUUID` and `CliqueId` across all 16 GPUs before engine launch.
4. Build the exact 8192-token corpus.
5. Before GB300 starts `trtllm-llmapi-launch`, preseed the complete
   fixed-batch environment because the 16 external MPI management workers
   exist before rank 0 creates its later worker subprocess. Rank 0 recomputes
   and validates this contract before engine construction. B300 lets TRT
   create its local MPI pool and does not need this host-side step.
6. Start one fresh TensorRT-LLM engine. Runtime capacity stays at the full
   GBS-derived value while synthetic pure-context warmup requests are capped
   at 65536 tokens. On B300 GBS128, reserve the full-runtime non-graph attention
   workspace before the first forward, then subtract the 12 GiB real-prefill
   transient from TRT's calibrated final KV budget while preserving the
   fixed-batch minimum cache cost. TRT's internal capacity probes run while
   the benchmark request gate is disarmed.
7. On B300 only, keep oversized FP8 activation quantization and DeepGemm
   calls bounded while preserving one executor-level prefill iteration.
8. Atomically arm the fixed-batch request gate after engine initialization.
9. Run the short full-batch warmup.
10. Run one measured generation.
11. Validate and filter 256 iteration stats.
12. After result and debug files are finalized, atomically publish a
    completion record. The GB300 host requires both the result and completion
    files before accepting success. If the external MPI world exits first, it
    waits up to 120 seconds for shared-filesystem visibility, then verifies
    both statuses and cancels the allocation.
13. Upload per-job result/debug/topology artifacts.
14. Collect `offline_aggregate.json`, `offline_summary.md`, and `agg_bmk.json`.

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

`results_bmk/agg_bmk.json` remains renderer-compatible. Standard throughput
and TPOT fields use acceptance-adjusted output-token metrics. Custom flat
fields retain raw decode-round TPOT, decode-step throughput, GBS, local batch,
token yield, and measured round count.

## Dispatch

```bash
BENCH_REF="$(git rev-parse HEAD)"
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='trt-bench' \
  -f "inputs[ref]=$BENCH_REF" \
  -f 'inputs[hardware-profile]=gb300' \
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

Each job uploads:

- `offline_result_gbsN.json`
- `offline_controller_gbsN.log`
- `offline_gpu_metrics_gbsN.csv` on B300
- `offline_gpu_metrics_gbsN_HOST.csv` on every GB300 node
- `offline_allocation_gbsN.log` on GB300
- `offline_rank_map_gbsN.tsv` on GB300
- `offline_topology_gbsN.log` on GB300
- `offline_completion_gbsN.json` on GB300
- `offline_debug_gbsN.tar.gz`

The debug archive contains the corpus manifest, worker log/result,
`warmup_iteration_stats.json`, `measured_iteration_stats.json`, and every-rank
perfect-router marker.

Debug in this order:

1. Read the result's `failure_kind`, `phase`, and `error`.
2. Read `offline_controller_gbsN.log`.
3. Extract the debug archive and inspect `worker.log`.
4. Use `jq` on iteration stats; do not dump the full JSON into the chat.
5. Confirm context-only iterations precede a consecutive full-batch decode
   range.
6. Confirm all rank markers are `0..7` for B300 or `0..15` for GB300 with
   identical fixed environment.
7. On GB300, confirm the rank map has four hosts, four local ranks per host,
   and global ranks exactly `0..15`; confirm every fabric summary reports four
   GPUs, four `Completed` states, four `Success` statuses, and the same
   non-empty `ClusterUUID` and `CliqueId`.
8. Confirm `offline_completion_gbsN.json` has the same `result_status` as the
   result and a matching return code. Its presence proves rank 0 finished
   copying the result and creating the debug archive before the host canceled
   the external MPI allocation.
9. For initialization stalls, use the controller heartbeat's `rank_progress`
   to identify which ranks reached warmup completion, clock synchronization,
   and executor worker start.
10. If ranks fail during shim installation, inspect `entry_failed` markers.
   If there are only `sitecustomize` rows and no `entry_start`, confirm the
   host logged `preseeded external MPI rank environment` before
   `trtllm-llmapi-launch`.
11. Confirm `fixed_batch_barrier.armed.json` was created only after the worker
   logged `engine initialization complete`.
12. On B300 GBS128, confirm every rank emitted
   `attention_workspace_preallocated` with target and allocated bytes
   `27111981056`, and `cuda_graph_workspace_bytes=0`.
13. On B300 GBS128, confirm every rank emitted `kv_prefill_reserve_applied` with
    reserve `12884901888`, minimum tokens `149504`, an exact
    configured-minus-adjusted delta, and adjusted bytes at least the reported
    minimum runtime KV bytes.
14. On B300 GBS128, confirm every rank emitted `fp8_prefill_gemm_chunked` for
    exactly 131072 rows, two 65536-row chunks, and synchronized chunks.
15. Inspect `memory.used` and `memory.free` in the applicable GPU telemetry
    files around full prefill.

Do not fix failures by reducing the global batch, splitting prefill across
executor iterations, enabling overlap scheduling, weakening schedule
validation, reducing MTP depth, or reverting to per-request TPOT. Internal
fused-op chunking is allowed only when the schedule proof still shows one
complete full-batch prefill iteration.
