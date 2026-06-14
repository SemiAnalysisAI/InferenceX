# TRT Bench Working Notes

These are operational notes for running and debugging `trt-bench`.

## Current Contract

- Branch-only; never merge to `main`.
- DeepSeek-V4 Pro FP4 with `b300` and `gb300` hardware profiles.
- Exact GBS `16`, `64`, `128`.
- Exact 8192-token prompts.
- MTP3, temperature 1, engine-global seed 42, EOS ignored.
- Perfect router, LM-head TP, and heuristic sparse top-k.
- Overlap scheduler disabled.
- Two warmup decode rounds.
- 256 measured full-batch decode rounds.
- First measured round skipped and upper-IQR outliers removed.
- Headline throughput is
  `GBS / decode-round TPOT / active_gpu_count`.
- MTP yield is separate.

The validated baseline is B300 DEP8 at TRT `c185066`. The current target is
GB300 NVL16 DEP16 using the DeepSeek-V4 development image from InferenceX
PR #1689.

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
- PDL enabled. ConfigurableMoE disabled to match the PR runtime path.
- CUDA graph contains only the exact fixed local batch for each row.
- Overlap remains disabled because this benchmark times synchronized decode
  rounds rather than the serving recipe's overlapped scheduler.

Capacity:

| GBS | Local/rank | `max_batch_size` | CUDA graph | `max_num_tokens` | Minimum KV tokens |
|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 1 | 1 | 8192 | 9344 |
| 64 | 4 | 4 | 4 | 32768 | 37376 |
| 128 | 8 | 8 | 8 | 65536 | 74752 |

Every GB300 shape fits the 65536-token warmup ceiling. The B300-only eager
attention workspace, 12 GiB KV reserve, packed-FP8 tactic guard, and
131072-row DeepGemm chunker are disabled.

Launch chain:

1. Import the ARM64 image on a GB300 compute node into the shared squash
   cache. Do not import it on the x86 runner/login node.
2. Allocate four nodes with `--gpus-per-node=4` and
   `--ntasks-per-node=4`.
3. Run a 16-task probe and require global ranks `0..15`, local ranks `0..3`
   on each of exactly four hosts.
4. Run a one-task-per-node full `nvidia-smi -q` Fabric probe. Require four
   GPUs plus four `State: Completed` and four `Status: Success` records per
   node, then require one shared non-empty `ClusterUUID` across all 16 GPUs.
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

Artifacts unique to GB300:

- `offline_rank_map_gbsN.tsv`
- `offline_topology_gbsN.log`
- `offline_gpu_metrics_gbsN_HOST.csv` for all four hosts

The topology log is proof that the allocation entered one 16-GPU NVLink
Fabric domain before the measured engine was started. The result itself
records the Slurm node list, Fabric `ClusterUUID`, and artifact names.

Status as of implementation: no GB300 row is considered validated until an
Actions artifact proves the exact 16-rank set, fabric checks, fixed-batch
schedule, 256-round window, and final flat renderer row. Record the canary and
full-sweep run IDs in this file after they complete.

### GB300 Canary History

Run `27502789238`, source
`3ca9a9febe452918641fc842e5e425553cf035a2`, reached Slurm allocation
`8707` on `im-gb300-r01-c011` through `c014`. Its rank artifact proves global
ranks `0..15`, four ranks per host, and local ranks `0..3`. The run stopped
before TRT because this driver rejects `nvidia-smi -q -d FABRIC` with exit
code `2`. Fabric data is present in full `nvidia-smi -q` output. The follow-up
uses the tested `gb300_fabric.py` parser and additionally requires one shared
non-empty `ClusterUUID` across all 16 GPUs.

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

GB300 full sweep:

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

GB300 GBS16 canary:

```bash
BENCH_REF="$(git rev-parse HEAD)"
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='trt-bench' \
  -f "inputs[ref]=$BENCH_REF" \
  -f 'inputs[hardware-profile]=gb300' \
  -f 'inputs[test-name]=DSV4 GB300 TRT fixed GBS16 canary' \
  -f 'inputs[global_batch_sizes]=16' \
  -f 'inputs[salloc-time]=60' \
  -f 'inputs[worker-timeout]=1800' \
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

- `decode_round_tpot_ms`: filtered full-batch TRT iteration latency
- `decode_step_tput_per_gpu`: direct Huawei-comparable rate
- `observed_tokens_per_step`: MTP output multiplier
- `output_tput_per_gpu`: step rate times output multiplier
- `wall_output_tput_per_gpu`: whole generation diagnostic
- `filter`: first-round skip and retained/outlier counts
- `schedule_validation`: exact selected iteration and local-batch proof

Flat renderer row:

- `tput_per_gpu` and `output_tput_per_gpu`: acceptance-adjusted output rate
- `mean_tpot`: equivalent output-token TPOT
- `decode_round_tpot_ms`: custom raw-round field
- `decode_step_tput_per_gpu`: custom raw-step field
- `global_batch_size`, `local_batch_size`, `measured_decode_rounds`: custom
  workload proof

Do not put the custom aggregate wrapper in `results_bmk`; the unofficial-run
API expects every JSON object there to be a flat benchmark row.

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
2. Download `offline-trt-job-gbsN`.
3. Read `offline_controller_gbsN.log`.
4. Extract `offline_debug_gbsN.tar.gz`.
5. Inspect `worker_result.json` and `worker.log`.
6. Query iteration stats:

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

7. Confirm rank markers are exactly `0..7` on B300 or `0..15` on GB300.
8. On GB300, inspect `offline_rank_map_gbsN.tsv` and
   `offline_topology_gbsN.log` before debugging TRT. A rank-placement or
   fabric failure is infrastructure, not a benchmark result.
9. On B300 GBS128, require `fp8_prefill_gemm_chunked` on ranks `0..7` for
   131072 rows, two 65536-row chunks, and synchronized chunks.
10. Check GPU telemetry, including `memory.used` and `memory.free`, for OOM,
   idle prefill gaps, or a hung rank.

Do not weaken the fixed-batch gate. Infrastructure errors may be retried on a
fresh node; capacity or schedule failures require an implementation fix.
