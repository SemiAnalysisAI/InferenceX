# TensorRT-LLM Offline Benchmark Ledger

This file is the evidence ledger for `trt-bench`. It records what was run,
which result supersedes which experiment, and the failures that produced the
current implementation.

Use [`utils/bench_offline/README.md`](utils/bench_offline/README.md) for the
runnable specification and dispatch commands. Use
[`utils/bench_offline/AGENTS.md`](utils/bench_offline/AGENTS.md) for the
invariants that must survive debugging.

## Current State

As of June 15, 2026:

- branch: `trt-bench`
- dispatched source: `0ec5b7d4c3992d1f98ce950b0cc20cd11f43561b`
- TensorRT-LLM source:
  `34a563ac6d8cc0ca7068c7f619e869fb8a625333`
- image:
  `nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1`
- model: `/scratch/models/DeepSeek-V4-Pro`
- recipe source: InferenceX PR #1689, Actions run `27164980476`,
  attempt 14
- fastest copied recipe:
  `ctx12dep4_gen1dep8_batch512_eplb384_mtp1.yaml`

The branch maximum remains rack GBS36864 at
`10262.766175` output tok/s/GPU.

The final Huawei-local-batch rack sweep is Actions run
[27555308252](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27555308252).
All three rows completed successfully and the renderer-compatible aggregate
contains exactly three flat rows.

## Metric Decision

The primary Huawei comparison is raw decode-step throughput:

```text
GB300/Huawei raw ratio =
    GB300 decode steps/s/GPU / Huawei decode steps/s/chip
```

To state that comparison in output tokens while holding MTP yield constant:

```text
Huawei same-yield output =
    Huawei decode steps/s/chip * GB300 measured tokens/step

GB300/Huawei same-yield output ratio =
    GB300 output tok/s/GPU / Huawei same-yield output
```

The two ratios are exactly equal. This is the comparison requested for the
rack GBS72/288/576 sweep. It does not credit GB300 or Huawei for a different
MTP depth or acceptance rate.

The separate own-yield ratio uses Huawei's published 2.44 tokens/step and
GB300's measured MTP1 yield. Keep it available, but do not use it as the
headline hardware comparison.

## Commit History

Before this consolidation, the branch contained 83 commits after merge base
`8f1f2137290bdf8e81f47240743614760c4f2670`.

| Phase | Commits | Range | Outcome |
|---|---:|---|---|
| B300 exploration | 19 | `3c74b504..6dfa72a8` | Built the offline harness, progress logging, tuning experiments, and early concurrency results |
| B300 fixed GBS | 30 | `5dcb554b..422bd56d` | Replaced concurrency inference with exact GBS, added schedule proof, renderer output, and pinned-image memory workarounds |
| GB300 NVL16 and PR profiles | 20 | `3ca9a9fe..85c49e1e` | Added four-node fabric validation, external-MPI environment handling, PR-aligned timing, and fixed-GBS GB300 results |
| GB300 NVL72 rack | 14 | `2976c179..0ec5b7d4` | Added nine synchronized TP8 engines, serialized/retried model loading, future-start barrier, and rack maximum results |

This consolidation adds explicit Huawei-at-measured-yield fields and removes
duplicated documentation. It does not change the measured TRT execution path.

## Result Ledger

### Superseded Concurrency Sweep

Actions run
[27483465692](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27483465692)
used configured serving-style concurrency and per-request averages:

| Configured concurrency | Raw derived steps/s/GPU | Derived output tok/s/GPU | Wall output tok/s/GPU |
|---:|---:|---:|---:|
| 16 | 110.00 | 333.92 | 189.74 |
| 32 | 160.35 | 488.44 | 270.56 |
| 64 | 291.03 | 932.20 | 482.11 |
| 128 | 390.34 | 1274.21 | 588.16 |
| 512 | 430.93 | 1351.94 | 625.39 |
| 1024 | 498.17 | 1582.72 | 731.58 |

These rows are historical only. TRT could queue and stagger requests because
the old prompt-token capacity was too small. Multiplying configured
concurrency by a mean per-request rate treated queued or prefill requests as
simultaneous decoders. At configured concurrency 1024, the result implied
only about 473 active requests. The fixed-GBS path supersedes this method.

### B300 Fixed GBS

Actions run
[27493336994](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27493336994),
source `9796f5d17c96ab56136b8b9b1e196b6e6db84426`.

Renderer:

```text
https://inferencemax-r4i4xgna4-semianalysisai.vercel.app/inference?unofficialrun=27493336994
```

| GBS | Local/GPU | Round ms | Steps/s/GPU | Tok/step | Output tok/s/GPU | Wall tok/s/GPU |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 2 | 22.069890 | 90.621203 | 3.001953 | 272.040604 | 198.178843 |
| 64 | 8 | 32.140069 | 248.910481 | 3.507324 | 873.009760 | 378.422175 |
| 128 | 16 | 36.831497 | 434.410801 | 3.298096 | 1432.728396 | 143.915720 |

All rows proved one full-batch prefill and 256 consecutive exact decode
rounds. GBS128 additionally proved the 12 GiB KV reserve, exact attention
workspace reservation, and two 65536-row prefill GEMM chunks.

### GB300 NVL16 Fixed GBS

Actions run
[27517035480](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27517035480),
source `c0a845521b51e5fb5eca5f9bb4ac2e3a6c60b43d`.

Renderer:

```text
https://inferencemax-r4i4xgna4-semianalysisai.vercel.app/inference?unofficialrun=27517035480
```

| GBS | Local/GPU | Round ms | Steps/s/GPU | Tok/step | Output tok/s/GPU | Wall tok/s/GPU | Raw/Huawei |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 25.921265 | 38.578364 | 3.003906 | 115.885788 | 90.025851 | 0.680394 |
| 64 | 4 | 32.375174 | 123.551461 | 3.509766 | 433.636669 | 212.253702 | 0.587892 |
| 128 | 8 | 34.231453 | 233.703196 | 3.450195 | 806.321670 | 367.156113 | 0.601971 |

Every row proved ranks `0..15`, four nodes, one shared Fabric UUID/clique,
one exact full-batch prefill, and 256 exact decode rounds.

PR #1689's TP16 serving row reports `2072.1585` output tok/s/decode-GPU, but
it used about 401 active decode requests, or roughly 25 per attention-DP
rank, plus 24 dedicated prefill GPUs. The fixed GBS128 row has exactly eight
requests per rank. Charging all 40 PR GPUs gives `828.8634` output
tok/s/GPU, close to this offline row's `806.3217`, but it is still a
different scheduler and population.

### Rack Synchronization Canary

Actions run
[27543065270](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27543065270),
source `ccddb60b`.

GBS72 completed with:

```text
round ms: 18.490630
steps/s/GPU: 54.081444
tokens/step: 1.769965
output tok/s/GPU: 95.722279
measured start skew: 0.000370 seconds
```

This run proved the common future-start barrier after NFS release visibility
varied by as much as 48.533 seconds. It is superseded by the current
Huawei-local-batch sweep at source `0ec5b7d4`.

### Rack Maximum

Actions run
[27545752641](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27545752641),
source `775a1451074966b871f1cbd57229894d393f4af0`.

Renderer:

```text
https://inferencemax-r4i4xgna4-semianalysisai.vercel.app/inference?unofficialrun=27545752641
```

| Rack GBS | Engine GBS | Local/GPU | Round ms | Steps/s/GPU | Tok/step | Output tok/s/GPU | Wall tok/s/GPU | Versus PR TP8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30960 | 3440 | 430 | 82.041011 | 5241.281069 | 1.806613 | 9468.968467 | 1382.349906 | -2.248095% |
| 36864 | 4096 | 512 | 90.548343 | 5654.438070 | 1.814993 | 10262.766175 | 1331.507904 | +5.946593% |

Both rows used one pass, 256 exact full-batch rounds, eight startup rounds
skipped, and 208 retained rounds after 40 upper-IQR outliers. GBS30960
retried replica `r00`; GBS36864 retried `r08`. Both allocations proved Fabric
UUID `8fe56262-d2bb-4602-b338-8898d34c4731` and clique `32766` on all
72 GPUs.

GBS36864 is the best validated result on the branch. The nine individual
TP8 replicas produced `10356.040023` to `10538.418796` output tok/s/GPU.
The published rack value is lower because each logical round takes the
slowest same-index replica.

Exact flat renderer rows:

```json
[
  {
    "benchmark_profile": "rack-tp8x9-mtp1",
    "conc": 30960,
    "decode_dp_attention": true,
    "decode_ep": 8,
    "decode_num_workers": 9,
    "decode_round_tpot_ms": 82.04101140682513,
    "decode_step_tput_per_gpu": 5241.281069387542,
    "decode_tp": 8,
    "disagg": false,
    "engine_max_batch_size": 512,
    "framework": "trt",
    "global_batch_size": 30960,
    "hw": "gb300-nv",
    "image": "nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1",
    "infmax_model_prefix": "dsv4",
    "is_multinode": true,
    "isl": 8192,
    "local_batch_size": 430,
    "mean_intvty": 22.02085689965943,
    "mean_tpot": 0.04541149350166595,
    "measured_decode_rounds": 256,
    "median_tpot": 0.045354886029864684,
    "model": "deepseek-ai/DeepSeek-V4-Pro",
    "num_decode_gpu": 72,
    "num_prefill_gpu": 72,
    "observed_tokens_per_step": 1.8066133720930233,
    "osl": 1024,
    "output_tput_per_gpu": 9468.968466853554,
    "p90_tpot": 0.04606294411368792,
    "p99_tpot": 0.0470255470529077,
    "precision": "fp4",
    "prefill_dp_attention": true,
    "prefill_ep": 8,
    "prefill_num_workers": 9,
    "prefill_tp": 8,
    "replica_count": 9,
    "spec_decoding": "mtp",
    "timing_source": "slowest_replica_trt_print_iter_log_host_step_time",
    "tput_per_gpu": 9468.968466853554
  },
  {
    "benchmark_profile": "rack-tp8x9-mtp1",
    "conc": 36864,
    "decode_dp_attention": true,
    "decode_ep": 8,
    "decode_num_workers": 9,
    "decode_round_tpot_ms": 90.54834338334891,
    "decode_step_tput_per_gpu": 5654.438069975255,
    "decode_tp": 8,
    "disagg": false,
    "engine_max_batch_size": 512,
    "framework": "trt",
    "global_batch_size": 36864,
    "hw": "gb300-nv",
    "image": "nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1",
    "infmax_model_prefix": "dsv4",
    "is_multinode": true,
    "isl": 8192,
    "local_batch_size": 512,
    "mean_intvty": 20.044465185924917,
    "mean_tpot": 0.04988908363103611,
    "measured_decode_rounds": 256,
    "median_tpot": 0.04974934500687279,
    "model": "deepseek-ai/DeepSeek-V4-Pro",
    "num_decode_gpu": 72,
    "num_prefill_gpu": 72,
    "observed_tokens_per_step": 1.8149931165907118,
    "osl": 1024,
    "output_tput_per_gpu": 10262.766175193558,
    "p90_tpot": 0.050719981593169344,
    "p99_tpot": 0.05213402061976046,
    "precision": "fp4",
    "prefill_dp_attention": true,
    "prefill_ep": 8,
    "prefill_num_workers": 9,
    "prefill_tp": 8,
    "replica_count": 9,
    "spec_decoding": "mtp",
    "timing_source": "slowest_replica_trt_print_iter_log_host_step_time",
    "tput_per_gpu": 10262.766175193558
  }
]
```

### Rack Huawei-Local-Batch Sweep

Actions run
[27555308252](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27555308252),
source `0ec5b7d4c3992d1f98ce950b0cc20cd11f43561b`.

The rack scales Huawei's local batches rather than copying Huawei's global
device count:

| Rack GBS | Engine GBS | Local/GPU | Huawei row |
|---:|---:|---:|---:|
| 72 | 8 | 1 | Huawei GBS16 |
| 288 | 32 | 4 | Huawei GBS64 |
| 576 | 64 | 8 | Huawei GBS128 |

Current results:

| Rack GBS | Local/GPU | Round ms | Steps/s/GPU | Tok/step | Output tok/s/GPU | Wall tok/s/GPU | Huawei raw | Huawei at GB300 yield | Same-yield ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 72 | 1 | 18.690088 | 53.504297 | 1.780816 | 95.281306 | 62.448993 | 56.70 | 100.972266 | 0.943638 |
| 288 | 4 | 23.188691 | 172.497882 | 1.795247 | 309.676373 | 253.218130 | 210.16 | 377.289193 | 0.820793 |
| 576 | 8 | 26.775983 | 298.775209 | 1.797852 | 537.153476 | 325.438665 | 388.23 | 697.979912 | 0.769583 |

GBS72:

- Slurm job `8817`
- no startup retry
- 210 retained rounds and 38 outliers
- measured start skew `0.000371` seconds
- all nine child windows selected local batch exactly 1
- every child had at least 536 full-batch rounds available

GBS288:

- Slurm job `8819`
- `r03` and `r05` timed out on model-load attempt 1 and succeeded on attempt 2
- 218 retained rounds and 30 outliers
- measured start skew `0.000324` seconds
- all nine child windows selected local batch exactly 4
- every child had at least 529 full-batch rounds available

GBS576:

- Slurm job `8823`
- `r05` and `r07` timed out on model-load attempt 1 and succeeded on attempt 2
- 202 retained rounds and 46 outliers
- measured start skew `0.000135` seconds
- all nine child windows selected iterations 4 through 259 at local batch
  exactly 8
- every child had at least 530 full-batch rounds available
- setup contained one context-only iteration and two staged mixed/partial
  iterations before the exact selected window

All three rows proved:

- ranks `0..71` on 18 hosts with four ranks per host
- one Fabric UUID `8fe56262-d2bb-4602-b338-8898d34c4731`
- one clique `32766`
- all 72 topology records `Completed` and `Success`
- nine successful TP8 child results
- one synchronized measured pass
- 256 exact full-batch rounds per child
- no mixed, partial, queued, or paused row in any selected window
- one successful top-level completion record
- a top-level renderer array with fixed GBS 72, 288, and 576

Renderer:

```text
https://inferencemax-r4i4xgna4-semianalysisai.vercel.app/inference?unofficialrun=27555308252
```

An unauthenticated `curl` returns HTTP 401. The downloaded `results_bmk`
artifact was validated directly: `agg_bmk.json` is a top-level array of
exactly three flat rows with GBS 72, 288, and 576.

Exact flat renderer rows, sorted by rack GBS:

```json
[
  {
    "benchmark_profile": "rack-tp8x9-mtp1",
    "conc": 72,
    "decode_dp_attention": true,
    "decode_ep": 8,
    "decode_num_workers": 9,
    "decode_round_tpot_ms": 18.69008768172491,
    "decode_step_tput_per_gpu": 53.50429687806098,
    "decode_tp": 8,
    "disagg": false,
    "engine_max_batch_size": 512,
    "framework": "trt",
    "global_batch_size": 72,
    "hw": "gb300-nv",
    "image": "nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1",
    "infmax_model_prefix": "dsv4",
    "is_multinode": true,
    "isl": 8192,
    "local_batch_size": 1,
    "mean_intvty": 95.28130646297056,
    "mean_tpot": 0.01049523812300614,
    "measured_decode_rounds": 256,
    "median_tpot": 0.01018109694649875,
    "model": "deepseek-ai/DeepSeek-V4-Pro",
    "num_decode_gpu": 72,
    "num_prefill_gpu": 72,
    "observed_tokens_per_step": 1.7808159722222223,
    "osl": 1024,
    "output_tput_per_gpu": 95.28130646297058,
    "p90_tpot": 0.011802564375057122,
    "p99_tpot": 0.013774931589452456,
    "precision": "fp4",
    "prefill_dp_attention": true,
    "prefill_ep": 8,
    "prefill_num_workers": 9,
    "prefill_tp": 8,
    "replica_count": 9,
    "spec_decoding": "mtp",
    "timing_source": "slowest_replica_trt_print_iter_log_host_step_time",
    "tput_per_gpu": 95.28130646297058
  },
  {
    "benchmark_profile": "rack-tp8x9-mtp1",
    "conc": 288,
    "decode_dp_attention": true,
    "decode_ep": 8,
    "decode_num_workers": 9,
    "decode_round_tpot_ms": 23.188690526769797,
    "decode_step_tput_per_gpu": 172.49788190420958,
    "decode_tp": 8,
    "disagg": false,
    "engine_max_batch_size": 512,
    "framework": "trt",
    "global_batch_size": 288,
    "hw": "gb300-nv",
    "image": "nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1",
    "infmax_model_prefix": "dsv4",
    "is_multinode": true,
    "isl": 8192,
    "local_batch_size": 4,
    "mean_intvty": 77.41909331882454,
    "mean_tpot": 0.012916710298864336,
    "measured_decode_rounds": 256,
    "median_tpot": 0.01294919218891659,
    "model": "deepseek-ai/DeepSeek-V4-Pro",
    "num_decode_gpu": 72,
    "num_prefill_gpu": 72,
    "observed_tokens_per_step": 1.7952473958333335,
    "osl": 1024,
    "output_tput_per_gpu": 309.67637327529815,
    "p90_tpot": 0.01349739966993427,
    "p99_tpot": 0.013837418723084768,
    "precision": "fp4",
    "prefill_dp_attention": true,
    "prefill_ep": 8,
    "prefill_num_workers": 9,
    "prefill_tp": 8,
    "replica_count": 9,
    "spec_decoding": "mtp",
    "timing_source": "slowest_replica_trt_print_iter_log_host_step_time",
    "tput_per_gpu": 309.67637327529815
  },
  {
    "benchmark_profile": "rack-tp8x9-mtp1",
    "conc": 576,
    "decode_dp_attention": true,
    "decode_ep": 8,
    "decode_num_workers": 9,
    "decode_round_tpot_ms": 26.775983300539526,
    "decode_step_tput_per_gpu": 298.7752087460707,
    "decode_tp": 8,
    "disagg": false,
    "engine_max_batch_size": 512,
    "framework": "trt",
    "global_batch_size": 576,
    "hw": "gb300-nv",
    "image": "nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:1.3.0-deepseek-v4-dev.1",
    "infmax_model_prefix": "dsv4",
    "is_multinode": true,
    "isl": 8192,
    "local_batch_size": 8,
    "mean_intvty": 67.14418448504836,
    "mean_tpot": 0.014893322596280541,
    "measured_decode_rounds": 256,
    "median_tpot": 0.014850871320783542,
    "model": "deepseek-ai/DeepSeek-V4-Pro",
    "num_decode_gpu": 72,
    "num_prefill_gpu": 72,
    "observed_tokens_per_step": 1.7978515625,
    "osl": 1024,
    "output_tput_per_gpu": 537.1534758803868,
    "p90_tpot": 0.01557451421187534,
    "p99_tpot": 0.01645714556244908,
    "precision": "fp4",
    "prefill_dp_attention": true,
    "prefill_ep": 8,
    "prefill_num_workers": 9,
    "prefill_tp": 8,
    "replica_count": 9,
    "spec_decoding": "mtp",
    "timing_source": "slowest_replica_trt_print_iter_log_host_step_time",
    "tput_per_gpu": 537.1534758803868
  }
]
```

## Failure-To-Fix Ledger

Only failures that produced a durable implementation decision are retained.

| Run | Symptom | Durable decision |
|---:|---|---|
| 27486168511 | Explicit KV token cap admitted only half the B300 batch | Use TRT's memory-derived cache; apply the narrow GBS128 reserve after calibration |
| 27490378501 | Gate deadlocked on 120/128 TRT calibration requests | Install early but arm only after `LLM(...)` returns |
| 27490833024 | Lowering runtime `max_num_tokens` created undersized attention metadata | Cap only the synthetic warmup request |
| 27491160719 | Eager attention workspace resize caused illegal memory access | Preallocate the exact GBS128 workspace |
| 27491999545 | Real B300 GBS128 prefill lacked transient memory | Subtract 12 GiB from final KV budget |
| 27492438399 | 131072-row FP8/DeepGemm prefill failed | Route large FP8 quantization and chunk only the oversized fused GEMM |
| 27511242827 | Concurrent perfect-router writes corrupted JSONL | Lock all shared appends and strict reads |
| 27511740130 | Root checkout cleanup hit unrelated `.nfs*` files | Use an isolated checkout/workspace |
| 27513364142 | Successful result appeared after the host's five-second grace | Preserve bounded shared-filesystem visibility polling |
| 27514818464 | Prior-pass inactive TRT stats appeared before measured prefill | Ignore only a precisely validated leading inactive tail |
| 27515257151 | GB300 65536-row packed-FP8 launch returned invalid argument | Use the existing TRT Triton quantizer above 32768 rows |
| 27516149323 | GB300 GBS128 synthetic 65536-token tuning exhausted temporary memory | Keep runtime capacity 65536 but tune pure-context warmup at 32768 |
| 27531206092 | Prefill-only allocator/loading settings left most rack engines idle | Copy the resolved decode environment, not prefill worker settings |
| 27533885582 | One of nine simultaneous engines stalled reading model shard 43 | Stop launching all model readers together |
| 27535038325 | One of three concurrent engines repeated the shard-43 stall | Serialize model-load admission |
| 27537092211 | One serialized engine still stalled transiently on shard 31 | Retry only the failed child, up to three attempts |
| 27539161854 | NFS release visibility caused 32.357 seconds of start skew | Publish a common start 90 seconds in the future |
| 27539161854 | Rack parent profile leaked into child metric import | Aggregate under the explicit TP8 engine profile |
| 27543065270 | Complete rack canary succeeded | Treat future-start barrier and serialized/retried loading as the baseline |

## Why Rack Jobs Take About An Hour

The measured decode pass itself takes seconds. Most elapsed time is startup:

1. request and acquire an 18-node Slurm allocation
2. validate 72 ranks and Fabric state
3. read the model through nine separate TP8 engines
4. admit model loading one engine at a time
5. capture CUDA graphs through batch 512
6. retry any child that misses the 600-second load gate
7. wait for all nine children
8. wait for the common future start
9. run and validate the measured pass
10. archive all child logs and wait for shared-filesystem visibility

Observed successful matrix-job durations:

- GBS72: 48m51s
- GBS288: 1h08m47s, including two model-load retries
- GBS576: 1h11m45s, including two model-load retries
- rack maximum points: roughly the same startup-dominated range

The 256-round measurement is intentionally one pass. Earlier three-pass
experiments were removed because they tripled measured work without solving
startup cost or improving the fixed-window definition.

## Result Acceptance Checklist

A final rack artifact is accepted only when all of these are true:

- source SHA, image, TRT revision, model identity, and recipe provenance match
- rack GBS, engine GBS, local batch, and max batch are exact
- rank map has 72 unique ranks, 18 hosts, and four ranks per host
- topology has 72 successful records and one Fabric UUID/clique
- all nine child node pairs are disjoint
- every child result is successful
- every child selects 256 consecutive exact full-batch rounds
- selected local batch min and max equal the requested local batch
- no selected mixed, partial, queued, or paused iteration exists
- all nine barrier-ready records and one release record exist
- measured start skew is at most 10 seconds
- rack timing uses the slowest same-index child host-step
- MTP counters cover the same selected window
- top-level completion status and controller return code agree
- `offline_aggregate.json` includes every expected row
- `agg_bmk.json` is a top-level array of flat rows
- the renderer schema validates; visual checking requires an authenticated
  InferenceMAX browser session

## Commands

The canonical dispatch, monitor, download, parsing, and local verification
commands are intentionally kept only in
[`utils/bench_offline/README.md`](utils/bench_offline/README.md).
