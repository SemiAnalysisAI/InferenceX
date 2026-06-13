# TRT Bench Working Notes

These notes are for the next agent running and debugging `trt-bench`.

## Scope

- Branch-only, never merge to `main`.
- B300, one node, eight GPUs allocated. DEP8 uses all eight; TP4/DEP4
  production-recipe experiments use four active GPUs.
- TensorRT-LLM only.
- Offline `LLM.generate()`, no server or generic InferenceX processing.
- Optimization workflows may use up to ten parallel single-node B300 jobs.
- Huawei-comparable rows remain DEP8/MTP3 at c8, c32, and c64.

## Pinned TRT Facts

The image tag identifies TensorRT-LLM source commit `c185066`.

- Tokenizer:
  `from tensorrt_llm.tokenizer.deepseek_v4 import DeepseekV4Tokenizer`
- Prompt fast path: `{"prompt_token_ids": ids}`
- Per-request telemetry:
  `completion.request_perf_metrics`
- Timing fields:
  `arrival_time`, `first_scheduled_time`, `first_token_time`,
  `last_token_time`
- Iteration fields: `first_iter`, `last_iter`
- Spec fields:
  `total_accepted_draft_tokens`, `total_draft_tokens`, `acceptance_rate`
- In run `27461421427`, the pinned PyTorch MTP path left all three raw spec
  values at zero while `last_iter - first_iter` showed 3.143 tokens/step.
  Treat zero proposed drafts as unavailable telemetry, not 0% acceptance.
- Effective MTP3 acceptance is derived as `(tokens_per_step - 1) / 3`.
- MTP config:
  `{"decoding_type": "MTP", "max_draft_len": 3}`
- Perfect router is read from unprefixed `ENABLE_PERFECT_ROUTER` in
  `tensorrt_llm._torch.modules.fused_moe.interface.MoE`.
- `MpiPoolSession` explicitly forwards environment names beginning with
  `TRTLLM` or `TLLM`.
- `trt_mpi_entry.worker_main` is installed into the pinned IPC proxy before
  `LLM` construction. It recreates `ENABLE_PERFECT_ROUTER` and marks every
  rank before importing TRT's real worker entry.
- The pinned PyTorch sampler does not apply request-level
  `SamplingParams.seed`. It creates one engine-global CUDA generator with
  seed `42` and advances it across requests and passes. CANN also globally
  seeds to `42`, but uses a different temperature-1 sampling implementation.
- Current optimization runs use one measured pass to reduce GPU time. Earlier
  runs used three passes after one-pass tuning showed output-dependent MTP
  variation; confirm any winner with a separate repeat before treating a
  marginal difference as stable.
- Fresh engine startup is dominated by model initialization plus CuTe DSL
  compilation. The branch mounts the image-keyed path
  `/data/trtllm-cache/dsv4-c185066-sm100a/cute-dsl`, and the direct
  `CUTE_DSL_CACHE_DIR` variable plus its `TRTLLM_*` forwarding alias must
  reach every active MPI rank marker. Cache-prime run `27475868179` created
  zero files there, so propagation is proven but caching is not.
- TRT's exact DeepSeek-V4 README defines ADP `max_batch_size` as per local
  rank. `LLM.generate()` enqueues global requests individually, and
  PyExecutor allows `tp_size * max_num_active_requests` total ADP requests.
  The CUDA graph runner pads each local rank to a configured graph batch.
  Therefore the legacy global graph sizes 8/32/64 can pad true local batches
  1/4/8 by 8x.

Do not assume a newer TRT release has the same field names. If the image
changes, inspect the exact source first.

## B300 Optimization Boundaries

The normal B300 recipe is an optimization source, but comparison claims have
stricter boundaries:

- It must not reduce MTP3 to MTP2 at high concurrency.
- LM-head TP in attention DP is now an explicit candidate because both the
  Huawei guide and the checked-in B300 production recipe enable it.
- It must not change to MegaMoe for this 8K matching workload unless that is
  a separately labeled non-comparable experiment.
- It must not substitute continuous batching or HTTP request timing.
- TP4 and alternate MTP-depth tests can be useful second-stage optimization
  work, but they do not have the same batch-per-chip/MTP3 Huawei match.

## First-Run Risks

1. Perfect-router alias propagation into all active MPI ranks. DEP8 must show
   exactly ranks `0..7`; TP4/DEP4 must show exactly ranks `0..3`. Every rank
   must record the same CuTe cache path, without assuming files are persisted.
2. Exact 8192-token prompt construction with the staged model tokenizer.
3. TRT argument names accepted by the pinned image.
4. Exact-concurrency CUDA graph memory at 512/1024.
5. One full-shape warmup plus one measured pass fitting the worker timeout.
6. `/scratch/models/DeepSeek-V4-Pro` being staged on the selected node.
7. Shared `/data/datasets` and `/data/squash` permissions.

## Fast Debug Commands

Dispatch only concurrency 8:

```bash
BENCH_REF="$(git rev-parse HEAD)"
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='trt-bench' \
  -f "inputs[ref]=$BENCH_REF" \
  -f 'inputs[test-name]=DSV4 B300 TRT offline bring-up c8' \
  -f 'inputs[concurrencies]=8'
```

Dispatch an explicit set of rows:

```bash
BENCH_REF="$(git rev-parse HEAD)"
CONCURRENCIES=32,64,128,256
TEST_NAME='DSV4 B300 TRT offline tuned c32-c256'
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='trt-bench' \
  -f "inputs[ref]=$BENCH_REF" \
  -f "inputs[test-name]=$TEST_NAME" \
  -f "inputs[concurrencies]=$CONCURRENCIES"
```

The top-level `ref` selects the branch's workflow definition. `inputs[ref]`
pins the checkout inside each benchmark job. Keep the top-level ref on
`trt-bench` and set `inputs[ref]` to the pushed commit SHA.

First bring-up dispatched with this command:

- Run: `27461421427`
- URL: `https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27461421427`
- Branch commit: `3c74b5048ffb4cdf3ad4867ae65d87171196452f`
- Dispatched: `2026-06-13T08:17:24Z`

List artifacts:

```bash
gh api /repos/SemiAnalysisAI/InferenceX/actions/runs/$RUN_ID/artifacts \
  --jq '.artifacts[].name'
```

Download one failed row:

```bash
gh run download "$RUN_ID" --repo SemiAnalysisAI/InferenceX \
  -n offline-trt-conc-8 -D /tmp/offline-trt-c8
tar -xzf /tmp/offline-trt-c8/offline_debug_conc8.tar.gz \
  -C /tmp/offline-trt-c8/debug
```

Inspect concise failures without dumping all request samples:

```bash
jq '{
  status,
  phase,
  failure_kind,
  error,
  candidate,
  aggregate,
  resolved_parallelism
}' /tmp/offline-trt-c8/debug/*_worker.json
```

## Acceptance Criteria

Do not call a row valid unless:

- status is `success`
- all prompts are exactly 8192 tokens
- every request emits exactly 625 tokens
- resolved shape matches the candidate and MTP3; LM-head TP matches the
  candidate
- perfect-router propagation validation passes for every active rank
- a single-candidate experiment contains one warmup and one measured pass
  from one fresh engine, with `concurrency` request samples
- legacy tuning also contains one pass per successful candidate and a fresh
  one-pass final engine
- derived and wall throughput are both present
- derived step throughput and step TPOT are present
- token/step and effective acceptance are present
- raw TRT acceptance is either populated or explicitly marked unavailable
- Huawei output and step-rate ratios are present only for matching DEP8/MTP3
  c8/c32/c64 rows; TP4/DEP4 rows must keep them null

Capacity failures at 512/1024 remain useful rows. They are not successful
performance measurements.

## Runtime Logging

- The launcher records allocation, timeout, telemetry start, controller start,
  and artifact finalization.
- The controller records corpus and tuning phases. While a worker is active,
  it emits one heartbeat every 60 seconds with the latest explicit worker
  phase and the worker-log filename.
- Each worker records engine initialization, warmup, all measured passes,
  perfect-router validation, aggregation, shutdown, and failure tracebacks.
- Launcher finalization reports the persistent CuTe cache path and file count.
- Native TRT output remains in the per-worker log and is not streamed into the
  controller log. This keeps Actions readable while preserving full debug
  detail in `offline_debug_concN.tar.gz`.
- Each pass has one flushed line before its timer starts and one after metrics
  extraction; neither line is included in the measured `LLM.generate()` wall
  interval.

## Parallel Optimization Matrix

The first optimization matrix is checked in at
`utils/bench_offline/b300_huawei_experiments.json`. It contains ten
single-candidate jobs and is deliberately all DEP8/MTP3:

- c8: prior wait30 control, wait30 plus LM-head TP, wait0 plus LM-head TP
- c32: prior wait30/balance-off control, the same with LM-head TP, and the
  production attention-DP wait30/balance-on/default-timeout knobs projected
  onto DEP8
- c64: prior wait0 control, wait0 plus LM-head TP, the production
  attention-DP wait30/balance-on/default-timeout knobs projected onto DEP8,
  and wait30/balance-off plus LM-head TP

Each job creates one fresh engine, performs one full-shape warmup, and records
one measured pass. The workflow fans out at most ten jobs, so it removes
the seven-engine serial multiplier that made c512 take 4h19.

Exact dispatch:

```bash
BENCH_REF="$(git rev-parse HEAD)"
EXPERIMENTS="$(jq -c . utils/bench_offline/b300_huawei_experiments.json)"
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='trt-bench' \
  -f "inputs[ref]=$BENCH_REF" \
  -f 'inputs[test-name]=DSV4 B300 TRT offline optimization' \
  -f "inputs[experiments]=$EXPERIMENTS" \
  -f 'inputs[salloc-time]=90' \
  -f 'inputs[worker-timeout]=3600'
```

The collector keys rows by experiment ID, so repeated concurrencies are
valid. Artifacts are named `offline-trt-job-EXPERIMENT_ID`.

The checked-in second-stage matrix repeats the c32 DEP8 control and LM-head-TP
candidate, then tests the production TP4 and DEP4 shapes:

```bash
BENCH_REF="$(git rev-parse HEAD)"
EXPERIMENTS="$(jq -c . utils/bench_offline/b300_stage2_experiments.json)"
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='trt-bench' \
  -f "inputs[ref]=$BENCH_REF" \
  -f 'inputs[test-name]=DSV4 B300 TRT offline stage2 one-pass' \
  -f "inputs[experiments]=$EXPERIMENTS" \
  -f 'inputs[salloc-time]=90' \
  -f 'inputs[worker-timeout]=3600'
```

Run `27476355772` dispatched that exact stage-two matrix on
`2026-06-13T19:11:30Z`, pinned to commit
`781065563a87740a094e4d5e70f19b4786f87fe1`.

The third-stage matrix tests TRT's documented per-local-rank ADP capacity
against legacy global controls. It remains DEP8/MTP3 and Huawei-comparable:

```bash
BENCH_REF="$(git rev-parse HEAD)"
EXPERIMENTS="$(
  jq -c . utils/bench_offline/b300_stage3_local_batch_experiments.json
)"
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='trt-bench' \
  -f "inputs[ref]=$BENCH_REF" \
  -f 'inputs[test-name]=DSV4 B300 TRT offline local-rank batch' \
  -f "inputs[experiments]=$EXPERIMENTS" \
  -f 'inputs[salloc-time]=90' \
  -f 'inputs[worker-timeout]=3600'
```

For DEP8 c8/c32/c64, `attention_dp_batch_mode=local-rank` resolves engine and
CUDA graph capacity to 1/4/8. Results are rejected if TRT resolves different
`max_batch_size`, `max_num_tokens`, or CUDA graph batch sizes.

## Huawei Comparison Audit

Huawei's c8/c32/c64-matched table is explicitly offline decode. Its published
throughput is step throughput because every row equals
`global_batch_size / step_tpot / chips`. Huawei also publishes 1.44 accepted
drafts for MTP3, or 2.44 emitted tokens per step.

Use two primary ratios:

1. Step-rate ratio: B300 derived decode steps/GPU divided by Huawei's
   published decode steps/chip. This isolates execution rate.
2. Actual output ratio: B300 emitted output tokens/GPU divided by Huawei step
   throughput times Huawei's own 2.44-token yield.

The prior artifacts contain enough request telemetry to reconstruct direct
step TPOT. The baseline was:

| Conc | B300 step TPOT ms | B300 step/s/GPU | Huawei step TPOT ms | Huawei step/s/chip | B300/Huawei step |
|---:|---:|---:|---:|---:|---:|
| 8 | 30.026 | 33.305 | 17.64 | 56.70 | 0.587 |
| 32 | 44.268 | 90.358 | 19.03 | 210.16 | 0.430 |
| 64 | 66.065 | 121.094 | 20.61 | 388.23 | 0.312 |

Based on the same prior final output-token results, B300/Huawei actual-output
ratios were:

- c8: `116.091 / (56.70 * 2.44) = 0.839`
- c32: `286.253 / (210.16 * 2.44) = 0.558`
- c64: `369.071 / (388.23 * 2.44) = 0.390`

The old TRT-yield-normalized ratios were `0.587`, `0.431`, and `0.313`, close
to the reconstructed direct step ratios because request-level token yield was
fairly uniform within each run. New results record direct step TPOT and step
throughput without relying on that approximation.

The current B300 baseline is plausibly behind for two independent reasons:

- Its decode-step rate is lower, increasingly so at the larger matched
  batches.
- Its observed MTP yield at c32/c64 is about 3.16/3.04 tokens per step, while
  Huawei reports 2.44. B300's higher yield helps output throughput, but not
  enough to overcome its slower step execution.

The Huawei guide also describes LM-head TP during decode, forced EPLB,
specialized fused SAS/LI/compressor/mHC kernels, and multiple streams
overlapping attention, compressor, routed/shared experts, and scheduler
metadata. The prior offline B300 baseline already used perfect routing and
fused mHC, but incorrectly left LM-head TP disabled. The first matrix isolates
that mismatch before exploring non-comparable TP4 or alternate-MTP shapes.

## Run History

### Run 27461421427

- Dispatched `2026-06-13T08:17:24Z` from commit
  `3c74b5048ffb4cdf3ad4867ae65d87171196452f`.
- Concurrency 8 completed successfully on `b300-015`, Slurm job `20771`.
- Six tuning attempts and one fresh three-pass final run completed.
- Winner: `wait10`, balance on, overlap on, CUDA graph on.
- Final mean token TPOT: `9.741 ms`.
- Final derived output throughput: `102.658 tok/s/GPU`.
- Final wall output throughput: `60.350 tok/s/GPU`.
- Final observed token yield: `3.143 tokens/step`.
- Effective MTP3 acceptance: `(3.143 - 1) / 3 = 71.4%`.
- Perfect-router proof contained eight `trt_mpi_entry` rank processes.
- All prompts were 8192 tokens and all outputs were 625 tokens.
- The run exposed two bounded reporting bugs: raw zero draft counters were
  shown as 0% instead of unavailable, and `provenance.git_revision` was null.

### Run 27462769691

- URL: `https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27462769691`
- Dispatched `2026-06-13T09:21:31Z`; completed
  `2026-06-13T10:09:23Z`.
- Ran commit `51c894535d8e780ac9561cf53872fb52ea8037ef`.
- Concurrency 8 completed successfully on `b300-015`, Slurm job `20781`.
- Winner: `wait60`, balance on, overlap on, CUDA graph on.
- Final mean token TPOT: `8.857 ms`.
- Final derived output throughput: `112.904 tok/s/GPU`.
- Final wall output throughput: `84.634 tok/s/GPU`.
- Final observed token yield: `3.459 tokens/step`.
- Effective MTP3 acceptance: `82.0%`.
- Commit provenance, null raw speculative counters, DEP8/MTP3 shape,
  eight-rank perfect-router propagation, 8192-token prompts, 625-token
  outputs, and three final passes all validated.
- The winner differed from run `27461421427`, and derived throughput differed
  by about 10%. Per-pass output digests also differed. Exact TRT source review
  showed that the pinned PyTorch sampler ignores request-level seeds and
  advances one engine-global seed-42 generator. The original one-pass tuning
  was therefore not strong enough to call the scheduler choice verified.
- Follow-up correction: pool three passes for every tuning candidate and
  require a 3% improvement before replacing the earlier candidate.

### Run 27463874862

- URL: `https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27463874862`
- Dispatched `2026-06-13T10:13:58Z`.
- Branch commit: `6c196d6b7f7cd10080c6f396bd9d43fcb4f7407b`.
- Concurrency: `8`.
- Purpose: verify three-pass candidate tuning and the 3% winner threshold
  before dispatching broader B300 concurrency points.
- Completed `2026-06-13T11:01:31Z` on `b300-015`, Slurm job `20789`.
- The final measurement itself succeeded with `wait30`, `9.163 ms` token
  TPOT, `109.130 tok/s/GPU` derived throughput, and `3.342 tokens/step`.
- Tuning verification failed: metadata requested three tuning passes, but
  every candidate artifact contained `pass_count=1` and `request_samples=8`.
  `trt_worker.py` still forced tune mode to one pass after the controller
  passed `--passes 3`.
- This is a bounded harness bug. Fix the worker pass-count branch and rerun c8
  before launching any broader concurrency points.

### Run 27464928729

- URL: `https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27464928729`
- Dispatched `2026-06-13T11:03:13Z`.
- Branch commit: `404d87e69024eb0aac808bacaf8e498bf78a75dc`.
- Concurrency: `8`.
- Purpose: final c8 gate after fixing tune mode to honor `--passes 3`.
- Required artifact proof: every successful tuning attempt must report
  `pass_count=3` and `request_samples=24`.
- Completed `2026-06-13T11:58:34Z` on `b300-016`, Slurm job `20800`.
- All six tuning attempts reported `pass_count=3`,
  `request_samples=24`, and three-element derived-throughput, wall-throughput,
  and output-digest arrays.
- Winner: `wait30`, balance on, overlap on, CUDA graph on.
- Three-pass tuning result: `114.063 tok/s/GPU` derived throughput.
- `wait10` reached `116.178 tok/s/GPU`, only `1.85%` above `wait30`;
  balance-off reached `115.723 tok/s/GPU`, only `1.46%` above. Neither met
  the 3% replacement threshold. Overlap-off was slower at
  `105.151 tok/s/GPU`.
- Fresh final result: `8.614 ms` token TPOT,
  `116.091 tok/s/GPU` derived throughput,
  `85.343 tok/s/GPU` wall throughput, and `3.486 tokens/step`.
- The fresh final derived result was `1.78%` above its tuning result, within
  the 3% stability threshold.
- Commit provenance, temperature-1/global-seed-42 sampling metadata, null raw
  counters, effective acceptance, DEP8/MTP3 shape, eight-rank perfect-router
  propagation, exact 8192-token prompts, and exact 625-token outputs all
  validated.
- This run passes the c8 gate. Broader c32/c64/c128/c256 measurements may
  proceed.

### Run 27466148578

- URL: `https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27466148578`
- Dispatched `2026-06-13T11:59:41Z`.
- Branch commit: `35b71eed55694a9e5bd87babc39d9f9851c5eaf7`.
- Concurrencies: `32,64,128,256`.
- Purpose: collect tuned low/mid-concurrency B300 results after the c8
  three-pass gate passed.
- Completed `2026-06-13T13:01:25Z`. The workflow conclusion is `failure`
  only because c128 and c256 hit the corpus-construction bug below. The c32
  and c64 jobs and their artifacts are valid.
- c32 ran on `b300-017`, Slurm job `20807`. Winner:
  `wait30-balance-off`. Three-pass tuning was `290.065 tok/s/GPU`; the fresh
  final was `286.253 tok/s/GPU`, a `-1.31%` drift.
- c32 final: `13.974 ms` token TPOT, `175.470 tok/s/GPU` wall throughput,
  `3.162 tokens/step`, `72.1%` effective MTP3 acceptance, and
  `1967.12 ms` mean TTFT. Huawei normalized token throughput was
  `664.630 tok/s/chip`, giving a B300/Huawei ratio of `0.431`.
- c64 ran on `b300-016`, Slurm job `20806`. Winner: `wait0`. Three-pass
  tuning was `371.771 tok/s/GPU`; the fresh final was
  `369.071 tok/s/GPU`, a `-0.73%` drift.
- c64 final: `21.676 ms` token TPOT, `216.654 tok/s/GPU` wall throughput,
  `3.036 tokens/step`, `67.9%` effective MTP3 acceptance, and
  `3533.63 ms` mean TTFT. Huawei normalized token throughput was
  `1178.769 tok/s/chip`, giving a B300/Huawei ratio of `0.313`.
- For both successful rows, all six tune workers and the fresh final worker
  completed three measured passes. Every measured and warmup request emitted
  exactly 625 tokens; DEP8/MTP3, LM-head TP off, exact 8192-token corpora,
  eight-rank perfect-router propagation, null raw speculative metrics, and
  commit provenance all validated.
- c128 and c256 failed during corpus construction before TRT initialization:
  some later InfiniteBench contexts rendered to 8191 or 8193 tokens but not
  8192 when truncating only at source-token boundaries. c32 and c64 continued.
- Bounded fix: when token-boundary truncation skips 8192, search recorded
  whitespace adjustments at the context/suffix boundary, then a small
  decoded context-tail trim. Never insert pad or synthetic token IDs.

### Run 27466362872

- URL: `https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27466362872`
- Dispatched `2026-06-13T12:09:22Z`.
- Branch commit: `e82c9902269e186214b09ce744b258d4c4940b99`.
- Concurrencies: `128,256`.
- Purpose: retry the two corpus-construction failures with recorded exact
  boundary/tail adjustment fallback.
- c128 completed successfully `2026-06-13T13:28:27Z` on `b300-015`,
  Slurm job `20811`.
- Of 128 prompts, 20 required one recorded `space` boundary adjustment.
  No prompt required context-tail trimming. All prompts encoded to exactly
  8192 tokens.
- c128 winner: `wait60`. Three-pass tuning was `634.922 tok/s/GPU`; the
  fresh final was `629.687 tok/s/GPU`, a `-0.82%` drift.
- c128 final: `25.409 ms` token TPOT, `105.283 tok/s/GPU` wall throughput,
  `3.195 tokens/step`, `73.2%` effective MTP3 acceptance, and
  `38404.93 ms` mean TTFT. The large TPOT/wall-throughput divergence is real:
  this benchmark tunes decode-token TPOT, while `wait60` delays scheduling.
- All six tune workers and the fresh final worker completed three measured
  passes with 384 request samples each. Exact output counts, DEP8/MTP3,
  LM-head TP off, eight-rank perfect-router propagation, null raw
  speculative metrics, and commit provenance all validated.
- c256 completed successfully `2026-06-13T14:15:22Z` on `b300-018`,
  Slurm job `20812`.
- Of 256 prompts, 25 required one recorded `space` boundary adjustment.
  No prompt required context-tail trimming. All prompts encoded to exactly
  8192 tokens.
- c256 winner: `wait60`. Three-pass tuning was `949.853 tok/s/GPU`; the
  fresh final was `926.062 tok/s/GPU`, a `-2.50%` drift and still within the
  3% stability bound.
- c256 final: `34.555 ms` token TPOT, `84.654 tok/s/GPU` wall throughput,
  `3.191 tokens/step`, `73.0%` effective MTP3 acceptance, and
  `105979.48 ms` mean TTFT. As at c128, the large decode-TPOT/wall divergence
  is caused by the selected `wait60` scheduling delay.
- All six c256 tune workers and the fresh final worker completed three
  measured passes with 768 request samples each. Exact output counts,
  DEP8/MTP3, LM-head TP off, eight-rank perfect-router propagation, null raw
  speculative metrics, and commit provenance all validated.
- The run and its collector completed successfully
  `2026-06-13T14:15:36Z`.

### Run 27467637477

- URL: `https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27467637477`
- Dispatched `2026-06-13T13:07:05Z`.
- Branch commit: `e82c9902269e186214b09ce744b258d4c4940b99`.
- Concurrencies: `512,1024`.
- Exact trigger:

  ```bash
  gh api -X POST \
    /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
    -f ref='trt-bench' \
    -f 'inputs[ref]=trt-bench' \
    -f 'inputs[test-name]=DSV4 B300 TRT offline c512-c1024 capacity' \
    -f 'inputs[concurrencies]=512,1024'
  ```

- Purpose: determine whether each high-concurrency row is measurable or an
  explicit `capacity_failure`.
- c1024 first attempt ran on `b300-019`, Slurm job `20820`, and is not a
  capacity result. At `2026-06-13T13:45:05Z`, UCX reported RoCE GID-table
  changes followed by `Transport retry count exceeded`; several MPI ranks
  aborted. The parent did not exit cleanly, so the 3600-second worker guard
  eventually recorded a timeout.
- c1024 prompt construction itself passed: 90 of 1024 prompts used one
  recorded `space` boundary adjustment, none used context-tail trimming, and
  every prompt encoded to exactly 8192 tokens.
- c512 completed successfully on `b300-016`, Slurm job `20821`.
- All six c512 tune workers and the fresh final worker completed three
  measured passes, with 1536 request samples and eight marked perfect-router
  rank processes each.
- Winner: `wait60`, balance on, overlap on, CUDA graph on.
- Final c512 mean token TPOT: `56.142 ms`.
- Final c512 derived output throughput: `1139.957 tok/s/GPU`.
- Final c512 wall output throughput: `54.373 tok/s/GPU`.
- Final c512 observed token yield: `3.196 tokens/step`.
- Final c512 effective MTP3 acceptance: `73.2%`.
- Final c512 mean TTFT: `346826.413 ms`.
- The final c512 shape resolved to DEP8, LM-head TP off, and MTP3. All 512
  prompts were exactly 8192 tokens; 55 used a recorded whitespace boundary
  adjustment and none used synthetic padding.
- `collect-results` completed and published both rows. The workflow conclusion
  is failure only because c1024 timed out; the c512 performance row is valid.

### Run 27469092334

- URL: `https://github.com/SemiAnalysisAI/InferenceX/actions/runs/27469092334`
- Dispatched `2026-06-13T14:11:06Z`.
- Branch commit: `2e837d99f11cfa6bf99f338d571e6aec7880b0ed`.
- Concurrency: `1024`.
- Exact trigger:

  ```bash
  gh api -X POST \
    /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
    -f ref='trt-bench' \
    -f 'inputs[ref]=trt-bench' \
    -f 'inputs[test-name]=DSV4 B300 TRT offline c1024 infra retry' \
    -f 'inputs[concurrencies]=1024' \
    -f 'inputs[worker-timeout]=5400'
  ```

- Purpose: retry c1024 on a fresh allocation after the `b300-019` RoCE/UCX
  failure. The longer guard avoids confusing a legitimate slow full-shape
  pass with the previous network-induced hang.
- The retry ran on `b300-015`, Slurm job `20831`. It had no UCX, NCCL, OOM,
  GPU, or capacity error. GPU telemetry showed all eight GPUs still near
  100% utilization when the 5400-second guard terminated the first
  `wait30` tune worker.
- One required warmup-plus-three-pass candidate did not complete in 90
  minutes. Six tuning candidates plus a fresh final engine therefore cannot
  fit the current 500-minute workflow and Slurm allocation without changing
  the benchmark method or making a much larger operational change.
- Stop c1024 here. It is `unmeasured_runtime_limit`, not
  `capacity_failure`, and must not be reported as a performance row.

## Stage-Four Kernel Matrix

`utils/bench_offline/b300_stage4_kernel_experiments.json` is the next bounded
ten-job offline matrix. Every job creates one engine, runs one full-shape
warmup, and records one measured pass.

The source-backed candidates are:

- CuTE DSL FP4 paged-MQA logits. TRT commit `c185066` contains the tuned
  implementation from TensorRT-LLM changes `#13929` and `#14133`.
- Tight `max_seq_len=8832`. The fixed workload needs 8817 committed sequence
  tokens, and 8832 is the next 128-token KV-block boundary.
- `print_iter_log=false`. This removes native TRT iteration printing only;
  benchmark phase logs and 60-second heartbeats remain enabled.
- DEP4 local-rank sizing. At global c64, engine and graph batch capacity is
  16 per active attention-DP rank instead of the legacy global 64.

Pinned TRT's DeepSeek-V4 config loader preserves checkpoint-derived sparse
attention values when the CuTE DSL flag is supplied. It uses explicit-field
checks for checkpoint `index_topk` and `window_size`, keeps the checkpoint's
full compression-ratio list, and rebuilds
`DeepSeekV4SparseAttentionConfig` with the requested DSL flag. The worker also
requires `IS_CUTLASS_DSL_AVAILABLE=true` so a labeled DSL row cannot silently
run the fallback kernel.

Dispatch only after run `27476767599` releases all ten B300 slots:

```bash
BENCH_REF="$(git rev-parse HEAD)"
EXPERIMENTS="$(
  jq -c . utils/bench_offline/b300_stage4_kernel_experiments.json
)"
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='trt-bench' \
  -f "inputs[ref]=$BENCH_REF" \
  -f 'inputs[test-name]=DSV4 B300 TRT offline kernel optimization' \
  -f "inputs[experiments]=$EXPERIMENTS" \
  -f 'inputs[salloc-time]=90' \
  -f 'inputs[worker-timeout]=3600'
```
