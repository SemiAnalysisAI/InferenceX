# TRT Bench Working Notes

These notes are for the next agent running and debugging `trt-bench`.

## Scope

- Branch-only, never merge to `main`.
- B300, one node, eight GPUs.
- TensorRT-LLM only.
- Offline `LLM.generate()`, no server or generic InferenceX processing.
- Start with concurrency 8 before spending GPUs on the full matrix.

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
- Tune candidates with three measured passes and require a 3% improvement.
  A single pass was too sensitive to output-dependent MTP acceptance.

Do not assume a newer TRT release has the same field names. If the image
changes, inspect the exact source first.

## Existing B300 Choices Intentionally Rejected

The normal B300 recipe is a useful environment reference, but this benchmark
does not inherit all of its behavior:

- It must not reduce MTP3 to MTP2 at high concurrency.
- It must not enable LM-head TP in attention DP.
- It must not change to MegaMoe for this 8K workload.
- It must not substitute continuous batching or HTTP request timing.

## First-Run Risks

1. Perfect-router alias propagation into all MPI ranks. The marker must show
   eight enabled `source=trt_mpi_entry` PIDs.
2. Exact 8192-token prompt construction with the staged model tokenizer.
3. TRT argument names accepted by the pinned image.
4. Exact-concurrency CUDA graph memory at 512/1024.
5. Three-pass final run fitting the per-worker timeout.
6. `/scratch/models/DeepSeek-V4-Pro` being staged on the selected node.
7. Shared `/data/datasets` and `/data/squash` permissions.

## Fast Debug Commands

Dispatch only concurrency 8:

```bash
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='trt-bench' \
  -f 'inputs[ref]=trt-bench' \
  -f 'inputs[test-name]=DSV4 B300 TRT offline bring-up c8' \
  -f 'inputs[concurrencies]=8'
```

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
- resolved shape is DEP8 with LM-head TP off and MTP3
- perfect-router propagation validation passes
- every successful tuning attempt contains three measured passes and
  `3 * concurrency` request samples
- final result contains three measured passes from one fresh final engine
- derived and wall throughput are both present
- token/step and effective acceptance are present
- raw TRT acceptance is either populated or explicitly marked unavailable
- Huawei conversion, when applicable, uses token/step

Capacity failures at 512/1024 remain useful rows. They are not successful
performance measurements.

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
- Status at dispatch: queued.
- c128 and c256 failed during corpus construction before TRT initialization:
  some later InfiniteBench contexts rendered to 8191 or 8193 tokens but not
  8192 when truncating only at source-token boundaries. c32 and c64 continued.
- Bounded fix: when token-boundary truncation skips 8192, search recorded
  whitespace adjustments at the context/suffix boundary, then a small
  decoded context-tail trim. Never insert pad or synthetic token IDs.
