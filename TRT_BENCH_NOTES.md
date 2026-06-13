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
- MTP config:
  `{"decoding_type": "MTP", "max_draft_len": 3}`
- Perfect router is read from unprefixed `ENABLE_PERFECT_ROUTER` in
  `tensorrt_llm._torch.modules.fused_moe.interface.MoE`.
- `MpiPoolSession` explicitly forwards environment names beginning with
  `TRTLLM` or `TLLM`.
- `trt_mpi_entry.worker_main` is installed into the pinned IPC proxy before
  `LLM` construction. It recreates `ENABLE_PERFECT_ROUTER` and marks every
  rank before importing TRT's real worker entry.

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
  -f 'inputs[test-name]=DSV4 TRT offline bring-up c8' \
  -f 'inputs[concurrencies]=8'
```

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
- final result contains three measured passes from one fresh final engine
- derived and wall throughput are both present
- token/step and weighted acceptance are present
- Huawei conversion, when applicable, uses token/step

Capacity failures at 512/1024 remain useful rows. They are not successful
performance measurements.
