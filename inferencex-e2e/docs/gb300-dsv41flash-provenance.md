# GB300 DeepSeek-V4.1-Flash AgentX result provenance

Audit date: 2026-10-22. Scope: `dsv41flash-fp4-gb300-dynamo-vllm-agentic-dspark-disagg`,
PRs [#4266](https://github.com/SemiAnalysisAI/InferenceX/pull/4266) and
[#4336](https://github.com/SemiAnalysisAI/InferenceX/pull/4336), and the serving recipes at
[`2633e9a4`](https://github.com/SemiAnalysisAI/InferenceX/commit/2633e9a443958ad87d3dbea1cb98bc829712fcaa).

The two DEP32 measurements reused by #4266 remain evidence for their historical
configuration. They do not validate the current DEP32 recipes. Keep their original
run, attempt, recipe fingerprint, and 80 GiB/rank provenance when displaying them
beside the #4336 result. The #4266 changelog already discloses this difference;
the reuse is not, by itself, a measurement error.

## Point-level decision

All disaggregated points below use DEP2 prefill. GPU counts are serving GPUs,
excluding dedicated frontend/benchmark resources and unused GPUs on partially
occupied nodes.

| Source run | Point | Serving GPUs | Measured prefill CPU KV per rank | Disposition |
|---|---|---:|---:|---|
| 39207089292 | 16P1D, DEP32 decode, c10888 | 64 | 80 GiB | Reuse as historical/default-selection result. Current recipe inherits 60 GiB and explicit active-request selection; current-config performance remains unmeasured by these two runs. |
| 39207089292 | 6P1D, DEP32 decode, c3018 | 44 | 80 GiB | Same historical qualification; a targeted current-config measurement is needed only to claim current-config performance. |
| 39382568249 | 9P1D, DEP16 decode, c7480 | 34 | **70 GiB** | Reuse. Its effective serving recipe matches the audited current recipe, including the per-point 70 GiB override. No repeat is needed for this audit. |
| 39207089292 | 12P1D, DEP16 decode, c9888 | 40 | 80 GiB | Has a performance artifact, but no successful eval job in the inspected attempts; not one of #4266's two appended points and no longer an active recipe. Retain as historical diagnostic evidence. |
| 39207089292 | 12P1D, DEP32 decode, c7388 | 56 | 80 GiB configured | No successful performance job/artifact found across the inspected attempts. Not a measured curve point and not a reason to revive a retired point. |
| 39382568249 | 10P1D, DEP16 decode, c8400 | 36 | 70 GiB configured | Failed/cancelled performance attempts; excluded from #4336's final matrix. Open [#4352](https://github.com/SemiAnalysisAI/InferenceX/pull/4352) proposes it separately; it cannot stand in for either DEP32 point. |

Run 39382568249 also contains aggregated TP4 c1/c16 results. Those belong to the
aggregated configuration and do not validate a disaggregated point.

There is **no matched topology/concurrency pair between the two source runs**.
Client throughput/latency points can be compared as different deployments on the
same workload, with their configuration labels retained. They cannot establish
the isolated effect of the routing change, the KV-budget change, or unchanged
performance after #4336. The two historical DEP32 points can be compared as
operating points within the old configuration family, not as a constant-topology
concurrency experiment.

## Effective recipes and measurement conditions

The benchmark jobs report these test revisions:

- Historical run: head `d9998aabcc9ed069b2f16a2a185a62703faa8812`;
  `BENCHMARK_REF=863f0a6840f64770645e2444bb6d9b82581f0331`.
- Updated run: head `b03874384cdb4886e2a61d41d14348c076cf1791`;
  `BENCHMARK_REF=76359017dba5b336231617f1dfb11f9febfe9c15`.

For each run, the disaggregated source recipe at the head and benchmark revision
is identical. Read the [historical recipe](https://github.com/SemiAnalysisAI/InferenceX/blob/863f0a6840f64770645e2444bb6d9b82581f0331/inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes/dsv41flash/vllm/gb300-fp4/agentx/disagg-variants.yaml)
and [updated recipe](https://github.com/SemiAnalysisAI/InferenceX/blob/76359017dba5b336231617f1dfb11f9febfe9c15/inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes/dsv41flash/vllm/gb300-fp4/agentx/disagg-variants.yaml),
rather than inferring executed settings from the final merged PR body.

| Setting | Run 39207089292 | Run 39382568249 / audited current serving recipes |
|---|---|---|
| Decode selection | No explicit `worker_selection` override; image/framework default | `decode-least-loaded`, `dynamo-two-tier-cost-fn`, `cache_threshold: 1.0` |
| Session affinity | No explicit soft-mode override | `DYN_ROUTER_SESSION_AFFINITY_MODE=soft` |
| Frontend KV events | No `no-kv-events` override | `no-kv-events: true` |
| Active-request weight | `10000000` already present | `10000000`; its presence alone does not identify the routing algorithm |
| Base prefill KV bytes/rank | 85899345920 = 80 GiB | 64424509440 = 60 GiB; c7480 overrides to 75161927680 = 70 GiB |
| Node exclusions | No recipe exclusion | Explicit exclusion of nodes with observed Engram host-allocation failures |
| AgentX client | `agentx-harness==1.0.11` | `agentx-harness==1.0.12` |

Both runs use `vllm/vllm-openai:cu134-nightly-aarch64-dev-20261019`,
Dynamo `1.6.0.dev20261010`, FP4 model weights, FP8 KV,
pinned prefill Engram tables (`use_thp: false`), HBM decode Engram,
and `VLLM_RS_TOKEN_IDS_AS_BYTES=1`. These settings are already present in the
historical run; their changes relative to #4336's base branch are not additional
differences between the two actual runs. The model image is identified by tag in
the result JSON; this audit does not establish container-digest or model-weight
revision equality.

The DEP32 overrides retain the same per-rank decode limits between the old and
current source: c10888 uses max-num-seqs 76, max-num-batched-tokens 760,
graph maximum 456, and async KV admits 5; c3018 uses 18, 180, 108, and 2.
The c7480 DEP16 point uses 126, 1260, 756, and 6. These controls and the changed
topologies preclude a routing-only attribution.

All three retained main points use:

- Full `semianalysis_cc_traces_weka_062126`, train split, 393 dataset entries.
- Formal `DURATION=3600` and `AIPERF_EXPERIMENTAL_FAST=0`.
- DSpark five speculative tokens with launcher-bound synthetic acceptance length
  3.51 on both roles; these are synthetic-acceptance AgentX measurements.
- Session headers enabled via `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID`.
- A warmup phase followed by a duration-based profile and cutoff drain.

Actual warmup completions were 120633, 33431, and 82862 for c10888, c3018, and
c7480. Profile summaries reported 128, 17, and 261 cancellations at cutoff,
respectively. The retained JSON records 327, 85, and 231
`InvalidInferenceResultError` records dropped across the source records; those
counts are not a measured profiling server-error rate. Preserve this accounting.

The 1.0.12 change in [#4318](https://github.com/SemiAnalysisAI/InferenceX/pull/4318)
and [agentx-harness#49](https://github.com/SemiAnalysisAI/agentx-harness/pull/49)
fixes warmup baseline labeling for server-counter deltas. Do not compare old and
new server-counter/cache rates as if their windows were identical. The three
JSONs have null measured GPU/CPU/overall cache-hit fields; their router estimate
is not measured CPU KV reuse or evidence that a lower KV limit is harmless.
If those telemetry metrics are needed, first reprocess retained raw data with
explicit phase boundaries. That is an offline task, not a reason for a sweep.

## Retained client metrics and provenance

Numbers below come from the small `bmk_agentic_*` JSON artifacts, specifically
`request_metrics`, not the live console's differently summarized table.
They describe different deployments; no before/after percentage is implied.

| Point | Artifact ID | Attempt | Output tokens/s/GPU | Mean TTFT (s) | Mean TPOT (ms) | Retained throughput window (s) |
|---|---:|---:|---:|---:|---:|---:|
| c10888 | 12182920722 | 1 | 6865.90 | 4.31432 | 4.88 | 3629.91517 |
| c3018 | 12228403071 | 3 | 3203.01 | 1.94269 | 2.93 | 3629.61211 |
| c7480 | 12257816084 | 2 | 7365.97 | 10.46110 | 6.36 | 3629.93873 |

- [c10888 benchmark job](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39207089292/job/117912468463);
  fingerprint `9fd788f3e6aa24950f4d777052f681027ad80f0ad13f68e8d373370f4928b153`.
- [c3018 benchmark job](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39207089292/job/118250601986);
  fingerprint `0afb4726de86203c6634fee7c583b38849f5b94c197e8898d010098757afd864`.
- [c7480 benchmark job](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39382568249/job/118503471538);
  fingerprint `966655192ae0e580578d060d0ee1b53456f2c7d854c831fe7f47022235d9d3a1`.

Both parent runs ended cancelled after retries of other jobs. Successful jobs
carried forward across attempts are not independent repeated measurements.
Successful eval jobs exist for c10888 (attempt 5), c3018 (attempt 1), and c7480
(attempt 6); whole-run cancellation alone does not invalidate their retained
performance artifacts or establish that the entire sweep passed.

## CPU KV metadata caveat

`allocated_cpu_dram_gb` is generated from a proportional host allowance, then
copied into results; it is not read from `SimpleCPUOffloadConnector.cpu_bytes_to_use`.
The GB300 runner declares 920000 MiB/node and four GPUs/node. The old
840 GiB comment in the master config was incorrect.

| Scope | Serialized/generated prefill allowance per DEP2 worker (decimal GB) | Actual CPU KV per DEP2 worker |
|---|---:|---:|
| Historical c10888/c3018 JSON | 183 | 160 GiB = 171.79869184 GB |
| Updated c7480 JSON | 160 | 140 GiB = 150.32385536 GB |
| Current c10888/c3018 recipe with shared metadata ratio 0.333333 | 160 (generated allowance, not a measurement) | 120 GiB = 128.84901888 GB |

The JSONs also report `decode_allocated_cpu_dram_gb` as 367/367/321. These are
proportional allowances, not proof of decode CPU KV allocation: the decode recipe
uses the NIXL consumer without a CPU offload connector and keeps Engram in HBM.
Display the recipe-derived KV budget with its role and units; retain original
artifact fields as source metadata. Do not reinterpret these allowances as
measured memory usage. Correcting documentation or presentation of these fields
requires no performance rerun.

## Reuse and minimal follow-up

1. Reuse c7480 for its matching updated serving recipe and keep the two DEP32
   results labeled historical/default selection/80 GiB per rank. Preserve source
   fingerprints; never relabel an old measurement with a current recipe hash.
2. Treat #4266 as **two appended historical points**, not a four-point
   replacement. Treat #4336's retained disaggregated point as **9P1D c7480 at
   70 GiB/rank**, not four measured 60 GiB points. Its base is still 60 GiB.
3. No additional measurement is necessary to retain accurately labeled historical
   curves or correct these descriptions. To claim current DEP32 performance,
   first look for existing artifacts matching the effective recipe and harness;
   neither of the two audited runs supplies them.
4. If no matching result exists and that claim is needed, measure only
   **16P1D DEP32 c10888** and **6P1D DEP32 c3018** with the current route and
   60 GiB/rank. Keep the image, workload, synthetic acceptance, warmup, formal
   duration, and per-point controls explicit, and collect the applicable eval
   evidence. A full sweep, c7480 repeat, or revival of removed points is unnecessary.
5. The [#4336 staging workflow](https://github.com/SemiAnalysisAI/InferenceX-app/actions/runs/39473519083)
   succeeded. The [#4266 staging workflow](https://github.com/SemiAnalysisAI/InferenceX-app/actions/runs/39479759413)
   failed its database-write check after ingest steps; this is a publication
   issue, not evidence that the benchmark must be rerun.
   [Merge Ingest for #4266](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39479975332)
   and [#4336](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39474277651)
   report success, but that alone does not verify the current production chart's
   per-point labels. Inspect the downstream rows/tooltip before claiming a
   production metadata correction.

This audit changes documentation and one misleading comment only. It does not
rewrite historical changelog entries, source artifacts, database rows, or benchmark
settings, and it launches no new measurements.
