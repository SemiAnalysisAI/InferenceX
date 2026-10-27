# Reproducing the measured C2 and C16 experiments

This recipe reproduces the serving settings used by [C2 run 40527165072](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/40527165072)
and [C16 run 40523300092](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/40523300092).
Both benchmark and result-validation steps passed. Both used natural MTP
acceptance, not the golden synthetic-acceptance schedule. Keep that distinction
visible when comparing or publishing results.

The executable settings are in `agentic.yaml`, combined with the image/model
binding in `configs/amd-master.yaml`. The measured references
[reproduction/c2.json](reproduction/c2.json) and
[reproduction/c16.json](reproduction/c16.json) preserve complete merged role
arguments/environment, 533 sanitized resolved server fields per run, relevant
image environment, client input configuration, original command, ordered patch
hashes, source hashes and artifact provenance. Original host addresses, ports and
output directories are observations and should be rebound to the new allocation.

| Setting | C2 | C16 |
|---|---|---|
| TP / DP / EP | 1 / 1 / 1 | 1 / 1 / 1 |
| MTP steps / topk / draft tokens | 3 / 1 / 4 | 3 / 1 / 4 |
| Context length | 262144 | 262144 |
| Static memory fraction | 0.9 | 0.88 |
| Prefill chunk | 16384 | 8192 |
| Admission / decode graph maximum | 2 / 2 | 32 / 16 |
| Target and MTP monokernels | enabled | not enabled |
| ReplaySSM | disabled | enabled |
| Fused HC | enabled | disabled |
| Fused QKV norm | enabled | not enabled |
| MLA persist | 1 | 1 |
| QSA paged prefill flag | not set by recipe | 0 |
| RMSNorm padding flag | not set by recipe | 1 |
| Tuned-kernel logging | not set by recipe | 1 |
| Rejection sampling | enabled | enabled |
| Acceptance policy | natural | natural |

The generation role sets `INFX_NATURAL_ACCEPTANCE=1`. The launcher honors it
before golden lookup and removes `SGLANG_SIMULATE_ACC_LEN`,
`SGLANG_SIMULATE_ACC_METHOD` and `SGLANG_SIMULATE_ACC_TOKEN_MODE` from the recipe.
Setting these variables to zero or dropping them from YAML alone would not
override automatic golden injection. Other concurrency variants retain their
existing acceptance policy. Draft precision is unchanged.

Use the master image digest
`sha256:9dc8237cf0fe52d29e0d9c2f920f4a932f2b8614505020b94b550ee6db73711c`
and the inherited 24 ordered patches from #3974. Their exact file hashes and
setup-script hash are in the measured references. No new kernel implementation
is introduced. Original implementation credit belongs to Chang, Nikolai,
Christian and Fan.

The measured client uses `agentx-harness==1.0.7`, streaming chat requests with
server token counting, trusted tokenizer code and chat-template application.
The public dataset loader is `semianalysis_cc_traces_weka_062126_256k`, backed by
`semianalysisai/cc-traces-weka-062126-256k`; logs identify dataset revision
`8fecd2fc56694469f758f0afbbb6335ad3043740`. There are 393 dataset entries,
random seed 42, 3600 seconds profiling, 10 warmup requests per lane, 1800 seconds
agentic warmup grace, 30 seconds profile grace, and a live failed-request
threshold of 0.1. The JSON input configurations preserve the other replay
timing defaults. No sampling override was passed on the recorded command;
preserve the trace loader and its sampling behavior instead of adding guessed
temperature/top-p values. The dataset loader currently resolves the repository
revision itself, so verify the resolved dataset revision on every reproduction.

For full performance runs, use the normal `e2e-tests.yml` dispatch against the
full 40-character PR commit SHA, with `agentx-fast=false`, `fail-fast=false` and
the following matrix command (replace `2` with `16` for C16):

```text
test-config --config-files configs/amd-master.yaml --config-keys qwen3.8next-fp4-mi355x-sglang-agentic-mtp --conc 2 --no-evals
```

Keep the default 3600-second duration. The two source runs skipped quality
evaluations; result validation does not certify semantic accuracy. Request
accounting records one `InvalidInferenceResultError` exclusion in each run.
Do not present these measurements as error-free or as an all-green full sweep.

Before measuring, read back resolved server arguments and environment after
binding, patch setup and launcher overrides. Verify rejection sampling, all
concurrency-specific flags and the absence of simulation variables. C2 logs
must show both monokernels initialized. Verify the same values again in the
definitive run. A YAML or matrix match alone is not runtime proof. A preliminary
configuration-only check should stop after normal artifact collection; the
full benchmark command above is not a short preflight mode.

## Limits of exact reproduction

Both measured servers used `revision=None`, and neither weights nor tokenizer
snapshot SHA was recorded in the available artifacts. The references explicitly
leave these pins unknown. Do not substitute the earlier historical experiment's
model revision and call it the revision used here. To guarantee identical
model bytes, recover the cached snapshot identity or file hashes from the source
allocation, then pin that verified revision in a separate validated change.

GPU identities were g15/692526014242 for C2 and g31/692524011214 for C16.
Different machines, caches and execution state can still change performance.
The recipe and archive reproduce known inputs; they do not guarantee identical
weights or performance until the remaining pins and effective workload have
been verified. Archive new run URL, SHA, fingerprint and patch/model/dataset
identities instead of assuming that a successful run proves equivalence.
