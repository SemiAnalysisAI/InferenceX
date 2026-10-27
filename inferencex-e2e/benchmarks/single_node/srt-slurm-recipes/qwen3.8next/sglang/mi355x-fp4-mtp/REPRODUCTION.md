# C2/C16 tuning with required golden acceptance

The current recipe retains the serving tuning and restores the original automatic golden selector. There is no natural-acceptance bypass. For Qwen3.8 NEXTN at three steps and the workflow's THINKING_MODE=thinking_on, the committed curve selects:

```text
SGLANG_SIMULATE_ACC_LEN=2.32
SGLANG_SIMULATE_ACC_METHOD=match-expected
SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
```

These values are generated from the golden YAML, not hard-coded in the recipe. Read them back from the bound config and server fingerprint in both points. Rejection sampling remains enabled, but acceptance is controlled by the golden schedule.

| Serving setting | C2 | C16 |
|---|---|---|
| TP / DP / EP | 1 / 1 / 1 | 1 / 1 / 1 |
| MTP steps / topk / draft tokens | 3 / 1 / 4 | 3 / 1 / 4 |
| Context length | 262144 | 262144 |
| Static memory / prefill chunk | 0.9 / 16384 | 0.88 / 8192 |
| Admission / decode graph maximum | 2 / 2 | 32 / 16 |
| Target and MTP monokernels | enabled | not enabled |
| ReplaySSM | disabled | enabled |
| Fused HC / fused QKV norm | enabled / enabled | disabled / not enabled |
| MLA persist | 1 | 1 |
| QSA paged prefill / RMSNorm padding | no override | 0 / 1 |
| Tuned-kernel logging | no override | 1 |
| Rejection sampling | enabled | enabled |

C1/C4/C8/C12 are unchanged. No new engine patch, precision override or sampling override is introduced. Image/model binding remains in configs/amd-master.yaml, with image digest sha256:9dc8237cf0fe52d29e0d9c2f920f4a932f2b8614505020b94b550ee6db73711c. The inherited 24 ordered patches come from #3974; original implementation credit belongs to Chang, Nikolai, Christian and Fan.

## Benchmark invariants

Use agentx-fast=false, 3600 seconds profiling, capped WEKA loader semianalysis_cc_traces_weka_062126_256k, 393 entries, seed 42 and standard 10 warmup requests per lane. Preserve streaming chat, server token counting, chat-template application, trace sampling, failure thresholds and grace periods. The harness requirement is agentx-harness==1.0.7. The golden collector's calibration sampling settings are not a replacement for the AgentX trace workload.

Previous logs identify dataset revision 8fecd2fc56694469f758f0afbbb6335ad3043740. Model/tokenizer snapshot revisions remain unrecorded (revision=None). Verify identities in new runs.

Dispatch e2e-tests.yml against the full PR commit SHA with:

```text
test-config --config-files configs/amd-master.yaml --config-keys qwen3.8next-fp4-mi355x-sglang-agentic-mtp --conc 2 16 --no-evals
```

This is a two-point performance retest, not a six-point sweep or quality evaluation. Verify golden variables and serving settings from actual runtime artifacts. A configuration-only preflight is a separate mode; agentx-fast or cancelling a full run does not implement it.

## Historical diagnostic records and audit limits

[reproduction/c2.json](reproduction/c2.json) and [reproduction/c16.json](reproduction/c16.json) are immutable historical natural-acceptance records, not the current recipe or compliant golden-acceptance evidence. Their INFX_NATURAL_ACCEPTANCE=1 setting must not be copied into this recipe. No natural-acceptance gain is claimed for the corrected submission.

The prohibited SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE flag is absent from the recipe and archived fingerprints, and the pinned SGLang source declares its default false. The tuning introduces no draft checkpoint, draft quantization, draft KV dtype or pruning override. Effective draft precision still requires comparison against checkpoint metadata and the pinned image's default handling; synthetic acceptance does not establish that condition.

The inherited setup applies engine patches. The engine-patch rule requires the introducing PR's waiver to cover them and be available on the default branch before merge. The audit did not find docs/waiver/3974.md on main; the stacked parent may need to supply and merge that waiver. This document does not grant a waiver or certify inherited draft precision. These merge prerequisites are separate from the requested C2/C16 retest.
