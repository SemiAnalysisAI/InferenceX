# Single-node SRT-Slurm migration

**English** | [中文](./single-node-srt-migration_zh.md)

This draft moves active single-node serving settings into native SRT-Slurm YAML.
The first candidate is built alongside the existing route; it is not a production
cutover. The existing master config, runner selection, dependency pin, and result
ingestion remain in use until the replacement has runtime evidence.

## Ownership and scope

Alec owns the single-node recipe migration. Cam owns the fork pin, AMD support,
and AMD runtime cleanup. Keep reusable execution changes small enough to send
upstream; the intended destination is NVIDIA SRT-Slurm, with the SemiAnalysisAI
fork as an intermediate dependency. Do not import the separate prepared-job or
publication systems from the earlier H100 pilot into this work.

At InferenceX main `4ab85c1e`, the active master files contain 110 single-node
configurations; deprecated archives are excluded:

| Vendor | SGLang | vLLM | TRT-LLM | ATOM | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| NVIDIA | 47 | 13 | 10 | 0 | 70 |
| AMD | 21 | 8 | 0 | 11 | 40 |

This is an inventory of migration work, not a claim that all these paths are
supported by the current SRT fork. The AMD stack and non-Slurm execution need
their own capability checks.

## First native recipe

[`8k1k.yaml`](../benchmarks/single_node/srt-slurm-recipes/dsr1/sglang/h200-fp8/8k1k.yaml)
ports `dsr1-fp8-h200-sglang` from
[`dsr1_fp8_h200.sh`](../benchmarks/single_node/fixed_seq_len/dsr1_fp8_h200.sh).
It uses native schema 2 and native `zip_override_concurrency` expansion: one
TP8 aggregate worker on one H200 node, with separate jobs at concurrency
4, 8, 16, 32, and 64. It preserves the model, image, 8k1k workload, server
environment, and explicit serving flags from the active recipe.

SRT owns allocation, container startup, endpoint selection, readiness, and server
cleanup. Account, partition, mounts, image caches, and exclusive allocation belong
to the cluster/launcher integration, not the workload YAML. The candidate disables
SRT observability to avoid adding a second telemetry workload beside the existing
GPU sampler; runtime qualification must still compare effective native defaults.

The native `custom` benchmark invokes
[`srt_fixed_sequence.sh`](../benchmarks/single_node/srt_fixed_sequence.sh).
This client uses the existing `run_benchmark_serving` helper and GPU sampler.
It preserves `10 * concurrency` requests, `2 * concurrency` warmups, random length
variation, completions API behavior, and the existing JSON result format. The
helper now accepts an explicit base URL so the client reaches the endpoint SRT
selected. Existing callers retain their previous local endpoint.

The client runs in the serving image and retains the legacy `sentencepiece`
installation. This is client compatibility glue, not a second server launcher.
The initial client rejects eval requests explicitly. Eval-only context handling,
eval artifacts, and cancellation qualification are required before cutover.

## Runtime inputs

The recipe provides model/workload inputs, including concurrency through native
override expansion. SRT provides `SRT_FRONTEND_HOST` and `SRT_FRONTEND_PORT`.
The launch integration must export `INFMAX_WORKSPACE` before submission; SRT
mounts it at `/infmax-workspace`. It must also supply these native overrides:

| Native override | Caller-owned value |
| --- | --- |
| `benchmark.env.RESULT_FILENAME` | Existing InferenceX result basename |
| `benchmark.env.RESULT_DIR` | `/logs`, SRT's existing per-job artifact mount |
| `benchmark.env.GPU_MONITOR_INTERVAL` | Explicit sampling interval in seconds |
| `benchmark.env.RUN_EVAL` | `"false"` for the throughput pilot |
| `benchmark.env.EVAL_ONLY` | `"false"` for the throughput pilot |

All environment values above are strings; use quoted YAML values with native
`--set`. Missing inputs fail before the client runs. No fallback runtime settings
are hidden in the workload recipe. Submit through the shared `apply_srt_recipe`
entrypoint so speculative ports later retain automatic golden-AL handling.

For a local allocation render, with SRT dependencies installed:

```bash
INFMAX_WORKSPACE="$PWD" PYTHONPATH=utils/srt-slurm/src srtctl dry-run \
  -f 'benchmarks/single_node/srt-slurm-recipes/dsr1/sglang/h200-fp8/8k1k.yaml:zip_override_concurrency[0]'
python -m pytest utils/test_srt_fixed_sequence.py
```

Without a cluster profile this renders SRT's generic scheduling defaults. It
validates configuration structure; it does not qualify a cluster or benchmark.

## Before enabling the replacement

- Add first-class recipe selection to the master/matrix/workflow contract and
  consume it in the existing pool launcher; retain one launcher per pool.
- Preserve both H200 runner paths: the current `h200` label includes
  `h200-dgxc-slurm` and `h200-cw`. Do not silently drop CoreWeave or pretend its
  Docker execution is already covered by a Slurm recipe.
- Stage the candidate in the job-local checkout and bind caller inputs using
  native `--set`; retain `nodes:1` and the existing result filename/metadata.
- Connect eval context, real-verification evals, result/eval artifact staging,
  and GPU power collection to the workflow without changing publication format.
- Compare legacy and native commands, then qualify startup, throughput, accuracy,
  power, cancellation, and cleanup on the same image/model/hardware. Coordinate
  existing smoke/vendor evaluation work rather than duplicating it.
- Expand to the other active single-node recipes after this path is qualified,
  including speculative decoding and KV offload. Coordinate AMD capabilities
  with Cam's fork work. Retire legacy scripts only after their callers move.

Keep the migration PR draft. Local schema checks and stubbed client tests do not
establish performance parity or GPU qualification.
