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

The client runs in the serving image and installs its `sentencepiece`, `datasets`,
and `pandas` dependencies before measurement. Recipe-owned `USE_CHAT_TEMPLATE`
retains the legacy client behavior, including chat formatting for MTP.
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

## Opt-in workflow pilot

[`configs/pilots/h200-srt.yaml`](../configs/pilots/h200-srt.yaml) selects only
`cluster:h200-dgxc`, 8k1k, TP8, and concurrencies 4, 16, and 64.
It now includes DeepSeek-R1 FP8 without speculation, DeepSeek-R1 FP8 MTP,
and Qwen3.5 FP8 with EP8. The MTP recipe preserves the embedded head,
EAGLE with two steps/three draft tokens, and real verification. Qwen preserves
its 9236-token context, FP8 KV cache, and graph capture size equal to concurrency;
native zipped overrides bind the graph size and client concurrency together.
The binder rejects a selector whose concurrency disagrees with the matrix. The search-space `srt-recipe` field
passes a native file/selector through the matrix and workflow to the existing
H200 pool launcher. Production `h200` coverage, including CoreWeave, is unchanged.

The launcher checks the recipe's model, image, precision, topology, and workload
against matrix metadata before submission. It resolves the model to its staged
cluster path and passes the exact recipe image URI to native SRT/Pyxis container
startup. It requests an exclusive node and binds concurrency and artifact inputs
with native `--set`. Missing model assets fail before submission; the pilot does
not depend on the legacy launcher's separately managed squash cache.
Plain `sglang` submissions use the shared automatic acceptance connector.

Submission uses native JSON output. The launcher waits for a successful Slurm
allocation exit, preserves the result basename, and stages raw results and GPU
sampling sidecars for the existing processor and uploads. A native log archive,
submission manifest, and SRT commit identify the run. Failed jobs retain available
artifacts; cancellation targets only the submitted job.

Dispatch the workflow definition from the draft branch, with `ref` set to the
exact pushed commit:

```bash
gh workflow run e2e-tests.yml --ref codex/single-node-srt-slurm \
  -f ref=<COMMIT> -f test-name='native H200 SRT pilot' \
  -f generate-cli-command='test-config --config-keys dsr1-fp8-h200-sglang --config-file configs/pilots/h200-srt.yaml --conc 4 --no-evals' \
  -f require-power=true
```

Submit only one recipe and one `--conc` value per E2E run, and wait for that
allocation to finish before submitting the next. Check cluster load first.
Use the lowest, middle, and highest configured points (4/16/64 here) instead
of a full sweep; compare existing matching baselines, then run a fresh legacy
point only if a difference needs investigation. This keeps GPU usage bounded.

The pilot requires `--no-evals`: evals are still rejected explicitly. A passing
throughput run alone is not accuracy or performance-parity qualification.

## Before enabling the replacement

- Preserve both H200 runner paths: the current `h200` label includes
  `h200-dgxc-slurm` and `h200-cw`. Do not silently drop CoreWeave or pretend its
  Docker execution is already covered by a Slurm recipe.
- Connect eval context, real-verification evals, and eval artifact staging;
  qualify the wired result and GPU power paths without changing publication format.
- Compare legacy and native commands, then qualify startup, throughput, accuracy,
  power, cancellation, and cleanup on the same image/model/hardware. Coordinate
  existing smoke/vendor evaluation work rather than duplicating it.
- Expand to the other active single-node recipes after this path is qualified,
  including speculative decoding and KV offload. Coordinate AMD capabilities
  with Cam's fork work. Retire legacy scripts only after their callers move.

Keep the migration PR draft. Local schema checks and stubbed client tests do not
establish performance parity or GPU qualification.
