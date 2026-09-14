# srt-slurm recipes

**English** | [中文](./RECIPES_zh.md)

InferenceX owns the recipes in this directory. Every NVIDIA srt-slurm launcher uses `setup_srt_slurm()` in [`runners/slurm_utils.sh`](../../../runners/slurm_utils.sh), makes a job-local Git clone of the pinned submodule, and copies this entire tree into `recipes/`. The shared helper records the actual revision in `srt-slurm-sha.txt`; power lanes copy that revision into `power-producer-sha.txt` for result validation.

The shared version is the Git submodule pointer at [`utils/srt-slurm`](../../../utils/srt-slurm), currently the merge of [NVIDIA/srt-slurm#407](https://github.com/NVIDIA/srt-slurm/pull/407). Update that submodule pointer when upgrading, then run the recipe and integration checks. Do not add model-specific checkout branches to launchers.

InferenceX requires srt-slurm 2.0 or newer and `schema: 2` recipes. Legacy recipe layouts are unsupported; migrate them before adding them to this tree.

## TileRT exception

For `FRAMEWORK=tilert`, `setup_srt_slurm()` fetches the SemiAnalysisAI/srt-slurm fork directly at `6bc3f306bdafa1edfb5dded2fcda8f1ccede1bde` into the job checkout. This is the schema-2 TileRT port in [SemiAnalysisAI/srt-slurm#13](https://github.com/SemiAnalysisAI/srt-slurm/pull/13). It is the only alternate checkout; its pin lives in that helper because the TileRT backend and router are absent from the NVIDIA pin. TileRT uses the same schema-2 recipe layout and native post-eval dispatch as NVIDIA. TileRT jobs need network access to the fork at setup time. Remove the fork exception once those features are available upstream.

## Schema 2 and master configuration

Recipes use `schema: 2`, `engine`, and `roles`. Each worker role owns its node count, worker count, GPU allocation, environment, and engine arguments. `resources` retains GPU hardware facts. `placement` controls the frontend and benchmark location, `services` describes auxiliary processes, and `dynamo.source` selects the Dynamo package or source revision.

| Recipe field | `configs/nvidia-master.yaml` field |
|---|---|
| `roles.prefill.workers` | `prefill.num-worker` |
| `roles.decode.workers` | `decode.num-worker` |
| `roles.prefill.args.tp-size` (SGLang) | `prefill.tp` |
| `roles.prefill.args.ep-size` (SGLang) | `prefill.ep` |
| `roles.prefill.args.enable-dp-attention` | `prefill.dp-attn` |
| `benchmark.concurrencies` | `conc-list` |
| Recipe path, optionally with an override selector | `additional-settings: CONFIG_FILE=recipes/...yaml` |

Keep the recipe and master configuration synchronized. The launcher executes the recipe; the master configuration supplies result labels and scheduling metadata. For aggregate recipes use `roles.agg`; `roles.decode.nodes: colocate` shares prefill nodes and contributes no additional worker nodes to scheduling.

All referenced recipes must be checked in: srt-slurm 2 ships curated examples instead of the historical `recipes/` archive. The initial migration restores 204 previously external recipes and two still-referenced AgentX recipes from InferenceX history. Existing recipe paths and override selectors continue to work.

## Migration and validation

Install the shared pin in an isolated environment, then use its CLI:

```bash
# Verify each supported recipe directory before rewriting it.
srtctl migrate --verify -f benchmarks/multi_node/srt-slurm-recipes/sglang
srtctl migrate --in-place -f benchmarks/multi_node/srt-slurm-recipes/sglang
# Repeat for vllm, trtllm and the other NVIDIA directories.
# Use the pinned TileRT fork when migrating tilert/.
python -m pytest utils/matrix_logic/ -q
python -m infx.matrix.generate full-sweep \
  --config-files configs/nvidia-master.yaml \
  --framework dynamo-sglang dynamo-trt dynamo-vllm --multi-node
```

The integration workflow installs the exact launcher pin and validates every schema-2 recipe, including all override variants. A passing local schema check does not replace the full hardware sweep and evals.

The initial migration also resolves compatibility issues that `srtctl migrate` cannot fix itself:

- Duplicate YAML keys retain the value selected by the former PyYAML loader.
- DCGM telemetry uses `collect_interval_ms: 1000` instead of `provider` and `default_frequency`. The collector derives its shutdown budget; an explicit ten-second budget is too short for the current validator. Power recipes keep discovery services on the head node so samples and benchmark windows share a clock. H200 custom recipes declare a default concurrency that the launcher replaces before submission.
- DeepSeek-V4 vLLM benchmarks use the supported `custom_tokenizer` loader. Retired `warmup_req_rate: inf` fields are removed; the current upstream client uses its fixed warmup rate of 250 requests per second.
- The power reader accepts both generations of samples CSV while validating utilization values and continuing to compute board energy from watts.
- Post-eval selection uses native `post_eval.command` and `post_eval.passthrough_env` with [`srt_eval.sh`](../srt_eval.sh). TRT AgentX recipes declare their existing Dynamo fork with `dynamo.source.git`; launchers no longer rewrite the srt-slurm source.

Append a new entry to the physical end of `perf-changelog.yaml` for every recipe or runtime change. Preserve all historical bytes. Validate the PR with `full-sweep-fail-fast`, including evals, before following the repository's review and artifact-reuse merge process.
