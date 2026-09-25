# srt-slurm recipes

**English** | [中文](./RECIPES_zh.md)

InferenceX owns the recipes in this directory. Every NVIDIA srt-slurm launcher uses `setup_srt_slurm()` in [`runners/slurm_utils.sh`](../../../runners/slurm_utils.sh), makes a job-local Git clone of the pinned submodule, and copies this entire tree into `recipes/`. The shared helper records the actual revision in `srt-slurm-sha.txt`; power lanes copy that revision into `power-producer-sha.txt` for result validation.

The shared version is the Git submodule pointer at [`utils/srt-slurm`](../../../utils/srt-slurm), currently [v2.2.1](https://github.com/NVIDIA/srt-slurm/releases/tag/v2.2.1) (`984180e5b8755aef85e9995048b5a16cb5336bce`). Update that submodule pointer when upgrading, then run the recipe and integration checks. Do not add model-specific checkout branches to launchers.

InferenceX requires srt-slurm 2.0 or newer and `schema: 2` recipes. Legacy recipe layouts are unsupported; migrate them before adding them to this tree.

## Directory and filename convention

Store every recipe at `<model-prefix>/<engine>/<gpu>-<precision>/<workload>/<recipe>.yaml`:

```text
dsr1/sglang/b200-fp4/8k1k/disagg-stp-mtp-variants.yaml
glm5.2/sglang/h200-fp8/agentx/disagg-1p1d-pcp8-tp8-dp8-mtp6-hicache.yaml
qwen3.5/trtllm/gb300-fp4/agentx/disagg-variants.yaml
```

- Use the master config's `model-prefix` and `precision` labels. Engines are `sglang`, `vllm`, `trtllm`, and `tilert`; frontend selection remains explicit inside the recipe. Hardware directories use GPU types such as `b200` and `gb300`, rather than cluster names.
- Workloads are `1k1k`, `8k1k`, or `agentx`. Existing bundles spanning several fixed sequence lengths use `fixed-seq-len`; keep their override selectors intact.
- Use lowercase, hyphen-separated filenames beginning with `agg` or `disagg`. Include topology and the settings that distinguish sibling recipes, such as parallelism, batch size, concurrency, MTP, offload, or cache configuration. Avoid dates, numbered latency/throughput labels, and repeating the model or hardware already in the path.
- In topology names, `1p4d` denotes prefill/decode worker counts, not necessarily physical nodes. Role-qualified `p-tp4` and `d-tp8` identify prefill/decode TP; `b` denotes batch size and `c` concurrency. The YAML is authoritative for runtime settings.
- Name override bundles `*-variants.yaml`. Multi-node AgentX recipes that differ only per configuration share one bundle per master-config entry, usually `agg-variants.yaml` or `disagg-variants.yaml`: `base` holds the shared settings and each former recipe becomes a named `override_<name>` block holding only its differences (plain overrides, not `zip_override_*`). Master entries select one with `CONFIG_FILE=recipes/<dir>/<bundle>.yaml:override_<name>`. Recipes read as text by a launcher, such as power recipes with top-level `telemetry:`, stay standalone. Keep distinct sweep entry files separate even when their contents match: recipe paths participate in eval grouping. The Qwen3.5 `*-stp-sweep.yaml` and `*-mtp-sweep.yaml` pair preserves that existing distinction.
- Update `CONFIG_FILE` and `EVAL_CONFIG_FILE` references in active and deprecated master configs, launcher path rules, workflow filters, and local documentation together when moving a file. Preserve upstream source URLs as provenance and leave historical performance-changelog entries unchanged. No aliases for the old layout are provided.

Shared runtime assets stay under `configs/` beside the model directories; they are not standalone recipes. The four files in `configs/dsv4-moe-load-balancer-configs/` are copied verbatim from NVIDIA/srt-slurm commit `deb1dfd9934398664f92d194169c183e009da83b`, preserving the EPLB initial expert assignments used by 17 DSV4 TRT recipes. `setup_srt_slurm()` stages them into the job checkout's `configs/` directory for the recipes' bind mounts. Keeping a recipe in this tree does not activate it; the master configs determine the benchmark matrix.

## TileRT exception

TileRT (`FRAMEWORK=tilert`) uses the same pinned NVIDIA checkout as every other framework. The TileRT backend and `tilert-router` frontend are absent from the NVIDIA pin, so [`runners/srt-slurm/patches/tilert-backend-router.patch`](../../../runners/srt-slurm/patches/tilert-backend-router.patch) adds them, ported from the SemiAnalysisAI/srt-slurm fork ([SemiAnalysisAI/srt-slurm#13](https://github.com/SemiAnalysisAI/srt-slurm/pull/13)). Remove the patch once those features are available upstream.

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

All referenced recipes must be checked in: srt-slurm 2 ships curated examples instead of the historical `recipes/` archive. The initial migration restores 204 previously external recipes and two still-referenced AgentX recipes from InferenceX history. Master-config paths follow the layout above; existing override selectors are preserved.

## Migration and validation

Install the shared pin in an isolated environment, then use its CLI:

```bash
# Verify each supported recipe directory before rewriting it.
srtctl migrate --verify -f benchmarks/multi_node/srt-slurm-recipes/dsr1/sglang
srtctl migrate --in-place -f benchmarks/multi_node/srt-slurm-recipes/dsr1/sglang
# Repeat for the other model/engine directories.
# Use the pinned TileRT fork for glm5.1/tilert/.
python -m pytest utils/matrix_logic/ -q
python -m infx.matrix.generate full-sweep \
  --config-files configs/nvidia-master.yaml \
  --framework dynamo-sglang dynamo-trt dynamo-vllm --multi-node
```

Validate recipes with the exact launcher pin, including all override variants. For a path-only reorganization, compare generated matrices before and after with the path mapping applied; all other fields, including eval selection and node counts, must match. A passing local schema check does not replace the full hardware sweep and evals.

The initial migration also resolves compatibility issues that `srtctl migrate` cannot fix itself:

- SGLang Model Gateway recipes use `frontend.type: sglang-router`; in v2.2.1, `sglang` selects a direct worker without a router.
- Duplicate YAML keys retain the value selected by the former PyYAML loader.
- DCGM telemetry uses `collect_interval_ms: 1000` instead of `provider` and `default_frequency`. The collector derives its shutdown budget; an explicit ten-second budget is too short for the current validator. Dedicated discovery-service placement is preserved from the original recipes. The pinned upstream runtime rejects telemetry with dedicated infrastructure nodes; this remains a power compatibility blocker rather than changing the original topology to satisfy validation. H200 custom recipes declare a default concurrency that the launcher replaces before submission.
- DeepSeek-V4 vLLM benchmarks use the supported `custom_tokenizer` loader. Retired `warmup_req_rate: inf` fields are removed; the current upstream client uses its fixed warmup rate of 250 requests per second.
- The power reader accepts both generations of samples CSV while validating utilization values and continuing to compute board energy from watts.
- Post-eval selection uses native `post_eval.command` and `post_eval.passthrough_env` with [`srt_eval.sh`](../srt_eval.sh). TRT AgentX recipes declare their existing Dynamo fork with `dynamo.source.git`; launchers no longer rewrite the srt-slurm source.

Append a new entry to the physical end of `perf-changelog.yaml` for every recipe or runtime change. Preserve all historical bytes. Validate the PR with `full-sweep-fail-fast`, including evals, before following the repository's review and artifact-reuse merge process.
