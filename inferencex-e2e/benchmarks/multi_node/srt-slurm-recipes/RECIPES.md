# srt-slurm recipes

InferenceX owns the recipes in this directory. For every NVIDIA srt-slurm launch, the srt driver ([`infx/launch/drivers/srt/checkout.py`](../../../infx/launch/drivers/srt/checkout.py)) makes a job-local Git clone of the pinned submodule and copies this entire tree into `recipes/`. It records the actual revision in `srt-slurm-sha.txt`; power lanes copy that revision into `power-producer-sha.txt` for result validation.

The shared version is the Git submodule pointer at [`utils/srt-slurm`](../../../utils/srt-slurm), currently [v2.43.4](https://github.com/NVIDIA/srt-slurm/releases/tag/v2.43.4) (`848f72d45b05af0fc082a9d1df49a8b4e7e61507`). Update that submodule pointer when upgrading, then run the recipe and integration checks. Do not add model-specific checkout branches to launchers.

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
- A master-config entry with two or more rows of one recipe design keeps them in one `<agg|disagg>[-<axis>]-variants.yaml`; the axis only tells bundles in one directory apart. `base` holds everything the points share, with `schema: 2` beside it, and each row gets one plain `override_<topology>_c<conc>` block (or a similarly point-identifying name, without dates or hardware) holding only its deltas. An override's `null` deletes the base key, so a null that a point needs, such as TensorRT-LLM's `cuda_graph_config:`, stays in `base`. Rows select it with `srt-recipe: <bundle>.yaml:override_<name>`, relative to `srt-recipe-dir`. Do not add `zip_override_*` blocks or share a bundle across entries, and keep one override per row: recipe references participate in eval grouping. An entry that reruns another entry's points selects that entry's overrides instead of copying them. Standalone files remain for single-row entries, adapted upstream recipes whose `# Source:` base materially differs, and points too different to share a base, such as a TP low-latency recipe beside wide-EP ones.
- Update `srt-recipe-dir`, `srt-recipe`, and `eval-srt-recipe` references in master configs, launcher path rules, workflow filters, and local documentation together when moving a file. Preserve upstream source URLs as provenance and leave historical performance-changelog entries unchanged. No aliases for the old layout are provided.

Shared runtime assets stay under `configs/` beside the model directories; they are not standalone recipes. The four files in `configs/dsv4-moe-load-balancer-configs/` are copied verbatim from NVIDIA/srt-slurm commit `deb1dfd9934398664f92d194169c183e009da83b`, preserving the EPLB initial expert assignments formerly used by the DSV4 TRT recipes; no checked-in recipe currently references them. The srt driver ([`infx/launch/drivers/srt/checkout.py`](../../../infx/launch/drivers/srt/checkout.py)) stages them into the job checkout's `configs/` directory for the recipes' bind mounts. Keeping a recipe in this tree does not activate it; the master configs determine the benchmark matrix.

## TileRT

TileRT uses the pinned upstream srt-slurm submodule. Recipes select `roles.prefill.engine: vllm`, `roles.decode.engine: tilert`, and `frontend.type: tilert-router`.

## Schema 2 and master configuration

Recipes use `schema: 2`, `engine`, and `roles`. Each worker role owns its node count, worker count, GPU allocation, environment, and engine arguments. `resources` retains GPU hardware facts. `placement` controls the frontend and benchmark location, `services` describes auxiliary processes, and `dynamo.source` selects the Dynamo package or source revision.

| Recipe field | `configs/nvidia-master.yaml` field |
|---|---|
| `roles.prefill.workers` | `prefill.num-worker` |
| `roles.decode.workers` | `decode.num-worker` |
| `roles.prefill.args.tp-size` (SGLang) | `prefill.tp` |
| `roles.prefill.args.ep-size` (SGLang) | `prefill.ep` |
| `roles.prefill.args.enable-dp-attention` | `prefill.dp-attn` |
| `benchmark.concurrencies` | `conc-list` (bound for `power: true` rows) |
| `telemetry:` | search-space `power: true` |
| Recipe directory, relative to this tree | entry-level `srt-recipe-dir` |
| Recipe file, optionally with a `base`, `override_<name>`, or `zip_override_<name>[<index>]` selector | search-space `srt-recipe`; `eval-srt-recipe` for eval-only real verification |

Keep the recipe and master configuration synchronized. The launcher executes the recipe; the master configuration supplies result labels and scheduling metadata. For aggregate recipes use `roles.agg`; `roles.decode.nodes: colocate` shares prefill nodes and contributes no additional worker nodes to scheduling.

All referenced recipes must be checked in: srt-slurm 2 ships curated examples instead of the historical `recipes/` archive. The initial migration restores 204 previously external recipes and two still-referenced AgentX recipes from InferenceX history. Master-config paths follow the layout above; existing override selectors are preserved.

## Recipe fragments

Active recipes, fixed-sequence and AgentX, single- and multi-node, are fragments: native srt-slurm YAML holding only recipe-specific settings. At launch the fragment is composed and bound:

1. The lane's shared block in [`configs/srt-recipes/`](../../../configs/srt-recipes) (`fixed-sequence-{single,multi}.yaml` or `agentic-{single,multi}.yaml`) is merged under the fragment (under `base` for bundles). A multi-node row with `power: true` also gets [`telemetry-dcgm.yaml`](../../../configs/srt-recipes/telemetry-dcgm.yaml), its exporter on the cluster's `srt-slurm.power-exporter-port`; the lane must allow power (`POWER_LANES`). The fragment wins: mappings merge and lists replace, so a fragment may keep telemetry tunables such as `collector_join_timeout_seconds`.
2. The row's selector picks the variant. A single-node variant may name its `benchmark.env.CONC` and `KV_OFFLOADING`, which keep that point paired with its tuning.
3. The binder ([`workload.py`](../../../infx/srt_slurm/workload.py)) writes the matrix point into the selected recipe: `model.path: hf:<model>`, `model.container: <image>`, `model.precision`, `identity.container.image` (as a registry reference) and `identity.model.repo` when the fragment declares `identity.container`/`identity.model`, and `benchmark.concurrencies` when telemetry is enabled. Fixed-sequence recipes also get `benchmark.env.ISL`/`OSL`; single-node recipes get `MODEL` and `CONC`, and fixed-sequence ones `RANDOM_RANGE_RATIO` and `USE_CHAT_TEMPLATE` (`true` exactly when the recipe speculates). AgentX recipes get the row's `KV_OFFLOADING`, and a DRAM point its budget, `TOTAL_CPU_DRAM_GB`. Multi-node AgentX recipes get the client's `RESULT_DIR`, `AIPERF_DATASET_MMAP_CACHE_DIR` and `HF_HUB_CACHE` from the launcher, which derives them from the volumes it mounts (`volume-mounts`, `agentic-volume-mounts`, lane mounts); a fragment that sets `HF_HOME` keeps its own cache. The multi-node client reads `CONC_LIST`, `CONC`, `MODEL` and the other matrix inputs from the job environment, so fragments do not copy them; single-node AgentX points get them as runtime arguments.
4. A component a repository setup script installs gets the master config's version. When `setup_script`, or a service `preamble` that runs `/configs/<script>`, is one of the installers in the binder's `INSTALLERS`, the binder writes the point's `router` or `kv-offload-backend` version as `ROUTER_VERSION` or `KV_OFFLOAD_BACKEND_VERSION` into the top-level `environment` (workers and static routers) or into that service's `env` (services do not inherit `environment`). A point that does not declare the version fails, and a `vllm-router==` pin in `SETUP_PIP_PACKAGES` must equal the master router version.

A fragment that sets a key the binder or launcher writes (`_POINT_KEYS`, `BOUND_ENV` in `workload.py`, and `ROUTER_VERSION` or `KV_OFFLOAD_BACKEND_VERSION` in any env; single-node variants may still name `CONC` and `KV_OFFLOADING`) fails before submission, even with the bound value. `hf:<model>` resolves to the cluster's staged checkpoint (`models.entries`) unless a `models.OVERRIDES` row serves the Hub snapshot, and the master image to its staged container.

Recipes do not hardcode host DRAM sizes either. A DRAM point's budget, the matrix `total-cpu-dram-gb` in decimal GB, is the cluster's `available-cpu-dram-mib` (capped at 3 TB) times the row's `dram-utilization` times the share of a node's GPUs it covers: the point's GPUs on a single node, or the GPUs the multi-node prefill (or aggregated) worker uses on each of its nodes. After binding, a whole value `'@dram.<name>'` becomes:

| Reference | Value |
| --- | --- |
| `@dram.total-gb` | the budget, GB |
| `@dram.total-bytes` | the budget, bytes |
| `@dram.per-gpu-gb` | the budget over the GPUs it covers, whole GB |
| `@dram.per-gpu-bytes` | the budget over the GPUs it covers, bytes |

References resolve to integers, or to decimal strings in env values and argument lists (e.g. after `--l1-size-gb`), including whole values inside JSON object strings such as `kv-transfer-config`. Per-rank pools take the `per-gpu` sizes, and Mooncake `global_segment_size` takes bytes because it reads `GB` as GiB. Unknown names, embedded references and references on a point without a DRAM budget fail. SimpleCPU, LMCache CPU and `--l1-size-gb` sizes must be references; measured HiCache, TRT-LLM host-cache and Mooncake sizes may stay literal if they fit the node's share. When a backend allocates several pools from the budget, such as the KV and Mamba HiCache pools of a hybrid model, the master's `dram-utilization` sizes one of them.

Recipes do not hardcode cluster hardware facts: a value `'@fabric.<name>'`, anywhere in the recipe, becomes the job cluster's `srt-slurm.fabric` field after binding (lists comma-joined; fields in [CONFIGS.md](../../../configs/CONFIGS.md#runners)). Matrix generation fails a selected variant whose reference is embedded in a longer string, names no field, or names one that a cluster its runner label reaches does not set. The recipe fingerprint hashes references as written, so it stays cluster-independent. A value a recipe chooses on purpose, such as a permuted rail order or a subset of the devices, stays literal.

Inspect what the launcher submits without a cluster:

```bash
uv run --extra recipes infx generate --config-key 'dsr1-fp8-h200-*' --output-dir /tmp/recipes
```

It writes one bound recipe per fixed-sequence or AgentX point, validated by the pinned srtctl, plus a `manifest.json` that maps each file to its matrix point and to the one cluster its runner label schedules on (`null` when the label spans several). Multi-node points take their cluster's binder inputs (DCGM exporter port, AgentX client paths, GPUs per node); a point that needs them fails unless its runner label names one cluster and the lane accepts the row's `power`. Fabric references take that cluster's facts; when the label spans several clusters they stay references. Launch-time edits such as the job name, health-check floor and runtime `--set` values are not applied.

## Recipe fingerprints

The planner gives every benchmark row a `recipe-fingerprint`: a SHA-256 of its matrix fields (except `conc` and `exp-name`) and the bound variant the launcher submits, composed with its shared block (and the telemetry block for `power: true`). Concurrency values, the job `name` and everything the launcher adds for a cluster stay out, so a recipe keeps one fingerprint across concurrencies and clusters. An `eval-srt-recipe` contributes only its path; rows without an srt-slurm recipe hash their matrix fields alone.

The planner selects variants without srtctl ([`variants.py`](../../../infx/srt_slurm/variants.py)), the way the launcher does, so a row that no variant serves fails planning instead of its launch. Before writing a `perf-changelog.yaml` entry, list the config keys whose results the change invalidates:

```bash
uv run python -m infx.matrix.changed --base origin/main
```

The base matrix comes from the base revision's own generator, and both sides are fingerprinted under their own recipes. Each listed key shows its point counts and the points (fingerprint and concurrency) that were added or removed. `--config-files` limits the comparison, and `--json` prints a machine-readable report.

## Migration and validation

Install the shared pin in an isolated environment, then use its CLI:

```bash
srtctl migrate --in-place -f benchmarks/multi_node/srt-slurm-recipes/dsr1/sglang
# Repeat for the other model/engine directories.
python -m pytest infx/tests/matrix/ -q
python -m infx.matrix.generate full-sweep \
  --config-files configs/nvidia-master.yaml \
  --framework dynamo-sglang dynamo-trt dynamo-vllm --multi-node
```

Validate recipes with the exact launcher pin, including all override variants. The current migration CLI has no `--verify`; use `srtctl dry-run` for each selected recipe after migration, supplying launcher-provided values such as `benchmark.concurrencies` through `--set` when needed. For a path-only reorganization, compare generated matrices before and after with the path mapping applied; all other fields, including eval selection and node counts, must match. A passing local schema check does not replace the full hardware sweep and evals.

The initial migration also resolves compatibility issues that `srtctl migrate` cannot fix itself:

- SGLang Model Gateway recipes use `frontend.type: sglang-router`; in v2.36.0, `sglang` selects a direct worker without a router.
- Duplicate YAML keys retain the value selected by the former PyYAML loader.
- DCGM telemetry uses `collect_interval_ms: 1000` instead of `provider` and `default_frequency`. The collector derives its shutdown budget; an explicit ten-second budget is too short for the current validator. Dedicated discovery-service placement is preserved from the original recipes. The pinned upstream runtime rejects telemetry with dedicated infrastructure nodes; this remains a power compatibility blocker rather than changing the original topology to satisfy validation.
- DeepSeek-V4 vLLM benchmarks use the supported `custom_tokenizer` loader. Retired `warmup_req_rate: inf` fields are removed; the current upstream client uses its fixed warmup rate of 250 requests per second.
- The power reader accepts both generations of samples CSV while validating utilization values and continuing to compute board energy from watts.
- Post-eval selection uses native `post_eval.command` and `post_eval.passthrough_env` with [`srt_eval.sh`](../srt_eval.sh). TRT AgentX recipes declare their existing Dynamo fork with `dynamo.source.git`; launchers no longer rewrite the srt-slurm source.

Append a new entry to the physical end of `perf-changelog.yaml` for every recipe or runtime change. Preserve all historical bytes. Validate the PR with `full-sweep-fail-fast`, including evals, before following the repository's review and artifact-reuse merge process.
