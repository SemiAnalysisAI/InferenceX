# Multinode benchmarks: architecture and the runner contract

Multinode benchmarks are orchestrated through
[srt-slurm](https://github.com/NVIDIA/srt-slurm)'s `srtctl`. The design rule
is the same one that governs single-node benchmarks:

- **`configs/`** declares *what* to run (image, model, recipe, parallelism,
  concurrencies). A multinode search-space point names its recipe via
  `CONFIG_FILE=recipes/...` in `additional-settings`.
- **`benchmarks/`** is **cluster-agnostic**. `run.sh` in this directory is
  the single orchestrator every NVIDIA multinode job goes through. It must
  run unchanged on any Slurm cluster — no hardcoded paths, partitions, or
  model locations.
- **`runners/`** is the pluggable, cluster-specific layer. The multinode
  branch of `runners/launch_<cluster>.sh` is a *cluster profile*: it may
  hardcode anything about its cluster, and its whole job is to satisfy the
  environment contract below and then `source` `run.sh`.
- **`configs/runners.yaml` `model-paths:`** is the registry of where model
  checkpoints are staged on each cluster. Profiles resolve `MODEL_PATH` /
  `SRT_SLURM_MODEL_PREFIX` / `SERVED_MODEL_NAME` from it via
  `runners/lib/multinode.sh:infx_resolve_model_paths` instead of if/elif
  ladders.

```
configs/*-master.yaml ──▶ matrix ──▶ benchmark-multinode-tmpl.yml
                                          │  bash runners/launch_<cluster>.sh
                                          ▼
                        runners/launch_<cluster>.sh   (cluster profile)
                          - cluster facts (partition, arch, gpus/node, ...)
                          - model path from configs/runners.yaml registry
                          - squash import (however this cluster needs it)
                          - optional hooks + srtslurm.yaml extras
                          │  source
                          ▼
                benchmarks/multi_node/srt_slurm/run.sh   (cluster-agnostic)
                          - resolve srt-slurm pin (sources.tsv, default main)
                          - clone + overlay srt-slurm-recipes/
                          - render srtslurm.yaml, srtctl apply, tail, collect
```

## The contract (`runners/launch_*.sh` → `run.sh`)

Required — cluster facts the profile must export before sourcing `run.sh`:

| Variable | Meaning |
|---|---|
| `INFX_CLUSTER` | Short cluster tag used in srtctl `--tags` (e.g. `gb300`) |
| `SLURM_ACCOUNT`, `SLURM_PARTITION` | Slurm submission settings |
| `INFX_GPUS_PER_NODE` | GPUs per compute node |
| `INFX_ARCH` | `x86_64` or `aarch64` (passed to `make setup ARCH=`) |
| `SQUASH_FILE`, `NGINX_SQUASH_FILE` | Enroot squash files, already imported/staged by the profile |
| `MODEL_PATH`, `SRT_SLURM_MODEL_PREFIX` | From the runners.yaml registry (`infx_resolve_model_paths`) |

Optional — defaults in parentheses:

| Variable | Meaning |
|---|---|
| `INFX_SLURM_TIME_LIMIT` (`4:00:00`) | srtslurm.yaml `default_time_limit` |
| `INFX_SRT_WORK_DIR` (`$GITHUB_WORKSPACE`) | Base dir for the per-run srt-slurm clone, venv, and outputs. Must be visible to compute nodes |
| `INFMAX_WORKSPACE` (`$GITHUB_WORKSPACE`) | Compute-visible InferenceX checkout (recipes mount it into containers) |
| `INFX_VENV_PYTHON` | Interpreter to pin the srtctl venv to (shared-FS venvs need a path that exists on compute nodes) |
| `INFX_CONTAINER_KEY` (`$IMAGE`) | Container alias key recipes reference |
| `INFX_EXTRA_CONTAINER_ALIASES`, `INFX_EXTRA_MODEL_ALIASES` | Extra alias keys mapped to `SQUASH_FILE` / `MODEL_PATH` |
| `INFX_SRTSLURM_EXTRA` | Verbatim YAML appended to srtslurm.yaml (`default_mounts:`, `use_*_sbatch_directive:` flags, ...) |
| `INFX_HEALTH_CHECK_MAX_ATTEMPTS` | Overrides the recipe's health-check budget (slow shared FS) |
| `INFX_NO_PREFLIGHT` (`0`) | Skip srtctl's login-node model-path stat (compute-only NVMe). Agentic jobs always skip |
| `SRTCTL_SETUP_SCRIPT` | `--setup-script` for srtctl apply. Recipe-specific values belong in the config entry's `additional-settings`, not in profiles |
| `SRT_SLURM_REPO`, `SRT_SLURM_REF` | Pin override; beats `sources.tsv` |
| `INFX_DRY_RUN` (`0`) | Print the resolved plan and stop before side effects |

Optional hooks — bash functions the profile may define; `run.sh` calls them
if they exist:

- `infx_hook_post_clone` — after the srt-slurm checkout (local patches).
- `infx_hook_edit_recipe <recipe-path>` — after the job-name/health-check
  edits (e.g. inject an `sbatch_directives: exclude:` for a bad node).
- `infx_hook_post_submit <job-id>` — after JOB_ID extraction (verification).
- `infx_hook_cleanup` — before the outputs cleanup at the end.

Workflow-provided variables (`benchmark-multinode-tmpl.yml`): `IMAGE`,
`MODEL`, `MODEL_PREFIX`, `PRECISION`, `FRAMEWORK`, `ISL`, `OSL`,
`CONC_LIST`, `CONFIG_FILE`, `IS_AGENTIC`, `RESULT_FILENAME`, `RUNNER_NAME`,
`RUN_EVAL`, `EVAL_ONLY`, worker topology (`PREFILL_*`, `DECODE_*`), etc.
Profiles must not overwrite these.

## The srt-slurm pin

The standard checkout is **`NVIDIA/srt-slurm` @ `main`** for every cluster,
scenario, and framework. `sources.tsv` (same directory) is the burn-down
registry of exceptions — shell-glob rows matched first-to-last on
`(cluster, scenario, framework, model-prefix, precision)`. Keep it empty;
when main breaks a family, pin it there with a comment and an upstream
issue, and remove the row when fixed. `SRT_SLURM_REPO` / `SRT_SLURM_REF`
env overrides (e.g. from `additional-settings`) beat the table for one-off
testing.

## Recipes and the custom benchmark

`benchmarks/multi_node/srt-slurm-recipes/` is overlaid wholesale onto the
clone (`vllm/` → `recipes/vllm/`, `sglang/` → `recipes/sglang/`,
`configs/*.sh` → `configs/`). `CONFIG_FILE` selects exactly one recipe, so
unused recipes are inert.

New recipes should use srt-slurm's `benchmark.type: custom` with
InferenceX's own client instead of srt-slurm's downstream sa-bench fork:

- fixed-seq-len: `benchmarks/multi_node/bench_serving_srt.sh` (drives
  `utils/bench_serving` and writes `results_concurrency_*` files in the
  layout the collector already understands)
- agentic-coding: `benchmarks/multi_node/agentic_srt.sh`

This keeps the measurement client identical across single-node and
multinode, and across clusters.

## Dry run

From any login node (or laptop) you can render a profile's plan without
touching the cluster:

```bash
IS_MULTINODE=true INFX_DRY_RUN=1 \
  IMAGE=... MODEL=... MODEL_PREFIX=dsv4 PRECISION=fp4 FRAMEWORK=dynamo-vllm \
  ISL=8192 OSL=1024 CONFIG_FILE=recipes/vllm/... RUNNER_NAME=gb300-nv_0 \
  GITHUB_WORKSPACE=$PWD RESULT_FILENAME=dryrun \
  bash runners/launch_gb300-nv.sh
```

which prints the resolved pin, work dir, srtslurm.yaml, and srtctl
arguments. `utils/multinode_contract/` runs this across the whole support
matrix in CI.
