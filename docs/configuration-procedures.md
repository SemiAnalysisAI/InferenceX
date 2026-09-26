# Configuration Procedures

<div align="center">

**English** | [中文](./configuration-procedures_zh.md)

</div>

Use this page for benchmark configuration, recipe, image, and runner changes. It is a procedure, not a field catalog: the linked implementation and schema remain authoritative.

## Source map

| Source of truth | What it controls |
| --- | --- |
| [`configs/CONFIGS.md`](../configs/CONFIGS.md) | Master-config and runner-config field contract |
| [`utils/matrix_logic/validation.py`](../utils/matrix_logic/validation.py) | Enforced Pydantic schema and topology invariants |
| [`utils/matrix_logic/generate_sweep_configs.py`](../utils/matrix_logic/generate_sweep_configs.py) | Matrix expansion, filtering, runner lookup, and emitted job metadata |
| [`configs/nvidia-master.yaml`](../configs/nvidia-master.yaml), [`configs/amd-master.yaml`](../configs/amd-master.yaml) | Executable benchmark definitions |
| [`configs/runners.yaml`](../configs/runners.yaml) | Schedulable labels, concrete runner names, and hardware facts |
| [`benchmarks/`](../benchmarks/) and [`runners/`](../runners/) | Runtime commands and launcher routing |
| [`perf-changelog.yaml`](../perf-changelog.yaml) | Append-only benchmark trigger log |
| [`AGENTS.md`](../AGENTS.md) | Repository-wide config, MTP, changelog, and sweep rules |

Delete retired entries from the active master configs; they are not archived. Git history and `perf-changelog.yaml` keep the historical settings. For partial retirements, remove only the retired scenarios. Delete retired AMD server-registry entries and model-specific setup from `benchmarks/multi_node/amd_utils/` as well. Preserve shared dependencies needed by retained SPEED-Bench collectors, including their scheduling scores. See the [deprecation rules](../AGENTS.md#deprecating-benchmark-configs).

## Dependency submodules

Git records the exact dependency commits. [`.gitmodules`](../.gitmodules) defines the repositories: AIPerf at `utils/aiperf`, NVIDIA srt-slurm at `utils/srt-slurm`. TileRT is a documented manual fork checkout in `setup_srt_slurm()`, not a separate submodule.

Initialize them before running benchmarks locally:

```bash
git submodule update --init
```

To upgrade, fetch and check out the desired commit inside the relevant submodule, then commit the updated submodule pointer in InferenceX. Benchmark workflows already initialize submodules. Slurm launchers make a local Git clone for each job so recipe staging and runtime writes do not modify the submodule, and record the actual commit for result provenance. NVIDIA setup clones locally; TileRT setup fetches its pinned fork commit over the network.

Single-node fixed-sequence recipes use NVIDIA upstream srt-slurm. ATOM recipes use
the native `atomesh` frontend with one aggregate worker and
`enable_multiple_frontends: false`. The router's pinned official image belongs in
`frontend.container_image`: older benchmark worker images do not include AToMesh.
Keep `model.container` aligned with the master config's worker `image`; changing the
router image does not require changing the worker image. TRT-LLM recipes use native
`engine.served_model_name`, without duplicating that flag in `roles.agg.extra_args`.
The former fork's direct ATOM frontend is not required.

### Cluster profiles

Launchers that use srt-slurm keep their cluster configuration in
[`runners/srt-slurm/<launcher>.yaml`](../runners/srt-slurm/). The native settings
(GPU count, scheduling directives, aliases, and mounts) are separate from workload recipes.
Only launchers with an existing srt-slurm path have a profile. Both B200 Nscale
srt-slurm paths share one profile, with path-specific container aliases supplied by
the launcher.

Call `write_srt_cluster_config <profile> srtslurm.yaml <uses_power>` from
[`runners/slurm_utils.sh`](../runners/slurm_utils.sh) after staging images and paths.
It writes the job-local config before `make setup`. `${NAME}` placeholders receive
explicit `--var NAME VALUE` inputs, never implicit process-environment substitution.
Optional `--model ALIAS PATH`, `--container ALIAS PATH`, and `--mount HOST CONTAINER`
arguments add or override mapping entries. Power jobs add the staged DCGM image through
the same writer. Missing variables fail before writing; values are substituted into
parsed YAML scalars so quotes and punctuation remain data, not YAML or shell syntax.

Keep model selection, cache preparation, and workload-dependent time limits in the
launcher. Do not add profiles for non-srt-slurm launchers or change their routing here.

Put per-allocation host checks and setup in
`runners/srt-slurm/hooks/<cluster>/setup.sh`, with cluster-specific helpers beside it.
The directory name matches the cluster profile's filename stem. Invoke the script
explicitly through `default_host_setup.commands` in that profile; scripts are not
auto-discovered. srt-slurm runs them on the selected allocated nodes, outside containers,
before starting services and workers. A failed check stops startup by default.
Pass configuration explicitly from the profile. If setup needs an undo step, keep it
in `teardown.sh` beside `setup.sh` and register it in `default_host_setup.teardown`.
These are job-owned hooks, not administrator-installed Slurm Prolog/Epilog scripts.
Only add hooks for clusters that need them; do not add empty scripts for every profile.

Put reusable host-check functions in `runners/srt-slurm/hooks/common.sh`; keep
cluster-only helpers beside `setup.sh`. The common file only defines functions:
sourcing it must not run checks, change environment variables, or initialize
benchmarks. Both setup hooks and benchmark scripts can reuse these functions
without importing benchmark initialization into host setup.

Hooks inject **cluster-specific host prerequisites only**, such as fabric checks or
required host-state preparation. Keep them small, workload-independent, and safe to
run repeatedly. Prefer native srt-slurm settings over shell code where possible.
Benchmark execution, model selection, engine flags, concurrency tuning, evaluation,
result collection, and job orchestration do not belong here. Do not patch engines or
containers, bypass failed checks, or hide runtime bugs with retries and ad hoc
workarounds; fix the owning component instead. Limit host changes to allocated nodes
and preserve resources used by other jobs.

## Procedure index

1. [Prepare a worktree](#prepare-a-worktree)
2. [Add a model + hardware recipe](#add-a-model--hardware-recipe)
3. [Change a master config](#change-a-master-config)
4. [Register and set up a runner](#register-and-set-up-a-runner)
5. [Register an srt-slurm recipe](#register-an-srt-slurm-recipe)
6. [Register an llm-d recipe](#register-an-llm-d-recipe)
7. [Update an image](#update-an-image)
8. [Add or change MTP](#add-or-change-mtp)
9. [Validate](#validate)
10. [Avoid schema and topology traps](#avoid-schema-and-topology-traps)
11. [Append the changelog safely](#append-the-changelog-safely)
12. [Stop conditions](#stop-conditions)

## Prepare a worktree

Source: [`docs/agent-guide.md`](./agent-guide.md), [`AGENTS.md`](../AGENTS.md).

From a clean repository root:

```bash
git status --short --branch
git fetch origin
git worktree add -b config/<slug> .worktrees/<slug> origin/main
cd .worktrees/<slug>
git status --short --branch
```

1. Confirm the path, branch, base commit, and status before editing.
2. Read `AGENTS.md`, then the closest working config, script, launcher, and recipe end to end.
3. Record the exact config key(s) the changelog and generator must select.
4. Preserve unrelated work. Do not reset, clean, rebase, or delete files you did not create.
5. Keep config work isolated until local generation succeeds. Do not consume GPU time to discover YAML or routing errors.

## Add a model + hardware recipe

Detailed source: [`.claude/commands/add-model-hardware.md`](../.claude/commands/add-model-hardware.md). Field source: [`configs/CONFIGS.md`](../configs/CONFIGS.md).
STP (Single Token Prediction) is vanilla autoregressive decoding with one token per forward pass. MTP (Multi-Token Prediction) predicts multiple tokens per forward pass through native heads or speculative decoding.

1. **Fix the identity.** Confirm the exact checkpoint ID, model prefix, precision, architecture, native context, target SKU, framework, and whether decoding is STP, native MTP, or draft-model speculation. Verify the image tag exists. Never invent one.
2. **Choose two kinds of sibling.** Read the same model on another SKU and another model on the target SKU. Read their srt-slurm recipes under [`benchmarks/single_node/srt-slurm-recipes/`](../benchmarks/single_node/srt-slurm-recipes/) and master-config entries.
3. **Add the srt-slurm recipe.** Put it at `benchmarks/single_node/srt-slurm-recipes/<model-prefix>/<engine>/<sku>-<precision>[-mtp]/8k1k.yaml`, with one `override_*` variant per matrix point. Preserve the proven sibling's engine args, env, parser flags, attention/MoE backend, KV-cache dtype, graph/eager mode, `setup_script`, and context handling.
4. **Add the master entry.** Use [`amd-master.yaml`](../configs/amd-master.yaml) for `mi*`. Otherwise, use [`nvidia-master.yaml`](../configs/nvidia-master.yaml). Set exact `image`, `model`, `model-prefix`, `runner`, `precision`, `framework`, scenarios, and supported search spaces.
5. **Size from evidence.** Mirror proven parallelism layouts and trim unsupported ones. Latency TP rows normally start at concurrency 1. Do not copy large-memory TP/EP layouts onto a smaller SKU.
6. **Check variant selection.** Every search-space row carries `srt-recipe:`, and each matrix point must match exactly one recipe variant by TP/GPU count, `CONC`, `KV_OFFLOADING` and image (`infx/srt_slurm/single_node.py::select_recipe`). No launcher routing is needed.
7. **Append one changelog entry** for the exact new key. See [Append the changelog safely](#append-the-changelog-safely).
8. **Validate syntax and generated output.** Inspect image, model, runner, ISL/OSL, `max-model-len`, concurrency, TP/PP/EP/DCP/PCP, and `spec-decoding`.

A `MODELS.md` row alone is not an executable recipe. The complete path is benchmark script + master entry + launcher routing + changelog trigger + generated matrix.

## Change a master config

Sources: [`configs/CONFIGS.md`](../configs/CONFIGS.md), [`validation.py`](../utils/matrix_logic/validation.py), [`generate_sweep_configs.py`](../utils/matrix_logic/generate_sweep_configs.py).

1. Locate the exact key and read its whole entry plus adjacent siblings.
2. Use only documented kebab-case fields. The schema forbids extras. A plausible-looking field is not accepted automatically.
3. Trace every changed field through generator output, workflow input, launcher, and benchmark script. YAML acceptance only proves shape, not runtime use.
4. Keep the correct layer authoritative:
   - master YAML: matrix identity, labels, search spaces, and emitted metadata.
   - benchmark script: serve/client behavior.
   - launcher: routing, mounts, model paths, image startup, and cluster behavior.
   - external/checked-in recipe: framework-specific multi-node runtime.
5. For topology changes, calculate GPUs before editing and compare with the target fleet.
6. For srt-slurm, update recipe and master entry together. For llm-d, update the llm-d recipe/orchestration and master entry together.
7. Append the trigger entry, generate only the affected key first, and inspect every emitted point.

Fixed-sequence `8192/1024` scenarios may set `require-power: true` to opt into validated measured power. The matrix passes this flag to standard sweeps and manual E2E throughput jobs; eval-only and AgentX rows do not inherit it. Omit the field to preserve existing behavior. Enable it only alongside the corresponding runtime and result adapter, then qualify the complete selected scope.

## Register and set up a runner

Setup source: [`utils/runner_setup/RUNNER_SETUP.md`](../utils/runner_setup/RUNNER_SETUP.md). Config source: [`configs/CONFIGS.md#runners`](../configs/CONFIGS.md#runners).

### Repository registration

1. Create `runners/launch_<base-name>.sh` for a new fleet, or update the existing launcher.
2. Add each exact registered runner name under the intended `labels:` key in [`configs/runners.yaml`](../configs/runners.yaml). New names use `<base-name>_<NN>` with zero-padded indices.
3. If generation needs fleet facts, add a matching `hardware:` entry with positive `available-cpu-dram-mib` and `gpus-per-node`.
4. Use an exact `cluster:<name>` label when facts depend on one physical fleet. Agentic configs require it.
5. Add/update master entries to use that label. Generate a targeted matrix and confirm the selected concrete names.

The runner-name prefix is load-bearing: workflow routing uses `launch_${RUNNER_NAME%%_*}.sh`. Therefore `<base-name>` must match a launcher and must not contain `_`.

### Host setup

1. Decide the runner user and shared storage. `_work` must be visible to login and compute nodes.
2. Confirm `curl`, `tar`, `tmux`, and, for Slurm, `sinfo`/`srun`/`sbatch` are on the registration shell's `PATH`.
3. Obtain repo-admin authentication and a fresh registration token. It expires after about one hour.
4. Run the documented [`setup.sh`](../utils/runner_setup/setup.sh) with token, runner URL, index range, base directory, base name, and labels.
5. Start with [`start_runners.sh`](../utils/runner_setup/start_runners.sh).
6. Verify every runner is **Idle** in [repository runner settings](https://github.com/SemiAnalysisAI/InferenceX/settings/actions/runners) before adding it to sweep traffic.
7. Verify launcher mounts for `_work`, HF cache, staged weights, and squash images from a compute node. Root containers must not leave root-owned files in the shared workspace.

The B300 DSXE Kimi-K3 AgentX path mounts its pre-staged target under
`/scratch/models` and separately exports and mounts `WRITABLE_MODELS_DIR` for
DSpark weights. Keep the draft directory on that persistent mount when reusing
the serving container; the read-only target mount cannot hold the draft.
Concurrent cells serialize draft staging with a per-model lock. Each cell lets
`hf download` validate or resume the existing cache before serving; a nonempty
directory is not a completion signal.

## Native TileRT power

TileRT's shared importer preserves Docker Hub image names and converts explicit registries such as `ghcr.io/team/image:tag` to Enroot's `docker://ghcr.io#team/image:tag` syntax. Existing `#` references are preserved. Valid cached squash images are reused without importing; a cache hit does not validate the registry import path. Invalid cached images are removed under the import lock before retrying the import.

The GLM-5.1 B200 Nscale 1k1k and 8k1k recipes select the prepared shared checkpoint, converted TileRT weights and squash cache, with allocation limits of 45 minutes for 1k1k and 90 minutes for 8k1k, including its full GSM8K eval. Since C1 is below automatic eval selection, use the PR `all-evals` label alongside `full-sweep-fail-fast` for full qualification. TileRT was added after the general GLM-5.1 retirement in [#2533](https://github.com/SemiAnalysisAI/InferenceX/pull/2533); [MODELS.md](../MODELS.md) records this retained scope. Changes still require the normal PR sweep, applicable quality evidence, sign-off and reuse before publication.

TileRT's eval wrapper calls the shared `run_eval` dispatcher without overriding its `run_lm_eval` client. It stages available artifacts after evaluation and preserves failures from either evaluation or staging. TCP readiness probes keep their socket inside a subshell and preserve the caller's diagnostic streams.

For GLM-5.1 on B200 Nscale, `MODEL_PATH` can select an existing shared checkpoint instead of the default `/scratch/models/GLM-5.1-FP8`. When it selects an HF snapshot, also set `HF_HUB_CACHE_HOST_PATH` to the existing cache root; TileRT mounts that root at the same absolute path so snapshot links to sibling blobs remain readable. Keep `TILERT_WEIGHTS_DIR` pointed at the separately converted decode weights.

Only fixed 8192/1024 `glm5.1-fp8-b200-tilert` requires native power. TileRT runs inside its returned `salloc` allocation, retains both role exit codes and drains collectors before staging audits. Exactly one physical node per role is supported. Other sequence lengths, AgentX and eval-only do not enable this collector. Hardware qualification and publication remain pending.

## Register an srt-slurm recipe

Mapping source: [`benchmarks/multi_node/srt-slurm-recipes/RECIPES.md`](../benchmarks/multi_node/srt-slurm-recipes/RECIPES.md). Checked-in recipes: [`benchmarks/multi_node/srt-slurm-recipes/`](../benchmarks/multi_node/srt-slurm-recipes/).

1. Locate the exact upstream [NVIDIA/srt-slurm](https://github.com/NVIDIA/srt-slurm) recipe and record its commit-pinned source path.
2. Stage the YAML under `benchmarks/multi_node/srt-slurm-recipes/<model-prefix>/<engine>/<gpu>-<precision>/<workload>/`, following the naming rules in `RECIPES.md`. Read the closest sibling and selected cluster launcher.
3. Map source fields to the master search-space entry: resource worker counts → `num-worker`, TP/EP/DP-attention → worker topology, benchmark concurrencies → `conc-list`, and recipe path → `additional-settings: ["CONFIG_FILE=..."]`.
4. Add/update the matching [`nvidia-master.yaml`](../configs/nvidia-master.yaml) entry in the same change. Keep worker counts, TP/PP/EP/DCP/PCP, hardware, router, transfer engine, and concurrency labels synchronized.
5. For an image bump, make recipe `model.container` exactly equal master `image`. The launcher uses the master image as the container-alias key.
6. Run the recipe's documented `srtctl` validation, then generate the master key and compare every frontend label/topology field with the recipe.
7. Append the changelog entry.

Do not ship one side alone. `srtctl` reads the recipe, while matrix generation reads the master config. Recipe-only changes can mislabel results. Master-only changes do not alter the deployed recipe.

## Register an llm-d recipe

Sources: [`benchmarks/llm-d/README.md`](../benchmarks/llm-d/README.md), [`benchmarks/multi_node/llm-d/README.md`](../benchmarks/multi_node/llm-d/README.md), and [`llm-d-recipes/`](../benchmarks/multi_node/llm-d-recipes/).

llm-d is not the srt-slurm path: InferenceX owns the Slurm allocation and starts one container per node.

1. Copy the nearest YAML under [`benchmarks/multi_node/llm-d-recipes/`](../benchmarks/multi_node/llm-d-recipes/) and set EPP plugins/scheduling, role-specific `extra-args`/`env`, and optional `slurm.time_limit`.
2. Add/update the `llmd-vllm` master entry. Set `multinode: true`, `disagg: true`, router metadata, `kv-p2p-transfer`, prefill/decode worker topology, concurrency, and `CONFIG_FILE=<basename>.yaml` in `additional-settings`.
3. Keep `PREFILL_NODES`, `DECODE_NODES`, `GPUS_PER_NODE`, and worker counts consistent with the allocation and with each role's DP/TP/EP layout.
4. Confirm [`submit.sh`](../benchmarks/multi_node/llm-d/submit.sh) → [`job.slurm`](../benchmarks/multi_node/llm-d/job.slurm) → [`server.sh`](../benchmarks/multi_node/llm-d/server.sh) propagation and the selected wrapper/launcher route.
5. Verify file discovery. The decode leader creates `/tmp/endpoints.yaml`. Prefill endpoints use vLLM port 8200, while decode endpoints use sidecar port 8000. Names must be unique, addresses must be literal IPv4, and ports must be strings in `1..65535`.
6. Confirm EPP loads discovery before Envoy receives traffic and role labels select the proper prefill/decode backends.
7. Generate the key, inspect topology and `additional-settings`, then append the changelog.

A missing/unset `CONFIG_FILE` silently selects the image's `/etc/epp/config.yaml` fallback and removes recipe-specific vLLM flags. Treat that as a validation failure unless fallback is explicitly intended.

## Update an image

Sources: [`AGENTS.md#non-negotiable-benchmark-invariants`](../AGENTS.md#non-negotiable-benchmark-invariants), the matching master configs, runtime scripts, and checked-in recipes.

1. Verify the exact upstream registry tag or digest exists and is appropriate for CUDA/ROCm and the target architecture.
2. Find every affected config key, runtime script, Dockerfile, and checked-in recipe. Do not assume the master YAML is the only image reference.
3. Update the master `image` and any required env vars, flags, package versions, or patches as one coherent change.
4. For srt-slurm, update `model.container` and keep it identical to master `image`.
5. For llm-d, distinguish the serving image selected by the master config from the build source in [`benchmarks/llm-d/Dockerfile`](../benchmarks/llm-d/Dockerfile). Update both only when the build contract changes.
6. Append a changelog entry selecting all affected keys (wildcards are allowed when intentional), including old/new versions and material runtime changes.
7. Generate each affected family and verify no stale tag survives in its runtime path.

## Add or change MTP

Sources: [`AGENTS.md#non-negotiable-benchmark-invariants`](../AGENTS.md#non-negotiable-benchmark-invariants), [MTP appendix in the model+hardware playbook](../.claude/commands/add-model-hardware.md#appendix--mtp--eagle3-spec-decoding-variant), and current [`*-mtp` srt-slurm recipes](../benchmarks/single_node/srt-slurm-recipes/).

1. Confirm native MTP modules versus an external draft. For a draft, verify exact model ID, method (for example `eagle3`), and recommended speculative-token count from the model/upstream recipe.
2. Copy a working sibling for the same model and backend. Preserve its speculative config, attention backend, token count, model patches, and dependency setup.
3. Every `*_mtp.sh` must pass `--use-chat-template` to `run_benchmark_serving`. Raw prompts silently depress acceptance.
4. Size graph capture for at least `CONC * (1 + NUM_SPEC_TOKENS)`, rounded as the sibling does and capped at the framework limit (the current vLLM playbook caps at 2048).
5. Keep backend differences: do not copy CUDA-only drafter attention pins or patches into ROCm recipes.
6. Set `spec-decoding: mtp` in the relevant search-space entries and add `_mtp` launcher suffix routing. For a draft-model mode supported by the schema, use the matching generated value deliberately. Do not infer it from a filename.
7. Add script + master entry + launcher routing + changelog together.
8. Run Bash syntax and generation checks. Inspect `spec-decoding`, draft/native method, token count, chat-template use, capture range, and resolved script.

### DeepSeek-V4-Pro-0813 DSpark on MI355X ATOM

`dsv4-fp4-mi355x-atom-agentic-mtp` keeps its historical key and `_mtp.sh`
filename, while its matrix uses `spec-decoding: draft_model`. The AMD launcher
routes both speculative metadata values to that script and mounts the shared
HF cache for the 0813 checkpoint. The recipe pins revision
`72e1d3230f6c080a530b0a1d46f8eb4602340597` and serves the resolved snapshot path;
an explicit `MODEL_PATH` must pass the same checkpoint checks. Before GPU
startup it verifies the config/index hashes, DSpark Markov/confidence heads,
all 66 shard headers and payload boundaries, and offline tokenizer loading.
This checks readability and completeness, not full weight-file hashes.

All ten AgentX throughput points use DSpark K6 (target verification length 7)
and the committed golden AL 3.77. C1/2/4/8/16 use TP8/EP1;
C48/64/96/128/256 use TP8/DPA8/EP8 with native RCCL. Each point runs for
3600 seconds. The C256 full GSM8K eval omits forced acceptance. Keep the
pinned `rocm/atom-dev:nightly_202609161445` image and GPU-only KV. C1 through C16 use
BF16 KV, while C48 and above retain FP8 KV; all points use the FP4 index cache,
8192-token checkpoints and DEP dense FULL graph ladder. Fixed q7 graphs are
captured in each new server; confirm target and DSpark draft capture in
`server.log`. Confidence schedules and ragged verification remain disabled.

`AGENTIC_TOKENIZER_PATH` optionally overrides AgentX's tokenizer source; its
default remains `MODEL`. This recipe sets it to the validated server snapshot.
`checkpoint_preflight.json`, `runtime_manifest.json` and `server_command.txt`
record model/source identity and requested settings. Successful startup,
graph capture and requests require runtime log evidence.

The pinned image is the official ATOM nightly
`rocm/atom-dev:nightly_202609161445`, which includes the merged
[ROCm/ATOM#2233](https://github.com/ROCm/ATOM/pull/2233) inference-mode fix.
The recipe does not patch AITER source at runtime; TP communication
fusion, DSpark K6 and graph capture use the implementation shipped in the image.

### MiniMax-M3 ATOM FlyDSL paged decode

`minimaxm3-fp4-mi355x-atom-agentic-mtp` uses
`rocm/atom-dev:nightly_202609231248` with `ATOM_PA_FLYDSL=1` and
`ATOM_PA_FLYDSL_PLAN=1`, following [ROCm/ATOM#2366](https://github.com/ROCm/ATOM/pull/2366)
and the [upstream recipe](https://github.com/ROCm/ATOM/blob/94cde4ba786f45b38c26ee8201444659e44f861f/recipes/MiniMax-M3-Agentic-InferenceX.md).
FlyDSL handles supported paged-decode shapes; its work planner balances dense
decode by actual context length. Unsupported shapes retain the Gluon fallback.
Verify the selected route and capture-time work-plan creation in `server.log`.

The change is limited to the image and the two FlyDSL variables in
`benchmarks/single_node/srt-slurm-recipes/minimaxm3/atom/mi355x-fp4-mtp/agentic.yaml`;
TP4 C32 is dropped; the TP4 C1-C28 and TP2 C1-C2 points, EAGLE3 K3, golden AL 2.78 and indexer CP
are unchanged.

### DeepSeek-V4.1-Flash DSpark

The GB200 DSpark recipe uses a minimum CUDA graph capture size of 64 tokens to cover concurrent AgentX subagents. This raises c1/c2/c4 from 8/16/32 to 64; c8 and above retain their existing sizes. The full trace, AL 3.51, and Engram UVA settings are preserved; low-concurrency tail latency improvements require CI confirmation.
The B200 DSpark recipe uses the same minimum capture size and preserves the same workload settings.
The GB300 DSpark recipe uses the same minimum capture size and preserves the same workload settings.
The H200 DSpark recipe uses the same minimum capture size and preserves the same workload settings.

B300 uses the same minimum capture size at c1/c2/c4. Its c1 CI comparison reduced request ITL P90/P99 from 38.74/41.42 ms to 2.62/3.45 ms; c2/c4 require CI confirmation.

The AgentX-only `dsv41flash-fp4-<sku>-vllm-agentic-dspark` recipes use
`vllm/vllm-openai:deepseekv41-flash-0909` at TP4 on Blackwell SKUs with native five-token DSpark,
probabilistic drafting. Throughput uses the [committed golden AL](../golden_al_distribution/dsv41flash_dspark.yaml) of 3.51 for thinking on and five draft tokens, with synthetic rejection sampling and adaptive verification disabled. Accuracy evals retain real block rejection and adaptive verification. `--engram-config '{"cpu_offload":true}'`
stores Engram embedding tables in pinned host DRAM accessed through UVA;
`kv-offloading: none` describes the separate, GPU-resident KV cache. MXFP4 expert
weights determine the recipe's `precision: fp4` label.

The GPU-specific entry points share the text-only serving behavior, `deepseek_v41` tokenizer and
parsers, 1M context, and the shared AgentX trace replay, power, metrics, and eval
helpers. The TP4 concurrency range is 1–128. The shared script sizes graph capture
for the six-token DSpark verification block. The launchers mount the repository at `/ix` for this recipe so
AgentX runtime directories are not created under `/workspace`. Launcher-specific model paths and persistent caches are reused.
The recipe probes the serving port on the compute node and selects an available
port if the preferred one is occupied. Serving, replay, metrics, and eval share
that endpoint.

The B300 entry also includes a TP2 variant at concurrency 2–128. Its dedicated
script uses `FULL_AND_PIECEWISE` CUDA graphs with explicit capture-size sets ending
at 2046 or 8190 tokens. It sets `--max-num-batched-tokens` to 2048 for concurrency
1–4 and TP2 concurrency 128, and to 8192 otherwise; `--max-num-seqs` is 256. The
TP2 concurrency-128 variant also sets `--gpu-memory-utilization 0.97`. Other SKUs
continue to use the shared script.

The GB300 launcher allows 7200 seconds for engine readiness. In [run 34504969146](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34504969146), the Rust frontend exhausted its 3600-second deadline while the engine was still capturing graphs; model loading alone took 18–23 minutes. This extends startup time without changing the benchmark duration or decoding settings.

GPU sweep and eval evidence is required before calling any recipe validated.

Source: [upstream recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash).

### DeepSeek-V4.1-Flash DSpark on ATOM

`dsv41flash-fp4-mi355x-atom-agentic-dspark` follows the
[upstream ATOM recipe](https://github.com/ROCm/ATOM/blob/53b11c9a665e786798785acbedfdfd4da3fb87c4/recipes/DeepSeek-V4.1-Flash-Agentic.md)
with `rocm/atom-dev:nightly_202609250902`. TP2 covers concurrency
`[1, 2, 8, 16, 32, 64]`; TP4 covers `[2, 8, 16, 32, 64]`, without expert
parallelism or KV offload. Every point uses BF16 KV, FP8 index cache, 128 maximum
sequences, 16K batched-token/prefill chunks, prefix caching with block size 16,
8K state checkpoints, compilation level 3 and FULL graphs. Concurrency 32 captures
every size from 1 through 32 plus 48, 64 and 128; other points use the upstream
sparse list. Five-token DSpark uses golden AL 3.51 for throughput and real
acceptance for eval, with the checkpoint's shipped draft and `dsml_v41` parser.

The existing MI355X launcher mounts this model's shared cache and the repository
at `/ix`, preserves Slurm's GPU allocation and routes `draft_model` to the new
`dsv41flash_fp4_mi355x_atom_mtp.sh` script. Canonical AgentX runs use the uncapped
`semianalysis_cc_traces_weka_062126` corpus, 3600 seconds per point and five warmup
requests per lane. Workflow duration overrides and `agentx-fast` remain available
for diagnostics. GPU sweep and eval validation is pending.

### DeepSeek-V4.1-Flash DSpark on H200

`dsv41flash-fp4-h200-vllm-agentic-dspark` is the H200 AgentX arm of the
DeepSeek-V4.1-Flash recipe. It shares `vllm/vllm-openai:deepseekv41-flash-0909` and the
text-only serving script with the Blackwell arms: `deepseek_v41` tokenizer and parsers,
1M context, native five-token DSpark with probabilistic drafting. Throughput uses the [committed golden AL](../golden_al_distribution/dsv41flash_dspark.yaml) of 3.51 for thinking on and five draft tokens, with synthetic rejection sampling and adaptive verification disabled. Accuracy evals retain real block rejection and adaptive verification.

The arm runs **TP8**, not the upstream TP4. Upstream verifies TP4 on one GB200 NVL4 tray
and states that the same layout becomes TP8 per role on 8-GPU nodes, which is what an
H200 DGXC node is.

`precision: fp4` labels the checkpoint's MXFP4 routed expert weights, matching the
Blackwell and MI355X arms on the identical checkpoint. Hopper has no FP4 tensor cores, so
those weights run through the upconverting MoE path; the label describes the checkpoint,
not the SKU's native arithmetic.

`--engram-config '{"cpu_offload":true}'` keeps the Engram tables in pinned host DRAM
reached through UVA, and `kv-offloading: none` describes the separate, GPU-resident KV
cache. Measured on the cluster, the offload moves 11.80 GiB per rank per table for two
tables across 8 ranks — 188.8 GiB — leaving roughly 35.9 GiB per GPU of resident weights
out of 141 GiB.

Trace corpus: the arm replays the uncapped `semianalysis_cc_traces_weka_062126` corpus,
not the 256k-capped `..._062126_256k` variant, because the model serves 1M context. The
recipe never names a corpus — `resolve_trace_source` picks the uncapped default only
because its `dsv4*` case arm also matches the `dsv41flash` prefix. That is load-bearing
and invisible at the call site, so `runners/test_dsv41flash_h200.py` pins it; narrowing
the arm would silently downgrade this recipe's traces.

**The H100 arm is separate.** H100 is not in the upstream hardware table, and the
blocker is not the weights. At 1M context the sparse attention indexer allocates a
`[max-num-batched-tokens, max-model-len]` logits buffer in
`fp8_fp4_paged_mqa_logits`, which at the default 8192 batched tokens is exactly 16 GiB.
That is a fixed startup cost paid during memory profiling, independent of concurrency, so
it fails at concurrency 1 on an 80 GB card even though the resident weights fit. The H100
arm therefore ships its own script with capped batched tokens instead of the shared
symlink; see the H100 section below.

The launcher mounts the repository at `/ix` for this recipe so AgentX runtime directories
are not created under `/workspace`, and it already mounts the shared HF cache, so the
script resolves the model through `HF_HUB_CACHE` rather than a per-node path. The recipe
probes the serving port on the compute node and selects an available one if the preferred
port is occupied; serving, replay, metrics, and eval share that endpoint.

### DeepSeek-V4.1-Flash DSpark on H100

Throughput uses the [committed golden AL](../golden_al_distribution/dsv41flash_dspark.yaml) of 3.51 for thinking on and five draft tokens, with synthetic rejection sampling and adaptive verification disabled. Accuracy evals retain real block rejection and adaptive verification.

`dsv41flash-fp4-h100-vllm-agentic-dspark` is the H100 AgentX arm of the
DeepSeek-V4.1-Flash recipe, added after the H200 arm and deliberately separate from
it. H100 is **not** in the upstream hardware table, which lists h200, gb200, gb300, and
mi350x.

Unlike the other SKUs, H100 does not use the shared `dsv41flash_fp4_vllm_mtp.sh`. It has
its own copy, because the shared flags cannot serve 1M context on an 80 GB card. At 1M
context the sparse attention indexer allocates a
`[max-num-batched-tokens, max-model-len]` logits buffer in `fp8_fp4_paged_mqa_logits`:
at the shared script's effective 8192 batched tokens that is 8192 x 1048576 x 2 bytes,
exactly 16.00 GiB. It is a fixed cost paid during startup memory profiling, independent
of concurrency, so it OOMed at concurrency 1 in run
[34467029236](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34467029236)
next to roughly 35.9 GiB per GPU of resident weights — trimming the concurrency list
cannot help.

The H100 script therefore caps `--max-num-batched-tokens` at 4096, putting the indexer
buffer at 8 GiB. Capping `--max-model-len` instead would shrink it just as well, but a
context cap forces the 256k-capped trace corpus onto a model that serves 1M, so batched
tokens is the right lever. The script also sets `--max-num-seqs` to twice the trajectory
concurrency rather than inheriting vLLM's default of 1024, sets
`--gpu-memory-utilization 0.92`, and enables `expandable_segments` because the failing
allocation left 1.04 GiB reserved but unallocated.

Those caps were validated with a single concurrency-1 `agentx-fast` run
([34485694183](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34485694183))
before any sweep was dispatched, which is the right order here: a full sweep that OOMs at
startup wastes every leg. That run came up healthy and reported the budget the
concurrency list is now derived from:

```
Available KV cache memory: 13.47 GiB
GPU KV cache size: 7,022,899 tokens
Maximum concurrency for 1,048,576 tokens per request: 6.70x
```

The original arm swept concurrency 1–4 under that 6.70x full-context estimate. The
follow-up sweep extends the same recipe to concurrency 8 and 16 to measure the real
AgentX saturation curve; these points may preempt if several trajectories approach 1M
tokens simultaneously. Buying more KV means
shrinking the indexer further — `--max-num-batched-tokens 2048` would free about 4 GiB
more — at the cost of chunking long-trace prefill harder. That trade is worth revisiting
once there is throughput data across the range.

`runners/launch_h100-dgxc-slurm.sh` previously resolved only the untagged
`_h100[_mtp].sh` script name, so no framework-tagged script could run on this cluster at
all. It now prefers `_h100_<framework>[_mtp].sh` first, as the h200 launchers have since
#392, and falls back to the untagged name for the recipes that predate framework tags. It
also mounts the repository at `/ix` for this recipe so AgentX runtime directories are not
created under `/workspace`.

Source: [upstream recipe](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml).

### DeepSeek-V4.1-Flash DSpark on SGLang

The H100 SGLang candidate sweeps DSpark at concurrency 1/2/4/8/16/20. It retains 8 SWA prefix tails per concurrency at C1/C2 and 32 at C4 and above. A matched one-hour comparison rejected a blanket 128-tail floor: C2 throughput improved only 1.7% while interactivity fell 44.5%. Completed STP comparisons did not contribute a measured frontier point, so STP is excluded from the selected sweep. The recipe interleaves 16 decode steps between prefill chunks, preserving trace content and context limits.

The same sweep also qualifies supported TP8/EP8/DP8 attention at C4/C8/C16/C20. DP uses a stock consistent-hash router with stable session keys, DP LM-head execution, and 64 SWA prefix tails per rank. Full C16 GSM8K passed on all 1,319 examples; its performance contribution remains under measurement. The native 1M context and the AgentX subagent/session semantics are preserved.

The nightly candidate uses `nightly-dev-cu13-20260922-582389ce`, native MXFP4 Marlin MoE, and `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank`. A resolved local snapshot lets the upstream allocator evict checkpoint file cache before allocating anonymous host tables. Draft precision follows the pinned image's default handling, including its WO_A FP8-to-BF16 conversion. GPU-specific block32 FP8 launch configurations use the upstream kernel and supported `SPLIT_K`/`SWAP_AB` options; checkpoint data, scales, output dtype and context limits remain unchanged. Small-batch configurations require FP32-reference and CUDA-graph validation at selection boundaries, followed by full-model accuracy and serving measurements. Larger batches retain their previous configurations. Full canonical qualification is still required.

`dsv41flash-fp4-<sku>-sglang-agentic-dspark` are the SGLang counterparts of the vLLM
arms, one PR per SKU across h100, h200, b200, b300, gb200, gb300 and mi355x. They follow the
[SGLang cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1),
which has no released SGLang version for this model yet. B200 pins the CUDA 13 nightly
`lmsysorg/sglang:nightly-dev-cu13-20260922-4cbf290f` by digest; the other NVIDIA arms use
`lmsysorg/sglang:dev-dsv41` and MI355X uses `lmsysorg/sglang:dev-dsv41-mi35x`.

B200 uses shipped-default DSpark across TP4/EP4 C1–128 and TP2/EP2 C1–8.
Engram stays in host DRAM with `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank`.
Shared host tables had zero huge-page backing in
[run 35626514270](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35626514270);
per-rank anonymous shards request huge pages without changing host sysctls.
Verify the actual backing percentage in each rank's startup log.

| B200 topology | Static memory fraction | Prefill chunk | SWA prefix tails |
| --- | ---: | ---: | --- |
| TP4/EP4, C1–128 | 0.80 | 4096 | `max(128, min(4096, 64 * CONC))` |
| TP2/EP2, C1–8 | 0.92 | 2048 | `128 * CONC` |

Both use 16 decode rounds between prefill chunks and cap running requests at
`min(2 * CONC, 64)`. Chunked-prefix caching retains SWA tails separately from full
KV, so low full-cache occupancy does not prove that reusable SWA capacity is
available. The concurrency-scaled tail budget shares the fixed static pool with
full KV. Validate actual cache sizes, transient memory, cache reuse, and the
throughput/interactivity frontier in the canonical sweep.

TP2 loads about 147.76 GiB of target and draft weights per GPU. The recipe
verifies the pinned stock loader's hash and enables PyTorch expandable allocator
segments without applying an engine patch. The September 22 nightly completed
startup and all 1,319 GSM8K examples at C8 (97.65% strict accuracy) in an isolated
Slurm diagnostic. Its post-eval packaging was recovered separately after a missing
wrapper variable; this is not a green official workflow. The full latest-image
sweep remains required. Draft precision remains upstream default.
The B200 launcher also converts pinned Docker digests to the installed Enroot
manifest-reference syntax and stops immediately on import failure.

DSpark uses the default precision shipped by the pinned official nightly, without custom draft quantization or precision patches. STP loads no draft; full accuracy and performance validation are still required.

GB200 pins official CUDA 13 nightly `20260923-06008c17` to manifest `sha256:5921361fcf358cdde4df1968c941c14157f418613b099ad7f3e5aeed6427ae15` (ARM64 `sha256:d49261d2edd82fed2dd6254c33e68871ccf7a399498e059ec91dc4453a5808c3`). It matches the published vLLM TP2/EP1 and TP4/EP1 grids at C1/2/4/8/16/32/64/128, with no DP attention. The earlier staged TP4/EP4 sweep is historical evidence, not qualification of these topologies.

The GB200 sweep contains only `dsv41flash-fp4-gb200-sglang-agentic-dspark`.
The unmeasured STP entry is excluded; adding it would require matched evidence
of a performance-frontier contribution. DSpark uses the default precision shipped
by the pinned official nightly, without custom draft quantization or precision
patches. Full accuracy and performance validation remain required.

GB200 TP4 reserves `min(64*CONC, 1024)` SWA prefix tails while retaining static memory
0.70 and chunk size 4096. The earlier TP4/EP4 C16 reserve left a measured 27.0M full-context
KV slots and 439,040 SWA slots; the cap avoids exhausting the measured 51.82 GiB
KV budget at high concurrency. Only TP4 C16 uses prefill/decode interval 16:
its canonical comparison improved p90 interactivity 13.65% for 0.30% lower
throughput, with p90 TTFT increasing from 2.35 to 3.51 seconds. Full GSM8K
passed all 1,319 samples. Other concurrency points still require the full sweep;
these C16 results do not establish a benefit at every concurrency.

GB200 TP2 uses static memory fraction 0.92, a 2048-token prefill chunk, `min(128*CONC,1024)` SWA tails, prefill/decode interval 16 and graph/running capacity bounded to 16 requests. These supported limits follow the completed B200 EP1 memory qualification; GB200 must independently pass loading, graph capture, full-context pool checks and every performance/evaluation cell. Expandable CUDA allocator segments reduce fragmentation without changing weights or precision. C64/C128 performance receives the partition maximum 12-hour allocation plus 30 minutes for workflow packaging; full warmup, the 3600-second scoring window and uncapped 1,319-question GSM8K remain unchanged.

The GB200 host-table layout is `per_rank`: its compute-node kernel enables
anonymous huge pages through `madvise`, while `shmem_enabled=never` prevents huge
pages for the shared memfd layout. Upstream allocates row shards in anonymous host
memory and requests 512 MiB huge pages with `MADV_HUGEPAGE`/`MADV_COLLAPSE`.
It preserves the original FP8 table weights and restores the two TP all-reduces;
inspect startup's actual resident/huge-page counts before claiming a benefit.

DSpark is the checkpoint's own bundled draft. SGLang exposes no EAGLE or MTP path and no
`--speculative-num-steps` knob for it; the recipes pass `--speculative-algorithm DSPARK
--speculative-dspark-block-size 5`. Throughput uses the same
[committed golden AL](../golden_al_distribution/dsv41flash_dspark.yaml) of 3.51 for thinking
on and five draft tokens through `SGLANG_SIMULATE_ACC_LEN` with `match-expected` and
`real-draft-token`; accuracy evals keep real verification. Thinking is off by default in
SGLang for this model, so the scripts set `SGLANG_DEFAULT_THINKING=1` and
`SGLANG_DSV41_REASONING_EFFORT=high` to measure the thinking-on regime the golden AL was
collected in.

Parallelism follows the verified cookbook cells: TP4/EP4 on Blackwell and MI355X, TP8/EP8 on
Hopper. The cookbook resolves the attention, MoE and FP8 GEMM backends automatically and
warns that overriding them falls back to the slow Triton block-FP8 matmul; the one exception
is its H200 cell, which pins `--attention-backend dsv4 --moe-runner-backend flashinfer_mxfp4`,
so the Hopper arms do the same. `--mem-fraction-static 0.8` is the cookbook's low-latency
setting. `--max-running-requests` is `2 * CONC` for AgentX subagent fan-out and the decode
graph batch covers it, floored at the cookbook's 64 and capped at 128.

Each SKU ships its own `dsv41flash_fp4_<sku>_sglang_mtp.sh` in its own PR. H100 is not in
the cookbook's hardware table, so its script differs: the Engram tables move to a single shared host copy
(`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`, the SGLang analogue of the vLLM arm's Engram
CPU offload) and the prefill chunk is capped at 4096, the same batched-token cap the vLLM
H100 arm needed for the sparse-attention indexer buffer on an 80 GB card. Concurrency
stops at 8 there until the KV ceiling is measured. MI355X has its own script with the
cookbook's ROCm environment (`SGLANG_USE_AITER=1`, `SGLANG_MOE_PADDING=1`,
`AITER_FLYDSL_FORCE_REDUCE=1`, `ROCM_QUICK_REDUCE_QUANTIZATION=NONE`),
`--disable-radix-cache`, and breakable prefill graphs capped at 4096 tokens.

The KV cache is GPU-resident on every arm, so `kv-offloading: none`. The launchers route
`dsv41flash` for `framework: sglang` the same way as for vLLM: the repository is mounted at
`/ix`, and the checkpoint resolves through each cluster's persistent HF cache (the writable
Lustre models directory on b300). `runners/launch_b200-nscale-compat.sh`,
`launch_b300-dsxe.sh`, `launch_gb200-nv.sh` and `launch_gb300-nv.sh` previously gated
those paths on `vllm` only.

GPU sweep and eval evidence is required before calling any of these arms validated.

## Validate

Run the smallest checks that cover the edited layers.

### YAML parse

```bash
python3 -c "import yaml; yaml.safe_load(open('configs/<nvidia|amd>-master.yaml')); yaml.safe_load(open('configs/runners.yaml')); yaml.safe_load(open('perf-changelog.yaml'))"
```

### Benchmark and launcher syntax

```bash
bash -n benchmarks/<path>/<script>.sh
bash -n runners/launch_<cluster>.sh
```

### Exact-key schema + matrix generation

```bash
uv run --no-project --exclude-newer PT12H --python 3.12 --with pydantic --with pyyaml \
  python -m infx.matrix.generate test-config \
  --config-files configs/<nvidia|amd>-master.yaml \
  --runner-config configs/runners.yaml \
  --config-keys <exact-key>
```

### Filtered family generation

```bash
uv run --no-project --exclude-newer PT12H --python 3.12 --with pydantic --with pyyaml \
  python -m infx.matrix.generate full-sweep \
  --config-files configs/<nvidia|amd>-master.yaml \
  --runner-config configs/runners.yaml \
  --model-prefix <prefix> \
  --framework <framework> \
  --precision <precision> \
  --runner-type <runner> \
  --seq-lens 8k1k
```

Use `--seq-lens 1k1k` only when explicitly selecting the retained `glm5.1-fp8-b200-tilert` configuration; other 1k1k coverage is retired.

Inspect, do not merely count, the emitted `model`, `image`, `runner`, scenario, concurrency, `max-model-len`, TP/PP/EP/DCP/PCP, prefill/decode worker blocks, hardware, router, KV transfer, eval flags, `additional-settings`, and `spec-decoding`.

If schema or generator behavior changed, run its focused suite:

```bash
python -m pytest utils/matrix_logic/ -v
```

For srt-slurm, also run the upstream recipe checker/`srtctl` command documented for that recipe. For llm-d, validate recipe YAML and exercise the allocation/discovery path on the intended Slurm fleet. Local matrix generation cannot prove endpoint discovery.

## Avoid schema and topology traps

Enforced details come from [`validation.py`](../utils/matrix_logic/validation.py) and are summarized in [`configs/CONFIGS.md`](../configs/CONFIGS.md):

- Schemas use `extra='forbid'`. Use kebab-case aliases exactly.
- Choose either `conc-start` + `conc-end` **or** non-empty `conc-list`, never both. Values must be positive and start must not exceed end.
- `pp`, `dcp-size`, and `pcp-size` are positive integers. `dcp-size` must divide `tp`.
- Per-worker GPU demand is `num-worker * tp * pp * pcp-size`. DCP reuses TP GPUs and does not multiply allocation.
- Single-node topology fields live in the search-space entry. Multi-node fields live independently under `prefill` and `decode`.
- Heterogeneous `hardware` must appear on both worker blocks or neither. It records result metadata and does not schedule runners.
- `disagg: true` requires `multinode: true` and `kv-p2p-transfer` at top level or on every search-space entry.
- Declare `router` and `kv-p2p-transfer` at exactly one scope: top level or search-space, not both.
- Router metadata requires its component's real name and release/package/commit version. An image tag is not a component version.
- Agentic configs require an exact `cluster:<name>` runner.
- Setting a field only emits an env/workflow value. Confirm the selected script consumes it.
- Scenario `max-model-len` is derived from ISL + OSL + slack. Do not hardcode the checkpoint's full context for an 8k1k recipe.

## Append the changelog safely

Sources: [`AGENTS.md#non-negotiable-benchmark-invariants`](../AGENTS.md#non-negotiable-benchmark-invariants), [`perf-changelog.yaml`](../perf-changelog.yaml).

1. Make all executable config changes first and identify the exact keys.
2. Append a new block at the physical end of `perf-changelog.yaml`:

```yaml
- config-keys:
    - <exact-key-or-intentional-wildcard>
  description:
    - "What changed"
    - "Image/topology/runtime detail"
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/<number>
```

3. Before the PR exists, the model+hardware playbook permits `pr-link: TBD`. Replace it with the real URL immediately after creating the PR.
4. Never prepend, insert chronologically, sort, reformat, or run a formatter over the file.
5. Never delete or normalize existing whitespace, including trailing spaces on blank separators. CI depends on historical bytes.
6. If the file conflicts with `main`, restore the current `main` version and re-append only this branch's entries. Do not hand-merge reordered history.
7. Parse the file and confirm the generated changelog selection includes the intended keys before requesting a sweep.

## Stop conditions

Stop before dispatching GPU work or claiming the configuration complete when any condition below holds. Obtain the missing fact or fix the source mismatch. Do not guess.

- Exact checkpoint, precision, architecture, native context, framework, draft model/method, or image tag is unverified.
- No proven sibling covers the target model/backend/SKU, and required runtime flags or memory limits remain unknown.
- Runner user, shared mounts, staged model path, GPU count, host DRAM, Slurm behavior, or root-file cleanup is unknown. Runner registration credentials are also a hard prerequisite for host setup.
- The registered runner prefix has no matching launcher, a matrix resolves to a nonexistent script, or the runner is not **Idle**.
- Calculated topology exceeds the fleet, DCP does not divide TP, heterogeneous hardware metadata is one-sided, or generated topology differs from the intended recipe.
- An srt-slurm recipe and master entry disagree, `model.container != image`, or upstream recipe validation has not run.
- An llm-d recipe is missing and would fall back unintentionally, allocation counts disagree, or endpoint discovery cannot satisfy literal-IPv4/unique-name/valid-port rules.
- An MTP script lacks chat-template benchmarking, the speculative method/token count is unverified, or graph capture exceeds the backend limit.
- The changelog change would modify historical bytes, is not at EOF, has a conflict, or still has `TBD` when the PR is otherwise ready for sweep.
- YAML, Bash, strict schema, exact-key generation, launcher simulation, or recipe validation fails.

A configuration is ready for sweep only when the executable files agree, the exact key generates, the runtime route exists, the changelog selects it, and all layer-specific checks above pass.

## DeepSeek-V4.1-Flash on MI355X

The `dsv41flash-fp4-mi355x-vllm-agentic-dspark` recipe extends [#2958](https://github.com/SemiAnalysisAI/InferenceX/pull/2958) to MI355X AgentX: TP4 and TP2, concurrency 1–128, native five-token DSpark. Throughput uses the [committed golden AL](../golden_al_distribution/dsv41flash_dspark.yaml) of 3.51 for thinking on and five draft tokens, with synthetic rejection sampling and adaptive verification disabled. Accuracy evals retain real block rejection but, unlike the CUDA arms, also keep adaptive verification disabled: it trims verification requests on device, which the ROCm `DeepseekV4IndexerBackend` does not support, and the engine refused to start with it enabled ([run 34651830283](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34651830283)). FP4 describes the MXFP4 experts; the checkpoint also contains MXFP8 weights.

Follow the AMD overrides in the merged [upstream recipe #968](https://github.com/vllm-project/recipes/pull/968): `VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_USE_AITER_MOE=1`, and `--moe-backend aiter`. The generic AITER selector lets vLLM pick the CK a8w4 experts, matching the DSV4-Pro MI355X recipe. The recipe pins `semianalysis_cc_traces_weka_062126` (the unfiltered corpus) via `WEKA_LOADER_OVERRIDE`. KV stays GPU-resident. Engram stayed on GPU under the upstream AMD defaults until [vllm-project/vllm#57491](https://github.com/vllm-project/vllm/pull/57491) widened the two `is_cuda()` gates to `is_cuda_alike()`. From that commit on, ROCm resolves an `EngramConfig` and `cpu_offload` defaults to on through `VLLM_PLE_CPU_OFFLOAD`, so the recipe sets `--engram-config` explicitly rather than leaning on that default. TP=2 always offloads, since the tables need 94.4 GiB per rank there; TP=4 keeps them resident through concurrency 64, where the KV pool is not the constraint, and offloads only at 128. The recipe likewise trims `--max-num-batched-tokens` only above concurrency 32, to 8192 at TP=2 c64 and TP=4 c128 and to 4096 at TP=2 c128, because the sparse-attention indexer and its companion per-rank buffers grow at roughly 4.4 MiB per batched token. Where that chunk falls below six times the API-server default of 1024 sequences, `--max-num-seqs` is capped at the graph-capture shape: DSpark verifies 1+5 tokens per sequence, and at 4096 against 1024 sequences the engram projection faults during profiling. The rule in every case is to spend device memory on KV only at the concurrencies that ran short of it, leaving the validated low-concurrency settings alone. Images built before that merge still reject the option on ROCm. The MI355X launcher uses the shared HF cache and mounts this model's repository at `/ix`, and exports `INFMAX_CONTAINER_WORKSPACE=/ix` so AgentX dependencies and outputs resolve inside that mount.

**GPU validation:** The recipe uses `vllm/vllm-openai-rocm:nightly-rocm100-3df4ae153eb385e27b52f26c81f8edb9e20b9984` (digest `sha256:eccb72b7…`, published 2026-09-21) on the ROCm 10.0 nightly channel that `kimik3-fp4-mi355x-vllm-agentic-mtp` already runs on this cluster. The sweep in [#3326](https://github.com/SemiAnalysisAI/InferenceX/pull/3326) qualifies that pin across TP4 and TP2 at concurrency 1–128, and is the only evidence for it: [run 34710937012](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34710937012) covered TP4 concurrency 1–32 plus eval-only concurrency 32 on the superseded `nightly-eed1f3d0c6043bd494424a22443ee198dd56f657`, so its points do not carry onto this image. The merged [upstream recipe #1006](https://github.com/vllm-project/recipes/pull/1006) documents the MI355X TP2 Engram offload and `--no-swa-bounded-replay`, and the merged [#968](https://github.com/vllm-project/recipes/pull/968) records the original AMD overrides and the complete InferenceX command. Follow the [AgentX procedure](./eval-agentx-procedures.md#7-run-agentx-fast-feedback-versus-canonical-evidence) for future runtime evidence; local generation and registry metadata alone are not GPU proof.

## DeepSeek-V4.1-Flash on MI300X and MI325X

`dsv41flash-fp4-mi300x-vllm-agentic-dspark` and `dsv41flash-fp4-mi325x-vllm-agentic-dspark`
copy the validated MI355X vLLM arm onto gfx942, on the `nightly-eed1f3d0` ROCm nightly the
MI355X arm used before it moved to the `nightly-rocm100` channel, and with the same
AMD overrides (`VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_USE_AITER_MOE=1`,
`VLLM_USE_BREAKABLE_CUDAGRAPH=1`, `--moe-backend aiter`, adaptive verification off). gfx942
is not in the upstream hardware table, and it has no FP4 MFMA: the plain `aiter` MoE
backend lets vLLM's selector skip the gfx950-only CK a8w4 experts, and pinning
`aiter_triton_mxfp4_bf16` (the Triton W4A16 kernel) is the first repair lever if startup
rejects every candidate.

Both arms run **TP8**, not the MI355X TP4: a 192 GB (MI300X) or 256 GB (MI325X) card must
hold its share of the 511 GB checkpoint plus the GPU-resident Engram tables (upstream AMD
defaults; no CPU offload) and still leave a 1M-context KV pool. MI300X additionally caps
`--max-num-batched-tokens` at 8192 because the sparse-attention indexer allocates a
`[batched-tokens, max-model-len]` fp8 logits buffer at startup (16 GiB at 8192, 32 GiB at
the MI355X arm's 16384). Concurrency is 1–32 on both.

`runners/launch_mi300x-amd.sh` and `runners/launch_mi325x-amds.sh` mount the checkout at
`/ix` for this checkpoint and rewrite `RESULT_DIR`, as the MI355X launcher does, so AgentX
runtime directories stay out of `/workspace`. The MI300X launcher also raises its Slurm
allocation from 180 to 480 minutes for this checkpoint: the HF cache there is node-local, so
the first arm on each node downloads 511 GB before serving. GPU sweep and eval evidence is
required before calling either arm validated.
