# Configuration Procedures

<div align="center">

**English** | [中文](configuration-procedures_zh.md)

</div>

Use this page for benchmark configuration, recipe, image, and runner changes. It is a procedure, not a field catalog: the linked implementation and schema remain authoritative.

## Source map

| Source of truth | What it controls |
| --- | --- |
| [`configs/CONFIGS.md`](../configs/CONFIGS.md) | Master-config and runner-config field contract |
| [`infx/matrix/validation.py`](../infx/matrix/validation.py) | Enforced Pydantic schema and topology invariants |
| [`infx/matrix/generate.py`](../infx/matrix/generate.py) | Matrix expansion, filtering, runner lookup, and emitted job metadata |
| [`configs/nvidia-master.yaml`](../configs/nvidia-master.yaml), [`configs/amd-master.yaml`](../configs/amd-master.yaml) | Executable benchmark definitions |
| [`configs/runners.yaml`](../configs/runners.yaml) | Schedulable labels, concrete runner names, and per-cluster records (`clusters:`) |
| [`benchmarks/`](../benchmarks) and [`infx/launch/`](../infx/launch) | Runtime commands, launch drivers, and workload launch policy |
| [`perf-changelog.yaml`](../perf-changelog.yaml) | Append-only benchmark trigger log |
| [`AGENTS.md`](../../AGENTS.md) | Repository-wide config, MTP, changelog, and sweep rules |

Delete retired entries from the active master configs; they are not archived. Git history and `perf-changelog.yaml` keep the historical settings. For partial retirements, remove only the retired scenarios. Delete unused recipes and model-specific setup as well. Preserve shared dependencies needed by retained SPEED-Bench collectors, including their scheduling scores. See the [deprecation rules](../../AGENTS.md#deprecating-benchmark-configs).

## Dependency submodules

Git records the exact dependency commits. [`.gitmodules`](../../.gitmodules) defines the repositories: AIPerf at `utils/aiperf`, NVIDIA srt-slurm at `utils/srt-slurm`. All srt-slurm jobs, including TileRT, use the pinned upstream submodule.

Initialize them before running benchmarks locally:

```bash
git submodule update --init
```

To upgrade, fetch and check out the desired commit inside the relevant submodule, then commit the updated submodule pointer in InferenceX. Benchmark workflows already initialize submodules. Slurm launchers make a local Git clone for each job so recipe staging and runtime writes do not modify the submodule, and record the actual commit for result provenance.

Single-node fixed-sequence recipes use NVIDIA upstream srt-slurm. ATOM recipes use
the native `atomesh` frontend with one aggregate worker and
`enable_multiple_frontends: false`. The router's pinned official image belongs in
`frontend.container_image`: older benchmark worker images do not include AToMesh.
Keep `model.container` aligned with the master config's worker `image`; changing the
router image does not require changing the worker image. TRT-LLM recipes use native
`engine.served_model_name`, without duplicating that flag in `roles.agg.extra_args`.
The former fork's direct ATOM frontend is not required.

### Cluster profiles

A cluster's srt-slurm settings live in its `clusters.<id>.slurm.srt-slurm` record in
[`configs/runners.yaml`](../configs/runners.yaml) (schema:
[`infx/clusters/slurm.py`](../infx/clusters/slurm.py)).
srt-slurm only runs on Slurm, so the profile is part of the Slurm sub-record. The native
settings (network interface, scheduling directives, the single-node time limit,
container, nginx and model aliases, volume and host mounts, the launch environment,
host setup, and literal `extra` keys) stay separate from workload recipes.

Before `make setup`, the srt driver ([`infx/launch/drivers/srt/`](../infx/launch/drivers/srt))
renders the job-local `srtslurm.yaml` (`config.py`) from that record plus job values:
staged images, resolved model paths, cache mounts, the time limit, and the DCGM
exporter image for power jobs. Values are written as YAML data, never substituted into
shell or YAML text, and `extra` cannot shadow a typed key.

NVIDIA profiles set `slurm.srt-slurm.extra.default_gpu_exporter` to the same `dcgm-exporter:4.6.0-4.8.3-distroless`
image the power path uses, with `/configs/dcgm-counters-noprof.csv` for Tachometer's
implicit DCGM exporter. Power recipes select the same CSV through
`telemetry.dcgm_exporter.command`, so one exporter version and one counter file serve
both paths. Tachometer reuses the power exporter when power telemetry owns it. Power,
energy and GPU utilization remain available; profiling and vGPU license counters are
omitted. This does not enable telemetry in opted-out recipes or qualify Tachometer
metrics as validated PowerX results. The existing 1000 ms Tachometer / 100 ms
power-exporter collection intervals and port 9401 are preserved.

Keep model selection, cache preparation, and workload-dependent time limits in the
srt driver's tables ([`lanes.py`](../infx/launch/drivers/srt/lanes.py),
[`models.py`](../infx/launch/drivers/srt/models.py),
[`power.py`](../infx/launch/drivers/srt/power.py)), not in the cluster record.

Put per-allocation host checks and setup in
`runners/srt-slurm/hooks/<cluster>/setup.sh`, with cluster-specific helpers beside it.
The directory name matches the cluster id. Register the script in the cluster's
`slurm.srt-slurm.host-setup` record (`script`, `env`, `timeout-s`, `nodes`), which the driver
renders as `default_host_setup`. Scripts are not auto-discovered. srt-slurm runs them on
the selected allocated nodes, outside containers, before starting services and workers.
A failed check stops startup by default. Pass configuration explicitly through
`host-setup.env`. If setup ever needs an undo step, add a `teardown` field to
`HostSetup` (infx/clusters/slurm.py) and render it as `default_host_setup.teardown`; no cluster
needs one today. These are job-owned hooks, not administrator-installed Slurm
Prolog/Epilog scripts. Only add hooks for clusters that need them; do not add empty
scripts for every cluster.

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

Source: [`docs/agent-guide.md`](agent-guide.md), [`AGENTS.md`](../../AGENTS.md).

From `inferencex-e2e/` in a clean checkout:

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

Detailed source: [`.claude/commands/add-model-hardware.md`](../../.claude/commands/add-model-hardware.md). Field source: [`configs/CONFIGS.md`](../configs/CONFIGS.md).
STP (Single Token Prediction) is vanilla autoregressive decoding with one token per forward pass. MTP (Multi-Token Prediction) predicts multiple tokens per forward pass through native heads or speculative decoding.

1. **Fix the identity.** Confirm the exact checkpoint ID, model prefix, precision, architecture, native context, target SKU, framework, and whether decoding is STP, native MTP, or draft-model speculation. Verify the image tag exists. Never invent one.
2. **Choose two kinds of sibling.** Read the same model on another SKU and another model on the target SKU. Read their srt-slurm recipes under [`benchmarks/single_node/srt-slurm-recipes/`](../benchmarks/single_node/srt-slurm-recipes) and master-config entries.
3. **Add the srt-slurm recipe.** Put it at `benchmarks/single_node/srt-slurm-recipes/<model-prefix>/<engine>/<sku>-<precision>[-mtp]/8k1k.yaml`, with one `override_*` variant per matrix point. Preserve the proven sibling's engine args, env, parser flags, attention/MoE backend, KV-cache dtype, graph/eager mode, `setup_script`, and context handling.
4. **Add the master entry.** Use [`amd-master.yaml`](../configs/amd-master.yaml) for `mi*`. Otherwise, use [`nvidia-master.yaml`](../configs/nvidia-master.yaml). Set exact `image`, `model`, `model-prefix`, `runner`, `precision`, `framework`, scenarios, and supported search spaces.
5. **Size from evidence.** Mirror proven parallelism layouts and trim unsupported ones. Latency TP rows normally start at concurrency 1. Do not copy large-memory TP/EP layouts onto a smaller SKU.
6. **Check variant selection.** Every search-space row carries `srt-recipe:`, and each matrix point must match exactly one recipe variant by TP/GPU count, `CONC`, `KV_OFFLOADING` and image (`infx/srt_slurm/single_node.py::select_recipe`). No launcher routing is needed.
7. **Append one changelog entry** for the exact new key. See [Append the changelog safely](#append-the-changelog-safely).
8. **Validate syntax and generated output.** Inspect image, model, runner, ISL/OSL, `max-model-len`, concurrency, TP/PP/EP/DCP/PCP, and `spec-decoding`.

A `MODELS.md` row alone is not an executable recipe. The complete path is srt-slurm recipe + master entry (`srt-recipe:`) + changelog trigger + generated matrix.

## Change a master config

Sources: [`configs/CONFIGS.md`](../configs/CONFIGS.md), [`validation.py`](../infx/matrix/validation.py), [`generate.py`](../infx/matrix/generate.py).

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

1. Add the fleet's `clusters.<id>` record to [`configs/runners.yaml`](../configs/runners.yaml) (node shape, workload env, models, and the scheduler sub-record: for Slurm the partition, volumes, squash cache and srt-slurm facts; schema in [`configs/CONFIGS.md#runners`](../configs/CONFIGS.md#runners)). Put launch rules that depend on model, framework, precision or recipe in [`infx/launch/policy.py`](../infx/launch/policy.py) or beside the one driver that reads them, never in a driver branch on the cluster id. A cluster on a new scheduler needs that scheduler's settings model under [`infx/clusters/`](../infx/clusters) and its backend under [`infx/launch/backends/`](../infx/launch/backends), one registry entry each, and no driver change; it runs only script-driver (`BENCH_SCRIPT_OVERRIDE`) points.
2. Add each exact registered runner name under the intended `labels:` key in [`configs/runners.yaml`](../configs/runners.yaml). New names use `<base-name>_<NN>` with zero-padded indices.
3. Add every runner name to exactly one `cluster:<id>` label matching that record. `python -m infx.launch run` resolves the cluster from the runner name, so a runner outside every cluster label fails validation and fails at launch.
4. Master entries whose facts depend on one physical fleet use that exact `cluster:<id>` label. Agentic configs require it.
5. Add/update master entries to use that label. Generate a targeted matrix and confirm the selected concrete names.

Routing is by `cluster:<id>` label, not by runner-name prefix. Keep `<base-name>` free of `_`: `_` separates it from the runner index.

### Host setup

1. Decide the runner user and shared storage. `_work` must be visible to login and compute nodes.
2. Confirm `curl`, `tar`, `tmux`, and, for Slurm, `sinfo`/`srun`/`sbatch` are on the registration shell's `PATH`.
3. Obtain repo-admin authentication and a fresh registration token. It expires after about one hour.
4. Run the documented [`setup.sh`](../utils/runner_setup/setup.sh) with token, runner URL, index range, base directory, base name, and labels.
5. Start with [`start_runners.sh`](../utils/runner_setup/start_runners.sh).
6. Verify every runner is **Idle** in [repository runner settings](https://github.com/SemiAnalysisAI/InferenceX/settings/actions/runners) before adding it to sweep traffic.
7. Verify launcher mounts for `_work`, HF cache, staged weights, and squash images from a compute node. Root containers must not leave root-owned files in the shared workspace.

## Register an srt-slurm recipe

Mapping source: [`benchmarks/multi_node/srt-slurm-recipes/RECIPES.md`](../benchmarks/multi_node/srt-slurm-recipes/RECIPES.md). Checked-in recipes: [`benchmarks/multi_node/srt-slurm-recipes/`](../benchmarks/multi_node/srt-slurm-recipes).

1. Locate the exact upstream [NVIDIA/srt-slurm](https://github.com/NVIDIA/srt-slurm) recipe and record its commit-pinned source path.
2. Stage the YAML under `benchmarks/multi_node/srt-slurm-recipes/<model-prefix>/<engine>/<gpu>-<precision>/<workload>/`, following the naming rules in `RECIPES.md`. Read the closest sibling and selected cluster launcher.
3. Map source fields to the master search-space entry: resource worker counts → `num-worker`, TP/EP/DP-attention → worker topology, benchmark concurrencies → `conc-list`, and recipe path → `additional-settings: ["CONFIG_FILE=..."]`.
4. Add/update the matching [`nvidia-master.yaml`](../configs/nvidia-master.yaml) entry in the same change. Keep worker counts, TP/PP/EP/DCP/PCP, hardware, router, transfer engine, and concurrency labels synchronized.
5. For an image bump, make recipe `model.container` exactly equal master `image`. The launcher uses the master image as the container-alias key.
6. Run the recipe's documented `srtctl` validation, then generate the master key and compare every frontend label/topology field with the recipe.
7. Append the changelog entry.

Do not ship one side alone. `srtctl` reads the recipe, while matrix generation reads the master config. Recipe-only changes can mislabel results. Master-only changes do not alter the deployed recipe.

## Register an llm-d recipe

Sources: [`benchmarks/llm-d/README.md`](../benchmarks/llm-d/README.md), [`benchmarks/multi_node/llm-d/README.md`](../benchmarks/multi_node/llm-d/README.md), and [`llm-d-recipes/`](../benchmarks/multi_node/llm-d-recipes).

llm-d is not the srt-slurm path: InferenceX owns the Slurm allocation and starts one container per node.

1. Copy the nearest YAML under [`benchmarks/multi_node/llm-d-recipes/`](../benchmarks/multi_node/llm-d-recipes) and set EPP plugins/scheduling, role-specific `extra-args`/`env`, and optional `slurm.time_limit`.
2. Add/update the `llmd-vllm` master entry. Set `multinode: true`, `disagg: true`, router metadata, `kv-p2p-transfer`, prefill/decode worker topology, concurrency, and `CONFIG_FILE=<basename>.yaml` in `additional-settings`.
3. Keep `PREFILL_NODES`, `DECODE_NODES`, `GPUS_PER_NODE`, and worker counts consistent with the allocation and with each role's DP/TP/EP layout.
4. Confirm [`submit.sh`](../benchmarks/multi_node/llm-d/submit.sh) → [`job.slurm`](../benchmarks/multi_node/llm-d/job.slurm) → [`server.sh`](../benchmarks/multi_node/llm-d/server.sh) propagation and the selected wrapper/launcher route.
5. Verify file discovery. The decode leader creates `/tmp/endpoints.yaml`. Prefill endpoints use vLLM port 8200, while decode endpoints use sidecar port 8000. Names must be unique, addresses must be literal IPv4, and ports must be strings in `1..65535`.
6. Confirm EPP loads discovery before Envoy receives traffic and role labels select the proper prefill/decode backends.
7. Generate the key, inspect topology and `additional-settings`, then append the changelog.

A missing/unset `CONFIG_FILE` silently selects the image's `/etc/epp/config.yaml` fallback and removes recipe-specific vLLM flags. Treat that as a validation failure unless fallback is explicitly intended.

## Update an image

Sources: [`AGENTS.md#non-negotiable-benchmark-invariants`](../../AGENTS.md#non-negotiable-benchmark-invariants), the matching master configs, runtime scripts, and checked-in recipes.

1. Verify the exact upstream registry tag or digest exists and is appropriate for CUDA/ROCm and the target architecture.
2. Find every affected config key, runtime script, Dockerfile, and checked-in recipe. Do not assume the master YAML is the only image reference.
3. Update the master `image` and any required env vars, flags, package versions, or patches as one coherent change.
4. For srt-slurm, update `model.container` and keep it identical to master `image`.
5. For llm-d, distinguish the serving image selected by the master config from the build source in [`benchmarks/llm-d/Dockerfile`](../benchmarks/llm-d/Dockerfile). Update both only when the build contract changes.
6. Append a changelog entry selecting all affected keys (wildcards are allowed when intentional), including old/new versions and material runtime changes.
7. Generate each affected family and verify no stale tag survives in its runtime path.

## Add or change MTP

Sources: [`AGENTS.md#non-negotiable-benchmark-invariants`](../../AGENTS.md#non-negotiable-benchmark-invariants), [MTP appendix in the model+hardware playbook](../../.claude/commands/add-model-hardware.md#appendix--mtp--eagle3-spec-decoding-variant), and current [`*-mtp` srt-slurm recipes](../benchmarks/single_node/srt-slurm-recipes).

1. Confirm native MTP modules versus an external draft. For a draft, verify exact model ID, method (for example `eagle3`), and recommended speculative-token count from the model/upstream recipe.
2. Copy a working sibling for the same model and backend. Preserve its speculative config, attention backend, token count, model patches, and dependency setup.
3. Every speculative fixed-sequence recipe variant must set `benchmark.env.USE_CHAT_TEMPLATE: "true"`; `select_recipe` rejects a speculative variant without it, and [`srt_fixed_sequence.sh`](../benchmarks/single_node/srt_fixed_sequence.sh) turns it into `--use-chat-template` for `run_benchmark_serving`. Raw prompts silently depress acceptance.
4. Size graph capture for at least `CONC * (1 + NUM_SPEC_TOKENS)`, rounded as the sibling does and capped at the framework limit (the current vLLM playbook caps at 2048).
5. Keep backend differences: do not copy CUDA-only drafter attention pins or patches into ROCm recipes.
6. Set `spec-decoding: mtp` in the relevant search-space entries and point their `srt-recipe:` at the `-mtp` recipe; `select_recipe` checks the recipe's speculative config against it. For a draft-model mode supported by the schema, use the matching generated value deliberately. Do not infer it from a filename.
7. Add recipe + master entry + changelog together.
8. Run YAML and generation checks. Inspect `spec-decoding`, draft/native method, token count, chat-template use, capture range, and resolved recipe variant.

### DeepSeek-V4-Pro-0813 DSpark on MI355X ATOM

`dsv4-fp4-mi355x-atom-agentic-mtp` keeps its historical key, while its matrix
uses `spec-decoding: draft_model`. Every row selects
`benchmarks/single_node/srt-slurm-recipes/dsv4/atom/mi355x-fp4-mtp/agentic.yaml`;
recipe selection treats `draft_model` as `mtp`. The recipe pins revision
`72e1d3230f6c080a530b0a1d46f8eb4602340597` and serves the resolved snapshot path;
an explicit `MODEL_PATH` must pass the same checkpoint checks. Before GPU
startup it verifies the config/index hashes, DSpark Markov/confidence heads,
all 66 shard headers and payload boundaries, and offline tokenizer loading.
This checks readability and completeness, not full weight-file hashes.

All ten AgentX throughput points use DSpark K6 (target verification length 7)
and the committed golden AL 3.77. C1/2/4/8/16 use TP8/EP1;
C48/64/96/128/256 use TP8/DPA8/EP8 with native RCCL. Each point runs for
3600 seconds. The C256 full GSM8K eval omits forced acceptance. Keep the
pinned `rocm/atom-dev:nightly_202609291501` image and GPU-only KV. C1 through C16 use
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
`rocm/atom-dev:nightly_202609291501` (ATOM `0.1.7.dev46+g74fd942b0`, ROCm 7.2.4),
which includes the merged
[ROCm/ATOM#2233](https://github.com/ROCm/ATOM/pull/2233) inference-mode fix.
From this image ATOM stores the checkpoint's `ue8m0` FP8 block scales as E8M0
on gfx950 by default ([ROCm/ATOM#2419](https://github.com/ROCm/ATOM/pull/2419));
the powers-of-two scales are represented exactly.
The recipe does not patch AITER source at runtime; TP communication
fusion, DSpark K6 and graph capture use the implementation shipped in the image.

### MiniMax-M3 ATOM FlyDSL paged decode and LMCache DRAM tier

`minimaxm3-fp4-mi355x-atom-agentic-mtp` uses
`rocm/atom-dev:nightly_202609281543` with `ATOM_PA_FLYDSL=1` and
`ATOM_PA_FLYDSL_PLAN=1`, following [ROCm/ATOM#2366](https://github.com/ROCm/ATOM/pull/2366)
and the [upstream recipe](https://github.com/ROCm/ATOM/blob/1423fceb08fbe88b2e35c77b320b3073b310a63f/recipes/MiniMax-M3-Agentic-InferenceX.md).
FlyDSL handles supported paged-decode shapes; its work planner balances dense
decode by actual context length. Unsupported shapes retain the Gluon fallback.
Verify the selected route and capture-time work-plan creation in `server.log`.

The TP2 C20/C25/C30 and TP4 C40/C48 points add LMCache's in-process CPU tier.
The `*_lmcache` variants in
`benchmarks/single_node/srt-slurm-recipes/minimaxm3/atom/mi355x-fp4-mtp/agentic.yaml`
list `{kv_connector: lmcache_offload, kv_role: offload}` under
`extra-kv-connectors`, which srtctl renders as
`--kv-transfer-config '{"kv_connector":"lmcache_offload","kv_role":"offload"}'`
(`runners/srt-slurm/patches/507-lmcache-server-atom-sglang.patch`). The
`LMCACHE_*` environment variables configure the tier. Each variant declares
`KV_OFFLOADING: dram` and the matrix `TOTAL_CPU_DRAM_GB`, and sizes
`LMCACHE_MAX_LOCAL_CPU_SIZE` at `TOTAL_CPU_DRAM_GB / TP` (257 GB per rank).
`PYTHONHASHSEED=0` is required: without it the ranks hash prompts to different
keys and the offload hit rate is zero. Verify the composed `kv_transfer_config`
line and non-zero `atom:lmcache_loaded_tokens` in `server.log`.

srtctl masks a partial-node worker to GPUs `0..TP-1`, which sit on NUMA node 0,
and HIP pins each rank's CPU tier on its GPU's node, so the tier and the weight
staging buffers (about 720 GB at TP2 and 1.23 TB at TP4) all come from node 0's
1.5 TB. `runners/srt-slurm/hooks/mi355x-amds/setup.sh` drops the page cache before
a `KV_OFFLOADING=dram` job so these pages are free. Without it, pinning reclaims
page cache, ranks finish minutes apart and ATOM's 300 s startup barrier times out
([run 36454319395](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/36454319395)).

TP4 C32 is dropped; the GPU-resident TP4 C1-C28 and TP2 C1-C2 points,
EAGLE3 K3, golden AL 2.78 and indexer CP are unchanged.

### DeepSeek-V4.1-Flash DSpark

The GB200 DSpark recipe uses a minimum CUDA graph capture size of 64 tokens to cover concurrent AgentX subagents. This raises c1/c2/c4 from 8/16/32 to 64; c8 and above retain their existing sizes. The full trace, AL 3.51, and Engram UVA settings are preserved; low-concurrency tail latency improvements require CI confirmation.
The B200 DSpark recipe sets explicit capture sizes per point, described below.
The GB300 DSpark recipe sets explicit capture sizes per point, described below.
The H200 DSpark recipe uses the same minimum capture size and preserves the same workload settings.

The B300 DSpark recipe sets explicit capture sizes per point, described below.

The B200 entry uses `vllm/vllm-openai:nightly-dev-x86_64-cu130-ac9126e58aa7` with FlashInfer
autotuning. TP4 covers concurrency 1–128. DEP2 (TP1 x DP2 + EP2, DeepGEMM MegaMoE) covers 8–32 and
DEP4 (TP1 x DP4 + EP4, MegaMoE) covers 64–128, both behind a consistent-hash vLLM Router. DEP2 keeps about
150 GiB of weights per B200 rank, so it caps batched tokens at 4096 and graph capture at 576 tokens.
All B200 points set `--gpu-memory-utilization 0.97`.
All points use `FULL_AND_PIECEWISE` CUDA graphs sized in multiples of the six-token verification block.

The AgentX-only `dsv41flash-fp4-<sku>-vllm-agentic-dspark` recipes use the per-SKU
`image` pinned in [`nvidia-master.yaml`](../configs/nvidia-master.yaml) (originally `vllm/vllm-openai:deepseekv41-flash-0909`) at TP4 on Blackwell SKUs with native five-token DSpark,
probabilistic drafting. Throughput uses the [committed golden AL](../infx/golden_al_distribution/dsv41flash_dspark.yaml) of 3.51 for thinking on and five draft tokens, with synthetic rejection sampling and adaptive verification disabled. Accuracy evals retain real block rejection and adaptive verification. `--engram-config '{"cpu_offload":true}'`
stores Engram embedding tables in pinned host DRAM accessed through UVA;
`kv-offloading: none` describes the separate, GPU-resident KV cache. MXFP4 expert
weights determine the recipe's `precision: fp4` label.

The GPU-specific entry points share the text-only serving behavior, `deepseek_v41` tokenizer and
parsers, 1M context, and the shared AgentX trace replay, power, metrics, and eval
helpers. Unless noted below, the TP4 concurrency range is 1–128. The shared script sizes graph capture
for the six-token DSpark verification block. The srt-slurm single-node path mounts the checkout at `/infmax-workspace`, so
AgentX runtime directories are not created under `/workspace`. Cluster model paths and persistent caches are reused.
The recipe probes the serving port on the compute node and selects an available
port if the preferred one is occupied. Serving, replay, metrics, and eval share
that endpoint.

The B300 entry uses `vllm/vllm-openai:nightly-dev-x86_64-cu130-ac9126e58aa7` with FlashInfer
autotuning. TP4 covers concurrency 1–16; DEP2 (TP1 x DP2 + EP2, DeepGEMM MegaMoE) covers 8–192
behind a consistent-hash vLLM Router, switching to MegaAttention at 128 and above. All points use
`FULL_AND_PIECEWISE` CUDA graphs sized in multiples of the six-token verification block.
Other SKUs continue to use the shared script.

The GB300 entry uses `vllm/vllm-openai:nightly-dev-arm64-cu130-ac9126e58aa7` with FlashInfer
autotuning. TP4 covers concurrency 1–16; DEP2 (TP1 x DP2 + EP2, DeepGEMM MegaMoE) covers 8–192
behind a consistent-hash vLLM Router, switching to MegaAttention at 128 and above. All points use
`FULL_AND_PIECEWISE` CUDA graphs sized in multiples of the six-token verification block.

The GB300 launcher allows 7200 seconds for engine readiness. In [run 34504969146](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34504969146), the Rust frontend exhausted its 3600-second deadline while the engine was still capturing graphs; model loading alone took 18–23 minutes. This extends startup time without changing the benchmark duration or decoding settings.

GPU sweep and eval evidence is required before calling any recipe validated.

Source: [upstream recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash).

### DeepSeek-V4.1-Flash DSpark on H200

`dsv41flash-fp4-h200-vllm-agentic-dspark` is the H200 AgentX arm of the
DeepSeek-V4.1-Flash recipe. It pins `vllm/vllm-openai:nightly-cd10ed6f9f6b37a8ace9cf380007e66fe12ec0c3` (shared with B200 and GB200) and shares the
text-only serving settings with the Blackwell arms: `deepseek_v41` tokenizer and parsers,
1M context, native five-token DSpark with probabilistic drafting. Throughput uses the [committed golden AL](../infx/golden_al_distribution/dsv41flash_dspark.yaml) of 3.51 for thinking on and five draft tokens, with synthetic rejection sampling and adaptive verification disabled. Accuracy evals retain real block rejection and adaptive verification.

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
and invisible at the call site, and no test pins it (the former `runners/test_dsv41flash_h200.py`
was removed in #3141); narrowing the arm would silently downgrade this recipe's traces.

**The H100 arm is separate.** H100 is not in the upstream hardware table, and the
blocker is not the weights. At 1M context the sparse attention indexer allocates a
`[max-num-batched-tokens, max-model-len]` logits buffer in
`fp8_fp4_paged_mqa_logits`, which at the default 8192 batched tokens is exactly 16 GiB.
That is a fixed startup cost paid during memory profiling, independent of concurrency, so
it fails at concurrency 1 on an 80 GB card even though the resident weights fit. The H100
arm therefore ships its own recipe settings with capped batched tokens; see the H100
section below.

The srt-slurm single-node path mounts the checkout at `/infmax-workspace`, so AgentX runtime
directories are not created under `/workspace`. It also mounts the shared HF cache, so the
recipe resolves the model through `HF_HUB_CACHE` rather than a per-node path. The recipe
probes the serving port on the compute node and selects an available one if the preferred
port is occupied; serving, replay, metrics, and eval share that endpoint.

### DeepSeek-V4.1-Flash DSpark on H100

Throughput uses the [committed golden AL](../infx/golden_al_distribution/dsv41flash_dspark.yaml) of 3.51 for thinking on and five draft tokens, with synthetic rejection sampling and adaptive verification disabled. Accuracy evals retain real block rejection and adaptive verification.

`dsv41flash-fp4-h100-vllm-agentic-dspark` is the H100 AgentX arm of the
DeepSeek-V4.1-Flash recipe, added after the H200 arm and deliberately separate from
it. H100 is **not** in the upstream hardware table, which lists h200, gb200, gb300, and
mi350x.

Unlike the other SKUs, H100 did not inherit the shared serving flags. Its recipe,
`benchmarks/single_node/srt-slurm-recipes/dsv41flash/vllm/h100-fp4-mtp/agentic.yaml`, carries its own,
because the shared flags cannot serve 1M context on an 80 GB card. At 1M
context the sparse attention indexer allocates a
`[max-num-batched-tokens, max-model-len]` logits buffer in `fp8_fp4_paged_mqa_logits`:
at the shared flags' effective 8192 batched tokens that is 8192 x 1048576 x 2 bytes,
exactly 16.00 GiB. It is a fixed cost paid during startup memory profiling, independent
of concurrency, so it OOMed at concurrency 1 in run
[34467029236](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34467029236)
next to roughly 35.9 GiB per GPU of resident weights — trimming the concurrency list
cannot help.

The H100 recipe therefore caps `--max-num-batched-tokens` at 4096, putting the indexer
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

This arm runs through its srt-slurm single-node recipe (`srt-recipe:`), which mounts the
checkout at `/infmax-workspace`, so AgentX runtime directories are not created under
`/workspace`.

Source: [upstream recipe](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml).

### DeepSeek-V4.1-Flash DSpark on SGLang

The GB300 DeepSeek-V4.1-Flash SGLang curve uses `dev-cu13-nightly-0924`:
TP4/EP1 at C1/C2 with GPU Engram and 4K prefill chunks, and TP4/EP4 at C4+
with host Engram and 16K chunks. TP2 is unchanged. Full sweep validation is pending.

The H100 SGLang candidate sweeps DSpark at concurrency 1/2/4/8/16/20. It retains 8 SWA prefix tails per concurrency at C1/C2 and 32 at C4 and above. A matched one-hour comparison rejected a blanket 128-tail floor: C2 throughput improved only 1.7% while interactivity fell 44.5%. Completed STP comparisons did not contribute a measured frontier point, so STP is excluded from the selected sweep. The recipe interleaves 16 decode steps between prefill chunks, preserving trace content and context limits.

The same sweep also qualifies supported TP8/EP8/DP8 attention at C4/C8/C16/C20. DP uses a stock consistent-hash router with stable session keys, DP LM-head execution, and 64 SWA prefix tails per rank. Full C16 GSM8K passed on all 1,319 examples; its performance contribution remains under measurement. The native 1M context and the AgentX subagent/session semantics are preserved.

The nightly candidate uses `nightly-dev-cu13-20260922-582389ce`, native MXFP4 Marlin MoE, and `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank`. A resolved local snapshot lets the upstream allocator evict checkpoint file cache before allocating anonymous host tables. Draft precision follows the pinned image's default handling, including its WO_A FP8-to-BF16 conversion. Block32 FP8 GEMMs use SGLang's default tilings; the custom block32 launch configurations were removed in [#3463](https://github.com/SemiAnalysisAI/InferenceX/pull/3463). Full canonical qualification is still required.

`dsv41flash-fp4-<sku>-sglang-agentic-dspark` are the SGLang counterparts of the vLLM
arms, one PR per SKU across h100, h200, b200, b300, gb200, gb300 and mi355x. They follow the
[SGLang cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1),
which has no released SGLang version for this model yet. B200, B300 and H100 pin the CUDA 13 nightly
`lmsysorg/sglang:nightly-dev-cu13-20260922-582389ce` by digest; GB300 pins `lmsysorg/sglang:dev-cu13-nightly-0924` by digest; GB200 and H200 use
`lmsysorg/sglang:nightly-dev-cu13-20260923-06008c17` (GB200 by digest), and MI355X pins
`lmsysorg/sglang:dev-dsv41-mi35x` by digest. Each master entry's `image` is authoritative.

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
Image staging ([`infx/launch/backends/slurm/squash.py`](../infx/launch/backends/slurm/squash.py)) converts pinned Docker
digests to Enroot's `registry#repository:digest` syntax on every cluster. It retries
transient import errors and fails the launch if the squash still does not validate.

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
[committed golden AL](../infx/golden_al_distribution/dsv41flash_dspark.yaml) of 3.51 for thinking
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

Each SKU ships its own recipe, `benchmarks/single_node/srt-slurm-recipes/dsv41flash/sglang/<sku>-fp4-mtp/agentic.yaml`, in its own PR. H100 is not in
the cookbook's hardware table, so its recipe differs: the Engram tables move to a single shared host copy
(`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`, the SGLang analogue of the vLLM arm's Engram
CPU offload) and the prefill chunk is capped at 4096, the same batched-token cap the vLLM
H100 arm needed for the sparse-attention indexer buffer on an 80 GB card. Concurrency
stops at 8 there until the KV ceiling is measured. MI355X has its own recipe with the
cookbook's ROCm environment (`SGLANG_USE_AITER=1`, `SGLANG_MOE_PADDING=1`,
`AITER_FLYDSL_FORCE_REDUCE=1`, `ROCM_QUICK_REDUCE_QUANTIZATION=NONE`),
`--disable-radix-cache`, and breakable prefill graphs capped at 4096 tokens.

The KV cache is GPU-resident on every arm, so `kv-offloading: none`. The srt-slurm
single-node path treats `framework: sglang` for `dsv41flash` the same way as vLLM: the
checkout is mounted at `/infmax-workspace`, and the checkpoint resolves through each
cluster's persistent HF cache.

GPU sweep and eval evidence is required before calling any of these arms validated.

## Validate

Run the smallest checks that cover the edited layers.

### YAML parse

```bash
python3 -c "import yaml; yaml.safe_load(open('configs/<nvidia|amd>-master.yaml')); yaml.safe_load(open('configs/runners.yaml')); yaml.safe_load(open('perf-changelog.yaml'))"
```

### Benchmark syntax and launch checks

```bash
bash -n benchmarks/<path>/<script>.sh
uv run python -c 'from infx.clusters import load_clusters; load_clusters()'
uv run pytest -q infx/tests/launch infx/tests/clusters
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

Inspect, do not merely count, the emitted `model`, `image`, `runner`, scenario, concurrency, `max-model-len`, TP/PP/EP/DCP/PCP, prefill/decode worker blocks, hardware, router, KV transfer, eval flags, `additional-settings`, and `spec-decoding`.

If schema or generator behavior changed, run its focused suite:

```bash
python -m pytest infx/tests/matrix/ -v
```

For srt-slurm, also run the upstream recipe checker/`srtctl` command documented for that recipe. For llm-d, validate recipe YAML and exercise the allocation/discovery path on the intended Slurm fleet. Local matrix generation cannot prove endpoint discovery.

## Avoid schema and topology traps

Enforced details come from [`validation.py`](../infx/matrix/validation.py) and are summarized in [`configs/CONFIGS.md`](../configs/CONFIGS.md):

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

Sources: [`AGENTS.md#non-negotiable-benchmark-invariants`](../../AGENTS.md#non-negotiable-benchmark-invariants), [`perf-changelog.yaml`](../perf-changelog.yaml).

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
7. Parse the file and confirm the generated changelog selection includes the intended keys before requesting a sweep. Request it by applying exactly one primary sweep label, normally `full-sweep-fail-fast` (see [PR primary and modifier labels](ci-procedures.md#pr-primary-and-modifier-labels)).

## Stop conditions

Stop before dispatching GPU work or claiming the configuration complete when any condition below holds. Obtain the missing fact or fix the source mismatch. Do not guess.

- Exact checkpoint, precision, architecture, native context, framework, draft model/method, or image tag is unverified.
- No proven sibling covers the target model/backend/SKU, and required runtime flags or memory limits remain unknown.
- Runner user, shared mounts, staged model path, GPU count, host DRAM, Slurm behavior, or root-file cleanup is unknown. Runner registration credentials are also a hard prerequisite for host setup.
- The registered runner is in no `cluster:<id>` label, a matrix resolves to a nonexistent script, or the runner is not **Idle**.
- Calculated topology exceeds the fleet, DCP does not divide TP, heterogeneous hardware metadata is one-sided, or generated topology differs from the intended recipe.
- An srt-slurm recipe and master entry disagree, `model.container != image`, or upstream recipe validation has not run.
- An llm-d recipe is missing and would fall back unintentionally, allocation counts disagree, or endpoint discovery cannot satisfy literal-IPv4/unique-name/valid-port rules.
- An MTP recipe lacks chat-template benchmarking, the speculative method/token count is unverified, or graph capture exceeds the backend limit.
- The changelog change would modify historical bytes, is not at EOF, has a conflict, or still has `TBD` when the PR is otherwise ready for sweep.
- YAML, Bash, strict schema, exact-key generation, cluster-record validation, launch tests, or recipe validation fails.

A configuration is ready for sweep only when the executable files agree, the exact key generates, the runtime route exists, the changelog selects it, and all layer-specific checks above pass.

## DeepSeek-V4.1-Flash on MI355X

The `dsv41flash-fp4-mi355x-vllm-agentic-dspark` recipe extends [#2958](https://github.com/SemiAnalysisAI/InferenceX/pull/2958) to MI355X AgentX: TP4 and TP2, concurrency 1–64, native five-token DSpark. Throughput uses the [committed golden AL](../infx/golden_al_distribution/dsv41flash_dspark.yaml) of 3.51 for thinking on and five draft tokens, with synthetic rejection sampling and adaptive verification disabled. Accuracy evals retain real block rejection but, unlike the CUDA arms, also keep adaptive verification disabled: it trims verification requests on device, which the ROCm `DeepseekV4IndexerBackend` does not support, and the engine refused to start with it enabled ([run 34651830283](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34651830283)). FP4 describes the MXFP4 experts; the checkpoint also contains MXFP8 weights.

Follow the AMD overrides in the merged [upstream recipe #968](https://github.com/vllm-project/recipes/pull/968): `VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_USE_AITER_MOE=1`, and `--moe-backend aiter`. The generic AITER selector lets vLLM pick the CK a8w4 experts, matching the DSV4-Pro MI355X recipe. The recipe pins `semianalysis_cc_traces_weka_062126` (the unfiltered corpus) via `WEKA_LOADER_OVERRIDE`. KV stays GPU-resident. Engram stayed on GPU under the upstream AMD defaults until [vllm-project/vllm#57491](https://github.com/vllm-project/vllm/pull/57491) widened the two `is_cuda()` gates to `is_cuda_alike()`. From that commit on, ROCm resolves an `EngramConfig` and `cpu_offload` defaults to on through `VLLM_PLE_CPU_OFFLOAD`, so the recipe sets `--engram-config` explicitly rather than leaning on that default. TP=2 always offloads, since the tables need 94.4 GiB per rank there; TP=4 keeps them resident, since its KV pool is not the constraint through concurrency 64. The recipe likewise trims `--max-num-batched-tokens` only above concurrency 32, to 8192 at TP=2 c64, because the sparse-attention indexer and its companion per-rank buffers grow at roughly 4.4 MiB per batched token. Where that chunk falls below six times the API-server default of 1024 sequences, `--max-num-seqs` is capped at the graph-capture shape: DSpark verifies 1+5 tokens per sequence, and at 4096 against 1024 sequences the engram projection faults during profiling. The rule in every case is to spend device memory on KV only at the concurrencies that ran short of it, leaving the validated low-concurrency settings alone. Images built before that merge still reject the option on ROCm. The MI355X launcher uses the shared HF cache and mounts this model's repository at `/ix`, and exports `INFMAX_CONTAINER_WORKSPACE=/ix` so AgentX dependencies and outputs resolve inside that mount.

**GPU validation:** The recipe uses `vllm/vllm-openai-rocm:nightly-rocm100-ac9126e58aa7bbab1856ba6593ba4d5003fea516`, a ROCm 10.0 nightly carrying vllm#58671 (paged MXFP4 sparse indexer), vllm#58655 (fused mHC Triton seams) and vllm#53492 (Gluon sparse-MLA kernel). [Run 36824408313](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/36824408313) from [#3571](https://github.com/SemiAnalysisAI/InferenceX/pull/3571) validated it across TP4 and TP2 at concurrency 1–64, and its eval-only TP4 concurrency 64 point scored GSM8K 0.9712 strict / 0.9704 flexible. Earlier sweeps qualified superseded pins, so their points do not carry onto this image: [#3420](https://github.com/SemiAnalysisAI/InferenceX/pull/3420) re-swept `nightly-rocm100-29468dde8b515031dc6d4d9d06bf0a2fa0442098`, [#3326](https://github.com/SemiAnalysisAI/InferenceX/pull/3326) qualified `nightly-rocm100-3df4ae153eb385e27b52f26c81f8edb9e20b9984`, and [run 34710937012](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34710937012) covered `nightly-eed1f3d0c6043bd494424a22443ee198dd56f657`. The merged [upstream recipe #1049](https://github.com/vllm-project/recipes/pull/1049) documents the MI355X sparse-MLA Gluon kernel, the MXFP4 sparse indexer and `--block-size 128`; the merged [#1006](https://github.com/vllm-project/recipes/pull/1006) documents the MI355X TP2 Engram offload and `--no-swa-bounded-replay`, and the merged [#968](https://github.com/vllm-project/recipes/pull/968) records the original AMD overrides and the complete InferenceX command. Follow the [AgentX procedure](./eval-agentx-procedures.md#7-run-agentx-fast-feedback-versus-canonical-evidence) for future runtime evidence; local generation and registry metadata alone are not GPU proof.

## DeepSeek-V4.1-Flash on MI300X and MI325X

`dsv41flash-fp4-mi300x-vllm-agentic-dspark` and `dsv41flash-fp4-mi325x-vllm-agentic-dspark`
copy the validated MI355X vLLM arm onto gfx942, on the ROCm 10.0 nightly
`nightly-rocm100-3df4ae153eb385e27b52f26c81f8edb9e20b9984`, and with the same
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
the MI355X arm's 16384). Concurrency is 1–32 on both. Both entries also carry smaller
layouts (MI300X TP4; MI325X TP2 and TP4) that move the Engram tables to host memory with
`--engram-config '{"cpu_offload":true}'`.

The srt-slurm single-node path mounts the checkout at `/infmax-workspace`, so AgentX runtime
directories stay out of `/workspace`. The HF cache on MI300X is node-local, so the first arm
on each node downloads 511 GB before serving. GPU sweep and eval evidence is required before
calling either arm validated.
