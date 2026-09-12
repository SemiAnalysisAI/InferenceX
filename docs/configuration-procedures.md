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
2. **Choose two kinds of sibling.** Read the same model on another SKU and another model on the target SKU. Also read the target [`runners/launch_*.sh`](../runners/) and shared [`benchmark_lib.sh`](../benchmarks/benchmark_lib.sh).
3. **Add the runtime script.** Put the single-node script under [`benchmarks/single_node/fixed_seq_len/`](../benchmarks/single_node/fixed_seq_len/). Preserve the proven sibling's env propagation, parser flags, attention/MoE backend, KV-cache dtype, graph/eager mode, cache setup, and context handling.
4. **Add the master entry.** Use [`amd-master.yaml`](../configs/amd-master.yaml) for `mi*`. Otherwise, use [`nvidia-master.yaml`](../configs/nvidia-master.yaml). Set exact `image`, `model`, `model-prefix`, `runner`, `precision`, `framework`, scenarios, and supported search spaces.
5. **Size from evidence.** Mirror proven parallelism layouts and trim unsupported ones. Latency TP rows normally start at concurrency 1. Do not copy large-memory TP/EP layouts onto a smaller SKU.
6. **Check launcher routing.** The launcher must resolve the new filename, including framework and `_mtp` suffixes. Simulate STP and MTP resolution and confirm each selected file exists.
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

## Register an srt-slurm recipe

Mapping source: [`benchmarks/multi_node/srt-slurm-recipes/RECIPES.md`](../benchmarks/multi_node/srt-slurm-recipes/RECIPES.md). Checked-in recipes: [`benchmarks/multi_node/srt-slurm-recipes/`](../benchmarks/multi_node/srt-slurm-recipes/).

1. Locate the exact upstream [NVIDIA/srt-slurm](https://github.com/NVIDIA/srt-slurm) recipe and record its commit-pinned source path.
2. Stage the YAML under the matching checked-in recipe tree. Read the closest sibling and selected cluster launcher.
3. Map source fields to the master search-space entry: resource worker counts → `num-worker`, TP/EP/DP-attention → worker topology, benchmark concurrencies → `conc-list`, and recipe path → `additional-settings: ["CONFIG_FILE=..."]`.
4. Add/update the matching [`nvidia-master.yaml`](../configs/nvidia-master.yaml) entry in the same change. Keep worker counts, TP/PP/EP/DCP/PCP, hardware, router, transfer engine, and concurrency labels synchronized.
5. For an image bump, make recipe `model.container` exactly equal master `image`. The launcher uses the master image as the container-alias key.
6. Run the recipe's documented `srtctl` validation, then generate the master key and compare every frontend label/topology field with the recipe.
7. Append the changelog entry.

Do not ship one side alone. `srtctl` reads the recipe, while matrix generation reads the master config. Recipe-only changes can mislabel results. Master-only changes do not alter the deployed recipe.

## Register an llm-d recipe

Sources: [`benchmarks/llm-d/README.md`](../benchmarks/llm-d/README.md), [`benchmarks/multi_node/llm-d/README.md`](../benchmarks/multi_node/llm-d/README.md), [`llm-d-recipes/`](../benchmarks/multi_node/llm-d-recipes/), and the current [`llmd-vllm` benchmark wrapper](../benchmarks/multi_node/dsv4_fp4_gb200_llmd-vllm-disagg.sh).

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

Sources: [`AGENTS.md#non-negotiable-benchmark-invariants`](../AGENTS.md#non-negotiable-benchmark-invariants), [MTP appendix in the model+hardware playbook](../.claude/commands/add-model-hardware.md#appendix--mtp--eagle3-spec-decoding-variant), and current [`*_mtp.sh` siblings](../benchmarks/single_node/fixed_seq_len/).

1. Confirm native MTP modules versus an external draft. For a draft, verify exact model ID, method (for example `eagle3`), and recommended speculative-token count from the model/upstream recipe.
2. Copy a working sibling for the same model and backend. Preserve its speculative config, attention backend, token count, model patches, and dependency setup.
3. Every `*_mtp.sh` must pass `--use-chat-template` to `run_benchmark_serving`. Raw prompts silently depress acceptance.
4. Size graph capture for at least `CONC * (1 + NUM_SPEC_TOKENS)`, rounded as the sibling does and capped at the framework limit (the current vLLM playbook caps at 2048).
5. Keep backend differences: do not copy CUDA-only drafter attention pins or patches into ROCm recipes.
6. Set `spec-decoding: mtp` in the relevant search-space entries and add `_mtp` launcher suffix routing. For a draft-model mode supported by the schema, use the matching generated value deliberately. Do not infer it from a filename.
7. Add script + master entry + launcher routing + changelog together.
8. Run Bash syntax and generation checks. Inspect `spec-decoding`, draft/native method, token count, chat-template use, capture range, and resolved script.

### DeepSeek-V4.1-Flash DSpark

The AgentX-only `dsv41flash-fp4-<sku>-vllm-agentic-dspark` recipes use
`vllm/vllm-openai:deepseekv41-flash-0909` at TP4 on Blackwell SKUs with native five-token DSpark,
probabilistic drafting. Throughput uses the [committed golden AL](../golden_al_distribution/dsv41flash_dspark.yaml) of 3.51 for thinking on and five draft tokens, with synthetic rejection sampling and adaptive verification disabled. Accuracy evals retain real block rejection and adaptive verification. `--engram-config '{"cpu_offload":true}'`
stores Engram embedding tables in pinned host DRAM accessed through UVA;
`kv-offloading: none` describes the separate, GPU-resident KV cache. MXFP4 expert
weights determine the recipe's `precision: fp4` label.

The GPU-specific entry points share the text-only serving script, `deepseek_v41` tokenizer and
parsers, 1M context, and the shared AgentX trace replay, power, metrics, and eval
helpers. Concurrency is 1–128. Model-runner selection and scheduler batching follow the
official single-node TP recipe defaults; graph capture covers concurrency times
the six-token DSpark verification block. The launchers mount the repository at `/ix` for this recipe so
AgentX runtime directories are not created under `/workspace`. Launcher-specific model paths and persistent caches are reused.
The recipe probes the serving port on the compute node and selects an available
port if the preferred one is occupied. Serving, replay, metrics, and eval share
that endpoint.
The GB300 launcher allows 7200 seconds for engine readiness. In [run 34504969146](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34504969146), the Rust frontend exhausted its 3600-second deadline while the engine was still capturing graphs; model loading alone took 18–23 minutes. This extends startup time without changing the benchmark duration or decoding settings.

GPU sweep and eval evidence is required before calling any recipe validated.

Source: [upstream recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash).

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

So the arm sweeps concurrency 1–4, under the 6.70x ceiling, so a trajectory replaying
near full context does not drive the batch into preemption. Buying more KV means
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
  --seq-lens 1k1k 8k1k
```

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
- Scenario `max-model-len` is derived from ISL + OSL + slack. Do not hardcode the checkpoint's full context for an 8k1k/1k8k recipe.

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

The draft `dsv41flash-fp4-mi355x-vllm-agentic-dspark` recipe extends [#2958](https://github.com/SemiAnalysisAI/InferenceX/pull/2958) to MI355X AgentX: TP4, concurrency 1–32, native five-token DSpark. Throughput uses the [committed golden AL](../golden_al_distribution/dsv41flash_dspark.yaml) of 3.51 for thinking on and five draft tokens, with synthetic rejection sampling and adaptive verification disabled. Accuracy evals retain real block rejection but, unlike the CUDA arms, also keep adaptive verification disabled: it trims verification requests on device, which the ROCm `DeepseekV4IndexerBackend` does not support, and the engine refused to start with it enabled ([run 34651830283](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34651830283)). FP4 describes the MXFP4 experts; the checkpoint also contains MXFP8 weights.

Follow the AMD overrides in [upstream recipe #946](https://github.com/vllm-project/recipes/pull/946): `VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_USE_AITER_MOE=1`, and `--moe-backend aiter_triton_mxfp4_bf16`. The recipe pins `semianalysis_cc_traces_weka_062126` (the unfiltered corpus) via `WEKA_LOADER_OVERRIDE`. KV stays GPU-resident; Engram follows upstream AMD defaults. Do not copy the NVIDIA `--engram-config` option: upstream currently rejects it on ROCm. The MI355X launcher uses the shared HF cache and mounts this model's repository at `/ix`, and exports `INFMAX_CONTAINER_WORKSPACE=/ix` so AgentX dependencies and outputs resolve inside that mount.

**GPU validation is pending:** The recipe uses `vllm/vllm-openai-rocm:deepseekv41-flash-0909`, the image referenced by the [upstream vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash?hardware=mi355x). The tag was published to Docker Hub on 2026-09-11 after AMD verification; it returned HTTP 404 in run 34466680355 before publication. Engram behavior and serving compatibility still require GPU validation. Follow the [AgentX procedure](./eval-agentx-procedures.md#7-run-agentx-fast-feedback-versus-canonical-evidence) for runtime evidence; local generation and registry metadata are not GPU proof.
