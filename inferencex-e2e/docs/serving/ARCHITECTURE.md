# Serving path: architecture

## 1. What this is

This branch is upstream InferenceX plus one extra launch path, `serving`. It runs VNG serving-YAML
benchmarks (AgentX, tau2, LiveKit, the E2/E3 prefix traces, and the synthetic ISL/OSL/cache grid at
fixed concurrency or under SLO search) on the VNG Slurm nodes. Upstream does the scheduling, image
staging, harness install and result processing; the VNG code turns a serving YAML into matrix
points and starts the engine, router and storage containers inside one right-sized Slurm job.
Design and decisions: `docs/superpowers/specs/2026-10-25-vng-serving-refactor-design.md` (repo root).

## 2. One benchmark point, end to end

```
utils/serving/dispatch.sh configs/serving/<x>.yaml
  validates locally with the same code as CI, requires a pushed HEAD, runs gh workflow run serving.yml
.github/workflows/serving.yml
  plan job: python -m infx.serving.to_matrix -> one point per (scenario variant x conc)
  bench job: benchmark-tmpl.yml per point, with config=<row> and serving=<SERVING_JSON>
.github/workflows/benchmark-tmpl.yml -> python -m infx.launch run          (upstream)
  resolve_cluster(runner) -> clusters.hcm-<node>
  policy.launch_path -> SERVING, because SERVING_JSON is set
infx/launch/drivers/serving.py run()
  1. validate SERVING_JSON (every bad field reported at once)
  2. stage_image for engine, DCGM, router, storage
  3. submit_batch(job.sh, gpus=N, time limit); cluster salloc-args right-size it
  4. stream the log, read the state, return its exit code
benchmarks/single_node/serving/job.sh       (inside the allocation)
  a. pick free ports on the node
  b. DCGM exporter step (--overlap)
  c. [storage] sidecar.sh storage (--overlap), e.g. the Mooncake master
  d. [router]  sidecar.sh router  (--overlap): sglang-router, vllm-router or a Dynamo frontend
  e. engine step: bench.sh in the engine container
benchmarks/single_node/serving/bench.sh
  client prepare (dataset fetch + sha256, model download), engine replicas,
  infx.bench wait on each replica, router /v1/models wait, client run
infx/serving/client.py run
  agentx profile --config results/scenario.yaml ..., then the upstream result CLIs
  (or infx.serving.sweep_result for search points), then writes the legacy label
```

| Step | Where |
|---|---|
| Validate and dispatch | `inferencex-e2e/utils/serving/dispatch.sh` |
| Points | `infx/serving/to_matrix.py:56` (`build`), rows at `:22`, `SERVING_JSON` at `:32` |
| Serving YAML schema | `infx/serving/schema.py:123` (`ServingSpec`), `:254` (`load_serving`) |
| Scenario render | `infx/serving/scenarios.py:173` (`render`) |
| Routing hook | `infx/launch/policy.py:87` |
| Driver | `infx/launch/drivers/serving.py:89` (`run`): staging `:111`, `submit_batch` `:116`, logs `:123`, state `:126` |
| Job env | `infx/launch/drivers/serving.py:62` (`job_env`) |
| Port picking | `benchmarks/single_node/serving/job.sh:16` |
| Storage / router steps | `job.sh:63`, `job.sh:98` |
| Engine step | `job.sh:109` |
| Replicas, one GPU slice each | `bench.sh:57` |
| Engine health wait | `bench.sh:76` |
| Router wait | `bench.sh:81` |
| Client prepare / run | `bench.sh:25`, `bench.sh:106`; `infx/serving/client.py:58`, `:72` |

If the engine or router dies before it is ready, the wait fails and prints the tail of
`server.log` or `router.log`. Cancelling the workflow cancels the Slurm job (upstream), and the
EXIT trap in `job.sh` stops the sidecar steps.

## 3. Where VNG code lives

Every VNG file has `serving` in its path, so `git grep -l serving` or
`git diff upstream/main --stat` lists all of it.

| File | Job |
|---|---|
| `benchmarks/single_node/serving/datasets/lock.yaml` | pinned file datasets, with `entries` |
| `infx/serving/datasets.py` | read the lock, fetch and sha256-check one dataset |
| `infx/serving/slo.py` | parse `metric:stat:op:threshold;...` |
| `benchmarks/single_node/serving/scenarios/*.yaml` | the six base AIPerf configs |
| `infx/serving/scenarios.py` | base + overrides / grid point / search / slo → final config and legacy label |
| `configs/runners.yaml` (VNG block) | `hcm-*` runner labels and cluster records |
| `infx/serving/schema.py`, `render.py` | serving YAML model and validation; command rendering |
| `infx/serving/to_matrix.py`, `utils/serving/dispatch.sh`, `.github/workflows/serving.yml` | points; local validation and dispatch; the workflow |
| `infx/serving/client.py`, `sweep_result.py` | in-container client; search result writer |
| `benchmarks/single_node/serving/{job,bench,sidecar}.sh` | allocation steps; engine step; router/storage sidecar |
| `infx/launch/drivers/serving.py` | the launch path |
| `configs/serving/*.yaml` | seven examples, one per pattern |
| `.agents/skills/{serving-bench,serving-runner-setup,result-report}` | skills |

Paths are relative to `inferencex-e2e/` except the workflow and the skills.

Upstream files touched (the only places a merge can conflict):

1. `configs/runners.yaml`: the VNG labels after the last upstream label, the four `hcm-*` records at the end of `clusters:`.
2. `infx/launch/policy.py:63`: `SERVING = "serving"` in `LaunchPath`; `:87-88`: return it when `SERVING_JSON` is set.
3. `infx/launch/drivers/__init__.py`: `serving` in the driver import, and `LaunchPath.SERVING: Route("slurm", serving.run)` in `ROUTES`.
4. `.github/workflows/benchmark-tmpl.yml:8`: the optional `serving` input; `:99`: `SERVING_JSON: ${{ inputs.serving }}`.
5. `.github/workflows/benchmark-tmpl.yml:311`: "Validate agentic result" is skipped for search points, which have no top-level `profile_export_aiperf.json`.

## 4. VNG Slurm

Facts measured on 2026-10-25: one cluster `test-cluster`, Slurm 26.11.0rc1, `select/cons_tres` with
`CR_CORE_MEMORY`, backfill, no preemption, pyxis on every node.

### One cluster record per node

There is no usable shared filesystem: `/shared` is a 492G NFS at
96%, and home directories are not shared. So each runner lives on its GPU node, and the HF cache
and squash files sit on that node's local disk. Each node is its own `clusters.hcm-<node>`, pinned
with `--nodelist` in `salloc-args`:

| Cluster | Node | Partition | GPUs | HF cache / squash |
|---|---|---|---|---|
| `hcm-hgx01` | hgx-h200-01 | `hardware-hcm` | 4 (gres declares 4 of 8; raise when infra fixes it) | `/mnt/hf_hub_cache`, `/mnt/containers` |
| `hcm-hgx02` | hgx-h200-02 | `hardware-hcm` | 8 | `/mnt/hf_hub_cache`, `/mnt/containers` |
| `hcm-rtx01` | hoanq3-rtx6000x2-slurm-1 | `rtx6000pro-hcm` | 2 | `/data/hf_hub_cache`, `/data/containers` |
| `hcm-rtx02` | hoanq3-rtx6000x2-slurm-2 | `rtx6000pro-hcm` | 2 | `/data/hf_hub_cache`, `/data/containers` |
| `han-h200-01` | hoanq3-h200-1x-han-1 | `test` | 1 | `/mnt/hf_hub_cache`, `/mnt/containers` |
| `han-h200-02` | hoanq3-h200-1x-han-2 | `test` | 1 | `/mnt/hf_hub_cache`, `/mnt/containers` |
| `han-h200-04` | hoanq3-h200-1x-han-4 | `test` | 1 | `/data/hf-cache`, `/data/containers` |

The serving YAML `runner:` takes a cluster label (`cluster:hcm-hgx02`), a pool label (`h200-hcm`,
`rtx6000pro-hcm`, `h200-han`) or one slot.

### Runner convention

Every pool uses the same convention: user `stackops` (enroot credentials and HF token in its
home), name `<cluster-id>_<NN>` with exactly one `cluster:<cluster-id>` label, directory
`<local disk>/gharunners/<cluster-id>/gharunnerNN/`, GitHub labels `slurm,<sku>,<pool>,cluster:<cluster-id>` (the YAML's `runner:` goes straight into `runs-on`), run by systemd. Slots per node = the most jobs you want at once
(hgx 4, rtx 2); Slurm queues a job that finds too few free GPUs. Setup and preflight:
`.agents/skills/serving-runner-setup`.

### Right-sized jobs, never exclusive

The nodes are shared with the team. A job asks for its GPUs
(`gres`), 16 CPUs per GPU and a per-GPU memory share (240000M on hgx, 110000M on rtx), and the
rest of the node stays free. These live in `salloc-args` because upstream `submit_batch` adds no
CPU or memory flags of its own; it passes `salloc-args` to `sbatch`.

### Ports picked at run time

Pyxis containers share the host network, and other jobs can run on the
same node, so fixed port bases would collide. `job.sh:16` picks each port free on the node from
20000-32767, below the Linux ephemeral range. This is why VNG has its own path instead of
upstream `srtctl`: srtctl uses fixed port bases and relies on exclusive nodes.

### Squash import

Squash records use `import: submit-host` and `single-node-import: true`. The runner sits on the node where the
job runs, so the image is imported where it is used.

## 5. Scenarios

Six base files in `benchmarks/single_node/serving/scenarios/`:

| Family | Dataset |
|---|---|
| `agentx` | public weka `062126` |
| `tau2` | lock `tau2-cs` |
| `livekit` | lock `livekit-111868-v1.1-warm` |
| `shared-prefix-e2` | lock `shared-prefix-e2` |
| `hidden-prefix-e3` | lock `hidden-prefix-e3` |
| `synthetic` | synthetic, with shared prefix |

A serving YAML picks a family and may set `overrides`. The keys allowed are `duration`, `dataset`
(same kind only: public to public, lock to lock), `max_context_length` (`null` removes the cap),
`generation` and `trace_idle_gap_cap`. Any other key fails validation, so cache-bust, warmup and seed
cannot change without a code review.

The synthetic family takes `workload:` instead:

```yaml
scenario: synthetic
workload: {isl: [8k, 32k], osl: ['512', 1k], cache: [0, 50, 90], search: true}
slo: "inter_token_latency:p90:lt:10"   # optional; replaces the default SLO
conc: 1                                # search finds the concurrency itself
```

Every ISL x OSL x cache point becomes one job in the same workflow run. `search: false` runs each
point at the `conc` values instead.

Old scenario names map to the new syntax like this:

| Old `scenario:` | New |
|---|---|
| `agentx-cc-062126` | `scenario: agentx` |
| `agentx-cc-062126-smoke` | `scenario: agentx` + `overrides: {duration: 180}` |
| `agentx-cc-062126-256k` | `scenario: agentx` + `overrides: {dataset: semianalysis_cc_traces_weka_062126_256k}` |
| `agentx-cc-062126-1m` | `scenario: agentx` + `overrides: {max_context_length: null}` |
| `agentx-cc-062126-topp95` | `scenario: agentx` + `overrides: {generation: {top_p: 0.95, temperature: 1.0}}` |
| `custom-agentic-tau2[-smoke]` | `scenario: tau2` (smoke: + `overrides: {duration: 180}`) |
| `custom-agentic-livekit111868-v1` / `-v2` | `scenario: livekit` + `overrides: {dataset: livekit-111868-v1}` / `v2` |
| `custom-agentic-livekit111868-smoke` | `scenario: livekit` + `overrides: {dataset: livekit-111868-v2, duration: 180}` |
| `agentx-shared-prefix-e2` | `scenario: shared-prefix-e2` |
| `agentx-hidden-prefix-e3[-smoke]` | `scenario: hidden-prefix-e3` (smoke: + `overrides: {duration: 180}`) |
| `sweep-isl<I>-osl<O>-cache<C>` | `scenario: synthetic` + `workload: {..., search: true}` + `conc: 1` |
| `fixed-isl<I>-osl<O>-cache<C>` | `scenario: synthetic` + `workload: {..., search: false}`, same `conc` list |
| `IX_SLO` in `env:` | `slo: "<same tuples>"` |

`infx/tests/serving/test_scenarios.py` checks parity: for each of the 20 legacy files kept as
fixtures in `infx/tests/serving/legacy_scenarios/`, base plus overrides renders the same YAML
document, comments aside.

LiveKit warm is a dataset, nothing more. aiperf TTFT therefore includes the 601 dial warm-up
requests (`max_tokens 1`), so avg and p50 TTFT read lower than the warm-up-filtered numbers in older
reports.

## 6. Results

Artifact names are upstream's (`bmk_agentic_<RF>`, `agentic_<RF>`). Exp-names keep the old rule:
`<name>_c<conc>`, `<name>_search`, and `<name>-<label>` for grid points.

`results/` in each artifact holds `serving.yaml`, `scenario.yaml`, `server[.N].log`,
`server_command[.N].txt`, `router.log`, `router_command.txt`, `storage.log`,
`storage_command.txt`, `dcgm.log`, `benchmark.log`, `benchmark_command.txt`, `dataset.txt` for
file datasets, and `aiperf_artifacts/`.

The aggregate JSON's `scenario` field holds the legacy label (`agentx-cc-062126-smoke`,
`sweep-isl8k-osl1k-cache50`, ...). Search points carry a `state`: `ok`, `cliff`,
`no_pass_in_range`, `no_failure_in_range`, `max_iterations` or `no_data`.

`.agents/skills/result-report` turns run links into `report.md` / `data.json`. It reads the
scenario from the aggregate first (`scenario_of` in `scripts/extract.py`).

## 7. Syncing upstream

```bash
git fetch upstream && git merge upstream/main
```

Conflicts can only land in the five touch points of section 3. After the merge, from
`inferencex-e2e/`:

```bash
git submodule update --init utils/srt-slurm
uv run --no-project --python 3.12 --with pytest --with ./utils/srt-slurm python -m pytest infx/tests/serving infx/tests/launch -q
```

Then check that these still exist with the same shape: `infx.bench.agentic.venv` (`Runtime`,
`bootstrap`), the `infx.bench wait` CLI, `SlurmBackend.stage_image` / `submit_batch` /
`stream_logs` / `state`, and the private `infx.results.agentic` helpers imported by
`infx/serving/sweep_result.py` (`_env_bool`, `_env_int`, `_gpu_shape`,
`_optional_component_metadata`, `_validate_kv_offload_env`). The serving tests fail if any of
them moves.

## 8. Not done here

- Multi-node PD disaggregation (upstream `srt-multi` and srt recipes).
- Single-node PD through `roles:` in the serving driver.
- Features still on unmerged branches: adaptive search on trace scenarios
  (`bench/gemma4-livekit-dyn-swa-arms`), LiveKit open-loop (`bench/gemma4-livekit-voice-openloop`),
  late-joining replicas / `scale_up` (`feat/serving-late-replica-mooncake`), `host_cpus` /
  `host_mem_gb` (`feat/serving-dynamo-vllm-replicas`), warm-up-filtered TTFT (`livekit_ttft.py`).
- The VNG fix that counts Mooncake `storage_*` hits as external cache hits in
  `infx/results/agentic/backends/sglang.py` is not ported. Aggregates of L3 runs under-report
  `server_external_cache_hit_rate`; `result-report` computes the tiers from raw metrics and is not
  affected. Candidate for an upstream PR.
- The other 46 legacy serving YAMLs stay on `vng-benchmark`; convert them with the table in
  section 5 when needed.
