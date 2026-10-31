# VNG serving refactor onto upstream InferenceX

Date: 2026-10-25. Branch: `refactor/vng-benchmark`, cut from `upstream/main` at `cfb7bdd0c`.
`vng-benchmark` (last upstream sync 2026-09-16, 148 own commits) stays untouched as the archive.

## Goal

Run every VNG benchmark (AgentX, fixed ISL/OSL, ISL/OSL search, custom file datasets) on
current upstream InferenceX. Keep the team's serving YAML as the entry point. Keep the VNG
custom code small and isolated, with a flow a new team member can follow end to end.
Dashboard ingest is out of scope.

## Why port instead of merge

A trial merge of `upstream/main` into `vng-benchmark` produced 84 conflicts. The core of the
VNG path was not modified upstream. It was removed:

| VNG depends on | Upstream now |
|---|---|
| `benchmarks/benchmark_lib.sh` (VNG-patched) | replaced by `infx.bench` (#3610, 2026-10-06) |
| `utils/aiperf` submodule (VNG fork) | `agentx-harness` from PyPI |
| `runners/launch_*.sh` | `python -m infx.launch run` + `infx/clusters` + `configs/runners.yaml` `clusters:` |
| top-level `benchmarks/ configs/ infx/ runners/ utils/` | moved under `inferencex-e2e/` (#3525, 2026-09-27) |

The VNG additions were almost entirely new files (+40.6k / -86 lines), so they port cleanly
onto the new layout.

## Decisions

1. **Serving YAML stays the front end.** The team writes `serve:` as a raw command, plus
   `router:`, `replicas:`, `scenario:`, `conc:`.
2. **Execution goes through a VNG `serving` driver inside `infx.launch`, not srtctl.**
   srtctl binds 40+ fixed ports (`srtctl/ports.py`). Pyxis shares the host network
   namespace. VNG nodes are shared with the team and must never be allocated exclusively,
   so the only safe way to place ports is to pick free ones on the node at run time.
   The driver uses upstream's public backend API (`stage_image`, `submit_batch`), which
   already applies right-sized gres/cpu/mem, `salloc-args` and cancellation.
   Multi-node PD disaggregation later uses upstream's `srt-multi` path with srt recipes;
   single-node PD can be added to the serving driver as `roles:`.
3. **One cluster record per node, identical runner slots, one runner convention.**
4. **Scenarios: six base files plus whitelisted overrides.** 63 files become 6.
5. **Keep `scenario: inferencex-agentx-mvp` in the trace scenarios.** Verified against
   harness 1.0.12 and 1.1.0: the `mooncake_trace` loader mismatch is an overridable
   violation and every file sets `unsafeOverride: true`. The only effect is
   `submission_valid=false`, which nothing in this repo reads. Removing the line breaks
   loading (`agentic cache warmup requires the agentic_replay timing mode`) and changes
   context-overflow accounting.

## Layout

Every VNG file has `serving` in its path. `git grep -l serving` lists the whole custom surface.

New files (never conflict on upstream sync):

```
.github/workflows/serving.yml                 # serving YAML -> matrix -> benchmark-tmpl.yml
inferencex-e2e/
  configs/serving/*.yaml                      # team serving YAMLs (was serving/)
  benchmarks/single_node/serving/
    job.sh                                    # runs inside the Slurm allocation
    scenarios/{agentx,tau2,livekit,shared-prefix-e2,hidden-prefix-e3,synthetic}.yaml
    datasets/lock.yaml
  infx/serving/                               # schema, scenario render, to_matrix, slo, datasets, client, sweep result
  infx/launch/drivers/serving.py
  infx/tests/serving/
  utils/serving/dispatch.sh
  docs/serving/ARCHITECTURE.md
.agents/skills/{serving-bench,result-report,serving-runner-setup}/  (+ .claude/skills symlinks)
```

Upstream files touched (additions only, under 15 lines total):

| File | Change |
|---|---|
| `.github/workflows/benchmark-tmpl.yml` | input `serving`, env `SERVING_JSON` |
| `inferencex-e2e/infx/launch/policy.py` | `LaunchPath.SERVING`; `launch_path` returns it when `SERVING_JSON` is set |
| `inferencex-e2e/infx/launch/drivers/__init__.py` | one `ROUTES` entry |
| `inferencex-e2e/configs/runners.yaml` | VNG labels and `clusters:` records (data) |

`e2e-tests.yml` is not touched. Upstream edits it often, so VNG dispatch uses its own `serving.yml`.

Not ported: the 17 `glm5.2*_h200_sglang.sh` arm scripts and their `nvidia-master.yaml`
entries, the bash launchers and `reap_orphan_aiperf.sh`, the `benchmark_lib.sh` patches, the
aiperf submodule, `remote-bench-e2e.yml` (BYO endpoint), `check_scenario_golden.py`,
`gen_sweep_scenarios.py`, `sweep_dispatch.sh`, and the old `docs/superpowers` design docs
(they stay on `vng-benchmark`).

## Flow of one benchmark point

```
utils/serving/dispatch.sh configs/serving/<x>.yaml
  validate locally (same code as CI), require pushed HEAD, gh workflow run serving.yml
serving.yml
  infx.serving.to_matrix -> rows (serving YAML x scenario variant x conc) + SERVING_JSON per row
  calls benchmark-tmpl.yml per row (artifact names unchanged: bmk_agentic_<RF>, agentic_<RF>)
benchmark-tmpl.yml -> python -m infx.launch run            (upstream, unchanged)
  resolve_cluster(RUNNER_NAME) -> clusters.<id>
  policy.launch_path -> SERVING
drivers/serving.py
  1. parse SERVING_JSON (pydantic, report every bad field before any work)
  2. backend.stage_image for engine, router, storage images
  3. backend.submit_batch(job.sh, gpus=N, time_limit)
  4. stream the log, read the state, return the exit code; outputs are already in the workspace
benchmarks/single_node/serving/job.sh (static bash, reads env only)
  a. pick free ports on the node for engine replicas, router, storage
  b. [storage] srun --overlap --container-image (e.g. Mooncake master)
  c. engine x replicas: srun --overlap, one GPU slice per replica, hf download, raw `serve:`
  d. [router] srun --overlap (sglang-router, vllm-router, Dynamo frontend)
  e. python -m infx.bench wait                              (upstream)
  f. python -m infx.serving.client
  g. trap EXIT stops the sidecar steps
```

`job.sh` contains no per-pool branches. Partition, node pinning, paths and squash dir all
come from the cluster record. Expected size is about 120 lines, replacing 870+ lines across
three launchers.

### Client (`infx/serving/client.py`)

- Writes the rendered scenario to `results/scenario.yaml` and runs
  `agentx profile -f results/scenario.yaml --url --model --tokenizer --concurrency
  --server-metrics --output-artifact-dir`. The harness venv comes from upstream
  `infx.bench.agentic.venv`.
- File datasets: resolve the lock key, download and verify sha256, export `IX_CUSTOM_DATASET_FILE`.
- Non-search scenarios: run upstream's result CLIs in the order `infx/bench/agentic/run.py`
  uses (`infx.results.agentic.process_agentic_result`, `analyze_benchmark_distributions`,
  `validate_agentic_result --failed-request-threshold 0.10`).
- Search scenarios: skip upstream validation and run `infx.serving.sweep_result` (ported
  `infx/results/sweep.py`; states `ok`, `cliff`, `no_pass_in_range`, `no_failure_in_range`,
  `max_iterations`, `no_data`).
- Writes the legacy scenario label (for example `agentx-cc-062126-smoke`,
  `sweep-isl8k-osl1k-cache50`) into the aggregate JSON's `scenario` field. `result-report`
  reads that field first.
- Depends on upstream CLI modules only, never on private functions. No upstream `infx.bench`
  edits.

### Failure handling

- The engine or router dies before it is ready: the wait step fails fast and prints the tail
  of `server.log` / `router.log`.
- The workflow is cancelled: upstream's launch callback cancels the Slurm job, and `job.sh`'s
  trap stops the sidecars.
- Logs stay where `result-report` expects them: `results/` holds `serving.yaml`,
  `server.log`, `router.log`, `storage.log`, `benchmark.log`, `aiperf_artifacts/`.

## Slurm and runners

Facts measured on 2026-10-25 (`slurm-client`, tiny `srun` probes):

- One cluster `test-cluster`, Slurm 26.11.0rc1, `select/cons_tres` + `CR_CORE_MEMORY`,
  backfill, no preemption, `AccountingStorageEnforce=none`. Pyxis on every node.
- Partitions: `hardware-hcm` (hgx-h200-01, -02; 8×H200, but hgx-01 declares only 4 GPUs in
  gres), `rtx6000pro-hcm` (2 nodes × 2 RTX PRO 6000), `test` (HAN 1×H200), `h100-han` (down).
- No usable shared filesystem: `/shared` is 492G NFS at 96%, and home directories are not
  shared. Every runner must live on its GPU node and use node-local paths.
- hgx runners: `stackops`, systemd, 4 identities per box named by flavor. rtx runners:
  `./run.sh` by hand as user `thanglq5` in `/data/thanglq5/runner-rtx0N`. HAN runners: offline.

### Cluster records

One `clusters.<id>` per node, pinned by `salloc-args`:

```yaml
labels:
  cluster:hcm-hgx01: [hcm-hgx01_00, hcm-hgx01_01, hcm-hgx01_02, hcm-hgx01_03]
  cluster:hcm-hgx02: [hcm-hgx02_00, hcm-hgx02_01, hcm-hgx02_02, hcm-hgx02_03]
  cluster:hcm-rtx01: [hcm-rtx01_00, hcm-rtx01_01]
  cluster:hcm-rtx02: [hcm-rtx02_00, hcm-rtx02_01]
  h200-hcm: [hcm-hgx01_00, ..., hcm-hgx02_03]
clusters:
  hcm-hgx01:
    gpus-per-node: 8
    available-cpu-dram-mib: <measured>
    scheduler: slurm
    slurm:
      partition: hardware-hcm
      exclusive: false
      gres: "gpu:nvidia_h200:{gpus}"
      cpus-per-gpu: 16
      salloc-args: [--nodelist=hgx-h200-01]
      volumes: {hf-hub-cache: {path: /mnt/hf_hub_cache}}
      squash: {dir: /mnt/containers, import: compute}
```

A job gets its GPUs, `cpus-per-gpu` per GPU, and a GPU-proportional share of host memory.
The rest of the node stays available to the team. Serving YAML `runner:` accepts a cluster
label (`cluster:hcm-hgx01`), a pool label (`h200-hcm`) or a slot name.

Exact values (`available-cpu-dram-mib`, `cpus-per-gpu`, squash `visibility`/`import`) are
settled during implementation against the record schema in `infx/clusters/slurm.py`.

### Runner convention (all pools)

1. User `stackops`. Enroot credentials and the HF token live in its home.
2. Name `<cluster-id>_<NN>`, exactly one `cluster:<cluster-id>` label.
3. Directory `<large-local-disk>/gharunners/<name>/` (`/mnt/...` on hgx, `/data/...` on rtx).
4. systemd (`svc.sh install stackops`). No tmux, no hand-run `run.sh`.
5. Slots per node = maximum concurrent jobs wanted (hgx 4, rtx 2). Slurm queues jobs that
   find too few free GPUs.

### Skill `serving-runner-setup`

A thin layer over upstream `utils/runner_setup/RUNNER_SETUP.md`:

1. Per-node preflight through a 1-CPU `srun`: runner user has a passwd entry; pyxis starts a
   container; enroot paths are writable with free space; local disk has room for HF cache and
   squash; Slurm NodeName matches slurmd; gres GPU count matches the hardware.
2. Install listeners with upstream `setup.sh` (`REPO_URL=https://github.com/vngcloud/InferenceX`,
   labels `slurm,<sku>,cluster:<id>`) as `stackops`, then systemd.
3. Add labels and the `clusters:` record from the template.
4. Smoke: dispatch `agentx` with `overrides: {duration: 180}`.
5. Pool notes: HAN `/` at 99% with enroot paths on `/`, slurmd name mismatch on `han-3`;
   hgx-01 gres declares 4 GPUs; rtx listeners move to `stackops`; `/shared` at 96%.

HAN returns only after the infra side passes the preflight. No code works around it.

## Scenarios

Six base files in `benchmarks/single_node/serving/scenarios/`:

| File | Dataset | Replaces |
|---|---|---|
| `agentx.yaml` | public weka `062126` | `agentx-cc-062126{,-smoke,-256k,-1m,-topp95}` |
| `tau2.yaml` | lock `tau2-cs` | `custom-agentic-tau2{,-smoke}` |
| `livekit.yaml` | lock `livekit-111868-v1.1-warm` | `custom-agentic-livekit111868-*` |
| `shared-prefix-e2.yaml` | lock `shared-prefix-e2` | `agentx-shared-prefix-e2` |
| `hidden-prefix-e3.yaml` | lock `hidden-prefix-e3` | `agentx-hidden-prefix-e3{,-smoke}` |
| `synthetic.yaml` | synthetic with shared prefix | 48 `fixed-*` / `sweep-*` files |

`datasets/lock.yaml` keeps its five keys and adds `livekit-111868-v1.1-warm` (revision
`6e0934c957565a42d52cafac096ec758a4302355`, `trace.v1.1-append-ideal-warm.jsonl`, sha256
`e26542392a202764849c05018f2f8ea3f9574277ffa3400615f0b4df23b0eb2f`, from
`bench/gemma4-livekit-dyn-swa-arms`). Lock entries gain an optional `entries` field, so a
dataset override carries its own entry count (v1: 1217, v1.1-warm: 671).

LiveKit warm is a dataset only: no `livekit_ttft.py`, no `*-plan` files. TTFT from aiperf
therefore includes the 601 warm-up requests (`max_tokens 1`), and its avg/p50 read lower
than the filtered numbers of earlier reports.

### Variants in the serving YAML

```yaml
scenario: agentx
overrides: {duration: 180}
```

`overrides` is a whitelist: `duration`, `dataset` (same kind only: public→public,
lock→lock), `max_context_length` (`null` removes the cap), `sampling`, `trace_idle_gap_cap`.
Any other key fails validation in `dispatch.sh`, so cache-bust, warmup and seed cannot drift
without review.

### Synthetic grid in one run

```yaml
scenario: synthetic
workload: {isl: [8k, 32k, 64k, 128k], osl: [512, 1k], cache: [0, 50, 90], search: true}
slo: {ttft_avg_ms: 8000, itl_p90_ms: 50}   # default; replaces the IX_SLO env var
conc: 1                                     # ignored when search: true
```

The cartesian product becomes one job per point in one workflow run. `search: false` is the
fixed-concurrency path driven by `conc`.

### Continuity

- Exp-names and the aggregate `scenario` label keep the legacy strings
  (`sweep-isl8k-osl1k-cache50`, `agentx-cc-062126-smoke`, ...). ARCHITECTURE.md carries the
  old→new mapping.
- Parity test: for every legacy file still referenced by a serving YAML, rendering base plus
  overrides equals the legacy file after `yaml.safe_load` (comments ignored). The old files
  are not ported until this passes.

## Skills and docs

- **`serving-bench`**: rewritten for the new paths, the six families plus `overrides`/`workload`/`slo`,
  the new runner table, and an old-scenario→new-syntax table. It keeps the ask-before-guess
  rule, the never-change-flags-on-crash rule and the single `dispatch.sh` command.
- **`result-report`**: no code change expected. Artifact names and the `aiperf_artifacts/`
  layout are unchanged upstream. The one break (the scenario regex on `--config scenarios/X.yaml`)
  is covered by the aggregate `scenario` label. Add one fixture for a new-style run.
- **`docs/serving/ARCHITECTURE.md`**:
  1. One-line summary.
  2. End-to-end flow with file:line.
  3. Where VNG code lives and the upstream touch points.
  4. VNG Slurm: per-node clusters, slots, right-sized allocation, run-time ports.
  5. Scenarios and the old→new mapping.
  6. Results and `result-report`.
  7. Upstream sync procedure: merge `upstream/main`; conflicts are limited to the four
     touched files; what to re-check afterwards.
  8. Out of scope / later.

  Run `humanizer` on it before committing.

## Testing

- Unit (`infx/tests/serving/`):
  - schema validation, including the overrides whitelist;
  - `SERVING_JSON` → env;
  - `job.sh` command rendering against golden files;
  - `policy.launch_path` picks `SERVING`;
  - client argv for the search and non-search paths;
  - sweep result states;
  - scenario parity.
- Live:
  - one smoke per node;
  - router + 2 replicas on an hgx;
  - two concurrent jobs on one hgx (ports);
  - one parity run of the same serving YAML (agentx smoke) on `vng-benchmark` and on this
    branch, metrics within run-to-run noise.

### Release gates (2026-10-26): passed

Each gate replays an old `vng-benchmark` run at one concurrency on `gn-benchmark`, branch
`refactor/vng-benchmark`. Metrics come from the `bmk_*` result JSON.

| Gate | New run | Old run | Result |
|---|---|---|---|
| Gemma4 vLLM TP4 LiveKit c50 | 40207508684 | 39717833048 | 8799 vs 8825 req, 7.32 vs 7.34 req/s, ITL p90 40.3 vs 39.5 ms, TTFT p90 431 vs 398 ms |
| Gemma4 SGLang TP4 LiveKit c50, local `.sqsh` image | 40208879855 | 39718341220 | 9425/9975 vs 9430/9980 ok/total, 220.4 vs 220.5 out tok/s, cache hit 0.925 both, same KV pool |
| GLM-5.3-Flash agentx1m c24, router | 40207499165 | 39889534365 | 703 vs 667 out tok/s, ITL p90 36 vs 39 ms, cache hit 0.962 vs 0.957, 266 failed requests in both, same KV pool |
| Qwen3.8-27B SLO search 8k/512/cache50, 1 GPU | 40208237757 | earlier HAN search | `state=ok`, `max_conc=40` in both, binding ITL p90 < 50 ms, measured cache hit 0.473 |

All runs are in `vngcloud/InferenceX`. Every gate stays within run-to-run noise, so the refactor is
ready for cutover step 4. Known difference: the result JSON no longer has `power_valid` or
`power_metric_schema_version`, and the run uploads no `gpu_metrics_*` artifact. Upstream dropped
that path; per-GPU power and energy stay in aiperf's `gpu_telemetry_export.jsonl`.

## Cutover

1. Port and unit-test on `refactor/vng-benchmark`.
2. Register the new runner slots next to the old ones on the same nodes. Slurm shares the GPUs.
3. Live tests above.
4. The team switches. Running `bench/*` branches finish on `vng-benchmark`.
5. Remove the old runners, including the two rtx runners under `thanglq5`. `vng-benchmark` becomes the archive.

Pushing this branch and renaming the main working branch need explicit approval first.

## Out of scope (later)

- Multi-node PD disaggregation (upstream `srt-multi` + srt recipes).
- Single-node PD via `roles:` in the serving driver.
- Features on unmerged branches:
  - adaptive search on trace scenarios (`bench/gemma4-livekit-dyn-swa-arms`);
  - LiveKit open-loop (`bench/gemma4-livekit-voice-openloop`);
  - late-joining replicas, `scale_up` (`feat/serving-late-replica-mooncake`);
  - `host_cpus` / `host_mem_gb` (`feat/serving-dynamo-vllm-replicas`);
  - warm-up-filtered TTFT (`livekit_ttft.py`).
- Remote/BYO endpoint benchmarking (`remote-bench-e2e.yml`).
- The HAN 1×H200 pool until its infra passes preflight.
- Dashboard ingest.
- Upstream PR for the aiperf `mooncake_trace` admit patch (`faf3f8b3`), optional.
