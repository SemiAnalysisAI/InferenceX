# Agent Operations Reference

Read only the section relevant to the current task.

## Translation terminology

Write natural technical Chinese used by ML infrastructure engineers. Preserve model names, hardware SKUs, framework names, flags, CLI identifiers, and environment variables. Clarify acronyms in English on first use where useful.

| English | Chinese |
|---|---|
| benchmark | 基准测试 |
| image (Docker) | 镜像 |
| config / configuration | 配置 |
| single-node / multi-node | 单节点 / 多节点 |
| speculative decoding | 投机解码 |
| inference | 推理 |
| throughput | 吞吐量 |
| latency | 延迟 |
| prefill / decode | 预填充 / 解码 |
| disaggregated (serving) | 分离式（推理） |
| expert parallelism | 专家并行 |
| sweep | 扫描 |
| launcher | 启动器 |
| artifact | 产物 |
| evaluation / eval | 评估 |

## Sweep labels and reuse

Primary labels, modifiers, canary, fail-fast, and the fork dispatch path are owned by [PR primary and modifier labels](../inferencex-e2e/docs/ci-procedures.md#pr-primary-and-modifier-labels) and [canary and fail-fast semantics](../inferencex-e2e/docs/ci-procedures.md#canary-and-fail-fast-semantics). Apply exactly one primary label, normally `full-sweep-fail-fast`.

Sweeps do not trigger while a PR has merge conflicts. For `inferencex-e2e/perf-changelog.yaml` conflicts, run `python3 -m infx.workflows.prepare_perf_changelog_merge resolve-conflict` while the conflict stages are present, as in [changelog conflict recovery](../inferencex-e2e/docs/ci-procedures.md#changelog-conflict-recovery). It keeps main's bytes and re-appends only the PR's entry. Never hand-merge, 3-way merge, or reformat the changelog.

Pushes to `main` never run a sweep: `.github/workflows/run-sweep.yml` is PR-only. A push that changes `inferencex-e2e/perf-changelog.yaml` runs `.github/workflows/merge-ingest.yml` (Merge Ingest). Its single `ingest` job validates the merged PR's authorized `/use <run_id>` (or legacy `/reuse-sweep-run`) command and source run, then dispatches ingest of that reused PR sweep. Without valid reuse authorization the run fails, and nothing is benchmarked or ingested. `[skip-sweep]` only skips PR benchmark setup; changelog validation and reuse authorization checks still run, and Merge Ingest ignores it.

Reuse is rejected while the PR currently carries `evals-only` or `agentx-fast`; this is checked when `/use` is acknowledged and at merge (and by `merge_with_reuse`). On a push these labels skip the reuse gate, so a primary label starts a fresh sweep. Source-run label history is not inspected, so pin only runs produced without those labels. Staging (`/stage-results`, also triggered by `/use <run_id>`) still requires a primary label on the PR and a source run created while one was applied. See `.github/workflows/README.md` and `uv run --project inferencex-e2e --extra workflows python -m infx.workflows.merge_with_reuse` for eligibility and merge behavior.

## Workflow dispatch and monitoring

One-offs dispatch `.github/workflows/e2e-tests.yml`. `.github/workflows/run-sweep.yml` is PR-triggered and `.github/workflows/merge-ingest.yml` is push-triggered; neither is dispatchable.

```bash
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='main' \
  -f 'inputs[ref]=my-feature-branch' \
  -f 'inputs[test-name]=DSR1 fp8 H200 sglang smoke' \
  -f 'inputs[generate-cli-command]=full-sweep --config-files configs/nvidia-master.yaml --model-prefix dsr1 --framework sglang --runner-type h200 --min-conc 4 --max-conc 4 --seq-lens 8k1k' \
  -f 'inputs[duration-override]='
```

The top-level `ref` selects the workflow definition and is normally `main`. `inputs[ref]` selects the repository revision under test. Direct config dispatches set `inputs[generate-cli-command]` with paths relative to the selected checkout's `inferencex-e2e/` directory. Trusted changelog-driven dispatches instead set both `inputs[changelog-base-ref]` and `inputs[changelog-head-ref]`. `duration-override` replaces per-config seconds, and `require-power` makes invalid multi-node power telemetry fatal.

For AgentX preflight, add `-F 'inputs[agentx-fast]=true'`. Official runs use 10 warmup requests per lane and a one-hour profile.

```bash
RUN_ID=$(gh run list --repo SemiAnalysisAI/InferenceX --workflow e2e-tests.yml \
  --event workflow_dispatch --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" --repo SemiAnalysisAI/InferenceX --exit-status
gh run view "$RUN_ID" --repo SemiAnalysisAI/InferenceX --log-failed
gh run cancel "$RUN_ID" --repo SemiAnalysisAI/InferenceX
```

The dispatch POST returns no body or run ID.

## Evaluation selection

Eval selection, the `--no-evals` / `--evals-only` / `--all-evals` flags, and changelog eval fields are owned by `inferencex-e2e/infx/evals/EVALS.md`.

## Power telemetry

Multinode srt-slurm results may include `power_valid`, `avg_power_w`, `avg_total_gpu_power_w`, `total_gpu_energy_j`, and joules per query/input/output/total token. Invalid telemetry records `power_valid: 0` without energy metrics and fails only with `REQUIRE_POWER=1`, or always on AgentX lanes whose launcher validates power. Single-node results carry no power fields until srt-slurm telemetry covers those lanes.

Multinode disaggregated results add `prefill_gpu_energy_j`, `decode_gpu_energy_j`, `prefill_avg_power_w`, `decode_avg_power_w`, `prefill_joules_per_input_token`, and `decode_joules_per_output_token`. Role energy covers the full formal benchmark window, not kernel-level phases, and the role watts are that energy divided by the same window and by the role's GPU count.

NVL72 packages that also carry srt-slurm's `power/cpu/` sub-package add `cpu_power_valid`, `avg_cpu_socket_power_w`, `avg_total_cpu_power_w`, and `total_cpu_energy_j`. They add `avg_total_module_power_w` and `total_module_energy_j` when the module sensor is exposed on every socket. They add `avg_total_cpu_rail_power_w` and `total_cpu_rail_energy_j` (ACPI CPU rail), and `avg_total_cpu_sysio_power_w` and `total_cpu_sysio_energy_j` (ACPI SysIO rail), each pair only when every Grace socket reports that rail without gaps. The rails are a non-additive breakdown inside the Grace socket total: never add them to it or subtract them from it. The CPU verdict is independent of `power_valid`. An invalid CPU leg fails the job only when power is required and the recipe declares `telemetry.cpu_power_exporter.source`; AgentX lanes whose launcher validates power always require it, and other lanes require it through `REQUIRE_POWER=1`; see [`inferencex-e2e/docs/results-and-ingestion.md`](../inferencex-e2e/docs/results-and-ingestion.md#measured-grace-cpu-side-power-nvl72).

Every power result, valid or invalid, carries `power_metric_schema_version`. Version 2 defines each unprefixed `joules_per_*` field as whole-deployment GPU-board energy over the named denominator; role-scoped energy uses the explicit `prefill_*` / `decode_*` keys. Rows without the field predate the whole-deployment switch and their unprefixed joules are not comparable across topologies.

For srt-slurm recipes, `telemetry.enabled: true` with `telemetry.dcgm_exporter` enables official energy collection. The Git submodule pointer at `inferencex-e2e/utils/srt-slurm` is the source of truth for every srt-slurm job, including TileRT. CI derives `POWER_PRODUCER_SHA` from the launcher stamp. The aggregate-power and AgentX power tests validate telemetry and provenance. These local tests do not prove hardware power collection. Eligible recipe-gated `dynamo-sglang` dcgm-power lanes are validated.

Power audit artifacts are named `power_audit_<result>` and contain the multinode `power_validation_<result>_*.json` sidecars. They are uploaded even when validation fails.

## Result artifacts and metrics

```bash
gh api /repos/SemiAnalysisAI/InferenceX/actions/runs/<RUN_ID>/artifacts --jq '.artifacts[].name'
gh run download <RUN_ID> --repo SemiAnalysisAI/InferenceX -n results_bmk -D ./results
jq -r '.[] | [.hw, .infmax_model_prefix, "\(.isl)/\(.osl)", (.tput_per_gpu | round)] | @tsv' \
  ./results/agg_bmk.json | column -t
```

Never dump raw result JSON. Core metrics are `tput_per_gpu`, `output_tput_per_gpu`, `mean_ttft`, `p99_ttft`, `mean_tpot`, and `mean_e2el`.

Artifacts:

- `results_bmk`: `agg_bmk.json`.
- `results_all`: all aggregated results, which may not exist.
- `eval_results_all`: `agg_eval_all.json`, which may not exist.
- `run-stats`: `run_stats.json`, containing nodes run and success status.
- `power_audit_<result>`: canonical power validity verdict and reason codes.
