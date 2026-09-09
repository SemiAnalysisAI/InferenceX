# Agent Operations Reference

Detailed operational reference moved out of `AGENTS.md` to keep the default agent context small. Read only the section relevant to the current task.

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

A PR sweep requires exactly one primary label:

- `sweep-enabled`: trim every parallelism configuration to its lowest concurrency. Use for most lightweight validation.
- `full-sweep-fail-fast`: canary-gated full sweep with matrix-scoped fail-fast. Recommended for image bumps, recipe changes, bring-up, and other full sweeps.
- `full-sweep-enabled`: canary-gated full sweep without fail-fast. Use when a flaky job must not cancel its matrix's in-flight work.
- `full-sweep-fail-fast-no-canary`: full, matrix-scoped fail-fast without the canary. Use when the canary is flaky or unrepresentative.
- `non-canary-full-sweep-enabled`: full sweep without canary or fail-fast. Use when both canary gating and fail-fast are unsuitable.

Modifiers:

- `all-evals` expands eval selection to every generated fixed-sequence configuration without suppressing throughput. It remains reuse-eligible with an eligible full-sweep label.
- `evals-only` suppresses throughput. Combining it with `all-evals` runs every eval and no throughput. It is not reuse-eligible.
- `agentx-fast` uses one deterministic warmup request per lane and a 20-minute AgentX profile. It does not affect fixed-sequence or eval jobs and is not reuse-eligible.

Fail-fast is matrix-scoped: one matrix failure does not cancel other matrices, and completed results remain valid. The failed job remains red.

Sweeps do not trigger while a PR has merge conflicts. For `perf-changelog.yaml` conflicts, follow `KLAUD_DEBUG.md` section 1.1: merge `origin/main`, restore the file byte-for-byte from `origin/main`, then append only the PR's entry at the tail. Never 3-way merge the changelog.

Pushes to `main` always enter sweep setup and either reuse approved artifacts or run an untrimmed full sweep. `[skip-sweep]` only skips PR benchmark setup. It never skips a main-branch sweep. It still permits changelog validation and reuse authorization checks.

Artifact reuse excludes runs with `evals-only` or `agentx-fast`. See `.github/workflows/README.md` and `utils/merge_with_reuse.sh` for eligibility and merge behavior.

## Workflow dispatch and monitoring

One-offs dispatch `.github/workflows/e2e-tests.yml`. `.github/workflows/run-sweep.yml` is push/PR-triggered and is not dispatchable.

```bash
gh api -X POST \
  /repos/SemiAnalysisAI/InferenceX/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='main' \
  -f 'inputs[ref]=my-feature-branch' \
  -f 'inputs[test-name]=DSR1 fp8 H200 sglang smoke' \
  -f 'inputs[generate-cli-command]=full-sweep --config-files configs/nvidia-master.yaml --model-prefix dsr1 --framework sglang --runner-type h200 --min-conc 4 --max-conc 4 --seq-lens 8k1k' \
  -f 'inputs[duration-override]='
```

The top-level `ref` selects the workflow definition and is normally `main`. `inputs[ref]` selects the repository revision under test. Direct config dispatches set `inputs[generate-cli-command]`. Trusted changelog-driven dispatches instead set both `inputs[changelog-base-ref]` and `inputs[changelog-head-ref]`. `duration-override` replaces per-config seconds, and `require-power` makes invalid fixed-sequence power telemetry fatal.

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

Full details live in `utils/evals/EVALS.md`.

`mark_eval_entries()` in `utils/matrix_logic/generate_sweep_configs.py` selects evals, which default to the 8k1k subset and run separately from throughput with `EVAL_ONLY=true`.

- `--no-evals`: skip evals.
- `--evals-only`: run the default selected eval subset and suppress throughput.
- `--all-evals`: expand selection to every generated fixed-sequence configuration. It composes with `--evals-only`.

For multi-node configurations, `--all-evals` creates one eval job per engine topology and runs every distinct `conc-list` value sequentially against that engine. Changelog `all-evals: true` suppresses throughput for that entry. The PR `all-evals` label expands selection only, while the `evals-only` label suppresses throughput. `utils/collect_eval_results.py` produces aggregated output.

## Power telemetry

Single-node fixed-sequence results may include `power_valid`, `avg_power_w`, `avg_total_gpu_power_w`, `total_gpu_energy_j`, and joules per query/input/output/total token. Invalid telemetry records `power_valid: 0` without energy metrics and fails only with `REQUIRE_POWER=1`.

Multinode disaggregated results add `prefill_gpu_energy_j`, `decode_gpu_energy_j`, `prefill_avg_power_w`, `decode_avg_power_w`, `prefill_joules_per_input_token`, and `decode_joules_per_output_token`. Role energy covers the full formal benchmark window, not kernel-level phases, and the role watts are that energy divided by the same window and by the role's GPU count.

Every power result — valid or invalid, single-node or multinode — carries `power_metric_schema_version`. Version 2 defines each unprefixed `joules_per_*` field as whole-deployment GPU-board energy over the named denominator; role-scoped energy uses the explicit `prefill_*` / `decode_*` keys. Rows without the field predate the whole-deployment switch and their unprefixed joules are not comparable across topologies.

For srt-slurm recipes, `telemetry: {provider: dcgm-power}` enables official energy collection. `runners/launch_gb200-nv.sh`, `runners/launch_gb300-nv.sh`, and `runners/launch_h200-dgxc-slurm.sh` are the source of truth for `POWER_SRT_SLURM_PIN`. CI derives `POWER_PRODUCER_SHA` from the launcher stamp. `utils/test_gb300_power_official_contract.py` exercises launcher routing; the aggregate-power and AgentX power tests validate telemetry, provenance, and lifecycle behavior. These local tests do not prove hardware power collection. Eligible recipe-gated `dynamo-sglang` dcgm-power lanes are validated.

Power audit artifacts are named `power_audit_<result>` and contain `power_validation_<result>.json` for single-node runs or `power_validation_<result>_*.json` for multinode runs. They are uploaded even when validation fails.

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
