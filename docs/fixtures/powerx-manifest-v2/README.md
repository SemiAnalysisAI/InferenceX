# PowerX source manifest v2

This small, synthetic one-GPU AgentX bundle is copied byte-for-byte between
InferenceX and InferenceX-app. It tests the file contract. It is **not** a GPU
capture, hardware qualification or publication receipt.

`artifacts/required-power-sweep-manifest/sweep_manifest.json` is the source
manifest. Evidence paths are relative to `artifacts/`, include their GitHub
artifact directory, and have SHA-256 hashes of the exact retained bytes.

## Contract

- `schema-version: 2` versions publication independently of measurement-row
  `power_metric_schema_version: 2`. Unsupported or unversioned required bundles
  fail closed; legacy optional bundles remain optional.
- `run-id`, `run-attempt`, and `head` name the tested source. A later rerun may
  retain successful earlier-attempt evidence from the same run and head. The
  consumer compares these fields with GitHub source metadata, including on reuse.
- `matrix` retains the ordinary planner declaration. Every required,
  non-evaluation matrix point must occur exactly once in `points`.
- `config_key` is the matrix entry's `exp-name`. `identity` binds model, hardware,
  framework, precision, recipe fingerprint, workload/sequence and concurrency.
  AgentX uses `agentic_traces` with null `isl`/`osl`; fixed sequence uses
  `single_turn`. Full recipe details remain in the matrix and hashed aggregate.
- `topology` declares aggregate versus P/D, single versus multi-node and physical
  GPU counts. `devices` binds actual nodes and GPU UUIDs to aggregate, prefill or
  decode roles with positive per-device energy. Both P/D roles are mandatory for
  disaggregated deployments. A local GPU index alone is insufficient.
- `measurement_window` contains finite Unix-second boundaries, with end after
  start. The evidence must describe the same window.
- `artifacts` contains confined relative paths, exact hashes and
  `validation_state: "valid"`. Missing files, path escapes, invalid verdicts,
  mismatched identities and invalid power reject required publication. Missing
  and invalid values remain distinct from numeric zero. Required energy and
  derived power must be strictly positive; measured zero fails that gate.

## Publication intent

Normal manifests emit `publication: {"mode":"incremental","replacement_scope":[]}`.
Incremental means **no authorized loss**; it does not change existing whole-curve
selection or automatically enable append-only. A complete refresh can pass by
retaining all stable point identities. Recipe/image fingerprint changes may need
a reviewed exact replacement.

`mode: "replacement"` permits only explicit entries containing:

- `curve_scope`: the canonical logical curve JSON identity;
- `previous_snapshot_workflow_run_id`: the exact database snapshot superseded;
- `removed_point_identities`: the exact set of lost stable identities.

No wildcard, stale snapshot or broader permission is implied. The producer's
ordinary path emits no destructive authorization. Authoring authorization needs
explicitly reviewed replacement intent and current snapshot evidence.

AgentX scope groups model/hardware/framework/precision/workload; AGG/P/D,
parallelism, offload and recipe variants are points in that curve. Fixed-sequence
scopes retain existing spec/disagg/offload boundaries. The consumer models latest
attempts, whole snapshots and same-image append-only inheritance before its first
publication write. Row counts or a stale materialized view cannot establish safety.

## Remaining publication boundary

Artifact validation runs before workflow migrations. Curve preflight runs before
workflow/config/benchmark upserts. It is a pure preflight, not an atomic commit:
concurrent writers and failures after the first write can still cause partial
publication. Follow-up work must stage the complete run, recheck the base-table
curve under a target-database publication lock, then atomically promote metadata
and benchmark rows. Sidecar preparation belongs outside that transaction.
Post-write DB/API checks remain necessary; exact-run equality alone does not prove
latest-curve visibility.

## Read-only fixture check in InferenceX-app

From the repository root:

```sh
bun packages/db/src/preflight-power-publication.ts \
  docs/fixtures/powerx-manifest-v2/artifacts 123 1 \
  bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
```

This opens no database connection, dispatches no work and writes no files. It
validates artifacts only; focused curve tests model previous published state.

## 中文说明

此目录是一份最小的合成 AgentX 样例，两个仓库保留完全相同的文件字节，
用于测试 producer 与 consumer 的契约；它不是实机采集或生产发布证据。

manifest schema v2 记录来源 run、attempt、测试 SHA、完整 matrix、逐点身份、
测量窗口、实际 node/GPU UUID/角色及产物哈希。必需功耗路径拒绝缺失、损坏、
不兼容或能量非正的证据；P/D 必须同时覆盖 prefill 和 decode。历史可选功耗
路径保留兼容行为。数值 0 不会被改写为“缺失”，但不能通过正能量门槛。

默认 incremental 只表示没有授权丢点，不会改变数据库整条曲线替换的语义。
有意替换必须精确指定旧快照及全部移除点，不能使用通配或过期授权。AgentX
的 AGG、P/D、offload 和配方属于同一条逻辑曲线，校验时必须一起比较。

产物校验先于 migration；曲线校验先于发布数据写入。当前实现不是原子发布，
仍需后续 staging、发布锁和事务性 promote 来处理并发及中途写入失败。
DB/API 写后校验仍保留，exactRun 校验不能单独证明 latest 曲线完整可见。
