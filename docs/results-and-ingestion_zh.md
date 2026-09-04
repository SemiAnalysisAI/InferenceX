# 结果与摄取

<div align="center">

[English](./results-and-ingestion.md) | **中文**

</div>

本页用于识别基准工件、检查其契约，并判断一次运行能否安全交给 InferenceX-app。生产端和摄取代码仍是权威来源。下列 InferenceX-app 链接固定在提交 [`3be1c34`](https://github.com/SemiAnalysisAI/InferenceX-app/tree/3be1c34a174f62fea2194f1133210e692e5bf415)。

## 权威来源

| 权威来源 | 控制内容 |
| --- | --- |
| [`benchmark-tmpl.yml`](../.github/workflows/benchmark-tmpl.yml)、[`benchmark-multinode-tmpl.yml`](../.github/workflows/benchmark-multinode-tmpl.yml) | 吞吐量、评测和 AgentX 工件的单配置名称、文件及上传规则 |
| [`utils/process_result.py`](../utils/process_result.py) | 固定序列吞吐量聚合架构及派生的每 GPU 指标 |
| [`utils/collect_results.py`](../utils/collect_results.py)、[`collect-results.yml`](../.github/workflows/collect-results.yml) | 将基准结果递归收集为 `agg_<prefix>.json` 和 `results_<prefix>` |
| [`utils/collect_eval_results.py`](../utils/collect_eval_results.py)、[`collect-evals.yml`](../.github/workflows/collect-evals.yml) | 评测发现、指标提取、批量并发选择及 `eval_results_<prefix>` |
| [`process_agentic_result.py`](../utils/agentic/aggregation/process_agentic_result.py)、[`request_metrics.py`](../utils/agentic/aggregation/request_metrics.py) | AgentX 聚合架构、原始记录过滤、请求计数和派生指标 |
| [`validate_agentic_result.py`](../utils/agentic/validation/validate_agentic_result.py) | AgentX 上传前错误率门禁 |
| [`run-sweep.yml`](../.github/workflows/run-sweep.yml)、[`recover-reused-ingest.yml`](../.github/workflows/recover-reused-ingest.yml) | 应用分发载荷及 source/merge 运行身份 |
| [InferenceX-app `prepare-ci-artifacts.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/src/prepare-ci-artifacts.ts)、[`ci-artifact-preparation.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/src/lib/ci-artifact-preparation.ts) | 跨运行工件选择、attempt 及复用来源信息 |
| [InferenceX-app `ingest-ci-run.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/src/ingest-ci-run.ts) | 端到端摄取顺序、配对、跳过、汇总和刷新 |
| [InferenceX-app `benchmark-mapper.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/src/etl/benchmark-mapper.ts)、[`eval-mapper.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/src/etl/eval-mapper.ts)、[`agentic-v3-flatten.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/src/etl/agentic-v3-flatten.ts) | 工件到数据库的架构和规范化 |
| [InferenceX-app 架构](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/migrations/001_initial_schema.sql)、[AgentX 迁移](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/migrations/008_agentic.sql) | 持久化自然键、JSONB 指标和 trace-replay 旁表 |
| [InferenceX-app 基准测试 API](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/app/src/app/api/v1/benchmarks/route.ts)、[模型常量](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/constants/src/models.ts) | 已发布结果查询与前端模型名称 |

## 索引

1. [各层身份](#各层身份)
2. [吞吐量工件](#吞吐量工件)
3. [评测工件](#评测工件)
4. [AgentX 工件](#agentx-工件)
5. [应用交接和复用运行](#应用交接和复用运行)
6. [摄取阶段](#摄取阶段)
7. [去重和失败行行为](#去重和失败行行为)
8. [来源不变量](#来源不变量)
9. [已发布结果 API](#已发布结果-api)
10. [安全检查](#安全检查)
11. [验证和停止条件](#验证和停止条件)

## 各层身份

不要用一种标识符替代另一种。

| 层 | 身份 | 含义 |
| --- | --- | --- |
| GitHub 工作流 | `(github_run_id, run_attempt)` | 一次执行 attempt。InferenceX-app 将其存为 `workflow_runs` 自然键。 |
| 已上传工件 | 一个工作流运行内的工件 `name` | 可下载的文件包。重跑可以再次上传完全相同的名称。 |
| 单配置结果 | `RESULT_FILENAME`，多点 AgentX 路径按并发各产一个文件时还包含 `_concN` 后缀 | 用于命名文件和同级工件包的运行时配置身份。它不是数据库键。 |
| 规范化配置 | 模型、硬件、框架、精度、推测方法、解耦状态、拓扑及相关配置字段 | InferenceX-app 在规范化后解析或创建一条 `configs` 记录。 |
| 吞吐量或 AgentX 数据库点 | `(workflow_run_id, config_id, benchmark_type, isl, osl, conc, offload_mode)`，空值视为相等 | 幂等基准身份。AgentX 使用 `benchmark_type=agentic_traces`、空 `isl`/`osl`，并将并发解释为用户数。 |
| 评测数据库点 | `(workflow_run_id, config_id, task, isl, osl, conc)` | 所有可空维度均以相同值填充时，聚合评测和单配置评测工件会汇入同一记录。当前唯一约束采用 PostgreSQL 普通空值语义，因此空 `isl`、`osl` 或 `conc` 不会与另一个空值冲突。 |
| 评测样本 | `(eval_result_id, doc_id)` | 单文档样本身份。 |
| AgentX 原始旁表 | `benchmark_results.trace_replay_id` | 从规范化 AgentX 数据点指向保留及预计算 trace 数据的链接。 |

复用时，这些区别尤其重要。工件字节可以来自 PR sweep，而变更日志元数据和摄取触发可以来自之后的 main 运行。存储后的基准记录仍归属于 source 运行及其 source attempt。

## 吞吐量工件

### 生产端和收集器

对于单节点固定序列任务，[`benchmark-tmpl.yml`](../.github/workflows/benchmark-tmpl.yml) 根据实验、精度、框架、TP/PP/DCP/PCP/EP/DP-attention、解耦、推测模式、并发和具体 runner 构建 `RESULT_FILENAME`。基准先写入 `<RESULT_FILENAME>.json`。[`utils/process_result.py`](../utils/process_result.py) 读取该文件并写入 `agg_<RESULT_FILENAME>.json`。工作流按下列身份上传：

```text
artifact: bmk_<RESULT_FILENAME>
file:     agg_<RESULT_FILENAME>.json
```

多节点模板在基础名称中编码 prefill 和 decode 拓扑、worker 数、模式、并发及 runner。一个 `bmk_<RESULT_FILENAME>` 工件中可以包含多个 `agg_<RESULT_FILENAME>_*.json` 文件。

[`collect-results.yml`](../.github/workflows/collect-results.yml) 通常接收 `result-prefix: bmk`。它下载 `bmk_*`，[`utils/collect_results.py`](../utils/collect_results.py) 再递归加载每个 JSON 文件，形成一个数组。交接身份为：

```text
artifact: results_bmk
file:     agg_bmk.json
shape:    array of benchmark row objects
```

收集器不验证记录、不排序，也不去重。成功解析 JSON 是它唯一的内容检查。应把 `results_bmk` 视为传输聚合，而不是所有记录均可用的证明。

### 吞吐量记录架构

固定序列转换器要求 runner、框架、精度、推测模式、结果文件名、ISL、OSL、解耦、模型前缀和镜像元数据。其基础输出包括：

| 字段组 | 重要字段 |
| --- | --- |
| 配置和路由 | `hw`、`model`、`infmax_model_prefix`、`framework`、`precision`、`spec_decoding`、`image`、`disagg`、`is_multinode`、`isl`、`osl`、`conc` |
| 单节点拓扑 | `tp`、`pp`、`dcp_size`、`pcp_size`、`ep`、`dp_attention` |
| 多节点拓扑 | `prefill_tp`、`prefill_pp`、`prefill_dcp_size`、`prefill_pcp_size`、`prefill_ep`、`prefill_dp_attention`、`prefill_num_workers`，对应的 `decode_*` 字段，`num_prefill_gpu`、`num_decode_gpu`，以及可选的 `prefill_hw`/`decode_hw` |
| 主要派生指标 | `tput_per_gpu`、`input_tput_per_gpu`、`output_tput_per_gpu` |
| 延迟和交互性 | 基准输入中每个以 `ms` 结尾的键都会从毫秒换算为秒，并移除 `_ms`。包含 `tpot` 的键还会产生其倒数 `intvty`。 |
| 可选运行时元数据 | 形式必须精确为 `{name, version}` 的 `router`、`kv_p2p_transfer`，以及在可用时由 `gpu_metrics.csv` 补入的实测功耗 |

单节点 GPU 数为 `tp * pp * pcp_size`。DCP 不会增加物理 GPU 数。多节点每 GPU 指标的分母使用声明的 prefill 和 decode GPU 数。无效或缺失的必需元数据会使转换失败。功耗聚合明确采用尽力而为模式，不能导致基准聚合失败。

InferenceX-app 将路由字段作为列或配置维度，并把数值测量存入 `benchmark_results.metrics` JSONB。映射器支持共享拓扑的 v1、拆分 prefill/decode 拓扑的 v2，以及嵌套 AgentX 指标的 v3。未知数值指标会被保留并产生警告，因此架构可以扩展，同时不会无提示地丢失数值数据。

### 功耗遥测停止与校验

[`benchmark_lib.sh`](../benchmarks/benchmark_lib.sh) 中的单节点 AMD 监控会等待每个已观测 GPU 都记录到有效的正数功耗样本，且时间戳不早于停止请求之后的第一个整秒。`AMD_MONITOR_STOP_TIMEOUT_S` 限制等待时长（默认 `30`；设为 `0` 可跳过等待）。没有可用 Unix 时间戳样本的数据流沿用原有的固定尾部等待。超时不代表覆盖有效：聚合器仍会拒绝未被样本完整包围的基准窗口。AgentX 取消运行时会跳过覆盖等待并停止监控，即使取消发生在正常停止的等待过程中。

[`aggregate_power.py`](../utils/aggregate_power.py) 不会将正式窗口之外的无效功耗行用于积分或边界覆盖。在边界搜索范围内，这些行会按 GPU 计入 `power_validation.json` 的 `boundary_degenerate_rows`，即使最终没有任何可用样本也会保留计数。窗口内的校验规则保持不变；这些计数仅供诊断，不能替代实测数据。

## 评测工件

### 单配置身份和收集

每个评测上传名为 `eval_<EXP_NAME>_<RESULT_FILENAME>`。当前允许的载荷包括 `meta_env.json`、`results*.json`、样本 JSONL、预测、SWE-bench 报告和轨迹文件。收集器只使用元数据和 lm-eval 结果 JSON 来生成聚合记录。

[`utils/collect_eval_results.py`](../utils/collect_eval_results.py) 执行以下规则：

1. 评测集是包含 `meta_env.json` 的根目录或一级子目录。
2. 候选结果文件必须能解析为对象并包含 `lm_eval_version`。
3. 传统评测集按修改时间贡献最新的候选文件。
4. 列表类型的 `eval_concs` 标识批量评测集。每个允许并发贡献最新的 `_concN` 结果。如果存在 `completed_eval_concs`，还会据此限制并发。
5. `results` 中的每个任务成为一条逻辑记录。
6. 主要 `score` 优先使用 strict 或 resolved exact match，其次使用 accuracy。标准误差单独保留。

收集交接形式为：

```text
artifact: eval_results_all
file:     agg_eval_all.json
shape:    array of one row per config, concurrency, and task
```

### 评测聚合架构

| 字段组 | 重要字段 |
| --- | --- |
| 配置 | `is_multinode`、`model_prefix`、`model`、`hw`、`framework`、`precision`、`spec_decoding`、`isl`、`osl`、`conc` |
| 拓扑 | `tp`、`ep`、`dp_attention`，以及拆分的 `prefill_*` 和 `decode_*` worker 字段 |
| 评测 | `task`、`score`、`score_name`、`score_se`、`em_strict`、`em_strict_se`、`em_flexible`、`em_flexible_se`、`n_eff` |
| 可追踪性 | `source`，即已下载工件树中被选中的结果 JSON 路径 |

应用也会读取每个未聚合的 `eval_*` 目录。`meta_env.json` 提供配置身份，`results_*.json` 提供 `lm_eval_version`、任务、原始数值指标和有效样本数。应用会规范化 strict 和 flexible exact-match 名称。样本文件按任务附加到解析后的评测记录。该双路径是有意设计。聚合记录用于摘要摄取，单配置文件则保留样本详情。

收集器解析失败时，`load_json` 会跳过文件，不会生成一条失败的评测记录。缺少 `meta_env.json`、没有可识别的 lm-eval 结果、`results` 对象为空，或并发不在 `completed_eval_concs` 中，都会导致不输出聚合记录。

## AgentX 工件

### 三种相关身份

AgentX 使用同一 `RESULT_FILENAME` 系列，但生成两类同级工件：

```text
aggregate artifact: bmk_agentic_<RESULT_FILENAME>
aggregate file:     <RESULT_FILENAME>.json
raw artifact:       agentic_<RESULT_FILENAME>
raw tree:           results/**, excluding inputs.json and profile_export_raw.jsonl
```

聚合工件匹配 `bmk_*` 收集模式，因此也会成为 `results_bmk/agg_bmk.json` 中的一条记录。原始同级工件不会交给 `collect_results.py`。InferenceX-app 移除 `bmk_` 和 `agentic_` 后缀前缀，将 `bmk_agentic_<suffix>` 与 `agentic_<suffix>` 配对。对于以 `_concN.json` 命名的文件，并发也参与 trace 同级工件查找。

服务器日志是单独的 `server_logs_<RESULT_FILENAME>` 工件。应用会使用完全移除前缀后的后缀作为回退，从而让 AgentX 记录找到不含 `agentic_` 前缀的日志工件。

### 原始输入和聚合架构

[`process_agentic_result.py`](../utils/agentic/aggregation/process_agentic_result.py) 可解析当前的 `results/aiperf_artifacts` 布局，也可解析只含一个子目录的嵌套布局。它要求存在 `profile_export.jsonl`，并在存在时读取以下输入：

| 输入 | 作用 |
| --- | --- |
| `profile_export.jsonl` | 单请求指标和生命周期元数据。必需。 |
| `profile_export_aiperf.json` | AIPerf 聚合元数据，包括已输出时的数据集来源。可选。 |
| `server_metrics_export.json` | 服务器缓存、KV-cache 和 token 指标。数据缺失时会生成空的或带警告的服务器指标，不会取代请求数据源。 |
| 服务器日志 | 特定框架的服务器指标和容量回退来源。 |

每条非空 JSONL 记录都会增加 `records_total`。`metadata.benchmark_phase` 不等于 `profiling` 的记录是 warmup 诊断，会被排除。含真值 `error` 的记录也会被排除并分类。没有 phase 的旧记录按 profiling 处理。保留记录数成为 `num_requests_successful`。完整计数保存在 `request_accounting` 中，包括 profiled、总丢弃、warmup 丢弃、错误丢弃和 `error_categories`。

AgentX 聚合的顶层身份和拓扑字段与基准摄取兼容。重要 AgentX 字段包括：

| 字段组 | 重要字段 |
| --- | --- |
| 场景和请求数 | `scenario_type: agentic-coding`、`num_requests_total`、`num_requests_successful`、`request_accounting` |
| 缓存配置 | `kv_offloading`、`kv_offload_backend`、可选的 `kv_p2p_transfer`、`allocated_cpu_dram_gb`、可选的 `router` |
| 来源 | 从 AIPerf `metadata.dataset` 复制的 `dataset` |
| 请求指标 | `request_metrics.qps`，包含 TTFT/E2EL/ITL/TPOT/交互性的 `latency` 块，token 分布、吞吐量、缓存和每 GPU 吞吐量 |
| 服务器指标 | `server_metrics.cache`、`kv_cache`、token 总数、来源详情，以及可能存在的 `warnings` |
| 兼容性 | `kv_cache_pool_tokens` 镜像 `server_metrics.kv_cache.gpu_total_tokens` |

应用会将嵌套 AgentX v3 值展平为规范指标键。例如 `median_ttft`、`p95_e2el`、`total_tput_tps`、`tput_per_gpu`、`server_gpu_cache_hit_rate` 和 `gpu_kv_cache_usage_pct`。p50 映射为 `median`。存在 full-response ITL 字段时优先使用它。交互性百分位数按对应 ITL 百分位数的倒数派生，使历史记录和当前记录采用同一定义。

正常上传前，单节点工作流会运行 [`validate_agentic_result.py`](../utils/agentic/validation/validate_agentic_result.py)。它要求聚合为对象，`request_count.avg` 是非负数值，已完成请求数为正，错误率不高于配置阈值。通过该门禁不表示没有失败请求。失败请求记录仍可通过 `request_accounting` 观察，但不参与性能指标计算。

## 应用交接和复用运行

[`run-sweep.yml`](../.github/workflows/run-sweep.yml) 向 InferenceX-app 分发 `ingest-results` 或 `ingest-agentic-results`。

- `source-run-id` 标识提供基准、评测、AgentX、日志和统计工件的工作流运行。
- `merge-run-id` 标识授权摄取并提供当前变更日志元数据的 main 分支工作流运行。
- 对于普通 main 运行，两个 ID 都等于 `github.run_id`。
- 对于复用的 PR sweep，`source-run-id` 是选定的 PR 运行，`merge-run-id` 是当前 main 运行。

InferenceX-app 的工件准备会从 source 运行中，为每个完全相同的工件名保留最新且未过期的上传。在复用模式下，它排除 source 运行的 `changelog-metadata`，要求 merge 运行中存在未过期的变更日志工件，并把该 merge 运行工件加入计划。没有未过期 source 工件，或复用时没有 merge 运行变更日志，都会直接失败。

只有两个 ID 不同时，它才写入 `reused-ingest-metadata/reuse_source_run.json`。该文件记录 source ID、source attempt、URL、PR、head SHA，以及 ingest ID、ingest attempt 和 URL。摄取期间，该元数据会把记录归属切回 source 运行和 source attempt。source PR 分支、SHA 和公开运行 URL 继续附在测量记录上。触发时间可以来自 merge 运行。这样不会把复用的 PR 测量伪装成 merge 运行产出的数据。

两个应用工作流都会把 merge 运行 ID 和 attempt 传入初始 CI 设置，因为它代表触发包。`ingest-ci-run.ts` 随后读取复用元数据，并在创建或解析 `workflow_runs` 前有意将其替换为 source 身份。

## 摄取阶段

InferenceX-app 按以下顺序执行。固定序列工作流超时为 30 分钟。包含大量 blob 的 AgentX 工作流超时为 180 分钟，并可选择 staging 或 production。

1. **准备工件。** 获取 source 和 merge 运行元数据，选择未过期工件，将每个工件下载到空目录，并在需要时写入复用元数据。
2. **运行数据库迁移。** 摄取不会假设目标架构已是最新状态。
3. **解析来源。** 读取复用元数据，获取 GitHub 元数据，预加载规范化配置记录，创建或解析 source `(github_run_id, run_attempt)` 工作流记录。若 source 运行或 attempt 位于 run overrides 中，此阶段即停止，不写结果。
4. **读取变更日志元数据。** 检测 `evals-only`。没有有效变更日志工件时，根据工作流元数据生成回退描述。Evals-only 运行跳过基准和运行统计阶段。
5. **摄取基准记录。** 读取 `results_bmk` 以及每个 `bmk_*` 或 `results_*` 目录。规范化模型、硬件、框架、精度、拓扑、场景、offload 模式、镜像和数值指标。解析配置，跳过 point overrides，批量 upsert 记录，只根据批量写入成功的记录构建 availability。
6. **附加大型旁路数据。** 链接服务器日志。配对 AgentX 聚合及原始工件。对于尚无 trace 链接的记录，worker 进程压缩原始 profile/server JSON，计算聚合统计、图表序列和请求时间线，随后持久化一条旁表记录并链接至基准记录。
7. **链接数据集来源。** 从 AgentX 聚合记录提取数据集 slug，并 upsert 一条 `run_datasets` 映射。一个工作流内出现多个 slug 是硬冲突。若数据集表中没有该 slug，则会报告，因为在数据集摄取前，时间线深层链接会失败。
8. **摄取运行统计。** 对非 evals-only 运行按硬件 upsert 成功数和总数。
9. **摄取评测摘要和样本。** 先处理 `eval_results_all`，再处理单配置 `eval_*` 目录。两者解析到相同的评测自然键。按任务和文档 ID 附加样本 JSONL。
10. **摄取变更日志并汇总。** Upsert 变更日志记录，打印新增、重复、跳过计数和未映射值，并写出供工作流告警使用的未映射实体报告。
11. **刷新并验证。** 刷新 `latest_benchmarks`。工作流随后应用已审计的 point overrides，运行 `admin:db:verify`，再使应用缓存失效。

该顺序是有意设计。基准记录成功写入前不会创建 availability。所有目标记录已经有旁表时，不会插入 trace blob。只有在迁移、摄取、overrides 和数据库验证之后，才会使缓存失效。

## 去重和失败行行为

多个边界都存在去重。诊断重复前，先确认所处边界。

| 边界 | 行为 |
| --- | --- |
| CI 中的工件准备 | 每个完全相同的工件名保留最新且未过期的上传。复用只会以 merge 运行副本替换变更日志元数据。 |
| 应用直接下载模式 | [`dedupeArtifactsByLogicalName`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/3be1c34a174f62fea2194f1133210e692e5bf415/packages/db/src/lib/github-artifacts.ts) 移除末尾 runner-pool 和 attempt token，并保留最新的逻辑工件，防止重试工件覆盖良好指标。 |
| 基准收集 | `collect_results.py` 附加每个已解析 JSON。它不做记录级去重。 |
| 基准数据库写入 | 按基准自然键执行 `ON CONFLICT`，更新指标、镜像、功耗 worker 及相关字段。当新工件缺少服务器派生的 `kv_cache_pool_tokens` 时会保留已有值。 |
| 评测数据库写入 | 维度完整且匹配的聚合记录和单配置记录会按评测自然键冲突。后一次写入刷新指标，并返回同一记录 ID 供样本附加。任一可空键维度为空时，PostgreSQL 当前的普通唯一约束不会对这些记录去重。 |
| 评测样本 | 按 `(eval_result_id, doc_id)` 冲突，防止文档重复。 |
| AgentX trace 旁表 | 先查询已有链接。所有基准记录都已链接时，跳过 blob 准备和插入。 |

失败数据与去重分开处理：

- 当数值类型 `num_requests_successful` 为零且存在 `num_requests_total` 时，应用基准映射器会丢弃该记录。这也包括总数为零的服务器启动失败。此规则防止无数据重试替换良好的自然键记录。
- AgentX 聚合从指标计算中排除 warmup 和错误请求记录，但在 `request_accounting` 中保留计数和类别。
- 未知模型或硬件、缺少固定序列 ISL/OSL/并发、无效 JSON、point overrides 和数据库错误都会被跟踪为跳过，不会成为占位记录。
- 评测收集器解析失败或缺少可识别结果文件时不输出记录。应用侧格式错误的单配置文件会产生警告或已跟踪的跳过。
- 数据库写入具备幂等性，因此部分摄取可以安全重跑。但不能忽略新增跳过计数、冲突的数据集来源、缺失的 AgentX 原始同级工件或失败的数据库验证。

## 来源不变量

1. **运行来源包含两个维度。** 必须记录运行 ID 和 attempt。不含 attempt 的运行 URL 可能指向最新重跑，而不是已摄取字节对应的执行。
2. **复用不会改变测量归属。** 指标和 source Git 元数据属于 `source-run-id`。只有 merge 运行变更日志和触发上下文来自 `merge-run-id`。
3. **工件来源必须可检查。** 保留精确工件名、上传时间、过期状态和选中文件。评测聚合中的 `source` 是路径线索，不能替代工作流运行和工件身份。
4. **数据集来源必须传递，不能推测。** AgentX 将 AIPerf `metadata.dataset` 复制到顶层 `dataset`。应用根据声明的 Hugging Face 数据集名派生仪表板 slug，不会根据基准名称猜测。
5. **一个 AgentX 工作流最多映射一个数据集。** 冲突 slug 会停止摄取。缺少来源信息的旧工件不会改动已有运行映射。
6. **规范化也是来源的一部分。** 诊断映射时，应保留原始 `model`、`infmax_model_prefix`、`hw`、`framework`、精度、镜像、拓扑和场景字段。数据库 config ID 是规范化身份，不是原始拼写。
7. **AgentX 原始数据必须与聚合指标配对。** 基准点可以在没有原始旁表的情况下摄取，但系统会报告同级工件缺失，详情视图也会不完整。

## 已发布结果 API

查询已经入库到 Dashboard 的结果时使用公共 API。查询尚未入库的 Run、原始输出、日志或调试证据时使用 GitHub Actions 产物。

```bash
INFERENCEX_API=https://inferencex.semianalysis.com/api/v1
curl --fail --compressed \
  "$INFERENCEX_API/benchmarks?model=DeepSeek-V4-Pro" \
  | jq 'first(.[] | select(.benchmark_type == "single_turn" and .isl == 8192 and .osl == 1024)) | {date, model, hardware, framework, precision, isl, osl, tput_per_gpu: .metrics.tput_per_gpu, run_url}'
```
该示例只返回一条有界的 8k1k 记录。将选中行作为证据前，应添加精确的 hardware、framework、precision 与 date 条件。

`model=` 使用链接的 InferenceX-app 模型常量所定义的前端显示名称。固定序列行使用数字 `isl` 和 `osl`；`agentic_traces` 行的长度为空，不要意外过滤掉它们。`view=calculator&sequence=8k/1k` 返回紧凑的插值数据。`date`、`runId`、`exact` 和 `exactRun` 用于限定历史或特定 Run 的查询。发现端点包括 `/availability`、`/workflow-info`、`/evaluations` 和 `/reliability`。

始终使用 `--compressed` 并通过 `jq` 过滤。不要输出原始基准测试 JSON，不要绕过缓存，也不要反复轮询 CDN 缓存结果。

## 安全检查

下列命令对 GitHub 都是只读操作。下载内容进入新建临时目录。只有在列出运行工件后，才替换 ID 和工件名。

### 检查运行元数据和工件清单

```bash
RUN_ID=<github-run-id>
REPO=SemiAnalysisAI/InferenceX

gh api "repos/$REPO/actions/runs/$RUN_ID" \
  --jq '{id,run_attempt,status,conclusion,event,head_branch,head_sha,html_url}'

gh api --paginate "repos/$REPO/actions/runs/$RUN_ID/artifacts?per_page=100" \
  --jq '.artifacts[] | [.id,.name,.created_at,.expires_at,.expired,.size_in_bytes] | @tsv'
```

如果 ID 不是数字、仓库错误、run attempt 不是目标执行、必需工件已过期，或无法按时间区分多个同名上传，请停止。

### 预览 source/merge 工件计划

在固定或已审查修订的 InferenceX-app checkout 中运行：

```bash
SOURCE_RUN_ID=<measurement-run-id> \
MERGE_RUN_ID=<main-trigger-run-id> \
INGEST_REPO=SemiAnalysisAI/InferenceX \
bun run admin:db:prepare:ci --dry-run
```

`--dry-run` 获取元数据并打印选中的工件计划，不下载工件，也不连接数据库。复用时，应确认基准数据来自 source 运行，而 `changelog-metadata` 来自 merge 运行。

### 下载并检查聚合架构

```bash
tmp="$(mktemp -d "${TMPDIR:-/tmp}/infx-results.XXXXXX")"

gh run download "$RUN_ID" -R "$REPO" -n results_bmk -D "$tmp/results_bmk"
jq 'type, length' "$tmp/results_bmk/agg_bmk.json"
jq '.[0] | {hw,model,infmax_model_prefix,framework,precision,scenario_type,isl,osl,conc,is_multinode,disagg}' \
  "$tmp/results_bmk/agg_bmk.json"

# Run this only when eval_results_all appeared in the inventory.
gh run download "$RUN_ID" -R "$REPO" -n eval_results_all -D "$tmp/eval_results_all"
jq 'type, length' "$tmp/eval_results_all/agg_eval_all.json"
jq '.[0] | {model_prefix,model,hw,framework,task,conc,score,score_name,n_eff,source}' \
  "$tmp/eval_results_all/agg_eval_all.json"
```

不要假定第零条记录代表正在调查的配置。确认架构后，应按稳定字段过滤：

```bash
jq --arg model '<model-prefix>' \
  '[.[] | select(.infmax_model_prefix == $model or .model_prefix == $model)] |
   map({scenario_type,hw,framework,precision,isl,osl,conc,num_requests_total,num_requests_successful,request_accounting})' \
  "$tmp/results_bmk/agg_bmk.json"
```

### 检查一个原始 AgentX 同级工件

从清单中选择精确的 `agentic_<RESULT_FILENAME>` 名称：

```bash
AGENTIC_ARTIFACT='agentic_<RESULT_FILENAME>'
gh run download "$RUN_ID" -R "$REPO" -n "$AGENTIC_ARTIFACT" -D "$tmp/agentic"

jq '{metadata,request_count,error_request_count,completed_request_count}' \
  "$tmp/agentic/results/aiperf_artifacts/profile_export_aiperf.json"

jq -s '{rows:length,
        profiling:map(select((.metadata.benchmark_phase // "profiling") == "profiling"))|length,
        warmup:map(select((.metadata.benchmark_phase // "profiling") != "profiling"))|length,
        errors:map(select(.error))|length}' \
  "$tmp/agentic/results/aiperf_artifacts/profile_export.jsonl"
```

部分历史原始工件使用只含一个子目录的嵌套 AIPerf 目录。如果上面的精确路径不存在，应先停止并检查下载目录名称，再选择文件。不要凭猜测展平、重命名或合并文件。

检查完成后，只删除刚刚创建的临时目录：

```bash
printf 'temporary inspection directory: %s\n' "$tmp"
rm -rf -- "$tmp"
```

## 验证和停止条件

只有全部适用检查通过，交接才算验证完成。

### 摄取前验证

- 明确目标 `github_run_id` 和 `run_attempt`。
- 预期吞吐量或 AgentX 数据点时，`results_bmk` 包含 JSON 数组。
- 预期聚合评测时，`eval_results_all` 包含 JSON 数组。需要样本详情时，对应单配置 `eval_*` 文件包仍然存在。
- 每个预期 AgentX 聚合都有对应的 `agentic_<suffix>` 原始同级工件。需要服务器派生指标时，服务器日志存在。
- 固定序列记录具有正数 `isl`、`osl` 和 `conc`。AgentX 记录具有 agentic 场景、正并发、请求计数，以及预期的 offload 和数据集元数据。
- source/merge dry-run 精确选择预期的 source 测量和 merge 变更日志。
- 工件过期时间为完整应用工作流留出足够时间，尤其是更长的 AgentX trace 处理路径。

### 摄取后验证

使用应用工作流日志和数据库 verifier，不能只根据工件存在来判断。确认：

- `ingest-ci-run` 显示的 source 运行和 source attempt 正确；
- 对于首次摄取或重跑，新增数和重复数合理；
- 已理解跳过计数、未映射实体、缺失数据集和缺失 trace 同级工件；
- 没有发生数据集来源冲突；
- AgentX 数据点已创建 trace-replay 链接，或明确显示链接已存在；
- 应用 overrides 后 `admin:db:verify` 通过；
- 验证之后执行了缓存失效。

### 停止条件

出现以下任一情况时，停止摄取或恢复调查：

- source 和 merge ID 或 attempt 有歧义；
- 选中的 source 没有未过期结果工件，或复用时没有未过期的 merge 运行变更日志；
- 聚合 JSON 格式错误、顶层结构错误，或不含预期记录；
- 固定序列数据点缺少身份维度，或 AgentX 聚合报告成功请求数为零；
- AgentX 验证错误率超过配置阈值；
- 无法确定性配对 AgentX 聚合和原始工件后缀；
- 一个工作流运行中出现多个数据集 slug，或缺少必需的数据集来源；
- 意外重复自然键必须依赖输入顺序才能决定最终值；
- 未映射的模型、硬件或精度会使目标记录被丢弃；
- 应用摄取报告数据库错误、缺失必需旁表、override 不匹配或数据库验证失败。

不要通过编辑已下载 JSON、更改 ID、抑制跳过或手工插入记录来修复这些情况。应在源头修复生产端、映射、来源或工件选择，再重跑幂等摄取。
