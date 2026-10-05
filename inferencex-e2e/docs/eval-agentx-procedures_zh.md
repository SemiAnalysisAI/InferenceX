# 评估与 AgentX 操作流程

<div align="center">

[English](eval-agentx-procedures.md) | **中文**

</div>


使用本页添加和运行评分 eval、操作 AgentX trace replay、保留证据，并判断长时间运行是否应继续。命令均假定当前目录为 `inferencex-e2e/`；请替换 `<ANGLE_BRACKETS>` 中的值。

若只需对已有服务运行回放客户端，请参阅[独立运行 AgentX-Harness](agentx-standalone_zh.md)。
该指南包含安装步骤和直接执行的 `aiperf profile` 命令，无需 CI 或 Slurm。

## 1. 选择正确的执行模式

若 PR sweep 只需测试吞吐量，请在相应的 `perf-changelog.yaml` 条目中设置
`no-evals: true`，并使用一个主要 sweep label（通常为 `full-sweep-fail-fast`）。这会跳过
这些条目的所有 eval 作业，不改变 benchmark 时长或 Prometheus 产物。
该选项默认为 false，且保留在 changelog 元数据中。其他条目仍可为同一配置
选择 eval；若需完全禁用，请在所有相关条目中设置该选项。设置 `no-evals` 的条目
不能同时设置 `all-evals`、`evals-only` 或 `eval-min-prefill-ep`，PR 也不能带有任一
eval modifier，这些组合都会被拒绝。这类运行提供吞吐量证据，不提供模型评估证据。

这里有两个不同层次：矩阵生成器决定**存在哪些作业**，运行时变量决定**已启动作业执行什么操作**。

| 需求 | 生成器参数（`infx.matrix.generate`）或工作流变量 | 运行时行为 |
|---|---|---|
| 常规 sweep | 不加 eval 选项 | 吞吐量作业，加上选定的 8k/1k eval 子集和 agentic GSM8K 子集 |
| 仅吞吐量 | `--no-evals` | 不生成 eval 作业 |
| 仅选定的 eval 子集 | `--evals-only` | 作业带有 `RUN_EVAL=true`、`EVAL_ONLY=true` |
| 仅运行所有符合条件的 eval | `--all-evals` | 等价于 `--evals-only --all-evals`；包含全部定长序列 8k/1k 行，以及单节点和多节点 agentic GSM8K 行 |
| 在一个 recipe 中先跑吞吐量再跑 eval | `RUN_EVAL=true`、`EVAL_ONLY=false` | 启动服务，运行吞吐量，然后执行 `python3 -m infx.bench eval` |
| 对新启动的服务仅运行 eval | `RUN_EVAL=true`、`EVAL_ONLY=true` | 启动器应用仅评估模式的服务设置，跳过吞吐量并运行评估 |

PR 上的 `all-evals` 标签则通过 [`infx.matrix.plan`](../infx/matrix/plan.py) 生成矩阵，会扩大 eval 选择范围并保留吞吐量作业。

默认选择会区分场景。单节点定长序列 eval 对每个 8k/1k 的模型/runner/framework/precision/并行配置分组选取符合条件的中位和最高并发；多节点 eval 对每种拓扑选取符合条件的最高并发。定长序列中低于 16 的并发不会被选中。所有 AgentX 模型（包括 Kimi K3 和 MiniMax M3）默认都会运行 GSM8K。单节点 agentic 行按模型/runner/framework/precision/spec-decoding/dp-attn/镜像分组，并在每组最高并发处评估：MTP、DP attention 和镜像不同的变体各自单独评估，而 TP/EP 和 KV offloading 不同的变体共用一次评估。多节点 agentic 行对每种拓扑选取符合条件的最高并发；若某个部署没有任何并发不低于 16 的拓扑，则在其最高并发处评估一次。Kimi K3 和 MiniMax M3 的行还会在每个生成的测试点（包括低并发测试点）额外运行厂商评估，因此它们的 GSM8K 是一条额外的 eval-only 行。所有评估都作为独立的 eval-only 作业运行，因此 agentic 吞吐量覆盖范围不变。参见 [`mark_eval_entries()` 和 `mark_all_eval_entries()`](../infx/matrix/generate.py)。

Kimi K3 在 AMD 和 NVIDIA 的单节点及多节点 recipe 上自动运行 `kimi-vendor` / `kimi_tool_call_schema_full`。完整套件对 204 个独立 schema 用例分别执行流式和非流式请求，共产生 408 项检查。快速诊断时，仍可显式设置工作流输入 `eval-framework=kimi-vendor` 和 `eval-suite=kimi_tool_call_schema`，运行一个用例、两项检查的冒烟评估。`--trim-conc` 只裁剪部署测试点，不缩减套件用例数。MiniMax M3 在两家硬件厂商上自动运行 `minimax-vendor` / `minimax_m3_full`，覆盖全部 102 个厂商用例；仍可通过显式覆盖选择单用例 `minimax_m3_smoke`。定长序列的 GSM8K 选择策略保持不变。

解读分数时应同时查看任务名和 `n_eff`：`kimi_tool_call_schema = 1.0, n_eff = 2` 表示一个 schema 用例在两种模式下均通过，不能视为完整套件结果或 GSM8K 分数。历史冒烟产物保留原有标识。Kimi 完整套件仍采用 `0.0` 的质量阈值，仅报告诊断分数；检查项缺失、准备失败和集成错误仍会使作业失败。参见[套件定义与产物约定](../infx/evals/EVALS.md#how)。

Kimi 厂商完整套件不再设置适配器层面的整进程超时。原生验证器的请求超时、引擎就绪等待上限，以及工作流和调度器的资源分配时限仍然生效。冒烟评估保留 900 秒超时；直接调用 Python 适配器时，可通过正数 `--timeout-seconds` 参数为任一套件显式设置超时。

在 PR 上，应将一个主要 sweep label（通常为 `full-sweep-fail-fast`）与 eval modifier 组合使用。`all-evals` 在不抑制吞吐量的情况下扩大覆盖范围；`evals-only` 会抑制吞吐量；两者一起使用时只运行所有符合条件的 eval。带有 `evals-only` 的运行不可复用，而常规 full sweep 和 `all-evals` full sweep 可以复用。添加或移除 modifier 会重启当前 sweep（[label 策略](../../.github/workflows/README.md#pr-eval-modifiers)）。

```bash
# Selected eval subset only
gh pr edit <PR_NUMBER> --repo SemiAnalysisAI/InferenceX \
  --add-label full-sweep-fail-fast --add-label evals-only

# Every eligible eval only
gh pr edit <PR_NUMBER> --repo SemiAnalysisAI/InferenceX \
  --add-label full-sweep-fail-fast --add-label all-evals --add-label evals-only
```

在占用 runner 前预览准确矩阵：

```bash
uv run --no-project --exclude-newer PT12H --python 3.12 --with pydantic --with pyyaml \
  python -m infx.matrix.generate \
  test-config \
  --config-keys qwen3.5-fp8-b200-sglang-agentic \
  --conc 1 \
  --evals-only \
  --config-files configs/nvidia-master.yaml | jq .
```

正确的 AgentX eval 行包含 `"scenario-type": "agentic-coding"`、`"run-eval": true` 和 `"eval-only": true`。工作流会在 [`.github/workflows/e2e-tests.yml`](../../.github/workflows/e2e-tests.yml#L328-L335) 中将生成的行拆分到吞吐量、定长序列 eval 和 agentic eval 作业。

## 2. 添加评分 eval

1. 按照 lm-evaluation-harness task 格式添加 `infx/evals/<task>.yaml`。固定 dataset/split、确定性生成设置、prompt 约定、filter 和主指标。可参考仓库内的 [`gsm8k.yaml`](../infx/evals/gsm8k.yaml) 或 [`gpqa_diamond.yaml`](../infx/evals/gpqa_diamond.yaml)。
2. 为 `task:` 指定稳定名称。分数阈值以该精确名称为键，收集后的行中也会出现该名称。
3. 在 [`infx/evals/thresholds.yaml`](../infx/evals/thresholds.yaml) 中添加最低可接受分数。通用下限放在 `default`；只有在确有依据需要模型专用下限时，才添加 `models.<model-prefix>.<task>`。
4. 如果 task 的主结果与 collector 的 strict/extract/accuracy 规则不兼容，请扩展 [`infx.results.evals`](../infx/results/evals.py) 中的 `extract_metrics()`。该函数接收已加载的 JSON 和显式来源信息；`build_rows()` 应用收集器的分数验证及元数据转换规则。发布为成功结果的行必须具有非 null 的 `score`。
5. 先运行一个显式的小切片并检查样本，再运行完整 split。`EVAL_LIMIT` 是 smoke test 控制项，不是可发布分数的运行设置。

对已经健康的 OpenAI-compatible 服务执行：

```bash
export MODEL='<HF_MODEL_ID>'
export MODEL_NAME='<SERVED_MODEL_NAME>'
export MODEL_PREFIX='<MODEL_PREFIX>'
export PORT='<PORT>'
export EVAL_ONLY=false IS_MULTINODE=false OPENAI_API_KEY=EMPTY
export EVAL_TASKS_DIR='infx/evals/<task>.yaml'
export EVAL_LIMIT='10'
EVAL_DIR="$(mktemp -d /tmp/eval_out-XXXXXX)"
PYTHONSAFEPATH=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python3 -m infx.bench eval \
  --endpoint "http://localhost:$PORT" --concurrency 16 --stage-to "$EVAL_DIR"
python3 -m infx.evals.validate_scores \
  --model-prefix "$MODEL_PREFIX" \
  --meta-env "$EVAL_DIR/meta_env.json" \
  --results-glob "$EVAL_DIR/results*.json"
```

完整 eval 需要取消 limit，并在干净且配置正确的服务上重复执行：

```bash
unset EVAL_LIMIT
EVAL_DIR="$(mktemp -d /tmp/eval_out-XXXXXX)"
PYTHONSAFEPATH=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python3 -m infx.bench eval \
  --endpoint "http://localhost:$PORT" --concurrency 16 --stage-to "$EVAL_DIR"
python3 -m infx.evals.validate_scores --model-prefix "$MODEL_PREFIX" \
  --meta-env "$EVAL_DIR/meta_env.json" --results-glob "$EVAL_DIR/results*.json"
```

请使用 Python 3.10 或更高版本运行这些命令，通常在服务容器内执行，因为 lm-eval 会通过 `uv pip` 把固定版本的 harness 安装到该 `python3` 中。该命令会把允许列表中的产物复制到 `--stage-to`，并在该目录写入 `meta_env.json`。并发取自 `--concurrency`，lm-eval 通过 `--model_args` 中的 `num_concurrent` 接收该值。命令不再读取 `EVAL_CONCURRENT_REQUESTS`。准确调用见 [`infx.bench.eval.lm_eval.run`](../infx/bench/eval/lm_eval.py#L121-L150)。

## 3. `EVAL_ONLY` 是 launcher 约定

必须在**启动服务前**设置 `EVAL_ONLY=true`。它不仅是评估命令内部的开关：

1. 对单节点定长序列作业，srt binder 会把服务上下文设为矩阵中的 `MAX_MODEL_LEN`（`isl + osl + 256`），SGLang 使用 `context-length`，TRT-LLM 使用 `max_seq_len` 和 `max_num_tokens`，vLLM 与 ATOM 使用 `max-model-len`。AgentX 测试点和多节点作业保留配方自身的上下文，多节点作业还可选择用于真实验证的 `EVAL_CONFIG_FILE`。
2. 仍会运行健康检查。在仅评估作业中，厂商评估框架还会在 `EVAL_ENDPOINT_READY_TIMEOUT_SECONDS` 内等待服务模型出现在 OpenAI chat 路由上。
3. 跳过吞吐量测试。
4. `python3 -m infx.bench eval` 按 `EVAL_MAX_MODEL_LEN` 确定每个 lm-eval 请求的预算。未设置时使用 `MAX_MODEL_LEN`，并以模型原生上限为界。
5. 同一命令会暂存产物并写入 `meta_env.json`，无论评估成功还是失败。

相关实现：[服务上下文](../infx/srt_slurm/single_node.py#L183-L194)、[请求预算](../infx/bench/eval/lm_eval.py#L77-L98)、[评估分派与失败策略](../infx/bench/eval/__init__.py#L74-L173) 和[工作流输入](../../.github/workflows/benchmark-tmpl.yml#L36-L53)。

原生多节点 post-eval 从 `/model` 读取挂载的检查点，并仅在评估进程中启用数据集下载，不改变工作进程环境。上下文查询先读取本地 `config.json` 中的数值上限，再回退到 Transformers；显式设置的 `EVAL_MAX_MODEL_LEN` 仍优先。

不要在吞吐量规格的服务已经运行后才切换 `EVAL_ONLY`，并假定 context 会随之变化。应通过 recipe 重启。Eval-only 模式会在暂存已有 artifact 后返回 eval 失败；在工作流中，上传步骤使用 `always()`，并位于分数校验前，因此失败证据仍会保留（[单节点上传与 gate](../../.github/workflows/benchmark-tmpl.yml#L449-L472)、[多节点上传与 gate](../../.github/workflows/benchmark-multinode-tmpl.yml#L477-L503)）。

## 4. 批量 eval 并发

空格分隔的 `--concurrency` 值会让多个并发点在**同一个存活的 engine 上依次执行**。多节点作业以这种方式传入 `EVAL_CONC`。它不会同时运行多个 harness。每个并发点内部，harness 最多发出该并发数的请求。

```bash
export MODEL='<HF_MODEL_ID>' MODEL_NAME='<SERVED_MODEL_NAME>' MODEL_PREFIX='<MODEL_PREFIX>'
export PORT='<PORT>' EVAL_TASKS_DIR='infx/evals/gsm8k.yaml'
export EVAL_ONLY=false IS_MULTINODE=false OPENAI_API_KEY=EMPTY
EVAL_DIR="$(mktemp -d /tmp/eval_out-XXXXXX)"
PYTHONSAFEPATH=1 PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python3 -m infx.bench eval \
  --endpoint "http://localhost:$PORT" --concurrency '16 32 64' --stage-to "$EVAL_DIR"
python3 -m infx.evals.validate_scores --expected-concs '16 32 64' \
  --meta-env "$EVAL_DIR/meta_env.json" --results-glob "$EVAL_DIR/results*.json"
```

批量 runner 会为每个点创建新的临时输出目录，用 `_conc<N>` 后缀暂存文件，并向 `meta_env.json` 写入以下数组：

- `eval_concs`：请求的点；
- `completed_eval_concs`：评估成功且至少暂存了一个产物的点；
- `failed_eval_concs`：评估或其暂存失败，或未暂存任何产物的点。

失败点会延迟报错，使所有已尝试点的 artifact 都能上传；随后 post-upload validator 会使作业失败。批量模式只接受正整数，且仅支持 `lm-eval`。参见[批量执行](../infx/bench/eval/__init__.py#L176-L211)、[产物后缀处理](../infx/bench/eval/stage.py#L20-L43) 和[manifest 校验](../infx/evals/validate_scores.py#L119-L216)。

对于多节点 `all-evals`，工作流通过连接拓扑的并发列表构造 `EVAL_CONC`（[分派](../../.github/workflows/e2e-tests.yml#L390-L392)）。如果缺少某点的 `_conc<N>` 结果或 completed manifest 条目，绝不能比较该点。

## 5. 校验分数，而不只是检查文件存在

运行：

```bash
python3 -m infx.evals.validate_scores \
  --thresholds infx/evals/thresholds.yaml \
  --meta-env meta_env.json \
  --results-glob 'results*.json'
```

批量运行还应加入独立确认的预期点：

```bash
python3 -m infx.evals.validate_scores \
  --expected-concs '16 32 64' \
  --thresholds infx/evals/thresholds.yaml
```

阈值按以下顺序解析：`models.<prefix>.<task>`、`default.<task>`，最后是 `--min-score`（默认 `0.85`）。默认检查名称以 `exact_match,` 开头、数值类型且非 stderr 的指标。当分数低于阈值、没有匹配指标、缺少请求的并发、metadata 有重复或无效值、任何点被标记为失败，或结果后缀与 manifest 不一致时，校验都会失败。当前权威下限位于 [`thresholds.yaml`](../infx/evals/thresholds.yaml)。参见[阈值解析](../infx/evals/validate_scores.py#L63-L73)与[校验流程](../infx/evals/validate_scores.py#L219-L373)。

手动的吞吐量+eval 组合 recipe 会上传 eval 输出，但模板的自动分数 gate 专用于 eval-only 作业。对手动或组合运行必须显式执行 validator。

## 6. 收集并检查 eval artifact

收集工作流会下载 `eval_*`，用 `infx/results/collect_eval_results.py` 聚合原始集合，上传 `eval_results_all/agg_eval_all.json`，并将表格写入 step summary（[`collect-evals.yml`](../../.github/workflows/collect-evals.yml)）。

```bash
RUN_ID='<RUN_ID>'
gh run download "$RUN_ID" --repo SemiAnalysisAI/InferenceX \
  --name eval_results_all --dir ./evals
jq -r '.[] | [.hw, .framework, .precision, .tp, .conc, .task,
  (.score * 100 | round | . / 100)] | @tsv' \
  ./evals/agg_eval_all.json | column -t
jq '[.[] | select(.hw == "B200")]' ./evals/agg_eval_all.json
```

当 aggregate 缺失或可疑时，下载原始证据：

```bash
gh run download "$RUN_ID" --repo SemiAnalysisAI/InferenceX \
  --pattern 'eval_*' --dir ./evals/raw
```

保留 `meta_env.json`、`results*.json` 和 `sample*.jsonl`。Aggregate 是导航工具，不能替代原始样本与 batch 完整性证据。

## 7. 运行 AgentX：快速反馈与 canonical 证据

`python3 -m infx.bench agentic` 会用 uv 自行构建客户端运行时（[`infx/bench/agentic/venv.py`](../infx/bench/agentic/venv.py)）。它在 `AIPERF_RUNTIME_DIR` 下新建 Python 3.11 venv（默认 `<tmp>/inferencex-agentic-<SLURM_JOB_ID 或 PID>`），以可编辑模式安装 `utils/aiperf` 及其声明的依赖，并安装 AIPerf 未声明的 client 依赖（[`requirements.txt`](../infx/bench/agentic/requirements.txt)）。随后它会在该 venv 的 Python 下重新运行自身。Recipe 通过 [`benchmarks/srt_agentic.sh`](../benchmarks/srt_agentic.sh) 调用它。

AgentX 是 AIPerf `agentx` trace replay，不是固定 token 的合成 benchmark。`agentx` scenario 负责 replay 默认值：每条 trajectory lane 额外执行十个 warmup 请求、warmup 排空上限为 1,800 秒、实时失败阈值为 0.10、trace 空闲上限为 300 秒。Recipe 可以用 `AGENTIC_WARMUP_GRACE_PERIOD` 提高排空上限，或用 `AIPERF_LIVE_FAILED_REQUEST_THRESHOLD` 放宽实时中止阈值；完成后的 profile 错误率超过 0.10 时仍会校验失败（[运行后校验](../infx/bench/agentic/run.py#L36-L38)）。Profile 使用配置的时长。`agentx-fast` 强制每条 lane 只运行一个 warmup 请求，并将 profile 设为 1,200 秒。它只影响单节点和多节点 AgentX 吞吐量；定长序列吞吐量与 eval 保持 canonical。Fast 运行不符合 artifact reuse 条件（[工作流策略](../../.github/workflows/README.md#agentx-fast-mode)、[fast replay 设置](../infx/bench/agentic/replay.py#L64-L65)）。

每个 AgentX 吞吐量并发点都必须使用新启动的服务。矩阵为每个点生成独立作业。`infx.launch` 会拒绝 `CONC_LIST` 不恰好等于其唯一正整数 `CONC` 的多节点 AgentX 吞吐量作业，replay client 也会拒绝与 `CONC` 不同的 `CONC_LIST`。AgentX 不清空缓存，也不复用正在运行的服务来测试另一个并发点。同一测试点的预热和正式测量共用服务。此规则不改变定长序列 sweep 或评分 eval 的批量执行行为。

对于多节点 srt-slurm 作业，benchmark client 与 frontend 可能运行在不同主机上。只要设置了 `SRT_FRONTEND_HOST`，replay 就以 `http://$SRT_FRONTEND_HOST:$SRT_FRONTEND_PORT` 为目标，否则使用显式提供的 `AIPERF_SERVER_URL`，两者都没有时才回退到 `http://localhost:$PORT`（[`_server_url`](../infx/bench/agentic/replay.py#L115-L122)）。

对于未发布到 package index 的 engine 或 router wheel，必须保证构建可复现且 artifact 不可变：在 launcher 旁签入源码 patch 与构建器，打 patch 前校验上游 wheel 的 digest，分配明确的 local version，并通过带 SHA256 fragment 的精确 URL 安装已发布 artifact。本地 backport 不得冒用尚未发布的上游版本号。

目标 canonical 运行（使用配置的 duration 和 warmup；不要加 fast 或 duration override）：

```bash
REF='<BRANCH_OR_SHA>'
gh workflow run e2e-tests.yml --repo SemiAnalysisAI/InferenceX --ref "$REF" \
  -f generate-cli-command='test-config --config-keys qwen3.5-fp8-b200-sglang-agentic --conc 1 --config-files configs/nvidia-master.yaml' \
  -f test-name='agentx-canonical-qwen35-c1'
```

快速诊断运行：

```bash
gh workflow run e2e-tests.yml --repo SemiAnalysisAI/InferenceX --ref "$REF" \
  -f generate-cli-command='test-config --config-keys qwen3.5-fp8-b200-sglang-agentic --conc 1 --config-files configs/nvidia-master.yaml' \
  -f test-name='agentx-fast-qwen35-c1' \
  -f agentx-fast=true
```

Fast 结果只能作为 bring-up 证据，绝不能替代 canonical candidate。小于 900 秒的 duration 会添加 AIPerf 的 `--unsafe-override` 并将 submission 标记为无效；只能用于 smoke 诊断（[源码](../infx/bench/agentic/replay.py#L105)）。Fast 运行健康后，必须对完全相同的 candidate 进行 canonical 运行，才能宣称 benchmark 成功。

## 8. 保留 trace 与运行 provenance

AgentX 默认 replay 已记录的 assistant response。实时服务输出会被测量，但构造后续 turn 时会丢弃。除非用 `WEKA_LOADER_OVERRIDE` 固定为 `semianalysis_cc_traces_weka_062126` 或 `semianalysis_cc_traces_weka_062126_256k`，否则所选 trace corpus 依赖模型 family；resolver 会同时记录 loader 与 Hugging Face dataset（[trace 解析](../infx/bench/agentic/traces.py#L20-L25)、[replay 语义](../infx/bench/agentic/replay.py#L150-L185)）。Replay 保留模型的原生上下文。客户端忽略 `MAX_MODEL_LEN`，只有显式设置的 `AIPERF_MAX_CONTEXT_LENGTH` 才会添加 AIPerf 的 `--max-context-length`。

立即记录 orchestration provenance：

```bash
RUN_ID='<RUN_ID>'
gh run view "$RUN_ID" --repo SemiAnalysisAI/InferenceX \
  --json url,headSha,headBranch,event,status,conclusion,createdAt,updatedAt,jobs \
  > run-provenance.json
```

下载 AgentX 证据：

```bash
gh run download "$RUN_ID" --repo SemiAnalysisAI/InferenceX \
  --pattern 'bmk_agentic_*' --dir ./agentx/aggregate
gh run download "$RUN_ID" --repo SemiAnalysisAI/InferenceX \
  --pattern 'agentic_*' --dir ./agentx/raw
gh run download "$RUN_ID" --repo SemiAnalysisAI/InferenceX \
  --pattern '*server_logs_*' --dir ./agentx/server-logs
gh run download "$RUN_ID" --repo SemiAnalysisAI/InferenceX \
  --pattern 'gpu_metrics_*' --dir ./agentx/gpu
```

每个并发点都应保留：

- `benchmark_command.txt`（准确 AIPerf 命令）和 `benchmark.log`；
- AIPerf `profile_export*`、`server_metrics_export.json`、plot 和 distribution analysis；
- aggregate JSON 及其 `dataset` 对象（`source_type`、loader、HF dataset/split、entry count）；
- server/frontend 日志以及所代表的每个 metrics endpoint；
- run URL/ID、attempt、head SHA、recipe/config 标识、image、topology、fast 标志和所有 override。

Runner 会在 replay 前写入命令，并在聚合后校验原始结果（[执行路径](../infx/bench/agentic/run.py#L135-L218)）。聚合会保留 dataset provenance 以及硬件/模型/拓扑字段（[aggregate 构造](../infx/results/agentic/__init__.py)）。工作流的 raw upload 会有意排除体积很大的 `inputs.json` 和 `profile_export_raw.jsonl`；如果调查需要这些文件，应在清理前从实时 allocation 保存（[单节点 artifact 约定](../../.github/workflows/benchmark-tmpl.yml#L382-L391)、[多节点约定](../../.github/workflows/benchmark-multinode-tmpl.yml#L466-L475)）。

## 9. 用实时证据调试长时间 AgentX 运行

GitHub Actions 是 orchestration/最终状态视图；cluster 是实时诊断来源。从 InferenceX Clusters canvas 获取 SSH alias、runner user 和受访问控制的路径。绝不要猜测或公开私有基础设施坐标。

解析准确的矩阵作业：

```bash
gh run view <RUN_ID> --repo SemiAnalysisAI/InferenceX --json jobs \
  --jq '.jobs[] | select(.name | test("agentic|AgentX"; "i")) |
        [.databaseId, .status, .conclusion, .name] | @tsv'
```

在 controller 上识别并核实 allocation：

```bash
squeue -u <RUNNER_USER> -o "%.8i %.8T %.10M %.20N %.100j"
scontrol show job -o <SLURM_JOB_ID> | tr " " "\n" | \
  grep -E '^(JobId|JobState|RunTime|TimeLimit|NodeList|WorkDir)='
```

根据 `WorkDir` 推导 `<LOG_DIR>`；srt-slurm 通常使用 `<WorkDir>/outputs/<SLURM_JOB_ID>/logs/`。先清点，再选择文件：

```bash
find "<LOG_DIR>" -maxdepth 1 -type f -print | sort
```

始终从头包含 custom benchmark 日志，然后加入所有与拓扑相关的 backend 和 frontend/router 日志：

```bash
ssh <CLUSTER_ALIAS> 'tail -f -n+1 "<LOG_DIR>/benchmark.out"'
tail -F -n+1 <BENCHMARK_LOG> <FRONTEND_LOG> <SERVER_LOGS...>
rg -n -i 'Phase |warmup|profiling|returned=|in_flight=|queue=|kv_usage=|prefix_cache_hit=|tput_|ERROR|Traceback|OOM|NCCL|RCCL|timeout|connection refused' <LOGS...>
```

拓扑规则：

- Aggregated：检查每个 aggregate backend；attention DP 可能暴露多个 metrics source，但它不是 disaggregation。
- Disaggregated：检查每个 prefill backend、每个 decode backend 以及 frontend/router。Decode pool 健康不能证明 prefill/KV transfer 健康。
- 确认 AIPerf 命令包含所有 `AIPERF_SERVER_METRICS_URLS`；缺少 endpoint 会产生片面而虚假的健康证据。

Summary 不明确时直接读取每个 endpoint：

```bash
curl -fsS '<METRICS_URL>' | \
  rg -i 'request|queue|cache|token|prefill|decode|error|fail'
```

通过重复 sample 跟踪趋势：running/waiting request、KV usage、prefix hit、input/output token rate、completed/cancelled/errored request、frontend routing balance，以及 disaggregated KV transfer。AIPerf 会为每条 server series 记录 endpoint identity（[metrics 接线](../infx/bench/agentic/replay.py#L125-L147)）。未设置 `AIPERF_SERVER_METRICS_URLS` 且 `SRTCTL_FRONTEND_TYPE` 不是 `dynamo` 时，replay 会从 `SRT_AGG_ENDPOINTS`，或从 `SRT_PREFILL_ENDPOINTS` 加 `SRT_DECODE_ENDPOINTS`，抓取每个 worker 的 `/metrics`。

应使用 phase marker，而不是 Slurm 总运行时间：

```bash
grep -E 'Phase warmup progress|WARMUP cache pressure|Phase warmup complete|Phase profiling started|Phase profiling complete|process_agentic_result' \
  "<LOG_DIR>/benchmark.out"
date -u
```

报告 phase 已用/剩余时间、最后日志更新时间、错误数、request/queue/KV 趋势、已检查的文件和 metric source，并分别给出预计 benchmark 完成时间与预计 GitHub 完成时间。在必需 artifact 上传且工作流接受它们之前，运行不能算 green。

## 10. 提前终止规则

当直接证据已经足以判定配置不合格时，应建议提前停止：

- 确定性 OOM、NCCL/RCCL 失败、parser crash 或 worker 缺失；
- 多次重复 sample 中 counter 与日志时间戳均没有前进；
- KV usage 长期接近 100%，queue 持续增长且 latency 已不可用；
- 吞吐量已经平台化，而更高并发只会恶化 TTFT/TPOT；
- 任意 disaggregated pool 或必需 metrics source 始终未注册；
- AIPerf 校验显示 completed request 为零，或错误率超过配置的 `0.10` 上限（[validator](../infx/results/agentic/validate_agentic_result.py#L48-L88)）。

如果 completion 持续增加且 queue 稳定，不要仅因模型加载、dataset 配置、warmup、cutoff drain 或 profiling 较慢而停止。任何取消前，都要捕获时间戳、准确拓扑、相关日志行、至少两个体现趋势的 metric sample、当前 phase 和诊断结论。

取消操作会修改共享基础设施。除非当前任务已明确授权，否则必须先询问。优先从 GitHub 取消，以便工作流执行 cleanup：

```bash
gh run cancel <RUN_ID> --repo SemiAnalysisAI/InferenceX
```

只有在获得明确批准且有具体理由时才使用 `scancel` 或终止进程；否则可能绕过 cleanup 或使 runner 残留。修复 recipe 后，先分派一个目标 fast e2e 点并实时检查，只有通过检查的 candidate 才值得进行 canonical 运行/完整 sweep。

## 完成检查清单

- 矩阵预览符合预期 scenario、topology、eval mode 与 concurrency。
- 完整 eval 未设置 `EVAL_LIMIT`；每个预期 batch 点都已完成且有带后缀的结果。
- `validate_scores.py` 针对预期 task/model 阈值通过。
- Aggregate 与 raw eval/AgentX artifact 均已下载且内部一致。
- 已记录 AgentX corpus、replay mode、准确命令、commit、image、recipe、topology 以及 fast/override 状态。
- 每个 backend/frontend 与 metrics source 都在实时证据中有所体现。
- Fast/smoke 结果明确标为诊断用途；只有 canonical candidate 用于最终比较。
- 在报告成功前，工作流与 artifact collection 均已得出 green 结论。
