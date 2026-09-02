# 评估与 AgentX 操作流程

<div align="center">

[English](./eval-agentx-procedures.md) | **中文**

</div>


使用本页添加和运行评分 eval、操作 AgentX trace replay、保留证据，并判断长时间运行是否应继续。命令均假定当前目录为仓库根目录；请替换 `<ANGLE_BRACKETS>` 中的值。

## 1. 选择正确的执行模式

这里有两个不同层次：矩阵生成器决定**存在哪些作业**，运行时变量决定**已启动作业执行什么操作**。

| 需求 | 生成器/工作流模式 | 运行时行为 |
|---|---|---|
| 常规 sweep | 不加 eval 选项 | 吞吐量作业，加上选定的 8k/1k eval 子集 |
| 仅吞吐量 | `--no-evals` | 不生成 eval 作业 |
| 仅选定的 eval 子集 | `--evals-only` | 作业带有 `RUN_EVAL=true`、`EVAL_ONLY=true` |
| 仅运行所有符合条件的 eval | `--all-evals` | 等价于 `--evals-only --all-evals`；包含全部定长序列 8k/1k 行，以及单节点和多节点 agentic GSM8K 行 |
| 在一个 recipe 中先跑吞吐量再跑 eval | `RUN_EVAL=true`、`EVAL_ONLY=false` | 启动服务，运行吞吐量，然后执行 `run_eval` |
| 对新启动的服务仅运行 eval | `RUN_EVAL=true`、`EVAL_ONLY=true` | launcher 扩大 eval context，跳过吞吐量并运行 eval |

默认选择会区分场景。单节点定长序列 eval 对每个 8k/1k 的模型/runner/framework/precision/并行配置分组选取符合条件的中位和最高并发；多节点 eval 对每种拓扑选取符合条件的最高并发。低于 16 的并发不会被选中。Agentic eval 需要显式启用；单节点和多节点 agentic 行均选择符合条件的最高并发。参见 [`mark_eval_entries()`](../utils/matrix_logic/generate_sweep_configs.py#L276-L396) 与 [`mark_all_eval_entries()`](../utils/matrix_logic/generate_sweep_configs.py#L399-L482)。

在 PR 上，应将一个主要 sweep label（通常为 `full-sweep-fail-fast`）与 eval modifier 组合使用。`all-evals` 在不抑制吞吐量的情况下扩大覆盖范围；`evals-only` 会抑制吞吐量；两者一起使用时只运行所有符合条件的 eval。带有 `evals-only` 的运行不可复用，而常规 full sweep 和 `all-evals` full sweep 可以复用。添加或移除 modifier 会重启当前 sweep（[label 策略](../.github/workflows/README.md#pr-eval-modifiers)）。

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
uv run --no-project --with pydantic --with pyyaml --python 3.12 \
  utils/matrix_logic/generate_sweep_configs.py \
  test-config \
  --config-keys qwen3.5-fp8-b200-sglang-agentic \
  --conc 1 \
  --evals-only \
  --config-files configs/nvidia-master.yaml | jq .
```

正确的 AgentX eval 行包含 `"scenario-type": "agentic-coding"`、`"run-eval": true` 和 `"eval-only": true`。工作流会在 [`.github/workflows/e2e-tests.yml`](../.github/workflows/e2e-tests.yml#L278-L293) 中将生成的行拆分到吞吐量、定长序列 eval 和 agentic eval 作业。

## 2. 添加评分 eval

1. 按照 lm-evaluation-harness task 格式添加 `utils/evals/<task>.yaml`。固定 dataset/split、确定性生成设置、prompt 约定、filter 和主指标。可参考仓库内的 [`gsm8k.yaml`](../utils/evals/gsm8k.yaml) 或 [`gpqa_diamond.yaml`](../utils/evals/gpqa_diamond.yaml)。
2. 为 `task:` 指定稳定名称。分数阈值以该精确名称为键，收集后的行中也会出现该名称。
3. 在 [`utils/evals/thresholds.yaml`](../utils/evals/thresholds.yaml) 中添加最低可接受分数。通用下限放在 `default`；只有在确有依据需要模型专用下限时，才添加 `models.<model-prefix>.<task>`。
4. 如果 task 的主结果与 collector 的 strict/extract/accuracy 规则不兼容，请扩展 [`extract_lm_metrics()`](../utils/collect_eval_results.py#L115-L197)。不要发布 `score` 为 null 的行。
5. 先运行一个显式的小切片并检查样本，再运行完整 split。`EVAL_LIMIT` 是 smoke test 控制项，不是可发布分数的运行设置。

对已经健康的 OpenAI-compatible 服务执行：

```bash
source benchmarks/benchmark_lib.sh
export MODEL='<HF_MODEL_ID>'
export MODEL_NAME='<SERVED_MODEL_NAME>'
export MODEL_PREFIX='<MODEL_PREFIX>'
export PORT='<PORT>'
export EVAL_TASKS_DIR='utils/evals/<task>.yaml'
export EVAL_CONCURRENT_REQUESTS='16'
export EVAL_LIMIT='10'
run_eval --framework lm-eval --port "$PORT"
append_lm_eval_summary
python3 utils/evals/validate_scores.py \
  --model-prefix "$MODEL_PREFIX" \
  --results-glob 'results*.json'
```

完整 eval 需要取消 limit，并在干净且配置正确的服务上重复执行：

```bash
unset EVAL_LIMIT
run_eval --framework lm-eval --port "$PORT"
append_lm_eval_summary
python3 utils/evals/validate_scores.py --model-prefix "$MODEL_PREFIX"
```

`run_lm_eval` 通过 `--model_args` 中的 `num_concurrent` 传递并发；它刻意采用环境变量，而不是 `run_eval` CLI 选项。准确调用见 [`run_lm_eval()`](../benchmarks/benchmark_lib.sh#L1080-L1162)。

## 3. `EVAL_ONLY` 是 launcher 约定

必须在**启动服务前**设置 `EVAL_ONLY=true`。它不仅是 `run_eval` 内部的开关：

1. `compute_eval_context_length`/`setup_eval_context` 选择请求的 eval context，并以模型原生上限为界。
2. launcher 将其连接到服务参数（`--context-length`、`--max-model-len` 或 framework 对应参数）。
3. 仍会运行健康检查。
4. 吞吐量路径立即返回或被跳过。
5. 运行 `run_eval` 和 artifact staging。

相关实现：[context 设置](../benchmarks/benchmark_lib.sh#L1049-L1078)、[eval 分派与失败策略](../benchmarks/benchmark_lib.sh#L1789-L1923) 和[工作流输入](../.github/workflows/benchmark-tmpl.yml#L79-L97)。

不要在吞吐量规格的服务已经运行后才切换 `EVAL_ONLY`，并假定 context 会随之变化。应通过 recipe 重启。Eval-only 模式会在暂存已有 artifact 后返回 eval 失败；在工作流中，上传步骤使用 `always()`，并位于分数校验前，因此失败证据仍会保留（[单节点上传与 gate](../.github/workflows/benchmark-tmpl.yml#L399-L417)、[多节点上传与 gate](../.github/workflows/benchmark-multinode-tmpl.yml#L466-L488)）。

## 4. 批量 eval 并发

空格分隔的 `EVAL_CONCURRENT_REQUESTS` 会让多个并发点在**同一个存活的 engine 上依次执行**，而不是同时运行多个 harness。每个并发点内部，harness 最多发出该并发数的请求。

```bash
source benchmarks/benchmark_lib.sh
export MODEL='<HF_MODEL_ID>' MODEL_NAME='<SERVED_MODEL_NAME>' MODEL_PREFIX='<MODEL_PREFIX>'
export PORT='<PORT>' EVAL_TASKS_DIR='utils/evals/gsm8k.yaml'
export EVAL_CONCURRENT_REQUESTS='16 32 64'
run_eval --framework lm-eval --port "$PORT"
append_lm_eval_summary
python3 utils/evals/validate_scores.py --expected-concs '16 32 64'
```

批量 runner 会为每个点创建新的临时输出目录，用 `_conc<N>` 后缀暂存文件，并向 `meta_env.json` 写入以下数组：

- `eval_concs`：请求的点；
- `completed_eval_concs`：eval 与 staging 均成功的点；
- `failed_eval_concs`：eval 或 staging 失败的点。

失败点会延迟报错，使所有已尝试点的 artifact 都能上传；随后 post-upload validator 会使作业失败。批量模式只接受正整数，且仅支持 `lm-eval`。参见 [`run_eval` batching](../benchmarks/benchmark_lib.sh#L1839-L1900)、[artifact 后缀处理](../benchmarks/benchmark_lib.sh#L1163-L1222) 和[manifest 校验](../utils/evals/validate_scores.py#L72-L171)。

对于多节点 `all-evals`，工作流通过连接拓扑的并发列表构造 `EVAL_CONC`（[分派](../.github/workflows/e2e-tests.yml#L397-L400)）。如果缺少某点的 `_conc<N>` 结果或 completed manifest 条目，绝不能比较该点。

## 5. 校验分数，而不只是检查文件存在

运行：

```bash
python3 utils/evals/validate_scores.py \
  --thresholds utils/evals/thresholds.yaml \
  --meta-env meta_env.json \
  --results-glob 'results*.json'
```

批量运行还应加入独立确认的预期点：

```bash
python3 utils/evals/validate_scores.py \
  --expected-concs '16 32 64' \
  --thresholds utils/evals/thresholds.yaml
```

阈值按以下顺序解析：`models.<prefix>.<task>`、`default.<task>`，最后是 `--min-score`（默认 `0.85`）。默认检查名称以 `exact_match,` 开头、数值类型且非 stderr 的指标。当分数低于阈值、没有匹配指标、缺少请求的并发、metadata 有重复或无效值、任何点被标记为失败，或结果后缀与 manifest 不一致时，校验都会失败。当前权威下限位于 [`thresholds.yaml`](../utils/evals/thresholds.yaml)。参见[阈值解析](../utils/evals/validate_scores.py#L61-L69)与[校验流程](../utils/evals/validate_scores.py#L174-L302)。

手动的吞吐量+eval 组合 recipe 会上传 eval 输出，但模板的自动分数 gate 专用于 eval-only 作业。对手动或组合运行必须显式执行 validator。

## 6. 收集并检查 eval artifact

收集工作流会下载 `eval_*`，用 `utils/collect_eval_results.py` 聚合原始集合，上传 `eval_results_all/agg_eval_all.json`，并将表格写入 step summary（[`collect-evals.yml`](../.github/workflows/collect-evals.yml)）。

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

保留 `meta_env.json`、`results*.json` 和 `sample*.jsonl`。Agentic SWE-bench 在单节点模板中还会上传 `agent_preds.json`、`predictions.jsonl`、`swebench_report_*.json` 和 trajectory 文件。Aggregate 是导航工具，不能替代原始样本与 batch 完整性证据。

## 7. 运行 AgentX：快速反馈与 canonical 证据

AgentX 是 AIPerf `inferencex-agentx-mvp` trace replay，不是固定 token 的合成 benchmark。仓库默认设置对每条 trajectory lane 额外执行十个 warmup 请求，并使用 recipe 配置的 profile 时长。`agentx-fast` 强制每条 lane 只运行一个 warmup 请求，并将 profile 设为 1,200 秒。它只影响单节点和多节点 AgentX 吞吐量；定长序列吞吐量与 eval 保持 canonical。Fast 运行不符合 artifact reuse 条件（[工作流策略](../.github/workflows/README.md#agentx-fast-mode)、[fast replay 设置](../benchmarks/benchmark_lib.sh#L2104-L2128)）。

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

对于受控的 Kimi-K3 MI355X 并发 1 实验，手动输入
`k3-perf-variant` 用于选择彼此隔离的运行时实验分支，同时保持生成的
recipe 不变。`m7tunedgemm` 分支只加入通过 changed-input graph replay
和严格 tuner 门槛的五条精确 M=7 BF16 配置。运行时必须固定分支 SHA，保持
`agentx-fast=false`，将 duration override 设为 1,200 秒，并在提升这些配置
之前用匹配 baseline 对原始请求记录进行比较。`tritonmla` 分支只禁用 AITER
MLA，使目标模型解码自动选择 `TRITON_MLA`；其余 AITER ROCm 路径保持启用，
并且只有在服务日志明确证明后端选择成功时才继续运行。对于亚毫秒级结果，
`m7tunedgemmaba` 会在同一个独占 Slurm 分配中依次运行全新 baseline、精确
M=7 候选方案和第二个全新 baseline，每个阶段运行 1,200 秒。候选方案保留
标准结果目录结构，两个 baseline 以及节点和阶段清单保存在
`results/same_node_aba/` 下。结构性实验 `spec3aba` 使用相同的
baseline/candidate/baseline 流程，只把并发 1 的 draft 数量从 6 改为 3，
同时保持合成接受长度 3.75。它会先执行 baseline/candidate 启动冒烟测试，
为每次启动隔离 Triton、TorchInductor、AITER JIT 和 Python 缓存，并在启动
下一台服务前验证 GPU、进程和共享内存都已清理。由于 Enroot 分配和主机监控
进程可能在服务启动前已经持有 KFD，harness 会先要求低 VRAM 且不存在 vLLM
进程，记录 KFD 和共享内存基线，并要求每个阶段恢复到完全相同的状态。baseline
捕获 2 到 14 的图尺寸，三 draft 候选方案捕获 2 到 8。符合 AgentX 规则的
`spec5goldenaba` 模式则比较使用黄金接受长度 `3.75` 的六 draft baseline 与
使用已提交黄金接受长度 `3.62` 的五 draft 候选方案，随后再运行一个全新的六
draft baseline。其余并发 1 设置保持不变，五 draft 候选方案捕获 2 到 12 的
图尺寸。当该深度通过归一化接受 token 效率门槛后，`spec5m6tunedaba` 会让
所有阶段都保持五 draft 和黄金接受长度 `3.62`，并比较镜像内置 BF16 配置与
运行 `33591999396` 选出的五条精确 M=6 gfx950 配置。launcher 只有在校验配置
校验和、AITER v0.1.19 源码哈希、gfx950 架构、256 CU 数量，以及确认镜像内置
配置尚未包含这些行后，才会生成临时运行时配置。同节点 A/B/A 模式的每个阶段
统一使用 `0.85` GPU 显存利用率，为重复启动 ROCm 服务保留足够的初始化空间；
配对比较中的该设置保持不变。

目标 AgentX SWE-bench smoke eval（前十个 instance，真实 agentic generation）：

```bash
gh workflow run e2e-tests.yml --repo SemiAnalysisAI/InferenceX --ref "$REF" \
  -f generate-cli-command='test-config --config-keys qwen3.5-fp8-b200-sglang-agentic --conc 1 --evals-only --config-files configs/nvidia-master.yaml' \
  -f test-name='swebench-smoke-qwen35-c1' \
  -f eval-framework=swebench \
  -f eval-limit='10' \
  -f swebench-gen-mode='agentic'
```

要得到可发布的 SWE-bench 分数，省略 `eval-limit`；不要使用 `single-shot`，它只是诊断逃生选项。SWE-bench generation/scoring 控制项以及完整 split 的 `0.50` 阈值在实现旁的 [`utils/evals/EVALS.md`](../utils/evals/EVALS.md#swe-bench-lite---framework-swebench) 中说明。

Fast 结果只能作为 bring-up 证据，绝不能替代 canonical candidate。小于 900 秒的 duration 或 `AIPERF_UNSAFE_OVERRIDE=true` 会添加 AIPerf 的 `--unsafe-override` 并将 submission 标记为无效；只能用于 smoke 诊断（[源码](../benchmarks/benchmark_lib.sh#L2266-L2268)）。Fast 运行健康后，必须对完全相同的 candidate 进行 canonical 运行，才能宣称 benchmark 成功。

## 8. 保留 trace 与运行 provenance

AgentX 默认 replay 已记录的 assistant response。实时服务输出会被测量，但构造后续 turn 时会丢弃。只有在明确要进行不同的 live-assistant 实验时，才设置 `AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=1`。除非用 `WEKA_LOADER_OVERRIDE` 固定，否则所选 trace corpus 依赖模型 family；resolver 会同时记录 loader 与 Hugging Face dataset（[trace 解析](../benchmarks/benchmark_lib.sh#L2023-L2102)、[replay 语义](../benchmarks/benchmark_lib.sh#L2104-L2270)）。

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

Runner 会在 replay 前写入命令，并在聚合后校验原始结果（[执行路径](../benchmarks/benchmark_lib.sh#L2320-L2360)）。聚合会保留 dataset provenance 以及硬件/模型/拓扑字段（[aggregate 构造](../utils/agentic/aggregation/process_agentic_result.py#L194-L272)）。工作流的 raw upload 会有意排除体积很大的 `inputs.json` 和 `profile_export_raw.jsonl`；如果调查需要这些文件，应在清理前从实时 allocation 保存（[单节点 artifact 约定](../.github/workflows/benchmark-tmpl.yml#L349-L358)、[多节点约定](../.github/workflows/benchmark-multinode-tmpl.yml#L455-L464)）。

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

通过重复 sample 跟踪趋势：running/waiting request、KV usage、prefix hit、input/output token rate、completed/cancelled/errored request、frontend routing balance，以及 disaggregated KV transfer。AIPerf 会为每条 server series 记录 endpoint identity（[metrics 接线](../benchmarks/benchmark_lib.sh#L2236-L2260)）。

应使用 phase marker，而不是 Slurm 总运行时间：

```bash
grep -E 'Phase warmup progress|WARMUP cache pressure|Phase warmup complete|Phase profiling started|Phase profiling complete|replay_rc=' \
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
- AIPerf 校验显示 completed request 为零，或错误率超过配置的 `0.10` 上限（[validator](../utils/agentic/validation/validate_agentic_result.py#L49-L87)）。

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
