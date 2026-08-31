<div align="center">

[English](./troubleshooting.md) | **中文**

</div>

# 故障排查

按照首个未能建立契约的层级对失败分类。重跑前保留证据，只修复负责该契约的层级；当证据不足以支持范围明确且可逆的操作时停止。下游任务可能在 `always()` 下执行或聚合空集合，因此其绿色状态不能证明上游基准测试或评测成功。

## 索引

- [事实来源](#事实来源)
- [修复前的证据](#修复前的证据)
- [失败层级矩阵](#失败层级矩阵)
- [Changelog 与矩阵](#changelog-与矩阵)
- [运行器与 AMD root 文件](#运行器与-amd-root-文件)
- [服务器](#服务器)
- [评测与收集](#评测与收集)
- [摄取](#摄取)
- [已知 KLAUD 案例](#已知-klaud-案例)
- [验证与停止条件](#验证与停止条件)

## 事实来源

- [`KLAUD_DEBUG.md`](../KLAUD_DEBUG.md) 记录反复出现的 Klaud-Cold/镜像升级事故及其已观测特征。它是事故知识，不替代当前工作流或评审政策。
- [`run-sweep.yml`](../.github/workflows/run-sweep.yml)、[`benchmark-tmpl.yml`](../.github/workflows/benchmark-tmpl.yml) 和 [`benchmarks/benchmark_lib.sh`](../benchmarks/benchmark_lib.sh) 定义编排、制品上传、服务器就绪、基准测试和评测行为。
- [`validate_perf_changelog.py`](../utils/validate_perf_changelog.py)、[`generate_sweep_configs.py`](../utils/matrix_logic/generate_sweep_configs.py) 和 [`validation.py`](../utils/matrix_logic/validation.py) 分别负责 changelog、矩阵和模式失败。
- [`utils/runner_setup/RUNNER_SETUP.md`](../utils/runner_setup/RUNNER_SETUP.md) 与 [`runners/`](../runners/) 负责预置和启动器路由。[`CONTRIBUTING.md`](../CONTRIBUTING.md#amd-cluster-never-leave-root-owned-files-in-runner-workspaces) 负责 AMD 工作区安全规则。
- [`utils/evals/EVALS.md`](../utils/evals/EVALS.md)、[`validate_scores.py`](../utils/evals/validate_scores.py) 和 [`collect_eval_results.py`](../utils/collect_eval_results.py) 分别负责评测执行、验证和收集。
- [失败摄取恢复命令](../.claude/commands/recover-failed-ingest.md)是带防护的恢复流程。下游事实来源是 InferenceX-app 的 [`ingest-results.yml`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/.github/workflows/ingest-results.yml)、[`prepare-ci-artifacts.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/packages/db/src/prepare-ci-artifacts.ts)、[`ingest-ci-run.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/packages/db/src/ingest-ci-run.ts) 和 [`benchmark-mapper.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/packages/db/src/etl/benchmark-mapper.ts)。

## 修复前的证据

重跑或编辑之前，记录：

1. 工作流 URL、run ID、attempt、event、branch、head SHA、失败任务/check、运行器名称和时间戳。
2. 精确配置键，以及生成后的模型、镜像、运行器、框架、场景、拓扑和并发。
3. 第一个错误和足以识别错误进程的前置日志上下文。将服务器与 Slurm 日志和工作流编排日志分开保存。
4. 制品名称、过期状态，以及与失败相关的小型结构化字段（`meta_env.json`、结果元数据、run stats、分数摘要）。不要转储大型聚合 JSON。
5. 已经成功的环节。矩阵从未发出任务、任务从未获得运行器，以及服务器从未健康是不同失败，即使它们最后都显示“无结果”。

不要先重跑：重跑可能替换日志、改变运行器/节点分配，或让确定性失败看起来像瞬时故障。

## 失败层级矩阵

| 首个失败层级 | 典型症状 | 安全证据 | 第一个安全操作 |
| --- | --- | --- | --- |
| Changelog/setup | `Deletions are not allowed`、YAML 格式错误、触发范围错误 | Setup 日志；精确 base/head changelog 字节与差异 | 恢复当前 main 字节，只追加本 PR 的预期条目，然后重跑验证器 |
| 矩阵/编排 | 生成器异常、矩阵为空/错误、任务跳过、标签冲突 | 精确生成 CLI；生成器/Pydantic 错误；PR 标签与 head 提交 | 本地复现并修复 master 配置/生成器/标签；不要绕过 setup |
| 运行器/分配 | 长时间排队、checkout `EACCES`、Slurm 未启动、Docker socket 或磁盘错误 | GitHub 运行器名称；启动器输出；`squeue`/`sacct`/节点状态；工作区属主 | 修复运行器基础设施或上报升级处理；不要为不健康节点调整配方 |
| 服务器 | 进程退出、日志未出现、健康检查未通过、OOM/内核/端口错误 | 服务器日志、PID 退出、`/health`、镜像 digest/tag、GPU/Slurm 日志 | 匹配精确特征；只修改一个有证据支持的运行时/镜像设置，或回退 |
| 评测 | `eval /` 失败、批次不完整、分数低于阈值、结果缺失 | `meta_env.json`、每个 `results*.json`、验证器输出、镜像与任务 | 修复评测/服务器/任务根因，重跑精确评测配置 |
| 收集 | “No eval results found”、聚合为空、评测被跳过但收集器绿色 | 底层 `eval /` 结论；制品树与元数据；收集器输出 | 恢复/修复上游制品契约；不要只依据收集器输出诊断服务层 |
| 摄取 | 面板行缺失/错误；目标 `trigger-ingest` 绿色但无有效数据 | 目标与来源运行元数据；未过期制品；应用工作流/ETL 日志；changelog 范围 | 使用带防护的恢复流程；绝不重跑失败的目标工作流 |

## Changelog 与矩阵

### Changelog

Setup 阶段的删除错误通常意味着陈旧分支或改变空白的合并使历史字节看起来被删除。遵循 [`KLAUD_DEBUG.md` §1.1](../KLAUD_DEBUG.md#11-perf-changelogyaml-deletion-not-allowed) 的规范修复：逐字采用当前 main 版本，然后只在末尾追加本 PR 条目。使用 [`validate_perf_changelog.py`](../utils/validate_perf_changelog.py) 对真实 base 与 head 进行验证。

不要对 `perf-changelog.yaml` 做三方合并或格式规范化。如果预期配置键、评测标志、场景范围或历史差异不明确，请停止。Changelog 新增内容和允许的 `pr-link` 修正行为由验证器强制执行，不能只依据目视有效的 YAML 解析结果。

### 矩阵与编排

在本地重跑与工作流完全相同的 `generate-cli-command`。先运行精确键 `test-config`，再运行同样过滤的 `full-sweep` 命令；检查发出的镜像、运行器、场景、拓扑、并发、评测标志和附加设置。命令与检查清单位于 [`testing_zh.md`](./testing_zh.md#先精确配置再过滤配置族)。

如果没有 GPU 任务启动，先检查 setup，再调查运行器。[`run-sweep.yml`](../.github/workflows/run-sweep.yml) 会拒绝冲突主标签，不会仅凭评测修饰标签运行，只在 PR head 上接受 `[skip-sweep]`，并等待合并冲突解决。修复输入或标签状态；不要手动分发不同矩阵并把它称为等价证据。

当本地矩阵与预期 PR 范围不完全一致时停止。生成器成功但配置错误并不是恢复成功。

## 运行器与 AMD root 文件

阅读服务器日志前，先分类排队/分配失败：

- 将生成的运行器标签映射到 [`configs/runners.yaml`](../configs/runners.yaml)，并检查从 [`runners/`](../runners/) 选择的启动脚本。确认目标集群、节点和 Slurm 作业。
- 使用工作流任务的运行器名称和分配时间戳关联 `squeue`/`sacct` 与节点状态。节点 drained、Pyxis 损坏、Docker socket 权限、磁盘耗尽或运行器 checkout 失败都属于基础设施证据。
- 仅在运行器恢复健康后重跑。不要仅为了避开坏节点而修改模型并行度、内存标志或镜像。
- 将访问权限、drained 节点、socket、存储和永久 Slurm 配置变更升级给集群运维人员。

[`KLAUD_DEBUG.md` §5](../KLAUD_DEBUG.md#5-cluster-infrastructure-amd-mi355x--mi300x--mi325x) 列出了已知 AMD 节点、Docker socket、磁盘和端口事故。除非当前节点证据再次确认，否则应把其中点名的节点状态视为历史记录。

### AMD root 属主工作区文件

其特征是 checkout 清理在 `benchmark_logs/logs/slurm_job-*` 上因 `EACCES` 失败。以 root 运行的 Slurm 容器在取消导致 teardown 跳过时可能遗留 root 属主目录，阻塞该运行器上所有后续任务。[`CONTRIBUTING.md`](../CONTRIBUTING.md#amd-cluster-never-leave-root-owned-files-in-runner-workspaces) 中的预防契约是：

1. 绝不以 root 身份写入运行器 `_work` 工作区；使用 `_work` 外的临时目录。
2. 如果无法避免，安装取消安全的清理逻辑，为所有 root 属主输出更改所有权或删除它们。
3. 中途取消测试运行并验证工作区仍然干净。

恢复时遵循 [`.claude/commands/clean-amd-mi355-runner-root-files.md`](../.claude/commands/clean-amd-mi355-runner-root-files.md)：使用文档规定的 sudo 跳板，执行范围仅限运行器 `_work` 的只读扫描，审查每一条路径；若任何匹配项位于 `_work` 外则停止；仅以相同范围删除已确认项，然后重新扫描确认零条目，之后再重跑。绝不要对 `/it-share` 执行无范围限制的 `rm -rf`。

## 服务器

[`wait_for_server_ready`](../benchmarks/benchmark_lib.sh) 会区分“服务器在日志出现前死亡”“服务器在健康前死亡”和进程存活但 `/health` 尚未通过。保留服务器日志和 PID 状态；仅有工作流最终超时不能构成诊断。

对于 RDMA 上的 Mooncake DRAM 层，宿主内存申请成功并不能证明 RNIC 可以用
4 KiB 页高效映射同一片内存。大容量 DRAM 层必须在 Docker 启动前于宿主预留
HugeTLB，并向容器传入 `MC_STORE_USE_HUGEPAGE=1` 和
`MC_STORE_HUGEPAGE_SIZE=2MB`。预留量应根据最终 DRAM 层总量、每个 TP rank
的本地 buffer 及 allocator 余量计算；`sysctl` 后必须验证
`HugePages_Total`，并在正常退出和信号退出时都恢复宿主原值。不要用
`max_mr_size` 替代 HugeTLB：它是 RNIC 能力值，不能减少基于 4 KiB 页的内存
所产生的页表项。

使用最早出现的具体特征：

- **镜像拉取/tag 失败：**修改运行时标志前验证精确 registry tag 或 digest 是否存在。[`KLAUD_DEBUG.md` §6](../KLAUD_DEBUG.md#6-docker-image-tag-gotchas) 警告不要从带日期的 nightly 推导 release tag。
- **权重/KV/CUDA graph OOM：**记录空闲显存、配置利用率、每 rank 并发、graph 限制和启动失败位置。只应用与匹配案例一致的设置；之后同时确认服务器启动和工作负载完成。
- **内核/架构断言或非法地址：**保留完整堆栈和 GPU 架构。优先使用已修复/固定的上游镜像或受支持后端，而不是未经评审的本地引擎补丁。
- **地址被占用：**终止进程前确认占用者及其集群所有者。不要杀死未经验证的 PID 或无关服务。
- **健康服务器在基准测试中死亡：**[`run_benchmark_serving`](../benchmarks/benchmark_lib.sh) 会监控服务器 PID。同时保留客户端与服务器日志，并按服务器的第一个错误分类，而不是按客户端后续连接错误分类。

如果拟议 workaround 会改变模型语义、减少模型 FLOPs、修补服务栈，或没有精确源代码 guard，请停止。当前 [PR 清单](./PR_REVIEW_CHECKLIST.md) 禁止推理引擎补丁，除非满足规定的豁免流程。

## 评测与收集

### 评测

阅读单个 `eval /` 任务，不要只看 `collect-evals`。对每个预期并发检查 `meta_env.json`、完成/失败元数据及其 `results*.json`。[`validate_scores.py`](../utils/evals/validate_scores.py) 会拒绝缺失结果文件、低于阈值的分数和零个已检查指标的运行。存在预期并发元数据时，它还会拒绝无效 manifest、重复/意外/缺失并发和失败批次。由于工作流调用时没有传入 `--expected-concs`，应单独检查非批处理的 `meta_env.json`；即使分数验证器成功退出，缺失或无效元数据仍然属于失败。

确认任务与镜像匹配生成配置。如果服务器在评测中失败，返回服务器层级。如果任务、阈值或 manifest 错误，修复其事实来源并重跑精确评测。不要接受结果被跳过、为空或不匹配的绿色任务。

### 收集

[`collect_eval_results.py`](../utils/collect_eval_results.py) 会发现平铺和嵌套制品布局，为旧版结果选择最新文件，为批处理的每个并发选择最新文件，并把批处理结果过滤为已完成并发。其 `> No eval results found to summarize.` 消息表示没有收集到行；它不是评测通过的证明。

收集为空时：

1. 检查底层 `eval /` 任务确实执行并成功，而不是被跳过。
2. 确认每个下载制品都有 `meta_env.json`，以及带 `lm_eval_version` 的可识别 lm-eval JSON。
3. 对批处理评测比较请求、完成、失败及文件名并发后缀。
4. 在下载的制品树上复现收集。

修复生产者或制品布局。不要添加伪造元数据，不要把失败并发标为完成，也不要修改收集器使其摄取不完整输出。

## 摄取

`trigger-ingest` 成功不能证明存在有效基准结果行：[`run-sweep.yml`](../.github/workflows/run-sweep.yml) 可以在收集之后触发下游处理，而取消/无结果目标仍可能到达该任务。验证目标运行的 event、workflow、branch、head SHA、changelog 差异、结果制品和下游 InferenceX-app 日志。

应用工作流在 [`ingest-results.yml`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/.github/workflows/ingest-results.yml) 中准备、迁移、摄取、应用覆盖并验证数据。[`prepare-ci-artifacts.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/packages/db/src/prepare-ci-artifacts.ts) 负责选择和下载来源/合并制品并写入复用元数据；[`ingest-ci-run.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/packages/db/src/ingest-ci-run.ts) 负责数据库摄取。[`benchmark-mapper.ts`](https://github.com/SemiAnalysisAI/InferenceX-app/blob/main/packages/db/src/etl/benchmark-mapper.ts) 中的失败行 guard 仅在 `num_requests_successful` 为数值零且 `num_requests_total` 为数值时跳过该行。

使用带防护的[失败摄取恢复流程](../.claude/commands/recover-failed-ingest.md)，不要重跑失败目标。它要求来源是已完成的 pull-request `run-sweep.yml`，结果制品未过期，来源提交属于原 PR，执行语义未变化，changelog 范围明确，并保留恢复 ancestry。

如果来源运行或制品不合格、无法证明来源 ancestry、来源 SHA 后配置/配方/镜像语义发生变化，或预期 changelog 范围不明确，请停止恢复。绝不要绕过 pending/失败检查，不要在挂接来源 ancestry 后重写恢复分支，也不要在未检查数据时相信目标的绿色 trigger。

## 已知 KLAUD 案例

使用 [`KLAUD_DEBUG.md`](../KLAUD_DEBUG.md) 识别精确特征，然后依据当前镜像、配方、硬件和政策验证，再应用其修复。

| 特征 | 已知案例与安全边界 |
| --- | --- |
| Setup 表示 changelog 历史被删除 | [§1.1](../KLAUD_DEBUG.md#11-perf-changelogyaml-deletion-not-allowed)：恢复 main 字节并只追加 PR 条目；绝不三方合并历史 |
| vLLM 权重加载/KV 分配 OOM | [§2](../KLAUD_DEBUG.md#2-vllm-v021x--v020x-gpu-oom-at-model-load)：降低利用率或使用记录的 profiler 设置前，先确认 memory-profiler 特征 |
| DEP decoder 在大型 CUDA graph capture 中失败 | [§2.1](../KLAUD_DEBUG.md#21-dep-cuda-graph-capture-oom-on-gb300)：按每个 DP rank 的负载而不是全局并发设置 sequence/graph 限制 |
| DSV4 在自定义 digest 上可用但在通用 SGLang 上 OOM | [§3](../KLAUD_DEBUG.md#3-custom-dsv4-image--generic-v0512-ooms)：通用 release 不是可直接替换项；保留/回退到已验证兼容镜像 |
| B300 DeepGemm 非法地址、EAGLE trtllm GEMM 失败或 flash-attn 架构断言 | [§4](../KLAUD_DEBUG.md#4-upstream-sglang-v0512-b300-regressions)：区分三种堆栈；使用受支持后端/上限或已修复/固定的上游镜像 |
| AMD drained/Pyxis、Docker socket、磁盘满或端口被占用 | [§5](../KLAUD_DEBUG.md#5-cluster-infrastructure-amd-mi355x--mi300x--mi325x)：确认当前节点状态并升级基础设施问题；不健康基础设施没有配方级修复 |
| 猜测的 Docker tag 返回 404 | [§6](../KLAUD_DEBUG.md#6-docker-image-tag-gotchas)：在 registry 验证精确 tag；不要推断命名规律 |
| `gh run rerun --failed` 被拒绝 | [§7](../KLAUD_DEBUG.md#7-ci-rerun-mechanics)：检查运行状态/结论；仅已完成的失败支持只重跑失败任务，取消运行需要完整重跑 |
| MiniMax M3 B300 MSA 报告 `q2k_indices` 不连续 | [§11](../KLAUD_DEBUG.md#11-minimax-m3-b300-msa-top-k-slice-is-non-contiguous)：识别 TP1/data-parallel-attention 暴露条件；优先使用上游已修复镜像。提交记录中的引擎补丁须符合当前清单/豁免要求 |

历史 KLAUD 标签或合并建议不能覆盖当前[扫描标签政策](../.github/AGENT_OPERATIONS.md#sweep-labels-and-reuse)、[`CONTRIBUTING.md`](../CONTRIBUTING.md) 或 [PR 清单](./PR_REVIEW_CHECKLIST.md)。

## 验证与停止条件

只有在重跑同一失败路径后原始特征消失、预期任务实际执行且预期结构化制品/数据出现，修复才算完成。**摄取是例外：**绝不重跑失败的目标工作流或任务；应验证新的恢复 push 及其下游应用摄取。记录新的 run/attempt，并与保留的失败证据比较。

遇到以下情况时停止并升级处理：

- 无法根据保留的日志和元数据识别首个失败层级；
- 删除、清理、终止 PID、变更节点或重写分支的范围不够精确且无法独立验证；
- workaround 会改变模型语义、架构 FLOPs、镜像来源，或在没有批准豁免时修补服务栈；
- 集群修复需要你不具备的权限或所有权；
- 评测完成状态、分数、镜像身份或制品来源缺失；
- 收集没有上游制品；
- 摄取来源资格、执行等价性、changelog 范围或 ancestry 不确定。

不要把反复重跑称为修复。如果相同输入只在运行器/节点分配变化后成功，请保留两次运行，并依据证据把原始失败分类为基础设施问题或疑似 flake。
