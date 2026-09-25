# 受信测量回执

[English](./measurement-receipts.md)

这个前置变更让托管回执签发流程和不可变产物传输先部署到受信的 `main`，随后才能对 native 产出端进行验收。它不选择 native pilot，不修改矩阵路由，不新增 runner launcher，也不分配新的 GPU 执行路径。随附的 Python 客户端和契约类型是可复用的验证及准备基础；现有基准测试选择逻辑保持不变。

## 合入顺序

1. 部署 InferenceX-app 回执读取端及迁移 `016_measurement_snapshots.sql`。记录实际部署 commit，并在启用 native 发布前验证所需回执版本。
2. 将这个控制端前置变更合入 InferenceX `main`。[签发工作流](../.github/workflows/phase1-receipt.yml) checkout 自身受信的工作流 commit，不执行候选代码来决定哪些测量结果可被接受。
3. 配置下文列出的已审查签发版本和已部署读取端版本。native 产出端的选择与分配属于另一个独立审查的变更，必须提供相应硬件验收证据。
4. 完整源运行成功后，先签发源回执并验证 staging；经批准的 merge 运行成功后，再签发独立发布记录。本地控制端测试不能证明读取端已部署、GPU 验收已通过或生产发布已完成。

## 策略与已审查输入

两个仓库都应配置 `INFX_RECEIPT_ISSUER_SHAS`，以逗号分隔已审查的 InferenceX 签发 commit，并将 `INFX_RECEIPT_ISSUER_WORKFLOW` 设置为 `.github/workflows/phase1-receipt.yml`。只要旧的不可变回执仍受支持，就应保留相应签发版本。在 InferenceX 中，将 `INFX_PHASE1_READER_REVISION` 固定到最终部署的应用 commit，并通过 `INFX_PHASE1_COLLECTOR_REVISION` 记录这个前置变更的受信 commit，供后续产出端验收使用。这些配置是部署前置条件，不能从调度 payload 推断。

维护者应先审查受信 `main` 上的 `qualification/phase1/*.json` 输入，再以 `kind: measurement` 调用签发流程。[Approval schema](../infx/workflows/phase1_publication.py) 要求 concurrency 为 1、2、4、8、16、20、24、28 的八个吞吐量点，以及实际 c28 GSM8K 评估，并记录源 run/attempt/head、逐点 execution 和 bundle 身份、native manifest digest 及实际语料版本。这些身份必须来自独立准备的控制记录；worker 的 `execution.json` 和产物名称不能自行授权预期契约。这个前置变更不包含虚构的批准文件。

[回执验证器](../infx/results/publication_receipt.py) 检查准确的 GitHub 产物 ID 及归属、API 与 ZIP digest、安全成员路径、预期执行身份、物理拓扑、规范化配置、必需指标、数据集元数据，以及完整的评估样本和过滤器覆盖。输入支持每个任务的原始 lm-eval 结果与元数据；聚合部署保留明确为零的拆分 worker 计数。紧凑的 version 1 回执在结果从 staging 进入生产时保留原始测量身份。

归档成员的 digest 通过每次读取 1 MiB 数据计算，避免将大型 AgentX trace 一次性载入内存。验证器将每个成员读取到 EOF，以校验 CRC 和实际字节数，同时保留单成员 10 GiB、单产物 20 GiB 的限制及原有路径检查。评估 JSONL 逐行解码，内存占用取决于最大的样本行和已记录的文档/过滤器身份；覆盖率和分数检查保持不变。摘要及执行 JSON 仍作为结构化输入进行验证。

## Staging、发布与恢复

[传输解析器](../infx/workflows/receipt_transport.py) 使用只读 API，从允许版本在 `main` 上成功完成的 `workflow_dispatch` 签发运行中查找唯一的已接受回执。它验证源运行原始 attempt，不用最近一次 rerun 替换。只要 API 清单包含 `native-execution-*`，就必须提供回执；证据缺失或无效时，不能回退到旧 native 导入方式。普通旧产物清单继续使用现有路径。

源回执签发后，使用现有经过授权的 staging 流程。生产发布前，审查 [PublicationRecord](../infx/workflows/phase1_record.py)：它引用原始回执产物及 digest，并独立绑定 merge run/SHA、changelog 产物及 digest、已部署的 app/ingest 版本。随后以 `kind: publication` 签发该记录。记录的 `ingest_sha` 必须匹配实际执行导入的应用 checkout，并应保留 `app_sha` 对应的部署证据。源回执不随发布而重写。

sweep 的自动导入任务在必需签发尚未完成时延后调度，让源或 merge 工作流能够成功结束。两个签发运行均完成后，现有[恢复工作流](../.github/workflows/recover-reused-ingest.yml) 会传递准确的回执和发布记录 ID，以及各自的 ZIP/JSON digest。native 生产发布始终需要后续记录，即使源运行和 merge 运行 ID 相同。已部署的应用在导入前独立验证传输和已接受 snapshot；中断恢复只能使用同一 snapshot。不要通过选择更新的同名产物来替代缺失的已接受产物。

## 本地验证

在这个前置变更的 checkout 中运行行为测试：

```bash
uv run --locked --group test pytest -q utils/test_benchmark_preparation.py \
  utils/test_python_benchmark_clients.py utils/test_phase1_receipt_control.py \
  utils/test_publication_receipt.py utils/test_receipt_transport.py
uvx --exclude-newer PT12H ruff@latest check infx
uvx --exclude-newer PT12H ruff@latest format --check infx
```

这些测试覆盖已安装包资源、prepared client 身份、子进程清理、九点批准契约构建、产物验证及不可变源回执/发布传输，不提交 Slurm 任务，也不写入生产数据库。现有验证边界见[测试说明](./testing_zh.md)和 [eval/AgentX 操作流程](./eval-agentx-procedures_zh.md)。
