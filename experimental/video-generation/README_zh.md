# H3 视频 CI 冒烟测试

[English](README.md) | **中文**

此实验任务在 InferenceX CI 内使用 SemiAnalysis H200 资源运行现有 H3
supervisor。首个目标是有时间上限的同版本冒烟测试：保留实际生成的 MP4，
完整验证视频和音频，记录请求耗时，并验证清理结果。此任务不写入原生
InferenceX 数据库，也不新增网站页面。

runner 支持两种冻结的 16:9 设置，均为 1344×768、24 FPS：请求 4 秒对应
107 帧，请求 8 秒对应 192 帧。这些帧数遵循固定版本 H3 的时间取整规则。
修改提示词或时长时应冻结新 plan，历史运行仍保留原始设置。

冒烟任务成功表示配置中要求的测量和证据已完成。未经校准的回归判定仍为
inconclusive，`ci_accepted: false`。新增这些文件或 CPU 测试通过，都不能
证明已经成功运行 H3。

## 准备现有运行环境

将仓库变量 `H3_SITE_CONFIG` 设置为 `cluster:h200-dgxc` 提交节点上
已审核 JSON 文件的绝对路径。以 [site.example.json](site.example.json) 为起点；
其中占位符不可直接执行。配置绑定持久工作目录、现有 rootfs 和准备记录、
仅负责进入容器的脚本及其 SHA256、容器内 Python、冻结的 supervisor spec
及其 SHA256、资源上限、任务标识，以及可选的历史分配收据。

spec 必须记录真实的算力和模型使用批准。选择手动 H3 路径仅请求执行该配置中
已审核的工作负载；调度参数不接受 shell 命令、模型路径、任意配置内容或其他
GPU 提供商。

提交节点需要 Python 3.11+、Git、Slurm 工具，以及配置中声明的共享路径访问权。
PyAV/NumPy、固定版本的 H3 运行时和模型必须已在现有环境中准备好。仅表示
rootfs 已创建的标记不能证明模型兼容。adapter 在申请资源前检查准备记录和
输入标识；不会安装依赖、导入镜像或新建 rootfs。

根据保存的可用进入命令调整
[runtime-entry.example.sh](runtime-entry.example.sh)，然后固定其摘要。脚本必须
进入现有 Enroot rootfs，将 `workspace.host` 挂载到 `/work`，保留 Slurm
step 的 GPU/CPU 绑定和元数据，转发传入命令并传回退出码。它将
`SLURM_STEP_GPUS` 中的全局编号转换为物理 UUID，导出
`H3_ASSIGNED_GPU_UUIDS`，并在 `H3_ORIGINAL_CUDA_VISIBLE_DEVICES` 中
保留原设备掩码。容器内准入会核对驱动实际看到的 UUID。不要把会申请资源的旧
launcher 当作进入脚本。

固定版本的 SGLang 运行时要求数字设备编号。每个 H3 子进程使用所选设备的
实际 NVML 编号，并在导入 SGLang 前核对 CUDA 驱动返回的有序 UUID。
设备枚举不一致时启动失败；归属锁和遥测仍使用分配的 UUID。

adapter 在申请资源前恢复本任务的分配收据。导入的收据必须匹配任务标识、Unix
所有者和调度器中的精确分配身份；提交结果不明确时禁止重复申请。默认 H200 站点是
`main` / `sa-shared`，其他站点通过 `site` 明确记录，见下文。新建独占分配预留八张 GPU；示例 step 使用四张
GPU、32 个 CPU 和 1 TiB 主机内存。固定版本的四 rank 加载器在 CPU 暂存权重时
超过了 256 GiB；1 TiB 是实际运行验证过的额度，并非测得的最低需求。预算按预留容量计算。

`resources.minutes` 是整个分配的时间上限，最多 240 分钟。step 为外层清理
预留五分钟，supervisor 的上限加十分钟必须不超过分配上限。例如：分配
90 分钟、step 85 分钟、supervisor 75 分钟。复用分配必须有足够剩余时间。
保留准备好的 rootfs，仅清理属于本次任务的进程和 step，仅释放本次执行拥有的
分配。

## 通过 InferenceX 调度

第一次运行必须在 CI 中进行。对于功能分支，使用已注册的 End-to-End Tests
工作流调用该分支的可复用 H3 工作流：

```bash
gh workflow run e2e-tests.yml --repo SemiAnalysisAI/InferenceX \
  --ref feat/h3-video-ci -f h3-video=true -f test-name=h3-first-smoke
```

调度前审核分支。首次调度者和重新运行者都必须具有 write、maintain 或 admin
权限。H3 路径检出 `github.sha`：`--ref` 同时选择工作流定义和源码；
普通 LLM 路径的 `ref` 输入不用于 H3。此模式跳过 LLM 矩阵生成、所有依赖它的
LLM 扫描和收集任务，以及对应的成功率计算。外部 PR 事件不能启动此路径。
独立工作流在默认分支注册后，也可手动调度 `h3-video.yml`。

开启优先级调度时，必须同时开启节点配额调度。排队任务在
`cluster:h200-dgxc` 上请求唯一的 `nodes:1`，以及原生
`ci-job-<priority>-<token>` 和 `ci-attempt-<attempt>` 标签。关闭优先级
调度时，沿用仓库的集群标签路径。GitHub 准入和 Slurm 资源验证是两个不同环节。
此任务使用原生工作流权限和调度准入，不新增 OIDC 服务，也不声称提供独立硬件
证明。

## 可选的服务负载测试

在已审核的 supervisor spec 中添加
`"serving": {"concurrency": 2, "delivery_deadline_seconds": 300}`，
并更新站点配置固定的 spec SHA256。这里的截止时间是操作者设置的示例，
不是经过校准的验收标准。并发数支持 1–32；省略 `serving` 时保持原有串行回归行为。
直接客户端提供 `--serving-concurrency` 和 `--delivery-deadline-seconds`；
不加 `--execute` 时仍然只预览。

每个任务针对一个受监督的服务端点测量一个并发数。工作线程下载完前一个视频
后提交下一个请求，媒体校验单独进行；预热仍然串行且单独记录。比较不同负载时，
应在独立任务中使用同一份固定的提示词、种子、生成参数和运行时，并包含显式
设置并发数为 1 的服务模式对照。不自动启动负载扫描。沿用现有 CI 的分配和
运行环境复用路径，保留每个请求的结果；远端完成状态不明确时停止新增提交。

此模式测量闭环交付吞吐，不代表固定到达速率或可持续服务容量。服务端排队与执行
时间戳、实际 batch 大小、多副本布局和完整部署成本仍为不可用。服务模式要求使用
未校准的策略。CPU 测试数据只验证测试工具，不能证明 H3 支持并发或具体硬件性能。

## 结果与本地检查

成功执行还会发布[前端结果契约](RESULTS_zh.md)：版本化的 `result.json`、
逐卡功率序列、分阶段能量与覆盖率，以及可离线打开的 `power-report.html`。
`h3-results-<run-id>-<attempt>` 包含索引、JSON Schema、双语指标说明，以及每次
原始执行的媒体、测量、日志和报告。原始执行身份与校验文件和本次导出身份分开保留。
遥测无效时功率和能量为空；导出失败会保留错误日志并返回失败状态。

复用已成功的 A/A 结果和已保留的硬件盘点时，在同一个可信入口提供原始 H3
运行编号（一至两个）以及盘点运行编号：

```bash
gh workflow run e2e-tests.yml --repo SemiAnalysisAI/InferenceX \
  --ref feat/h3-video-ci -f h3-video=true \
  -f h3-reuse-run-ids=34291306687,34293342829 \
  -f h3-inventory-run-id=34297499754 -f test-name=h3-power-export
```

此路径使用托管 CPU 导出，跳过原生 H200 作业。首次调度者和重新运行者仍须通过
相同权限检查。托管导出独立验证已验收的 H3 执行和已完成的硬件盘点作业，核对
产物校验清单、Git/CI/Slurm 身份及相同物理 GPU UUID。不执行新的 GPU 查询或
模型请求。源产物必须仍在 GitHub 保留期内。

省略 `h3-inventory-run-id` 时，会通过原生 Slurm 路径执行新的盘点。CI 核对源提交、
原始产物和持久化 Slurm 收据，在相同节点和 GPU UUID 上复用现有容器做只读检查。
分配上限为十分钟，即最多 1.3333 个预留 GPU 小时；先检查本任务已有分配是否可
复用。盘点不加载 H3 模型，清理后释放自有分配。后来的功率上限不能补写为历史
生成时的设置。新盘点上传 `h3-hardware-<run-id>-<attempt>`；仅使用 CPU 的重放
则下载已保留的盘点产物。已验证的原始硬件证据会复制到每份结果中。

已保留的[盘点运行 34297499754](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34297499754)
（[原始盘点产物](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34297499754/artifacts/10083702100)）
在 `worker-10` 上完成 Slurm **82290.0**，于 `2026-09-09T01:02:44Z` 观测原来的
四个 UUID。记录的 NVIDIA H200 PCI 设备/子系统编号 `233510DE` / `18BE10DE`
对应 H200 SXM，厂家最大可配置 TDP 为**每张 700 W**。在这次后续观测中，四张卡的
配置、实际执行、默认及最大功率上限均为 700 W；历史生成时的设置仍未知。
原始盘点记录中的 unknown 分类保持不变；导出器使用新 producer 提交解析原始
XML，同时支持带或不带 `0x` 的 PCI 编号，不重新查询硬件。
[已保留的 A/A 测量及限制](RESULTS_zh.md#已观测的-aa-证据)提供具体数据。

原始工作负载失败时仍上传失败证据，但不会启动成功结果导出。

每次尝试都会上传 `h3-video-<run-id>-<attempt>`，保留 14 天，并关闭媒体压缩。
失败后也执行上传，内容包括 adapter 的完整输出：收据、原始 MP4、遥测、
校验和，以及可用时的便携报告。缺失的报告或媒体保持缺失，不以测试素材替代。
原始持久证据保存在配置的工作目录。请在 GitHub 保留期结束前保存完整 artifact。

冒烟模式中，返回 0 表示两个角色的全部预热和测量均完成，耗时证据、重新解码后的
媒体和资源清理均通过验证；返回 1 表示已完成的负载包含无效结果；返回 2 表示执行
或证据验证失败。延迟/保真度阈值单独报告：冒烟执行成功后，比较结果仍可能失败
或不确定。回归模式还要求通过现有的校准与验收门槛。

```bash
cd experimental/video-generation
PYTHONPATH=../.. uv run --no-project --python 3.12 \
  --with 'av==16.1.0' --with 'numpy==2.3.5' \
  --with 'pytest>=8,<9' --with 'jsonschema>=4,<5' python -m pytest -q
bash -n runtime-entry.example.sh
```

[Test H3 Video](../../.github/workflows/test-h3-video.yml) 对相关变更运行上述 CPU
检查和工作流 lint。这些检查不会调用模型、调度器或 GPU。真实 CI 执行和
artifact 检查是独立的验收证据。

编译缓存保留在持久化存储中，不上传为测量证据。

## 跨硬件服务测量

`h3-preflight-only=true` 仅记录所选 CI runner 的身份、SSH 主机公钥和命令路径，
不分配或查询 GPU。独立的 `h3-site-preflight` 产物不代表性能结果或运行时验收。
该模式支持 `mi355x-amds`；AMD 视频生成仍未接入。预检不可同时重放历史结果。
各硬件站点使用独立的工作流并发组，并保留原有 Slurm 校验。

现有 `serving-smoke` 模式支持每档 4–200 条测量请求，数量由
`plan.cases × plan.repetitions` 决定；并发档位保持 1、2、4。每档 20 条
产生 60 条测量请求，`warmup_runs: 1` 时另有 3 条独立预热。失败和未启动
请求保留在预先确定的分母中。跨硬件固定相同模型文件、提示词/种子、视频规格
和质量要求，明确记录运行时构建与部署拓扑差异。小样本分位数属于初步结果，
闭环并发扫描不能证明持续开放到达负载下的服务容量。

默认配置保持 H200 的 `main` / `sa-shared`。可选 `site` 明确记录 `cluster`、
`partition`、`account` 和 `gpu_model`；当前允许 `h200-dgxc`、`h100-dgxc`、
`b200-nscale`。允许配置不等于实测通过。通过 `h3-cluster` 选择站点，
`h3-site-config` 指向 runner 上已准备并审核的 JSON 文件；二者必须匹配。
`resources.allocated_gpus` 单独记录全部分配卡数，`resources.gpus` 记录实际参与卡数；
H100 整节点分配需记录 8 张卡。总分配上限提高至 240 分钟，保留原有清理余量。
AMD 的运行时与设备接入尚未实现，不可用的硬件或指标不能用 fixture 数据代替。
