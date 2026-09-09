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
所有者和调度器中的精确分配身份；提交结果不明确时禁止重复申请。固定站点是
`main` / `sa-shared`。新建独占分配预留八张 GPU；示例 step 使用四张
GPU、32 个 CPU 和 1 TiB 主机内存。固定版本的四 rank 加载器在 CPU 暂存权重时
超过了 256 GiB；1 TiB 是实际运行验证过的额度，并非测得的最低需求。预算按预留容量计算。

`resources.minutes` 是整个分配的时间上限，最多 90 分钟。step 为外层清理
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

## 结果与本地检查

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
uv run --no-project --python 3.12 \
  --with 'av==16.1.0' --with 'numpy==2.3.5' \
  --with 'pytest>=8,<9' python -m pytest -q
bash -n runtime-entry.example.sh
```

[Test H3 Video](../../.github/workflows/test-h3-video.yml) 对相关变更运行上述 CPU
检查和工作流 lint。这些检查不会调用模型、调度器或 GPU。真实 CI 执行和
artifact 检查是独立的验收证据。

编译缓存保留在持久化存储中，不上传为测量证据。
