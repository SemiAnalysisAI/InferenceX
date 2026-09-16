# OperatorX GitHub Actions

[English](CI.md) | **中文**

[OperatorX Sweep](../../.github/workflows/operatorx-sweep.yml) 支持手动选择
`h100-dgxc`（默认）或 `h200-dgxc`。PR 只在 GitHub 托管运行器上生成执行计划；
GPU 执行必须通过 `workflow_dispatch` 触发。首个验证目标是 H100。

## 触发运行

GitHub 注册该工作流后，选择 **OperatorX Sweep → Run workflow**，指定源码分支，
并保留初始默认值：`pool=h100-dgxc`、`backends=torch`、`testlists=gemm_perf`、
`world_sizes=1`、`chunk_size=50`。这会生成一个包含 11 个 BF16 GEMM 形状的分片。
新工作流可能需要先进入默认分支，GitHub 才允许手动触发。

```bash
gh workflow run operatorx-sweep.yml --repo SemiAnalysisAI/InferenceX \
  --ref <branch> -f pool=h100-dgxc -f backends=torch \
  -f testlists=gemm_perf -f world_sizes=1 -f chunk_size=50
```

其他 NVIDIA 后端和测试列表需要显式选择，不能视为已经通过 Hopper 验证。
不支持的操作会保留在结果中。后端导入错误、基准错误，以及没有任何成功结果，
都会使分片失败。先验证 BF16 GEMM，再验证范围受限的集合通信和兼容的 MoE 组合。
不要假定面向 Blackwell 的 FP4 内核可以在 Hopper 上运行。

## 执行约定

- 托管规划步骤校验输入，按容器镜像分组后端，区分 world size 和 MoE 并行参数组合，
  并将形状拆成大小受限的分片。最多支持 256 个分片。world size 仅允许 1、2、4、8；
  未被所选大小覆盖的形状会计入 `excluded_shapes`。
- 每个 Actions 分片独占一个八 GPU Slurm 节点。GPU 进程数等于所选 world size。
  准入沿用优先级评分器，以及 `ci-job-*`、`ci-attempt-*` 和唯一的 `nodes:1` 标签。
  初始并发上限为两个分片。两个调度开关都必须保持启用。
- 运行器设置来自 CollectiveX 已纳入版本控制的平台配置。源码按工作流 SHA 检出，
  再复制到共享 squash 父目录下的私有、计算节点可见目录。结果不依赖提交主机的
  `/tmp` 在计算节点上可见。
- 规划步骤解析镜像 digest。导入操作加锁，并按镜像与 digest 缓存，导入后再次核对
  digest。标签发生变化或无法解析时运行失败，避免错误标注测量所用镜像。
  规划和导入主机都必须能匿名读取镜像。
- 启动器等待分配、导入和执行完成。Slurm 分配限时 45 分钟；Actions 允许 70 分钟，
  包含排队与清理时间。Slurm 作业名与 Actions 运行器名称一致。
- 信号处理和工作流的 `always()` 恢复步骤会取消已记录的分配、停止写入、保留部分结果，
  然后删除暂存源码。清理失败时保留暂存目录供调查。如果运行器主机失联，Slurm 时间限制
  是最后的资源释放保障。
- CI 严格模式在每个操作结束后原子写入 rank-zero 结果检查点，写入发生在内核计时之外。
  原有非 CI 计时循环保持不变。

## 产物与重跑

`operatorx-manifest-<run_id>` 记录请求的案例、镜像 digest 和源码 SHA，失败作业重跑时
仍可使用。每次尝试分别上传 `operatorx-shard-<run_id>-<attempt>-<shard>`，包含执行元数据、
分配/导入/基准日志、状态和已生成的原始结果 JSON。
启动失败时可能只有日志；取消时的检查点仅代表部分覆盖。分片成功要求实际测量成功，
不能仅凭 Slurm 提交成功。结果环境信息记录工作流运行、尝试、分片、源码 SHA 和镜像 digest。

使用 `gh run download` 下载产物。保留原始文件及来源信息；
`scripts/consolidate_results.py` 不属于 CI 流程。仪表盘接入属于独立工作。

## 本地验证

生成计划需要 Python 3.11 或更新版本。计算节点控制代码使用现有的 Python 3.10+
Slurm 主机环境。CPU 测试执行真实规划器、基准编排和启动器，仅替换外部 GPU/Slurm 依赖。

```bash
uv run --no-project --python 3.12 --with pytest --with pyyaml \
  python -m pytest experimental/operatorx/tests/ -q
```

实际验收还需要带产物的 Hopper smoke 运行、失败分片重跑，以及确认释放分配的取消测试。
CPU 检查不能证明 GPU 兼容性或集群存储可见性。

最终覆盖汇总为每个请求分片选择最新产物尝试，保留此前尝试中已成功的分片，
并在任一分片缺失或失败时报告失败。汇总区分请求形状数和结果行数，
因为一个形状可能在多个后端上执行。

清理失败后，可在单分片触发中将 `recovery_run_id` 设为同一运行器池中近期的 OperatorX
运行 ID。工作流下载执行产物，在申请新节点前重试分配与暂存清理。恢复流程校验运行、
运行器池和私有暂存父目录；不要选择无关或过旧的 Slurm 执行。

`cleanup.log` 记录用于确认分配已释放的活动作业查询。查询使用当前用户的作业列表，
因为直接查询已删除的作业 ID，即使分配已终止，也可能返回 Slurm 错误。
