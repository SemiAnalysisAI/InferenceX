# OperatorX GitHub Actions

[English](CI.md) | **中文**

[OperatorX Sweep](../../.github/workflows/operatorx-sweep.yml) 支持手动选择
`h100-dgxc`（默认）、`h200-dgxc`、`b200-nscale`、`b300`、`gb200`、`gb300`、
`mi300x`、`mi325x` 或 `mi355x`。
PR 只在 GitHub 托管运行器上生成执行计划；GPU 执行必须通过 `workflow_dispatch` 触发。

| GPU | 运行器池 | 每个物理节点的 GPU 数 | 镜像平台 | 结果集群标识 |
| --- | --- | ---: | --- | --- |
| H100 | `h100-dgxc` | 8 | `linux/amd64` | `h100_dgxc_8x` |
| H200 | `h200-dgxc` | 8 | `linux/amd64` | `h200_dgxc_8x` |
| B200 | `b200-nscale` | 8 | `linux/amd64` | `b200_nscale_8x` |
| B300 | `b300` | 8 | `linux/amd64` | `b300_dsxe_8x` |
| GB200 | `gb200` | 4 | `linux/arm64` | `gb200_nvl72_4x` |
| GB300 | `gb300` | 4 | `linux/arm64` | `gb300_nvl72_4x` |
| MI300X | `mi300x` | 8 | `linux/amd64` | `mi300x_amds_8x` |
| MI325X | `mi325x` | 8 | `linux/amd64` | `mi325x_amds_8x` |
| MI355X | `mi355x` | 8 | `linux/amd64` | `mi355x_8x` |

GB200/GB300 每次使用一个四卡计算托盘，不会占用整个 NVL72 机架。所有运行器池的
稠密 GEMM 都使用 `world_sizes=1`，TFLOPS 始终按单卡计算。硬件信息来自 CollectiveX
平台配置，并在规划和执行阶段分别校验。

## 触发运行

GitHub 注册该工作流后，选择 **OperatorX Sweep → Run workflow**，指定源码分支，
并保留初始默认值：`pool=h100-dgxc`、`backends=torch`、`testlists=gemm`、
`world_sizes=1`、`chunk_size=500`。这会将完整的 GEMM 测试列表拆分为有界分片（目前为 7,212 个测试、15 个分片）。
列表中包含所选后端不支持的精度，以及可能超出设备显存的形状。不支持的测试会保留
在结果中；实际的内核和显存分配错误仍会使 CI 失败。运行完整列表不代表其中每个
测试都能在 H100 上执行。
新工作流可能需要先进入默认分支，GitHub 才允许手动触发。

```bash
gh workflow run operatorx-sweep.yml --repo SemiAnalysisAI/InferenceX \
  --ref <branch> -f pool=h100-dgxc -f backends=torch \
  -f testlists=gemm -f world_sizes=1 -f chunk_size=500
```

快速检查基础设施时，可显式选择 `testlists=gemm_perf` 和 `chunk_size=50`
（11 个 BF16 测试）。其他 NVIDIA 后端和测试列表需要显式选择，
不能视为已经通过 Hopper 验证。
不支持的操作会保留在结果中。后端导入错误、基准错误，以及没有任何成功结果，
都会使分片失败。先验证 BF16 GEMM，再验证范围受限的集合通信和兼容的 MoE 组合。
不要假定面向 Blackwell 的 FP4 内核可以在 Hopper 上运行。

## 执行约定

- 托管规划步骤校验输入，按容器镜像分组后端，区分 world size 和 MoE 并行参数组合，
  并将形状拆成大小受限的分片。最多支持 256 个分片。world size 仅允许 1、2、4、8，
  且不能超过一个物理节点的 GPU 数，因此 GB200/GB300 不接受 8。
  未被所选大小覆盖的形状会计入 `excluded_shapes`。
- 每个 Actions 分片独占一个物理 Slurm 节点，包含四张或八张 GPU。GPU 进程数等于所选 world size。
  准入沿用优先级评分器，以及 `ci-job-*`、`ci-attempt-*` 和唯一的 `nodes:1` 标签。
  所有分片均可被调度，由优先级/节点调度器控制物理节点准入。额外的 GitHub matrix
  并发上限可能将标签分配给尚未获准启动的作业，导致停滞。两个调度开关都必须保持启用。
- 运行器设置来自 CollectiveX 已纳入版本控制的平台配置。源码按工作流 SHA 检出，
  再复制到共享 squash 父目录或配置中可写的 `storage_roots` 路径（GB200）下的私有目录，
  该目录必须在计算节点上可见。B300 沿用 CollectiveX，从系统账户数据库读取计算节点可见
  的账户主目录；显式配置的 `stage_dir` 优先。结果不依赖提交主机的 `/tmp` 在计算节点上可见。
- 规划步骤解析镜像 digest。导入操作加锁，并按镜像、digest 和 CPU 架构缓存，导入后再次核对
  digest。标签发生变化或无法解析时运行失败，避免错误标注测量所用镜像。
  规划和导入主机都必须能匿名读取镜像。导入前校验主机 CPU 架构，并在已分配的计算节点
  上执行。B300 提交主机缺少该镜像所需的解压空间，因此沿用推理启动器的计算节点导入方式。
  Enroot 和 GNU parallel 使用私有临时目录；Enroot 使用显式 registry 地址及运行器池指定的
  缓存路径。分配请求保留 account、QoS 和隔离节点
  列表，并沿用 B300/GB 平台的 remap-root 与内存设置。B300 与推理启动器一致，由
  partition/account 选择 QoS；原来的 `batch_1_qos` 覆盖值会被当前集群拒绝。
  原配置中的隔离节点名称也不存在于该运行器池，已移除；Slurm 仍会遵循节点的 drain
  状态。GB300 继续使用其配置的 QoS 和隔离节点列表。
- 启动器等待分配、导入和执行完成。Slurm 分配限时 45 分钟；Actions 允许 70 分钟，
  包含排队与清理时间。Slurm 作业名与 Actions 运行器名称一致。
- 信号处理和工作流的 `always()` 恢复步骤会取消已记录的分配、停止写入、保留部分结果，
  然后删除暂存源码。工作流显式允许最多 180 秒等待 Slurm epilog 和节点释放，覆盖 H200
  的延迟释放情况。清理失败时保留暂存目录供调查。如果运行器主机失联，Slurm 时间限制
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
uv run --no-project --python 3.12 --with pytest --with pyyaml --with torch --with numpy \
  python -m pytest experimental/operatorx/tests/ -q
```

实际验收还需要在每个所选运行器池上执行带产物的 smoke 运行、失败分片重跑，以及确认释放分配的取消测试。
CPU 检查不能证明 GPU 兼容性或集群存储可见性。

最终覆盖汇总为每个请求分片选择最新产物尝试，保留此前尝试中已成功的分片，
并在任一分片缺失或失败时报告失败。汇总区分请求形状数和结果行数，
因为一个形状可能在多个后端上执行。

清理失败后，可在单分片触发中将 `recovery_run_id` 设为同一运行器池中近期的 OperatorX
运行 ID。工作流下载执行产物，在申请新节点前重试分配与暂存清理。恢复流程校验运行、
运行器池和私有暂存父目录；不要选择无关或过旧的 Slurm 执行。

`cleanup.log` 记录用于确认分配已释放的活动作业查询。查询使用当前用户的作业列表，
因为直接查询已删除的作业 ID，即使分配已终止，也可能返回 Slurm 错误。

## AMD 执行

`platforms.json` 在 CollectiveX 配置之上补充 AMDS Slurm 运行器池。
AMD 接受单卡 `torch` GEMM 和 `torch,aiter` attention。ROCm PyTorch 通过 `torch.cuda` 使用 HIP 事件计时；
FP8 在 gfx942 上选择 FNUZ，在 gfx950 上选择 OCP。不支持的格式会明确记录。
暂存目录由 `RUNNER_TEMP` 推导，位于共享运行器根目录下、`_work` 之外。容器不写入
源码检出目录。MI300X/MI325X 显式传递 `/dev/kfd` 和 `/dev/dri`；CPU 请求沿用各推理启动器。

## Attention

`testlists=attention_perf` 包含八个 BF16/FP16 MHA/GQA 及物化 MLA 的 prefill/decode
测试；`attention` 运行完整的 2,315 个测试。NVIDIA 使用 `backends=torch`，AMD 还支持
`backends=torch,aiter`。Attention 记录微秒延迟。不支持的精度或布局会明确记录；显存分配
或内核错误仍使 CI 失败。严格模式保留不支持的后端/算子组合，不会静默丢弃计划覆盖。

PyTorch 为矩形 decode 输入使用右下对齐的因果掩码，分组 KV 在计时前展开。两个 MLA
后端只测量物化 Q/K/V 的 attention，不包含压缩缓存投影或 RoPE。AITER 直接调用
`flash_attn_func`，保留原生分组 KV 和右下对齐的因果语义。当前支持统一 BF16/FP16、
连续 KV，以及不超过 256 且能被八整除的 head dimension；其他请求记录为不支持，
不会回退到 torch。实验算子变更记录在相邻的 `perf-changelog.yaml`，与根目录中受推理
配置键约束的变更日志分开维护。


## Kimi K3 路由专家 MoE 基准测试配置

选择 `testlists=kimi_k3_moe_perf`、`backends=vllm`、`world_sizes=1`，运行八个
BF16 路由专家测试：本地 token 数为 1、16、128、1024，分别使用 EP8 和 TP8 形状。
通用配置依据 [vLLM #50082](https://github.com/vllm-project/vllm/pull/50082)：
896 个专家、top-16、hidden 7168、intermediate 3072。EP8 分配 112 个专家，
intermediate 为 3072；TP8 分配 896 个专家，intermediate 为 384。
两者均在**单张 GPU** 上运行，不创建实际分布式通信组，也不测量通信。

界面明确标注为 **Kimi K3 (vLLM benchmark profile)**。它测量 vLLM 的通用 SiLU
路由专家内核，使用计时前生成的合成本地均匀随机路由。计时包含融合专家实现中的
 token 排序、gate/up GEMM、SiLU-and-multiply、down GEMM 和加权归约；不包含
router/top-k 计算、共享专家和通信。该配置不测量已发布 Kimi K3 层的 SITU 激活、
3584 维潜在专家路径、潜在投影或共享专家；这些需要独立的原生层测试配置。
无需模型权重或 Hugging Face 凭据。

NVIDIA 使用 `vllm/vllm-openai:v0.19.0`（amd64/arm64），AMD 复用现有 ROCm 镜像。
需显式选择 `backends=vllm`。不支持的精度、路由和共享专家请求会记录为 unsupported；
导入或内核执行错误使 CI 失败。

单卡路由矩阵乘法有效 TFLOPS 为
`6*num_tokens*top_k*hidden*(intermediate/routed_tensor_parallel_size)/(latency_us*1e6)`。
本地 top-k 全部指向本地专家表，与通用基准测试的 EP 模拟方式一致。不要再次除以 EP，
也不要乘以节点分配的 GPU 数。该指标不计激活或路由的 FLOP，不代表完整模型吞吐量。

### GPU 验证状态

完整的八项 BF16 测试已在 H200、MI300X 和 MI325X 上通过。H100 目前在执行内核前失败：Enroot 导入 vLLM 镜像时，在 `/tmp` 和 `/var/tmp` 均无法转换 OCI whiteout。CollectiveX swap-blocks 也记录了相同的主机限制。H100 需先恢复镜像导入环境，才能提供性能数据；本次改动不修改节点配置。B200、B300、GB200、GB300 和 MI355X 的任务正在等待共享 GPU 资源。注册 GPU 池不代表已完成运行时验证。
