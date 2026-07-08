# 多节点基准测试契约

<div align="center">

[English](./README.md) | **中文**

</div>

`benchmarks/multi_node` 下的文件只描述工作负载。它们不得选择集群、提交调度任务、映射主机存储、选择容器权限，也不得根据主机名推断网络策略。这些决策必须由 `runners` 下的代码负责。

工作流和主配置提供工作负载意图：

- 模型和镜像：`MODEL`、`IMAGE`
- 请求形状：`ISL`、`OSL`、`CONC_LIST`、`RANDOM_RANGE_RATIO`
- 推理引擎：`FRAMEWORK`、`SPEC_DECODING`
- 拓扑：`PREFILL_*` 和 `DECODE_*`

集群启动器将这些意图解析为本地资源，并注入一个入口：

```bash
export MULTINODE_LAUNCHER="$GITHUB_WORKSPACE/runners/<cluster>/submit.sh"
bash "benchmarks/multi_node/<recipe>.sh"
```

`run_disaggregated.sh` 校验工作负载契约，统一 EP/DP 和节点列表的表达方式，然后调用注入的启动器。启动器通过已导出的环境变量接收工作负载，并负责补充模型路径、调度器账号与分区、设备名、主机挂载、容器运行时配置和日志位置等集群参数。

## 添加配方

大多数配方应当只尾调用 `disaggregated_recipe.sh`。只有在模型或引擎行为适用于所有集群时，才应在调用前增加导出变量。主机路径、节点名、Slurm 参数和 Docker 配置都属于启动器配置，而不属于配方配置。

## 添加集群

在 `runners/launch_<cluster>.sh` 下创建启动器；如有需要，再在 `runners/<cluster>/` 下创建集群适配器。在 `configs/runners.yaml` 中声明公共模型到本地存储的映射以及缓存根目录，并通过 `runners/lib/runner_config.sh` 读取。启动器负责其余集群策略。适配器负责启动容器或提交调度任务，并安排容器内执行可移植的运行时脚本。

`runners/mi355x-amds` 中的 MI355X AMDS 实现是 Slurm + Docker 适配器的参考。NVIDIA srt-slurm 启动器通过生成的 `srtctl` 配置遵循同一职责边界。

## srt-slurm

所有 NVIDIA 启动器都通过 `runners/lib/srt_slurm.sh` 使用 `configs/runners.yaml` 中声明的仓库、分支和不可变 `main` 快照。启动器不得自行克隆或选择其他 srt-slurm 分支。该辅助脚本会覆盖 InferenceX 内已检入的配方，以纯数据方式获取仅存在于锁定旧快照中的配方，解析配方选择器，并将 `sa-bench` 条目转换为 srt-slurm 的自定义基准测试接口。自定义命令运行 `srt_benchmark.sh`，后者调用 InferenceX 仓库内的基准测试客户端。Agentic 等已有的自定义基准测试会原样保留。

尚未上游合入的小型兼容补丁存放在 `runners/srt-slurm-patches/`，并通过 `git apply --check` 应用于该精确快照。这些补丁不得选择其他分支或修订版本。
