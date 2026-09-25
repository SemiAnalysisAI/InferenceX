[English](architecture.md) | **中文**

# CollectiveX 实现结构

CollectiveX 包含两条执行路径：分布式专家并行(EP)通信，以及单 GPU 上的 vLLM 块复制。
工作流和启动器负责编排进程，EP 适配器实现统一的 Python 接口。配置、测量语义、产物名称
和退出状态构成各层之间的接口约定。

## 从触发到结果

1. [collectivex-sweep.yml](../../../.github/workflows/collectivex-sweep.yml) 接收手动触发请求。
   [sweep_matrix.py](../sweep_matrix.py) 展开工作负载和平台配置。每个分片按 GPU 运行器池、
   后端、模式、节点数和精度组织用例，每个用例包含完整的 token 数量梯度。不支持的请求
   用例仍保留在记录中，但不进入执行矩阵。
2. [ci.py](../ci.py) 调用 Python [执行控制器](../runtime/execution.py)申请硬件资源，
   并将源码暂存到隔离目录。Slurm 运行器池为每个 GPU
   启动一个 `run_ep.py` 进程；台湾 Docker 运行器池使用 `torchrun`。分片内的用例顺序执行。
3. [run_ep.py](../bench/run_ep.py) 初始化 GPU，根据 `BACKENDS` 延迟导入适配器模块，再建立
   进程组、构造适配器，最后调用 [ep_harness.run_sweep](../bench/ep_harness.py)。
4. 测试驱动准备输入、校验正确性、在支持的模式下测量 graph 回放（其他模式测独立组件与连续调用周期），然后再次校验正确性。
   [ep_results.py](../bench/ep_results.py) 归约样本，由 rank 0 写入 case-attempt JSON，
   并在各 rank 间统一退出状态。
5. 启动器收集 JSON 文件。工作流生成延迟和带宽摘要，并上传 `cxshard-*` 产物。
   清理流程负责释放资源，也覆盖启动器被中断的情况。用例失败仍会使分片失败；部分完成
   或失败的执行也会尝试上传已有结果。

## Python 模块职责

| 模块 | 职责 |
| --- | --- |
| [run_ep.py](../bench/run_ep.py) | CLI 输入、按需后端分派、运行时初始化和版本记录 |
| [ep_backend.py](../bench/ep_backend.py) | 抽象通信接口、`RankInputs`、`WorkloadSpec`、确定性输入和 FP8 不变量 |
| [ep_measurement.py](../bench/ep_measurement.py) | `EPTiming` 预热与 graph/eager 计时模板、CUDA event、跨 rank 归约、分位数和 `PointSamples` |
| [ep_oracle.py](../bench/ep_oracle.py) | 独立参考计算、各接收布局的校验，以及共用的清理和合并结果校验 |
| [ep_results.py](../bench/ep_results.py) | 用例身份、字节数计算、产物格式、原子写入和结果日志 |
| [ep_harness.py](../bench/ep_harness.py) | 单个用例中按既定顺序执行的正确性校验与测量阶段 |
| [ep_legacy.py](../bench/ep_legacy.py) | DeepEP 与 UCCL 兼容的 legacy Buffer API 所共用的操作 |
| `ep_deepep_v2.py`、`ep_uccl.py`、`ep_mori.py`、`ep_nccl.py`、`ep_flashinfer.py` | 各厂商库特有的构造、通信、接收视图和清理逻辑 |
| [routing.py](../bench/routing.py) | 确定性路由、激活值、源 token 身份和局部性统计 |

`EPBackend` 从 `EPTiming` 继承计时实现。DeepEP 和 UCCL 还继承 `LegacyBufferOperations`，
但普通模式的通信实现和各库的量化器仍位于各自适配器中。在 API 一致的路径上，经过参考
变换的校验输入复用适配器的常规合并操作。后端库仍在 GPU 初始化之后延迟导入。

## 运行时模块职责

[ci.py](../ci.py) 是工作流和手动执行的统一入口。`matrix`、`extract`、`execute`、`finalize`
和 `cleanup` 将调度与产物上传留给 Actions，执行生命周期由 Python 管理。工作流直接调用
宿主机 venv 中的解释器，避免其 `PATH` 和 `VIRTUAL_ENV` 替换容器内固定镜像的 Python。

| 模块 | 职责 |
| --- | --- |
| [config.py](../runtime/config.py) | 运行器池资源请求、配置优先级、容器选项和用例/块复制参数 |
| [scheduler.py](../runtime/scheduler.py) | `simple-slurm` 步骤、资源生命周期、子进程、私有日志、文件锁和信号处理 |
| [probe.py](../runtime/probe.py) | 硬件与网络探测、设备选择、链路层规则和镜像摘要查询 |
| [storage.py](../runtime/storage.py) | 镜像缓存身份与导入，以及隔离源码暂存和结果收集 |
| [build.py](../runtime/build.py) | 精确源码/依赖版本、源码准备、受锁保护的构建缓存和后端环境 |
| [node.py](../runtime/node.py) | 标准库节点工具、逐节点准备，以及 `exec` 前的 rank 环境加载 |
| [execution.py](../runtime/execution.py) | 资源分配重试、准备、用例顺序执行和可恢复清理 |
| [docker.py](../runtime/docker.py) | 无 Slurm 运行器池执行、Docker 构建缓存和本次容器清理 |

宿主机使用 [simple-slurm](https://github.com/amq92/simple_slurm)，版本固定在 `collectivex`
可选依赖中。保留 `salloc --no-shell` 和针对指定作业的 `squeue`/`scancel` 调用，以保持原有
资源分配生命周期。库负责执行 `srun`；传入其 shell 字符串接口的参数均做转义，无值开关和
Pyxis 选项显式传递。仓库中不再维护 CollectiveX Bash 启动器。

共享存储或容器尚不可用时，宿主机通过标准输入发送仅依赖标准库的 zipapp，在计算节点上
运行工具。后端按节点准备一次，并将白名单环境变量写入私有 JSON。rank 启动代码加载该
环境，从 Slurm 获取 rank 身份，再执行原有基准测试。用例参数直接使用 Python 列表，
不再生成 NUL 分隔参数文件或 shell 环境脚本。

`execution.json` 和 `jobid` 记录资源归属，供独立的工作流清理步骤恢复。只有确认资源分配
已终止后，才收集结果并删除暂存目录；无法确认时保留恢复记录。信号退出保持 `128 + signal`。
失败后仍收集已生成的结果，最终清理仅处理本次执行已记录的容器。

## 独立的块复制路径

`backend=swap-blocks` 选择 [swap_matrix.py](../swap_matrix.py)、
执行控制器（Slurm）或 Docker 执行器，以及
[run_swap_blocks.py](../bench/run_swap_blocks.py)。该路径使用函数与回调，在单 GPU 上运行，
采用独立的墙钟计时方式和 `collectivex-swap-blocks-v1` 产物格式。其测量约定见
[swap-blocks_zh.md](swap-blocks_zh.md)。

## 验证边界

[测试](../tests/) 覆盖矩阵生成、CLI 参数传递、正确性模型、event 放置、完整产物生成、
缓存行为、源码暂存和运行时错误处理。修改运行时辅助代码时，还需运行
`experimental/operatorx/tests/`：OperatorX 也使用平台配置，并复制运行时目录。
CPU 测试无法证明 GPU 通信性能，相关证据仍需通过真实资源分配获得。
[测量方法](methodology_zh.md)定义了测量语义。
