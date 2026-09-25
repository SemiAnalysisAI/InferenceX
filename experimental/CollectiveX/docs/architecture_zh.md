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
2. [启动器](../launchers/) 申请硬件资源，并将源码暂存到隔离目录。Slurm 运行器池为每个 GPU
   启动一个 `run_ep.py` 进程；台湾 Docker 运行器池使用 `torchrun`。分片内的用例顺序执行。
3. [run_ep.py](../bench/run_ep.py) 初始化 GPU，根据 `BACKENDS` 延迟导入适配器模块，再建立
   进程组、构造适配器，最后调用 [ep_harness.run_sweep](../bench/ep_harness.py)。
4. 测试驱动准备输入、校验正确性、测量独立组件与连续调用周期，然后再次校验正确性。
   [ep_results.py](../bench/ep_results.py) 归约样本，由 rank 0 写入 case-attempt JSON，
   并在各 rank 间统一退出状态。
5. 启动器收集 JSON 文件。工作流生成延迟和带宽摘要，并上传 `cxshard-*` 产物。
   清理流程负责释放资源，也覆盖启动器被中断的情况。用例失败仍会使分片失败；部分完成
   或失败的执行也会尝试上传已有结果。

## Python 模块职责

| 模块 | 职责 |
| --- | --- |
| [ep_case.py](../bench/ep_case.py) | 用例 ID、CLI 输入、token 数量梯度和运行时版本格式化；不导入厂商库 |
| [ep_backend.py](../bench/ep_backend.py) | 抽象通信接口、`RankInputs`、`WorkloadSpec`、确定性输入和 FP8 不变量 |
| [ep_timing.py](../bench/ep_timing.py) | `EPTiming`：预热、分发与合并的配对规则、独立计时区间和两条计时链 |
| [ep_measurement.py](../bench/ep_measurement.py) | CUDA event 计时、跨 rank 归约、百分位数和 `PointSamples` |
| [ep_oracle.py](../bench/ep_oracle.py) | 独立参考计算、各接收布局的校验，以及共用的清理和合并结果校验 |
| [ep_results.py](../bench/ep_results.py) | 字节数计算、产物格式、原子写入和结果日志 |
| [ep_harness.py](../bench/ep_harness.py) | 单个用例中按既定顺序执行的正确性校验与测量阶段 |
| [ep_legacy.py](../bench/ep_legacy.py) | DeepEP 与 UCCL 兼容的 legacy Buffer API 所共用的操作 |
| `ep_deepep_v2.py`、`ep_uccl.py`、`ep_mori.py`、`ep_nccl.py`、`ep_flashinfer.py` | 各厂商库特有的构造、通信、接收视图和清理逻辑 |
| [routing.py](../bench/routing.py) | 确定性路由、激活值、源 token 身份和局部性统计 |

`EPBackend` 从 `EPTiming` 继承计时实现。DeepEP 和 UCCL 还继承 `LegacyBufferOperations`，
但普通模式的通信实现和各库的量化器仍位于各自适配器中。在 API 一致的路径上，经过参考
变换的校验输入复用适配器的常规合并操作。后端库仍在 GPU 初始化之后延迟导入。

## 运行时模块职责

[runtime/common.sh](../runtime/common.sh) 是供调用方 `source` 的入口，负责日志和运行器
配置加载，再将下列模块加载到同一个 shell 中：

| 模块 | 职责 |
| --- | --- |
| [network.sh](../runtime/network.sh) | 网络设备选择、链路层规则和网络校验 |
| [slurm.sh](../runtime/slurm.sh) | 资源分配、分布式会合、rank 身份、健康检查和资源释放 |
| [images.sh](../runtime/images.sh) | 镜像身份、导入锁、缓存复用和导入重试 |
| [sources.sh](../runtime/sources.sh) | 后端源码与版本固定、精确 commit 暂存和缓存挂载 |
| [staging.sh](../runtime/staging.sh) | 计算节点可见的源码隔离、结果收集和暂存目录清理 |
| [execution.sh](../runtime/execution.sh) | 用例执行、后端准备和启动器清理 trap |

[prepare_backend.sh](../runtime/prepare_backend.sh) 校验容器内网络并写入 rank 环境文件。
它加载 [build_common.sh](../runtime/build_common.sh) 中的工具链发现和受锁保护的缓存安装
逻辑，以及各个[后端构建模块](../runtime/backends/)。启动器在申请资源前暂存固定版本的
源码，各节点在启动 GPU rank 前完成后端准备。修改安装器时，应一并检查缓存身份、就绪
条件和错误处理。

## 独立的块复制路径

`backend=swap-blocks` 选择 [swap_matrix.py](../swap_matrix.py)、
[launch_swap-blocks.sh](../launchers/launch_swap-blocks.sh) 和
[run_swap_blocks.py](../bench/run_swap_blocks.py)。该路径使用函数与回调，在单 GPU 上运行，
采用独立的墙钟计时方式和 `collectivex-swap-blocks-v1` 产物格式。其测量约定见
[swap-blocks_zh.md](swap-blocks_zh.md)。

## 验证边界

[测试](../tests/) 覆盖矩阵生成、CLI 参数传递、正确性模型、event 放置、完整产物生成、
缓存行为、源码暂存和运行时错误处理。修改运行时辅助代码时，还需运行
`experimental/operatorx/tests/`：OperatorX 也使用平台配置，并复制运行时目录。
CPU 测试无法证明 GPU 通信性能，相关证据仍需通过真实资源分配获得。
[测量方法](methodology.md)定义了测量语义。
