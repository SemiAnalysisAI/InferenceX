<div align="center">

[English](./TRAINIUM3_HARDWARE_ACCOUNTING.md) | **中文**

</div>

# OperatorX 校准所用的 Trainium3 硬件口径

## 范围

本次评估仅覆盖当前 `trn3pd98.3xlarge` 主机可见的 Trainium3 分区。计算校准
单位为一个 LNC=2 逻辑 NeuronCore（`NC_v4d`）：两个物理 NeuronCore-v4 共享
一个 HBM bank。分区内其他逻辑设备保持空闲。GEMM 评估不涉及集合通信、
NeuronLink 假设，也不外推至 UltraServer。

## 实机清点结果

环境清单嵌入精简后的 OperatorX 结果。2026-07-31 的主机信息如下：

| 证据 | 观测值 |
|---|---|
| EC2 实例类型 | `trn3pd98.3xlarge` |
| PCI 设备 | 一个 `NeuronDevice (Trainium3)` |
| `neuron-ls -j` 物理核心数 | `nc_count: 8` |
| `neuron-ls -j` 暴露的 ID | `0, 1, 2, 3` |
| 设备内存 | 154,618,822,656 字节 = 144 GiB |
| Torch/XLA 交叉检查 | 四个设备，类型为 `NC_v4d` |
| NKI 编译目标 | `trn3`，LNC=2 |
| NKI | `0.3.0+23928721754.g18aa1271` |
| Neuron 编译器 | `2.24.5133.0+58f8de22` |
| Neuron runtime | `2.31.24` |
| Neuron profiler/tools | `2.29.18.0` |

八个与四个核心的表面差异来自 LNC=2：芯片有八个物理 NeuronCore-v4，
每两个组成一个逻辑 NeuronCore，因此共暴露四个逻辑设备。NKI 编译产物记录
`lnc: 2`，基准测试以二路 grid 启动内核。

## 公开规格交叉检查

主要资料：

- [AWS Trainium3 架构](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.29.1/about-neuron/arch/neuron-hardware/trainium3.html)
- [AWS 逻辑 NeuronCore 配置](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/logical-neuroncore-config.html)
- [AWS Trainium3 NKI 架构指南](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium3_arch.html)
- [AWS NKI LNC 指南](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.29.1/nki/get-started/about/lnc.html)

AWS 公布的每芯片规格为八个物理核心、144 GiB HBM、671 BF16/FP16/TF32
TFLOP/s、2,517 MXFP8/MXFP4 TFLOP/s、183 FP32 TFLOP/s 和 4.9 TB/s HBM
带宽。按四个 LNC=2/HBM-bank 单元均分后，审计结果如下：

| 指标 | 公布的整芯片值 | 推导的 LNC=2 值 | 当前 SimulatorX `trn3` | 本次处理方式 |
|---|---:|---:|---:|---|
| HBM 容量 | 144 GiB | 36 GiB | 36 GiB | 一致 |
| HBM 带宽 | 4.9 TB/s | 1.225 TB/s | 1.225 TB/s | 保持当前值，并标注公开资料冲突 |
| BF16 峰值 | 671 TFLOP/s | 167.75 TFLOP/s | 158 TFLOP/s | 按当前配置评估 |
| MXFP8/MXFP4 峰值 | 2,517 TFLOP/s | 629.25 TFLOP/s | 630 TFLOP/s | 基本一致 |
| FP32 峰值 | 183 TFLOP/s | 45.75 TFLOP/s | 40 TFLOP/s | 不在 BF16 试验范围内 |
| 静态内存延迟 | 未公布 | 未知 | 200 ns | 暂定值 |
| 启动延迟 | 未公布 | 未知 | 300 ns | 暂定值 |

NKI 架构指南目前写的是 4.7 TB/s，而 Trainium3 架构页写的是 4.9 TB/s。
SimulatorX 使用 4.9 TB/s 并均分到四个 HBM-bank 单元。本次评估记录这一冲突，
但不修改设备配置。同样，本次直接报告当前 158 TFLOP/s BF16 配置的基线，
不会悄然替换成由 671 TFLOP/s 推导出的 167.75 TFLOP/s。

## 统一口径

- OperatorX 请求的 shape 是模型逻辑 GEMM。
- NKI 实际执行及 padding 后的 shape 单独记录。
- 每个保留的精简样本中，NKI profile 的 `hardware_flops` 与实际执行 shape 的
  `2*M*N*K` 偏差不得超过 0.01%。这既保持严格的身份诊断，也允许 profiler
  推导计数器遗漏边界 tile 的贡献。
- SimulatorX 在一个 `trn3` LNC=2 设备上预测请求的逻辑 shape。
- NKI padding 属于真实实现开销，因此保留在模型误差中，不通过预测 padding 后
  shape 来掩盖。
- Profile 的 HBM 读写计数作为实现实际流量保留，不要求等于逻辑 A+B+C 字节数。
- 结果仅适用于该分区上的一个 LNC=2，不对集合通信或整机相关性作出结论。
