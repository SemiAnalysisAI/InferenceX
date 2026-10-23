# ubenchX

[English](README.md) | **中文**

独立微基准测试工具，无需依赖 OperatorX。请在已安装相应设备库的环境中，
从仓库根目录运行命令。

| 工具 | 测量内容 | 依赖 |
|------|----------|------|
| [gemm](gemm/bench.py) | 稠密矩阵乘法的延迟和 TFLOP/s | 支持 CUDA/ROCm 的 PyTorch，或适配昇腾的兼容 `torch_npu` 与 CANN |
| [mem_bw](mem_bw/bench.py) | 设备内存拷贝带宽，计入读取和写入 | 支持 CUDA/ROCm 的 PyTorch 与 Triton |
| [sm_l2_distance](sm_l2_distance/bench.py) | 各 SM 的 L2 指针追踪延迟差异 | NVIDIA GPU 与 CUDA 工具包（`nvcc`） |
| [gpc_query](gpc_query/) | 通过线程块集群调度查询 GPC/TPC 配置 | 支持 `sm_90+` 的 NVIDIA GPU 与 CUDA 工具包 |

## GEMM

```bash
python ubenchX/gemm/bench.py --device npu --shape 8192 8192 8192
python ubenchX/gemm/bench.py --device cuda --shape 8192 8192 8192
```

`--device cuda` 同时适用于 NVIDIA CUDA 和 AMD ROCm 版本的 PyTorch。
`--device-index` 选择逻辑设备，默认值为 `0`。昇腾需要相互兼容的驱动、
CANN、PyTorch 和 `torch_npu` 环境。`--dtype` 默认为 `bf16`，也支持 `fp16`
和 `fp32`。每次运行的输入与输出使用同一种数据类型。运算不包含偏置或激活函数；
尚未实现 FP8 和 FP4。
精度控制选项沿用运行环境的设置，并在相关接口可用时记录。

矩阵形状按 **M K N** 顺序指定，对应 `A[M,K] @ B[K,N]`。可重复使用
`--shape` 指定多个形状，也可通过 `--shapes shapes.json` 读取 JSON 列表，
例如 `[[256, 4096, 4096], [8192, 8192, 8192]]`。这两种输入方式不能混用。

脚本以 eager 模式运行 `torch.matmul`，先预热五次，再通过设备事件采集十个
耗时样本，报告排序后的第六个样本（上中位数）。每次计时区间之前会将一个
临时缓冲区清零，清零操作不计入耗时。缓冲区默认采用设备报告的 L2 缓存大小；
`--flush-mib 256` 可显式设为 256 MiB。临时缓冲区写入用于准备缓存状态，
不保证完全驱逐缓存中的数据。

TFLOP/s 按 `2*M*K*N / elapsed_seconds / 1e12` 计算。可选参数
`--peak-tflops 378` 将调用方指定的稠密计算峰值作为分母，额外计算 MFU；
请根据实际设备与数据类型选择分母。该指标反映 GEMM 利用率，并非整个模型的
MFU。结果以 CSV 格式输出至 stdout，运行环境元数据以 JSON 格式输出至 stderr。
CSV 每行的 `timings_us` 包含全部十个样本，同时记录 `status` 和 `error`。
`ai` 按每个输入读取一次、输出写入一次估算 FLOP/byte，并非实测内存流量。
某个形状运行失败时会记录错误并继续后续形状；只要有形状失败，最终退出码即为 1：

```bash
python ubenchX/gemm/bench.py --device npu --shapes shapes.json \
  --dtype bf16 --flush-mib 256 --peak-tflops 378 > gemm.csv 2> gemm-metadata.json
```

GEMM 路径尚未在加速器硬件上验证，未附带硬件实测结果。脚本不将输出与数值参考结果比对。
