# ubenchX

[English](README.md) | **中文**

独立微基准测试工具，无需依赖 OperatorX。请在已安装相应设备库的环境中，
从仓库根目录运行命令。

| 工具 | 测量内容 | 依赖 |
|------|----------|------|
| [gemm](gemm/bench.py) | 稠密矩阵乘法的延迟和 TFLOP/s | 支持 CUDA/ROCm 的 PyTorch，或适配昇腾的兼容 `torch_npu` 与 CANN |
| [mem_bw](mem_bw/bench.py) | 设备内存拷贝带宽，计入读取和写入 | 支持 CUDA/ROCm 的 PyTorch 与 Triton |
| [l2_distance/nvidia](l2_distance/nvidia/bench.py) | 各 SM 的 L2 指针追踪延迟差异 | NVIDIA GPU 与 CUDA 工具包（`nvcc`） |
| [l2_distance/amd](l2_distance/amd/bench.py) | 各 CU 的 L2 / Infinity Cache 指针追踪延迟差异 | AMD GPU（`gfx942`、`gfx950`）与 ROCm（`hipcc`、`rocminfo`） |
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

## L2 距离

```bash
python ubenchX/l2_distance/nvidia/bench.py             # NVIDIA：SM 间，链表大小 = L2
python ubenchX/l2_distance/amd/bench.py                # AMD：CU 间，链表大小 = 单个 XCD 的 L2（4 MB）
python ubenchX/l2_distance/amd/bench.py --chain-mb 64  # AMD：链表驻留在 Infinity Cache
python ubenchX/l2_distance/amd/plot_distance.py ubenchX/l2_distance/amd/results/distance.csv
python ubenchX/l2_distance/amd/plot_distance.py --iods 4 results/distance.csv   # MI300X：四个 IOD
```

`l2_distance/nvidia` 是 CUDA 的 SM 间基准测试；`l2_distance/amd` 是其面向 CDNA3/CDNA4 的
CU 间对应版本。每个 CU 上运行一个单 wavefront 的工作组，
通过申请全部 LDS 将其固定为每 CU 一个；该工作组追踪一个由 128 字节行组成的随机链表。
每次依赖加载都是设备作用域（`sc1`）加载，会绕过 CU 的向量 L1 并由所在 XCD 的 L2 提供数据，
并用 `s_memtime` 以着色器时钟周期计时。每跳延迟先暂存在 LDS 中，因此计时循环不会发出任何
全局存储。CU 通过 `HW_ID` 和 `XCC_ID` 硬件寄存器识别，并以 `cu = xcd << 6 | se << 4 | cu_id`
的形式报告。

链表大小决定被测的缓存层级。默认值等于 HIP 报告的单个 XCD 的 L2 容量（MI300X 与 MI355X 为
4 MB）。`--chain-mb 64` 超过 L2，驻留在 256 MB 的 Infinity Cache 中；`--chain-mb 1024` 则连
Infinity Cache 也超过。每个 XCD 都有私有且组织方式相同的 L2，因此驻留在 L2 的链表不会显示
跨 XCD 的结构，只反映 CU 在其 XCD 内的位置（SE0/SE1 与 SE2/SE3 两半）。驻留在 Infinity Cache
的链表则显示 IOD 局部性：在 MI355X（两个 IOD）上为 XCD 0-3 与 XCD 4-7；在 MI300X（四个 IOD）上为
XCD 配对 {0,1}、{2,3}、{4,5}、{6,7}，同一 IOD 上的两个 XCD 无法区分，跨 IOD 的距离分为三个层级。
`plot_distance.py` 先按 IOD（`--iods` 指定连续 XCD id 分组数，默认 2，MI300X 为 4；`--iod-groups`
可显式指定分组，或用 `cluster` 从数据聚类得到分组）分组，再按 XCD 和着色器引擎对 CU 分组。在 LDS 为
64 KB 的型号（MI300X）上，4 MB 链表分两段计时，并在下一段开始前重新触碰其行。

`--num-passes` 以打乱的起点偏移进行多轮平均，`--num-hops` 限制每个 CU 的计时跳数，
`--warm-hops` 设置计时前填充缓存的非计时遍历。成对 CSV（`cu_a, cu_b, mean_abs_diff,
xcd_a, xcd_b`）输出到 stdout；`results/cu_info.csv` 保存每个 CU 的 XCD、SE、CU id 和平均延迟，
`results/latency_profile.csv` 保存第一个 CU 的排序延迟曲线。
