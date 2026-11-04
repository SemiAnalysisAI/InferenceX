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
| [trn3_dma_bw](trn3_dma_bw/) | Trainium3 HBM 与 SBUF 之间的 DMA 带宽，覆盖静态、硬件和软件描述符生成 | 安装 Neuron 驱动和 Docker 的 Trainium3 实例 |

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

## Trainium3 DMA 带宽

`trn3_dma_bw` 测量 Trainium3 上 HBM 到 SBUF（load）和 SBUF 到 HBM（store）的 DMA 带宽，
覆盖 LNC=1 下的单个 NeuronCore，以及一颗芯片上 8 个 NeuronCore 同时运行的情况。
它比较 `nisa.dma_copy` 的三种描述符生成（DGE）模式：`none`（执行前写入 HBM 的静态描述符）、
`hwdge`（运行时由 DGE 硬件模块生成）和 `swdge`（运行时由 GpSimd 引擎生成）。
每个 NKI kernel 是一组连续发出的 `[pdim, fdim]` fp16 拷贝；同一方向的三个 kernel
仅 `dge_mode` 不同。

命令在 `trn3-ubench` 镜像（Neuron SDK 2.33，NKI 0.7）中运行。`run.sh` 负责构建镜像、
传入所有 `/dev/neuron*` 设备、将仓库挂载到 `/InferenceX`，并以调用者的 uid 运行。
请从仓库根目录执行：

```bash
NEURON_TARGET=trn3pre OUT_DIR=ubenchX/trn3_dma_bw/runs/single CORE=0 \
  KERNELS="load_static load_hwdge load_swdge store_static store_hwdge store_swdge" \
  ubenchX/trn3_dma_bw/run.sh ubenchX/trn3_dma_bw/sweep.sh

NEURON_TARGET=trn3pre OUT_DIR=ubenchX/trn3_dma_bw/runs/allcore KERNEL=load_static FDIM=8192 \
  TOTAL_MIB=256 NCORES=8 NUM_EXEC=20000 PROFILE_EVERY=5000 \
  ubenchX/trn3_dma_bw/run.sh ubenchX/trn3_dma_bw/allcore.sh

ubenchX/trn3_dma_bw/run.sh python ubenchX/trn3_dma_bw/parse.py \
  ubenchX/trn3_dma_bw/runs/single/* --csv single.csv
ubenchX/trn3_dma_bw/run.sh python ubenchX/trn3_dma_bw/parse.py \
  ubenchX/trn3_dma_bw/runs/allcore/*/solo ubenchX/trn3_dma_bw/runs/allcore/*/core? --csv allcore.csv
ubenchX/trn3_dma_bw/run.sh python ubenchX/trn3_dma_bw/report.py --single single.csv --allcore allcore.csv
```

`NEURON_TARGET` 取 `trn3` 表示 Trainium3 硅片修订版 v2，取 `trn3pre` 表示修订版 v1；
若 NEFF 面向比芯片更新的修订版构建，运行时会拒绝加载。`sweep.sh` 对 `KERNELS` 中的每个
kernel 在 pdim=128 下测试 fdim 256 到 16384（每行 0.5 到 32 KiB），并对 `load_static`
额外测试 pdim 32 和 64；每个测试点约以 1,024 次拷贝为上限，以缩短编译时间。
`allcore.sh` 先在 core 0 上运行单核基线，再在 `NCORES` 个核心上各启动一个进程同时运行；
每个进程连续执行 `NUM_EXEC` 次，每隔 `PROFILE_EVERY` 次采集一次 profile，
使采集到的执行都落在所有核心同时忙碌的时间段内。

带宽取自 Neuron profiler 中单次被采集执行的逐包 `DmaPacket` 表：有效载荷字节数除以
从突发传输第一个包开始到最后一个包结束的时间，因此其中的空闲间隙也计入耗时。
突发传输取搬运字节最多的 DMA 队列类型（静态为 `input`，hwdge 为 `hardware_dynamic`，
swdge 为 `software_dynamic`）。4 字节的完成写入不计入有效载荷。使用 swdge 时，
load kernel 最后的输出写回与突发传输共用同一队列类型；`parse.py` 按字节数将其截去
（`trimmed_bytes`），若与突发传输时间重叠则报错。`bytes_ok` 确认 trace 包含突发传输的
全部字节，`kernel_GBps_lb` 用同样的字节数除以整次执行时间。`report.py` 输出三种模式的
对比表、逐 kernel 表和整芯片表；单核 MBU 以 612.5 GB/s（芯片 HBM 4.9 TB/s 除以
8 个 NeuronCore）为分母，整芯片 MBU 对各核心中位数求和。`plot.py` 按模式绘制带宽随消息
大小的变化（需要 `matplotlib`）。

`results/trn3pd98_2026-11-03_*` 测于搭载修订版 v1 硅片（`trn3pre`）的 trn3pd98.3xlarge。
每行 8 KiB 及以上时，三种模式单核均达到 588-611 GB/s；每行 16 KiB 时，8 个核心合计达到
4.58-4.62 TB/s。
每行 0.5 KiB 时，load/store 在静态描述符下为 412/544 GB/s，hwdge 为 108/110，
swdge 为 94/91：每次 hwdge 拷贝至少耗时约 0.6 us，每次 swdge 拷贝约 0.7 us，
因此小拷贝受指令速率限制，而非 HBM。

kernel 不将传输数据与参考结果比对。`store_static` fdim=512 的 profile 只采集到约一半的包
（`bytes_ok` 为 false）。目前仅覆盖 LNC=1。
