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
| [trn3_psum_bw](trn3_psum_bw/) | Trainium3 上 VectorE 与 ScalarE 的 PSUM 到 SBUF `tensor_copy` 带宽，含单引擎与双引擎并发 | 安装 Neuron 驱动和 Docker 的 Trainium3 实例 |
| [trn3_exp_bw](trn3_exp_bw/) | Trainium3 上 VectorE 与 ScalarE 的 exp 及激活函数吞吐量 | 安装 Neuron 驱动和 Docker 的 Trainium3 实例 |
| [trn3_matmul](trn3_matmul/) | Trainium3 TensorE bf16 矩阵乘法在不同 N 和 M 下的发射间隔与延迟，含固定或每次新加载的 stationary | 安装 Neuron 驱动和 Docker 的 Trainium3 实例 |
| [full_chip_matmul](full_chip_matmul/) | Trainium3 整芯片 TensorE 矩阵乘法 TFLOP/s（bf16、fp8 double_row、MXFP8）：八个 NeuronCore 同时运行与单核单独运行的对比，以及各设备性能 profile 下的限速读数 | 安装 Neuron 驱动和 Docker 的 Trainium3 实例，以及 AWS 的 `performance_profile_tool`（需 `sudo`） |

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

命令在 `trn3-ubench` 镜像（Neuron SDK 2.33，NKI 0.7，由 `ubenchX/trn3_docker` 构建）中运行。`run.sh` 负责构建镜像、
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
使采集到的执行都落在所有核心同时忙碌的时间段内。LNC=1 时每颗芯片有 8 个核心编号
（芯片 k 为核心 8k 到 8k+7），因此在多芯片主机上 `CORE` 同时决定芯片，`allcore.sh`
始终使用芯片 0。

带宽取自 Neuron profiler 中单次被采集执行的逐包 `DmaPacket` 表：有效载荷字节数除以
从突发传输第一个包开始到最后一个包结束的时间，因此其中的空闲间隙也计入耗时。
突发传输取搬运字节最多的 DMA 队列类型（静态为 `input`，hwdge 为 `hardware_dynamic`，
swdge 为 `software_dynamic`）。4 字节的完成写入不计入有效载荷。使用 swdge 时，
load kernel 最后的输出写回与突发传输共用同一队列类型；`parse.py` 按字节数将其截去
（`trimmed_bytes`），若与突发传输时间重叠则报错。`bytes_ok` 确认 trace 包含突发传输的
全部字节，`kernel_GBps_lb` 用同样的字节数除以整次执行时间。`report.py` 输出三种模式的
对比表、逐 kernel 表、整芯片表，并在同时给出两个 CSV 时输出单核与整芯片 MBU 对比；单核 MBU 以 612.5 GB/s（芯片 HBM 4.9 TB/s 除以
8 个 NeuronCore）为分母，整芯片 MBU 对各核心中位数求和。`plot.py` 按模式绘制带宽随消息
大小的变化：输入单核 CSV 时绘制单核带宽，加 `--chip` 并输入多核 CSV 时绘制整芯片总带宽
（需要 `matplotlib`）。

`results/trn3pd98_2026-11-03_*` 测于搭载修订版 v1 硅片（`trn3pre`）的 trn3pd98.3xlarge。
每行 8 KiB 及以上时，三种模式单核均达到 588-611 GB/s；每行 16 KiB 时，8 个核心合计达到
4.58-4.62 TB/s。
每行 0.5 KiB 时，load/store 在静态描述符下为 412/544 GB/s，hwdge 为 108/110，
swdge 为 94/91：每次 hwdge 拷贝至少耗时约 0.6 us，每次 swdge 拷贝约 0.7 us，
因此小拷贝受指令速率限制，而非 HBM。

`results/trn3_48xlarge_rev2_*` 测于搭载修订版 v2 硅片（`trn3`）的 trn3.48xlarge 的芯片 0；
`trn3_48xlarge_rev2_2026-11-05_report.md` 收录 `report.py` 输出的表格。其整芯片数据对每个
kernel 在 `sweep.sh` 的每个 fdim 上运行 `allcore.sh`，`TOTAL_MIB` 上限与 `sweep.sh` 相同。
每行 8 KiB 及以上时，单核结果与 v1 相差不超过 0.2%。小拷贝在 hwdge 下明显更快：每行
0.5 KiB 时，load/store 在 hwdge 下为 398/476 GB/s（v1：108/110），静态描述符下为 492/544
（v1：412/544），swdge 下为 95/93。每次 hwdge 拷贝耗时约 0.14-0.17 us，而非约 0.6 us；
hwdge 发出每个包含多行的 4 KiB 包，而 v1 每行一个包；swdge 每次拷贝仍约 0.7 us。
8 个核心合计在每行 16 KiB 时达到 4.71-4.81 TB/s（v1：4.58-4.62），hwdge load/store 在
每行 2 KiB 时达到 4.57/4.61 TB/s（v1：3.42/3.52）。大行下整芯片带宽的提升可能部分来自
实例不同，而不仅是硅片修订版。整芯片 MBU 比单核 MBU 低 0 到 3.2 个百分点：在拷贝开销
限制带宽时（swdge 每次拷贝 256 KiB 及以下，hwdge 64-128 KiB）差距约为 0；每次拷贝
1 MiB 及以上时差距为 2-3 个百分点，单核独占时达到其份额的 99-99.8%，整芯片达到 4.9 TB/s
的 96-98%（v1 在 2 MiB 时差距为 4.5-5.9 个百分点）。单核峰值约 611 GB/s，约等于其 16 个
DMA 引擎各约 38 GB/s 之和，因此单核 MBU 可能反映的是引擎上限而非 HBM。

kernel 不将传输数据与参考结果比对。两个修订版上 `store_static` fdim=512 的 profile
都只采集到约一半的包（`bytes_ok` 为 false；`report.py` 以 `*` 标记）。目前仅覆盖 LNC=1。

## Trainium3 PSUM 到 SBUF 拷贝带宽

`trn3_psum_bw` 在 LNC=1 下测量单个 Trainium3 NeuronCore 上 VectorE 与 ScalarE 读取 PSUM（TensorE
输出缓冲区）的速度。每个 kernel 是一次由 1,024 条 fp32 `[pdim, fdim]` `nisa.tensor_copy`（PSUM 到
SBUF）组成的突发：`copy_vector` 与 `copy_scalar` 只用一个引擎；`dual_alternate` 让两个引擎各执行
1,024 次拷贝，按程序顺序交替，读取不同的 PSUM tile 并写入各自的 SBUF 环（kernel 拆分 PSUM 读出的
常见做法）。fdim <= 512（每个 partition 一个 2 KiB fp32 PSUM bank）时，拷贝轮流使用 8 个 PSUM
tile。它使用 `trn3_dma_bw` 的 Docker 镜像和 `run.sh`：

```bash
NEURON_TARGET=trn3 OUT_DIR=ubenchX/trn3_psum_bw/runs/r0 CORE=0 KERNELS="copy_vector copy_scalar" \
  ubenchX/trn3_dma_bw/run.sh ubenchX/trn3_psum_bw/sweep.sh
NEURON_TARGET=trn3 OUT_DIR=ubenchX/trn3_psum_bw/runs/dual_r0 CORE=0 KERNELS=dual_alternate \
  FDIMS="64 128 256 512 1024 2048" ubenchX/trn3_dma_bw/run.sh ubenchX/trn3_psum_bw/sweep.sh

ubenchX/trn3_dma_bw/run.sh python ubenchX/trn3_psum_bw/parse.py ubenchX/trn3_psum_bw/runs/r*/* --csv psum.csv
ubenchX/trn3_dma_bw/run.sh python ubenchX/trn3_psum_bw/parse.py ubenchX/trn3_psum_bw/runs/dual_r0/* --csv dual.csv
ubenchX/trn3_dma_bw/run.sh python ubenchX/trn3_psum_bw/report.py psum.csv
```

`sweep.sh` 在 pdim=128 下运行 fdim 64 到 4096（整个 PSUM），并为 `copy_vector` 额外运行 pdim 32 和
64；`FDIMS` 可覆盖尺寸。带宽来自 profiler 的逐指令表：读取的 PSUM 字节数除以第一条拷贝开始到最后
一条拷贝结束的时间。对 `dual_alternate`，`parse.py` 还给出每个引擎在其自身忙碌窗口内的速率以及两
个窗口的重叠程度。`report.py` 拟合 每次拷贝 ns = 固定开销 + 字节数 / 流式速率。`plot.py` 和
`plot_partition.py` 画每次拷贝的时间，`plot_bpc.py` 画每周期字节数（单引擎 CSV 用 `--solo`），
`plot_overlay.py` 对比单引擎与双引擎（均需要 `matplotlib`）。

`results/trn3_48xlarge_rev2_*` 在一台 revision-v2 芯片的 trn3.48xlarge 的 NeuronCore 0 上测得；
`trn3_48xlarge_rev2_2026-11-06_report.md` 收录 `report.py` 的表格。两个引擎的流式速率都是
614 GB/s，即每个 1.2 GHz 周期 512 字节、每个 partition 每周期一个 fp32。每次拷贝的固定开销在
VectorE 上约 15 个周期（12.5 ns），在 ScalarE 上约 8 个周期（6.7 ns），因此 32 KiB 拷贝分别达到
497 和 545 GB/s，2 MiB 拷贝达到 612 和 613 GB/s。两次重复扫描相差不超过 0.1%。两个引擎同时拷贝
时，各自都保持单引擎速率，且忙碌窗口完全重叠，总带宽翻倍：每引擎 32 KiB 时 993 GB/s，每引擎
1 MiB 时 1,219 GB/s（每周期 1,016 字节）。只覆盖 LNC=1 和 fp32，且不校验拷贝的数据。

## Trainium3 exp 与激活函数吞吐量

`trn3_exp_bw` 在 LNC=1 下测量单个 Trainium3 NeuronCore 上超越函数的指令速率。每个 kernel 是一次
由 1,024 条指令组成的突发，每条指令读取一个 `[128, fdim]` SBUF tile（除特别说明外为 fp32）并写回
SBUF：

- Softmax exp，fp32 到 bf16，减去行最大值并在整个突发中累加行和：`exp_vector_softmax` 在 VectorE
  上使用 `nisa.exponential`，`exp_scalar_softmax` 在 ScalarE 上使用 `nisa.activation(nl.exp)`。
- 同样的 softmax exp，16 位输入，bf16 到 bf16：`exp_vector_softmax_bf16` 在 VectorE 上使用
  `nisa.exponential`。行最大值与行和仍为 fp32。
- ScalarE 上不带 scale、bias 或归约的 `nisa.activation`：`act_silu`（fp32 到 bf16），以及
  `act_sigmoid`、`act_rsqrt`、`act_softplus`、`act_sqrt`、`act_tanh`（fp32 到 fp32）。

```bash
NEURON_TARGET=trn3 OUT_DIR=ubenchX/trn3_exp_bw/runs/r0 CORE=0 \
  KERNELS="exp_vector_softmax exp_vector_softmax_bf16 exp_scalar_softmax act_silu act_sigmoid act_rsqrt act_softplus act_sqrt act_tanh" \
  ubenchX/trn3_dma_bw/run.sh ubenchX/trn3_exp_bw/sweep.sh

ubenchX/trn3_dma_bw/run.sh python ubenchX/trn3_exp_bw/parse.py ubenchX/trn3_exp_bw/runs/r0/* --csv act.csv
ubenchX/trn3_dma_bw/run.sh python ubenchX/trn3_exp_bw/report.py act.csv
```

`sweep.sh` 在 pdim=128 下运行 fdim 8 到 4096。`parse.py` 从 profiler 的逐指令表计算发射间隔
（II）：（最后一条开始时间 - 第一条开始时间）/ 1,023，并用 trace 中的引擎时钟（1.2 GHz）换算成
周期。`report.py` 在 fdim >= 256 上拟合 II 周期 = 发射开销 + fdim x 每元素周期数（可用
`--fit-min-fdim` 修改下限）。`plot.py` 画 softmax exp 的每指令周期数，加 `--epc` 则画整个引擎的每
周期元素数，CSV 中的每个 kernel 一条曲线，图例中标注其精度；`plot_act.py` 画激活函数的每周期元素
数（均需要 `matplotlib`）。

`results/trn3_48xlarge_rev2_2026-11-06_*` 在一台 revision-v2 芯片的 trn3.48xlarge 的 NeuronCore 0
上测得。ScalarE 每个 partition 每周期处理一个 fp32 元素（每周期 128 个，即文档中的 fp32 宽度），
六种激活函数的耗时完全相同，固定发射开销均为 22.1 个周期，softmax exp 为 23.1 个周期。VectorE 的
`nisa.exponential` 每个 partition 每周期处理两个元素（每周期 256 个，即文档中的 fp32 宽度），从
fdim 256 起发射开销为 25.1 个周期，但 fdim 低于 128 时每条指令至少约 56 个周期。fdim 为 8 / 64 /
512 / 4096 时，整个引擎每周期处理的元素数在 VectorE 上为 18 / 147 / 233 / 253，在 ScalarE 上为
33 / 94 / 123 / 127：fdim 低于 32 时 ScalarE 更快，fdim 较大时 VectorE 接近 ScalarE 的两倍。在
fp32 输入下未观察到架构指南所说的 VectorE exp 4 倍优势；bf16 输入的测量见下文 2026-11-24 的结果。
结果未与参考值比对。

`results/trn3n_48xlarge_rev2_2026-11-24_exp*` 在一台 revision-v2 芯片的 trn3n.48xlarge 的
NeuronCore 0 上重测了两个 softmax exp kernel，并新增 `exp_vector_softmax_bf16`。输入和输出均为
bf16 时，VectorE 的 `nisa.exponential` 每个 partition 每周期处理四个元素（每周期 512 个），是其
fp32 速率的两倍、ScalarE fp32 速率的四倍，从 fdim 512 起发射开销为 26.1 个周期；按输入字节计，
两种精度的流式速率同为 1,229 GB/s。每条指令约 56 个周期的下限一直持续到 fdim 128，是 fp32 输入时
的两倍，因此 fdim 不超过 64 时 bf16 输入并不比 fp32 输入快。fdim 为 8 / 64 / 512 / 4096 时，bf16
输入下整个引擎每周期处理 18 / 146 / 425 / 499 个元素。报告在 fdim >= 512 上拟合，三个 kernel 在该
范围内均为线性。两个 fp32 kernel 在每个 fdim 上与 2026-11-06 的结果相差不超过 1 个周期。未测量
ScalarE 的 bf16 输入。

## Trainium3 Tensor Engine 矩阵乘法时序

`trn3_matmul` 在 LNC=1 下测量单个 Trainium3 NeuronCore 上 TensorE 矩阵乘法的发射间隔（II）和延迟。
每个 kernel 是一次由 1,024 条 `nisa.nc_matmul` 指令组成的突发，bf16 x bf16 到 fp32 PSUM，K=128，
`[128, M]` 的 stationary tile 和 `[128, N]` 的 moving tile 均常驻 SBUF。第 i 次矩阵乘法覆盖写入
PSUM tile i % 8，因此各次矩阵乘法互不依赖：

- `mm_bf16_fixed` 复用同一个 stationary tile，编译器只生成一条 LoadStationary（LDWEIGHTS）和
  1,024 条 MATMUL。
- `ls_bf16_alt` 在两个 stationary tile 之间交替，因此每次矩阵乘法都有自己的 LDWEIGHTS。

```bash
NEURON_TARGET=trn3 OUT_DIR=ubenchX/trn3_matmul/runs/r0 CORE=0 \
  ubenchX/trn3_dma_bw/run.sh ubenchX/trn3_matmul/sweep.sh
NEURON_TARGET=trn3 OUT_DIR=ubenchX/trn3_matmul/runs/alt_r0 CORE=0 KERNEL=ls_bf16_alt \
  ubenchX/trn3_dma_bw/run.sh ubenchX/trn3_matmul/sweep.sh

ubenchX/trn3_dma_bw/run.sh python ubenchX/trn3_matmul/parse.py ubenchX/trn3_matmul/runs/r0/* --csv fixed.csv
ubenchX/trn3_dma_bw/run.sh python ubenchX/trn3_matmul/plot_fixed.py fixed.csv fixed_ii.png
```

`sweep.sh` 在 M=128 下运行 N 8 到 512，在 N=512 下运行 M 16 到 96，每个点执行 12 次，每 4 次
profile 一次。`parse.py` 读取 profiler 的逐指令表。TensorE 在突发开始时处于限速状态，trace 的
Throttle 表记录其恢复全速的时刻。对恢复之后开始的 MATMUL，II 为（最后一条开始时间 - 第一条开始
时间）/（条数 - 1），延迟为单条 MATMUL 从开始到结束时间的中位数，二者均用 trace 中的 TensorE
时钟（2.4 GHz）换算成周期。第一次 profile 的执行常常全程限速；绘图取三次 profile 执行的中位数。
`plot_fixed.py` 画 II 随 N 的变化（加 `--latency` 则画延迟）及其预测值，`plot_m.py` 画二者随 M 的
变化（均需要 `matplotlib`）。

`results/trn3_48xlarge_rev2_2026-11-10_*` 在一台 revision-v2 芯片的 trn3.48xlarge 的 NeuronCore 0
上测得。

- **单个 stationary tile 时的 II：** N 从 64 到 512 时为 N + 6 个周期，即 128x128 阵列每周期推进
  一列 moving 数据，外加 6 个周期的发射开销。N = 512 时达到 77.7 TFLOPS，为 2.4 GHz 下 bf16 峰值
  78.6 TFLOPS 的 99%。更小的矩阵乘法比 N + 6 更慢：N = 32 时为 42 个周期，N = 8 时为 30 个周期。
- **延迟：** 在每个 N 上都与 N + 393 相差不超过 6 个周期，从 N = 8 的 398 个周期到 N = 512 的
  905 个周期。393 个周期中，127 个用于填充各行，127 个用于排空各列，另有约 139 个周期无法用阵列
  解释。因此 N = 512 时同时在执行的矩阵乘法为 1.75 个，N = 8 时为 13 个。
- **M：** N = 512 时，M 从 16 到 128 的 II 均为 518 个周期，延迟均为 905 个周期。M 只决定阵列中
  有多少列装载权重，因此 TFLOPS 与 M 成正比。
- **每次矩阵乘法使用新的 stationary tile：** N 从 64 起，II 和延迟都不变，因为每次加载与前一次
  矩阵乘法重叠。N = 32 时 II 升至 46 个周期，N = 8 时升至 80 个周期。

只覆盖 bf16、K = 128 和互不依赖的矩阵乘法；一串相互依赖的矩阵乘法每次都要付出完整延迟。结果未与
参考值比对。

## Trainium3 整芯片矩阵乘法 TFLOP/s 与限速

`full_chip_matmul` 在 LNC=1 下让一颗 Trainium3 芯片的全部八个 NeuronCore 同时运行 TensorE 矩阵
乘法，把整芯片的 TFLOP/s 与单核单独运行时的八倍相比较，并在运行期间记录限速情况。它在用 AWS 的
`performance_profile_tool`（AWS 提供的二进制，不在本仓库中）设置的每个设备性能 profile 下各测
一遍。三个 kernel 沿用单核基准测试中固定 stationary 的 tile，M = 128，N = 512：

- `mm_bf16`：`nisa.nc_matmul`，bf16，K = 128（即 `trn3_matmul` 的 `mm_bf16_fixed`）。
- `mm_fp8dr`：`nisa.nc_matmul` 加 `perf_mode=double_row`，float8_e4m3，K = 256。
- `mm_mx`：MXFP8，`nisa.nc_matmul_mx`，float8_e4m3fn_x4 加 e8m0 scale，K = 512。

每个 kernel 是一个在设备上执行的循环（`nl.dynamic_range`），共 `REPS` 次迭代，每次迭代有 `NUM_MM`
条矩阵乘法，写入 PSUM tile i % 8，因此一次执行可以连续发射约 0.25 s 的矩阵乘法而不返回主机。
neuronx-cc 要求跨越循环边界仍存活的张量位于 HBM，所以每次迭代还要加载并转换操作数、排空 PSUM，
每次 1.8 ms 的迭代中约占 20 us。

```bash
docker build -t trn3-ubench ubenchX/trn3_docker
IMAGE=trn3-ubench PROFILE_TOOL=$HOME/performance_profile_tool \
  PROFILES="as_found 2.0ghz_tpb_1.2ghz_soc" RESTORE_PROFILE=2.4ghz_tpb_1.2ghz_soc \
  KERNELS="mm_bf16 mm_fp8dr mm_mx" N=512 NUM_MM=8192 REPS=128 PROBE_REPS=1024 SOLO_S=60 CHIP_S=180 \
  IDLE_S=10 NCORES=8 CHIP=0 PROBE_LAST_S=12 SAMPLE_S=0.5 NEURON_TARGET=trn3 \
  OUT_DIR=ubenchX/full_chip_matmul/runs/r1 ubenchX/full_chip_matmul/sweep.sh

ubenchX/trn3_dma_bw/run.sh python ubenchX/full_chip_matmul/parse.py ubenchX/full_chip_matmul/runs/r1 --prefix chip
ubenchX/trn3_dma_bw/run.sh python ubenchX/full_chip_matmul/report.py --prefix chip --bin-s 15 > chip_report.md
python ubenchX/full_chip_matmul/plot.py --prefix chip --out chip_timeline.png \
  --profiles as_found 2.0ghz_tpb_1.2ghz_soc
```

`sweep.sh` 在主机上运行，需要能免密 `sudo` 执行 profile 工具。对 `PROFILES` 中的每个名字，它先设置
该 profile（`as_found` 表示保持节点现状），用 `te_clock.py` 记录每个设备的 TensorE 时钟，再在
`IMAGE` 容器中为每个 kernel 运行 `fullchip.sh`；`OUT_DIR` 相对于仓库根目录。只要它改过 profile，
退出时（包括失败或被中断时）就会设置 `RESTORE_PROFILE`。该工具会对节点上所有 Neuron 设备设置
profile；在 Trainium3 上，设备固件不支持它的 `--get-profile` 和 `--get-supported-profiles`，因此
无法读出节点起始的 profile，需要由调用方指定要恢复到的 profile。

`fullchip.sh` 先编译 kernel，然后在 `sample_throttle.py` 每 `SAMPLE_S` 秒采样一次 sysfs 的同时，
先让一个核单独运行 `SOLO_S` 秒（`solo`），再让芯片 `CHIP` 的 `NCORES` 个核一起运行 `CHIP_S` 秒
（`chip`），之前、之间和之后各空闲 `IDLE_S` 秒。每个核对应一个 `run_point.py` 进程，通过 libnrt
加载 NEFF（`NEURON_RT_VISIBLE_CORES`、`NEURON_LOGICAL_NC_CONFIG=1`）；所有进程在同一墙上时钟时刻
开始，连续调用 `nrt_execute` 并为每次调用计时。

TFLOP/s 按每条矩阵乘法指令 2 K M N 个 FLOP 计算，与单核基准测试一致。单核 TFLOP/s 为
`REPS` x `NUM_MM` x 2 K M N x 执行次数除以在 `nrt_execute` 内花费的时间，因此包含每次迭代的操作数
加载和 PSUM 排空。整芯片 TFLOP/s 是各核之和，scaling 为整芯片 TFLOP/s 除以 `NCORES` 乘以 solo
TFLOP/s。利用率为 TFLOP/s 除以该 profile 时钟下的 TensorE 峰值：每个核每个时钟周期 2 K M 个 FLOP，
2.4 GHz 下 bf16 为 78.6 TFLOP/s（即 `trn3_matmul` 使用的峰值），fp8 double_row 为 157.3，MXFP8 为
314.6，整芯片为其八倍。`report.py` 的时钟取自 TensorE 计数器。

限速通过三种方式记录：

- **sysfs 限速统计。** `sample_throttle.py` 读取芯片上每个核的
  `/sys/devices/virtual/neuron_device/neuron<chip>/neuron_core<c>/stats/hardware/ham_throttle/throttle`。
  `mean`、`max` 和 `min` 是以 255 为满的限速水平，255 表示全速，空闲的核读数为 128（50%）。驱动
  大约每 0.9 s 刷新一次这条记录。
- **TensorE 计数器。** 同一次采样还读取 `tensor_engine/pe_cntrs`。`mm_active` / 2^14 + `idle`
  按 TensorE 时钟递增，`mm_active` / 2^14 统计正在推进 moving 列的周期数，由此得到不依赖主机计时的
  TFLOP/s。这些单位是从测量结果推断的，没有文档说明。
- **Profiler。** 每次运行结束前 `PROBE_LAST_S` 秒，第一个核执行一次带 profile 的 kernel，迭代次数
  为 `PROBE_REPS`。profiler 的 Throttle 表给出这次探测期间的限速水平及其来源（`activity_0`、
  `activity_1`、`power_3`、`temp_0`），Instruction 表用于确认 opcode 和 dtype。profiler 就绪期间该核
  会空闲约一秒，其他核在此期间加速，而且被 profile 过的核的 sysfs 统计不再更新，因此每次运行只做
  一次探测，TFLOP/s 只统计探测之前的时间。

`results/trn3n_48xlarge_rev2_2026-11-24_full_chip_*` 在一台 revision-v2 芯片的 trn3n.48xlarge 的
芯片 0 上测得：每个 kernel 单核运行 60 s、八核运行 180 s，速率分别取探测之前的 48 s 和 168 s。
`_report.md` 是 `report.py` 生成的表格，`_timeline.png` 是八核运行随时间的变化，
`_profile_tool.txt` 是 profile 工具的输出和 TensorE 时钟，`_throttle_raw.jsonl.gz` 是全部 sysfs 采样。

| profile | 模式（kernel） | solo TFLOP/s（利用率） | 整芯片 TFLOP/s（利用率） | 整芯片 / (8 x solo) | 八核时 sysfs 限速均值 | profiler 给出的来源与水平 |
|---|---|---|---|---|---|---|
| 起始状态（TensorE 2.4 GHz） | bf16（`mm_bf16`） | 76.86（97.7%） | 614.0（97.6%） | 0.999 | 99.1% | 无 |
| 起始状态 | fp8 double_row（`mm_fp8dr`） | 153.72（97.7%） | 1,127.5（89.6%） | 0.917 | 90.9% | `power_3`，91.7% |
| 起始状态 | MXFP8（`mm_mx`） | 297.08（94.4%） | 1,727.8（68.7%） | 0.727 | 73.6% | `power_3`，73.9% |
| `2.0ghz_tpb_1.2ghz_soc` | bf16（`mm_bf16`） | 64.06（97.7%） | 512.4（97.7%） | 1.000 | 99.2% | 无 |
| `2.0ghz_tpb_1.2ghz_soc` | fp8 double_row（`mm_fp8dr`） | 128.15（97.8%） | 1,025.1（97.8%） | 1.000 | 99.2% | 无 |
| `2.0ghz_tpb_1.2ghz_soc` | MXFP8（`mm_mx`） | 248.35（94.7%） | 1,766.9（84.3%） | 0.889 | 90.7% | `power_3`，91.2% |

- **起始状态下，八核同时运行 fp8 double_row 或 MXFP8 会被功耗限速。** 单核单独运行时从不被限速：其
  sysfs 均值保持在 99.2%，其中已包含循环迭代之间的 50% 水平。八核时整芯片只达到单核速率八倍的
  91.7%（fp8 double_row）和 72.7%（MXFP8），每个核减速的幅度相同（各核速率相差不超过 0.1%），sysfs
  均值和 profiler 给出的水平按相同比例下降，profiler 标出的来源是 `power_3`。bf16 除了在 10 到
  40 s 之间有一次 1.3% 的下探外不被限速。
- **限速作用于整颗芯片，并且有延迟。** 八个核启动后 1 到 2 s 限速生效，15 到 30 s 时最深（1,105 和
  1,707 TFLOP/s），约 90 s 后稳定（1,133 和 1,727）。某个核为探测而暂停时，其余七个核会加速约半秒，
  fp8 double_row 回到单核速率；探测本身开始时不受限速，约 0.5 s 内重新被限速，因此远短于一秒的
  探测会低估限速。
- **AWS 建议的 profile 无法设置。** `--profile 2.0ghz_tpb_1.1ghz_soc` 两次都在全部 16 个设备上失败，
  报错 `Failed to set performance profile on device N: No such file or directory`（内核日志：
  `fw_io_execute_request_new: seq: 86, cmd: 3 failed 1`），时钟仍为 2.4 GHz。
  `2.0ghz_tpb_1.2ghz_soc` 可以设置，于是改测它；之后计数器显示全部 16 个设备均为 2.000 GHz。
- **2.0 GHz 下 fp8 double_row 不再被限速，但更慢。** 单核速率按 2.0 / 2.4 下降。bf16 和 fp8
  double_row 扩展到八核的比例为 1.000，整芯片总量为 512 和 1,025 TFLOP/s，低于起始状态下带限速
  测得的 614 和 1,128。MXFP8 仍被功耗限速到 88.9%，整芯片总量上升 2.3% 至 1,767 TFLOP/s：对 MXFP8 而言，
  两种 profile 下整芯片都只能给出约 1,730 到 1,770 TFLOP/s。
- **恢复。** 节点被设回 `2.4ghz_tpb_1.2ghz_soc`，即 AWS 所说的默认 profile；之后全部 16 个设备读数
  均为 2.400 GHz，重跑 MXFP8（报告中的 `restored_2.4ghz_tpb_1.2ghz_soc`，以及第一次设置之后的
  `set_2.4ghz_tpb_1.2ghz_soc`）与起始状态的结果相比，solo 相差不超过 0.1%，八核运行每 15 s 的速率
  相差不超过 0.5%。驱动的 `/sys/class/neuron_device/current_perf_profile` 在这次设置后读数为 6，
  在 `2.0ghz_tpb_1.2ghz_soc` 下为 7；从未设置过 profile 的节点读数为 0，即驱动在设备复位后赋的值。

`results/trn3n_48xlarge_rev2_2026-11-24_full_chip_node3_*` 在另一台 trn3n.48xlarge 的芯片 0 上
重复了起始状态下的测量，该节点从未设置过 profile
（`current_perf_profile` 为 0，2.400 GHz），单核运行 30 s、八核运行 102 s（速率分别取 18 s 和
90 s）。这颗芯片在每种模式下都被限速得更多：

| 模式（kernel） | solo TFLOP/s（利用率） | 整芯片 TFLOP/s（利用率） | 整芯片 / (8 x solo) | 第一个节点同样 90 s |
|---|---|---|---|---|
| bf16（`mm_bf16`） | 76.85（97.7%） | 563.9（89.6%） | 0.917 | 613 |
| fp8 double_row（`mm_fp8dr`） | 153.71（97.7%） | 1,032.7（82.1%） | 0.840 | 1,123 |
| MXFP8（`mm_mx`） | 267.68（85.1%） | 1,592.0（63.3%） | 0.743 | 1,728 |

它的八核总量在三种模式下（包括 bf16）都比第一个节点低 8%，主要来源是 `power_3`，`temp_0` 占每次
探测的 8% 到 18%。在这颗芯片上，单个 MXFP8 核本身就会被限速：起始为 297 TFLOP/s，1.5 s 后为 270，
15 s 后为 265，profiler 标出的来源是 `temp_0`。两个节点的测量时间和运行时长都不同，因此这只说明限速在不同
芯片或节点之间存在差异，并不能说明原因；上面的 profile 对比全部在同一颗芯片上完成。

芯片 0 上运行时的 LNC=1 核 0 到 7 对应 sysfs 的 `neuron_core` 4、5、6、7、2、3、0、1（逐核单独
运行，根据其 `pe_cntrs` 判断）。sysfs 记录在每次采样中都打印 `Source attribution unavailable`，
因此限速来源只来自 profiler。profile 对比只在一个节点的一颗芯片上进行，N = 512，stationary tile
固定；没有读取功耗和温度（`stats/hardware/temperature` 返回 `Operation not supported`）。结果未与
参考值比对。
