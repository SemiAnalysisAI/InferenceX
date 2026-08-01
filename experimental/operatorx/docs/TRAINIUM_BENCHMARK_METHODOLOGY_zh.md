<div align="center">

[English](./TRAINIUM_BENCHMARK_METHODOLOGY.md) | **中文**

</div>

# Trainium3 NKI GEMM 基准测试方法

## 测量口径

校准使用 NKI 原生 standalone profiling 路径得到的设备执行时间：

1. 面向 `trn3`、以二路 grid/LNC 编译 `@nki.jit` BF16 GEMM；
2. 通过 NKI Spike runtime 加载 NEFF；
3. 使用稳定复用的设备 buffer 执行十次不采集 profile 的预热；
4. 采集 21 份独立 NTFF 设备 profile；
5. 使用 `neuron-profile view --output-format summary-json` 解析每份 profile。

保留的 `device_execution_us` 分布来自 profile 的 `total_time`，不包含框架调度和
编译时间。NKI `benchmark` 延迟还包含主机与设备之间的数据传输，因此不作为校准
指标。该流程遵循 AWS [`nki.profile`](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.profile.html)
所描述的 NKI 原生 profiling 模型；针对已安装的 NKI 0.3，直接使用 standalone
编译器与 Spike runtime，使现有 OperatorX `@nki.jit` 内核无需 PyTorch/XLA 即可
完成 profiling。

## Shape 集合与 NKI padding

测试列表严格复用 TPU 探索的 20 个 BF16 shape，即
`testlists/tpu_gemm_sweep.json`，覆盖解码/预填充 M、方形 GEMM，以及
8192↔28672 前馈投影。

当前 OperatorX NKI BF16 GEMM 会按以下约束 padding：

- M：2048 的倍数；
- N：LNC=2 下为 2048 的倍数；
- K：1024 的倍数。

20 个请求对应十个不同的 padding 后可执行程序。映射到同一个程序的请求会复用
该程序的 21 份原生设备样本和 NEFF hash。每行同时记录请求与执行 shape、逻辑与
执行 FLOPs/字节数，以及 padding FLOPs 比例。SimulatorX 预测请求的逻辑 shape，
因此 NKI padding 会作为实现开销保留在误差中。

后续的 `testlists/trainium_gemm_tile_boundaries.json` 集合包含 20 个请求，覆盖
2048 行、2048 列和 1024 reduction 维度的 padding 边界，共对应 11 个可执行
程序，并包含八个完全对齐的对照 shape。边界前一位、边界值及边界后一位的请求会
有意形成共享可执行程序的延迟平台；完全对齐的对照组用于区分原生组件行为与
padding 影响。

用于模型开发的后续集合 `testlists/trainium_gemm_exact_holdout.json` 包含 24 个
完全对齐的可执行程序：M/N 取 {2048, 4096, 6144, 8192}，K 取
{1024, 2048, 3072, 4096}，每个 K 恰好出现六次。其中 16 行标记为 `train`；
另外八行完整留出四组 M/N 组合，每组包含两个 K。冻结后的数据划分可确保输出
wave 尾项不会在拟合时看到用于评估的 M/N 组合。

纵向验证集合 `testlists/trainium_gemm_longitudinal_validation.json` 包含 103 个
请求：从 OperatorX `testlists/gemm.json` 中可复现筛选出的 75 个 BF16 GEMM、
24 个完全对齐的外推 shape，以及四个校准锚点。它们对应 38 个可执行程序，其中
29 个未出现在校准网格中。同一个冻结集合会以带 UTC 时间戳的独立快照重复运行，
且绝不会加入 phase 模型的校准数据。

## 验收门槛

只有满足下列条件的可执行程序才会被接受：

1. **语义：**确定性的 BF16 全一输入，在抽样位置得到预期输出 K，并覆盖 padding
   后的最后一行与最后一列。
2. **放置：**NKI 目标为 `trn3`；编译器元数据记录 LNC=2；启动 grid 使用两个物理
   NeuronCore-v4。
3. **编译程序：**保留 NEFF 与 compiler-info 的 SHA-256；每份 profile 都包含
   matmul 指令。
4. **工作量身份：**`hardware_flops` 必须严格等于
   `matmul_instruction_count × 16,777,216`。profiler 的 trace 窗口最多可遗漏
   静态调度 matmul 指令的 0.1%；对于小型可执行程序，容差下限为两条指令。
   编译 shape 与输出正确性仍是相互独立的强制门槛；该容差只适用于彼此严格一致的
   profile 推导计数器。
5. **计时：**精确保留 21 个正数 `total_time` 样本，并给出
   p10/p50/p90/min/max。
6. **流量证据：**每份 profile 都包含正数 HBM 读写计数。
7. **驻留状态：**输入输出只分配一次，并在预热和 profiling 间复用，因此标记为
   `steady_state`。

旧版 `nki.profile` 仅接收 shape 描述时不会使用真实输入值。本工具改为通过 Spike
用真实 NumPy BF16 buffer 执行 NKI 0.3 编译后的 NEFF，在保留相同原生 NTFF 计时
机制的同时完成独立正确性检查。

## 精简证据与清理

`scripts/trainium_gemm_sweep.py` 一次执行完整 shape 集合。NEFF、NTFF、MLIR/BIR、
编译输出及日志会暂存在新建目录中。所有 trace 完成解析且精简 JSON 写入后，默认
删除整个目录。`--keep-artifacts` 仅用于显式调试，正式入库运行不使用该参数。

精简数据集保留：

- 每个不同可执行程序的 21 个设备时间及选定设备计数器；
- 请求与执行 shape；
- 验证结果与驻留标签；
- NEFF/compiler-info hash 与 NEFF 大小；
- Neuron 设备清单及 NKI/编译器/runtime/profiler 版本；
- UTC 开始/完成时间戳及调用方指定的纵向运行标签；
- 完整的测量与测试列表契约。

正式扫描结束后，不会将原始 NTFF、NEFF、编译 dump 或运行日志提交到仓库。

## 复现

在该 Trainium3 分区的 `experimental/operatorx` 目录执行：

```bash
export PATH=/opt/aws/neuron/bin:/opt/aws_neuronx_venv_pytorch_2_9/bin:$PATH
export PYTHONPATH="$PWD/.."

python scripts/trainium_gemm_sweep.py \
  --json-out data/trn3_lnc2_gemm_sweep_20260731.json \
  --samples 21 \
  --warmup 10
```

宽覆盖集合由入库的 OperatorX 源列表生成，并作为相互独立的快照运行：

```bash
python scripts/build_trainium_longitudinal_corpus.py \
  --output testlists/trainium_gemm_longitudinal_validation.json

python scripts/trainium_gemm_sweep.py \
  --testlist testlists/trainium_gemm_longitudinal_validation.json \
  --json-out data/trn3_lnc2_gemm_longitudinal_20260801_snapshot_a.json \
  --run-label 20260801-snapshot-a \
  --samples 21 \
  --warmup 10
```

命令会逐个打印不同的可执行程序，写入精简结果，并确认临时 profile 根目录已删除。

边界后续扫描采用相同的测量契约：

```bash
python scripts/trainium_gemm_sweep.py \
  --testlist testlists/trainium_gemm_tile_boundaries.json \
  --json-out data/trn3_lnc2_gemm_tile_boundaries_20260731.json \
  --samples 21 \
  --warmup 10
```

完全对齐的训练/留出网格同样采用一次性测量契约：

```bash
python scripts/trainium_gemm_sweep.py \
  --testlist testlists/trainium_gemm_exact_holdout.json \
  --json-out data/trn3_lnc2_gemm_exact_holdout_20260731.json \
  --samples 21 \
  --warmup 10
```
