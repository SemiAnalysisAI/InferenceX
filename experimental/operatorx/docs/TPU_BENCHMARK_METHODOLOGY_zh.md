<div align="center">

[English](./TPU_BENCHMARK_METHODOLOGY.md) | **中文**

</div>

# TPU 算子基准测试方法

## 范围与后端策略

OperatorX 应调用实现目标算子的最高层生产 API。如果已有可用的库算子，不应仅为提高基准测试
结果而用手写的底层内核替代它。对于本次普通 Linear/GEMM 试点，JAX 后端调用编译后的
`jnp.dot`。

试点使用两个 BF16 形状：

| 名称 | M | N | K | 性能区间 |
|---|---:|---:|---:|---|
| `tpu-gemm-skinny` | 16 | 8192 | 8192 | 类似解码，内存与调度开销明显 |
| `tpu-gemm-square` | 8192 | 8192 | 8192 | 计算受限 |

这些形状记录在 `testlists/tpu_gemm_pilot.json` 中。

## 计时口径

不要用 `min()` 混合不同含义的测量值。应分别报告以下口径：

1. **`device_execution_us`（标准校准值）：** 完成编译和预热后，采集 XProf
   `TRACE_ONLY_XLA` profile。使用唯一的 `jax.profiler.StepTraceAnnotation`，
   测量 TPU 侧 XLA module 的关键路径。一个复合算子可能被降级为相互重叠的多个 HLO
   事件，因此通常不能简单累加事件时长。
2. **`queued_throughput_us`（快速扫描代理值）：** 连续提交一批相同的已编译算子，
   保留所有输出，对整个输出集合执行阻塞等待，再用总完成时间除以批大小。采集多个批次并报告
   p10/p50/p90，同时报告每个算子的入队时间，以便识别主机供给瓶颈。只有在对应硬件、后端和
   算子区间上通过 XProf 校准后，才能把该代理值作为标准 `latency_us`。
3. **`dispatch_to_ready_us`（应用可见延迟）：** 测量一次已编译提交到
   `block_until_ready()` 返回的时间。它包含 Python/PJRT 调度与同步开销，不能直接与
   仅计算设备时间的成本模型比较。

所有编译和预热必须位于计时区间之外，输入在计时前必须已经位于设备上。结果应记录样本数量、
JAX/jaxlib/libtpu 版本、设备拓扑和数据驻留模式。

## 数据驻留

每个样本前重新创建 JAX array 并不能证明片上缓存处于冷状态。必须使用以下标签之一：

- `steady_state`：重复使用稳定的 HBM buffer；
- `cold_hbm`：使用大于片上 VMEM 的轮换 buffer 工作集，并尽可能通过 profiler
  counter 验证；
- `residency_unknown`：无法确认 HBM 驻留时使用。

试点脚本使用 `steady_state`。冷 HBM 验证属于后续工作。

## Ironwood 计量口径

一个 Ironwood 物理芯片包含两个独立暴露的 chiplet，JAX 将每个 chiplet 显示为一个设备。
Google 公布的 2307 BF16 TFLOP/s 和 7380 GB/s HBM 带宽均为每个物理芯片的数据，因此
单设备 GEMM 应使用 1153.5 BF16 TFLOP/s 作为峰值归一化基准。每个 chiplet 拥有独立的
96 GiB 内存空间。由于公开带宽是芯片级数据，在计算带宽利用率之前还需再次核实 chiplet
带宽归一化方式。

不能把单 chiplet 的 JAX 结果直接与 SimulatorX 的物理芯片设备模型比较。应在
SimulatorX 中建模单个 chiplet，或者在真实硬件上把 GEMM 分片到两个 chiplet。

## Ironwood 试点数据

在 `tpu7x-8` 主机上使用 JAX 0.11.0 得到：

| 形状 | 当前 runner | XProf 设备时间 | 修正后的排队代理值 | 代理误差 |
|---|---:|---:|---:|---:|
| `16×8192×8192` | 51.2 µs | 44.35 µs | 46.67 µs | +5.2% |
| `8192×8192×8192` | 1382.2 µs | 1244.75 µs | 1246.38 µs | +0.13% |

对于细长形状，同步的稳态调用约为 155 µs，比 XProf 设备时间高约 250%。对于方形形状，
XProf 测得约 883 TFLOP/s，相当于单 chiplet BF16 峰值的 76.6%。

这些数据支持在快速试点中使用排队代理值，但还不足以证明它可以替代所有 TPU 算子的标准设备
测量。

## 二十形状校准扫描

扩展后的 `testlists/tpu_gemm_sweep.json` 包含 20 个 BF16 GEMM：隐藏维度
4096 和 8192 下的解码与预填充 M 值、从 1024 到 8192 的方形 GEMM，以及
8192↔28672 的前馈投影。每一行都通过了正确性、单设备放置、StableHLO、优化后 HLO
和 XProf 事件数完全匹配等门槛。

仓库中的紧凑数据集 `data/tpu7x_gemm_sweep_20260725.json` 对每个形状使用 21 个
module 级 XProf 样本。整个扫描的 `(p90-p10)/p50` 中位数为 0.91%，最大值为
3.58%。相比之下，排队代理值对 `16×4096×4096` 高估 303%，对
`256×4096×4096` 高估 222%，对 `2048³` 高估 133%，对 `1024³` 高估
826%；只有最大的一批 kernel 才收敛到 1% 以内。因此，跨形状成本模型校准必须以
**XProf module 时长为标准值，而不能使用排队吞吐代理值**。

## 验证门槛

试点可以生成一套自包含的验证产物。每个形状只有通过以下全部检查才算有效：

1. **语义：** 抽样九个输出元素，用对应 BF16 输入行、列执行独立的 NumPy FP32
   点积并进行比较。
2. **设备放置：** 断言 A、B 和输出均只位于指定的单个 JAX 设备上，同时记录已编译程序的
   输入和输出 sharding。
3. **逻辑程序：** 保存 StableHLO，并要求存在输入、输出维度正确的
   `stablehlo.dot_general`。
4. **编译后程序：** 保存实际已编译可执行文件的优化 HLO，并要求其为 `jit_dot`
   module。记录 TPU emitter、window 配置、scoped memory 分配和跨程序预取决策。
5. **设备计时：** 每个形状单独采集 profile，使用 `TRACE_ONLY_XLA`、只 profile
   一个芯片、仅保留命名的主机 annotation，并关闭 Python tracing。解析 module 级
   `jit_dot(<fingerprint>)` 事件，并要求事件数与指定样本数完全一致。

解析器把 module 事件作为可执行程序的关键路径。子 HLO 事件仅用于诊断，绝不累加其时长，
因为这些事件可能重叠。XProf 的 `raw_bytes_accessed` 是编译器/profiler 的访问量计数；
即使它与逻辑 tensor 字节数一致，也不能证明所有访问都到达了 HBM。

在验证主机上，两个形状都通过了全部门槛，使用的设备是
`TPU_0(process=0,(0,0,0,0))`，即 `TPU7x` chiplet 0：

| 形状 | 抽样最大相对误差 | XProf p50 | 排队 p50 | 代理误差 |
|---|---:|---:|---:|---:|
| `16×8192×8192` | 0.289% | 43.34 µs | 46.99 µs | +8.42% |
| `8192×8192×8192` | 0.279% | 1243.34 µs | 1246.82 µs | +0.28% |

StableHLO 保留了指定的 BF16 dot 和 `[1] × [0]` contracting dimensions。优化后的
TPU HLO 把两个 GEMM 都表示为使用 `EmitAllBatchInSublanes` 的 convolution fusion，
但选择了不同的 window 配置。细长形状使用跨程序预取，方形形状则没有。XProf 报告的逻辑
工作量符合预期：

| 形状 | Model FLOPs | Raw bytes accessed |
|---|---:|---:|
| `16×8192×8192` | 2,147,483,648 | 134,742,016 |
| `8192×8192×8192` | 1,099,511,627,776 | 402,653,184 |

## 复现试点

在 InferenceX worktree 的 OperatorX 目录中执行：

```bash
cd ~/InferenceX/.worktrees/tpu-exploration/experimental/operatorx

PYTHONPATH=.. \
~/SimulatorX/.venv/bin/python scripts/tpu_gemm_pilot.py \
  --json-out /tmp/operatorx-tpu-gemm-pilot.json
```

查看机器可读结果：

```bash
jq '.rows[] | {
  name,
  shape,
  dispatch_to_ready_us,
  queued_throughput_us,
  enqueue_us_per_op,
  achieved_tflops,
  pct_of_chiplet_peak
}' /tmp/operatorx-tpu-gemm-pilot.json
```

运行完整的正确性、设备放置、HLO 和 XProf 验证：

```bash
cd ~/InferenceX/.worktrees/tpu-exploration/experimental/operatorx

VALIDATION_DIR="$(mktemp -d /tmp/operatorx-tpu-validation.XXXXXX)"

PYTHONPATH=.. \
python scripts/tpu_gemm_pilot.py \
  --artifacts-dir "$VALIDATION_DIR" \
  --profile \
  --profile-iters 5 \
  --json-out "$VALIDATION_DIR/results.json"

echo "$VALIDATION_DIR"
```

查看验证摘要：

```bash
jq '.rows[] | {
  name,
  device,
  correctness,
  placement,
  compiler_algorithm: .hlo.compiler_algorithm,
  device_execution_us,
  queued_proxy_error_pct,
  device_achieved_tflops,
  device_pct_of_chiplet_peak
}' "$VALIDATION_DIR/results.json"
```

独立重新解析任一已保存的 Perfetto trace：

```bash
PYTHONPATH=.. \
python -m operatorx.runners.tpu.xprof \
  "$VALIDATION_DIR/profiles/tpu-gemm-skinny" \
  --module-name jit_dot \
  --expected-samples 5 \
  --annotation-name operatorx_tpu-gemm-skinny
```

运行已校准的 20 形状扫描，并且每个 profile 只保留紧凑的 Perfetto trace：

```bash
cd ~/InferenceX/.worktrees/tpu-exploration/experimental/operatorx

SWEEP_DIR="$(mktemp -d /tmp/operatorx-tpu-gemm-sweep.XXXXXX)"

PYTHONPATH=.. \
~/SimulatorX/.venv/bin/python scripts/tpu_gemm_pilot.py \
  --testlist testlists/tpu_gemm_sweep.json \
  --sync-iters 5 \
  --batches 3 \
  --batch-iters 16 \
  --artifacts-dir "$SWEEP_DIR" \
  --profile \
  --profile-iters 21 \
  --profile-retention perfetto \
  --json-out "$SWEEP_DIR/results.json"

PYTHONPATH=.. \
~/SimulatorX/.venv/bin/python scripts/compact_tpu_gemm_results.py \
  "$SWEEP_DIR/results.json" \
  /tmp/tpu7x-gemm-sweep-compact.json
```

压缩脚本保留验证结果、聚合计时、设备和编译器方案证据，同时删除主机名、命令行路径、原始
XProf 样本、sharding 表示和产物路径。查看标准测量值：

```bash
jq '.rows[] | {
  name,
  shape,
  xprof_p50_us: .device_execution_us.p50,
  xprof_samples: .device_execution_us.samples,
  queued_proxy_error_pct,
  compiler_algorithm: .hlo.compiler_algorithm
}' /tmp/tpu7x-gemm-sweep-compact.json
```

为了在多个形状之间摊销 profiler 启动和转换开销，可以在同一个 XProf session 中采集
整个 testlist，并在所有形状结束后只解析一次 Perfetto trace：

```bash
cd ~/InferenceX/.worktrees/tpu-exploration/experimental/operatorx

BATCH_DIR="$(mktemp -d /tmp/operatorx-tpu-gemm-batch.XXXXXX)"

PYTHONPATH=.. \
~/SimulatorX/.venv/bin/python scripts/tpu_gemm_batched_xprof.py \
  --testlist testlists/tpu_gemm_sweep.json \
  --profile-iters 21 \
  --artifacts-dir "$BATCH_DIR" \
  --profile-retention perfetto \
  --json-out "$BATCH_DIR/results.json"
```

脚本会在启动 profiler 前完成每个形状的编译、预热和验证。在单个 trace 内，它按顺序提交
带有唯一 annotation 的形状区块，并同步每一次采样提交。逐次同步使内存占用不随
`profile_iters` 增长；否则，即使单次调用能够放入 HBM，保留 21 个数 GiB 的输出也可能
导致 OOM。主机 annotation 与 TPU 事件
使用不同的时间线，因此解析器不会按 annotation 时间窗口分组。它会要求总事件数完全匹配，
再按照声明的形状顺序分配单调递增的 XLA `run_id` 组。每一行都会记录所分配的 run ID、
module fingerprint 和时间范围重叠的子 HLO 诊断信息。

所有已准备的输入和编译上下文都会一直驻留到单次采集结束。如果某个 testlist 接近 chiplet
的 HBM 容量，应把它拆成多个 testlist，并分别执行一个 batch；每个 batch 仍然可以让其中
所有形状共同摊销一次 XProf 启动成本。

### 构建来自真实工作负载的 BF16 语料库

`testlists/gemm.json` 是来自 InferenceX 的 GEMM 历史并集。语料库构建工具会把每个有效且
唯一的 `(M,N,K)` 转换为 BF16，应用当前 kernel-local chunked-prefill 上限
`M <= 32768`，为每一行生成稳定且唯一的名称，并用 best-fit 算法把形状打包成受内存上限
约束的 XProf manifest：

```bash
cd ~/InferenceX/.worktrees/tpu-exploration/experimental/operatorx

CORPUS_DIR=/tmp/operatorx-all-current

~/SimulatorX/.venv/bin/python scripts/build_tpu_gemm_corpus.py \
  --source testlists/gemm.json \
  --output-dir "$CORPUS_DIR" \
  --max-m 32768 \
  --batch-memory-gib 12
```

在 2026-07-25 的代码版本上，该命令生成 3,117 个可测试形状和 37 个 batch。
`excluded.json` 会明确记录 2,172 个超过 kernel-local M 上限的唯一形状，以及 24 个维度
非正的无效源条目；这些条目不会被静默地当成硬件测量结果。等 InferenceX 的 matrix 配置
与 runner 元数据重新一致后，应优先从当前 master matrix 直接重新生成。

逐个执行 batch，并为每个 batch 使用独立的 artifact 目录：

```bash
mkdir -p "$CORPUS_DIR/results" "$CORPUS_DIR/artifacts"

for MANIFEST in "$CORPUS_DIR"/chunks/chunk-*.json; do
  CHUNK="$(basename "$MANIFEST" .json)"
  PYTHONPATH=.. ~/SimulatorX/.venv/bin/python \
    scripts/tpu_gemm_batched_xprof.py \
      --testlist "$MANIFEST" \
      --profile-iters 21 \
      --artifacts-dir "$CORPUS_DIR/artifacts/$CHUNK" \
      --profile-retention perfetto \
      --json-out "$CORPUS_DIR/results/$CHUNK.json"
done
```

3,117 个形状的运行已完成：共 37 个 batch，每个形状包含 21 个 module 样本。
成功 batch 的累计耗时为 13,798.4 秒（3 小时 49 分 58 秒）。对于 643 个非规则
形状，优化后的 TPU executable 会填充维度，因此 XProf 的 `model_flops` 和
`raw_bytes_accessed` 与逻辑工作量不同。这两个字段是编译器物理工作量的诊断信息，
不能作为通用的逻辑工作量验证门槛。batch 结果会同时记录逻辑值、编译器值和相等标志。

通过当前标准 OperatorX 入口运行同一 testlist：

```bash
cd ~/InferenceX/.worktrees/tpu-exploration/experimental/operatorx

PYTHONPATH=.. \
WORLD_SIZE=1 \
OPERATORX_CLUSTER=v7x_4x \
OPERATORX_BACKENDS=jax \
~/SimulatorX/.venv/bin/python -m operatorx \
  --platform tpu \
  --testlists tpu_gemm_pilot
```

标准入口会把运行结果写入 `results/tpu/v7x_4x/`。当前 `latency_us` 仍使用旧的
`min(single_shot, pipelined)` 方法；在 runner 重构完成前，应使用试点脚本中分离后的
指标来决定基准测试方法。

官方参考资料：

- [JAX 基准测试指南](https://docs.jax.dev/en/latest/benchmarking.html)
- [JAX 异步调度](https://docs.jax.dev/en/latest/async_dispatch.html)
- [使用 XProf 分析 JAX 计算](https://openxla.org/xprof/jax_profiling)
- [TPU7x 架构](https://docs.cloud.google.com/tpu/docs/tpu7x)
