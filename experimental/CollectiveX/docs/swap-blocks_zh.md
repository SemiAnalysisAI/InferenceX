# vLLM 块复制基准测试

[English](./swap-blocks.md) | **中文**

`bench/run_swap_blocks.py` 在单个 CUDA 或 ROCm GPU 上测量
`from vllm._custom_ops import swap_blocks`，需要安装兼容的 vLLM。
可在该环境中直接使用 Python，或选择下方的独立 GPU Action。
无需 `torchrun`，也不会执行 EP 工作负载。
结果记录已安装的 vLLM 版本，同时支持旧版三参数接口和显式传入
`block_size_in_bytes` 的接口。

```bash
python3 experimental/CollectiveX/bench/run_swap_blocks.py \
  --directions h2d d2h d2d --block-bytes 4096 65536 1048576 \
  --num-blocks 1 16 256 --layout random --seed 0 \
  --device 0 --warmup 32 --iterations 100 --output /tmp/swap-blocks.json
```

`h2d` 和 `d2h` 使用锁页主机内存；`d2d` 使用同一 GPU 上的独立缓冲区。
CPU int64 映射将每个选定源块复制一次，目标位置为连续布局或由固定种子生成的
随机排列。缓冲区使用 uint8，块大小以字节精确指定。额外的两个块保持不变。
测量前后逐位检查目标数据、未复制的块和源数据。失败时以非零状态退出，
不写入新的结果文件。

每个样本在 GPU 完全同步后，用主机单调时钟测量一次调用，包含 Python/C++
提交开销和最终设备同步。内存分配、映射构建、初始化、正确性检查及预热均不计入。
因此结果表示包含主机开销的独立复制端到端延迟，而非纯 DMA 时间或与推理重叠时的
吞吐量。重复调用复用同一组缓冲区和映射，不属于冷缓存测量。

独立的 `collectivex-swap-blocks-v1` JSON schema 包含原始样本、以微秒表示的
nearest-rank p50/p90/p95/p99 延迟，以及各延迟分位数对应的有效载荷 GB/s
（`num_blocks * block_bytes / elapsed_seconds / 1e9`）。复制字节仅计算一次，
不累计读写流量。记录还包含方向、布局、种子、API 类型、设备和运行时版本。
EP 汇总与带宽工具不读取该 schema。输出目录必须已存在；基准测试不创建目录。
较大的测试点需要为传输缓冲区和 CPU 正确性参考数据预留内存。

GPU smoke 检查可使用 `--block-bytes 257 --num-blocks 4 --warmup 1
--iterations 2` 并覆盖全部三个方向。可选 GPU 测试同样执行这些复制及正确性检查：

```bash
python3 -m unittest discover experimental/CollectiveX/tests -p 'test_swap_blocks.py' -v
```

仅有 CPU 的环境会执行测量和映射测试，并跳过真实 GPU 测试。

## 独立 GitHub GPU Action

在 **CollectiveX Sweep** 中选择 `backend: swap-blocks`，或执行：

```bash
gh workflow run collectivex-sweep.yml --ref codex/collectivex-swap-blocks \
  -f backend=swap-blocks -f swap_profile=smoke \
  -f swap_image=vllm/vllm-openai:v0.25.1
```

合并后使用 `--ref main`。该模式只调度一个 `h200-dgxc` 任务，优先级队列需求为
`nodes:1`，独占分配一个物理节点并运行一个 GPU 进程。不构建 EP 库，也不执行 EP
测试点。EP 筛选参数应留空；`only_sku` 可留空或设为 `h200-dgxc`。
现有的 `all` 选项仍只覆盖 EP。调用方指定的官方 vLLM 镜像通过现有 CollectiveX
容器缓存导入，并从计算节点可见的隔离暂存目录运行。

`smoke` 覆盖三个方向、两种布局、257/4096/65536/262144 字节（最大 256 KiB）的块大小及 1/4/16/64/256/1024/2048 个块，
每个测试点预热 4 次、采样 20 次，共 168 个测试点。`standard` 使用
4096/65536/1048576 字节的块大小和相同的块数量，预热 32 次、采样 100 次，共 126 个测试点。
两种配置均在计时前后检查真实 GPU 复制；缺少 GPU 或兼容 vLLM 时直接失败。

下载 `cxshard-swap-blocks-<run_id>-<attempt>` 可获得两个 JSON 结果文件，
其中记录实际 GPU、框架版本、镜像、源代码 SHA、正确性状态和测量数据。
失败时同样执行现有的资源分配和暂存目录清理。CPU CI 独立运行，不能证明 GPU 正确性。
