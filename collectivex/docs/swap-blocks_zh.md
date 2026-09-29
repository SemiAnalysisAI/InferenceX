# vLLM 块复制基准测试

[English](swap-blocks.md) | **中文**

`bench/run_swap_blocks.py` 在单个 CUDA 或 ROCm GPU 上测量
`from vllm._custom_ops import swap_blocks`，需要安装兼容的 vLLM。
可在该环境中直接使用 Python，或通过下方的 GitHub Action 触发该套件。
无需 `torchrun`，也不会执行 EP 工作负载。
结果记录已安装的 vLLM 版本，同时支持旧版三参数接口和显式传入
`block_size_in_bytes` 的接口。

```bash
python3 collectivex/bench/run_swap_blocks.py \
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
可选参数 `--max-payload-bytes` 在分配内存之前排除 `block_bytes * num_blocks`
超过上限的组合；没有可执行测试点时直接失败。JSON 的 `selection` 记录请求网格、
预算及被排除的组合与原因；被排除的组合没有计时或正确性结果。
该有效载荷上限不包含两个保护块及 CPU 参考缓冲区。

GPU smoke 检查可使用 `--block-bytes 257 --num-blocks 4 --warmup 1
--iterations 2` 并覆盖全部三个方向。可选 GPU 测试同样执行这些复制及正确性检查：

```bash
python3 -m unittest discover collectivex/tests -p 'test_swap_blocks.py' -v
```

仅有 CPU 的环境会执行测量和映射测试，并跳过真实 GPU 测试。

## GitHub GPU Action

swap-blocks 是 **CollectiveX Sweep** 的一个套件。设置 `suites: swap-blocks`（或 `ep,swap-blocks`
同时运行两者），或执行：

```bash
gh workflow run collectivex-sweep.yml --ref main \
  -f suites=swap-blocks -f swap_profile=smoke
```

`only_sku` 留空时运行所有已注册的 GPU 池，也可单独指定 `h200-dgxc`、`h100-dgxc`、`b200-nscale`、
`b300`、`gb200`、`gb300`、`mi300x`、`mi325x` 或 `mi355x`。`exclude_skus` 接受以逗号分隔的排除列表。
EP 筛选项（`backend`、`ep_sizes`、`modes`）仅作用于 `ep` 套件；未选择该套件时，矩阵会拒绝这些筛选项。

每个池生成一个分片，包含两个用例，每种布局一个。分片使用该池自己的启动器运行，因此沿用该池的
资源分配、节点校验、镜像导入和清理流程；它申请一个节点、一个 GPU，时长 45 分钟。`config.py`
把每个用例编码为 `run_swap_blocks.py` 的参数，rank 包装脚本执行它，而不是 `run_ep.py`。

测试网格和镜像都写在 `configs/swap_sweep.json` 中。CUDA 与 AMD 池使用其中指定的官方 vLLM 镜像；
GB 池通过注册表的 `image_platform` 选择 ARM64 镜像，`sku_images` 可为某个池固定镜像。如需其他 vLLM
版本，请在分支上修改配置后从该分支触发。每种配置都覆盖三个方向和两种布局，并在计时前后检查真实 GPU
复制；后端准备阶段会在任何用例运行前确认能导入 `swap_blocks`。`smoke`（默认，块最大 256 KiB，168 个
测试点）和 `standard`（4 KiB 至 1 MiB，126 个测试点）的全部测试点都在 2 GiB 有效载荷上限内。
`large-blocks` 扫描 257 B 至 1 GiB 的块，**复制有效载荷上限为 1 GiB**，超出的组合会被排除并记录
（测量 294 个测试点，排除 126 个）；每个缓冲区另含两个保护块，因此 1 GiB 块的测试点每个缓冲区分配
3 GiB，另需 CPU 正确性参考缓冲区。

下载 `cxshard-<sku>-swap-blocks-<run_id>-<attempt>` 可获得两个 JSON 结果文件，文件名为
`<case_id>_<timestamp>-c<index>.json`，其中 case_id 为 `<sku>-swap-blocks-<profile>-<layout>`。
结果记录实际 GPU、框架版本、镜像、源代码 SHA、正确性状态和测量数据；EP 汇总表会跳过这些文档。
CPU CI 独立运行，不能证明 GPU 正确性。

无法自行导入镜像的池可以在 `sku_images` 中指定运维预置的镜像缓存（`staged_image_dir`）。H100 即如此：
它先在 `/mnt/nfs/lustre/containers` 中按推理启动器的文件命名规则查找与请求标签完全一致的镜像，
有效的 squash 可直接复用，无需在计算 pod 内导入；文件不存在时走常规导入路径。
`refresh_image=true` 会绕过预置缓存，要求重新导入。工作流为 H100 选择 `/var/tmp` 作为镜像导入
临时目录，其他平台使用 `/tmp`。
