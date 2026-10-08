# MI355X 上的原生 Kimi-K3 PD

[English](k3-pd-native.md) | **中文**

Kimi-K3 prefill/decode 分离沿用共享 Python launcher 和原生 srt-slurm 生命周期。Recipe 管理 serving 设置，master config 管理矩阵标识。

## Serving 配置

每个 worker 使用 TP8、MXFP4 主模型权重、FP8 KV、GMU 0.90 和 1,048,576 token 上下文。DSpark 使用原始发布的 checkpoint，沿用固定框架版本的默认加载方式。AgentX 吞吐自动选择 golden acceptance，精度评测使用真实 block rejection。

并发指全局客户端并发，下表 D 侧上限按每个 worker 计。

| 拓扑 | 并发 | GPU 数 | P/D DCP | Draft token 数 | P CPU offload | P/D 最大序列数 | P/D token budget |
| --- | ---: | ---: | --- | ---: | --- | --- | --- |
| 1P1D | 1 | 16 | 1/1 | 7 | 无 | 2/2 | 16384/512 |
| 1P1D | 48 | 16 | 8/8 | 4 | 1799 GB | 96/96 | 8192/512 |
| 1P2D | 24 | 24 | 8/8 | 4 | 1799 GB | 48/24 | 8192/512 |
| 1P2D | 48 | 24 | 8/8 | 4 | 1799 GB | 96/48 | 8192/512 |
| 1P3D | 24 | 32 | 8/8 | 4 | 1799 GB | 48/16 | 8192/512 |

P 仅在 c1 使用 PIECEWISE graph，其余 P 配置关闭 graph；D 均使用 FULL_DECODE_ONLY。c48 的 P 配置保留显式 scratch reclaim。SimpleCPUOffloadConnector 仅在 P 侧启用，D 使用 MoRIIO READ。1P3D c24 继承 1P2D c24 的策略，只增加一个 D 并调整每个 D 的序列上限。

## 配置职责

- `configs/amd-master.yaml` 注册 `kimik3-fp4-mi355x-vllm-disagg-agentic` 及对应拓扑元数据。
- `benchmarks/multi_node/srt-slurm-recipes/kimik3/vllm/mi355x-fp4/agentx/disagg-variants.yaml` 管理 worker/router 镜像、connector、graph 和 serving 参数。参照现有 AMD disagg recipe，NIC 选择放在 P/D role env；`/dev/infiniband` 挂载限定到当前 recipe。
- `configs/runners.yaml` 声明目标 checkpoint 位置及具名 draft/provider volumes。K3 路径选择只读挂载 draft checkpoint 和节点本地 Ionic provider。
- `infx/launch/` 管理任务提交、取消及产物收集。共享 backend 准备 worker 镜像；recipe 使用原生 Pyxis/Enroot digest 引用指定 router 镜像，由 frontend 的任务 step 导入。

RDMA 注册和 CPU offload 使用集群继承的锁页内存限额。部署到其他集群时，应验证 worker 实际限额、routed request 和 CPU offload 读写。集群特定设置限定在对应 workload 内。

## 官方镜像

Master 与 recipe 固定同一不可变 worker 镜像：

`vllm/vllm-openai-rocm:nightly-81198e97ba7eee2a22540caaa756b7fdddcb4d93@sha256:a401e4f46872dcde77e07ae3d7fa2d3a71b7f2773714316116532e2b39deb179`

Worker 运行该镜像提供的 vLLM。

## 参数归属

每个 master 配置点从 recipe 的共享 `base` 中选择一个 `override_<topology>_c<concurrency>`。Engine flags 放在 `roles.prefill.args` 或 `roles.decode.args`，worker 环境变量放在相应的 `env`。YAML anchor 共享配置，原生 override 展开每个点的差异。JSON 类型的 engine 参数仍写成 JSON 字符串，因为当前固定版本的 srt-slurm CLI renderer 对普通字典只做字符串转换，不做 JSON 编码。

Custom benchmark 使用运行时发现的 `SRT_FRONTEND_HOST` 和 `SRT_FRONTEND_PORT`。其 `benchmark.env` 提供客户端配置，不提供 engine flags。`KV_OFFLOADING` 和 `TOTAL_CPU_DRAM_GB` 描述 connector 配置并与 master 元数据一致，不负责配置 server 内存。`AIPERF_LIVE_FAILED_REQUEST_THRESHOLD=0.01` 只控制实时中止，已完成 profiling 的错误率门槛由共享 collector 管理。

SSM state 仍显式设置为 `float32`；SiTUv2 activation 选择和 request-ID 随机化遵循固定镜像的上游默认值。State layout、DCP collective 选择、非阻塞 collective 和 c48 scratch reclaim 保留为显式兼容设置。

## Discovery 依赖

[srt-slurm #508](https://github.com/NVIDIA/srt-slurm/pull/508) 将实际分配的 discovery endpoints 绑定到显式 MoRIIO/MultiConnector 模板，同时保留 CPU-offload sibling。[补丁清单](../runners/srt-slurm/patches/README.md) 统一记录其上游版本。沿用 `runners/srt-slurm/patches/` 机制，将 backport 应用到每个任务的私有 checkout。共享 srt-slurm pin 包含该功能后即可移除 patch 及其 README 条目。

PR 生成器按仓库默认策略、样本数和评分门槛选择精度评测。最终栈合入前须满足仓库的 full-sweep 和评测要求，主 sweep 标签用于启用 GPU 验证。
