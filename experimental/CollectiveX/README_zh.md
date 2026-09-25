[English](README.md) | **中文**

# CollectiveX

CollectiveX 是实验性的混合专家(MOE)专家并行(EP)通信基准测试。它衡量不同 EP 库和加速器系统的分发、合并及配对往返延迟，并上传中立的结果产物。

独立的 [vLLM `swap_blocks` 基准测试](docs/swap-blocks.md)（[中文](docs/swap-blocks_zh.md)）衡量锁页 CPU↔GPU 内存复制和同 GPU 内的块复制，使用独立的正确性检查以及延迟/带宽 JSON 输出。

CollectiveX 负责调度基准测试、在实际分配的资源上执行并上传每次运行生成的中立产物。它不验证这些产物，不做晋级、排名、推荐、筛选，也不决定下游展示什么。下游展示和比较由使用方负责。完整测量方法见 [docs/methodology.md](docs/methodology.md)（[中文](docs/methodology_zh.md)）。

## 执行配置

工作负载采用 packed 放置，每个后端/拓扑使用一个固定的 `fixed-profile` 资源配置，不做调优扫描。合并始终使用 BF16。分发精度是扫描维度：包含 BF16 对照，以及上游支持 FP8 分发的后端（DeepEP V2、MoRI、UCCL-EP、FlashInfer EP）的 FP8 用例。`normal` 模式由调用方预量化；`low-latency` 模式下 DeepEP 和 UCCL-EP 内核从 BF16 内部量化，MoRI 仍由调用方预量化，FlashInfer 没有该模式。调用方量化计入分发时间，因为生产前向路径同样承担这项开销。本版 NCCL EP 仅测 BF16。路由仅覆盖 uniform，用例分为两种模式：

- `normal` 使用 `layout-and-dispatch-v1`：按 rank 去重的 token 载荷，以及仅对激活做无权 rank 求和的合并。覆盖完整解码和预填充 token 梯度。
- `low-latency` 使用各后端面向解码的内核。DeepEP 使用旧版 `deep_ep.Buffer` IBGDA `low_latency_dispatch`/`low_latency_combine`，接收缓冲按专家填充，在源端按门控权重合并。UCCL-EP 使用相同的旧版 `Buffer` 内核；本测试的 EP8 通过 NVLink 上的 `cudaIpc` 通信，不走 CPU 代理。MoRI 使用分阶段调用的 `AsyncLL`（每个计时组件内包含 send+recv，即 SGLang 部署的低延迟内核），保留与 `IntraNode` 相同的紧凑布局和无权 rank 求和。此模式仅用于解码，按 SKU 的 `ll_backends` 能力表启用，运行集合与 `normal` 不同。

当前 DeepEP V2 覆盖 H100/H200 的 EP8，以及 B200、GB200/GB300 的 EP8 和 EP16。B200 nscale 裸机池具有原生 IB 和 gdrdrv，可运行 IBGDA 跨节点低延迟路径；虚拟化池不具备这个条件。GB 的 EP16 仍位于 MNNVL scale-up 域内。MoRI 覆盖 MI300X/MI325X/MI355X EP8。UCCL-EP 低延迟仅覆盖 H100/H200/B200 EP8：上游在 2026-07-13 将 `kNumMaxTopK` 从 9 改为 16（uccl#1016，早于当前固定提交六天），导致 `kNumMaxTopK + 1 <= num_warp_groups * num_warps_per_group` 在 AMD 的 16 个 warp group 上无论 CU 数量都无法成立。这是特定版本回归，并非硬件限制；AMD 保留 UCCL-EP normal 模式。

NCCL EP v0.2 增加 H100/H200/B300 EP8，以及 B200/GB200/GB300 EP8 和 EP16 低延迟用例。`nccl-extensions` wheel 补齐了此前缺失的 combine-recv fence；更早的 [NVIDIA/nccl#2303](https://github.com/NVIDIA/nccl/issues/2303) 信号复用问题已通过单 handle 修复。B300 唯一的低延迟用例是 `candidate` 级 NCCL EP，没有生产级解码覆盖。DeepEP V2 在 B300 上不生成 LL 行（见下表中的 IBGDA 地址句柄问题）；`_ll_runnable` 只加入可运行的组合，因此该限制以文档记录，而非矩阵中的分类行呈现。

单节点 EP8 的低延迟载荷走 NVLink/XGMI，不需要 `/dev/gdrdrv`（已在缺少该设备的 H200 上验证）。NVSHMEM/IBGDA 只有在多节点 EP16 scale-out 时才承担线上载荷。但旧版 Buffer 在 EP8 仍会自行启用 IBGDA，这也是 B300 失败的原因。

固定计时配置来自 `configs/sweep.json`：256 轮，每轮 8 次计时迭代，即每个组件 2048 个样本；每轮、每个点、每个被测组件前，先执行 32 次同步的完整往返预热。组件顺序逐轮轮换；每次迭代先取跨 rank 最大值，再计算 nearest-rank p50/p90/p95/p99。带密钥的 BLAKE2b 计数器保证不同运行时生成逐字节相同的路由和门控权重。

上述组件测的是 **fresh entry**：每个计时窗口前后排空 GPU，反映空闲流水线的进入成本，而非连续解码成本。每行还记录**链式配对周期**：4 轮，每轮连续提交 128 个 dispatch→combine 对，中间不做主机同步，丢弃前 16 对作为流水线填充，共 448 个观测，跨 rank 取中位数。每轮运行两条配套计时链：先运行仅带操作事件的 floors 链，再运行仅带外层配对事件的 period 链。旧版每对六个事件的单链在设备快于主机时，将四次内部 `record()` 的成本计入周期，形成近似固定的 10–30µs 主机开销，导致全平台 T=1 结果高出 20–38%。

对于存在 `components.pair_period` 的行，该字段是主延迟指标。此口径于 2026-08-06 在 B200/H200/GB200 手工参考与双链全平台产物核对后启用；`summarize.py` 通过脚注明确星号列口径。floors 链输出各操作窗口的跨 rank 最小值 `chain_floor_us`；period 链同时提供 `chain_health.pair_spread_us`（跨 rank 节奏一致性）、`interpair_gap_us`（已发布窗口外的每对成本，用于防止仪器开销重新混入，也区分同步主导的 `period − Σfloors` 残差）和 `settle_drift_us`（后半段减前半段周期，验证 `chain_drop` 假定的收敛）。残差正负的含义见方法文档。

不发布链式单操作中位数：跨 rank 等待会落在不同操作窗口内，在一个 rank 上稳定、跨 rank 任意分配，只有配对总周期守恒。现有字段没有重命名或重新定义，扫描 `version` 仍为 1，使用方应检查 `components.pair_period` 是否存在。

链式运行有两重检查：每轮最终合并输出与相同路径的排空配对比较（`correctness.chain_last_output_passed`，差值在 `correctness.chain_last_output_error`）；每个梯度点还对链留下的状态再次运行完整 oracle（`correctness.post_chain_state_passed`）。后者始终作为门禁，前者仅在每对都执行 staging 时作为门禁；staging 外提时为 `null`，因为两条路径的合并输入均不对应自身的分发结果，不可比较。相同 H100 用例在外提时误差为容差的 1000×–2966×，不外提时恰为零。详见正确性章节。该 `null` 是已知、范围明确的证据缺口：仅在 FP8 自由连续运行中出现、且不留下状态的错误可能逃过所有门禁；`CX_FP8_CONSUME=dequant` 是持续保留的诊断入口。

每行的 `roundtrip` 都表示分发后合并的通信过程。专家输出 staging 位于其外，单独报告为 `stage`。FP8 的 staging 是测试框架用来代替专家 GEMM 的辅助工作；生产 GEMM 原生读取 FP8，不会单独物化 BF16 副本。因此 **`stage` 不能累加到总延迟，也不能跨后端比较**。此前 MoRI BF16 和 FlashInfer BF16 将 staging 复制包含在链内，且 `version` 未变，应通过 `implementation.stage_excluded_from_roundtrip` 和 `stage` 是否存在来区分。完整约定见 [方法文档](docs/methodology.md)。

正确性使用独立于实现的 oracle，复现两级归约：scale-up 域内 FP32 累加，再将各域部分和转为 BF16 发送到 scale-out。合并门限为最大逐元素相对误差小于 `8 * 2^-8`，分母下限为 0.02，适用于 scale-up 和跨节点 scale-out。FP8 oracle 对语义载荷执行同样的逐 token 量化往返，使分发载荷比较保持逐位精确，合并容差不变；量化由模型明确处理，不靠放宽容差容忍。任一 rank 或点失败都会在结果中将该用例标为不合格。

矩阵覆盖 H100、H200、B200、B300、GB200、GB300、MI300X、MI325X、MI355X。`sweep_matrix.py` 展开请求的 SKU、后端、EP 规模和 token 梯度，提取严格的分片控制，并拒绝缺失、过期、格式错误或被篡改的控制。`--only-sku`、`--exclude-skus`、`--ep-sizes`、`--precisions` 用于选子集。每次调度重新生成矩阵，不固定摘要或用例数量。

| 系统 | EP8 | EP16 |
|---|---|---|
| H100/H200/B200/B300 | 1x8 NVLink，scale-up | 2x8 NVLink + RDMA，scale-out |
| MI300X/MI325X/MI355X | 1x8 XGMI，scale-up | 2x8 XGMI + RDMA，scale-out |
| GB200/GB300 | 2x4 MNNVL，scale-up | 4x4 MNNVL，scale-up |

物理主机数不决定 scope：两种 GB 拓扑都在一个 72-GPU MNNVL scale-up 域内。

| 后端 | 引擎可用性 | 当前范围 |
|---|---|---|
| DeepEP V2 | `production`；vLLM `--all2all-backend deepep_v2`、SGLang `--moe-a2a-backend deepep` | normal 使用 PR #605 的 `ElasticBuffer`，包含 #630/#640 修复：scale-up 走 LSA，x86 EP16 走 GIN。BF16 与 `use_fp8_dispatch` 分块 e4m3fn 分发并列。LL 使用旧版 `deep_ep.Buffer` IBGDA 解码内核、按专家填充布局、加权合并和 `use_fp8` e4m3fn；覆盖启用的 EP8，以及 GB 和 B200 nscale 裸机 EP16。B300 即使 EP8 也会自行启用 NVSHMEM IBGDA，地址句柄创建报 `ibgda.cpp:2234 Unable to create ah`，八个 rank 均 rc255。`NVSHMEM_DISABLE_IB=1` 无效；在 b300-002、b300-011 上设置和不设置均失败。 |
| MoRI | `production`；vLLM `--all2all-backend mori_*`、SGLang `--moe-a2a-backend mori` | normal 在所有 CDNA SKU 的 EP8 使用 `IntraNode`；mi355x EP16 通过 Pollara RoCE 使用 `InterNodeV1` 和默认 `STATIC_HEAP`。TW 没有跨节点 GPU 网络，EP16 不支持。此前两项调用方问题分别是把 dispatch 返回的 recv-slot 索引而非本 rank 路由传给 `combine()`（ROCm/mori#475、#546），以及强制 `MORI_SHMEM_MODE=VMM_HEAP`，在 ROCm 7.2 上令堆进入缓存池、破坏跨节点合并部分和（ROCm/mori#610）。同分支 A/B：STATIC_HEAP 在运行 34939333022 全梯度通过，VMM_HEAP 在 34941185534 复现；适配器现在拒绝非 STATIC_HEAP。mi355x 的 UCCL-EP EP16 CPU 代理可通信但仅约 6 GB/s，远低于文档的 82 GB/s，注册方式/traffic class 不改变，怀疑 ionic 驱动。LL 使用分阶段 `AsyncLL`（旧行使用 `IntraNodeLL`，通过 `kernel_generation` 区分），仅解码/EP8。FP8 由调用方预量化：gfx942 e4m3fnuz，gfx950 e4m3fn；合并保持 BF16、`quant_type=none`。 |
| UCCL-EP | `candidate`，没有引擎暴露该选择项 | [UCCL](https://github.com/uccl-project/uccl) 是 DeepEP API 兼容替代，CPU 代理通过 `libibverbs` 发起 GPUDirect RDMA，不依赖 NVSHMEM/IBGDA，以软件实现消息顺序、原子操作和流控。scale-up 为单节点 NVLink/XGMI `cudaIpc`，不支持 MNNVL。normal 为旧版 `Buffer` 的无权 rank 求和；LL 为旧版低延迟加权合并，解码/EP8。normal FP8 调用方预量化，分块 e4m3fn（gfx942 为 e4m3fnuz）；LL 传 BF16，由内核内部量化；合并 BF16。覆盖 H100/H200/B200 和 MI300X/MI325X/MI355X EP8。EP16 网络可连通、轻用例正确，但重 token 用例 CPU 代理吞吐量无法满足统一用例时限，暂记为不支持。 |
| NCCL EP | `candidate`，NVIDIA 原生库，但没有引擎选择项 | [NCCL EP v0.2](https://github.com/NVIDIA/nccl-extensions) 使用 NCCL Device API，节点内 LSA，节点间 GIN。v0.2 起由 `nccl-extensions` wheel 提供 `nccl.ep`，并固定 `nccl4py` 提供 `nccl.core`；替代 2026-06-11 后未更新的 `NVIDIA/nccl` 中 `contrib/nccl_ep`。normal 使用 `HIGH_THROUGHPUT`，FLAT `[N, hidden]` 接收和无权 rank 求和。HT 要求 `ZeroCopyMode.ON`：接收 token 平面由 NCCL 分配，collective 注册为对称窗口，dispatch/combine 直接使用。LL 使用推理引擎的 rank-major 预归约约定，在单个 LSA/MNNVL 域内启用零复制，scale-out 关闭。LL 覆盖 H100/H200/B300 EP8 及 B200/GB200/GB300 EP8+EP16，梯度恢复到完整 256-slot 缓冲。v0.1 缺少 DeepEP #642 的 shared-memory fence，GB300 T=256 曾五次中一次错误 0.4704（正常 0.0039）；v0.2 `ll_ep.cuh` 在 `emptyBarriers` arrive 前加入 `fence_view_async_shared`。本版只测 BF16；FP8 `DS_FP8E3M4` 和实验性 NVFP4 合并仍需单独接入。仅 NVIDIA/CUDA 13，六个 NVIDIA SKU 均覆盖 EP8/EP16；GB 为 MNNVL scale-up，其余为 2x8 GIN scale-out。v0.1 GIN 在 x86 RoCE/IB 均故障，v0.2 已在 B200/H100/H200 实机复验，B300 按同一修复启用，待该池复验。 |
| FlashInfer EP | `production`；vLLM `--all2all-backend flashinfer_nvlink_one_sided` | [FlashInfer](https://github.com/flashinfer-ai/flashinfer) `MoeAlltoAll` 为 TensorRT-LLM 单边 MNNVL all-to-all，各 rank 直接写对端 workspace，合并读回，无 send/recv 配对或 NVSHMEM。仅 normal、GB200/GB300，EP8/EP16 均在 scale-up 域内。FP8 由调用方按块预量化为 e4m3fn，并以第四份载荷携带每 128 元素块的 FP32 scales；合并强制 BF16，C++ `toNvDataType` 仅接受 fp16/bf16/fp32。0.6.16 之前按载荷 dtype 做 BF16 top-k 配对树归约，每层舍入；oracle 使用 `combine_reduction="topk-slot-tree"` 建模，不放宽容差。0.6.16 改为 FP32 累加，适配器按安装版本切换模型。 |

DeepEP V2 指 [PR #605](https://github.com/deepseek-ai/DeepEP/pull/605) 引入的 `ElasticBuffer`，不是更新的旧版 `Buffer`。固定源码为上游 `main`，含 [#630](https://github.com/deepseek-ai/DeepEP/pull/630)（无 GIN 时纯 scale-up 初始化）、[#640](https://github.com/deepseek-ai/DeepEP/pull/640)（不将 NCCL 共享内存映射误认成重复库）、[#642](https://github.com/deepseek-ai/DeepEP/pull/642)（修复 [#700](https://github.com/deepseek-ai/DeepEP/issues/700) Blackwell 顶部梯度损坏的 LL combine fence）。此前固定在 #605 合并前分支的 #630 head，尚无这些后续修复。scale-up 必须证明实际 LSA team 覆盖整个 EP world，否则失败；x86 EP16 必须使用 GIN 混合路径，两逻辑 scale-out 域由两个物理 RDMA rank 表示，各域八个 scale-up rank；GB EP16 仍使用 LSA。是否尝试某 SKU/后端/EP 组合由能力决定，是否成功由基准测试退出码决定。

## 工作流与产物

`.github/workflows/collectivex-sweep.yml` 包含两个 job。`setup` 根据 `backend`、`only_sku`、`exclude_skus`、`ep_sizes` 生成公开 SKU 矩阵并上传。`sweep` 为每个矩阵单元提取严格的、被 Git 忽略的 `.shards/<id>.json` 控制，每个分片使用一次资源分配，需要时在申请前拉取固定 DeepEP 源码，并通过 `always()` 上传产物，失败或部分运行也会上传。

每个分片输出逐用例 JSON 和简短的机械汇总。成功以基准测试自身退出码为准；无额外完整性或隐私验证，失败或不支持的单元不生成合成记录。流程不晋级运行、不构建数据集、不推进发布通道；中立产物就是交付物，由使用方下载并决定展示方式。

工作流不传入或上传操作员凭据。运行器本地覆盖和选择器留在运行器上。逐步骤日志留在运行器供排障，结果只含方法文档列出的字段。

## 运行器配置

各 SKU 的 Slurm 和存储参数来自 registry 中的受版本控制基线。可选本地 JSON 位于 `$XDG_CONFIG_HOME/inferencex/collectivex.json` 或 `COLLECTIVEX_OPERATOR_CONFIG`，逐字段覆盖基线。缺少 registry 项、未知字段或非 JSON 输入均失败，配置从不作为 Shell 执行。重复 JSON key **不会**被拒绝，`json.load` 静默保留最后一个；其他 SKU 的条目不验证，因此 SKU 拼写错误会被忽略。GHA 不传操作员 secret，没有本地文档时完全使用受版本控制基线。

所有公开平台数据在 `configs/platform_config.json`：架构/产品、镜像与平台、固定放置、启动器、可运行后端/EP 组合、scale-out `fabric` 身份（NIC 与交换机，使相同 GPU 的不同网络池可分开登记）、操作员默认值、RDMA 选择器。本地操作员文档可覆盖默认值。启动器声明并检查实际需要的字段，`sweep_matrix.py` 根据放置字段推导 EP 拓扑，默认扫描所有登记 SKU。

每个非 MNNVL EP16 放置还必须给出经操作员确认的 `rdma_devices`。`socket_ifname` 固定跨节点接口；未指定时使用所分配节点的默认路由接口（h200-dgxc、b300）。可选字段为 `ib_gid_index`、`rdma_service_level`、`rdma_traffic_class`、`rail_isolated`、`rdma_fabric`；service level 和 traffic class 映射到 MoRI RDMA/IO QoS 环境。CollectiveX 不启发式选择管理网络或 HCA。非 MNNVL scale-out 每个节点必须在准备后端前证明接口和活动 HCA 端口存在；scale-up/MNNVL 清除这些覆盖。scale-out 将 NCCL/RCCL 固定到 `IB` 并精确匹配 HCA，避免 socket 回退误标为 RDMA。`NCCL_IB_MERGE_NICS=0` 禁止双端口融合，因为融合会关闭 DeepEP V2 EP16 所需的 NCCL GIN。独立 rail 子网、无跨 rail 路由时，`rail_isolated=1` 还设置 `NCCL_CROSS_NIC=0`。

仅当所有所选 HCA 端口都报告 Ethernet 链路层时应用 `ib_gid_index`，选择批准的 RoCE GID。原生 InfiniBand 保留 HCA/service level 固定，但不设置 RoCE GID，让 NVSHMEM/NCCL 使用原生 LID。混合 Ethernet/InfiniBand 列表会被拒绝。

`rdma_fabric: "efa"` 表示 AWS Elastic Fabric Adapter 池（B300 为每节点 16 个 EFA 的 p6-b300.48xlarge）。EFA 不是 verbs HCA，没有链路层、GID 表、service level 或 traffic class；NCCL 经集群 enroot hook 挂载的 aws-ofi-nccl libfabric 插件访问。探测允许未指定链路层，但设备必须在各节点 ACTIVE。IB 选择器族（`NCCL_IB_*`、`NVSHMEM_HCA_LIST`/IBGDA、MoRI、UCCL）保持未设置，通过 `NCCL_NET_PLUGIN=ofi`、`FI_PROVIDER=efa` 选插件，NVSHMEM 使用 libfabric，容器没有插件库则准备失败。

GIN 走插件实现，由 `OFI_NCCL_GIN_TYPE` 选择 proxy 或 GDAKI，需要节点 GDRCopy 2.5+、已加载 `gdrdrv` 和 `libgdrapi`。缺失时 GIN 初始化失败、NCCL 禁用 Device API、`ncclEpCreateGroup` 返回 `ncclInvalidUsage`。2026-09-11 在 b300-dsxe 验证该池没有 gdrcopy，故 registry 暂停其 EP16。该池也未启用 DeepEP V2 LL：旧版 Buffer 单节点也初始化 RDMA NVSHMEM，设置或不设置 `NVSHMEM_REMOTE_TRANSPORT=none` 均以 “nvshmem detect topo failed”/状态 28 退出（34521825749、34523594787）；无需 NVSHMEM 的 HT ElasticBuffer 可通过。

`stage_dir` 是检出目录和工作流 workspace 外、预先存在、归运行器所有、非符号链接且组/其他用户不可写的共享基目录，在运行器和所有分配节点上路径一致。作业只创建带标记的 0700 子目录，验证跨节点读写可见性，资源释放后只移除该子目录。不挂载运行器原检出目录；AMD 不在镜像缓存下暂存源码。AMD 未指定 `stage_dir` 时，在共享运行器文件系统的标准 `_work` 旁推导私有基目录，不使用 root 所有的 squash 缓存。

H200/B200/B300 可以省略 `stage_dir`：在验证过的操作系统账户 home 下建立 0700 基目录，不依赖工作流临时 `HOME`。H100 可在共享容器目录旁建立私有基目录，确保计算节点可见。规范 B300 执行忽略旧 `stage_dir`，总用经过验证的、计算节点可见的账户 home，加执行 ID 后缀隔离并行 worker。规范 GB300 同样忽略旧的组可写 `stage_dir`，在账户 home 下派生执行专用私有基目录。各节点从暂存树准备后端。

Enroot 根据镜像 tag 和平台导入按运行划分的 squash，避免复用其他运行导入的文件系统。镜像 tag/平台来自各 SKU registry。DeepEP V2 源码版本固定在 `runtime/build.py`，获取后严格核对提交、检查 `ElasticBuffer`，并按架构、镜像、提交缓存集群本地构建。容器只看到固定 `/cx-cache` 挂载。

## 本地检查

```bash
uv sync --locked --all-extras --group test
.venv/bin/python -m pytest experimental/CollectiveX/tests experimental/operatorx/tests -q
python3 experimental/CollectiveX/sweep_matrix.py --backend all --out /tmp/cx-matrix.json >/dev/null
```

核心路径为 `configs/`、`sweep_matrix.py`、`summarize.py`、`bench/`、`runtime/`、`ci.py` 和 `tests/`。[实现导览](docs/architecture_zh.md)介绍调用流程、类和模块边界。工作流执行逻辑全部使用 Python；作业步骤使用 `collectivex` extra 中固定版本的 `simple-slurm`，资源分配仍保持原有 `salloc`/`squeue`/`scancel` 生命周期。
