[English](methodology.md) | **中文**

# CollectiveX EP 基准测试方法

CollectiveX 调度专家并行(EP)通信基准测试，在实际加速器资源上运行，上传每次执行生成的中立产物。它**不**验证这些产物，不做晋级、排名、推荐、选择、隐藏，也不决定使用方展示什么。前端读取中立的矩阵、结果和汇总，自行决定覆盖与展示。本页描述用例如何调度、测量、检查和记录，不是发布或准入规范。

## 产品边界

CollectiveX 是通信微基准测试，用于：

- 在同一芯片/拓扑上比较 EP 库。
- 在相同工作负载下跨系统比较 EP 延迟和逻辑载荷带宽。
- 显式呈现不支持、失败、无效或不稳定的用例。

没有独立相关性研究时，它不能预测推理服务吞吐量。

## 矩阵

当前工作负载为 `deepseek-v4-pro`（DeepSeek-V4-Pro 1.6T，即 InferenceX 其余测试中的 `dsv4`）：hidden 7168、top-k 6、384 个路由专家、packed 放置，各后端/拓扑固定一个资源配置。合并始终为 BF16。分发扫描 BF16 对照和上游支持的 FP8（DeepEP V2、MoRI、UCCL-EP、FlashInfer EP）。normal 模式由调用方预量化；LL 模式 DeepEP/UCCL-EP 从 BF16 内部量化，MoRI 仍预量化。

调用方量化计入**被测分发窗口内**，因为生产前向的关键路径也需要它。DeepEP V2、UCCL-EP、FlashInfer 使用一个融合内核，与 eager 参考逐位比对；MoRI 只是 dtype cast，无需融合及此检查。因此 FP8 normal 测量量化加传输，BF16 只测传输，不能和该变化之前的 FP8 行比较。扫描 `version` 刻意保持 1；`implementation.stage_excluded_from_roundtrip` 与 `stage` 的存在性是这一代变化的区分字段。

这两个字段不能区分所有历史版本。持久存储中相同单元的比较表明：加入 FP8 量化后，小 T 分发中位数增加 64%–110%；扩大 staging 外提范围让 FlashInfer 往返下降 46%、MoRI BF16 下降 23%；MoRI 改为外部输入缓冲配 16 warps 后，gfx950 FP8 合并增加 33%–65%；NCCL EP HT 按接收数量设置合并规模，合并下降 10%–41%；双链计时使各平台 `pair_period` 下降 5%–42%。只有前两项可由上述字段识别。

**存在 `chain_health.interpair_gap_us`** 表示双链计时。单条六事件链的行仅有 `pair_spread_us`，其 `pair_period` 混入主机常量，应视为有缺陷数据，而非另一种有效口径。MoRI 缓冲模式和 NCCL EP 合并规模变化**没有逐行区分字段**，`kernel_generation` 也相同，必须从存储中排除变化前的行。

未来若提升扫描版本，**跳过 2，直接用 3**。开发期间短暂产生的 `version: 2` 文档目前仅因前端只接受 `[1]` 而不可见；支持 2 会把开发中间状态重新当作有效数据。

`fp8_consume=dequant` 是验证入口，不是第二项指标、扫描轴或默认值。其额外成本可由已有输出估算：

    dequant roundtrip  ~=  roundtrip + stage        (+2.4% .. -0.1%, b200 LL fp8 ladder)

估计略高，因为链式运行摊薄启动成本（全语料 `rt/(d+s+c)` 中位数 0.93）。反向推导不成立：用 `dequant - stage` 重建 native，在 T=1 误差 -11.6%、T=64 误差 -5%，直到 T=256 才收敛，恰在解码主指标所在区间最差。因此测 native、推导 dequant，不能反过来。该入口可复现历史 deepep-v2/uccl-ep 数字（运行 30177021271 的 T=1：302.0µs 对 302.5µs），不能复现现在只转换有效接收行的 MoRI FP8 或外提前的 BF16 往返。

这是**固定每调用成本，不随载荷成比例**。DeepEP V2 解码 T=1 时 FP8 相对 BF16 分发 p50 增加：H100 65µs、B200 59µs、B300 27µs、GB300 57µs；到 T=64 基本保持在 1–2µs 波动内，随后下降。T=512 时 H100/B200 略为负值，载荷减半抵消了成本。FlashInfer GB300 约 107µs，使用自己的 codec 和第四份 FP8 分发载荷。T=1 的 FP8 搬运字节更少，额外成本来自每调用工作，不是通信；它比融合量化自身的 1.5–3.6µs 设备时间高一个数量级以上。开始事件前没有主机同步，近空闲 stream 让主机启动成本落入计时窗口。融合编译降低而非制造这一成本：同样窗口下 H100 eager 量化还会多 33–39µs。生产是在已忙碌的 stream 中提交一个量化操作，因此 FP8 normal 小 T 数字是本套测试最不代表生产的部分；应在梯度顶端比较 FP8/BF16。LL 不受影响，因为量化在内核中，或 API 明确要求预量化输入。NCCL EP 本版只有 BF16。支持精度由 `BACKEND_PRECISIONS` 限定，不生成不支持的组合。normal 使用 `layout-and-dispatch-v1`，LL 使用各自的解码内核语义。

- `ep-core`：对模型 token 梯度进行 uniform 路由。`deepseek-v4-pro` 解码 T=1..512、预填充 T=1024..8192，均为 2 的幂。梯度与模型一起定义在 `configs/sweep.json`。

后端可截断梯度，但必须报告 `workload.ladder_measured`、`ladder_dropped`、`ladder_cap`，不得静默丢弃。DeepEP V2 LL 固定接收容量和上限均为 256，因此只丢弃 T=512。

该限制曾临时用于控制回归：DeepEP LL combine 在 Blackwell（B200/GB200/GB300，EP8/EP16、两种精度、MNNVL/RDMA）T=256 每调用约 1.5%–3.3% 概率损坏一个 token 行，范数仍匹配到四位有效数字，Hopper 正常。上游 #642 加 CTA 级 fence，确保共享内存读取完成后才复用 staging 缓冲。旧 pin 在此修复之前，曾把梯度限制到 128，直到改用上游 main。接收容量由常量而非 `max(ladder)` 决定，避免截断改变通信流量和 FP8 反量化规模。

`sweep_matrix.py` 展开请求的 SKU、后端、EP 规模和 token 梯度，提取严格分片控制。`--only-sku`、`--exclude-skus`、`--ep-sizes`、`--precisions` 只选子集，不改变约定。每次调度重新生成矩阵，不固定摘要或用例数。

| 系统 | EP8 | EP16 |
|---|---|---|
| H100/H200/B200/B300 | 1x8 NVLink，scale-up | 2x8 NVLink + RDMA，scale-out |
| MI300X/MI325X/MI355X | 1x8 XGMI，scale-up | 2x8 XGMI + RDMA，scale-out |
| GB200/GB300 | 2x4 MNNVL，scale-up | 4x4 MNNVL，scale-up |

**虚拟化池可能测到 hypervisor 限制而非网络能力。** 相同拓扑/流量下 h200-dgxc EP16 跨节点成本约为 B300/H100 的三倍，但 EP8 正常。每节点约 34 GB/s，标称 8x400G，即每 GPU-NIC 对仅约 4.2 GB/s，裸机 H100 可达线速。将 NIC-PE 映射调整为 socket 本地 NIC 没有改善（478µs 对 480µs），排除了选择器，指向 guest 内 GDR 路径整体退化。已退役的 b200-nscale 池曾有相同现象。在确认宿主 ACS/IOMMU 配置前，应把虚拟化池 EP16 视为硬件下限。

物理主机数不决定 scope；两种 GB 组合都位于一个 72-GPU MNNVL scale-up 域内。

不支持的组合在矩阵中显式分类。DeepEP V2 是 #605 的 `ElasticBuffer`，固定上游 main，含 #630 纯 scale-up 初始化、#640 排除 NCCL 共享内存映射的库匹配、#642 LL combine fence。scale-up 必须证明 NCCL Device API LSA team 覆盖全 EP world；x86 EP16 使用 GIN 混合路径，两个逻辑 scale-out 域对应两个物理 RDMA rank，每域八个 scale-up rank。GB EP16 使用 LSA。

MoRI EP8 在所有 CDNA SKU 使用 `IntraNode`。mi355x EP16 用 Pollara RoCE 上的 `InterNodeV1` 和默认 `STATIC_HEAP`。此前须移除两项缺陷：`combine()` 应接收本 rank 路由而非 dispatch 的 recv-slot 索引（ROCm/mori#475、#546）；强制 `MORI_SHMEM_MODE=VMM_HEAP` 会在 ROCm 7.2 使请求 uncached 的 VMM 内存落入 cached 池，跨节点部分和读到旧 cache line（ROCm/mori#610），T≈64 起随机损坏，预填充几乎必现。同分支 A/B：STATIC_HEAP 在运行 34939333022 的 BF16/FP8 全梯度通过，VMM_HEAP 在 34941185534 复现。适配器现在拒绝其他堆模式。TW 无跨节点 GPU 网络，EP16 不支持。mi355x UCCL-EP EP16 仅约 6 GB/s，远低于上游的 82 GB/s，DMA-BUF/peer-memory 或 traffic class 都不改变，怀疑 ionic 驱动；b200-nscale 的裸机 IB 上 UCCL-EP EP16 已登记且运行正常。

MoRI 使用 MANUAL 和固定配置，与引擎一致：vLLM/SGLang 都不设 `MORI_EP_LAUNCH_CONFIG_MODE`，节点内 dispatch/combine 都为 block_num 80、rdma_block_num 0、`warp_num_per_block` 16，不传逐调用覆盖；使用默认外部输入缓冲，SGLang 明确设置它。两个选择必须配套：MoRI 调优表按 `zero_copy` 区分，外部输入约 16 warps，注册缓冲约 4–8。在 MI300X/MI325X/MI355X，注册缓冲下 16 相比 8 warps：T=128 合并慢 13%–18%，T=512 慢 61%–78%；引擎实际使用的外部输入下，16 warps 在 T=128 快 14%–19%、T=256 快 26%–27%，所有预填充梯度快 9%–14%，仅 T<32 时 8 warps 快 0.2–2.5µs。所有组合正确。

外部输入由内核按接收数量自行复制，BF16 直接传递 dispatch 输出，`stage` 标为不可用（null 分位数、零样本）；FP8 仍需反量化。这里测的是引擎配置，不是 MoRI 峰值。按 shape 调块数/warp 数可更快，但无引擎选择，AUTO 也不能一致复现：gfx950 没有 IntraNodeLL combine 表和 normal IntraNode dispatch BF16 规则，AUTO 对这两项使用默认、另两项调优，结果依赖 MoRI 版本。MI355X 把注册模式排除的 BF16 stage 加回后，注册缓冲/8 warps 仍在 T=512 快 15%、T=8192 快 8%；gfx942 未复现，因此未部署配置距离峰值的诚实范围为 0%–15%。本文所述 LL 配置沿用 normal 元组；SGLang 的低延迟为 8-warps `AsyncLL`，而这里记录的 `IntraNodeLL` 单调用方式与其不同，不能视为引擎先例。

UCCL-EP 保留旧版 DeepEP `Buffer` API 和无权 rank 求和，CPU 代理以普通 `libibverbs` 发起 GPUDirect RDMA，不使用 NVSHMEM/IBGDA，软件处理顺序、原子操作、流控。scale-up 为单物理节点 NVLink/XGMI `cudaIpc`，从不使用 MNNVL；EP16 使用各 SKU 同样的 RDMA 网络。

NCCL EP 是 NVIDIA 的 NCCL Device API 原生 MOE 通信。v0.2 起 `nccl-extensions` 提供 `nccl.ep`，固定的 `nccl4py` 提供 `nccl.core`。normal 的 `HIGH_THROUGHPUT` 使用 FLAT `[N, hidden]` 接收和无权 rank 求和，与 `layout-and-dispatch-v1` 完全匹配。仅 NVIDIA/CUDA 13，六个 NVIDIA SKU 均支持 EP8/EP16；GB 为 MNNVL，H100/H200/B200/B300 为 2x8 GIN scale-out。v0.1 GIN 在 x86 RoCE/IB 的 `nccl_ep.cc` 同样失败；v0.2 在 B200/H100/H200 全解码/预填充梯度实机通过，B300 按同一修复启用，待该池复验。

FlashInfer EP 是 TensorRT-LLM 单边 MNNVL `MoeAlltoAll`，直接写对端 workspace，合并读回，无 send/recv 配对或 NVSHMEM，因此仅 GB200/GB300 EP8/EP16。0.6.15 及之前 top-k 使用载荷 dtype 的展开配对树，每层 BF16 舍入；oracle 精确复现树，而非放宽容差。0.6.16 改为 FP32，适配器按安装版本选模型。normal 覆盖完整 token 梯度。其 FP8 可运行却不在任何已部署路径：vLLM 此传输只接受 nvfp4/mxfp8/bf16，`OFF_PATH_PRECISIONS` 将 FP8 排除在默认矩阵之外；显式 `--precisions fp8` 可用于与 DeepEP/UCCL 匹配字节/块大小的比较。这是精度筛选唯一会增加行的情况。

`low-latency` 增加各后端解码内核。DeepEP 使用旧版 `deep_ep.Buffer` LL API，按专家填充接收，源端按 top-k 权重合并；EP8 使用 `allow_nvlink_for_low_latency_mode`，NVSHMEM/IBGDA 和 `/dev/gdrdrv` 仅在 EP16 scale-out 承载线上载荷，H200 无 gdrdrv 的 EP8 已验证。本文该模式段落记录的 MoRI `IntraNodeLL` 是单调用纯节点内解码内核，保留紧凑、按 rank 去重布局和无权 rank 求和；分阶段 `AsyncLL` 不符合该单调用约定。LL 仅解码，依据 `ll_backends` 逐组合启用，不能由 normal 支持情况推断。

DeepEP V2 LL 启用 H100/H200 EP8，B200 nscale 裸机和 GB200/GB300 EP8/EP16；MoRI 为 MI300X/MI325X/MI355X EP8；UCCL-EP 为 H100/H200/B200 EP8，传 `is_intranode` 后使用 `cudaIpc`、不启动代理。AMD LL 因上游在 pin 前六天把 `kNumMaxTopK` 从 9 改 16，host assert 在 16 warp groups 上无法成立而关闭。NCCL EP 为 H100/H200/B300 EP8 和 B200/GB200/GB300 EP8/EP16，`LOW_LATENCY` 使用 EXPERT_MAJOR 接收、源端加权合并。

NCCL v0.1 复用了 DeepEP 修复前代码，缺 #642 的 `fence.proxy.async.shared::cta`；GB300 EP8 T=256 五次中一次错误 0.47（正常 0.0039），各梯度都有竞态。临时 T≤128 只减少暴露，并非安全边界，绿色截断结果也不可发布。v0.2 在 `ll_ep.cuh` 的 `emptyBarriers` arrive 前加入 `fence_view_async_shared`，满足恢复条件，启用完整缓冲梯度，仍由 oracle 把关。更早的旧 peer signal 卡死 [NVIDIA/nccl#2303](https://github.com/NVIDIA/nccl/issues/2303) 已由单 handle 修复。本文记录 B300 的 NCCL EP 为 `candidate` LL 覆盖。是否尝试由能力决定，是否成功以产物为准。

## 工作负载身份

按 `configs/sweep.json` 中的 seed 生成全局 token 批次，再按源 rank 切片。seed 属于工作负载身份，写入每个用例。带密钥的 BLAKE2b 计数器以 `(token, slot, attempt, stream)` 为坐标，跨运行时生成逐字节一致的专家索引和门控权重；测试框架必须证明各 rank 实际路由相同才可成功。

流量区分：

- token-expert assignment，决定专家计算负载。
- 按 rank 去重的 token 载荷副本，决定 EP 激活通信量。

适配器不能自行生成路由，也不能把两个量混用。

## 测量

normal 使用 `layout-and-dispatch-v1`，分发包括布局和通信，合并为激活的无权 rank 求和。专家输出 staging 位于独立合并窗口和配对往返窗口之外；所有行的 `roundtrip` 都是分发后合并，只有 staging 实际发起设备工作时才报告独立 `stage`。`CX_FP8_CONSUME=dequant` 是刻意将转换放回链内的验证例外。

FP8 的 `stage` 是测试辅助工作，不是推理服务独立阶段：把接收的 FP8 转成 BF16 合并输入，代替生产中原生消费 FP8、输出 BF16 的专家 GEMM。测试衡量 collective 而非完整层，因此 **stage 不可加到总延迟，也不可跨后端比较**。DeepEP/UCCL normal 只转有效行，LL 转整个 `[experts, cap * ranks, hidden]` 填充平面；MoRI 只转有效行，FlashInfer 只转已填槽。生产中量化格式不匹配的 fallback 才会单独反量化，例如 vLLM 的 `block_k` 不匹配 DeepEP 块大小；dequant 入口模拟此情况，不是默认快路径。

`implementation.stage_excluded_from_roundtrip` 表示“存在设备 staging 且已外提”。`false` 不等于往返包含 staging：`stage` 不存在时只是指针传递（NCCL EP 和直接把接收缓冲交给 combine 的 BF16 行）；`stage` 存在且该值为 false，才表示 dequant 将转换放回链内。仅凭 false 去减 stage 会减掉从未支付的成本。每个组件声明可用性、来源和样本数；仅支持配对的 API 独立组件为 null，`isolated_sum` 为推导值。

主延迟为 `components.pair_period`；旧行缺该字段时为逐迭代跨 rank MAX 的 roundtrip p99。主指标切换曾等待六事件链缺陷修复，2026-08-06 在手工参考与双链全平台产物核对后放开（31092783122、31089556516）。两种口径均输出 p50/p99，汇总也都展示。fresh-entry 使用 MAX，因为层要等最慢 rank 完成；它也把进入错位计入对应组件。相同 H200 LL 解码中，DeepEP/UCCL BF16 每迭代跨度约 9.3µs，NCCL 约 2.6µs；前两者 FP8 内核量化较重、自然对齐，跨度降到约 2.8µs。没有合理方法统一减掉这一项，MAX 对不同路径的额外成本并不相同。

因此每行还输出 `cross_rank_min_us`（MIN，排除错位的下限）和 `cross_rank_spread_us`（每迭代 MAX−MIN）。MAX/MIN 构成区间；若两个单元 MAX 差距小于较大跨度，数据不足以区分。按 roundtrip p50 排序，只有 MAX/MIN 顺序一致才判优。多节点解码不要按 MAX 的 p99 排名，它往往被最差 rank 停顿主导；旁边 MIN 的 p99 才是同步成本尾部。独立组件继承前一操作退出错位，主要用于残余等待诊断，可比较量是配对往返。

### 链式配对周期

前面的 fresh-entry 在每个窗口前后排空，样本从空闲流水线开始，rank 每次重新错位。解码循环连续运行，支付的是每个 MOE 层的稳态**周期**。`benchmark_chain` 连续提交 dispatch→combine 对，把 CUDA event 入 stream，**循环内不做主机同步**。每点 4 轮，每轮 128 对，丢弃前 16 对填充；梯度与 Pass 2 一样逐轮旋转。配对与 `run_roundtrip` 完全相同：dispatch、已准备的 combine 输入（dequant 模式才内联 stage）、combine；配对 API 保持约定，stage 排除规则与 roundtrip 一致。

每轮有两条配套链，避免统计量承担自身采集成本。最初每对六次 `record()`；在设备比主机提交更快的小 T 区间，事件随提交立即执行，窗口退化为主机耗时，四个内部事件和胶水逻辑混入发布周期。全平台表现为不随 T 变化的 10–30µs，T=1 高出 20%–38%。现在先运行仅含四个操作窗口事件的 floors 链，再运行仅含两外层事件的 period 链，两个 collective 中间无插桩；两个 record 的主机成本落在配对间隙。仍有 eager 启动下限，CUDA graphs 解码每对的主机成本会更低。

两条链只发布五项统计量：

- `components.pair_period`，来源 `chained-median`：period 链逐对周期，跨 rank 取 MEDIAN。排空组件完成成本适用 MAX；周期是速率，collective 将 rank 锁在同一节奏，MAX 会把偶发慢 rank 当作流水线速度。
- `chain_floor_us.dispatch`/`.combine`，来源 `chained-cross-rank-min`：floors 链窗口跨 rank 取 MIN，最晚进入者等待最少，近似操作下限，与 profiler 内核时间约在 10% 内一致。它用于判断通信占比，不要求 `period − Σfloors` 为零。正值表示 MIN 刻意剔除的 rank 间等待；同步主导时可与 floor 和相当且不随 T 变化。GB200 FlashInfer normal 解码在运行 31089556516 各梯度约为 BF16 70µs/FP8 100µs，到 T=512/预填充随着工作量增加消失，即 `period = max(sync budget, work)`；此时 pair spread 仅 2.7–8.5µs，而残差 62–107µs，因为周期守恒、等待在窗口间迁移。负值则是 floors 链自身四事件/对约 10–12µs 的主机成本在设备过快时膨胀窗口（同运行的 GB200/H200 FP8 LL），不是重叠，也不是两链收敛差异。大正残差不意味着六事件缺陷复发：旧缺陷同时影响所有厂商/网络，并落入 `interpair_gap_us`；同步残差有后端特异性，间隙仍小。
- `chain_health.pair_spread_us`：逐迭代跨 rank 配对 MAX−MIN，证明中位数是否有意义。若相对 pair period 很大，说明有节流或慢 rank，不能视为稳态周期。
- `chain_health.interpair_gap_us`：每轮起点到起点间隔中位数减配对窗口中位数，是发布窗口**外**的每对成本（框架两次 record 和间隙停顿），也是防止六事件问题回归的产物内证据。
- `chain_health.settle_drift_us`：每轮后半段减前半段周期中位数，保留正负，跨 rank 取绝对值最大者。`chain_drop` 假定已收敛，该字段给出证据；未收敛或降频会留下漂移。

**不发布链式单操作中位数或 p99。** 无主机同步时，不同 rank 在不同操作窗口内等待。一个 rank 上数字稳定，但跨 rank 任意：rank 3 的 dispatch 长，可能正对应 rank 5 的 combine 长；相同配置跨运行可双稳态，配对和却守恒。单操作中位数只反映该次等待落在哪里。MIN 才能去掉它，p99 会把同样噪声重新引入尾部。

fresh-entry 的 `roundtrip`、`dispatch`、`combine`、`stage`、`isolated_sum`、`cross_rank_min_us`、`cross_rank_spread_us` 保持原义和 256×8 采样。存量行不重新解释/测量。检查 `components.pair_period` 是否存在；缺失表示早于链式测量，不能在主指标列直接与新行排名。汇总混合两者时有脚注。

**发布周期始终指自由连续运行。** rank 最多漂移约一次迭代，要求接收平面能容忍：后端按 dispatch 双缓冲、严格配对，或每个操作在可复用 handle 上完成。此前未审计的 DeepEP V2 normal 在 2026-08-06、pin `01dc3aaa`、当时 dgxc RoCE/GIN 上手工跑了 T=128、EP8/EP16、两种精度的 256 对无同步链；全部通过、输出有限、输入不变、跨 rank 周期差小于 1µs。链与同步对照：EP8 BF16 105.4/125.4µs，FP8 216.6/272.3µs；EP16 838/863µs 和 820/897µs。不能自由运行的后端应修复，而非增加测量变体：每对重新对齐会加约 10µs，并去掉要测的跨对重叠，不能与自由周期共列。

所有 HT 被测 dispatch 都包含路由工作。NCCL `ncclEpUpdateHandle` 是准备当前 top-k 路由的逐 step collective，生产每层都改变路由，且按 handle 全容量更新。旧版和 NVIDIA `ep_bench` 一样把 update 放在窗口外，理由是引入与梯度最大容量相关的成本；这恰是生产实际承担的成本。现在计入窗口，`kernel_generation` 为 `nccl-ep-v02-ht-routed`（LL 为 `nccl-ep-v02-ll`），v02 区分 `nccl-extensions` mover；早期 `nccl-ep-ht`/`nccl-ep-ht-routed` 属于不同口径或实现。LL update 立即返回，内核在计时 dispatch 内读缓存路由。其他后端已经包含此成本：UCCL dispatch 内调用 `get_dispatch_layout`，DeepEP/MoRI/FlashInfer 每次传路由。

统一计时配置定义在 `configs/sweep.json` 并写入各用例：

- fresh-entry：256×8 = 2048 观测。
- 链式：4×(128−16) = 448 个周期观测，floors 另有 448 个，先 floors 后 period。每次调用已产生 128 对，没必要匹配 fresh-entry 轮数，否则只会增加时长而无收敛收益。
- 每轮、每点、每个可用被测组件及每轮链前，执行 32 次同步完整 dispatch-stage-combine 预热。
- `trial_order` 每轮轮换组件顺序，同时旋转 token 梯度；链式轮次使用同样旋转。
- fresh-entry 每迭代跨 rank MAX 后取 nearest-rank p50/p90/p95/p99；链式按前述 median/min。

`measurement.sampling` 同时记录两部分；仅 sample_count 无法还原，128×4 和 512×1 不是同一测量。

主指标为链式周期，旧行用 roundtrip p99。decode/prefill 表示该 MOE collective 对应的服务区间，相同 shape 不改变计时原语。沿梯度递增，每个 shape 在正确性检查前执行 8 次不计时完整往返，稳定时钟、网络和缓冲；所有 shape 预热、检查完才开始计时，conditioning 不记录为样本。

与厂商表格并非同口径：这里 eager 逐调用包含内核启动和 rank 进入错位，fresh-entry 跨 rank 取 MAX；厂商可能只用 profiler 测内核（DeepEP/UCCL LL）、回放 CUDA graphs（MoRI）、跨 rank 取均值、用 sleep/摊销 barrier 消除错位，或选调优扫描最优值。健康网络上预期本测试主指标约高 5%–10%，可用 `cross_rank_min_us` 对照。相同口径下，MoRI dispatch 为其随库调优最优值的 0.96 倍；B300 DeepEP V2 与公开 8x2 结果差 3% 内；FlashInfer 在八个字节归一点与 NVIDIA 单边内核差 4% 内。

逻辑载荷带宽：

`wire_payload_bytes / measured_latency_seconds`

每行两种字节统计均排除专家元数据、padding、后端容量。`byte_provenance` 是统一可比基准：每个唯一 `(token, dest-rank)` 一份激活。`wire_byte_provenance` 是内核实际搬运量：normal 和 MoRI LL 等按 rank 去重布局与前者一致；DeepEP/UCCL/NCCL LL 在 combine 内按 top-k 加权，每个 `(token, expert)` 一份。专家同处一个 rank 时，去重统计只是后者下限，例如 NCCL LL EP8 T=128 低估 34%。所有 GB/s 都采用 wire 基准，旧产物没有该字段时回退到去重值，只会低估。`logical_copies` 明示 `routed`/`assignments`/`wire`，不静默混用。

BF16 每值 2 字节、无 scales；FP8 每值 1 字节，DeepEP/UCCL/FlashInfer 分块 codec 另携每 128 元素的 FP32 scale（FlashInfer 为第四载荷），MoRI e4m3 cast 无 scales。合并仍 BF16，两个方向字节可不同，往返为逐字段相加。直接测得的延迟不受统计口径影响。没有原语模型或传输计数器时，不发布算法/总线带宽、线速/物理链路利用率；逻辑带宽不能标为物理带宽。`rate_at_latency_percentile` 是字节/token 除以对应延迟分位数；p99 延迟对应低尾服务速率，不是逆速率分布的 p99。

## 正确性

独立 oracle 使用专家特异的确定性变换，防止错误专家路由也通过 identity 往返。每 rank、每点验证：

1. 目标 rank/专家、源 token、重数、门控权重和接收数量。
2. 计时前的分发载荷及元数据。
3. 计时前的合并输出。
4. 全部计时样本期间语义输入不变。
5. 计时后再次检查分发载荷/元数据和合并输出。
6. 每轮**自由链自身的最终合并输出**与相同 dispatch→combine 路径的排空配对比较。
7. 对自由链留下的状态再做一次 2–5 的完整检查。

2–5 只检查排空调用，若没有 6/7，一个排空正确、连续执行损坏的后端可能看起来最快。6 是运行方式 A/B，不是 oracle：每轮最后同步后、所有计时窗口外，以 oracle 容差逐元素比较链末输出和新排空配对，不要求位相同，因为 combine 不保证归约顺序确定。7 在最后一轮后对同步完成的 communicator 重跑完整 oracle，检查链是否破坏后续状态。链内部每对覆盖前一对输出，若保留或归约所有输出，会在计时内增加设备工作或约 O(iters×T×hidden) 内存；因此只出现在中间对、最终恢复的缺陷不在证据范围内。6 记录 `correctness.chain_last_output_passed`（跨轮 AND）；7 为 `correctness.post_chain_state_passed`，并入 `correctness.passed`，失败即用例失败。

6 **仅在每对执行 staging 时作为门禁**，否则为 null。这一边界经过测量。`stage_excluded_from_roundtrip` 成立时（默认所有 FP8，`stage_device_work` 即 FP8 标志），链把一个预热 dispatch 的替代输入复用于全部 128 对。链末合并和排空参考都不消费自身 dispatch 对应输入，因此不可要求一致。

H100/DeepEP V2/EP8 仅改变是否外提的 A/B：

| staging | `chain_last_output_error` | 相对 `COMBINE_REL_TOL` |
|---|--:|--:|
| 外提（`fp8_consume=native`） | 31–93 | **1000×–2966×** |
| 每对执行（`fp8_consume=dequant`） | 所有梯度 `0.0` | 逐位相同 |

normal 和 LL 均如此（31180411148、31185184372、31185233991），BF16 不外提，对照误差 0。差异由刻意外提导致，并非传输损坏。把链的 staged 输入交给排空对也无效：共享输入仍不匹配两次 dispatch，只有逐对 staging 才可比较。门禁在所有 BF16 和 dequant FP8 行保留；null 表示没有提出该比较问题，不是比较失败。若改变边界，应依据误差量级，不能只看 verdict；此前一天内仅看 verdict 导致两次错误判断，量级探针各一次便澄清。

null 留下的缺口是三条件交集：只在自由连续执行时出现、不给后续留下状态、BF16 对照不走该路径。少一个条件仍会触发门禁。DeepEP LL T=256 缺陷影响两精度，BF16 今天会失败；卡死由用例超时保护捕获。外提链也不消费自身 FP8 接收，因此 dispatch 和 combine 都有暴露面。`chain_health` 证明一致性，不证明工作完成；稳定缩短 combine 的缺陷仍可能发布快周期。`CX_FP8_CONSUME=dequant` 逐对消费自身 dispatch，恢复 FP8 检查；FP8 周期变化而 BF16 没有同向变化、以及首次发布新 FP8 后端前，应运行此探针。FlashInfer BF16 也外提 workspace staging，所有精度均 null，dequant 不覆盖它，此处仍待后续补足。

6 还报告跨 rank MAX 的最差相对误差 `correctness.chain_last_output_error`，**无论 verdict 成败都报告**，应先看量级。假定真正损坏远超 `COMBINE_REL_TOL`，普通内核非确定性远低于它；这并非免费假设。FlashInfer 的 BF16 slot-tree 累加比 FP32 粗糙，单个 verdict 无法区分传输损坏和容差过严，两者应对相反，误差量级才是依据。缺量级的失败不应单独当作损坏证据。

旧 `chain_regime_passed` 已更名为 `post_chain_state_passed`：链后 oracle 证明的是状态，不是链输出。旧字段 null 表示链未运行；现在启动前必须验证链预算，新产物始终为布尔值。`chain_last_output_passed` 是另一项，staging 外提时合法为 null，包括 dequant 之外所有 FP8。

normal 为激活无权 rank 求和。oracle 在各 rank 先建立门控加权专家聚合，再从实际通信值推导两级归约：目标 rank FP32 聚合转 BF16，与适配器一致；同一 NVLink/MNNVL 域内 FP32 累加，各域再转 BF16 发送 scale-out，最后求和。`ep_size <= scale_up_domain`（所有 EP8 和 MNNVL EP16）只有一域，无跨域舍入；RoCE EP16 每节点一份 BF16 部分和。显式建模域级 cast，使最大逐元素相对误差门限可保持 `< 8 * 2^-8`，分母下限 0.02；漏掉该 cast 时 EP16 误差约 0.048，会越界。

LL 的源端加权合并由内核给每个专家消息乘 top-k 权重，适配器准备未加权专家变换。专用 `(source, expert)` oracle 按 BF16 专家消息的加权和求期望，无域级中间值，因为在源 rank 归约；交付 assignment 多重集和各专家数量与路由核对，使用同样门限。FP8 oracle 在载荷比较和合并期望前执行后端精确逐 token 量化往返，载荷逐位一致，容差不变。量化由模型处理，不靠宽容差吸收；这是正确性门禁，不是通信误差估计。任一 rank/点失败即用例不合格。前后 dispatch 对照规范源 token 元数据及期望输出；原生接收槽可非确定性分配，不要求物理接收顺序固定。

## 结果产物

每个原始文档包含 `record_type: "case-attempt"`、单一 `version`、`generated_at`，以及：

- `identity`：`case_id`、`attempt_ordinal`、`case_factors`（SKU、后端、EP、模式、精度、阶段、suite、工作负载、拓扑）、`allocation_factors`（run id、attempt、源码 SHA）。
- `workload`：`cross_rank_consistent`，是否证明跨 rank 路由一致。
- `measurement`：实际分发/合并 dtype、语义、`payload_unit`（`token-rank`）、`sampling`、逐点 `rows`。合并 BF16，分发 BF16 或 SKU 对应 FP8。
- `implementation`：后端、kernel generation、`maturity`。`production` 表示 vLLM `--all2all-backend` 或 SGLang `--moe-a2a-backend` 可选，`candidate` 表示真实库但没有引擎选择项，数字描述库而非可部署配置；registry 的 `backend_maturity` 相同。另含 `fp8_consume`、`combine_reduction`、`library_version`、`stage_excluded_from_roundtrip`、`chained_period`，分别记录消费模型、oracle 归约/库版本及 staging/链式世代。
- `topology`：SKU/产品、放置、`gpus_per_node`、节点、scale-up 域、scope、topology_class、world size，及 `scale_up_transport`、`scale_out_transport` 两组件和合成的 `transport`（如 `nvlink`、`nvlink-rdma`）。
- `runtime`：vendor、framework（torch 版本）、accelerator_runtime（torch 构建 CUDA/HIP 版本）、collective_library（进程实际加载的 NCCL/RCCL 及版本）。
- `provenance`：挂载镜像 tag、源码 SHA。
- `outcome`：status（success/invalid）和 reasons。

每个 rows 项包含 fresh-entry components、`components.pair_period`、`chain_floor_us`、`chain_health`、字节统计、token 速率、正确性、负载和 fanout；逐点统计原地汇总，不拆成文档。每个实际执行用例只写一份原始结果，不支持/未运行组合不生成合成记录。

## 身份

标识采用可读因子串：

- `case_id`：`{sku}-{backend}-{workload}-{mode}-{phase}-ep{ep}-{routing}-{precision}`，各因子做 slug 规范化。
- `attempt_ordinal`：区分同一 case_id 重复执行的正整数。

后端 pin 在 `runtime/build.py`，拉取后严格比较提交；加载的 DeepEP V2 还检查必需 `ElasticBuffer` API。使用方可据此匹配/区分配置；后端自身不计算 cohort、控制比较、敏感性配对、资格或推荐，展示与比较规则由读取方决定。

## 执行隔离

每个非 MNNVL scale-out 使用操作员固定的 socket/RDMA 选择器；启动器拒绝缺失/不完整配置，在后端初始化前逐节点探测接口、活动 HCA 端口和 GID，不以默认路由、继承环境或传输回退替代。scale-up/MNNVL 清除 profile。NVIDIA scale-out 强制 `NCCL_NET=IB`，AMD 由 RCCL 选插件；两者精确匹配 HCA。`NCCL_IB_MERGE_NICS=0` 防止双端口融合关闭 DeepEP EP16 所需 GIN，rail-isolated 时另设 `NCCL_CROSS_NIC=0`。选择器来自 tracked registry 和可选操作员覆盖，仅出现在 0600 私有日志。

源码暂存到检出/workspace 外、预先存在、归运行器所有、组/其他用户不可写的共享基目录。父进程先解析准确的执行子目录，再复制；所有节点从该树准备后端。清理确认分配结束后仅删除该子目录。DeepEP 源码在申请前按精确版本获取，初始化固定 fmt 子模块并应用所需本地补丁。

H200/B200/B300 可在已验证、计算节点可见的操作系统账户 home 下派生私有基目录；H100 则在共享容器目录旁，而非镜像存储内。规范 B300 忽略旧 `stage_dir`，采用账户 home；UID 映射的 Actions 进程仅在该准确基目录 owner 匹配私有父目录时接受。显式 stage/其他池仍严格要求有效 UID 所有权。执行 ID 后缀隔离并行 worker。当前 NFS 可能将新建基目录映射为 UID 0，仅接受该创建路径，不接受预先存在的 root 所有目录。GB300 同样忽略旧组可写 stage_dir，在已验证 home 下派生执行专用私有基目录。

## 镜像固定与构建隔离

Enroot 按 `(image platform, image reference)` 导入一个 squash，每个集群暂存一次，同镜像的运行复用。启动时提交节点查询 registry manifest digest，与 sidecar 比较；摘要变化意味着 tag 移动，必须重新导入。无法解析摘要（无外网或暂时故障）时复用现有缓存；`refresh_image` 可强制更新，仅丢弃启动前的文件，使并行用例仍只导入一次。每次复用前都用 `unsquashfs -l` 验证。

镜像内 DeepEP 检查准确包版本与 API。源码构建的 V2 使用独立 0700 集群缓存，只挂载为 `/cx-cache`，路径绑定 CPU/GPU 架构、镜像和提交，不作为产物上传。逐执行源码/结果 stage 独立且可丢弃，复用前运行时探测失败即拒绝。运行器 UID 位于可信集群边界；缓存防止过期/意外改动，不防同 UID 恶意作业。仅未发布的半成品可自动重置；完整性或运行检查失败的缓存保留并拒绝使用，避免删掉其他分配正在用的文件。

## 中立产物交付

没有结果服务、附加存储或托管对象存储。每分片一次资源分配，产出逐用例 JSON 和小型机械汇总，以 `always()` 上传 GitHub artifacts，失败/部分执行也上传。成功按基准测试退出码判定；上传前无完整性或隐私验证，失败或不支持组合无合成记录。

流程不晋级运行、不构建数据集、不推进通道。产物就是输出；下游展示和比较由使用方负责。

## 历史数据

历史数字 schema 3–5 不属于当前产物，保留为历史诊断证据，当前扫描既不生成也不读取。
