# DSv4.1 对比实验

[English](README.md) | [中文](README_zh.md)

本实验仅用于研究对比，不作为正式性能提交。历史 SGLang H200 测量按要求对
目标模型和 DSpark 均使用 W4A8；原生 vLLM H200 则选择 Marlin BF16 MoE 激活
和 FP8 Indexer，保留为明确标注条件的基线。所有引擎均使用发布权重与原生
内核，不修改服务引擎。

`experiment.py --mode both --output /logs/research --gpu-count 8` 先运行现有固定长度客户端，
再单独采集 16 步 CPU/GPU 服务 trace，最后在服务引擎空闲时用 GPU 0 测量独立
Engram gate。只有最初未启用 profiler 的测量属于服务性能结果。原始 trace 和算子
JSON 保存在 server-log artifact 中。剖析请求沿用相同聊天格式、精确长度及 DSpark
设置，使用种子 12345，并在预热后清空缓存；同时校验服务端返回的 token 数量。

Engram 配置为 T=1/72/128/512/1024/4096/8192/16384、D=5120、H=4，使用 BF16
激活、FP32 归一化权重、epsilon=1e-20, clamp=1e-6。在 T=512 和 8192 时掩盖奇数行，
并额外测量 T=8192 的无掩码情形。原生 Triton gate 没有掩码输入，因此有掩码配置
包含额外的 `torch.where` 操作；原生 gate 也在内核中相乘 q/k 归一化权重。
与预先合并权重或单内核掩码实现比较时，必须说明这些差异。

每种实现先预热三次，再执行十次目标调用，每次之前在计时范围外运行 256 MiB
FP16 ArgMax。报告 profiler 命名作用域内 GPU 内核耗时之和的中位数；CUDA event
耗时包含间隙和 profiler 开销，须单独保留。保存全部样本与 trace。正确性检查采用
BF16 容差与 PyTorch 公式对照，并要求掩码行与输入逐位一致；报告实际误差，该检查
不等同于模型精度评测。算子计时期间服务模型仍驻留显存。

更完整的实验清单包括 32 GPU 长上下文服务、完整 MoE、Attention/Indexer
前处理及后处理、稠密和稀疏 Indexer、Sparse MLA和 Engram hash。
报告跨平台比值前，必须记录形状、精度、计时边界、缓存协议、预热、重复次数和统计
口径。特定架构的流水线计数器不能直接相互替代。

## 减少芯片数量的服务对比与剖析产物

主要对比使用四张 B200/B300 测量单请求 8K，八张测量 128K、全局 batch
384/1536/2560。长上下文客户端先单独预热 32 个输出 token 并清空缓存，再提交一批
互不重复的完整并发请求。发送已渲染聊天模板的 token ID，校验服务端实际长度，
并保留每个流式 token 计数的时间戳。去掉开头和末尾各八个 decode chunk 后，所有
请求必须存在共同的内部 decode 窗口，否则判定对比无效。普通客户端指标、稳定流式
指标和模型计时分别保留；禁用 prefill/decode 交替，使排队的前缀先完成。第二批复用
缓存前缀并生成 1024 个 token，待所有请求至少生成 64 个 token 后采集八步剖析；
该批不作为性能结果。

CI 现在直接发布含研究 trace 与数据的 `profiles_*` artifact。旧 trace 仍可从
server-log tar 包或结果报告中的 release 下载。`analyze_trace.py` 根据 CUDA launch
关联信息归属模型阶段，同时保留重叠耗时之和与区间并集。

## vLLM 生产版 Engram 门控

`vllm_engram.py --output <directory> --image <image> --kernel-sha256 <hash>`
直接导入已安装的 `_fused_engram_post_wkv_kernel`，沿用生产路径的网格、步长、
block 大小和 warp 数。运行前校验内核源码哈希，不修改或复制引擎内核。
测量范围仅为 post-WKV 门控，不含 embedding 查表、WKV 投影和输出分配。
BF16 归一化权重与生产路径一致；另测 FP32 权重以保留此前独立测试的数据类型。
每组包含无掩码、全有效掩码，以及 T=512/8192 时的奇数行掩码，掩码在内核内部处理。
沿用 ArgMax 缓存清理协议，预热三次后测量十次。每个用例检查参考公式、掩码行逐位
保持不变，以及十次生产内核事件。保存原始 trace、全部计时、误差和软件版本。
这些测量不替代原有 SGLang 数据，也不是完整的服务性能测试。

## vLLM 服务重测

`vllm_serving.py` 先运行常规定长客户端，再启用 vLLM 原生剖析。研究配置使用固定
`ddd6fbca` 镜像、四卡或八卡、与设备数相同的 TP/EP、DSpark 七个草稿 token，
并关闭前缀缓存。接受长度仍由启动器输入。选用 Python API 前端以提供剖析端点。
单独预热后的请求最多记录十六次引擎迭代，然后显式停止并导出剖析。客户端校验
8192/256 token 用量及每个配置 rank 的 GPU 内核 trace。这是新框架的测量，
不会将原有 SGLang 结果改标为 vLLM。

`vllm_cohort.py` 使用原生 completion token ID 增量及
`X-data-parallel-rank` 请求头。每个唯一 prompt 先生成一个 token 以预热，
再单独测量并保持缓存亲和性；必须存在全批次共同解码窗口。初始八卡配置
使用 TP8/EP8、DP1（不是 DP attention）、原生 `uniform_random` 路由及五个草稿。
均匀随机路由不等同于确定性的轮转均衡。剖析单独生成 1024 个输出 token，
并要求每个配置 GPU 都产出 trace。

共享定长 shell 客户端支持 `FRAMEWORK=vllm`，并选择已有 completion 后端。
shell 集成检查通过实际执行脚本验证客户端参数，不需要 GPU 或实际运行 pip。

## vLLM 原生稠密 Indexer

`vllm_indexer.py` 导入 vLLM 的 DeepGEMM 包装层、原生 TopK 分派器及候选选择器。
输入为 B12、六个 query、32 heads、D128、page128、TopK512，可选输出 2048 个
候选块，每块八个位置。使用 `--physical-lengths` 设置物理压缩 K=64K/128K；
源集成传入的长度已压缩。不传该标志可复现早期半长度诊断，两者必须分开。
源测试 fixture 未公开，因此不能据此宣称直接硬件加速比。MXFP4 输入为独立随机的可表示编码，
E8M0 scale 为 1，head weight 为 1/32，原生稠密权重使用 FP32。逐 query 校验
分数和 TopK 阈值；候选选择计入耗时，但未独立校验其输出。按正序、逆序、正序
执行三轮，每轮预热三次、剖析 1000 次。报告 GPU 内核耗时之和的均值，不含
调度元数据及输入准备，保留全部原始 trace 与样本。

`vllm_sparse_indexer.py` 测量原生分页稀疏流水线，包括候选展开及排序、调度构建、
稀疏 logits、DeepSelect TopK 和逻辑索引映射。使用 72 个 query 行，每行 2048
个独立且不重复的候选块，每块八个位置；通过 `--physical-kv-tokens` 明确物理 K，
例如 131072，或早期诊断的 65536。head weight 为 BF16。
预热三次后测量三次，保留全部样本。分数按 BF16 容差 `rtol=atol=0.02` 校验；
原生 TopK 必须返回不重复的候选位置，且分数不低于原生 TopK 阈值。
`analyze_trace.py` 也识别 vLLM 原生 `execute_` 标记，将关联 GPU 时间跨度标记为
`VLLM_EXECUTE`，不假定其包含所有草稿或采样工作，也不将其等同于客户端 TPOT。

原生 DP8/EP8 变体使用共享 CPU Engram 存储，每个 DP 副本内部为 TP1。
客户端在前缀预热和测量间保持固定 DP rank；图捕获和序列预算按逐 DP batch 设置。
DP 变体保留全局 batch 384/1536/2560，TP8 长上下文基线仅保留 batch384。

`vllm_cohort_sweep.py --concurrencies 384 1536 2560` 复用按每个 DP 副本
320 个序列配置的 DP8/EP8 服务。每个用例使用独立客户端进程，分别保存原始结果、
事件和剖析文件清单。仅首个用例使用标准 CI 结果路径，全部结果同时复制到
`research/cohort-sweep`。用例失败即停止后续执行，并保留已有结果。剖析校验仅统计
该波新增文件，不覆盖前面用例的 trace。启动和前缀预热不计入稳态解码计时。
这是研究扫描，不是三个独立配置的正式 CI 矩阵测量。

长上下文扫描通过 vLLM 原生设置使用 95% 的设备内存。每个用例开始前，驱动
读取所有 DP 引擎在所配置长上下文下的启动日志 KV 容量估计。仅当忽略输出
token 后的乐观容量上界仍低于所需逐 DP batch 时跳过，并记录为
`not_run_capacity_bound`，绝不填入计时结果。若缺少任一 rank 的容量证据，
仍由正常的实测窗口校验决定有效性。通过容量估计不代表该 batch 必然可运行。

合并长上下文扫描使用 ISL=131072、OSL=1024；这是在源表未公布输出长度时
明确选择的协议。更长输出为 HTTP 请求入队后留出全批次内部测量窗口。
单请求测试仍为 8192/256，每个用例都校验精确 token 数。

流式客户端请求连续 usage 计数，不请求 token ID 回显。vLLM 的 token ID 响应
会在首块回显完整 prompt，给 128K 大批次带来大量额外流量。完成 token 数
仍使用服务端原生累计计数，最终 usage 块不会重复计数。

长上下文配置在昂贵的前缀预热前，先对八个 rank 执行 64 输入/128 输出的协议
探测，验证原生流式 usage 和 DP 路由，保存记录并在失败时提前退出。探测计时
不作为性能结果。

## 明确的混合 KV 布局

FlashInfer 稀疏 MLA 基线在两个 KV 池中均使用普通 FP8 行。更接近对应缓存位宽
的变体显式选择 `FLASHMLA_MEGA_ATTN_DSV41` 和
`kv-cache-dtype: nvfp4_ds_mla`：窗口记录使用 MXFP8 group32/E8M0 scale
（D512 时为 528 字节），压缩记录使用 NVFP4 group16/E4M3 scale（288 字节）。
这些 scale 格式与 BF16 scale 记录不同；数据位宽相同不代表数值完全一致。
原 FP8-KV 结果仍保留为标明条件的基线。混合变体用于 B200/B300，Hopper
原生路径支持的精度和布局不同。

Indexer 新测量使用 `--physical-lengths` 明确设置物理压缩 K=64K/128K；源集成
传入的长度已经压缩。原半长度模式保留用于复现早期诊断，不与新数据混用。
稀疏脚本必须传入 `--physical-kv-tokens`，例如 131072 或早期诊断的 65536。
清单记录最小及最大可见物理 K。稠密输入权重和分数为 FP32；源文档描述的是
BF16 中间舍入及 head reduction，不应据此认定 API 输入权重就是 BF16。

trace 分类将 Mega Attention 内核单列为 `fused_attention_rope_cast`，因为它们
同时包含 Attention、RoPE 和输出量化。分类之间的耗时转移涉及融合边界变化，
不能直接当作延迟收益。

长上下文入队现在通过原生 `/pause?mode=keep&clear_cache=false` 和 `/resume`
控制。先编码请求体，收到所有 HTTP 响应头后再等待明确配置的五秒 IPC 稳定期，
随后释放生成；暂停期间任何请求都不得推进。这是 HTTP 入队屏障，不是核心队列
状态的证明，仍必须通过共同测量窗口及逐 worker 剖析 batch 校验。使用八个 API
进程。前缀预热每个请求生成 64 个 token，同时预热 prefill 和 decode。协议探测
在每个 DP rank 上使用 64 输入/128 输出，并要求至少八个进度块。客户端 TTFT
包含人为暂停，不作为在线延迟基准；稳态解码计时排除入队过程。剖析校验要求每个
DP/TP worker 都有真实 GPU 内核和目标生成 batch，观测值保存在
`profile-validation.json`。

## 原生 BF16 Sparse MLA

`vllm_sparse_mla.py` 调用已安装的生产内核 `flash_mla_sparse_fwd`。
输入为 4096 个 query、64 个 head、D512，以及 8192 行原始和 2048 行压缩
BF16 KV，每个 query 选择 128 个窗口/原始条目和 512 个压缩条目。
因果及无限制索引样例是明确限定的对应实验，不复现未公开的源索引分布。
计时不包含索引构造、KV gather 和反量化。Sink logits 为零，原生 LSE 不包含 sink。
先将所有 query/head 与独立 FP32 参考计算核对，再执行三轮测量，
每轮预热 3 次、分析 300 次调用。保留 GPU 内核时长、标注跨度和原始 trace；
这些数据属于重复相同输入的热态测量。

每个请求建立新的 HTTP 连接。下一批大请求体的编码可能超过服务端 keepalive
时限，因此不跨测量阶段复用闲置控制连接。该修改仅影响客户端传输，
不改变引擎执行或计时校验。

## 原生 Engram 哈希

`vllm_engram_hash.py` 在 8192/16384 token 下测量已安装的
`NgramHashState.forward` V2 路径。采用发布模型的 bucket 布局与明确的
恒等压缩 token 映射，计算第 1/14 层、2/3/4-gram 和八个 head。
单个 prefill 请求使用确定性 token ID，没有 dead token 或外部历史。
计时排除 tokenizer 归一化、初始化和 embedding 查询。原生调用包含 token
历史解析，与接受预构造 n-gram 窗口的参考边界不同。所有输出整数均与独立
标量计算逐一校验；三轮各保留 300 个内核时长样本及 trace。
原生 gate 内核已包含权重转换和乘积计算，不额外添加独立权重准备时长。

## 原生 Indexer K 前处理

`vllm_indexer_k_prologue.py` 在 T=72、RoPE 维度64下，测量生产
`ReplicatedLinear` 的 BF16 512→128 投影以及 `indexer_k_norm_rope_store`。
使用正常单 rank vLLM 上下文，测试 page64/128 与跨步 page64 存储。
所有 GPU 测量原生 FP8，Blackwell 额外测量 MXFP4。压缩比1保证每行
输出一个 key；不计入 compressor 生成。缓存采用原生数值/缩放分区布局，
不冒充参考实现的 mode0/mode1 变体。BF16 投影与 FP64 累加核对，
精确校验缩放值、检查所有反量化结果误差界及跨步存储保护区。
十组相邻预热/目标调用保留内核时长之和的中位数、GPU scope 跨度及原始
trace。此协议不清空缓存。

## 原生 Attention 后处理

`vllm_attention_epilogue.py` 调用生产 `deep_gemm_fp8_o_proj`，配合
V4.1 量化配置的 `ColumnParallelLinear`/`RowParallelLinear`。
输入 X[T,64,512]，八组4096→1024，再进行8192→5120投影，
T=1/16/72/128/160/192/224/256。记录原生加载后的内核选择与权重类型，
不强制后端。受控权重样例中 E8M0 字节127代表缩放1.0。
计时前使用独立 FP32/BF16 参考计算检查有限输出及相对 L2 误差小于0.08，
保留实测误差，不声称逐位一致。每个 T 测量192次；每次前执行256MiB
FP16 ReduceSum（FP32累加），不计入目标时长。保留 GPU 内核时长之和
与 eager scope 跨度。Mega Attention 将旋转/转换融合进 attention，计时边界不同。

长上下文 sweep 显式启用原生 scale-out API，并在 benchmark 容器设置
`VLLM_COHORT_TOKEN_API=1`。向 `/inference/v1/generate` 提交 token ID，
使用 `sampling_params.detokenize=false`、连续 usage 计数，不回显 prompt token。
这样将 token 生成与文本反分词分开，保持采样、长度、DP 路由、入队和全 batch
校验不变。其他调用者继续使用 completion 端点。本地 HTTP 行为测试检查原生
请求及流式计数，仍要求执行八 rank 运行时探针。这是诊断性协议修改，并非已证明
之前的合并输出由反分词导致。

## 原生 Indexer QW 前处理

`vllm_indexer_qw_prologue.py` 在 T=72/128/4096/8192、H5120、R1280、
32 head、D128、RoPE64下，运行生产量化 query 投影、BF16评分权重投影
及 `fused_indexer_q_rope_quant`。Query 投影使用 V4.1 量化配置；
Blackwell 输出 MXFP4 query，Hopper 输出 FP8。FP32 权重输出对应
dense indexer 接口；缩放因子为1/sqrt128和1/sqrt32。
包含输入 QR 量化，与已量化 QR 起点不同；原生 BF16 中间舍入也与 FP32
直接传递不同。所有行先进行独立投影与后处理校验，再分析20次调用。
同时发布最小值、中位数、eager 跨度与误差，不推断源样例完全一致。

## 原生 Attention 前处理

`vllm_attention_prologue.py` 测量合并 QA/KV 投影、原生 Q/KV 归一化
（支持时融合 QR 量化）、QB 投影及分页缓存 Q/KV RoPE/写入。
形状T72/H5120/R1280/heads64/D512/RoPE64，epsilon1e-6，block128。
Blackwell 使用528字节 MXFP8 记录；Hopper 的584字节记录保留 BF16
RoPE 尾部。全部 Q/KV 行与独立 FP32/BF16 参考核对（相对L2小于0.10），
并检查未写入缓存行。热态及256MiB ArgMax驱逐后各测量十次，保留中位数、
极差/中位数和 eager 跨度。原生后端使用输入量化时包含其成本；参考起点
已量化，KV 使用scale1。Mega Attention 中融合的 Q RoPE 边界不同。

## 完整八 rank MoE

`vllm_complete_moe.py` 使用已安装的 `DeepseekV4MegaMoEExperts` 与原生
`deep_gemm_mega_moe`，随后执行一个 `DeepseekV4MLP` shared expert 并原地
相加，遵循原生串行 shared-expert 顺序。以 torchrun 在八张 GPU 上运行
DP8/EP8/TP1，每个 case 单独建立进程组。保留八个 h×n/expert-count 形状；
明确将 n 解释为 gate/up 拼接宽度（intermediate=n/2），token 数解释为每 rank。
源表中这些定义不充分，必须保留限定说明。

受控样例使用各专家不同、可精确表示的 FP4 权重，六条唯一循环路由和非均匀
归一化路由权重。计时前用闭式 FP64 参考检查全部 rank 的所有输出。
包含一个 shared expert，排除路由选择和权重准备。分别采集20次无 profiler
CUDA-event 时长及20次带 profiler 调用；逐次取最慢 rank，再报告其最小值与
中位数，不能把八个 rank 的时长相加当延迟。保留全部 rank 的 trace。
路由输入为原生 FP8/group128，专家权重为 MXFP4/E8M0 group32，shared MLP
使用原生 MXFP8 后端。

固定镜像的 FlashInfer Mega 适配器无法导入顶层 `deep_gemm`；选择的原生 vLLM
路径直接加载其内置库，不修改镜像。该 Mega 实现需要 SM100 系列硬件，H200
不适用于此 A8W4 Mega 对比；这不意味着 H200 无法运行 MoE。

B200 global1536 的专用后续实验保持原生 token API、混合 KV、DP8/EP8、
输入/输出长度、draft 和接受长度配置不变，将 prefill 预算8192→4096、
每 DP 最大序列320→192、graph cap2048→1152（192×6）。GPU 内存比例仍为0.95。
用于检查为2560批次预留的资源是否不必要地排除了1536。保留原容量界证据，
明确新配置；只有运行时校验才能证明其容纳能力及完整 batch 结果。

B200 global2560 专用点使用 prefill 预算2048、每 DP320序列及 graph cap1920，
内存比例保持0.95。完整 decode 步骤需要320×6=1920 token，仍在调度预算内。
原生滑窗入队容量取决于最大在途 token 数，因此缩小 prefill 预留可显著改变
容量。保留之前特定配置的容量界，不称其为普遍硬件极限。仍必须通过真实完整
批次测量及全部 worker 的 profile 校验。

B300 global2560 后续实验使用2048输出 token、最大模型长度133376。
1024输出的尝试虽全部完成，但按要求裁剪后没有共同区间（原始重叠7.784秒，
裁剪后−3.623秒）。保留全部校验，通过延长生成提供观察时间，不放宽验证。
源报告未公开输出长度，必须明确此实验选择。Profile 波次改为
max(1024, 实测输出长度)，避免长输出案例的 profile 暗中缩回较短长度。
现有1024输出案例的协议和结果保持不变。


## 测量区间代表性复核

复核配置在 B200/B300 的 global2560 使用相同2048输出及2048/320/1920 prefill/sequence/graph预算。B300 global1536 与 B200 使用相同4096/192/1152预算及1024输出。保留抢占和 KV-cache 指标，记录整个区间及四个固定子区间的逐请求进度。请求完成及独立 profile 通过不代表稳态性能成立；新结果须经过代表性复核后才可发布。


B200 TP4 global384 profile 使用四张 GPU、TP4/EP4/DP1、131072/2048、prefill4096、max-sequences384、graph cap2304，采用原生混合 KV 并校验四个 rank。此配置不是 DP4。


失败复测：B200 global2560 在2048输出下裁剪区间无交集，改为4096输出，单独标记为诊断结果。TP4预填充降至2304（384×6验证宽度），使用单 API worker；DP1省略路由头，admission失败保留原始请求错误。修改仍需运行验证，区间和profile校验不变。


修正原生 DSpark 预算：五个草稿需每请求六个验证 token 加四个额外输入槽位。DP320使用 input3200/scheduled1920；TP4 batch384使用 input3840/scheduled2304。原2048/2304输入预算仅覆盖验证，不能覆盖草稿。TP4必须设置 `enable-scale-out: true` 注册 token 端点，此开关与 DP 数量无关。
