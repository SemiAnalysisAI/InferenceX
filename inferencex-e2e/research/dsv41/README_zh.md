# DSv4.1 对比实验

[English](README.md) | [中文](README_zh.md)

本实验仅用于研究对比，不作为正式性能提交。H200 的 W4A8 激活设置也作用于
DSpark 草稿头，用户已明确要求在本次对比中启用。所有引擎均使用发布权重与原生
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

更完整的实验清单包括 32 GPU 长上下文服务、通信与完整 MoE、Attention/Indexer
前处理及后处理、稠密和稀疏 Indexer、Sparse MLA、Engram hash、单 GPU CPU offload。
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
