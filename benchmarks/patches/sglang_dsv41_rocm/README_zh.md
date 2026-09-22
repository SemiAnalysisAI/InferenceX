# DeepSeek V4.1 ROCm 注意力后端移植

[English](README.md) | **中文**

此独立 Python 包将官方预览镜像中支持压缩比 1/2 的 ROCm 注意力实现移植到固定的
9 月 22 日 ROCm 10 nightly 镜像。9 月 21 日原版 nightly 在图捕获阶段报错
`No indexer pool for compression ratio 4`。仅 HIP `deepseek_v41` 注册路径使用
此包；其他模型和硬件路径仍使用 nightly 原有实现。

`provenance.json` 记录两个不可变镜像摘要、原始文件哈希和适配后文件哈希。
预览镜像的构建历史标记源代码覆盖层为 `f8f290f2`，但该提交无法公开解析，因此以
镜像摘要为准。预览版使用 ROCm 7.2.4 和 Triton `3.7.0+amd.rocm7.2.0.git89002410`；
新 nightly 使用 ROCm 10 和 Triton `3.8.0+git4cff872c.rocm10.0.0`。两个镜像均使用
AITER `4ad99832823dde2315b361cbd3b54b1c5c12acd5`，不使用预览二进制替换 nightly 二进制。
9 月 21 日与 22 日 SGLang 的六个安装器目标文件逐字一致。限定范围的 GPU 回归测试
现已在 9 月 22 日镜像上通过，详见下文；完整精度与性能验证仍未完成。

适配包括隔离导入、对接 nightly 已迁移的候选索引和图捕获 API，并通过 `get_exec()`
读取内核配置。nightly 在 HIP 上禁用融合低压缩比内核，因此保留其非融合压缩器的
权重布局。nightly 模型在注意力后执行逆 RoPE，不传入预览实现的可选融合逆 RoPE 参数。

`install.py` 在复制文件到任务临时容器前检查注册文件及所有源文件哈希，拒绝不匹配
的注册文件版本，并将安装后注册文件哈希记录到结果目录。必须在导入注意力注册模块前
运行；重复安装结果一致。

DSpark 保留固定 nightly 镜像的默认量化行为，不安装自定义草稿权重或投影
精度转换补丁。V4.1 gfx950 的 block-FP8 适配器复用 nightly 已有的 Triton 分组量化
和 GEMM 内核，以避开通用 UE8M0 包装函数错误选择的 CUDA 专用 JIT 头文件。
分组大小 32、E4M3 范围、1e-10 绝对最大值下限和向上取整的二次幂缩放均保持不变。
分发仅针对 HIP 上的 V4.1，其他模型保持原路径。GPU 数值与图捕获测试覆盖此适配器。

ROCm V4 融合 RMSNorm 辅助函数将激活量化分组硬编码为 128。仅对 V4.1，保留其
已有的 BF16 归一化输出，由线性层执行配置中的 32 宽 UE8M0 量化，避免向 WQ_B
传递不兼容的缩放张量。归一化与模型权重不变。`validate_model_norm.py` 覆盖安装后的
辅助函数、线性层、非连续 QKV 切片和图重放。

HIP 注册的主机 Engram 表需要使用 `hipHostGetDevicePointer` 返回的设备地址；
CPU 地址可能不同，直接传给 GPU 内核会触发内存访问错误。安装器只适配主机表的
查找地址，保留 CPU 加载张量、现有 gather 内核、表数据及默认量化。
GPU 常驻表和 CUDA 行为不变。共享表及每 rank 表均已在 MI355X 上通过精确输出
和图重放检查。模型规模的主机表性能仍待验证；配方暂时继续使用 GPU 常驻 Engram。

启动成功或有限样本评估通过不代表完整性能与准确性验证通过。
源代码改编自 SGLang，适用随附的 Apache 2.0 许可证。

V4.1 共享专家 MLP 使用通过小型包装器调用 nightly AITER 掩码激活内核：AITER 融合路径要求宽度按 128 对齐并输出 128 分组缩放，而 TP4 V4.1 共享专家的分区宽度为 576，缩放分组为 32。

配方保留交错的 gate/up 权重，并复现官方预览镜像中的 `AITER_BF16_FP8_MOE_BOUND=0`、模型专用 A8W4 调优 CSV 及 hipBLASLt 偏好。CSV 原样复制并记录精确来源哈希；AITER 源码、二进制及草稿权重加载保持不变。必须先验证这些默认设置下的正确性，再开展性能资格验证。仅 V4.1 移植官方预览版的 FP4 分离缓冲区索引存取方法，这些方法在 nightly 中缺失。现有 HIP 分配布局、FP4 数值及最近偶数舍入保持不变。

V4.1 设置 `q_head_norm=False`。nightly 的 HIP 融合 Q/K 内核始终对 Q
执行 RMS 归一化，而官方预览版会将此模型排除在该优化之外。V4.1 配方设置
`SGLANG_OPT_USE_FUSED_QK_NORM_ROPE=0`，保留模型原有的不对 Q 归一化的
非融合路径。来源记录包含预览版模型、nightly 模型及融合内核的精确源码哈希。

## 9 月 22 日 ROCm10 回归验证

[GPU 证据](evidence-rocm10.json)记录主机指针原始回溯及缓存修复前后的结果。镜像的通用
`find_library` 返回 `libamdhip64.so.5`，PyTorch 实际加载
`/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libamdhip64.so.7`。
加载前者会在 `hipHostGetDevicePointer` 内崩溃，并非清理阶段的问题。适配器现在从进程
已有运行时解析符号；`dladdr` 确认绑定 SDK 库，共享表及每 rank 表的精确输出与图回放均通过。

原始 FlashMLA store 在字节偏移超过 2 GiB 时发生 int32 溢出。GPU 复现测试为错误地址也
保留自有分配空间，在页大小 64/128/256 下确认 FP8 数值和尺度写错位置。
[上游 SGLang #40351](https://github.com/sgl-project/sglang/pull/40351)将 `loc` 扩为 int64；
`wide_store.py` 只提取原 FlashMLA store 并应用同一类型转换，仅 HIP V4.1 缓存选择此路径。
安装后的缓存方法通过九个边界用例的即时执行及图回放，包含填充字节与 BF16 RoPE；其他
模型原有路径也通过验证。这些修复不改变 KV 格式、量化、检查点或草稿精度。

启用 `expandable_segments:True` 时，完整目标图捕获在 AITER 的
`hipIpcGetMemHandle` 图缓冲区注册中报 `invalid argument` 并退出。仅将该分配器设置
改为 `False` 后，目标与 DSpark 启动及三条真实请求通过：输入 52/719/2,159 token，
输出 19/23/25 token，均回答 42 且未达到输出上限。配方采用已验证的原生分配器。
这不代表完整 GSM8K、长上下文内存或性能已通过；旧设置曾用于缓解旧镜像长预填充碎片，
因此仍须明确验证相应工作负载。
