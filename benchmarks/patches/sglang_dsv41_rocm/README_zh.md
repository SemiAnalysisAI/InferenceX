# DeepSeek V4.1 ROCm 注意力后端移植

[English](README.md) | **中文**

此独立 Python 包将官方预览镜像中支持压缩比 1/2 的 ROCm 注意力实现移植到固定的
9 月 21 日 nightly 镜像。原版 nightly 在图捕获阶段报错
`No indexer pool for compression ratio 4`。仅 HIP `deepseek_v41` 注册路径使用
此包；其他模型和硬件路径仍使用 nightly 原有实现。

`provenance.json` 记录两个不可变镜像摘要、原始文件哈希和适配后文件哈希。
预览镜像的构建历史标记源代码覆盖层为 `f8f290f2`，但该提交无法公开解析，因此以
镜像摘要为准。两个镜像均使用 ROCm 7.2.4、AITER
`4ad99832823dde2315b361cbd3b54b1c5c12acd5` 和 Triton
`3.7.0+amd.rocm7.2.0.git89002410`。没有使用预览镜像的二进制替换 nightly 二进制。

适配包括隔离导入、对接 nightly 已迁移的候选索引和图捕获 API，并通过 `get_exec()`
读取内核配置。nightly 在 HIP 上禁用融合低压缩比内核，因此保留其非融合压缩器的
权重布局。nightly 模型在注意力后执行逆 RoPE，不传入预览实现的可选融合逆 RoPE 参数。

`install.py` 在复制文件到任务临时容器前检查注册文件及所有源文件哈希，拒绝不匹配
的注册文件版本，并将安装后注册文件哈希记录到结果目录。必须在导入注意力注册模块前
运行；重复安装结果一致。

初期仅验证 STP。DSpark 原生 FP8 草稿 `wo_a` 精度问题单独修复前，此后端拒绝启动
DSpark。启动成功或有限样本评估通过不代表完整性能与准确性验证通过。
源代码改编自 SGLang，适用随附的 Apache 2.0 许可证。
