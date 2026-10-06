# AMD 功耗 exporter 镜像

[English](README.md) | **中文**

仅使用 CPU 的 `AMD exporter image` 工作流构建可获取新读数的 exporter，并向 `ghcr.io/semianalysisai/amd-device-metrics-exporter` 发布唯一标签。它不启动 GPU 基准测试，也不修改生产数据。

官方 ROCm nightly 工作流 `36867107435` 的原始产物已被上游删除。现在复用 InferenceX 工作流 `36985488002` 保留的 Enroot 转换产物 `11217112734`，校验 squash SHA256 `0f2cbb4e2c41c1506db74c6653ea09ec5e7d42236d9532e2bc38fb9e0be5879e` 及原始来源回执。其源码为 `9d0eb8c88af99f1dbe2914d803382092671456d9`。该运行时包含上游 255 W 哨兵值修复。`cache.patch` 基于 [functionstackx/device-metrics-exporter#1](https://github.com/functionstackx/device-metrics-exporter/pull/1) 的可配置 GPUGet 缓存修改，并补充默认缓存回归测试和零缓存并发测试。

`base-runtime.json` 从保留的官方 OCI 归档提取原始 Docker 配置，以及 61 个关键路径的文件哈希和符号链接目标。恢复过程先验证 GPUAgent、入口脚本、AMD-SMI、ROCm sysdeps 和配套二进制，再使用原始环境变量、入口、标签及工作目录导入根文件系统。Enroot 已合并镜像层、规范化所有权和权限，并生成 `/etc` 启动文件，因此重建基础镜像使用新标识，不能称为原始 OCI 镜像。之后仅在新增镜像层替换 `/home/amd/bin/server`。`AMD_GPU_GET_CACHE_TTL=0s` 强制每次 exporter 请求发起新的 GPUGet RPC。这消除了 exporter 响应复用；硬件实际刷新率、采集开销和基准测试中的一秒采集仍需实机验证。

工作流校验补丁摘要，运行缓存竞态测试，构建 Linux/amd64 二进制，检查版本与非法配置拒绝行为，并验证重建基础镜像层及运行时配置未变。随后发布提交专属标签，生成 Enroot squash，并将 `SHA256SUMS`、`provenance.json` 保存为 `amd-dme-cache-squash` 产物。使用方应记录发布的注册表摘要及 squash 校验值。`base-recovery.json` 记录恢复工作流和产物、原始来源回执、运行时清单哈希、已验证文件数及重建镜像标识。源码提交、补丁哈希、GPUAgent 提交和工作流版本分别记录；二进制中的 ROCm 提交明确标记为未知。

在已审阅分支手动触发，或推送专用分支 `chore/powerx-amd-exporter-image`。CPU 作业上限为 30 分钟，需要 `packages: write` 权限，不移动 `latest` 标签。镜像构建通过只代表准备完成，不代表 AMD 功耗测量已验收。
