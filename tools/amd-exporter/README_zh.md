# AMD 功耗 exporter 镜像

[English](README.md) | **中文**

仅使用 CPU 的 `AMD exporter image` 工作流构建可获取新读数的 exporter，并向 `ghcr.io/semianalysisai/amd-device-metrics-exporter` 发布唯一标签。它不启动 GPU 基准测试，也不修改生产数据。

基础镜像来自官方 ROCm nightly 工作流 `36867107435`，源码为 `9d0eb8c88af99f1dbe2914d803382092671456d9`。使用前校验归档 SHA256 和镜像配置摘要。该运行时包含上游 255 W 哨兵值修复。`cache.patch` 基于 [functionstackx/device-metrics-exporter#1](https://github.com/functionstackx/device-metrics-exporter/pull/1) 的可配置 GPUGet 缓存修改，并补充默认缓存回归测试和零缓存并发测试。

只替换 `/home/amd/bin/server`，保留原有 ROCm、GPUAgent 镜像层及入口。`AMD_GPU_GET_CACHE_TTL=0s` 强制每次 exporter 请求发起新的 GPUGet RPC。这消除了 exporter 响应复用；硬件实际刷新率、采集开销和基准测试中的一秒采集仍需实机验证。

工作流校验补丁摘要，运行缓存竞态测试，构建 Linux/amd64 二进制，检查版本与非法配置拒绝行为，并验证基础镜像层及运行时配置未变。随后发布提交专属标签，生成 Enroot squash，并将 `SHA256SUMS`、`provenance.json` 保存为 `amd-dme-cache-squash` 产物。使用方应记录发布的注册表摘要及 squash 校验值。源码提交、补丁哈希、GPUAgent 提交和工作流版本分别记录；二进制中的 ROCm 提交明确标记为未知。

在已审阅分支手动触发，或推送专用分支 `chore/powerx-amd-exporter-image`。CPU 作业上限为 30 分钟，需要 `packages: write` 权限，不移动 `latest` 标签。镜像构建通过只代表准备完成，不代表 AMD 功耗测量已验收。
