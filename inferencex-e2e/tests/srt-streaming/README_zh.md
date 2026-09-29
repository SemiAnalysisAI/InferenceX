# SRT 流式传输冒烟测试

[English](README.md) | **中文**

在 B300 上测试 NVIDIA/srt-slurm#539，不加载模型。仅运行服务的 SRT 作业连续
90 秒输出 stdout/stderr，同时由 Tachometer 采集真实的主机进程指标。测试使用
一个节点，Slurm 时间上限为十分钟。

工作流从指定的 SRT 提交构建 Tachometer，确保上传器使用同一提交中的原子 Arrow
写入实现，不下载已发布的 Tachometer 二进制。运行产物保存二进制校验和及源码提交。

部署 Dash 收集 API 后，为此 PR 添加 `srt-streaming-test` 标签。构建在 GitHub
托管机器上执行，冒烟测试通过 B300 调度器运行。在推送无关更改前移除该标签；带有
此标签的推送会再次运行测试。现有仓库 Secret `SRT_STATUS_ENDPOINT` 和
`SRTCTL_STATUS_TOKEN` 提供共用收集端点和 bearer token。

打开 InferenceX Dash 的 `/srt` 页面，选择工作流输出的 `b300-dsxe` 作业 ID。
运行期间检查实时日志标记，完成后检查 Tachometer 数据。产物检查验证本地输出；
还必须在 Dash 中确认数据成功送达。此测试不发布基准测试结果。
