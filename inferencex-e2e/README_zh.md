# InferenceX 端到端基准测试

[English](README.md) | **中文**

模型推理服务基准测试、配置、启动器、结果处理工具及性能变更日志均位于此目录。
请从[文档索引](docs/index_zh.md)开始阅读。

在此目录中运行端到端命令：

```bash
cd inferencex-e2e
uv run --locked python -m infx.matrix.generate test-config \
  --config-files configs/nvidia-master.yaml --config-keys <key>
```

Python 项目清单与锁文件保留在仓库根目录，`uv` 会从此目录向上查找它们。
GitHub 工作流、仓库规范及 CODEOWNERS 也保留在仓库根目录。
工作流手动触发时的生成器参数使用相对此目录的路径；历史版本仍使用原有执行目录。

本目录保留一份 `LICENSE`，以便仅挂载此项目目录的容器仍能生成许可证归属信息。
