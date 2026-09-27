# InferenceX 端到端基准测试

<div align="center">

[English](README.md) | **中文**

</div>

此项目包含模型推理服务基准测试、配置、启动器、Python 工具、文档及性能历史记录。
请从[文档索引](docs/index_zh.md)开始阅读。

[Python 项目清单](pyproject.toml)与[锁文件](uv.lock)由此项目维护。
从仓库根目录执行以下命令，安装并运行本地工具：

```bash
cd inferencex-e2e
uv sync --locked
uv run --locked python -m infx.matrix.generate test-config \
  --config-files configs/nvidia-master.yaml --config-keys <key>
```

将 `<key>` 替换为选定的配置键。工作流手动触发时的生成器参数也使用相对此目录的路径。
依赖组与检查命令详见[测试指南](docs/testing_zh.md)。

仓库规范、GitHub 工作流及共享的 [Python 版本](../.python-version)仍位于仓库根目录。
本目录的[许可证](LICENSE)确保仅挂载此项目目录的容器仍可生成许可证归属信息。
