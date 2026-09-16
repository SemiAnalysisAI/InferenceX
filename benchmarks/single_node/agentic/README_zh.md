# Agentic 单节点基准

[English](README.md) | **中文**

**MVP / 实验性。** 此目录中的内容均非正式 InferenceX 基准。结果不在
https://inferencex.com 发布，也不应作为引用依据。

这些启动脚本用于开发和验证 agentic-coding 场景，之后才会将其升级为正式支持。
脚本按尽力而为原则维护，主要展示环境变量、场景路由及结果路径等基础流程的参考实现。
具体模型和配置可能随时无法运行，尤其是多节点支持目前尚未成为正式功能。

## DRAM KV 卸载内存策略

Agentic 场景通过 `kv-offloading` 指定资源层，通过 `kv-offload-backend` 指定后端实现。
`kv-offloading` 通常为 `none` 或 `dram`；使用 `dram` 时必须指定后端：

```yaml
- dram-utilization: 0.80
  search-space:
  - { tp: 4, kv-offloading: dram, kv-offload-backend: { name: vllm-native }, conc-list: [16, 32] }
  - { tp: 8, kv-offloading: none, conc-list: [16, 32] }
```

启用卸载时，必须提供 `kv-offload-backend.name`。`version` 可选：框架原生实现没有
独立版本时省略；LMCache、Mooncake 等独立发布的软件包应填写版本。

Agentic 矩阵生成默认使用 3600 秒时长。可复用工作流的调用方仍可显式覆盖 `duration`。

Agentic 主配置必须使用准确的 `cluster:<name>` runner 标签，确保搜索空间中的所有点
落在同一硬件集群。节点主机内存统一声明在 `configs/runners.yaml` 对应标签的 `hardware` 中：

```yaml
hardware:
  cluster:b300-dsxe:
    available-cpu-dram-mib: 3977095
    gpus-per-node: 8
```

矩阵生成器结合主配置中的利用率和 runner 硬件信息，生成总预算：
`floor(min(available MiB, 2,861,022) * 1,048,576 * utilization * tp / gpus-per-node / 1,000,000,000)`。
`2,861,022 MiB` 是十进制 3 TB 的 DRAM 上限。例如，八 GPU B300 节点在 80% 利用率下，
TP4 获得 1,199 GB，TP8 获得 2,399 GB。

基准脚本必须使用 `TOTAL_CPU_DRAM_GB`，不能以模型专用常量替代。采用每 rank 或每池
容量设置的后端必须相应拆分此总预算。DSv4 SGLang 是例外，因为它只暴露
`--hicache-ratio`；模型启动脚本限制其经验测得的比率，以保持在生成的字节预算以内。

NVMe 和 DRAM + NVMe 组合仅在分支内的 [B200 卸载交叉点实验](../../../experiments/agentx-offload/README_zh.md)
中启用。实验选择标记、容量及规范工作流要求参见该文档。
