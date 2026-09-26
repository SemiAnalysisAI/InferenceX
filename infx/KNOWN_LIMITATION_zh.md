# 基准测试客户端的限制

<div align="center">

[English](./KNOWN_LIMITATION.md) | **中文**

</div>

当使用较短的输入和输出，以极高的每秒查询数（QPS）测试 Gemma 1B 或 Llama 8B 等小模型时，服务基准测试客户端可能成为性能瓶颈。

InferenceX 目前侧重于更大的模型、更长的输入和输出序列，以及交互延迟和单用户吞吐量。支持高 QPS 的小模型工作负载需要采用多进程基准测试客户端。
