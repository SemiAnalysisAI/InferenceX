# DeepSeek V4.1 ROCm Engram CPU-offload experiment

This patch is based on vLLM `af1c01499b289be555c475669ba50a88e96d846e`.
It adds an AMD TP-sharded pinned-host embedding implementation and permits
explicit DeepSeek V4.1 Engram configuration on ROCm. It does not enable
Engram DP sharding or shared-memory DP storage.

The common FP8/E8M0 lookup kernel, hash history, TP head ownership, graph
staging, and model math are unchanged. Omitted Engram configuration keeps the
existing ROCm HBM behavior. Offloaded tables use the existing ROCm-capable
`get_accelerator_view_from_cpu_tensor` helper. Cached device aliases are
refreshed if the host parameter storage is replaced.

`apply.sh` checks the image pin and applies the complete patch with
`git apply --check`. It runs `rocm_preflight.py` before model loading.
The preflight checks real TP2 shard outputs against an independent dequantized
reference, masked IDs, noncontiguous hash inputs, graph replay with changed
IDs, pinned allocation under a GPU device context, and storage replacement.
The two shard owners are exercised on one GPU; real two-GPU collectives are
qualified by the serving sweep and evaluation jobs.

GPU preflight success alone does not qualify this serving recipe. Require
the full AgentX sweep plus the c32 real-rejection evaluation result.
AgentX throughput retains golden AL 3.51; evaluation uses real block rejection.
This experiment is AgentX only. Do not run fixed-sequence or 8k1k benchmarks.

The upstream-ready source and tests are bundled for human review. No
upstream submission or approval is implied by this patch.

<details>
<summary>中文</summary>

此补丁基于 vLLM `af1c01499b289be555c475669ba50a88e96d846e`，
为 AMD 添加按 TP 分片的 pinned-host Engram 存储，并允许在 ROCm 上显式配置
DeepSeek V4.1 Engram。不支持 Engram DP 分片或 DP 共享主机内存。
公共查找 kernel、哈希历史、TP 分片、图捕获暂存和模型计算均保持不变。
未提供 Engram 配置时，ROCm 仍使用原有 HBM 路径。

启动前会校验镜像并执行真实 GPU 查找和图回放测试；单卡上分别验证两个 TP
分片，真实双卡通信由服务扫描和评估验证。仅通过预检不代表配方通过验证。
仅运行 AgentX，必须完成全扫描及 c32 真实拒绝采样评估，不运行固定序列或 8k1k。
上游候选补丁需人工审核，此目录不代表已提交或获上游批准。

</details>
