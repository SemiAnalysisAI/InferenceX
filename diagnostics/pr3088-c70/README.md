# C70 serial-preparation diagnostic

Diagnostic-only branch based on recipe candidate `aec2e2c6b`. Do not merge these execution limits into the recovery PR.

Run exactly one Kimi K3 B300 C70 benchmark, no eval, `require-power=true`, `duration-override=1200`, `agentx-fast=false`, and empty PR labels. Preserve full dataset reconstruction and normal warmup. The native allocation is capped at 90 minutes (8 GPUs, at most 12 GPU-hours); GitHub job is capped at 100 minutes. Require the existing image cache and inspect the allocated node's effective Enroot runtime before extracting a container. Stop if the cache is unavailable or a retained runtime requires ownership/compatibility review. No automatic retry.

This can test whether serial preparation gets beyond the previous OOM and whether the short measured interval has valid power. It cannot qualify the full one-hour workload, the other failed concurrency points, quality evals, performance equivalence, or merge reuse.

<details><summary>中文</summary>

本诊断分支基于候选 `aec2e2c6b`，执行限制不合入修复 PR。仅运行 B300 C70 一个点，不运行 eval；保留完整数据准备与正常 warmup，将测量缩为 1200 秒并要求有效功耗。Slurm 上限 90 分钟（8 GPU，最多 12 GPU-hours），GitHub job 上限 100 分钟。必须使用已有镜像缓存；创建容器前检查计算节点的 Enroot 环境，缓存缺失或发现需核实的旧容器时立即停止。不自动重试。

结果只用于判断串行准备能否越过原 OOM，以及短测量区间的功耗是否有效；不能替代完整 sweep、其他失败点、质量 eval、性能对比或合并复用验收。

</details>
