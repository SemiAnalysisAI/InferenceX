# MI355X 上的原生 Kimi-K3 PD

[English](k3-pd-native.md) | **中文**

此配置通过 InferenceX 的 Python launcher 和原生 srt-slurm 编排运行 Kimi-K3 prefill/decode 分离。主配置与所选 recipe 管理基准拓扑和调优，不增加另一套 Bash launcher。

## 配置归属

- `configs/amd-master.yaml` 选择 `kimik3-fp4-mi355x-vllm-disagg-agentic`，提供矩阵标识和结果元数据。
- `benchmarks/multi_node/srt-slurm-recipes/kimik3/vllm/mi355x-fp4/agentx/disagg-variants.yaml` 管理角色、图配置、并发、worker/router 镜像及 FP32 SSM 状态。目标权重仍为 MXFP4，KV 为 FP8，GPU memory utilization 为 0.90。
- Draft checkpoint 使用上游 DSpark 路径，不转换权重精度。吞吐由 InferenceX 自动选择实测 golden acceptance，eval 保持真实校验。低延迟配置使用 DSpark K7 且无 CPU offload；其余配置使用 K4 与 prefill SimpleCPUOffload。
- `configs/runners.yaml` 管理已部署模型、draft 挂载、网络设备、worker 网络环境、memlock 和镜像导入策略。具名 srt 路径为该工作负载启用 recipe 镜像准备。
- `infx/launch/` 管理导入、安装、提交、取消和结果保留。Recipe 镜像由任务私有的 srt-slurm Python 环境解析，不在 launcher 解释器中导入依赖。Worker 镜像不一致会在导入前失败，所选 recipe 的 router 镜像也通过现有后端准备。

高并发 prefill 保留 `HSA_NO_SCRATCH_RECLAIM=0`，decode 不变。这是显式运行策略，不是临时源码补丁。不引入 BF16 SSM、workspace 开发、READ-credit/QP 或 router 算法改动。

## 官方镜像与临时集成

Worker 使用官方 `vllm/vllm-openai-rocm:nightly-ac68c3087215e0a4f3cdfa218508c6aada57235d`，固定 amd64 digest 为 `sha256:e3fdfb382f2b567718ab6de49a14f5d5695dad84efc6dfd9c38f661b1a763e19`。主配置和 recipe 使用完全相同的镜像标识，替换原自定义构建，不携带 shared-MR 修改。

该 nightly 已包含 [vLLM#57700](https://github.com/vllm-project/vllm/pull/57700)。Discovery 模板还依赖尚未进入 srt-slurm pin 的 [srt-slurm#508](https://github.com/NVIDIA/srt-slurm/pull/508)。必需的 engine backport 和镜像/provider 兼容前置放在单独、可删除的 debug commit 中，不混入框架或配置提交。

这些临时修改不符合直接合入的引擎补丁政策；待上游发布后删除：

1. 确认官方 ROCm nightly 包含必需的 engine 修复，同步更新 worker 镜像的两处引用，验收后删除对应临时 setup 和 patch 文件。
2. 独立升级 srt-slurm 子模块到包含 PR #508 的版本，删除临时的任务内 cherry-pick 及其测试。替换 worker 镜像不会升级 srt-slurm。
3. 追加性能 changelog，并按正常 smoke、sweep、eval 门槛验证新的源码与镜像组合。删除临时补丁只能让源码栈干净，不能替代运行验收和 review。

## 验证范围

Python launcher 迁移的行为测试覆盖镜像选择、导入前拒绝错误输入、使用外部调度/安装桩运行真实启动入口、结果归档、失败传播及取消。镜像仓库和源码检查证明镜像身份与上游能力包含关系，不证明设备/provider 兼容性或 RDMA 稳定性。

此前的 [c48 run](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/36556488243) 使用不同 harness 版本和自定义镜像，仅作历史记录，不是本次官方 nightly 候选的验收证据。此处不宣称新的 GPU 实测、吞吐、精度或完全优雅退出结果。
