# MI355X 上的原生 Kimi-K3 PD

[English](k3-pd-native.md) | **中文**

此 draft 通过 InferenceX 共享 Python launcher 和原生 srt-slurm 生命周期接入 Kimi-K3 prefill/decode 分离，不增加另一套 launcher 或 router 算法。

## 配置归属

- `configs/amd-master.yaml` 选择 `kimik3-fp4-mi355x-vllm-disagg-agentic`，提供矩阵标识和结果元数据。
- `benchmarks/multi_node/srt-slurm-recipes/kimik3/vllm/mi355x-fp4/agentx/disagg-variants.yaml` 管理拓扑、worker/router 镜像及 serving 设置：MXFP4 目标权重、FP32 SSM、FP8 KV 和 GMU 0.90。
- 保留现有 1P1D c1/c10/c48 与 1P2D c24/c48。低延迟配置使用 DSpark K7 且不做 CPU offload；其他配置使用 K4 和 prefill SimpleCPUOffload。不改变 draft 权重或精度；吞吐自动选择 golden acceptance，eval 使用真实校验。
- `configs/runners.yaml` 声明已部署模型、draft/fabric 挂载、worker 网络设置、memlock 和镜像导入策略。`infx/launch/` 使用任务私有 srt-slurm Python 解析 recipe 镜像，导入前拒绝 worker 镜像不一致，准备 recipe 指定的 router 镜像，并沿用共享提交、取消和结果收集。

高并发 prefill 保留 `HSA_NO_SCRATCH_RECLAIM=0`。原有图模式、传输设置及 1% 请求错误门槛不变。不加入 BF16 SSM、workspace 开发、shared-MR、QP/credit 实现或客户端取消补丁。

## 固定运行栈与可删除 debug 层

Worker 镜像为 `vllm/vllm-openai-rocm:nightly-ac68c3087215e0a4f3cdfa218508c6aada57235d@sha256:e3fdfb382f2b567718ab6de49a14f5d5695dad84efc6dfd9c38f661b1a763e19`，master 和 recipe 使用完全相同的不可变标识。镜像已包含 vLLM #57700，不再携带该 PR 的 backport。

临时 engine setup 只应用两个随源码交付的 Python 运行时补丁：同步 READ 清零保护和仅用于 synthetic 的 draft gather 回退。分别校验 SHA256 和 `git apply --check`，不替换编译扩展，也不下载浮动 PR head。Parser 使用 nightly 原实现，不携带 EOF 回补。独立 router 跳过 engine setup；worker 缺少 vLLM 或补丁不兼容时终止启动。

| 依赖 | 固定来源 | 作用 |
| --- | --- | --- |
| [vLLM #59164](https://github.com/vllm-project/vllm/pull/59164) | `c5b1350f1f2bf10a128127a9b85e93d5f9f18e62` | 从新分配 KV 页的清零列表中排除同步 READ 目标。 |
| Synthetic draft gathering 实验 | 针对固定 nightly 的 `k3-synthetic-unproposed-drafts.patch` | 仅在启用 synthetic acceptance 时恢复 #58784 之前的输入 gather；普通/block 验证仍拒绝未提出的槽位。这是 benchmark 兼容实验，不是通用正确性修复。 |
| [srt-slurm #508](https://github.com/NVIDIA/srt-slurm/pull/508) | 已测版本 `51cee8904a0b402a834887a26008adb79b8cd26b` | 在匹配任务的临时 checkout 中补齐 connector 模板内的 discovery 拓扑绑定。 |

READ zeroing 补丁哈希为 `3da3746e85d53e4a3b17062b4113475d31a86cc07418e87ef5b4bb0c906cad20`。不加入 #58968 的 EOF、心跳、FULL-context、draft-fence 或传输所有权补丁。

Synthetic 实验补丁哈希为 `33a6f792b13d39705a50562ca037a1d3c49dc054a8bdd8539fd8a154667f39df`。自动 golden acceptance 曲线、draft 模型/数量、parser 与错误门槛不变。该实验恢复首次调度占位槽的历史处理方式，可能改变首次可见 token 和后续生成；结果应与普通模型质量评估区分，尚未证明性能口径等价。

Debug 层还保留集群声明的单文件只读 `ionic-provider` 挂载，用于镜像与内核 ABI 兼容，与 shared-MR 无关。镜像内 libibverbs 核心库和其他 provider 不变。本集成不替换 srt-slurm 仓库 URL、子模块 pin 或 AIPerf 源码。

## 删除条件

1. 官方 worker 镜像包含 #59164 并通过验收后，同步更新 master/recipe，删除对应补丁及应用步骤。Synthetic 兼容问题定案后单独删除该实验，它不是等待上游合入的修复。全部运行时补丁移除后，再删除 worker setup、recipe 引用及其测试。
2. 独立升级共享 srt-slurm pin 到包含 #508 的版本，再删除任务内 cherry-pick 及专项测试。更换 worker 镜像不会升级 srt-slurm。
3. 只有替换镜像在目标平台不挂载 provider 也能打开全部预期 RDMA 设备，才删除兼容挂载。设备枚举不能代替 RDMA 流量验收。
4. 追加性能 changelog，并在合入前完成适用 smoke、sweep 和 eval。临时 engine 补丁仍属于 draft debug 集成，不代表获得上游镜像政策豁免。

## 证据与剩余验收

本轮 PR sweep 临时保留全部五个一小时吞吐点，仅选取原生生成的 1P2D c48 完整 GSM8K 精度任务。独立 debug commit 在原生矩阵生成后收窄 PR #3582 的评测范围，不改变 evaluator 输入、真实 block rejection、样本上限、分数门槛或 benchmark 判据；遇到意外范围会直接报错，不静默缩减。其他 PR 和手动 workflow 不受影响。宣称满足仓库完整评测覆盖之前必须删除该选择提交；本轮省略的厂商检查不视为已验收。

当前候选不带 EOF。此前 EOF 回归和吞吐结果仅作为另一套 parser 栈的历史证据，不代表当前候选已通过验证。

配对 [1P1D c48 一小时作业](https://github.com/billishyahao/InferenceMINI/actions/runs/36847087330) 使用相同不可变 worker 镜像、#59164 和字节一致的 EOF 运行时代码。Warmup 为 531 valid / 0 empty；profile 为 4040 valid / 4 empty（0.0989%），完成提交、结果导出及原生收尾。这是另一套适配 harness 的运行支持证据，不是本 PR 当前 commit 的 sweep 或精度认证。

此前 [InferenceX 完整特性对照](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/36800732704) 使用更广的补丁栈；后续仅带 #59164 的 [InferenceX 作业](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/36814088271) 在排队且未获得 GPU runner 时取消，均不构成本次收束候选的验收。最小组合仍需当前 PR 的 1P2D、sweep/eval 和已加载服务收尾验证；省略其他已定位修复不等于证明其故障不可能发生。
