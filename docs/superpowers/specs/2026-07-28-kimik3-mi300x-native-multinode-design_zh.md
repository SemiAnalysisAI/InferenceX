# Kimi K3 MI300X 原生多节点 AgentX 设计

<div align="center">

[English](./2026-07-28-kimik3-mi300x-native-multinode-design.md) | **中文**

</div>

## 摘要

为 AMD MI300X 集群增加一个显式启用的 Slurm 原生多节点路径，用聚合式
vLLM 运行 Kimi K3 AgentX。第一个垂直切片使用两个 8-GPU 节点和
TP8 × PP2 拓扑运行普通目标模型，并完成 AgentX canary。第二个堆叠改动在
同一生命周期上增加 DSpark 投机解码。

本设计是 Slurm 上的 Kimi K3 支持。Kubernetes、P/D 分离式推理、
MI325X 以及 vLLM 核心修改均不在范围内。

## 当前约束

- Kimi K3 MXFP4 大约为 1.5–1.56 TB。单个 8×MI300X 节点的总 HBM
  容量与模型大小接近，没有安全的运行时余量。
- 可用 MI300X 节点均提供 8 张 gfx942 GPU，以及节点本地的
  `/raid/hf-hub-cache`。
- 现有 `launch_mi300x-amds.sh` 是单节点路径，并且把总张量并行度和
  每节点 GPU 请求数混为同一个值。
- `/raid` 是节点本地存储；一个节点上的模型缓存无法被另一个节点看到。
- controller 当前以
  `SystemComment=slurm_cred_create failure, holding job` 挂起真实作业。
  这是外部执行门槛，基准测试代码不应绕过它。在该问题修复前，可以继续
  本地实现、CPU/静态测试并提交 draft PR；只有集群门槛通过后，改动才能
  被视为 runtime-ready 或 merge-ready。
- 当前 AMD Kimi K3 参考实现为 InferenceX PR #2351（MI355X 普通
  vLLM）和 #2367（MI355X DSpark）。PR #2353 提供 H200 上的 Kimi K3
  原生多节点生命周期。MI300X 实现复用这些已验证的契约，但不依赖
  NVIDIA 专用注意力后端。
- 上游 Kimi K3 recipe 只将 MI350X/MI355X gfx950 标记为 verified。
  但是 `vllm/vllm-openai-rocm:kimi-k3` 使用
  `AITER_ROCM_ARCH=gfx942;gfx950` 构建，包含 AMD Kimi K3 实现及 Kimi K3
  AITER kernel。镜像内已提交的 Kimi K3 tuned MoE 表只覆盖 gfx950，
  因此 gfx942 正确性必须经过 preflight，不能直接假设。

## 方案比较

### 1. 从现有 MI300X 启动器委派到隔离的原生多节点路径

在 `launch_mi300x-amds.sh` 开头增加一个显式启用的分支：

```bash
if [[ "${NATIVE_MULTINODE:-0}" == "1" ]]; then
    exec bash runners/launch_mi300x-amds-native-multinode.sh
fi
```

新启动器单独负责资源分配、每节点容器准备、服务生命周期、AgentX、
产物和清理。委派分支之后的现有单节点路径保持逐字节不变。

**决策：采用。** 该方案在保留现有 runner 标签和矩阵路由的同时，将
回归风险降至最低。

### 2. 增加全新的 runner 类型和启动器

隔离程度最高，但还需要新增自托管 runner 标签、runner group 配置及
workflow 路由。物理资源池已经使用 `cluster:mi300x-amds`，额外身份只会
增加运维工作。

**决策：不采用。**

### 3. 通过 srt-slurm 运行

B200 K3 工作证明了该方案，但原生多节点聚合式 vLLM 仍依赖尚未完成的
srt-slurm 上游工作和补丁。本次 MI300X 垂直切片不需要 P/D 编排。

**决策：本任务不采用。** 如果后续范围扩展到分离式推理，再重新评估。

## 交付切片

### K3 PR A：普通 vLLM 垂直切片

- 显式启用的 MI300X 原生多节点启动器。
- 两节点 TP8 × PP2 Kimi K3 服务入口。
- 并发 `[1, 2, 4, 8]` 的普通 AgentX 配置。
- 静态、矩阵生成、生命周期和清理测试。
- 一个基于准确 head commit 的两节点并发 1 canary，随后运行有界扫描。

当前 matrix generator 会有意为每个多节点 AgentX concurrency 生成一个
独立 job。因此上述并发阶梯会产生四个独立的 Slurm allocation，而不是
让四个点复用同一个 server allocation。

### K3 PR B：DSpark

- DSpark 服务入口，或基于普通入口的轻量模式封装。
- `Inferact/Kimi-K3-DSpark`、7 个投机 token，以及已提交的 Kimi K3
  黄金接受长度契约。
- 吞吐量使用合成接受，评估使用真实 block verification。
- 同样使用 `[1, 2, 4, 8]`，确保普通模式和 DSpark 可直接比较。

只有在 PR A 的真实集群 canary 通过后才开始 PR B，从而将分布式启动
故障和投机解码故障分开。

## 拓扑契约

首版只支持以下拓扑：

```text
节点数             = 2
每节点 GPU 数      = 8
world size         = 16
张量并行           = 8
流水线阶段         = 2
聚合 worker 数     = 1
decode worker 数   = 0
```

TP8 保持在单个 8-GPU 节点内，PP2 跨越节点边界。该拓扑与 H200 和 B200
K3 工作所使用的可行方案一致，也避免在每次张量并行操作中执行跨节点
TP16 collective。

启动器必须分离以下概念：

```text
MULTINODE_NODE_COUNT=2
MULTINODE_GPUS_PER_NODE=8
PREFILL_TP=8
PREFILL_PP_SIZE=2
WORLD_SIZE=16
```

每节点 Slurm GRES 请求绝不能从 `TP` 推导。

## 配置契约

AMD master 配置使用现有聚合式多节点 schema：

```yaml
kimik3-fp4-mi300x-vllm-agentic:
  image: vllm/vllm-openai-rocm:kimi-k3
  model: moonshotai/Kimi-K3
  model-prefix: kimik3
  runner: cluster:mi300x-amds
  precision: fp4
  framework: vllm
  multinode: true
  disagg: false
  scenarios:
    agentic-coding:
    - search-space:
      - spec-decoding: none
        kv-offloading: none
        conc-list: [1, 2, 4, 8]
        prefill:
          num-worker: 1
          tp: 8
          pp: 2
          ep: 1
          dp-attn: false
          additional-settings:
          - "NATIVE_MULTINODE=1"
        decode:
          num-worker: 0
          tp: 8
          pp: 2
          ep: 1
          dp-attn: false
```

原生 MI300X 路径会拒绝任何不满足以下条件的配置：AgentX、vLLM、
聚合式、两节点、TP8 × PP2、每节点完整占用 8 张 GPU。首版故意保持
严格且狭窄的契约。

服务入口只复用 #2351 已经实际运行过、且与架构无关的 Kimi K3 AMD
契约：ROCm Kimi K3 镜像、fast safetensors 以及必要的 parser/serving
参数。在 staging 模型之前，先用单 GPU gfx942 preflight 分别测试 AITER
a16w4 基线和显式启用的 a8w4 路径，只把通过的模式固化到 PR A 配置中。
AITER 模式、显存利用率和 batching 数值都保持为显式输入；没有证据时，
实现不得照搬 MI355X 上只适用于 gfx950 的调优结果。

## Slurm 与进程生命周期

启动器按以下顺序执行：

```text
验证环境和拓扑
→ 分配 2 个独占节点，每节点 1 个 task、8 张 GPU
→ 根据 Slurm task rank 解析 rank-0 hostname
→ 在所有分配节点验证目标模型缓存
→ 在所有分配节点导入或验证容器 squash
→ 每节点启动一个 vLLM rank
→ 等待 rank-0 health endpoint
→ 使用 --overlap 在 rank 0 启动 AgentX client
→ 持久化结果和有界诊断产物
→ 停止 server step
→ 取消并回收 allocation
```

服务端接收：

```text
MULTINODE_NODE_COUNT=2
MULTINODE_GPUS_PER_NODE=8
MULTINODE_NODE_RANK=$SLURM_PROCID
MULTINODE_MASTER_ADDR=<rank-0 hostname>
```

vLLM 命令使用：

```text
--tensor-parallel-size 8
--pipeline-parallel-size 2
--nnodes 2
--node-rank <0|1>
--master-addr <rank-0 hostname>
```

rank 0 提供 OpenAI endpoint，rank 1 增加 `--headless`。

ROCm 使用 PR #2351/#2367 中的 Kimi K3 AMD 环境。MI300X 路径不得加入
NVIDIA 专用的 `FLASHMLA` 或 `FLASHINFER_MLA` 后端。初次 bring-up 仅使用
GPU 常驻 KV；host KV offload 是后续优化，而不是模型加载的替代方案。

## 模型与镜像 staging

正式基准测试作业不会在线下载模型权重：

- 目标模型：`moonshotai/Kimi-K3`
- DSpark draft：`Inferact/Kimi-K3-DSpark`
- 主机缓存：`/raid/hf-hub-cache`
- 容器缓存：节点本地，并使用 `unsquashfs -l` 验证

第一次 canary 在选定的两个节点完成 staging，并把 allocation 固定到
该节点对。启用 CI sweep 前，将目标模型复制到所有符合条件的 MI300X
runner 节点；在 PR B 之前将 draft 同步到同一节点池。

staging 支持续传，并使用节点本地锁串行化。每个节点必须满足：

```text
目标模型 staging 前至少有 2 TB 可用空间
目标 snapshot 存在且完整
DSpark 运行前 draft snapshot 存在
容器 squash 可读
```

如果所需缓存缺失，production launcher 会直接失败，不会在有时限的
基准测试作业中启动 1.5 TB 下载。

## 日志、产物与清理

- 服务 stdout/stderr 写入 GitHub Actions workspace 之外、由主机用户
  所有的 scratch 目录。
- AgentX 写入挂载的 scratch 结果目录，主机只把最终结果和有界诊断文件
  复制到 workspace。
- 即使作业被取消，也不得在 workspace 中残留 root-owned 文件。
- `EXIT`、`INT` 和 `TERM` 清理会停止 server step、快照已有诊断信息、
  调用 `scancel`，并等待 allocation 消失。
- 任意 server rank 失败时，通过 `--kill-on-bad-exit=1` 终止整个
  `srun`。
- readiness 的上限为 7,200 秒；如果 server step 提前退出，则立即失败。

## 验证

### CPU/静态门槛

- 所有修改过的 shell 脚本通过 `bash -n`。
- 只为新增 config key 生成矩阵。
- 现有 matrix suite 保持全绿。
- 启动器测试覆盖：
  - native mode 会委派，而默认模式不会；
  - world size 与每节点 GPU 请求不会混淆；
  - 不支持的拓扑在 allocation 前失败；
  - 两个节点都参与镜像与缓存 preflight；
  - rank 0 提供服务，rank 1 使用 headless；
  - 失败和 signal 路径都会取消 allocation。

### 集群门槛

1. controller credential 修复后，真实单节点 `hostname` 作业成功。
2. 单 GPU 容器 preflight 识别出 gfx942，成功导入 AMD Kimi K3 实现，
   并通过一个 Kimi K3 shape 的 AITER MoE smoke test。它会记录 a16w4
   或 a8w4 哪个模式有效；如果两者都失败，则明确记录为 vLLM/AITER
   依赖，而不是在 InferenceX launcher 中偷偷绕过。
3. 两节点诊断输出两个不同 hostname，并在每节点看到 8 张 GPU。
4. 两个节点都通过模型与容器 preflight。
5. 普通 vLLM health 成功，并能服务一个直接请求。
6. AgentX 并发 1 生成有效结果，且无残留 Slurm job。
7. 四个独立 job 的普通模式有界扫描全绿。
8. PR B 使用 DSpark 重复门槛 5–7，并运行规定的评估路径。

## 非目标

- Kubernetes。
- MI325X。
- P/D 分离式推理。
- srt-slurm 修改。
- 在 InferenceX recipe PR 中隐藏 vLLM 或 Kimi K3 kernel 修改。如果
  gfx942 preflight 证明需要上游修复，则将其明确记录为独立依赖。
- 设施级功耗测量。
- rack-scale 或超过两节点的调优。
- 第一个普通垂直切片中的 CPU/DRAM KV offload。
