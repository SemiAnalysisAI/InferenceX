# 配置操作规程

<div align="center">

[English](configuration-procedures.md) | **中文**

</div>

本页用于基准配置、配方、镜像和 runner 变更。它是操作规程而非字段目录；所链接的实现和 schema 始终是权威来源。

## 权威来源图

| 权威来源 | 控制内容 |
| --- | --- |
| [`configs/CONFIGS.md`](../configs/CONFIGS.md) | 主配置和 runner 配置的字段契约 |
| [`infx/matrix/validation.py`](../infx/matrix/validation.py) | 强制执行的 Pydantic schema 和拓扑不变量 |
| [`infx/matrix/generate.py`](../infx/matrix/generate.py) | 矩阵展开、过滤、runner 查找和生成的作业元数据 |
| [`configs/nvidia-master.yaml`](../configs/nvidia-master.yaml)、[`configs/amd-master.yaml`](../configs/amd-master.yaml) | 可执行的基准定义 |
| [`configs/runners.yaml`](../configs/runners.yaml) | 可调度标签、具体 runner 名称和各集群记录（`clusters:`） |
| [`benchmarks/`](../benchmarks) 和 [`infx/launch/`](../infx/launch) | 运行时命令、启动驱动和工作负载启动策略 |
| [`perf-changelog.yaml`](../perf-changelog.yaml) | 只允许追加的基准触发日志 |
| [`AGENTS.md`](../../AGENTS.md) | 仓库级配置、MTP、changelog 和 sweep 规则 |

退役的配置项直接从启用的主配置中删除，不再归档；历史设置由 Git 历史和 `perf-changelog.yaml` 保留。仅弃用部分场景时，只删除已退役的场景。同时删除不再使用的配方和模型专用初始化逻辑。保留 SPEED-Bench 采集器仍需使用的共享依赖，包括调度评分。参见[弃用规则](../../AGENTS.md#deprecating-benchmark-configs)。

## 依赖子模块

Git 记录依赖的精确提交版本。[`.gitmodules`](../../.gitmodules) 定义各仓库：AIPerf 位于 `utils/aiperf`，NVIDIA srt-slurm 位于 `utils/srt-slurm`。包括 TileRT 在内的所有 srt-slurm 作业均使用固定的上游子模块。

本地运行基准测试前，先初始化子模块：

```bash
git submodule update --init
```

升级时，在对应子模块中获取并检出目标提交，再将更新后的子模块指针提交到 InferenceX。基准测试工作流已配置为自动初始化子模块。Slurm 启动器为每个作业创建本地 Git 克隆，避免配方准备和运行时写入修改子模块，并记录实际提交以供结果溯源。

单节点固定序列长度配方使用 NVIDIA 上游 srt-slurm。ATOM 配方使用原生 `atomesh`
frontend、一个聚合 worker，并设置 `enable_multiple_frontends: false`。旧版基准 worker
镜像不包含 AToMesh，因此通过 `frontend.container_image` 单独固定路由器的官方镜像。
`model.container` 必须与主配置中的 worker `image` 一致；更换路由器镜像无需更换 worker
镜像。TRT-LLM 配方使用原生 `engine.served_model_name`，不再通过 `roles.agg.extra_args`
重复传入该参数。不再依赖此前分叉中的 ATOM 直连 frontend。

### 集群配置文件

集群的 srt-slurm 设置保存在 [`configs/runners.yaml`](../configs/runners.yaml) 中该集群的
`clusters.<id>.slurm.srt-slurm` 记录里（schema：[`infx/clusters/slurm.py`](../infx/clusters/slurm.py)）。
srt-slurm 只在 Slurm 上运行，因此该配置属于 Slurm 子记录。原生设置
（网络接口、调度指令、单节点时间上限、容器/nginx/模型别名、卷挂载与主机挂载、启动环境、主机设置以及字面量 `extra` 键）与工作负载配方分开维护。

srt 驱动（[`infx/launch/drivers/srt/`](../infx/launch/drivers/srt)）在 `make setup` 之前（`config.py`）
根据该记录和作业取值生成作业本地的 `srtslurm.yaml`。作业取值包括已暂存的镜像、解析后的模型路径、
缓存挂载、时间上限，以及功耗作业所需的 DCGM exporter 镜像。取值以 YAML 数据写入，
绝不替换进 shell 或 YAML 文本，`extra` 也不能覆盖类型化的键。

模型选择、缓存准备以及依赖工作负载的时间上限保留在 srt 驱动的表中
（[`lanes.py`](../infx/launch/drivers/srt/lanes.py)、[`models.py`](../infx/launch/drivers/srt/models.py)、[`power.py`](../infx/launch/drivers/srt/power.py)），不写进集群记录。

每次分配的主机检查和准备放在 `runners/srt-slurm/hooks/<cluster>/setup.sh`，集群专用辅助脚本放在其旁边。
目录名与集群 id 一致。在该集群的 `slurm.srt-slurm.host-setup` 记录（`script`、`env`、`timeout-s`、`nodes`）
中登记脚本，驱动会将其生成为 `default_host_setup`。脚本不会被自动发现。srt-slurm 在选定的已分配节点上、
容器之外、启动服务和 worker 之前运行这些脚本；检查失败默认会停止启动。通过 `host-setup.env` 显式传入配置。
如果以后需要撤销步骤，请为 `HostSetup`（infx/clusters/slurm.py）增加 `teardown` 字段，并将其生成为
`default_host_setup.teardown`；目前没有集群需要。这些是作业自有的 hook，不是管理员安装的
Slurm Prolog/Epilog 脚本。只为确有需要的集群添加 hook，不要为每个集群添加空脚本。

可复用的主机检查函数放在 `runners/srt-slurm/hooks/common.sh`，集群专用辅助函数放在 `setup.sh` 旁边。
common 文件只定义函数：source 它不得运行检查、修改环境变量或初始化基准测试。setup hook 与基准脚本都可以复用
这些函数，而不会把基准初始化带入主机设置。

hook 只注入**集群专用的主机前提条件**，例如网络结构检查或必要的主机状态准备。保持其短小、与工作负载无关，
并可安全重复运行。能用原生 srt-slurm 设置表达时，优先使用原生设置而不是 shell 代码。基准执行、模型选择、
引擎参数、并发调优、评测、结果收集和作业编排都不属于 hook。不要用 hook 给引擎或容器打补丁、绕过失败的检查，
或用重试和临时变通掩盖运行时缺陷；应修复负责的组件。主机改动只限于已分配节点，并保留其他作业正在使用的资源。

### DCGM counters

NVIDIA 集群配置通过 `slurm.srt-slurm.extra.default_gpu_exporter`，让 Tachometer 默认启动的 DCGM exporter
使用与功耗路径相同的 `dcgm-exporter:4.6.0-4.8.3-distroless` 镜像和
`/configs/dcgm-counters-noprof.csv`。功耗配方通过 `telemetry.dcgm_exporter.command`
选择同一份 CSV，因此两条路径共用同一 exporter 版本和同一 counters 文件。当功耗采集
已启动 exporter 时，Tachometer 复用该实例。保留功耗、能耗和 GPU 利用率，省略 profiling
与 vGPU license counters。这不会开启配方已关闭的采集，也不代表 Tachometer 指标已通过
PowerX 严格校验。现有的 Tachometer 1000 ms / 功耗 exporter 100 ms 采集间隔及 9401 端口
保持不变。

## 规程索引

1. [准备 worktree](#准备-worktree)
2. [添加模型 + 硬件配方](#添加模型--硬件配方)
3. [修改主配置](#修改主配置)
4. [注册并设置 runner](#注册并设置-runner)
5. [注册 srt-slurm 配方](#注册-srt-slurm-配方)
6. [注册 llm-d 配方](#注册-llm-d-配方)
7. [更新镜像](#更新镜像)
8. [添加或修改 MTP](#添加或修改-mtp)
9. [验证](#验证)
10. [避开 schema 和拓扑陷阱](#避开-schema-和拓扑陷阱)
11. [安全追加 changelog](#安全追加-changelog)
12. [停止条件](#停止条件)

## 准备 worktree

来源：[`docs/agent-guide.md`](agent-guide.md)、[`AGENTS.md`](../../AGENTS.md)。

从干净工作树中的 `inferencex-e2e/` 目录开始：

```bash
git status --short --branch
git fetch origin
git worktree add -b config/<slug> .worktrees/<slug> origin/main
cd .worktrees/<slug>
git status --short --branch
```

1. 编辑前确认路径、分支、基准提交和状态。
2. 阅读 `AGENTS.md`，再完整阅读最接近且可工作的配置、脚本、launcher 和配方。
3. 记录 changelog 和生成器必须选择的精确配置 key。
4. 保留无关工作。不要 reset、clean、rebase，也不要删除并非由你创建的文件。
5. 本地生成成功前保持配置工作隔离；不要为了发现 YAML 或路由错误而消耗 GPU 时间。

## 添加模型 + 硬件配方

详细来源：[`.claude/commands/add-model-hardware.md`](../../.claude/commands/add-model-hardware.md)。字段来源：[`configs/CONFIGS.md`](../configs/CONFIGS.md)。
STP（Single Token Prediction，单 Token 预测）是每次前向传播生成一个 Token 的标准自回归解码。MTP（Multi-Token Prediction，多 Token 预测）通过原生预测头或投机解码在每次前向传播中预测多个 Token。

1. **固定身份。**确认精确 checkpoint ID、model prefix、精度、架构、原生上下文、目标 SKU、框架，以及解码方式是 STP、原生 MTP 还是 draft-model 推测。验证镜像 tag 确实存在；绝不能编造。
2. **选择两类同类项。**阅读同一模型在其他 SKU 上的实现，以及目标 SKU 上的另一个模型。阅读它们在 [`benchmarks/single_node/srt-slurm-recipes/`](../benchmarks/single_node/srt-slurm-recipes) 下的 srt-slurm 配方及主配置条目。
3. **添加 srt-slurm 配方。**放在 `benchmarks/single_node/srt-slurm-recipes/<model-prefix>/<engine>/<sku>-<precision>[-mtp]/8k1k.yaml`，每个矩阵点对应一个 `override_*` 变体。保留已验证同类项中的引擎参数、env、parser 参数、attention/MoE backend、KV-cache dtype、graph/eager 模式、`setup_script` 和上下文处理。
4. **添加主配置条目。**`mi*` 使用 [`amd-master.yaml`](../configs/amd-master.yaml)，其他使用 [`nvidia-master.yaml`](../configs/nvidia-master.yaml)。精确设置 `image`、`model`、`model-prefix`、`runner`、`precision`、`framework`、scenario 和支持的搜索空间。
5. **依据证据确定规模。**复制已验证的并行布局并删除不支持的布局。延迟型 TP 行通常从并发 1 开始；不要把大显存 SKU 的 TP/EP 布局复制到小显存 SKU。
6. **检查变体选择。**每个搜索空间行都带有 `srt-recipe:`，每个矩阵点必须按 TP/GPU 数、`CONC`、`KV_OFFLOADING` 和镜像恰好匹配一个配方变体（`infx/srt_slurm/single_node.py::select_recipe`），无需 launcher 路由。
7. **追加一条 changelog**，精确选择新 key；参见[安全追加 changelog](#安全追加-changelog)。
8. **验证语法和生成结果。**检查 image、model、runner、ISL/OSL、`max-model-len`、并发、TP/PP/EP/DCP/PCP 和 `spec-decoding`。

仅添加 `MODELS.md` 行并不会产生可执行配方。完整路径是：srt-slurm 配方 + 主配置条目（`srt-recipe:`）+ changelog 触发项 + 生成的矩阵。

## 修改主配置

来源：[`configs/CONFIGS.md`](../configs/CONFIGS.md)、[`validation.py`](../infx/matrix/validation.py)、[`generate.py`](../infx/matrix/generate.py)。

1. 定位精确 key，完整阅读其条目及相邻同类项。
2. 只使用文档列出的 kebab-case 字段。schema 禁止额外字段；看起来合理的字段不会自动被接受。
3. 沿生成器输出、workflow 输入、launcher 和基准脚本追踪每个被修改字段。YAML 能通过只能证明形状正确，不能证明运行时已使用。
4. 保持各层的权威边界：
   - 主 YAML：矩阵身份、标签、搜索空间和输出元数据；
   - 基准脚本：服务端/客户端行为；
   - launcher：路由、挂载、模型路径、镜像启动和集群行为；
   - 外部/检入的配方：框架特定的多节点运行时。
5. 对拓扑变更，先计算 GPU 用量，再与目标 fleet 对照。
6. srt-slurm 必须同时更新配方和主条目；llm-d 必须同时更新 llm-d 配方/编排和主条目。
7. 追加触发条目，先只生成受影响的 key，并检查每个生成点。

固定序列 `8192/1024` 场景可设置 `require-power: true`，要求经过验证的实测功耗。矩阵将此标记传递给标准 sweep 和手动 E2E 吞吐作业；eval-only 和 AgentX 行不继承该标记。省略此字段可保留现有行为。仅在对应 runtime 和结果适配器同时交付时启用，然后验证完整选定范围。

## 注册并设置 runner

设置来源：[`utils/runner_setup/RUNNER_SETUP.md`](../utils/runner_setup/RUNNER_SETUP.md)。配置来源：[`configs/CONFIGS.md#runners`](../configs/CONFIGS.md#runners)。

### 仓库注册

1. 在 [`configs/runners.yaml`](../configs/runners.yaml) 中为 fleet 添加 `clusters.<id>` 记录（节点形状、工作负载环境、模型以及调度器子记录：Slurm 为分区、卷、squash 缓存和 srt-slurm 事实；schema 见 [`configs/CONFIGS.md#runners`](../configs/CONFIGS.md#runners)）。依赖模型、框架、精度或配方的启动规则写进 [`infx/launch/policy.py`](../infx/launch/policy.py) 或唯一读取它的驱动旁边，绝不在驱动中按集群 id 分支。新调度器上的集群需要在 [`infx/clusters/`](../infx/clusters) 下新增该调度器的设置模型、在 [`infx/launch/backends/`](../infx/launch/backends) 下新增其后端，各登记一行，无需修改驱动；这类集群只运行 script 驱动（`BENCH_SCRIPT_OVERRIDE`）的点。
2. 在 [`configs/runners.yaml`](../configs/runners.yaml) 预期的 `labels:` key 下添加每个精确的已注册 runner 名称。新名称使用 `<base-name>_<NN>`，索引必须两位补零。
3. 把每个 runner 名称加入且仅加入一个与该记录对应的 `cluster:<id>` 标签。`python -m infx.launch run` 通过 runner 名称解析集群，因此不属于任何集群标签的 runner 会导致校验失败，并在启动时失败。
4. 事实依赖某个物理 fleet 的主条目使用对应的精确 `cluster:<id>` 标签；agentic 配置强制要求该标签。
5. 添加/更新主条目以使用该标签。生成目标矩阵并确认选择了正确的具体名称。

路由依据 `cluster:<id>` 标签，而不是 runner 名称前缀。`<base-name>` 不得包含 `_`：`_` 用于分隔名称与 runner 序号。

### 主机设置

1. 确定 runner 用户和共享存储。登录节点与计算节点都必须能看到 `_work`。
2. 确认注册 shell 的 `PATH` 中有 `curl`、`tar`、`tmux`；Slurm 还需要 `sinfo`/`srun`/`sbatch`。
3. 获取仓库管理员认证和新的注册 token；token 大约一小时后过期。
4. 按文档运行 [`setup.sh`](../utils/runner_setup/setup.sh)，传入 token、runner URL、索引范围、基础目录、基础名称和标签。
5. 用 [`start_runners.sh`](../utils/runner_setup/start_runners.sh) 启动。
6. 将 runner 加入 sweep 流量前，在[仓库 runner 设置页](https://github.com/SemiAnalysisAI/InferenceX/settings/actions/runners)确认每个 runner 都是 **Idle**。
7. 从计算节点验证 launcher 对 `_work`、HF cache、预置权重和 squash 镜像的挂载。root 容器不得在共享 workspace 留下 root 所有的文件。

## 注册 srt-slurm 配方

映射来源：[`benchmarks/multi_node/srt-slurm-recipes/RECIPES.md`](../benchmarks/multi_node/srt-slurm-recipes/RECIPES.md)。检入的配方：[`benchmarks/multi_node/srt-slurm-recipes/`](../benchmarks/multi_node/srt-slurm-recipes)。

1. 定位精确的上游 [NVIDIA/srt-slurm](https://github.com/NVIDIA/srt-slurm) 配方，并记录固定到 commit 的来源路径。
2. 将 YAML 放在 `benchmarks/multi_node/srt-slurm-recipes/<model-prefix>/<engine>/<gpu>-<precision>/<workload>/` 下，遵循 `RECIPES_zh.md` 中的命名规范。阅读最接近的同类项和所选集群 launcher。
3. 将来源字段映射到主配置搜索空间条目：资源 worker 数 → `num-worker`；TP/EP/DP-attention → worker 拓扑；基准并发 → `conc-list`；配方路径 → `additional-settings: ["CONFIG_FILE=..."]`。
4. 在同一变更中添加/更新匹配的 [`nvidia-master.yaml`](../configs/nvidia-master.yaml) 条目。同步 worker 数、TP/PP/EP/DCP/PCP、hardware、router、传输引擎和并发标签。
5. 更新镜像时，使配方 `model.container` 与主配置 `image` 完全相同；launcher 使用主配置镜像作为 container alias key。
6. 运行配方所记录的 `srtctl` 验证，再生成主配置 key，并把每个前端标签/拓扑字段与配方逐一比对。
7. 追加 changelog 条目。

不得只提交一侧：`srtctl` 读取配方，而矩阵生成读取主配置。仅改配方可能给结果贴错标签；仅改主配置不会改变实际部署的配方。

## 注册 llm-d 配方

来源：[`benchmarks/llm-d/README.md`](../benchmarks/llm-d/README.md)、[`benchmarks/multi_node/llm-d/README.md`](../benchmarks/multi_node/llm-d/README.md)、和 [`llm-d-recipes/`](../benchmarks/multi_node/llm-d-recipes)。

llm-d 不是 srt-slurm 路径：InferenceX 自己持有 Slurm allocation，并在每个节点启动一个容器。

1. 复制 [`benchmarks/multi_node/llm-d-recipes/`](../benchmarks/multi_node/llm-d-recipes) 下最接近的 YAML，设置 EPP plugin/scheduling、角色特定 `extra-args`/`env`，以及可选 `slurm.time_limit`。
2. 添加/更新 `llmd-vllm` 主条目。设置 `multinode: true`、`disagg: true`、router 元数据、`kv-p2p-transfer`、prefill/decode worker 拓扑、并发，以及 `additional-settings` 中的 `CONFIG_FILE=<basename>.yaml`。
3. 保持 `PREFILL_NODES`、`DECODE_NODES`、`GPUS_PER_NODE` 和 worker 数与 allocation 及各角色 DP/TP/EP 布局一致。
4. 确认 [`submit.sh`](../benchmarks/multi_node/llm-d/submit.sh) → [`job.slurm`](../benchmarks/multi_node/llm-d/job.slurm) → [`server.sh`](../benchmarks/multi_node/llm-d/server.sh) 的传递，以及所选 wrapper/launcher 路由。
5. 验证文件发现：decode leader 生成 `/tmp/endpoints.yaml`；prefill endpoint 使用 vLLM 端口 8200，decode endpoint 使用 sidecar 端口 8000；名称唯一；地址为 IPv4 字面量；端口是 `1..65535` 范围内的字符串。
6. 确认 EPP 在 Envoy 收到流量前完成 discovery 加载，且角色标签为请求阶段选择正确的 prefill/decode backend。
7. 生成 key，检查拓扑和 `additional-settings`，再追加 changelog。

`CONFIG_FILE` 未设置或文件缺失时，会静默选择镜像内 `/etc/epp/config.yaml` fallback，并移除配方特定 vLLM 参数。除非明确打算使用 fallback，否则应将其视为验证失败。

## 更新镜像

来源：[`AGENTS.md#non-negotiable-benchmark-invariants`](../../AGENTS.md#non-negotiable-benchmark-invariants)、对应主配置、运行时脚本与检入的 Recipe。

1. 验证精确的上游 registry tag 或 digest 确实存在，并适用于 CUDA/ROCm 和目标架构。
2. 找出所有受影响的配置 key、运行时脚本、Dockerfile 和检入配方。不要假设主 YAML 是唯一镜像引用。
3. 将主配置 `image` 与所需 env、参数、软件包版本或补丁作为一个一致变更更新。
4. 对 srt-slurm，更新 `model.container` 并保持其与主配置 `image` 完全一致。
5. 对 llm-d，区分主配置选择的服务镜像和 [`benchmarks/llm-d/Dockerfile`](../benchmarks/llm-d/Dockerfile) 中的构建来源；仅在构建契约变化时同时更新两者。
6. 追加选择全部受影响 key 的 changelog 条目（有意覆盖多个 key 时可以使用通配符），并列出旧/新版本及实质运行时变更。
7. 生成每个受影响的配置族，确认其运行时路径中没有残留旧 tag。

## 添加或修改 MTP

来源：[`AGENTS.md#non-negotiable-benchmark-invariants`](../../AGENTS.md#non-negotiable-benchmark-invariants)、[模型+硬件 playbook 的 MTP 附录](../../.claude/commands/add-model-hardware.md#appendix--mtp--eagle3-spec-decoding-variant)和现有 [`*-mtp` srt-slurm 配方](../benchmarks/single_node/srt-slurm-recipes)。

1. 确认使用原生 MTP 模块还是外部 draft。使用 draft 时，从模型/上游配方验证精确模型 ID、方法（例如 `eagle3`）和建议 speculative token 数。
2. 复制相同模型和 backend 的可工作同类项。保留其 speculative config、attention backend、token 数、模型补丁和依赖设置。
3. 每个投机解码的定长配方变体都必须设置 `benchmark.env.USE_CHAT_TEMPLATE: "true"`；`select_recipe` 会拒绝缺少该设置的投机解码变体，[`srt_fixed_sequence.sh`](../benchmarks/single_node/srt_fixed_sequence.sh) 会将其转换为传给 `run_benchmark_serving` 的 `--use-chat-template`。原始 prompt 会静默降低 acceptance。
4. graph capture 至少按 `CONC * (1 + NUM_SPEC_TOKENS)` 确定规模，采用同类项的取整方式，并限制在框架上限内（当前 vLLM playbook 上限为 2048）。
5. 保留 backend 差异：不要把 CUDA 专用 drafter attention pin 或补丁复制到 ROCm 配方。
6. 在相应搜索空间条目设置 `spec-decoding: mtp`，并将其 `srt-recipe:` 指向 `-mtp` 配方；`select_recipe` 会据此校验配方的 speculative 配置。若使用 schema 支持的 draft-model 模式，要有意设置匹配的生成值；不要根据文件名推断。
7. 同时添加配方 + 主配置条目 + changelog。
8. 运行 YAML 和生成检查；检查 `spec-decoding`、draft/native 方法、token 数、chat-template 使用、capture 范围和解析出的配方变体。

### MI355X ATOM 上的 DeepSeek-V4-Pro-0813 DSpark

`dsv4-fp4-mi355x-atom-agentic-mtp` 保留历史配置 key，
矩阵元数据改为 `spec-decoding: draft_model`。所有行均选择
`benchmarks/single_node/srt-slurm-recipes/dsv4/atom/mi355x-fp4-mtp/agentic.yaml`；
配方选择时将 `draft_model` 视为 `mtp`。配方固定 revision
`72e1d3230f6c080a530b0a1d46f8eb4602340597`，以实际 snapshot 路径启动服务；
显式传入的 `MODEL_PATH` 也必须通过相同检查。GPU 启动前核对 config/index 哈希、
DSpark Markov/confidence head、全部 66 个分片的 header 与 payload 边界，
并离线加载 tokenizer。这验证可读性和完整性，不计算完整权重文件哈希。

全部十个 AgentX 性能点使用 DSpark K6（target 验证长度为 7）和已提交的
golden AL 3.77。C1/2/4/8/16 使用 TP8/EP1；C48/64/96/128/256 使用
TP8/DPA8/EP8 原生 RCCL。每个性能点运行 3600 秒。C256 全量 GSM8K 不传强制
接受率参数。保留固定的 `rocm/atom-dev:nightly_202609291501` 镜像和 GPU KV；C1 至 C16
使用 BF16 KV，C48 及以上继续使用 FP8 KV，所有任务均使用 FP4 index cache、
8192-token checkpoint 和 DEP dense FULL graph 阶梯。每个新服务进程重新捕获固定
q7 图；必须从 `server.log` 确认 target 和 DSpark draft capture 完成。confidence
schedule 和 ragged verification 保持关闭。

`AGENTIC_TOKENIZER_PATH` 可覆盖 AgentX 的 tokenizer 来源，默认仍为 `MODEL`；
本配方将其设置为已验证的服务 snapshot。`checkpoint_preflight.json`、
`runtime_manifest.json` 和 `server_command.txt` 保存模型/源码身份及请求的配置。
成功启动、graph capture 和请求执行仍需运行时日志证明。

固定镜像为官方 ATOM nightly `rocm/atom-dev:nightly_202609291501`（ATOM `0.1.7.dev46+g74fd942b0`，
ROCm 7.2.4），已包含已合入的
[ROCm/ATOM#2233](https://github.com/ROCm/ATOM/pull/2233) inference-mode 修复。
自该镜像起，ATOM 在 gfx950 上默认以 E8M0 存储检查点的 `ue8m0` FP8 block scale
（[ROCm/ATOM#2419](https://github.com/ROCm/ATOM/pull/2419)），2 的幂次 scale 可被精确表示。
配方不再在运行时修改 AITER 源码；TP 通信融合、DSpark K6 和 graph capture
直接使用镜像内实现。

### MiniMax-M3 ATOM FlyDSL paged decode 与 LMCache DRAM 层

`minimaxm3-fp4-mi355x-atom-agentic-mtp` 按照
[ROCm/ATOM#2366](https://github.com/ROCm/ATOM/pull/2366) 和
[上游配方](https://github.com/ROCm/ATOM/blob/1423fceb08fbe88b2e35c77b320b3073b310a63f/recipes/MiniMax-M3-Agentic-InferenceX.md)，
使用 `rocm/atom-dev:nightly_202609281543`，启用 `ATOM_PA_FLYDSL=1` 和
`ATOM_PA_FLYDSL_PLAN=1`。FlyDSL 处理支持的 paged-decode shape，work planner
按实际上下文长度均衡 dense decode 工作量；不支持的 shape 仍回退至 Gluon。
从 `server.log` 核对实际路由，以及 work plan 是否在图捕获时创建。

TP2 C20/C25/C30 与 TP4 C40/C48 点位启用 LMCache 进程内 CPU 层。
`benchmarks/single_node/srt-slurm-recipes/minimaxm3/atom/mi355x-fp4-mtp/agentic.yaml`
中的 `*_lmcache` 变体在 `extra-kv-connectors` 下列出
`{kv_connector: lmcache_offload, kv_role: offload}`，srtctl 将其渲染为
`--kv-transfer-config '{"kv_connector":"lmcache_offload","kv_role":"offload"}'`
（`runners/srt-slurm/patches/507-lmcache-server-atom-sglang.patch`），CPU 层由
`LMCACHE_*` 环境变量配置。每个变体声明 `KV_OFFLOADING: dram` 和矩阵的
`TOTAL_CPU_DRAM_GB`，并将 `LMCACHE_MAX_LOCAL_CPU_SIZE` 设为
`TOTAL_CPU_DRAM_GB / TP`（每 rank 257 GB）。`PYTHONHASHSEED=0` 必须设置：否则各
rank 对同一 prompt 计算出不同的 key，卸载命中率为零。在 `server.log` 中核对组装后的
`kv_transfer_config` 日志和非零的 `atom:lmcache_loaded_tokens`。

srtctl 将非整节点 worker 限定在 GPU `0..TP-1`，均位于 NUMA 节点 0；HIP 又把每个 rank
的 CPU 层 pin 在其 GPU 所在的节点上，因此 CPU 层与权重 staging buffer（TP2 约 720 GB，
TP4 约 1.23 TB）全部来自节点 0 的 1.5 TB 内存。`runners/srt-slurm/hooks/mi355x-amds/setup.sh`
会在 `KV_OFFLOADING=dram` 的 job 启动前释放 page cache，保证这些内存页空闲；否则 pin
内存时需要回收 page cache，各 rank 完成时间相差数分钟，ATOM 启动时 300 秒的屏障会超时
（[run 36454319395](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/36454319395)）。

移除 TP4 C32；GPU 常驻 KV 的 TP4 C1-C28、TP2 C1-C2 点位、EAGLE3 K3、golden AL 2.78
和 indexer CP 保持不变。

### DeepSeek-V4.1-Flash DSpark

GB200 的 DSpark 配方将 CUDA graph 最小捕获范围设为 64 tokens，以覆盖 AgentX 子代理并发。这会将 c1/c2/c4 的上限从 8/16/32 提升至 64；c8 及以上保持原有大小。完整轨迹、AL 3.51 和 Engram UVA 配置保持不变；需通过 CI 验证低并发尾延迟改善。
B200 的 DSpark 配方按测试点显式设置捕获尺寸，详见下文。
GB300 的 DSpark 配方按测试点显式设置捕获尺寸，详见下文。

B200 条目使用 `vllm/vllm-openai:nightly-dev-x86_64-cu130-ac9126e58aa7`，开启 FlashInfer autotune。
TP4 覆盖并发 1–128；DEP2（TP1 x DP2 + EP2，DeepGEMM MegaMoE）覆盖 8–32，DEP4（TP1 x DP4 + EP4，DeepGEMM MegaMoE）覆盖
64–128，两者均前置一致性哈希 vLLM Router。DEP2 每个 B200 rank 约有 150 GiB 权重，因此 batched tokens
上限为 4096，CUDA graph 捕获上限为 576 tokens。所有 B200 测试点设置 `--gpu-memory-utilization 0.97`。
所有测试点使用 `FULL_AND_PIECEWISE` CUDA graph，捕获尺寸为六 token 验证块的倍数。
H200 的 DSpark 配方使用相同的最小捕获范围，并保持相同的工作负载配置。

B300 的 DSpark 配方按测试点显式设置捕获尺寸，详见下文。

仅运行 AgentX 的 `dsv41flash-fp4-<sku>-vllm-agentic-dspark` 配方使用
[`nvidia-master.yaml`](../configs/nvidia-master.yaml) 中按 SKU 固定的 `image`（最初为 `vllm/vllm-openai:deepseekv41-flash-0909`），在 Blackwell SKU 上采用 TP4、原生五 token DSpark、
概率采样草稿。吞吐测试使用[已提交的黄金 AL](../infx/golden_al_distribution/dsv41flash_dspark.yaml)：thinking 开启、五个草稿 token 对应 3.51，采用合成拒绝采样并关闭自适应验证。准确率 eval 保留真实块拒绝采样和自适应验证。
`--engram-config '{"cpu_offload":true}'` 将 Engram 嵌入表放在固定页主机 DRAM
中，通过 UVA 访问；`kv-offloading: none` 描述的是另行保留在 GPU 上的 KV cache。
专家权重为 MXFP4，因此配方标记为 `precision: fp4`。

各 GPU 入口共用纯文本服务行为，使用 `deepseek_v41` tokenizer 和解析器、1M 上下文，
以及共享的 AgentX 轨迹回放、功耗、指标和 eval helper。除下文另有说明外，TP4 的并发范围为 1–128。
共享脚本按六 token DSpark 验证块设置 CUDA graph capture。srt-slurm 单节点路径将检出挂载到 `/infmax-workspace`，避免在 `/workspace`
下创建 AgentX 运行目录。沿用集群的模型路径和持久化缓存。配方在计算节点探测服务端口，首选端口被占用时选择可用端口，
服务、回放、指标和 eval 共用同一端点。所有配方都必须获得 GPU sweep 和 eval
证据后才能视为已验证。

B300 条目使用 `vllm/vllm-openai:nightly-dev-x86_64-cu130-ac9126e58aa7`，开启 FlashInfer autotune。
TP4 覆盖并发 1–16；DEP2（TP1 x DP2 + EP2，DeepGEMM MegaMoE）覆盖 8–192，前置一致性哈希 vLLM
Router，并发 128 及以上改用 MegaAttention。所有测试点使用 `FULL_AND_PIECEWISE` CUDA graph，捕获尺寸为
六 token 验证块的倍数。其他 SKU 继续使用共享脚本。

GB300 条目使用 `vllm/vllm-openai:nightly-dev-arm64-cu130-ac9126e58aa7`，开启 FlashInfer autotune。
TP4 覆盖并发 1–16；DEP2（TP1 x DP2 + EP2，DeepGEMM MegaMoE）覆盖 8–192，前置一致性哈希 vLLM
Router，并发 128 及以上改用 MegaAttention。所有测试点使用 `FULL_AND_PIECEWISE` CUDA graph，捕获尺寸为
六 token 验证块的倍数。

GB300 launcher 将引擎就绪等待时间设为 7200 秒。在[运行 34504969146](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34504969146) 中，仅模型加载就耗时 18–23 分钟；Rust frontend 达到 3600 秒期限时，引擎仍在捕获 CUDA graph。此次仅延长启动等待时间，基准测试时长和解码设置保持不变。

来源：[上游配方](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)。

### H200 上的 DeepSeek-V4.1-Flash DSpark

`dsv41flash-fp4-h200-vllm-agentic-dspark` 是 DeepSeek-V4.1-Flash 配方的 H200 AgentX
分支。它固定使用 `vllm/vllm-openai:nightly-cd10ed6f9f6b37a8ace9cf380007e66fe12ec0c3`（与 B200、GB200 相同），并与 Blackwell 分支共用纯文本服务
设置：`deepseek_v41` tokenizer 和解析器、1M 上下文、原生五 token DSpark（概率采样草稿）。吞吐测试使用[已提交的黄金 AL](../infx/golden_al_distribution/dsv41flash_dspark.yaml)：thinking 开启、五个草稿 token 对应 3.51，采用合成拒绝采样并关闭自适应验证。准确率 eval 保留真实块拒绝采样和自适应验证。

该分支使用 **TP8**，而非上游的 TP4。上游在一个 GB200 NVL4 tray 上验证 TP4，并说明在
8 GPU 节点上同一布局每个角色变为 TP8，而 H200 DGXC 节点正是 8 GPU 节点。

`precision: fp4` 标记检查点中 MXFP4 的路由专家权重，与同一检查点的 Blackwell 和
MI355X 分支保持一致。Hopper 没有 FP4 tensor core，因此这些权重走上转换的 MoE 路径；
该标签描述检查点，而非 SKU 的原生算力。

`--engram-config '{"cpu_offload":true}'` 将 Engram 表放在固定页主机 DRAM 中，通过 UVA
访问；`kv-offloading: none` 描述的是另行驻留 GPU 的 KV cache。集群实测：卸载在 8 个 rank
上为两张表各移出每 rank 11.80 GiB，共 188.8 GiB，使每 GPU 的驻留权重从 141 GiB 中约占
35.9 GiB。

轨迹语料：该分支回放未截断的 `semianalysis_cc_traces_weka_062126` 语料，而不是 256k
截断的 `..._062126_256k` 变体，因为该模型服务 1M 上下文。配方本身并未指定语料 ——
`resolve_trace_source` 选中未截断的默认值，仅仅是因为其 `dsv4*` 分支同时匹配了
`dsv41flash` 前缀。这一依赖在调用处并不可见却至关重要，且目前没有测试固定它（原
`runners/test_dsv41flash_h200.py` 已在 #3141 中删除）；收窄该分支会静默地降级本配方的轨迹。

**H100 分支单独实现。** H100 不在上游硬件表中，且瓶颈不在权重。在 1M 上下文下，稀疏
注意力 indexer 会在 `fp8_fp4_paged_mqa_logits` 中分配一个
`[max-num-batched-tokens, max-model-len]` 的 logits 缓冲区，在默认 8192 batched tokens
下恰好为 16 GiB。这是显存 profiling 阶段固定支付的启动开销，与并发无关，因此即使驻留
权重放得下，在 80 GB 卡上并发 1 也会失败。因此 H100 分支使用独立的配方设置并收窄 batched
tokens，详见下文 H100 小节。

srt-slurm 单节点路径将检出挂载到 `/infmax-workspace`，避免在 `/workspace` 下创建 AgentX 运行目录；它还
挂载了共享 HF 缓存，因此配方通过 `HF_HUB_CACHE` 解析模型，而不依赖各节点的独立路径。
配方在计算节点探测服务端口，首选端口被占用时选择可用端口，服务、回放、指标和 eval 共用
同一端点。

### H100 上的 DeepSeek-V4.1-Flash DSpark

吞吐测试使用[已提交的黄金 AL](../infx/golden_al_distribution/dsv41flash_dspark.yaml)：thinking 开启、五个草稿 token 对应 3.51，采用合成拒绝采样并关闭自适应验证。准确率 eval 保留真实块拒绝采样和自适应验证。

`dsv41flash-fp4-h100-vllm-agentic-dspark` 是 DeepSeek-V4.1-Flash 配方的 H100 AgentX
分支，在 H200 分支之后加入，并有意与其分开。H100 **不在**上游硬件表中（该表列出
h200、gb200、gb300、mi350x）。

与其他 SKU 不同，H100 不沿用共享服务参数，其配方
`benchmarks/single_node/srt-slurm-recipes/dsv41flash/vllm/h100-fp4-mtp/agentic.yaml` 使用独立参数，
因为共享参数无法在 80 GB 卡上服务 1M 上下文。在 1M 上下文下，稀疏注意力 indexer 会在
`fp8_fp4_paged_mqa_logits` 中分配 `[max-num-batched-tokens, max-model-len]` 的 logits
缓冲区：按共享参数实际生效的 8192 batched tokens 计算，即 8192 x 1048576 x 2 字节，
恰好 16.00 GiB。这是启动阶段显存 profiling 固定支付的开销，与并发无关，因此在
[34467029236](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34467029236)
中于并发 1 即 OOM（此时每 GPU 驻留权重约 35.9 GiB）—— 收窄并发列表无济于事。

因此 H100 配方将 `--max-num-batched-tokens` 限制为 4096，使 indexer 缓冲区降至 8 GiB。
改为收窄 `--max-model-len` 同样有效，但上下文上限会迫使一个服务 1M 上下文的模型使用
256k 截断语料，因此 batched tokens 才是正确的调节点。脚本还将 `--max-num-seqs` 设为
轨迹并发的两倍（而非沿用 vLLM 默认的 1024）、设置 `--gpu-memory-utilization 0.92`，
并启用 `expandable_segments`，因为失败的分配留下了 1.04 GiB 已保留但未分配的显存。

这些上限在调度任何 sweep 之前，已由单个并发 1 的 `agentx-fast` 运行
（[34485694183](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34485694183)）
验证 —— 这一顺序很重要：启动即 OOM 的全量 sweep 会浪费每一个 leg。该运行健康启动，并
报告了当前并发列表所依据的预算：

```
Available KV cache memory: 13.47 GiB
GPU KV cache size: 7,022,899 tokens
Maximum concurrency for 1,048,576 tokens per request: 6.70x
```

原始分支扫描并发 1–4，处于 6.70x 满上下文估算上限之下。后续 sweep 保持配方不变，
将并发扩展到 8 和 16，以测量 AgentX 的实际饱和曲线；如果多条轨迹同时接近 1M token，
这些点可能发生抢占。若要获得更多 KV，需要进一步缩小 indexer —— `--max-num-batched-tokens 2048` 可再释放约
4 GiB —— 代价是长轨迹 prefill 的分块更细。待有跨并发的吞吐数据后可重新权衡。

该分支通过其 srt-slurm 单节点配方（`srt-recipe:`）运行，配方将检出挂载到 `/infmax-workspace`，
避免在 `/workspace` 下创建 AgentX 运行目录。

来源：[上游配方](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml)。

### SGLang 上的 DeepSeek-V4.1-Flash DSpark

GB300 DeepSeek-V4.1-Flash SGLang 曲线使用 `dev-cu13-nightly-0924`：
C1/C2 使用 TP4/EP1、GPU Engram 和 4K prefill chunk；C4+ 使用 TP4/EP4、
主机 Engram 和 16K chunk。TP2 不变，仍待全量 sweep 验证。

H100 SGLang 候选配方在并发 1/2/4/8/16/20 下测试 DSpark。C1/C2 按并发数的 8 倍保留 SWA 前缀尾部，C4 及以上按 32 倍保留。相同条件下的一小时对比否决了统一的 128 尾部下限：C2 吞吐量仅提高 1.7%，交互性能却下降 44.5%。已完成的 STP 对比没有贡献实测性能前沿点，因此所选 sweep 不包含 STP。配方在预填充分块之间插入 16 步解码，轨迹内容和上下文限制保持不变。

同一 sweep 还会在 C4/C8/C16/C20 下验证受支持的 TP8/EP8/DP8 attention。DP 使用原生一致性哈希路由器与稳定会话键、DP LM-head，以及每 rank 64 个 SWA 前缀尾部。C16 的完整 GSM8K 已通过全部 1,319 个样本；其性能贡献仍在测量中。原生 1M 上下文与 AgentX 子代理/会话语义保持不变。

nightly 候选配方使用 `nightly-dev-cu13-20260922-582389ce`、原生 MXFP4 Marlin MoE，以及 `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank`。解析后的本地快照路径让上游分配器能够在分配匿名主机表之前清理检查点文件缓存。draft 精度遵循固定镜像的默认处理，包括 WO_A 从 FP8 到 BF16 的转换。block32 FP8 GEMM 使用 SGLang 的默认 tiling；自定义 block32 启动配置已在 [#3463](https://github.com/SemiAnalysisAI/InferenceX/pull/3463) 中移除。仍需完成规范全量 sweep 验收。


`dsv41flash-fp4-<sku>-sglang-agentic-dspark` 是 vLLM 配方在 h100、h200、b200、b300、gb200、gb300
与 mi355x 上的 SGLang 对应版本（每个 SKU 一个 PR），遵循
[SGLang cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1)。
该模型尚无正式发布的 SGLang 版本。B200、B300 与 H100 通过 digest 固定 CUDA 13 nightly 镜像
`lmsysorg/sglang:nightly-dev-cu13-20260922-582389ce`；GB300 通过 digest 固定 `lmsysorg/sglang:dev-cu13-nightly-0924`；GB200 与 H200 使用
`lmsysorg/sglang:nightly-dev-cu13-20260923-06008c17`（GB200 通过 digest 固定），MI355X 通过 digest 固定
`lmsysorg/sglang:dev-dsv41-mi35x`。以各主配置条目的 `image` 为准。

B200 在 TP4/EP4 C1–128 与 TP2/EP2 C1–8 全部使用上游默认 DSpark。
Engram 保留在主机 DRAM，设置 `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank`。
[run 35626514270](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/35626514270)
中共享主机表的大页覆盖率为零；每个 rank 的匿名分片无需修改主机 sysctl 即可申请大页。
必须从每个 rank 的启动日志核实实际大页比例。

| B200 拓扑 | 静态显存比例 | Prefill chunk | SWA 前缀尾部数 |
| --- | ---: | ---: | --- |
| TP4/EP4，C1–128 | 0.80 | 4096 | `max(128, min(4096, 64 * CONC))` |
| TP2/EP2，C1–8 | 0.92 | 2048 | `128 * CONC` |

两者均在 prefill chunk 之间执行 16 轮 decode，并将运行请求上限设为
`min(2 * CONC, 64)`。分块前缀缓存将 SWA 尾部与完整 KV 分开保留，因此完整缓存
占用率低并不证明有足够的可复用 SWA 容量。随并发调整的尾部预算与完整 KV 共享
固定静态内存池。正式扫描需验证实际缓存大小、临时内存、缓存复用率以及吞吐与交互性能前沿。

TP2 每 GPU 加载约 147.76 GiB 的目标和草稿权重。配方校验固定原始加载器的哈希，
启用 PyTorch 可扩展分配段，不应用引擎补丁。9 月 22 日 nightly 在独立 Slurm 诊断中
成功启动，并在 C8 完成全部 1,319 道 GSM8K，严格准确率为 97.65%。评测后的打包因
诊断脚本缺少环境变量而失败，之后单独恢复；这不等于官方工作流全绿。仍需完成最新镜像
的完整 sweep。草稿精度保留上游默认值。
镜像暂存（[`infx/launch/backends/slurm/squash.py`](../infx/launch/backends/slurm/squash.py)）在所有集群上将固定 Docker digest
转为 Enroot 的 `registry#repository:digest` 格式；遇到暂时性导入错误会重试，squash 仍无法校验通过时启动失败。

DSpark 使用固定官方 nightly 默认提供的精度，不应用自定义草稿量化或精度补丁。STP 不加载草稿模型；完整准确率和性能验证仍然必需。

GB200 固定官方 CUDA 13 nightly `20260923-06008c17`，manifest 为 `sha256:5921361fcf358cdde4df1968c941c14157f418613b099ad7f3e5aeed6427ae15`（ARM64 为 `sha256:d49261d2edd82fed2dd6254c33e68871ccf7a399498e059ec91dc4453a5808c3`）。拓扑与已发布 vLLM 一致：TP2/EP1 和 TP4/EP1，均覆盖 C1/2/4/8/16/32/64/128，不启用 DP attention。此前已暂存的 TP4/EP4 sweep 仅为历史证据，不能替代本次拓扑验证。

GB200 sweep 仅包含 `dsv41flash-fp4-gb200-sglang-agentic-dspark`。
移除尚无实测依据的 STP 条目；若要加入，须通过匹配测试证明其对性能前沿有贡献。
DSpark 使用固定官方 nightly 默认提供的精度，不应用自定义草稿量化或精度补丁。
完整准确率和性能验证仍然必需。

GB200 TP4 保留 `min(64*CONC, 1024)` 个 SWA prefix tails，并维持 static memory 0.70
和 chunk size 4096。此前 TP4/EP4 C16 实测保留 2,700 万个 full-context KV slots 与 439,040 个 SWA slots；
该上限避免高并发时耗尽实测 51.82 GiB KV 预算。仅 TP4 C16 使用 prefill/decode interval 16：
canonical 对比中 p90 interactivity 提升 13.65%，吞吐下降 0.30%，p90 TTFT 从 2.35 秒增至
3.51 秒。完整 GSM8K 的 1,319 个样本通过验证。其他并发点仍需完整 sweep；
C16 结果不能证明该设置在所有并发下均有收益。

GB200 TP2 使用 0.92 静态显存比例、2048-token 预填充块、`min(128*CONC,1024)` 个 SWA tails、16 的 prefill/decode interval，graph 与 running capacity 上限为 16 个请求。这些受支持的限制参考已完成的 B200 EP1 显存验证；GB200 仍须独立通过加载、图捕获、完整上下文缓存池以及全部性能和准确率测试。可扩展 CUDA allocator segments 仅减少碎片，不改变权重或精度。C64/C128 性能任务获得分区允许的最大 12 小时 allocation，工作流额外预留 30 分钟打包产物；完整预热、3600 秒计分时段及无样本限制的 1,319 题 GSM8K 均保持不变。

GB200 的主机表布局为 `per_rank`：计算节点内核通过 `madvise` 启用匿名大页，
而 `shmem_enabled=never` 阻止共享 memfd 布局使用大页。上游在匿名主机内存中
按行分片，并通过 `MADV_HUGEPAGE`/`MADV_COLLAPSE` 请求 512 MiB 大页。
该方式保留原始 FP8 表权重，并恢复两次 TP all-reduce；声称性能收益前，
须检查启动日志中的实际驻留内存与大页覆盖量。

DSpark 是检查点自带的草稿模型。SGLang 对它不提供 EAGLE 或 MTP 路径，也没有
`--speculative-num-steps` 参数；配方传入 `--speculative-algorithm DSPARK
--speculative-dspark-block-size 5`。吞吐测试通过 `SGLANG_SIMULATE_ACC_LEN`（`match-expected`、
`real-draft-token`）使用同一[已提交的黄金 AL](../infx/golden_al_distribution/dsv41flash_dspark.yaml)：
thinking 开启、五个草稿 token 对应 3.51；准确率 eval 保留真实验证。SGLang 对该模型默认关闭
thinking，因此脚本设置 `SGLANG_DEFAULT_THINKING=1` 与 `SGLANG_DSV41_REASONING_EFFORT=high`，
以测量黄金 AL 所采集的 thinking 开启状态。

并行方式遵循 cookbook 已验证的配置：Blackwell 与 MI355X 为 TP4/EP4，Hopper 为 TP8/EP8。
cookbook 会自动解析 attention、MoE 与 FP8 GEMM 后端，并警告手动覆盖会退回到较慢的 Triton
块 FP8 matmul；唯一例外是其 H200 配置显式指定 `--attention-backend dsv4
--moe-runner-backend flashinfer_mxfp4`，Hopper 配方与之保持一致。`--mem-fraction-static 0.8`
为 cookbook 的低延迟设置。`--max-running-requests` 为 `2 * CONC` 以容纳 AgentX 子代理扇出，
decode 图的 batch 覆盖该值，下限为 cookbook 的 64，上限为 128。

每个 SKU 在各自的 PR 中提供独立配方 `benchmarks/single_node/srt-slurm-recipes/dsv41flash/sglang/<sku>-fp4-mtp/agentic.yaml`。H100 不在 cookbook 的
硬件表中，其配方有所不同：Engram 表移至
单一共享主机副本（`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`，对应 vLLM 配方的 Engram CPU
offload），prefill 分块上限设为 4096，即 vLLM H100 配方在 80 GB 显卡上为稀疏注意力 indexer
缓冲区所需的相同批处理 token 上限。在测得 KV 上限之前，该配方并发止于 8。MI355X 也有独立
配方，包含 cookbook 的 ROCm 环境变量（`SGLANG_USE_AITER=1`、`SGLANG_MOE_PADDING=1`、
`AITER_FLYDSL_FORCE_REDUCE=1`、`ROCM_QUICK_REDUCE_QUANTIZATION=NONE`）、
`--disable-radix-cache`，以及上限 4096 token 的 breakable prefill 图。

所有配方的 KV cache 均常驻 GPU，因此 `kv-offloading: none`。srt-slurm 单节点路径对 `framework: sglang`
的 `dsv41flash` 处理方式与 vLLM 相同：检出挂载到 `/infmax-workspace`，检查点通过各集群的持久 HF 缓存解析。

在获得 GPU sweep 与 eval 证据之前，不得将这些配方视为已验证。

## 验证

运行覆盖被修改层的最小检查。

### YAML 解析

```bash
python3 -c "import yaml; yaml.safe_load(open('configs/<nvidia|amd>-master.yaml')); yaml.safe_load(open('configs/runners.yaml')); yaml.safe_load(open('perf-changelog.yaml'))"
```

### 基准语法和启动检查

```bash
bash -n benchmarks/<path>/<script>.sh
uv run python -c 'from infx.clusters import load_clusters; load_clusters()'
uv run pytest -q infx/tests/launch infx/tests/clusters
```

### 精确 key schema + 矩阵生成

```bash
uv run --no-project --exclude-newer PT12H --python 3.12 --with pydantic --with pyyaml \
  python -m infx.matrix.generate test-config \
  --config-files configs/<nvidia|amd>-master.yaml \
  --runner-config configs/runners.yaml \
  --config-keys <exact-key>
```

### 过滤后的配置族生成

```bash
uv run --no-project --exclude-newer PT12H --python 3.12 --with pydantic --with pyyaml \
  python -m infx.matrix.generate full-sweep \
  --config-files configs/<nvidia|amd>-master.yaml \
  --runner-config configs/runners.yaml \
  --model-prefix <prefix> \
  --framework <framework> \
  --precision <precision> \
  --runner-type <runner> \
  --seq-lens 8k1k
```

必须检查而非仅计数所生成的 `model`、`image`、`runner`、scenario、并发、`max-model-len`、TP/PP/EP/DCP/PCP、prefill/decode worker block、hardware、router、KV transfer、eval flag、`additional-settings` 和 `spec-decoding`。

如果修改了 schema 或生成器行为，运行其聚焦测试：

```bash
python -m pytest infx/tests/matrix/ -v
```

对 srt-slurm，还要运行该配方文档指定的上游 recipe checker/`srtctl` 命令。对 llm-d，要验证配方 YAML 并在目标 Slurm fleet 上实际检查 allocation/discovery 路径；本地矩阵生成无法证明 endpoint discovery。

## 避开 schema 和拓扑陷阱

强制规则来自 [`validation.py`](../infx/matrix/validation.py)，并在 [`configs/CONFIGS.md`](../configs/CONFIGS.md) 汇总：

- Schema 使用 `extra='forbid'`；必须精确使用 kebab-case alias。
- `conc-start` + `conc-end` 与非空 `conc-list` 二选一，绝不能同时使用。值必须为正数，start 不得大于 end。
- `pp`、`dcp-size` 和 `pcp-size` 是正整数。`dcp-size` 必须整除 `tp`。
- 每个 worker 的 GPU 需求为 `num-worker * tp * pp * pcp-size`；DCP 复用 TP GPU，不增加 allocation 乘数。
- 单节点拓扑字段位于搜索空间条目；多节点字段分别位于 `prefill` 和 `decode` 下。
- 异构 `hardware` 必须同时出现在两个 worker block，或两边都不出现。它记录结果元数据，不负责 runner 调度。
- `disagg: true` 要求 `multinode: true`，并要求在顶层或每个搜索空间条目提供 `kv-p2p-transfer`。
- `router` 和 `kv-p2p-transfer` 必须只在一个 scope 声明：顶层或搜索空间，不能两边都有。
- Router 元数据要求组件真实名称及 release/package/commit 版本；镜像 tag 不是组件版本。
- Agentic 配置要求精确 `cluster:<name>` runner。
- 设置字段只会生成 env/workflow 值。必须确认被选择的脚本实际消费它。
- Scenario 的 `max-model-len` 由 ISL + OSL + slack 推导；不要为 8k1k 配方硬编码 checkpoint 的完整上下文。

## 安全追加 changelog

来源：[`AGENTS.md#non-negotiable-benchmark-invariants`](../../AGENTS.md#non-negotiable-benchmark-invariants)、[`perf-changelog.yaml`](../perf-changelog.yaml)。

1. 先完成所有可执行配置变更，并识别精确 key。
2. 在 `perf-changelog.yaml` 物理文件末尾追加新 block：

```yaml
- config-keys:
    - <exact-key-or-intentional-wildcard>
  description:
    - "What changed"
    - "Image/topology/runtime detail"
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/<number>
```

3. PR 创建前，模型+硬件 playbook 允许 `pr-link: TBD`；创建 PR 后立即替换为真实 URL。
4. 绝不能 prepend、在中间按时间插入、排序、重新格式化，也不能对文件运行 formatter。
5. 绝不能删除或标准化现有空白，包括空白分隔行上的尾随空格。CI 依赖历史字节。
6. 如果文件与 `main` 冲突，恢复当前 `main` 版本，只重新追加本分支条目。不要手动合并已经重排的历史。
7. 请求 sweep 前解析文件，并确认生成的 changelog 选择包含预期 key。请求时必须且只能添加一个主 sweep 标签，通常为 `full-sweep-fail-fast`（参见 [PR 主标签与修饰标签](ci-procedures_zh.md#pr-主标签与修饰标签)）。

## 停止条件

出现以下任何条件时，在派发 GPU 工作或宣称配置完成前停止。取得缺失事实或修复来源不一致；不要猜测。

- 精确 checkpoint、精度、架构、原生上下文、框架、draft model/方法或镜像 tag 尚未验证。
- 没有覆盖目标模型/backend/SKU 的已验证同类项，且所需运行时参数或内存限制仍未知。
- runner 用户、共享挂载、预置模型路径、GPU 数、host DRAM、Slurm 行为或 root 文件清理未知。主机设置还必须先有 runner 注册凭据。
- 已注册 runner 不在任何 `cluster:<id>` 标签中、矩阵解析到不存在的脚本，或 runner 不是 **Idle**。
- 计算出的拓扑超过 fleet、DCP 不能整除 TP、异构 hardware 元数据只写一侧，或生成拓扑与目标配方不一致。
- srt-slurm 配方与主条目不一致、`model.container != image`，或尚未运行上游配方验证。
- llm-d 配方缺失并会意外 fallback、allocation 数不一致，或 endpoint discovery 无法满足 IPv4 字面量/唯一名称/有效端口规则。
- MTP 配方缺少 chat-template 基准、speculative 方法/token 数未验证，或 graph capture 超过 backend 上限。
- changelog 变更会修改历史字节、没有位于 EOF、存在冲突，或 PR 已准备请求 sweep 但仍保留 `TBD`。
- YAML、Bash、严格 schema、精确 key 生成、集群记录校验、启动测试或配方验证失败。

只有当所有可执行文件一致、精确 key 能生成、运行时路由存在、changelog 能选择该 key，且以上各层检查全部通过时，配置才可以进入 sweep。

## MI355X 上的 DeepSeek-V4.1-Flash

配方 `dsv41flash-fp4-mi355x-vllm-agentic-dspark` 将 [#2958](https://github.com/SemiAnalysisAI/InferenceX/pull/2958) 扩展至 MI355X AgentX：TP4 与 TP2、并发 1–64、原生五 token DSpark。吞吐测试使用[已提交的黄金 AL](../infx/golden_al_distribution/dsv41flash_dspark.yaml)：thinking 开启、五个草稿 token 对应 3.51，采用合成拒绝采样并关闭自适应验证。准确率 eval 保留真实块拒绝采样，但与 CUDA 分支不同，同样关闭自适应验证：它会在设备端裁剪验证请求，而 ROCm 的 `DeepseekV4IndexerBackend` 不支持该操作，启用后引擎拒绝启动（[运行 34651830283](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34651830283)）。FP4 表示 MXFP4 专家权重；检查点还包含 MXFP8 权重。

遵循已合并的[上游配方 #968](https://github.com/vllm-project/recipes/pull/968) 中的 AMD 设置：`VLLM_ROCM_USE_AITER=1`、`VLLM_ROCM_USE_AITER_MOE=1` 和 `--moe-backend aiter`。通用 AITER 选择器允许 vLLM 选择 CK a8w4 专家内核，与 DSV4-Pro MI355X 配方一致。配方通过 `WEKA_LOADER_OVERRIDE` 固定使用完整语料 `semianalysis_cc_traces_weka_062126`。KV 驻留 GPU。在 [vllm-project/vllm#57491](https://github.com/vllm-project/vllm/pull/57491) 将两处 `is_cuda()` 判断放宽为 `is_cuda_alike()` 之前，Engram 按上游 AMD 默认设置常驻 GPU。自该提交起，ROCm 会解析 `EngramConfig`，且 `cpu_offload` 经由 `VLLM_PLE_CPU_OFFLOAD` 默认开启，因此配方显式设置 `--engram-config`，而不依赖该默认值。TP=2 始终下放，因为此时表每 rank 需 94.4 GiB；TP=4 始终保持常驻，因为在并发 64 及以下 KV 池并非瓶颈。同样地，配方仅在并发高于 32 时调低 `--max-num-batched-tokens`：TP=2 c64 为 8192，因为稀疏注意力 indexer 及其配套的每 rank 缓冲区按每个批量 token 约 4.4 MiB 增长。当该分块低于 API server 默认 1024 序列所需的六倍时，`--max-num-seqs` 会被限制为 CUDA graph 捕获的规模：DSpark 每序列验证 1+5 个 token，4096 对 1024 序列会使 engram 投影在 profiling 阶段崩溃。所有情况下的原则一致：只在确实出现 KV 不足的并发点上把设备内存让给 KV，保持低并发处已验证的设置不变。早于该合并的镜像在 ROCm 上仍会拒绝该选项。MI355X launcher 使用共享 HF 缓存，并将此模型的仓库挂载至 `/ix`，同时导出 `INFMAX_CONTAINER_WORKSPACE=/ix`，确保 AgentX 依赖与输出路径位于该挂载中。

**GPU 验证：** 配方使用 `vllm/vllm-openai-rocm:nightly-rocm100-ac9126e58aa7bbab1856ba6593ba4d5003fea516`，该 ROCm 10.0 nightly 包含 vllm#58671（分页 MXFP4 稀疏 indexer）、vllm#58655（融合 mHC Triton seams）与 vllm#53492（Gluon sparse-MLA kernel）。[#3571](https://github.com/SemiAnalysisAI/InferenceX/pull/3571) 的[运行 36824408313](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/36824408313) 在 TP4 与 TP2、并发 1–64 下验证了该镜像，其中仅评测的 TP4 并发 64 数据点 GSM8K 为 strict 0.9712 / flexible 0.9704。此前的 sweep 验证的都是已被取代的镜像，其数据点不能沿用到本镜像：[#3420](https://github.com/SemiAnalysisAI/InferenceX/pull/3420) 重新扫描了 `nightly-rocm100-29468dde8b515031dc6d4d9d06bf0a2fa0442098`，[#3326](https://github.com/SemiAnalysisAI/InferenceX/pull/3326) 验证了 `nightly-rocm100-3df4ae153eb385e27b52f26c81f8edb9e20b9984`，[运行 34710937012](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34710937012) 覆盖的是 `nightly-eed1f3d0c6043bd494424a22443ee198dd56f657`。已合并的[上游配方 #1049](https://github.com/vllm-project/recipes/pull/1049) 记录了 MI355X 的 sparse-MLA Gluon kernel、MXFP4 稀疏 indexer 与 `--block-size 128`；已合并的 [#1006](https://github.com/vllm-project/recipes/pull/1006) 记录了 MI355X 的 TP2 Engram 卸载与 `--no-swa-bounded-replay`，已合并的 [#968](https://github.com/vllm-project/recipes/pull/968) 则记录了最初的 AMD 设置与完整的 InferenceX 命令。后续运行时证据请遵循 [AgentX 流程](./eval-agentx-procedures_zh.md)；仅有本地矩阵生成和镜像元数据不能证明 GPU 验证完成。

## MI300X 与 MI325X 上的 DeepSeek-V4.1-Flash

`dsv41flash-fp4-mi300x-vllm-agentic-dspark` 与 `dsv41flash-fp4-mi325x-vllm-agentic-dspark`
将已验证的 MI355X vLLM 配方复制到 gfx942，使用 ROCm 10.0 nightly
`nightly-rocm100-3df4ae153eb385e27b52f26c81f8edb9e20b9984`，以及相同的 AMD 设置
（`VLLM_ROCM_USE_AITER=1`、`VLLM_ROCM_USE_AITER_MOE=1`、`VLLM_USE_BREAKABLE_CUDAGRAPH=1`、
`--moe-backend aiter`、关闭自适应验证）。gfx942 不在上游硬件表中，且没有 FP4 MFMA：通用的
`aiter` MoE 后端允许 vLLM 选择器跳过仅 gfx950 可用的 CK a8w4 专家内核；若启动时所有候选均被
拒绝，首选修复手段是固定 `aiter_triton_mxfp4_bf16`（Triton W4A16 内核）。

两个配方均运行 **TP8** 而非 MI355X 的 TP4：192 GB（MI300X）或 256 GB（MI325X）显卡需容纳
511 GB 检查点的分片以及常驻 GPU 的 Engram 表（上游 AMD 默认，不做 CPU offload），并仍留出
1M 上下文 KV 池。MI300X 另将 `--max-num-batched-tokens` 上限设为 8192：稀疏注意力 indexer
在启动时分配 `[batched-tokens, max-model-len]` 的 fp8 logits 缓冲区（8192 时 16 GiB，MI355X
配方的 16384 时 32 GiB）。两者并发均为 1–32。两个条目还包含更小的布局（MI300X 的 TP4；MI325X 的 TP2 与 TP4），
通过 `--engram-config '{"cpu_offload":true}'` 将 Engram 表移至主机内存。

srt-slurm 单节点路径将检出挂载到 `/infmax-workspace`，使 AgentX 运行目录不落在 `/workspace` 下。MI300X
上的 HF 缓存为节点本地，每个节点上的首次运行需先下载 511 GB。在获得 GPU sweep 与 eval 证据之前，
不得将任一配方视为已验证。
