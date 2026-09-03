# 配置操作规程

<div align="center">

[English](./configuration-procedures.md) | **中文**

</div>

本页用于基准配置、配方、镜像和 runner 变更。它是操作规程而非字段目录；所链接的实现和 schema 始终是权威来源。

## 权威来源图

| 权威来源 | 控制内容 |
| --- | --- |
| [`configs/CONFIGS.md`](../configs/CONFIGS.md) | 主配置和 runner 配置的字段契约 |
| [`utils/matrix_logic/validation.py`](../utils/matrix_logic/validation.py) | 强制执行的 Pydantic schema 和拓扑不变量 |
| [`utils/matrix_logic/generate_sweep_configs.py`](../utils/matrix_logic/generate_sweep_configs.py) | 矩阵展开、过滤、runner 查找和生成的作业元数据 |
| [`configs/nvidia-master.yaml`](../configs/nvidia-master.yaml)、[`configs/amd-master.yaml`](../configs/amd-master.yaml) | 可执行的基准定义 |
| [`configs/runners.yaml`](../configs/runners.yaml) | 可调度标签、具体 runner 名称和硬件事实 |
| [`benchmarks/`](../benchmarks/) 和 [`runners/`](../runners/) | 运行时命令和 launcher 路由 |
| [`perf-changelog.yaml`](../perf-changelog.yaml) | 只允许追加的基准触发日志 |
| [`AGENTS.md`](../AGENTS.md) | 仓库级配置、MTP、changelog 和 sweep 规则 |

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

来源：[`docs/agent-guide.md`](./agent-guide.md)、[`AGENTS.md`](../AGENTS.md)。

从干净的仓库根目录开始：

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

详细来源：[`.claude/commands/add-model-hardware.md`](../.claude/commands/add-model-hardware.md)。字段来源：[`configs/CONFIGS.md`](../configs/CONFIGS.md)。
STP（Single Token Prediction，单 Token 预测）是每次前向传播生成一个 Token 的标准自回归解码。MTP（Multi-Token Prediction，多 Token 预测）通过原生预测头或投机解码在每次前向传播中预测多个 Token。

1. **固定身份。**确认精确 checkpoint ID、model prefix、精度、架构、原生上下文、目标 SKU、框架，以及解码方式是 STP、原生 MTP 还是 draft-model 推测。验证镜像 tag 确实存在；绝不能编造。
2. **选择两类同类项。**阅读同一模型在其他 SKU 上的实现，以及目标 SKU 上的另一个模型。还要阅读目标 [`runners/launch_*.sh`](../runners/) 和共享 [`benchmark_lib.sh`](../benchmarks/benchmark_lib.sh)。
3. **添加运行时脚本。**把单节点脚本放入 [`benchmarks/single_node/fixed_seq_len/`](../benchmarks/single_node/fixed_seq_len/)。保留已验证同类项中的 env 传递、parser 参数、attention/MoE backend、KV-cache dtype、graph/eager 模式、缓存设置和上下文处理。
4. **添加主配置条目。**`mi*` 使用 [`amd-master.yaml`](../configs/amd-master.yaml)，其他使用 [`nvidia-master.yaml`](../configs/nvidia-master.yaml)。精确设置 `image`、`model`、`model-prefix`、`runner`、`precision`、`framework`、scenario 和支持的搜索空间。
5. **依据证据确定规模。**复制已验证的并行布局并删除不支持的布局。延迟型 TP 行通常从并发 1 开始；不要把大显存 SKU 的 TP/EP 布局复制到小显存 SKU。
6. **检查 launcher 路由。**launcher 必须能解析新文件名，包括 framework 和 `_mtp` 后缀。模拟 STP 与 MTP 解析，并确认每个被选择的文件都存在。
7. **追加一条 changelog**，精确选择新 key；参见[安全追加 changelog](#安全追加-changelog)。
8. **验证语法和生成结果。**检查 image、model、runner、ISL/OSL、`max-model-len`、并发、TP/PP/EP/DCP/PCP 和 `spec-decoding`。

仅添加 `MODELS.md` 行并不会产生可执行配方。完整路径是：基准脚本 + 主配置条目 + launcher 路由 + changelog 触发项 + 生成的矩阵。

## 修改主配置

来源：[`configs/CONFIGS.md`](../configs/CONFIGS.md)、[`validation.py`](../utils/matrix_logic/validation.py)、[`generate_sweep_configs.py`](../utils/matrix_logic/generate_sweep_configs.py)。

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

## 注册并设置 runner

设置来源：[`utils/runner_setup/RUNNER_SETUP.md`](../utils/runner_setup/RUNNER_SETUP.md)。配置来源：[`configs/CONFIGS.md#runners`](../configs/CONFIGS.md#runners)。

### 仓库注册

1. 对新 fleet 创建 `runners/launch_<base-name>.sh`，或更新现有 launcher。
2. 在 [`configs/runners.yaml`](../configs/runners.yaml) 预期的 `labels:` key 下添加每个精确的已注册 runner 名称。新名称使用 `<base-name>_<NN>`，索引必须两位补零。
3. 如果生成过程需要 fleet 事实，添加匹配的 `hardware:` 条目，并设置正数 `available-cpu-dram-mib` 和 `gpus-per-node`。
4. 当事实依赖某个物理 fleet 时使用精确 `cluster:<name>` 标签；agentic 配置强制要求该标签。
5. 添加/更新主条目以使用该标签。生成目标矩阵并确认选择了正确的具体名称。

runner 名称前缀是关键契约：workflow 通过 `launch_${RUNNER_NAME%%_*}.sh` 路由。因此 `<base-name>` 必须匹配一个 launcher，且不得包含 `_`。

### 主机设置

1. 确定 runner 用户和共享存储。登录节点与计算节点都必须能看到 `_work`。
2. 确认注册 shell 的 `PATH` 中有 `curl`、`tar`、`tmux`；Slurm 还需要 `sinfo`/`srun`/`sbatch`。
3. 获取仓库管理员认证和新的注册 token；token 大约一小时后过期。
4. 按文档运行 [`setup.sh`](../utils/runner_setup/setup.sh)，传入 token、runner URL、索引范围、基础目录、基础名称和标签。
5. 用 [`start_runners.sh`](../utils/runner_setup/start_runners.sh) 启动。
6. 将 runner 加入 sweep 流量前，在[仓库 runner 设置页](https://github.com/SemiAnalysisAI/InferenceX/settings/actions/runners)确认每个 runner 都是 **Idle**。
7. 从计算节点验证 launcher 对 `_work`、HF cache、预置权重和 squash 镜像的挂载。root 容器不得在共享 workspace 留下 root 所有的文件。

## 注册 srt-slurm 配方

映射来源：[`benchmarks/multi_node/srt-slurm-recipes/RECIPES.md`](../benchmarks/multi_node/srt-slurm-recipes/RECIPES.md)。检入的配方：[`benchmarks/multi_node/srt-slurm-recipes/`](../benchmarks/multi_node/srt-slurm-recipes/)。

1. 定位精确的上游 [NVIDIA/srt-slurm](https://github.com/NVIDIA/srt-slurm) 配方，并记录固定到 commit 的来源路径。
2. 将 YAML 暂存到匹配的检入配方目录。阅读最接近的同类项和所选集群 launcher。
3. 将来源字段映射到主配置搜索空间条目：资源 worker 数 → `num-worker`；TP/EP/DP-attention → worker 拓扑；基准并发 → `conc-list`；配方路径 → `additional-settings: ["CONFIG_FILE=..."]`。
4. 在同一变更中添加/更新匹配的 [`nvidia-master.yaml`](../configs/nvidia-master.yaml) 条目。同步 worker 数、TP/PP/EP/DCP/PCP、hardware、router、传输引擎和并发标签。
5. 更新镜像时，使配方 `model.container` 与主配置 `image` 完全相同；launcher 使用主配置镜像作为 container alias key。
6. 运行配方所记录的 `srtctl` 验证，再生成主配置 key，并把每个前端标签/拓扑字段与配方逐一比对。
7. 追加 changelog 条目。

不得只提交一侧：`srtctl` 读取配方，而矩阵生成读取主配置。仅改配方可能给结果贴错标签；仅改主配置不会改变实际部署的配方。

## 注册 llm-d 配方

来源：[`benchmarks/llm-d/README.md`](../benchmarks/llm-d/README.md)、[`benchmarks/multi_node/llm-d/README.md`](../benchmarks/multi_node/llm-d/README.md)、[`llm-d-recipes/`](../benchmarks/multi_node/llm-d-recipes/) 和当前 [`llmd-vllm` 基准 wrapper](../benchmarks/multi_node/dsv4_fp4_gb200_llmd-vllm-disagg.sh)。

llm-d 不是 srt-slurm 路径：InferenceX 自己持有 Slurm allocation，并在每个节点启动一个容器。

1. 复制 [`benchmarks/multi_node/llm-d-recipes/`](../benchmarks/multi_node/llm-d-recipes/) 下最接近的 YAML，设置 EPP plugin/scheduling、角色特定 `extra-args`/`env`，以及可选 `slurm.time_limit`。
2. 添加/更新 `llmd-vllm` 主条目。设置 `multinode: true`、`disagg: true`、router 元数据、`kv-p2p-transfer`、prefill/decode worker 拓扑、并发，以及 `additional-settings` 中的 `CONFIG_FILE=<basename>.yaml`。
3. 保持 `PREFILL_NODES`、`DECODE_NODES`、`GPUS_PER_NODE` 和 worker 数与 allocation 及各角色 DP/TP/EP 布局一致。
4. 确认 [`submit.sh`](../benchmarks/multi_node/llm-d/submit.sh) → [`job.slurm`](../benchmarks/multi_node/llm-d/job.slurm) → [`server.sh`](../benchmarks/multi_node/llm-d/server.sh) 的传递，以及所选 wrapper/launcher 路由。
5. 验证文件发现：decode leader 生成 `/tmp/endpoints.yaml`；prefill endpoint 使用 vLLM 端口 8200，decode endpoint 使用 sidecar 端口 8000；名称唯一；地址为 IPv4 字面量；端口是 `1..65535` 范围内的字符串。
6. 确认 EPP 在 Envoy 收到流量前完成 discovery 加载，且角色标签为请求阶段选择正确的 prefill/decode backend。
7. 生成 key，检查拓扑和 `additional-settings`，再追加 changelog。

`CONFIG_FILE` 未设置或文件缺失时，会静默选择镜像内 `/etc/epp/config.yaml` fallback，并移除配方特定 vLLM 参数。除非明确打算使用 fallback，否则应将其视为验证失败。

GB200 DSpark AgentX 配置使用 `cluster:gb200-nv`：TP8/DEP8 跨两个节点（8 GPU），1P-DEP8/1D-DEP8 跨四个节点（16 GPU）。ARM64 镜像摘要包含 router v0.10.0；AgentX 不挂载旧版路由器二进制文件。使用 `token-load-scorer` 的配置必须显式声明 `inflight-load-producer`，以提供其依赖的未缓存 token 数据。launcher 使用预先存储的 `DeepSeek-V4-Pro-0813` checkpoint，而非旧版 V4-Pro 权重。

`recipe.py` 为吞吐测试注入 DSpark golden AL；`EVAL_ONLY=true` 保留真实验证。保持自适应验证关闭：它根据置信度缩减草稿 token 预算，会改变固定 K 的 golden AL。Mooncake 嵌入式存储即使设置 `enable_offload: false` 也属于 DRAM 卸载（该开关控制 SSD）。需声明 `kv-offloading: dram`、`kv-offload-backend: { name: mooncake }` 和 `dram-utilization`；运行时将每节点预算均分给四个 GPU rank。Mooncake 使用 InfiniBand HCA `mlx5_0,mlx5_1,mlx5_3,mlx5_4`；`mlx5_2` 和 `mlx5_5` 为以太网设备。普通 TP8/DEP8 声明 `none`。

服务发现排除无 API 的 TP follower。`agentic.sh` 检查各服务节点的 vLLM `/metrics`，导出 `AIPERF_METRIC_URLS` 和 `AIPERF_SERVER_METRICS_URLS`，再通过 AIPerf 的 `--server-metrics` 传递。Envoy 和 P/D sidecar 的指标不能替代 vLLM 指标。

## 更新镜像

来源：[`AGENTS.md#non-negotiable-benchmark-invariants`](../AGENTS.md#non-negotiable-benchmark-invariants)、对应主配置、运行时脚本与检入的 Recipe。

1. 验证精确的上游 registry tag 或 digest 确实存在，并适用于 CUDA/ROCm 和目标架构。
2. 找出所有受影响的配置 key、运行时脚本、Dockerfile 和检入配方。不要假设主 YAML 是唯一镜像引用。
3. 将主配置 `image` 与所需 env、参数、软件包版本或补丁作为一个一致变更更新。
4. 对 srt-slurm，更新 `model.container` 并保持其与主配置 `image` 完全一致。
5. 对 llm-d，区分主配置选择的服务镜像和 [`benchmarks/llm-d/Dockerfile`](../benchmarks/llm-d/Dockerfile) 中的构建来源；仅在构建契约变化时同时更新两者。
6. 追加选择全部受影响 key 的 changelog 条目（有意覆盖多个 key 时可以使用通配符），并列出旧/新版本及实质运行时变更。
7. 生成每个受影响的配置族，确认其运行时路径中没有残留旧 tag。

## 添加或修改 MTP

来源：[`AGENTS.md#non-negotiable-benchmark-invariants`](../AGENTS.md#non-negotiable-benchmark-invariants)、[模型+硬件 playbook 的 MTP 附录](../.claude/commands/add-model-hardware.md#appendix--mtp--eagle3-spec-decoding-variant)和现有 [`*_mtp.sh` 同类项](../benchmarks/single_node/fixed_seq_len/)。

1. 确认使用原生 MTP 模块还是外部 draft。使用 draft 时，从模型/上游配方验证精确模型 ID、方法（例如 `eagle3`）和建议 speculative token 数。
2. 复制相同模型和 backend 的可工作同类项。保留其 speculative config、attention backend、token 数、模型补丁和依赖设置。
3. 每个 `*_mtp.sh` 都必须向 `run_benchmark_serving` 传入 `--use-chat-template`；原始 prompt 会静默降低 acceptance。
4. graph capture 至少按 `CONC * (1 + NUM_SPEC_TOKENS)` 确定规模，采用同类项的取整方式，并限制在框架上限内（当前 vLLM playbook 上限为 2048）。
5. 保留 backend 差异：不要把 CUDA 专用 drafter attention pin 或补丁复制到 ROCm 配方。
6. 在相应搜索空间条目设置 `spec-decoding: mtp`，并添加 `_mtp` launcher 后缀路由。若使用 schema 支持的 draft-model 模式，要有意设置匹配的生成值；不要根据文件名推断。
7. 同时添加脚本 + 主配置条目 + launcher 路由 + changelog。
8. 运行 Bash 语法和生成检查；检查 `spec-decoding`、draft/native 方法、token 数、chat-template 使用、capture 范围和解析出的脚本。

## 验证

运行覆盖被修改层的最小检查。

### YAML 解析

```bash
python3 -c "import yaml; yaml.safe_load(open('configs/<nvidia|amd>-master.yaml')); yaml.safe_load(open('configs/runners.yaml')); yaml.safe_load(open('perf-changelog.yaml'))"
```

### 基准和 launcher 语法

```bash
bash -n benchmarks/<path>/<script>.sh
bash -n runners/launch_<cluster>.sh
```

### 精确 key schema + 矩阵生成

```bash
uv run --no-project --with pydantic --with pyyaml --python 3.12 \
  utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files configs/<nvidia|amd>-master.yaml \
  --runner-config configs/runners.yaml \
  --config-keys <exact-key>
```

### 过滤后的配置族生成

```bash
uv run --no-project --with pydantic --with pyyaml --python 3.12 \
  utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files configs/<nvidia|amd>-master.yaml \
  --runner-config configs/runners.yaml \
  --model-prefix <prefix> \
  --framework <framework> \
  --precision <precision> \
  --runner-type <runner> \
  --seq-lens 1k1k 8k1k
```

必须检查而非仅计数所生成的 `model`、`image`、`runner`、scenario、并发、`max-model-len`、TP/PP/EP/DCP/PCP、prefill/decode worker block、hardware、router、KV transfer、eval flag、`additional-settings` 和 `spec-decoding`。

如果修改了 schema 或生成器行为，运行其聚焦测试：

```bash
python -m pytest utils/matrix_logic/ -v
```

对 srt-slurm，还要运行该配方文档指定的上游 recipe checker/`srtctl` 命令。对 llm-d，要验证配方 YAML 并在目标 Slurm fleet 上实际检查 allocation/discovery 路径；本地矩阵生成无法证明 endpoint discovery。

## 避开 schema 和拓扑陷阱

强制规则来自 [`validation.py`](../utils/matrix_logic/validation.py)，并在 [`configs/CONFIGS.md`](../configs/CONFIGS.md) 汇总：

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
- Scenario 的 `max-model-len` 由 ISL + OSL + slack 推导；不要为 8k1k/1k8k 配方硬编码 checkpoint 的完整上下文。

## 安全追加 changelog

来源：[`AGENTS.md#non-negotiable-benchmark-invariants`](../AGENTS.md#non-negotiable-benchmark-invariants)、[`perf-changelog.yaml`](../perf-changelog.yaml)。

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
7. 请求 sweep 前解析文件，并确认生成的 changelog 选择包含预期 key。

## 停止条件

出现以下任何条件时，在派发 GPU 工作或宣称配置完成前停止。取得缺失事实或修复来源不一致；不要猜测。

- 精确 checkpoint、精度、架构、原生上下文、框架、draft model/方法或镜像 tag 尚未验证。
- 没有覆盖目标模型/backend/SKU 的已验证同类项，且所需运行时参数或内存限制仍未知。
- runner 用户、共享挂载、预置模型路径、GPU 数、host DRAM、Slurm 行为或 root 文件清理未知。主机设置还必须先有 runner 注册凭据。
- 已注册 runner 前缀没有匹配 launcher、矩阵解析到不存在的脚本，或 runner 不是 **Idle**。
- 计算出的拓扑超过 fleet、DCP 不能整除 TP、异构 hardware 元数据只写一侧，或生成拓扑与目标配方不一致。
- srt-slurm 配方与主条目不一致、`model.container != image`，或尚未运行上游配方验证。
- llm-d 配方缺失并会意外 fallback、allocation 数不一致，或 endpoint discovery 无法满足 IPv4 字面量/唯一名称/有效端口规则。
- MTP 脚本缺少 chat-template 基准、speculative 方法/token 数未验证，或 graph capture 超过 backend 上限。
- changelog 变更会修改历史字节、没有位于 EOF、存在冲突，或 PR 已准备请求 sweep 但仍保留 `TBD`。
- YAML、Bash、严格 schema、精确 key 生成、launcher 模拟或配方验证失败。

只有当所有可执行文件一致、精确 key 能生成、运行时路由存在、changelog 能选择该 key，且以上各层检查全部通过时，配置才可以进入 sweep。
