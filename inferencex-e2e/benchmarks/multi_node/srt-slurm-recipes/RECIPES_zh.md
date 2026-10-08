# srt-slurm 配置

[English](RECIPES.md) | **中文**

InferenceX 负责维护本目录中的配置。每次 NVIDIA srt-slurm 启动时，srt 驱动（[`infx/launch/drivers/srt/checkout.py`](../../../infx/launch/drivers/srt/checkout.py)）都会为作业创建固定版本子模块的本地 Git 克隆，并将整个目录复制到 `recipes/`。它将实际提交记录到 `srt-slurm-sha.txt`；功耗测试路径还会将其复制到 `power-producer-sha.txt`，供结果校验使用。

统一版本由 [`utils/srt-slurm`](../../../utils/srt-slurm) 的 Git 子模块指针指定，目前为 [v2.43.4](https://github.com/NVIDIA/srt-slurm/releases/tag/v2.43.4)（`848f72d45b05af0fc082a9d1df49a8b4e7e61507`）。升级时更新该子模块指针，然后运行配置和集成检查。不要在启动器中新增按模型选择检出版本的分支。

InferenceX 要求 srt-slurm 2.0 或更新版本，且配置必须声明 `schema: 2`。不支持旧版配置结构；加入本目录前必须先完成迁移。

## 目录和文件命名规范

所有配置统一存放在 `<model-prefix>/<engine>/<gpu>-<precision>/<workload>/<recipe>.yaml`：

```text
dsr1/sglang/b200-fp4/8k1k/disagg-stp-mtp-variants.yaml
glm5.2/sglang/h200-fp8/agentx/disagg-1p1d-pcp8-tp8-dp8-mtp6-hicache.yaml
qwen3.5/trtllm/gb300-fp4/agentx/disagg-variants.yaml
```

- 使用主配置中的 `model-prefix` 和 `precision` 标签。引擎目录为 `sglang`、`vllm`、`trtllm` 或 `tilert`；前端仍在配置内显式声明。硬件目录使用 `b200`、`gb300` 等 GPU 型号，不使用集群名称。
- 工作负载目录为 `1k1k`、`8k1k` 或 `agentx`。已有的跨序列长度配置集合放在 `fixed-seq-len` 下，保留其覆盖项选择器。
- 文件名使用小写字母和连字符，以 `agg` 或 `disagg` 开头。包含拓扑及用于区分同目录配置的关键参数，例如并行方式、批大小、并发数、MTP、卸载或缓存设置。避免日期、带序号的延迟/吞吐量标签，以及重复目录中已有的模型或硬件信息。
- 拓扑名中的 `1p4d` 表示预填充/解码 worker 数，不一定等于物理节点数。`p-tp4` 和 `d-tp8` 分别标识预填充和解码 TP；`b` 表示批大小，`c` 表示并发数。运行参数以 YAML 为准。
- 覆盖项集合使用 `*-variants.yaml` 命名。仅在各配置间存在差异的多节点 AgentX 配置，按主配置条目合并为一个集合，通常为 `agg-variants.yaml` 或 `disagg-variants.yaml`：`base` 保存共享设置，每个原配置成为一个具名 `override_<name>` 块，只包含其差异（使用普通覆盖项，而非 `zip_override_*`）。主配置行通过 `srt-recipe: <bundle>.yaml:override_<name>` 选择其一，路径相对于条目的 `srt-recipe-dir`。即使内容相同，也保留独立扫描入口：配置路径参与评估分组。Qwen3.5 的 `*-stp-sweep.yaml` 和 `*-mtp-sweep.yaml` 保留了这一既有区别。
- 移动文件时，同步更新主配置中的 `srt-recipe-dir`、`srt-recipe` 和 `eval-srt-recipe` 引用，以及启动器路径规则、工作流过滤器和本地文档。保留上游来源 URL，并保持历史性能变更日志不变。不为旧目录结构提供别名。

共享运行时资源保留在模型目录旁的 `configs/` 中，不属于独立基准测试配置。`configs/dsv4-moe-load-balancer-configs/` 中的四个文件原样取自 NVIDIA/srt-slurm 提交 `deb1dfd9934398664f92d194169c183e009da83b`，保留了此前 DSV4 TRT 配置使用的 EPLB 初始专家分配；目前没有已提交的配置引用这些文件。srt driver（[`infx/launch/drivers/srt/checkout.py`](../../../infx/launch/drivers/srt/checkout.py)）将这些文件复制到作业仓库的 `configs/` 目录，供配置中的绑定挂载使用。将配置文件放入本目录不会启用该配置；实际基准测试矩阵由主配置决定。

## TileRT

TileRT 使用固定版本的上游 srt-slurm 子模块。配置指定 `roles.prefill.engine: vllm`、`roles.decode.engine: tilert` 和 `frontend.type: tilert-router`。

## Schema 2 与主配置

配置使用 `schema: 2`、`engine` 和 `roles`。每个工作角色集中声明节点数、实例数、GPU 分配、环境变量和引擎参数。`resources` 保留 GPU 硬件信息，`placement` 控制前端和基准测试客户端的位置，`services` 描述辅助进程，`dynamo.source` 指定 Dynamo 软件包或源码提交。

| 配置字段 | `configs/nvidia-master.yaml` 字段 |
|---|---|
| `roles.prefill.workers` | `prefill.num-worker` |
| `roles.decode.workers` | `decode.num-worker` |
| `roles.prefill.args.tp-size`（SGLang） | `prefill.tp` |
| `roles.prefill.args.ep-size`（SGLang） | `prefill.ep` |
| `roles.prefill.args.enable-dp-attention` | `prefill.dp-attn` |
| `benchmark.concurrencies` | `conc-list`（`power: true` 的行由绑定器写入） |
| `telemetry:` | 搜索空间条目的 `power: true` |
| 配置目录，相对于本目录 | 条目级 `srt-recipe-dir` |
| 配置文件，可附带 `base`、`override_<name>` 或 `zip_override_<name>[<index>]` 选择器 | 搜索空间条目的 `srt-recipe`；仅评估的真实验证使用 `eval-srt-recipe` |

配置文件和主配置必须同步更新。启动器执行配置文件；主配置提供结果标签和调度元数据。聚合式配置使用 `roles.agg`；`roles.decode.nodes: colocate` 表示解码角色与预填充角色共享节点，不增加调度所需的工作节点数。

所有被引用的配置都必须纳入版本控制：srt-slurm 2 提供精选示例，不再携带历史 `recipes/` 目录。本次迁移补齐了 204 个此前依赖外部仓库的配置，并从 InferenceX 历史记录恢复了两个仍被引用的 AgentX 配置。主配置路径遵循上述目录结构，原有覆盖项选择器保持不变。

## 配置片段

当前启用的配置（定长序列和 AgentX，单节点和多节点）均为片段：只包含该配置特有设置的原生 srt-slurm YAML。启动时先组合片段，再绑定测试点：

1. 通道的共享块（[`configs/srt-recipes/`](../../../configs/srt-recipes) 中的 `fixed-sequence-{single,multi}.yaml` 或 `agentic-{single,multi}.yaml`）合并到片段之下（配置集合则合并到 `base` 之下）。共享块设置基准测试客户端，多节点定长序列共享块还将 `benchmark.env.TOKENIZER` 指向挂载的检查点 `/model`。`power: true` 的多节点主配置行还会合并 [`telemetry-dcgm.yaml`](../../../configs/srt-recipes/telemetry-dcgm.yaml)，其导出器使用集群的 `srt-slurm.power-exporter-port`；该通道必须允许功耗测量（`POWER_LANES`）。片段优先：映射逐层合并，列表整体替换，因此片段可以保留 `collector_join_timeout_seconds` 等遥测调优参数。
2. 主配置行的选择器选出变体。单节点变体可以声明自身的 `benchmark.env.CONC`，使该并发数与其调优参数保持配对。
3. 绑定器（[`workload.py`](../../../infx/srt_slurm/workload.py)）将矩阵测试点写入选中的配置：`model.path: hf:<model>`、`model.container: <image>` 和 `model.precision`；片段声明了 `identity.container`/`identity.model` 时写入 `identity.container.image`（采用镜像仓库引用形式）和 `identity.model.repo`；启用遥测时写入 `benchmark.concurrencies`。定长序列配置还会获得 `benchmark.env.ISL`/`OSL`；单节点配置获得 `MODEL` 和 `CONC`，其中定长序列配置还获得 `RANDOM_RANGE_RATIO` 和 `USE_CHAT_TEMPLATE`（当且仅当配置启用投机解码时为 `true`）。多节点 AgentX 配置从启动器获得客户端的 `RESULT_DIR`、`AIPERF_DATASET_MMAP_CACHE_DIR` 和 `HF_HUB_CACHE`，启动器根据其挂载的卷（`volume-mounts`、`agentic-volume-mounts` 和通道挂载）推导这些路径；设置了 `HF_HOME` 的片段保留自身的缓存。多节点客户端从作业环境读取 `CONC_LIST`、`CONC`、`MODEL` 等矩阵输入，因此片段无需复制这些值；单节点 AgentX 测试点则通过运行时参数获得它们。

片段若设置了上述任一绑定键、`telemetry.enabled`、`benchmark.env.CONC_LIST` 或 AgentX 客户端路径（`RESULT_DIR`、`AGENTIC_OUTPUT_DIR`、`HF_HUB_CACHE`、`HUGGINGFACE_HUB_CACHE`，多节点还包括 `AIPERF_DATASET_MMAP_CACHE_DIR`），即使取值相同，也会在提交前失败。`hf:<model>` 解析为集群预置的检查点（`models.entries`），除非 `models.OVERRIDES` 中的某一行改用 Hub 快照；主配置镜像解析为预置的容器，因此配置不再引用别名。

配置不硬编码集群硬件信息：配置中任意位置的 `'@fabric.<name>'` 值在绑定后替换为作业所在集群的 `srt-slurm.fabric` 字段（列表以逗号连接；字段见 [CONFIGS.md](../../../configs/CONFIGS.md#runners)）。schema 未定义的名称、嵌在更长字符串中的引用，或集群未设置的字段，都会在提交前失败；生成矩阵时，若某行选中的变体所引用的字段在其运行器标签可达的任一集群上未设置，也已失败。启动前不会解析引用：配置指纹按原样对引用计算哈希，因此与集群无关。配置有意选择的值（例如重新排列的 rail 顺序或设备子集）保持字面值。

无需集群即可查看启动器实际提交的内容：

```bash
uv run --extra recipes infx generate --config-key 'dsr1-fp8-h200-*' --output-dir /tmp/recipes
```

该命令为每个定长序列或 AgentX 测试点及变体写出一个已绑定的配置，经固定版本的 srtctl 校验，并生成 `manifest.json`，将每个文件映射到对应的矩阵测试点，以及其运行器标签调度到的唯一集群（标签跨多个集群时为 `null`）。多节点测试点会获得启动器根据该集群推导的绑定器输入：通道允许该行的 `power: true` 时加入 DCGM 遥测块（导出器使用该集群的 `power-exporter-port`），以及 AgentX 客户端路径。需要这些输入的测试点，若其运行器标签未指向唯一集群，或通道拒绝其功耗测量，则会失败。fabric 引用取该集群的值；标签跨多个集群时保持引用原样。作业名称、健康检查下限和运行时 `--set` 值等启动时修改不会应用。

## 配置指纹

规划器为每个基准测试行计算 `recipe-fingerprint`：对该行的矩阵字段（`conc`、`exp-name` 和指纹本身除外）连同启动器为其提交的具体配置计算 SHA-256。该配置是与共享块组合后的片段（`power: true` 的行还包括遥测块）、该行选中的变体，以及绑定的模型、镜像、精度和定长序列的序列长度。并发值（`benchmark.env.CONC`、`CONC_LIST`、`benchmark.concurrencies`）和作业 `name` 不计入指纹，启动器为集群添加的内容（预置检查点和容器路径、挂载、AgentX 客户端缓存路径、导出器端口、网络 fabric 设置）也不计入，因此同一配置在其服务的各个并发数以及运行它的各个集群上保持同一个指纹。`eval-srt-recipe` 只贡献其路径，因为仅评估运行不产生基准测试结果。没有 srt-slurm 配置的行只对其矩阵字段计算哈希。

规划器与启动器一样选择变体，但不依赖 srtctl（[`variants.py`](../../../infx/srt_slurm/variants.py)），因此没有任何变体能服务的行会在规划阶段失败，而不是在启动时失败。编写 `perf-changelog.yaml` 条目前，先列出该变更使其结果失效的配置键：

```bash
uv run python -m infx.matrix.changed --base origin/main
```

基准侧矩阵由基准修订版自身的生成器生成，两侧均按各自的配置计算指纹。每个列出的配置键会显示其测试点数量，以及新增或移除的测试点（指纹和并发数）。`--config-files` 限定比较范围，`--json` 输出机器可读的报告。

## 迁移与验证

在隔离环境中安装统一版本，然后使用其 CLI：

```bash
srtctl migrate --in-place -f benchmarks/multi_node/srt-slurm-recipes/dsr1/sglang
# 对其他模型/引擎目录重复执行。
python -m pytest infx/tests/matrix/ -q
python -m infx.matrix.generate full-sweep \
  --config-files configs/nvidia-master.yaml \
  --framework dynamo-sglang dynamo-trt dynamo-vllm --multi-node
```

使用启动器指定的确切提交验证配置，包括全部覆盖变体。当前迁移 CLI 已不提供 `--verify`；迁移后对每个选中的配置运行 `srtctl dry-run`，必要时通过 `--set` 提供启动器注入的值，例如 `benchmark.concurrencies`。仅调整路径时，应按路径映射比较变更前后的生成矩阵；其他字段（包括评估选择和节点数）必须完全一致。本地配置校验通过不能替代完整硬件扫描和准确性评估。

本次迁移还修复了 `srtctl migrate` 无法自动处理的兼容性问题：

- SGLang Model Gateway 配置使用 `frontend.type: sglang-router`；在 v2.36.0 中，`sglang` 表示不经过路由器的独立工作进程。
- 对重复的 YAML 键，保留原 PyYAML 加载器实际采用的值。
- DCGM 遥测使用 `collect_interval_ms: 1000`，替代 `provider` 和 `default_frequency`。采集器自动推导退出等待时间；原先显式设置的十秒不满足当前校验要求。保留原配置中服务发现进程的专用节点部署方式。固定的上游版本不支持在专用基础设施节点上启用遥测；该功耗兼容性问题仍待解决，不通过改变原有拓扑来绕过校验。对于功耗测量行，绑定器根据测试点设置 `benchmark.concurrencies`。
- DeepSeek-V4 vLLM 基准测试使用受支持的 `custom_tokenizer` 加载器。删除已废弃的 `warmup_req_rate: inf` 字段；当前上游客户端的预热速率固定为每秒 250 个请求。
- 功耗读取器兼容两代 samples CSV，校验利用率字段，并继续根据瓦特数计算 GPU 板级能耗。
- 评估选择通过原生 `post_eval.command` 和 `post_eval.passthrough_env` 调用 [`srt_eval.sh`](../srt_eval.sh)。TRT AgentX 配置通过 `dynamo.source.git` 声明原有的 Dynamo 分支仓库，启动器不再改写 srt-slurm 源码。

每次修改配置或运行时，都必须在 `perf-changelog.yaml` 的物理末尾追加新条目，保留全部历史内容及空白。合并前使用 `full-sweep-fail-fast` 验证 PR（包括评估），再按仓库规定完成审查及产物复用合并流程。
