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
- 覆盖项集合使用 `*-variants.yaml` 命名。仅在各配置间存在差异的多节点 AgentX 配置，按主配置条目合并为一个集合，通常为 `agg-variants.yaml` 或 `disagg-variants.yaml`：`base` 保存共享设置，每个原配置成为一个具名 `override_<name>` 块，只包含其差异（使用普通覆盖项，而非 `zip_override_*`）。主配置通过 `CONFIG_FILE=recipes/<dir>/<bundle>.yaml:override_<name>` 选择其一。启动器以文本方式读取的配置（例如带顶层 `telemetry:` 的功耗配置）保持独立文件。即使内容相同，也保留独立扫描入口：配置路径参与评估分组。Qwen3.5 的 `*-stp-sweep.yaml` 和 `*-mtp-sweep.yaml` 保留了这一既有区别。
- 移动文件时，同步更新当前及已弃用主配置中的 `CONFIG_FILE`、`EVAL_CONFIG_FILE`，以及启动器路径规则、工作流过滤器和本地文档。保留上游来源 URL，并保持历史性能变更日志不变。不为旧目录结构提供别名。

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
| `benchmark.concurrencies`（运行时绑定） | 所选 `conc-list` |
| 配置路径，可附带覆盖项选择器 | `additional-settings: CONFIG_FILE=recipes/...yaml` |

配方与主配置中的拓扑和调优设置必须保持同步。主配置提供服务镜像、模型、ISL/OSL
及所选并发数；InferenceX 在选择调优变体后绑定这些值。保留与调优设置配套的并发 zip
选择器，并显式声明独立固定的角色/辅助服务镜像。

本目录中活跃的原生固定序列长度文件是不完整配方片段，仅按原生 YAML 结构保存
配方特有设置。InferenceX 自动将其与扁平的
`configs/srt-recipes/fixed-sequence-multi.yaml` 共享块组合；单节点片段使用
`fixed-sequence-single.yaml`。无需逐配方源注册表、单独调优目录或 include/模板
语法。运行时与 `infx generate` 共用加载器，递归合并映射、替换列表，并在变体集合的
`base` 下应用共享字段。主配置提供模型、镜像、精度、长度和并发数；引擎调优及有意
设置的覆盖项仍保留在片段中。

固定序列脚本统一启用 chat template，使用 `0.8` 的 random-range ratio。
`infx generate` 仅向空输出目录写入完整原生配方和 manifest；生成配方不纳入版本
控制。命令及生成器范围详见
[运行时工作负载绑定](#运行时工作负载绑定)。

聚合式配置使用 `roles.agg`；`roles.decode.nodes: colocate` 表示解码角色与预填充
角色共享节点，不增加调度所需的工作节点数。

所有被引用的配置都必须纳入版本控制：srt-slurm 2 提供精选示例，不再携带历史 `recipes/` 目录。本次迁移补齐了 204 个此前依赖外部仓库的配置，并从 InferenceX 历史记录恢复了两个仍被引用的 AgentX 配置。主配置路径遵循上述目录结构，原有覆盖项选择器保持不变。

## 运行时工作负载绑定

所有活跃的原生固定序列长度配方仍保留在原有的
`benchmarks/single_node/srt-slurm-recipes/` 和
`benchmarks/multi_node/srt-slurm-recipes/` 路径下，只以原生 YAML 结构保存配方特有
设置。共享字段放在扁平的
[`fixed-sequence-single.yaml`](../../../configs/srt-recipes/fixed-sequence-single.yaml) 和
[`fixed-sequence-multi.yaml`](../../../configs/srt-recipes/fixed-sequence-multi.yaml) 块中。
InferenceX 根据 `IS_AGENTIC=0` 和 `IS_MULTINODE` 自动选择共享块；配方片段无需
include、模板语法、注册表或单独的调优文件。

运行时与 `infx generate` 均通过 `infx.srt_slurm.common.load_recipe` 合并共享块和
配方片段。映射递归合并，片段中的列表替换共享列表；对于变体集合，加载器将共享块
应用到 `base` 下。先选择调优变体，再绑定主配置值，保留与 CUDA graph 或批处理设置
配套的并发 zip 选择器。

| 主配置/运行时值 | 绑定的配方字段 |
|---|---|
| `image` / `IMAGE` | `model.container`，以及已有的 `identity.container.image` |
| `model` / `MODEL` | `model.path`、已有的 `identity.model.repo`，以及自定义客户端的 `benchmark.env.MODEL` |
| `precision` / `PRECISION` | 固定序列配方的 `model.precision` |
| `isl`、`osl` / `ISL`、`OSL` | 内置客户端的 `benchmark.isl` / `benchmark.osl`，或自定义客户端的 `benchmark.env.ISL` / `OSL` |
| 所选并发数 / `CONC`、`CONC_LIST` | 使用该字段时的 `benchmark.concurrencies`，以及自定义客户端的并发环境变量 |

模型加载保留集群映射与预先准备机制；执行时使用镜像缓存，溯源仍保留原始引用。
引擎量化、并行方式、调优、独立的角色/辅助服务镜像、有意设置的模型别名、tokenizer
覆盖项和 draft model 仍在片段中显式声明。`python3 -m infx.bench fixed-seq` 的 SRT 客户端统一为所有运行启用 chat
template，并将 random-range ratio 设为 `0.8`；配方不再重复或切换这些设置。

在 `inferencex-e2e/` 下为所选矩阵点生成完整原生 YAML：

```bash
uv run --extra recipes infx generate \
  --config-key qwen3.5-fp8-b300-sglang --output-dir /tmp/infx-recipes
```

输出目录必须为空，将收到 manifest 和完整原生配方，供检查并使用固定上游版本验证。
生成配方属于不纳入版本控制的输出，不要提交，也不要将不完整片段直接传给 `srtctl`。
生成器支持原生固定序列长度的单节点和多节点路径，明确拒绝 AgentX 和脚本工作负载。现有 AgentX 运行时绑定仍然可用。

## 迁移与验证

在隔离环境中安装统一版本。运行上游验证命令前，先生成完整原生配方；已提交的
固定序列片段不能作为独立的 `srtctl` 输入。例如，在 `inferencex-e2e/` 下运行：

```bash
uv run --extra recipes infx generate \
  --config-key qwen3.5-fp4-gb300-dynamo-sglang --output-dir /tmp/infx-recipes
python -m pytest infx/tests/matrix/ -q
python -m infx.matrix.generate full-sweep \
  --config-files configs/nvidia-master.yaml \
  --framework dynamo-sglang dynamo-trt dynamo-vllm --multi-node
```

使用启动器指定的确切提交验证配置，包括全部覆盖变体。当前迁移 CLI 已不提供 `--verify`；迁移后对每个选中的配置运行 `srtctl dry-run`，必要时通过 `--set` 提供启动器注入的值，例如 `benchmark.concurrencies`。仅调整路径时，应按路径映射比较变更前后的生成矩阵；其他字段（包括评估选择和节点数）必须完全一致。本地配置校验通过不能替代完整硬件扫描和准确性评估。

本次迁移还修复了 `srtctl migrate` 无法自动处理的兼容性问题：

- SGLang Model Gateway 配置使用 `frontend.type: sglang-router`；在 v2.36.0 中，`sglang` 表示不经过路由器的独立工作进程。
- 对重复的 YAML 键，保留原 PyYAML 加载器实际采用的值。
- DCGM 遥测使用 `collect_interval_ms: 1000`，替代 `provider` 和 `default_frequency`。采集器自动推导退出等待时间；原先显式设置的十秒不满足当前校验要求。保留原配置中服务发现进程的专用节点部署方式。固定的上游版本不支持在专用基础设施节点上启用遥测；该功耗兼容性问题仍待解决，不通过改变原有拓扑来绕过校验。H200 自定义配置声明默认并发数，提交前由启动器替换。
- DeepSeek-V4 vLLM 基准测试使用受支持的 `custom_tokenizer` 加载器。删除已废弃的 `warmup_req_rate: inf` 字段；当前上游客户端的预热速率固定为每秒 250 个请求。
- 功耗读取器兼容两代 samples CSV，校验利用率字段，并继续根据瓦特数计算 GPU 板级能耗。
- 评估选择通过原生 `post_eval.command` 和 `post_eval.passthrough_env` 调用 [`srt_eval.sh`](../srt_eval.sh)。TRT AgentX 配置通过 `dynamo.source.git` 声明原有的 Dynamo 分支仓库，启动器不再改写 srt-slurm 源码。

每次修改配置或运行时，都必须在 `perf-changelog.yaml` 的物理末尾追加新条目，保留全部历史内容及空白。合并前使用 `full-sweep-fail-fast` 验证 PR（包括评估），再按仓库规定完成审查及产物复用合并流程。
