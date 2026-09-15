# srt-slurm 配置

[English](./RECIPES.md) | **中文**

InferenceX 负责维护本目录中的配置。所有 NVIDIA srt-slurm 启动器均调用 [`runners/slurm_utils.sh`](../../../runners/slurm_utils.sh) 中的 `setup_srt_slurm()`，为作业创建固定版本子模块的本地 Git 克隆，并将整个目录复制到 `recipes/`。共享函数将实际提交记录到 `srt-slurm-sha.txt`；功耗测试路径还会将其复制到 `power-producer-sha.txt`，供结果校验使用。

统一版本由 [`utils/srt-slurm`](../../../utils/srt-slurm) 的 Git 子模块指针指定，目前为 [v2.2.1](https://github.com/NVIDIA/srt-slurm/releases/tag/v2.2.1)（`984180e5b8755aef85e9995048b5a16cb5336bce`）。升级时更新该子模块指针，然后运行配置和集成检查。不要在启动器中新增按模型选择检出版本的分支。

InferenceX 要求 srt-slurm 2.0 或更新版本，且配置必须声明 `schema: 2`。不支持旧版配置结构；加入本目录前必须先完成迁移。

## 目录和文件命名规范

所有配置统一存放在 `<model-prefix>/<engine>/<gpu>-<precision>/<workload>/<recipe>.yaml`：

```text
dsr1/sglang/b200-fp4/8k1k/disagg-stp-mtp-variants.yaml
glm5.2/sglang/h200-fp8/agentx/disagg-1p1d-pcp8-tp8-dp8-mtp6-hicache.yaml
qwen3.5/trtllm/gb300-fp4/agentx/disagg-1p7d-dep4-tep8-c7-b1-mtp-kvoffload.yaml
```

- 使用主配置中的 `model-prefix` 和 `precision` 标签。引擎目录为 `sglang`、`vllm`、`trtllm` 或 `tilert`；前端仍在配置内显式声明。硬件目录使用 `b200`、`gb300` 等 GPU 型号，不使用集群名称。
- 工作负载目录为 `1k1k`、`8k1k` 或 `agentx`。已有的跨序列长度配置集合放在 `fixed-seq-len` 下，保留其覆盖项选择器。
- 文件名使用小写字母和连字符，以 `agg` 或 `disagg` 开头。包含拓扑及用于区分同目录配置的关键参数，例如并行方式、批大小、并发数、MTP、卸载或缓存设置。避免日期、带序号的延迟/吞吐量标签，以及重复目录中已有的模型或硬件信息。
- 拓扑名中的 `1p4d` 表示预填充/解码 worker 数，不一定等于物理节点数。`p-tp4` 和 `d-tp8` 分别标识预填充和解码 TP；`b` 表示批大小，`c` 表示并发数。运行参数以 YAML 为准。
- 覆盖项集合使用 `*-variants.yaml` 命名。即使内容相同，也保留独立扫描入口：配置路径参与评估分组。Qwen3.5 的 `*-stp-sweep.yaml` 和 `*-mtp-sweep.yaml` 保留了这一既有区别。
- 移动文件时，同步更新当前及已弃用主配置中的 `CONFIG_FILE`、`EVAL_CONFIG_FILE`，以及启动器路径规则、工作流过滤器和本地文档。保留上游来源 URL，并保持历史性能变更日志不变。不为旧目录结构提供别名。

共享运行时资源保留在模型目录旁的 `configs/` 中，不属于独立基准测试配置。`configs/dsv4-moe-load-balancer-configs/` 中的四个文件原样取自 NVIDIA/srt-slurm 提交 `deb1dfd9934398664f92d194169c183e009da83b`，保留了 17 个 DSV4 TRT 配置使用的 EPLB 初始专家分配。`setup_srt_slurm()` 将这些文件复制到作业仓库的 `configs/` 目录，供配置中的绑定挂载使用。将配置文件放入本目录不会启用该配置；实际基准测试矩阵由主配置决定。

## TileRT 例外

当 `FRAMEWORK=tilert` 时，`setup_srt_slurm()` 直接从 SemiAnalysisAI/srt-slurm 分支仓库获取提交 `6bc3f306bdafa1edfb5dded2fcda8f1ccede1bde`，检出到作业目录。该版本为 [SemiAnalysisAI/srt-slurm#13](https://github.com/SemiAnalysisAI/srt-slurm/pull/13) 中支持 schema 2 的 TileRT 移植。这是唯一的备用检出路径；由于统一的 NVIDIA 版本尚未包含 TileRT 后端和路由器，该例外的固定提交在共享函数中指定。TileRT 使用与 NVIDIA 相同的 schema 2 配置结构和原生评估调度。TileRT 作业在准备阶段需要通过网络访问分支仓库。上游支持这些功能后，应删除此分支仓库例外。

## Schema 2 与主配置

配置使用 `schema: 2`、`engine` 和 `roles`。每个工作角色集中声明节点数、实例数、GPU 分配、环境变量和引擎参数。`resources` 保留 GPU 硬件信息，`placement` 控制前端和基准测试客户端的位置，`services` 描述辅助进程，`dynamo.source` 指定 Dynamo 软件包或源码提交。

| 配置字段 | `configs/nvidia-master.yaml` 字段 |
|---|---|
| `roles.prefill.workers` | `prefill.num-worker` |
| `roles.decode.workers` | `decode.num-worker` |
| `roles.prefill.args.tp-size`（SGLang） | `prefill.tp` |
| `roles.prefill.args.ep-size`（SGLang） | `prefill.ep` |
| `roles.prefill.args.enable-dp-attention` | `prefill.dp-attn` |
| `benchmark.concurrencies` | `conc-list` |
| 配置路径，可附带覆盖项选择器 | `additional-settings: CONFIG_FILE=recipes/...yaml` |

配置文件和主配置必须同步更新。启动器执行配置文件；主配置提供结果标签和调度元数据。聚合式配置使用 `roles.agg`；`roles.decode.nodes: colocate` 表示解码角色与预填充角色共享节点，不增加调度所需的工作节点数。

所有被引用的配置都必须纳入版本控制：srt-slurm 2 提供精选示例，不再携带历史 `recipes/` 目录。本次迁移补齐了 204 个此前依赖外部仓库的配置，并从 InferenceX 历史记录恢复了两个仍被引用的 AgentX 配置。主配置路径遵循上述目录结构，原有覆盖项选择器保持不变。

## 迁移与验证

在隔离环境中安装统一版本，然后使用其 CLI：

```bash
# 重写前先验证每个受支持的配置目录。
srtctl migrate --verify -f benchmarks/multi_node/srt-slurm-recipes/dsr1/sglang
srtctl migrate --in-place -f benchmarks/multi_node/srt-slurm-recipes/dsr1/sglang
# 对其他模型/引擎目录重复执行。
# 迁移 glm5.1/tilert/ 时，使用固定提交的 TileRT 分支仓库。
python -m pytest utils/matrix_logic/ -q
python -m infx.matrix.generate full-sweep \
  --config-files configs/nvidia-master.yaml \
  --framework dynamo-sglang dynamo-trt dynamo-vllm --multi-node
```

使用启动器指定的确切提交验证配置，包括全部覆盖变体。仅调整路径时，应按路径映射比较变更前后的生成矩阵；其他字段（包括评估选择和节点数）必须完全一致。本地配置校验通过不能替代完整硬件扫描和准确性评估。

本次迁移还修复了 `srtctl migrate` 无法自动处理的兼容性问题：

- SGLang Model Gateway 配置使用 `frontend.type: sglang-router`；在 v2.2.1 中，`sglang` 表示不经过路由器的独立工作进程。
- 对重复的 YAML 键，保留原 PyYAML 加载器实际采用的值。
- DCGM 遥测使用 `collect_interval_ms: 1000`，替代 `provider` 和 `default_frequency`。采集器自动推导退出等待时间；原先显式设置的十秒不满足当前校验要求。保留原配置中服务发现进程的专用节点部署方式。固定的上游版本不支持在专用基础设施节点上启用遥测；该功耗兼容性问题仍待解决，不通过改变原有拓扑来绕过校验。H200 自定义配置声明默认并发数，提交前由启动器替换。
- DeepSeek-V4 vLLM 基准测试使用受支持的 `custom_tokenizer` 加载器。删除已废弃的 `warmup_req_rate: inf` 字段；当前上游客户端的预热速率固定为每秒 250 个请求。
- 功耗读取器兼容两代 samples CSV，校验利用率字段，并继续根据瓦特数计算 GPU 板级能耗。
- 评估选择通过原生 `post_eval.command` 和 `post_eval.passthrough_env` 调用 [`srt_eval.sh`](../srt_eval.sh)。TRT AgentX 配置通过 `dynamo.source.git` 声明原有的 Dynamo 分支仓库，启动器不再改写 srt-slurm 源码。

每次修改配置或运行时，都必须在 `perf-changelog.yaml` 的物理末尾追加新条目，保留全部历史内容及空白。合并前使用 `full-sweep-fail-fast` 验证 PR（包括评估），再按仓库规定完成审查及产物复用合并流程。
