# srt-slurm 配置

[English](./RECIPES.md) | **中文**

InferenceX 负责维护本目录中的配置。所有 NVIDIA srt-slurm 启动器均调用 [`runners/slurm_utils.sh`](../../../runners/slurm_utils.sh) 中的 `setup_srt_slurm()`，为作业创建固定版本子模块的本地 Git 克隆，并将整个目录复制到 `recipes/`。共享函数将实际提交记录到 `srt-slurm-sha.txt`；功耗测试路径还会将其复制到 `power-producer-sha.txt`，供结果校验使用。

统一版本由 [`utils/srt-slurm`](../../../utils/srt-slurm) 的 Git 子模块指针指定，目前为 [NVIDIA/srt-slurm#407](https://github.com/NVIDIA/srt-slurm/pull/407) 的合并提交。升级时更新该子模块指针，然后运行配置和集成检查。不要在启动器中新增按模型选择检出版本的分支。

## TileRT 例外

GLM-5.1 TileRT 配置暂时保留 schema 1。当 `FRAMEWORK=tilert` 时，`setup_srt_slurm()` 直接从 SemiAnalysisAI/srt-slurm 分支仓库获取提交 `d1e6c97b3baf3e87103b6d83189544c3c7d61c38`，检出到作业目录。这是唯一的备用检出路径；由于统一的 NVIDIA 版本尚未包含 TileRT 后端和路由器，该例外的固定提交在共享函数中指定。TileRT 作业在准备阶段需要通过网络访问分支仓库。上游支持这些功能后，应删除此例外并迁移对应配置。

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

所有被引用的配置都必须纳入版本控制：srt-slurm 2 提供精选示例，不再携带历史 `recipes/` 目录。本次迁移补齐了 204 个此前依赖外部仓库的配置，并从 InferenceX 历史记录恢复了两个仍被引用的 AgentX 配置。原有配置路径和覆盖项选择器仍可使用。

## 迁移与验证

在隔离环境中安装统一版本，然后使用其 CLI：

```bash
# 重写前先验证每个受支持的配置目录。
srtctl migrate --verify -f benchmarks/multi_node/srt-slurm-recipes/sglang
srtctl migrate --in-place -f benchmarks/multi_node/srt-slurm-recipes/sglang
# 对 vllm、trtllm 和其他 NVIDIA 目录重复执行；排除 tilert。
python -m pytest utils/matrix_logic/ -q
python -m infx.matrix.generate full-sweep \
  --config-files configs/nvidia-master.yaml \
  --framework dynamo-sglang dynamo-trt dynamo-vllm --multi-node
```

集成工作流安装启动器指定的确切提交，验证所有 schema-2 配置，包括全部覆盖变体。本地配置校验通过不能替代完整硬件扫描和准确性评估。

本次迁移还修复了 `srtctl migrate` 无法自动处理的兼容性问题：

- 对重复的 YAML 键，保留原 PyYAML 加载器实际采用的值。
- DCGM 遥测使用 `collect_interval_ms: 1000`，替代 `provider` 和 `default_frequency`。采集器自动推导退出等待时间；原先显式设置的十秒不满足当前校验要求。功耗配置将服务发现进程放在主节点上，确保采样和基准测试窗口使用同一时钟。H200 自定义配置声明默认并发数，提交前由启动器替换。
- DeepSeek-V4 vLLM 基准测试使用受支持的 `custom_tokenizer` 加载器。删除已废弃的 `warmup_req_rate: inf` 字段；当前上游客户端的预热速率固定为每秒 250 个请求。
- 功耗读取器兼容两代 samples CSV，校验利用率字段，并继续根据瓦特数计算 GPU 板级能耗。
- 评估选择通过原生 `post_eval.command` 和 `post_eval.passthrough_env` 调用 [`srt_eval.sh`](../srt_eval.sh)。TRT AgentX 配置通过 `dynamo.source.git` 声明原有的 Dynamo 分支仓库，启动器不再改写 srt-slurm 源码。

每次修改配置或运行时，都必须在 `perf-changelog.yaml` 的物理末尾追加新条目，保留全部历史内容及空白。合并前使用 `full-sweep-fail-fast` 验证 PR（包括评估），再按仓库规定完成审查及产物复用合并流程。
