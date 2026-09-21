# 单节点 SRT-Slurm 迁移

[English](./single-node-srt-migration.md) | **中文**

本草稿将仍在使用的单节点服务设置迁移到原生 SRT-Slurm YAML。首个候选实现与现有路径并行构建，尚未切换生产路由。在取得实际运行证据之前，主配置、runner 选择、依赖固定版本和结果导入流程继续沿用现有实现。

## 分工与范围

Alec 负责单节点配方迁移。Cam 负责固定分叉版本、AMD 支持及 AMD 运行时清理。可复用的执行改动应保持小而独立，便于提交上游；最终目标是使用 NVIDIA SRT-Slurm，SemiAnalysisAI 分叉只是过渡依赖。本项工作不引入此前 H100 试点的独立 prepared-job 或发布系统。

在 InferenceX main `4ab85c1e` 中，排除 deprecated 归档后，活跃主配置包含 110 个单节点配置：

| 厂商 | SGLang | vLLM | TRT-LLM | ATOM | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NVIDIA | 47 | 13 | 10 | 0 | 70 |
| AMD | 21 | 8 | 0 | 11 | 40 |

这是待迁移清单，不代表当前 SRT 分叉已支持所有路径。AMD 功能和非 Slurm 执行需要分别确认。

## 首个原生配方

[`8k1k.yaml`](../benchmarks/single_node/srt-slurm-recipes/dsr1/sglang/h200-fp8/8k1k.yaml) 对应活跃配置 `dsr1-fp8-h200-sglang`，来源为 [`dsr1_fp8_h200.sh`](../benchmarks/single_node/fixed_seq_len/dsr1_fp8_h200.sh)。它使用原生 schema 2 和 `zip_override_concurrency` 展开：在一个 H200 节点上启动一个 TP8 聚合 worker，并发 4、8、16、32、64 分别运行独立作业。模型、镜像、8k1k 工作负载、服务环境和显式服务参数均取自现有配方。

SRT 负责资源分配、容器启动、端点选择、就绪检查及服务清理。账号、分区、挂载、镜像缓存和独占分配属于集群与启动器集成，不放入工作负载 YAML。候选配方关闭 SRT observability，避免在现有 GPU 采样器之外增加一套遥测负载；验收时仍须比较实际生效的原生默认设置。

原生 `custom` benchmark 调用 [`srt_fixed_sequence.sh`](../benchmarks/single_node/srt_fixed_sequence.sh)，复用现有 `run_benchmark_serving` 和 GPU 采样器。它保留 `10 * concurrency` 个请求、`2 * concurrency` 次预热、随机长度变化、completions API 行为和现有 JSON 结果格式。共享 helper 新增显式 base URL 参数，让客户端连接 SRT 选定的端点；现有调用仍使用原来的本地端点。

客户端在服务镜像中运行，并保留原有 `sentencepiece` 安装步骤。这是客户端兼容胶水，不是第二套服务启动器。初版客户端明确拒绝 eval 请求；切换前必须完成 eval-only 上下文处理、评测产物接入和取消验收。

## 运行时输入

配方提供模型及工作负载参数，并通过原生覆盖展开并发。SRT 提供 `SRT_FRONTEND_HOST` 和 `SRT_FRONTEND_PORT`。启动集成须在提交前导出 `INFMAX_WORKSPACE`，由 SRT 挂载到 `/infmax-workspace`，并提供以下原生覆盖：

| 原生覆盖字段 | 调用方负责的值 |
| --- | --- |
| `benchmark.env.RESULT_FILENAME` | 现有 InferenceX 结果文件基本名称 |
| `benchmark.env.RESULT_DIR` | `/logs`，SRT 已创建的作业产物挂载 |
| `benchmark.env.GPU_MONITOR_INTERVAL` | 显式指定的采样间隔秒数 |
| `benchmark.env.RUN_EVAL` | 吞吐试点使用 `"false"` |
| `benchmark.env.EVAL_ONLY` | 吞吐试点使用 `"false"` |

上述环境值均为字符串；通过原生 `--set` 传递时使用带引号的 YAML 值。缺失输入会在客户端启动前报错，工作负载配方不隐藏运行时回退设置。提交应经过共享 `apply_srt_recipe`，确保后续推测解码迁移保留自动 golden-AL 选择。

安装 SRT 依赖后，可在本地渲染资源分配：

```bash
INFMAX_WORKSPACE="$PWD" PYTHONPATH=utils/srt-slurm/src srtctl dry-run \
  -f 'benchmarks/single_node/srt-slurm-recipes/dsr1/sglang/h200-fp8/8k1k.yaml:zip_override_concurrency[0]'
python -m pytest utils/test_srt_fixed_sequence.py
```

未提供集群 profile 时，该命令使用 SRT 的通用调度默认值。它验证配置结构，不代表集群或基准已经验收。

## 显式启用的工作流试点

[`configs/pilots/h200-srt.yaml`](../configs/pilots/h200-srt.yaml) 仅选择 `cluster:h200-dgxc`、8k1k、TP8 和并发 4。搜索空间的 `srt-recipe` 字段将原生文件及选择器经矩阵和工作流传给现有 H200 池启动器。生产 `h200` 覆盖（包括 CoreWeave）保持不变。

启动器在提交前核对配方与矩阵中的模型、镜像、精度、拓扑和工作负载。它使用集群已暂存的模型路径，将配方指定的准确镜像 URI 交给原生 SRT/Pyxis 启动容器，申请独占节点，并通过原生 `--set` 绑定并发与产物参数。模型资源缺失会在提交前失败；试点不依赖旧启动器单独管理的 squash 缓存。普通 `sglang` 提交也经过共享的自动 acceptance 连接器。

提交使用原生 JSON 输出。启动器验证 Slurm 分配成功结束，保留结果文件名，并将原始结果和 GPU 采样附属文件交给现有处理及上传流程。原生日志归档、提交清单和 SRT commit 用于追踪运行来源。失败时保留已有产物，取消操作仅针对本次提交的作业。

从草稿分支触发工作流，将 `ref` 设为已推送的准确 commit：

```bash
gh workflow run e2e-tests.yml --ref codex/single-node-srt-slurm \
  -f ref=<COMMIT> -f test-name='native H200 SRT pilot' \
  -f generate-cli-command='test-config --config-keys dsr1-fp8-h200-sglang --config-file configs/pilots/h200-srt.yaml --no-evals' \
  -f require-power=true
```

试点必须传 `--no-evals`，目前仍明确拒绝 eval。一次吞吐运行通过不代表准确性或性能一致性已验收。

## 启用替代路径之前

- 保留两个 H200 runner 路径：当前 `h200` 标签同时包含 `h200-dgxc-slurm` 和 `h200-cw`。不能静默删除 CoreWeave 覆盖，也不能把其 Docker 执行视为已被 Slurm 配方覆盖。
- 接入 eval 上下文、真实验证评测和评测产物准备，并验收已接入的结果与 GPU 功耗路径，不改变发布格式。
- 比较旧路径与原生路径的命令，在相同镜像、模型和硬件上验收启动、吞吐、准确性、功耗、取消及清理。协调现有 smoke/vendor 评测工作，避免重复执行。
- 首条路径验收后，再扩展到其他活跃单节点配方，包括推测解码和 KV offload；AMD 能力与 Cam 的分叉工作协调。只有调用方完成迁移后才删除旧脚本。

迁移 PR 保持草稿。本地 schema 检查和使用外部进程桩的客户端测试不能证明性能一致或 GPU 验收通过。
