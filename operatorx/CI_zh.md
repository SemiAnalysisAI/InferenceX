# OperatorX GitHub Actions

[English](CI.md) | **中文**

[OperatorX Sweep](../.github/workflows/operatorx-sweep.yml) 只通过 `workflow_dispatch` 运行（PR 只运行托管规划器）。

| GPU | 运行器标签 | 每节点 GPU 数 | 镜像平台 | 结果集群标识 |
| --- | --- | ---: | --- | --- |
| H100 | `cluster:h100-dgxc`（默认） | 8 | `linux/amd64` | `h100_dgxc_8x` |
| H200 | `cluster:h200-dgxc` | 8 | `linux/amd64` | `h200_dgxc_8x` |
| B200 | `cluster:b200-nscale` | 8 | `linux/amd64` | `b200_nscale_8x` |
| B300 | `cluster:b300-dsxe` | 8 | `linux/amd64` | `b300_dsxe_8x` |
| GB200 | `cluster:gb200-nv` | 4 | `linux/arm64` | `gb200_nvl72_4x` |
| GB300 | `cluster:gb300-nv` | 4 | `linux/arm64` | `gb300_nvl72_4x` |
| MI300X | `cluster:mi300x-amd` | 8 | `linux/amd64` | `mi300x_amds_8x` |
| MI325X | `cluster:mi325x-amds` | 8 | `linux/amd64` | `mi325x_amds_8x` |
| MI355X | `cluster:mi355x-amds` | 8 | `linux/amd64` | `mi355x_8x` |

## 触发运行

```bash
gh workflow run operatorx-sweep.yml --repo SemiAnalysisAI/InferenceX \
  --ref <branch> -f runner=cluster:h100-dgxc -f backends=vllm \
  -f testlists=gemm -f world_sizes=1 -f chunk_size=500 -f ingest=false
```

- 输入：`runner`、`backends`（`vllm`；AMD 另有 `torch`）、`testlists`、`mode`（`timing` | `counters`）、`world_sizes`（1、2、4、8）、`chunk_size`（1-500）、`recovery_run_id`、`ingest`（默认 true；测试运行设为 `false`）。
- `mode=counters`：每个算子在 Nsight Compute（NVIDIA）或 rocprofv3（AMD，12 轮采集：请用更小的 `chunk_size`）下运行一次；原始文件在 `results/counters/`；延迟受分析器干扰。
- 冒烟测试：`testlists=gemm_perf`、`chunk_size=50`。`attn_*` 测试列表与 gemm / moe 列表分开触发。

## 执行约定

- 规划：按后端镜像、`parallel` 拆分和 InferenceX 配方（`recipes.py`：镜像、启动环境变量、`vllm serve` 参数）分片，并按 `chunk_size` 切块；最多 256 个分片；没有配方的测试项使用 `containers.toml` 中的后端镜像。
- 每个分片独占一个 Slurm 节点（GB200/GB300：一个四卡托盘，不支持 world size 8）；分配时限 45 分钟，作业时限 70 分钟。
- 镜像：规划时解析摘要，在分配到的节点上用 enroot 导入，按镜像 + 摘要 + 架构缓存；规划后标签发生变化则分片失败。
- Rank：拆分的 world size 中每块 GPU 一个进程，通过 env:// 汇合；运行器设置来自 CollectiveX 平台配置，并由 `platforms.json` 覆盖。
- 结果：每个算子后保存检查点；不支持的测试项作为结果行保留，错误或没有成功行会使分片失败；产物为 `operatorx-manifest-<run_id>` 和 `operatorx-shard-<run_id>-<attempt>-<shard>`；除非 `ingest=false`，结果会导入 OperatorX 数据库。
- 清理：取消和 `always()` 步骤会取消分配并保留部分结果；清理失败后，用 `recovery_run_id` 重新运行单个分片。

## 本地测试

```bash
uv run --no-project --python 3.12 --with pytest --with pyyaml --with torch --with numpy \
  python -m pytest operatorx/tests/ -q
```
