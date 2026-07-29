# 在两台 MI300X 节点上运行 Kimi K3

<div align="center">

[English](./kimik3-mi300x-native-multinode.md) | **中文**

</div>

本文档面向运维人员，介绍需显式启用的原生多节点路径：用普通 vLLM 在两台
8×MI300X 节点上服务 Kimi K3，并在其之上运行 AgentX。

> **该启动器不会下载权重。** Kimi K3 MXFP4 约 1.5 TB。如果某个节点上缺少权重，
> 作业会立即失败，并在报错中指出缺失的路径。预置权重是运行基准测试之前
> 单独执行的步骤。

## 支持范围

| | |
|---|---|
| 场景 | agentic-coding（AgentX trace replay） |
| 框架 | 普通 vLLM，聚合式（不做 P/D 分离） |
| 硬件 | 2 节点 × 8 张 gfx942 MI300X |
| 并行度 | 节点内 TP8，跨节点 PP2，EP1 |
| 并发 | 每个作业取 1、2、4 或 8 中的一个 |
| KV cache | 仅 GPU 常驻 |

其他配置都会在分配任何资源之前被拒绝。TP8 保持在单节点内，因此张量并行的
collective 不跨网络；PP2 才让模型放得下 —— 单节点的总 HBM 对 1.5 TB 的
checkpoint 没有安全余量。

MI325X、Kubernetes、P/D 分离、srt-slurm 和 DSpark 都不在本文范围内。

## 如何启用

矩阵 key 是 `configs/amd-master.yaml` 中的 `kimik3-fp4-mi300x-vllm-agentic`。
它的 prefill worker 带一个额外设置：

```yaml
additional-settings:
- "NATIVE_MULTINODE=1"
- "KIMIK3_NODELIST=chi-mi300x-043,chi-mi300x-054"
- "AITER_SITUV2_A8W4=0"
```

`NATIVE_MULTINODE=1` 选择原生多节点启动器；`KIMIK3_NODELIST` 将 allocation
固定到已经预置节点本地 snapshot 的节点对。`runners/launch_mi300x-amds.sh`
检查前者，然后交给 `runners/launch_mi300x-amds-native-multinode.sh`。其他
所有 MI300X 配置仍然走原本不变的单节点路径。

启动器要求下列环境变量，多节点 workflow 模板已经全部提供：

```text
IS_MULTINODE=true          SCENARIO_TYPE=agentic-coding    IS_AGENTIC=1
FRAMEWORK=vllm             MODEL_PREFIX=kimik3             PRECISION=fp4
SPEC_DECODING=none         DISAGG=false                    KV_OFFLOADING=none
PREFILL_NUM_WORKERS=1      PREFILL_TP=8                    PREFILL_PP_SIZE=2
PREFILL_EP=1               PREFILL_DP_ATTN=false
DECODE_NUM_WORKERS=0       DECODE_TP=8                     DECODE_PP_SIZE=2
DECODE_EP=1                DECODE_DP_ATTN=false
CONC_LIST=<1 2 4 8 之一>   KIMIK3_NODELIST=<已预置节点 A,已预置节点 B>
IMAGE  MODEL  RESULT_FILENAME  RUNNER_NAME
```

可选参数及其默认值：

| 变量 | 默认值 | 用途 |
|---|---|---|
| `KIMIK3_MODEL_CACHE_ROOT` | `/raid/hf-hub-cache/models--moonshotai--Kimi-K3` | 节点本地模型缓存 |
| `KIMIK3_NODELIST` | 必填；矩阵固定为 `chi-mi300x-043,chi-mi300x-054` | 持有已预置 snapshot 的准确节点对 |
| `KIMIK3_SQUASH_DIR` | `/raid/hf-hub-cache/inferencex/squash` | 节点本地镜像目录 |
| `HF_HUB_CACHE_MOUNT` | `/raid/hf-hub-cache/inferencex/agentx-hub` | client HF 缓存挂载的宿主机侧路径 |
| `HF_HUB_CACHE` | `/hf-hub-cache` | 容器侧路径；AgentX 在此缓存 trace 语料 |
| `KIMIK3_SLURM_TIME_MINUTES` | `480` | allocation 墙钟时间 |
| `KIMIK3_STARTUP_TIMEOUT_SECONDS` | `7200` | 等待 `/health` 的上限 |
| `KIMIK3_CLEANUP_TIMEOUT_SECONDS` | `120` | 清理时等待作业消失的上限 |
| `KIMIK3_PRESCANCEL_TIMEOUT_SECONDS` | `15` | 唯一需要在 `scancel` 之前完成的清理步骤的上限 |
| `KIMIK3_IMAGE_LOCK_TIMEOUT_SECONDS` | `3600` | 等待节点本地镜像锁 |

## 每个节点上的模型布局

`/raid` 是节点本地存储，因此一个节点上的 snapshot 对另一个节点不可见。
两个分配到的节点都需要各自的副本，且 revision 相同：

```text
/raid/hf-hub-cache/models--moonshotai--Kimi-K3/
├── refs/main                       # 40 位 revision
└── snapshots/<revision>/
    ├── config.json
    ├── model.safetensors.index.json
    └── model-*.safetensors
```

Hugging Face snapshot 通常包含指向同级 `blobs/` 目录的符号链接。因此 server
会只读挂载整个 `models--moonshotai--Kimi-K3` 缓存，并通过
`/models-cache/snapshots/<revision>` 解析模型；若只挂载 snapshot，weight
链接会失效。

`runners/mi300x_native_node_preflight.sh` 会在每个分配到的节点上运行，
按顺序检查：

1. 恰好 8 个 GPU agent，且全部为 `gfx942`；
2. `refs/main` 中是 40 位 revision；
3. snapshot 目录及其 `config.json` 存在；
4. `model.safetensors.index.json` 中列出的**每一个** shard 都存在且非空。

第 4 步最关键。同步不完整的 snapshot 在表面上看不出问题，直到几个小时后
模型加载失败才暴露。随后启动器会比对两个节点的记录：必须看到两个不同的
hostname、且 revision 一致，才会启动 server。

## 每个节点上的容器镜像

镜像逐节点导入到 `/raid/hf-hub-cache/inferencex/squash`。有三个路径被刻意
排除：

- `/home` 和 `/nvme_home` 是同一个几乎写满的 NFS 导出，而镜像约 30.8 GB；
- `/raid/squash` 无法由 `gharunner` 用户创建。

导入过程使用节点本地 `flock` 串行化，因此同一节点上的并行作业会等待而不是
互相竞争。已有效的镜像直接复用；缺失或损坏的镜像会重新导入到同目录下的临时
文件，用 `unsquashfs -s` 校验后才原子移动到位。如果导入中途失败，trap 只会
删除临时文件，绝不动此前已校验通过的镜像。

Enroot 的 cache 和 temp 目录也固定在同一个节点本地目录下，不会写到 `$HOME`。

## AITER_SITUV2_A8W4

镜像用 `AITER_ROCM_ARCH=gfx942;gfx950` 构建，但原始 Kimi K3 tuned MoE 表只覆盖
gfx950。单 GPU gfx942 probe 已经用常量和随机输入验证 A16W4 的 `tile_m=32`
路径，因此矩阵固定为：

```text
AITER_SITUV2_A8W4=0
```

入口脚本仍允许诊断运行显式传入 `0` 或 `1`，其他值会在分配资源前被拒绝。
启动时会打印最终取值，因此每次运行都会留下记录。

该 probe 没有覆盖 vLLM 启动 profiling 使用的
`token=4096, inter_dim=512`。未调优的 gfx942 fallback 原本会选择
`tile_m=128`，其两个 BF16 输入缓冲区需要 128 KiB LDS，而硬件上限只有
64 KiB。当前固定的 AITER overlay 只把 gfx94x A16W4 fallback 限制到
`tile_m=32`；gfx950 和非 A16W4 路径保持不变。

## 本地验证

以下检查都不需要 GPU、Slurm、容器导入或网络：

```bash
python3 -m pytest utils/matrix_logic/ -q
python3 utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files configs/amd-master.yaml \
  --config-keys kimik3-fp4-mi300x-vllm-agentic \
  --scenario-type agentic-coding \
  --no-evals
bash -n runners/launch_mi300x-amds.sh
bash -n runners/launch_mi300x-amds-native-multinode.sh
bash -n runners/mi300x_native_node_preflight.sh
bash -n benchmarks/multi_node/agentic/kimik3_fp4_mi300x_vllm.sh
```

矩阵生成应输出四行，并发分别是 1、2、4、8。

要在不启动任何服务的情况下查看某个 rank 实际会执行的 vLLM 命令：

```bash
KIMIK3_VLLM_DRY_RUN=1 MULTINODE_NODE_RANK=0 MULTINODE_MASTER_ADDR=<head-node> \
  bash benchmarks/multi_node/agentic/kimik3_fp4_mi300x_vllm.sh
```

## 集群门槛（按顺序）

启用 sweep 之前依次执行。每一步都是可以叫停的检查点。

1. 单 GPU gfx942 preflight：确认 AITER MoE 路径可用，并记录 a16w4 还是 a8w4
   有效。这个结果才最终确定 `AITER_SITUV2_A8W4`。
2. 在选定的节点对上预置目标 snapshot，且不使用 `/home`。
3. 在两个节点上运行节点 preflight。预期得到同一个 revision，且各自都有有效
   镜像。
4. 启动普通 vLLM，并对 rank 0 发一个直接请求。
5. 运行并发 1 的 AgentX。检查结果有效，并且没有残留的 Slurm 作业。
6. 运行完整的 1/2/4/8 扫描。

DSpark（PR B）只有在门槛 5 通过后才开始，这样分布式启动的问题不会被误判为
投机解码的问题。

## 故障排查

**`missing model revision pointer ...` 或 `missing model snapshot directory ...`**
该节点上没有预置 snapshot。预置后重跑；启动器不会替你下载。

**`missing weight shard(s) ...`**
snapshot 不完整。请把传输补齐，不要删掉 index 了事。

**`nodes hold different model revisions: ...`**
两个分配到的节点 snapshot 版本不一致。重新预置较旧的那个节点，或者把
allocation 固定到版本一致的节点对上。

**`imported image ... failed unsquashfs validation`**
导入产出的 squash 有问题。临时文件已被删除；检查 `/raid` 剩余空间后重跑。

**`the vLLM server step exited with code N before becoming healthy`**
某个 rank 在启动阶段挂了。server 日志最后 200 行会直接打印出来，完整日志作为
`multinode_server_logs.tar.gz` 上传。`--kill-on-bad-exit=1` 意味着一个 rank
失败会带走两个。

**`local memory (...) exceeds limit (...) in function 'moe_gemm1_0'`**
gfx942 A16W4 fallback 选择的 MoE tile 超过了 64 KiB LDS 上限。确认固定版本的
`fused_moe.py` overlay 已安装，并检查 AITER dispatch 名称包含 `t32`，而不是
`t128`。

**`the vLLM server did not become healthy within Ns`**
跨两个节点加载 1.5 TB 本来就慢。先在日志里确认两个 rank 都还活着，再考虑调大
`KIMIK3_STARTUP_TIMEOUT_SECONDS`。

**Slurm 作业在运行结束后仍然存在**
清理逻辑在每条退出路径（包括 `INT`、`TERM`、`HUP`）上都会调用 `scancel` 并
轮询 `squeue`，workflow 也会在每次运行前后按作业名取消。如果仍有残留，
执行 `scancel --name="$RUNNER_NAME"` 即可清除。
