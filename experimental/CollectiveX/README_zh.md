# CollectiveX

<div align="center">

[English](./README.md) | **中文**

</div>

CollectiveX 是实验性的 MoE 专家并行通信基准，用于测量不同 EP 库和加速器系统的
dispatch、combine 及配对 roundtrip 延迟。

> 发布暂停：历史 schema 3-5 数据仅供诊断。目前没有数据集获准用于排名、推荐或回归基线。

## v1 执行配置

每个调度用例均采用 BF16、后端调优资源和 packed placement。显式指定的 mode 选择以下两个
契约之一：

- Normal mode 使用 `layout-and-dispatch-v1`、按 rank 去重的 token payload 和 activation-only
  combine。核心覆盖使用 uniform routing，并保留一个 Zipf 敏感性场景；EPLB 只作为 Zipf
  的修正方案测量。
- Low-latency mode 使用 `expert-packed-weighted-combine-v1`、token-expert payload 和
  gate-weighted combine，并且只调用真正的 DeepEP V1 或 UCCL low-latency API。该模式仅覆盖
  解码，绝不与 normal mode 共用排名 cohort。其他后端在此 suite 中均显式标为 unsupported。

两种模式统一使用 `fixed-512-v1`：64 trials x 8 timed iterations；每个 trial/point 的每个被测
组件前执行 32 次同步完整 roundtrip warmup。先测 roundtrip；每次 iteration 先取跨 rank 最大值，
再按 nearest-rank 计算 p50/p90/p95/p99，主要延迟指标为 roundtrip p99。stdlib 整数计数器
生成逐字节一致的 routing 和 gate weights。

规范矩阵覆盖 H100、H200、B200、B300、GB200、GB300、MI325X 和 MI355X。矩阵请求
608 个 cases / 1,600 个 token points：364 个可运行 cases / 940 个 points，并形成 58 个可执行
workflow shards/allocation cells；另有 244 个显式 unsupported cases / 660 个 points。
`sweep_matrix.py` 物化每个 token ladder，并拒绝缺失、过期、格式错误或被修改的 shard controls。
分片按 SKU round-robin 发出，使受限的 GHA matrix 尽早使用所有 runner pools。

| 系统 | EP8 | EP16 |
|---|---|---|
| H100/H200/B200/B300 | 1x8 NVLink，scale-up | 2x8 NVLink + RDMA，scale-out |
| MI325X/MI355X | 1x8 XGMI，scale-up | 2x8 XGMI + RDMA，scale-out |
| GB200/GB300 | 2x4 MNNVL，scale-up | 4x4 MNNVL，scale-up |

物理主机数量不能决定通信范围：两种 GB 拓扑都位于同一个 72-GPU MNNVL scale-up domain 内。

| 后端 | 当前范围 |
|---|---|
| DeepEP V1 | 镜像固定的 `deep_ep.Buffer`：提供 normal 和原生 low-latency API；x86 使用 upstream v1.2.1，arm64 使用镜像内 GB fork |
| DeepEP V2 | PR #605 `ElasticBuffer` 加 #630：scale-up 使用 LSA，x86 EP16 scale-out 使用 GIN；JIT 可复现并绑定 source/SASS |
| DeepEP Hybrid | 固定的 `HybridEPBuffer`：x86 EP16 使用 multi-domain RDMA/DOCA；GB EP8/EP16 位于同一个 MNNVL communication domain |
| UCCL | Hopper 上固定的 0.1.1 wheel 和 wrapper，提供 normal 和原生 low-latency API；Blackwell 显式标为 unsupported |
| NCCL/RCCL A2A | 可移植的 rank-deduplicated payload 加 expert/routing-metadata reference |
| MoRI | EP8 使用 MI325X AsyncLL 或 MI355X IntraNode；EP16 固定使用 2x8 XGMI + RDMA 上的 InterNodeV1 |

FlashInfer 不在 v1 范围内，因为已测试的 EP path 在运行时存在间歇性失败。该问题不会被误报为
平台能力限制；在证明有稳定的固定实现后可重新加入。

DeepEP V2 指 [DeepEP PR #605](https://github.com/deepseek-ai/DeepEP/pull/605) 引入的
`ElasticBuffer` 实现，而不是更新的 legacy `Buffer` build。固定 source 使用最小化的 upstream
[PR #630](https://github.com/deepseek-ai/DeepEP/pull/630) 后续修复：其 parent 是 #605 merge
tree，唯一 source 变更是修复 GIN 不可用时的纯 scale-up 初始化。Scale-up cases 请求 NCCL
Device API LSA；若实际建立的 LSA team 未覆盖整个 EP world，则直接失败。x86 EP16 scale-out
cases 必须使用启用 GIN 的 hybrid path，其精确拓扑为两个逻辑 scale-out domains（由两个物理
RDMA ranks 表示）、每个 domain 八个 scale-up ranks；GB EP16 仍是 MNNVL scale-up，因此继续
使用 LSA。隔离构建会记录 API、source、loaded libraries、generated JIT source、executable
SASS 与 raw CUBIN diagnostics。当前 H100 runner pool 被明确标记为 V2 unsupported，因为 NCCL
2.30.4 报告其 EP8 communicator 不具备 Device API symmetric-memory 支持；只有该 pool 的
runtime 支持全 rank CUDA P2P/LSA 后才能重新启用。其他 NVIDIA SKU 在 GPU outcome 通过 native
correctness 和 publication gates 前仍为 unvalidated。

v1 已移除的轴包括 cached-layout `[cl]`、runtime-visible `[rv]`、FP8、quantized combine、
额外 routing distributions、activation profiles、uneven allocation、placement permutations、
model envelopes 和 scaling studies。

## Workflow 与产物

`.github/workflows/collectivex-sweep.yml` 生成 public-SKU matrix，提取严格且被忽略的
`.shards/<id>.json` control，每个 shard 执行一次 allocation，对结果 JSON 做隐私检查并上传
raw GitHub artifacts。运行默认使用 `release_tag=unversioned`，仅供诊断。V1 运行必须显式选择
`release_tag=v1`；setup 随后要求固定的完整 matrix digest，并生成绑定 run、attempt 与 source 的
`cxrelease-v1-*` marker。Partial 或 filtered 运行无法获得该 marker。

`.github/workflows/collectivex-sweep.yml` 的 `publish-v1` 操作是显式的 V1 gate。它只接受一个位于
qualification index 1、成功且带 V1 tag 的首次尝试 sweep run ID，重新校验 GitHub metadata 与
release marker，并在 runner 本地可丢弃工作区中执行 `publisher.py`。只有完整通过 promotion、隐私检查和内容寻址的数据集才会以
`cxpublication-v1-*` 上传；raw artifacts 与 publisher private workspace 永不暴露给前端。

系统不需要 results server、attached store、Vercel storage、GCP、Neon、managed database 或
managed object store。前端使用已有的 server-side `GITHUB_TOKEN`，即时发现最新成功且按版本隔离
的 publication workflow，下载其 NDJSON artifact，校验 ZIP layout、UTF-8/NDJSON 结构、schema、
promotion 状态与 SHA-256，随后提供带版本的 channel URL 和 immutable dataset URL。UI 保留显式
benchmark-version selector；V2 及后续版本必须使用独立的 release tag 与 publication identity。
完整 validation contract 见 [docs/methodology_zh.md](docs/methodology_zh.md)。

## Runner 配置

Runner 本地 Slurm 和 storage 值使用严格的 per-SKU JSON 文档，路径为
`$XDG_CONFIG_HOME/inferencex/collectivex.json` 或 `COLLECTIVEX_OPERATOR_CONFIG`。该 mode-0600、
同 owner、非 symlink 文件位于 checkout 外且永不上传。未知 runners、fields、duplicate keys、
endpoint literals、unsafe paths 和非 JSON 输入均 fail closed；配置绝不作为 shell 执行。GHA
仅将加密的 `COLLECTIVEX_OPERATOR_CONFIG_V1` 内容传给 launcher；launcher 验证后只导出所选
SKU 的 allowlisted values，并在 allocation 前删除临时副本。必需 JSON fields 如下：

| SKU | 变量 |
|---|---|
| `h100-dgxc`, `b200-dgxc` | `partition`, `account`, `squash_dir`, `stage_dir` |
| `h200-dgxc` | `partition`, `squash_dir`, `stage_dir` |
| `b300` | `partition`, `account`, `squash_dir`, `stage_dir` |
| `gb200` | `partition`, `account`, 有序 `storage_roots` |
| `gb300` | `partition`, `account`, `squash_dir`, `stage_dir`, `enroot_cache_path` |
| `mi325x`, `mi355x` | `partition`, `squash_dir`, `stage_dir` |

每个已选中的非 MNNVL EP16 placement 还必须提供 `socket_ifname` 和 `rdma_devices`，用来指定
operator 审核过的 fabric；还可配置 allowlisted
`ib_gid_index` 与 `rdma_service_level`。CollectiveX 不会通过启发式规则选择 management route 或
HCA。Allocation 完成后，每个非 MNNVL scale-out 节点都必须证明所有已配置 interface 与 active
HCA port 存在，之后才允许初始化 backend。Scale-up 和 MNNVL job 会清除这些 overrides。
Scale-out NCCL/RCCL 固定使用 `IB` 与精确匹配的 HCA selectors；如果无法使用 RDMA，job 会失败，
而不会回退到 socket 后仍被错误标记为 RDMA。

`stage_dir` 必须是 checkout 与 workflow workspace 外预创建且由 runner owner 持有的 base，
不能经过 symlink，group 和 world 都不能写入，并且 runner 与所有 allocation 节点必须以相同路径
访问。Job 只创建带 marker 的 mode-0700 execution child，验证跨节点读写可见性，并在
allocation teardown 后只删除该 child；不会挂载 runner checkout，也不会在 AMD image storage
下创建 stage。

导入前，每个 Docker Hub tag 都通过有界 registry requests 解析，并且必须匹配固定 digest；拒绝
digest-qualified overrides。Enroot imports 使用固定 filesystem epoch 和带版本、绑定 registry
digest 的 cache key。每个已挂载 squash 都重新计算 hash，同时记录 verified registry digest 和
local squash hash。镜像提供的 DeepEP 会按精确 wheel 和 installed-file fingerprints 检查；
source-built backends 使用固定 commits 和 runtime-verified GPU targets。DeepEP V2 的 mode-0700
cluster-local build cache 由版本化 build recipe、verified image、architecture、upstream
trees 和 dependency pins 共同寻址；container 只看到固定的 `/cx-cache` mount，且该 cache 永不
进入 result artifacts。
固定的 V2 与 Hybrid source 在每个 workflow 中只获取一次。每个 job 都会验证完整 archive，仅
提取自身精确 backend root，只允许指向 archive 内 regular file 的受限相对 leaf symlink，并在
staging 前重新核对 Git tree 与 submodule pins。
Compute containers 仅接收显式 environment allowlist。Private host、address、device、NIC、
credential、workspace 和 path 数据只保留在加密配置、忽略的 operator notes 或有界 mode-0600
runner logs 中，永不上传。

## 本地检查

```bash
uv run --with-requirements experimental/CollectiveX/requirements.txt \
  python -m unittest discover experimental/CollectiveX/tests -p 'test_*.py'
uv run --with-requirements experimental/CollectiveX/requirements.txt \
  python experimental/CollectiveX/sweep_matrix.py --backends all --out /tmp/cx-matrix.json >/dev/null
uv run --with-requirements experimental/CollectiveX/requirements.txt \
  python experimental/CollectiveX/publisher.py --store-root "$COLLECTIVEX_STORE_ROOT" verify
bash -n experimental/CollectiveX/runtime/*.sh experimental/CollectiveX/launchers/*.sh
```

核心路径为 `capability.py`、`configs/`、`contracts.py`、`schemas/`、`sweep_matrix.py`、
`publisher.py`、`runtime/`、`launchers/` 和 `tests/`。
