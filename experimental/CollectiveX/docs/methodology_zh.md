# CollectiveX EP v1 契约

<div align="center">

[English](./methodology.md) | **中文**

</div>

本文档定义新的 CollectiveX 结果。历史运行笔记是 evidence，不是 contract。

## 产品边界

CollectiveX 是通信 microbenchmark，用于：

- 在同一 chip/topology 上比较 EP libraries；
- 在相同 workload 下比较不同系统的 EP latency 和 logical payload bandwidth；
- 展示 unsupported、failed、invalid 和 unstable evidence，同时避免污染决策。

若没有单独的 correlation study，它不能预测 serving throughput。

## 矩阵

提升后的 workload 为 `deepseek-v3-v1`：hidden 7168、top-k 8、256 routed experts、BF16
dispatch 和 combine、packed placement，以及 backend-tuned resources。每个 case 都显式选择
normal `layout-and-dispatch-v1` 或 low-latency `expert-packed-weighted-combine-v1` 语义。

- `ep-core-v1`：uniform routing；decode T=1..128 的 2 次幂；prefill T=256/512。
- `ep-routing-v1`：Zipf，EPLB off/on；decode T=128；prefill T=512。
- `ep-low-latency-v1`：使用 DeepEP V1/UCCL 原生 low-latency API；uniform decode T=1..128 的
  2 次幂；capability contract 会拒绝其他后端，不会伪造 low-latency path。
- 规范矩阵范围：请求 608 个 cases / 1,600 个 token points；364 个可运行 cases / 940 个
  points，分布在 58 个可执行 workflow shards/allocation cells；244 个 unsupported cases / 660 个
  points。

| 系统 | EP8 | EP16 |
|---|---|---|
| H100/H200/B200/B300 | 1x8 NVLink，scale-up | 2x8 NVLink + RDMA，scale-out |
| MI325X/MI355X | 1x8 XGMI，scale-up | 2x8 XGMI + RDMA，scale-out |
| GB200/GB300 | 2x4 MNNVL，scale-up | 4x4 MNNVL，scale-up |

物理主机数量不能定义通信范围。两个 GB 配置都位于同一个 72-GPU MNNVL scale-up domain 内。

Unsupported combinations 是 terminal outcomes，不会被静默跳过。DeepEP V2 指 PR #605
引入的 `ElasticBuffer`，并固定使用 upstream PR #630 的最小纯 scale-up 修复。V2 的 scale-up
cases 请求 NCCL Device API LSA；若实际建立的 LSA team 未覆盖整个 EP world，则直接失败。x86
EP16 scale-out 使用启用 GIN 的 hybrid path，并要求两个逻辑 scale-out domains（由两个物理 RDMA
ranks 表示）、每个 domain 八个 scale-up ranks。GB EP16 仍是 MNNVL scale-up，因此使用 LSA。
Source 中声明的 NVIDIA capabilities 在 GPU outcomes 通过 native oracle 和 publisher gates 前仍为
unvalidated。当前 runner pool 上的 H100 V2 在 v1 中被声明为 unsupported，因为 NCCL 2.30.4
报告其 EP8 communicator 不具备 Device API symmetric-memory 支持；只有该 pool 恢复全 rank
CUDA P2P/LSA 支持后才能重新加入。已移除的轴包括 `[cl]`、`[rv]`、quantization、alternate
activation/routing profiles、uneven allocation、placement
permutations、model envelopes 和 scaling。
FlashInfer 因可重复出现的间歇性执行失败而排除在 v1 外；这些失败不会转为 planned-unsupported
coverage。
MoRI EP8 在 normal mode 下使用 MI325X AsyncLL 或 MI355X IntraNode。EP16 固定使用 2x8 XGMI +
RDMA 上的 InterNodeV1，配置为 96 blocks、64 RDMA blocks、8 warps、每个 PE 一个 QP，以及
external input。MoRI 的 AsyncLL transport 不属于真正的 low-latency suite contract，也绝不会
以该模式标注。

## Workload 身份

一个 canonical workload 在 global token batch 上生成，再按 source rank 切分。Expert indices
和 gate weights 会序列化。Activations 使用带版本的整数计数器公式，其 BF16 值在不同 runtime
中精确一致；完整身份绑定到 manifest。Manifest 还绑定 shape/EP coordinates 和 oracle version。
SHA-256 覆盖 canonical bytes 和 parameters；重新生成 library RNG 不能证明身份一致。

Routing traffic 区分：

- token-expert assignments，决定 expert compute load；
- rank-deduplicated token payload copies，决定 EP activation traffic。

Adapters 不得生成 routing，也不得将两种量相互解释。

## 测量

Normal mode 使用 `layout-and-dispatch-v1`：dispatch timing 包括 layout 和 communication，combine
通过 unweighted rank-sum path 返回 activation payload。Low-latency mode 使用
`expert-packed-weighted-combine-v1`：DeepEP V1/UCCL 原生 API dispatch token-expert assignments，
并执行 gate-weighted combine。Expert-output staging 不计入 isolated combine timing，但计入被测
paired roundtrip。每个 component 声明 availability、origin、start/end states、stage scope 和 sample
count。仅有 paired API 时，isolated components 报 null。`isolated_sum` 为派生值，不用于
throughput 或 recommendations。Mode 属于 series identity；normal 和 low-latency evidence 不能
共用排名 cohort。

每个被测 component 均使用 `fixed-512-v1`：

- 64 trials x 8 timed iterations = 512 observations；
- 每个 trial/point 的每个可用被测 component 前，执行 32 次同步完整
  dispatch-stage-combine warmups；
- 先测 roundtrip，再测 isolated dispatch 和 combine，并使用固定的 per-phase conditioning ladder；
- 每次 iteration 先取跨 rank 最大 latency，再以 nearest-rank 计算 p50/p90/p95/p99。

被测 roundtrip p99 是 headline latency。Retries 保持为独立 attempts；后续成功不会抹除早期失败。
Decode 和 prefill 表示一个 MoE-layer collective 所代表的 serving regime；在其他 shape 相同时，
它们不会改变 timed primitive。

NCCL/RCCL reference 是 end-to-end Python adapter，而不是 bare fabric primitive。其 dispatch
boundary 包含 layout、count exchange、device-to-host split synchronization、fresh receive
allocation，以及四次 payload/metadata all-to-all；activation-only combine 还包含一次 all-to-all 和
scatter/reduction。因此其 p99 测量完整 reference-adapter boundary，可能对 host/scheduler 敏感。
它可作为 portable system control，但不得标记为 fabric、link、bus 或 single-collective latency。

带版本的 conditioning 和 EPLB planner contracts（reference trace、redundant count 和
placement/remap version）属于 scheduled 和 evidence identity。

Logical payload bandwidth 为：

`logical_payload_bytes / measured_latency_seconds`

Normal-mode payload bytes 使用按 rank 去重的 token-rank activations；low-latency bytes 使用
token-expert assignments。两种模式都在命名边界上加入必需 scale bytes，并排除 expert metadata、
padding 和 backend buffer capacity。若没有定义 primitive model 或 transport counters，不发布
algorithm bandwidth、bus bandwidth、wire utilization 或 physical-link utilization。Logical
bandwidth 绝不能标为 physical bandwidth。已发布 payload 和 token rates 命名为
`rate_at_latency_percentile`：bytes 或 tokens 除以对应 latency percentile。它们是 p99 latency
下的 lower-tail service rates，不是 inverted rate distribution 的 p99 percentiles。

## 正确性

与实现无关的 oracle 使用 expert-specific deterministic transform，使错误 expert routing 无法
通过 identity roundtrip。它对每个 rank 和 point 验证：

1. destination rank/expert、source token、multiplicity、gate weight 和 receive counts；
2. timing 前的 dispatched payload 和 metadata；
3. timing 前的 combined output；
4. 所有 timed samples 期间 semantic inputs 不变；
5. timing 后再次验证 dispatched payload/metadata 和 combined output。

Normal-mode adapters 使用 activation-only、unweighted rank-sum combine。Oracle 在 combine 前
构造每个 rank 的 gate-weighted expert aggregate，独立计算 `sum(gate * expert(token))`，并检查
dispatch metadata 和 transformed output。Low-latency adapters 单独验证 expert-packed
source/expert assignment、原生 gate weights 和 gate-weighted combined output。两个契约都使用
已记录的 `rtol=0.05` 和 `atol=0.02` 检查每个 element。任一 rank 或 point 失败都会使 case
不合格。Pre/post dispatch evidence 按
canonical source-token order 计算 hash。Native receive slots 可能非确定性分配，因此 physical
receive order 不作为 correctness property。

## Native 结果

单个 raw case document 使用 `format: "collectivex.ep.v1"`，拒绝未知 fields，并包含：

- `case`：稳定 case ID、suite、required tier 和 coordinate；
- `workload`：canonical identity 和 logical MoE shape；
- `measurement`：sampling、component states、timing 和 byte accounting；
- `implementation`：实例化 class/API、固定 source、loaded libraries 和 resources；
- `topology`：requested 和 realized SKU、devices、placement、scale-up domain 和 transport；
- `provenance`：source SHA、image/squash hashes、allocation、run 和 attempt；
- `rows`：point latency、byte accounting、token rate、correctness、load、fanout 和 anomaly evidence；
- `outcome`：`success`、`failed`、`invalid`、`diagnostic` 或 `unsupported`，以及 reasons。

Raw result documents 和 exact samples 会先经过临时 GitHub delivery artifacts，再由 publisher
归档到 private bundle；它们不会进入 public tree。Private environment details 只保留在本地
mode-0600 logs 和忽略的 operator notes 中；不会归档或发布。每个 expected case 有一个 terminal
selected outcome，同时保留每次 attempt。

## 身份与比较

Canonical JSON 生成三个完整 SHA-256 IDs：

- `series_id`：除 token coordinate 和 repeat allocation 外的所有 locked factors；
- `point_id`：`series_id` 加 token coordinate；
- `evidence_id`：`point_id` 加 allocation/run/attempt/sample checksum。

Locked factors 包括 workload bytes、measurement 和 sampling contract、resources、realized
topology、implementation/build、loaded libraries、image/squash、runtime 和 source SHA。
Deferred code generation 会在 measurement 前捕获，并在之后再次捕获。DeepEP V2 使用固定的
NVCC random seed，并绑定最终 cache keys、generated-source hashes 与 executable-SASS hashes；
raw CUBIN bytes 仅保留为 private diagnostics。Hybrid 绑定实际自动调优配置与完整 kernel-key
set，同时将各 rank 的 shared-object hashes 仅保留为 private diagnostics。本地构建的 extension
hashes 属于 diagnostic；其固定 source trees、build recipe、runtime 与 dependencies 仍绑定到
series。
Series identity 包含 case ID；case ID 绑定完整 scheduled token ladder，以及固定的 percentile、
rank-reduction、conditioning、warmup 和 correctness semantics。

Controlled comparison 只声明一个 contrast：

- `library`：backend implementation 及其 tuned resource profile 可以不同；realized system、
  workload、EP、resource policy、source 和 measurement 必须匹配；
- `chip`：受控 platform contrast。完整 realized system/topology 和 tuned resource profile 可以不同，
  但 workload、EP、placement class、resource policy、backend lineage、source 和 measurement 必须
  匹配。它不是 silicon-only comparison；
- `system`：保留所有 hardware/backend 差异，同时匹配 workload、EP 和 measurement；
- `routing`：routing distribution/EPLB 可以不同，但 static implementation build/generator、system、
  model shape、resource profile 和 measurement 必须匹配。未启用 EPLB 的 Uniform 和 Zipf 复用
  同一 generated implementation；EPLB 的 physical-expert/JIT configuration 是显式 treatment
  difference。

任何未声明的 mismatch 都会拒绝 overlay。Chip/system results 描述 measured systems，而非仅描述
silicon。

## Evidence 策略

Capability declarations 说明可以尝试什么；artifacts 决定 evidence status。Promotion 要求完整的
expected coverage，不能有 missing、extra、duplicate、malformed 或 heterogeneous case。Public
coverage 保留每个 matrix disposition；promotion 要求每个 runnable case 在所有 selected runs 中
成功，且每个 planned-unsupported case 始终为 unsupported。只有固定 canonical full-v1 matrix，
且具有 decision-grade library、chip、system 和 routing cohort，才能推进 `dev-latest`；partial
matrices 仍为 diagnostic。Full-matrix digest 有意绑定精确 workflow shard grouping 和 requested
cases，因此即使 case coverage 不变，修改 `--max-cases` 或 SKU round-robin scheduling order 也只
会产生 diagnostic-only runs。Superseded retries、planned-unsupported outcomes 和 unstable
comparison cohorts 可以用于诊断展示，但不能排名或推荐；promoted dataset 中每个成功的 required
series 都必须保持 decision-grade。Runnable case 的任何 failed、invalid 或 diagnostic retry 都会
阻止 promotion，即使后续 retry 成功。Routing cohorts 是 comparable-experimental sensitivities，
不会产生 configuration recommendations；official library/platform/system cohorts 才能产生可执行
recommendations。

一个 point 只有在三个独立 workflow runs 和 allocation IDs 均通过 correctness、identity、
provenance、tail gates、p50/p99 repeat-stability thresholds 和 stable ordering 后才成为
decision-grade。Eligibility、controlled cohorts、sensitivity pairs 和 recommendations 由
publisher 而非 frontend 计算。

## 执行隔离

每个非 MNNVL scale-out case 都使用 operator 固定的 socket 与 RDMA selectors。Launcher 会拒绝
缺失或不完整的 profile，并在 backend 初始化前逐个 allocation 节点检查已配置 interface、active
HCA port 与指定 GID。它不会改用 default route、继承的 runner environment 或 transport
fallback。Scale-up 和 MNNVL case 会清除该 profile；scale-out NCCL/RCCL 强制设置
`NCCL_NET=IB` 并精确匹配 HCA。Selector values 只保留在加密配置和 mode-0600 private logs 中。

Repository staging 使用 checkout 与 workflow workspace 外预创建的 shared base；该 base 由
runner owner 持有，group/world 均不可写。父进程在复制前解析精确 execution child，以
runner-owned marker 声明所有权，并验证所有 allocation 节点读写的是同一份 bytes。Cleanup 会
等待 allocation teardown 得到确认，并只删除该 child，包括可安全识别的未完成 claim。同一 run
的 V2/Hybrid source archive 会在固定 member 数和解压大小上限内完整验证，并且只提取所选 fixed
root；仅当相对 leaf symlink 指向同一 backend root 内的 regular member 时才允许创建，之后还要
通过精确 Git tree/submodule 校验。

## 产物验证与即时交付

不使用 self-hosted service、Vercel storage、GCP、Neon、managed database 或 managed object
store。Publication workflow 仅将 runner 本地临时存储用作可丢弃的 validation 与 promotion
工作区：

```text
$COLLECTIVEX_STORE_ROOT/
  private/incoming/          # write-once downloaded GHA attempts
  private/bundles/<sha256>/  # immutable source archives, native results/samples, matrix, checksums
  private/quarantine/        # rejected attempts plus machine-readable reasons
  public/datasets/<sha256>/  # immutable sanitized frontend datasets
  public/channels/           # small atomic pointers: latest-attempt, dev-latest
  locks/
```

Private 和 public trees 使用不同 permissions。JSON manifests 和 checksums 是权威记录；可重建
catalog 仅为 index。Raw sweep artifacts 只是 publisher 的临时输入；只有清理并完成 promotion
的 NDJSON 会保留为前端 publication artifact。

Container tags 会与固定 registry digests 核对。Enroot imports 使用固定
`SOURCE_DATE_EPOCH` 和 versioned cache generation；每个 mounted squash 都重新计算 hash 并纳入
series identity。Image-provided DeepEP 也按精确 per-architecture wheel 和 installed-file
fingerprints 检查，因此 stale cache 不能继承固定 source identity。
Source-built DeepEP V2 使用独立的 mode-0700 cluster-local cache，并且只以 `/cx-cache` 挂载。
其 content key 绑定版本化 build recipe、verified image digest、CPU/GPU architecture、
upstream source trees 和固定 build dependencies。该 cache 既不是 artifact，也不是 publisher
input；每次执行的 source/results stage 仍然隔离且可丢弃，并在复用前以 marker 和 runtime probe
fail closed。Runner UID 属于受信任的 cluster boundary：该 cache 用于防止 stale 或意外修改，
不防御恶意的同 UID job。只有从未发布的 partial build 才能自动重置；已发布 cache 一旦未通过
integrity 或 runtime 检查，将保持原样并被拒绝，避免并发 allocation 正在使用的文件被删除。

Publication 采用 fail-closed：

1. 获取 exclusive filesystem lock，并在 destination filesystem 上 stage；
2. 解析前归档 source bytes；
3. 要求精确 matrix-declared artifact set，并拒绝每个未消费 archive member；
4. 验证 strict schemas、privacy、checksums、identities、timing 和精确 matrix outcomes；
5. 写入 checksums 和 `COMPLETE`，fsync，然后原子 rename private bundle；
6. 构建并验证 sanitized content-addressed dataset，fsync，然后原子 rename；
7. 仅在全部 promotion gates 通过后原子替换 `dev-latest.json`。

Rejected attempts 可以更新工作区中的 `latest-attempt`，但不能更新 `dev-latest`。工作区会随
publication runner 销毁，且绝不连接到前端。只有三个选定 bundles 全部推进 `dev-latest` 后才会
生成 artifact。

`publisher.py ingest` 接受精确 matrix，并为每个 GitHub artifact 接受一个 `--artifact` directory
或 ZIP。`promote` 接受显式 immutable bundle IDs。默认 `verify` 要求 `latest-attempt`；若存在
`dev-latest` 也会验证，而显式 `--channel dev-latest` 则要求其存在。Workflow 只会将通过验证并
清理后的 dataset 复制到单记录 `collectivex_public_v1_<sha256>.ndjson` artifact。Raw artifacts 和
private workspace 内容绝不打包进应用。

Sweeps 默认使用 `release_tag=unversioned`。选择 `v1` 时必须匹配固定的完整 matrix digest，并
生成绑定 run ID、attempt、source SHA 与 matrix SHA-256 的 marker。手动 publication workflow
只接受三个唯一、成功、来自同一 source SHA 的 `CollectiveX Sweep` run IDs；它会重新校验
metadata 与精确 markers，下载 immutable artifacts，并将相同 provenance assertions 传给
`publisher.py ingest`。Partial、filtered、untagged、跨 source、失败或过期的输入都会 fail closed。

前端使用 server-side GitHub read token，即时发现最新成功且按版本隔离的 publication run，并
下载 publication artifact。它要求 ZIP 根目录只有一个 NDJSON entry，校验 UTF-8、schema、
promotion 状态及 filename/body SHA-256，随后提供短期缓存的带版本 channel pointer 和 immutable
带版本 dataset URL。Benchmark-version selector 当前只显示 V1；后续版本必须使用独立的 release
与 publication identity。前端不会虚构 missing values、选择 retries，或重新计算 decision
eligibility。

## 历史数据

Numeric schemas 3-5 不在 v1 publisher 和 frontend reader 范围内。它们仍是 historical
diagnostic evidence，不能作为 `dev-latest` 初始数据或驱动 v1 decisions。
