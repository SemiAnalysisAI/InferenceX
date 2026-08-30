# Kimi-K3 MI355X vLLM 当前基线

<div align="center">

[English](./kimi-k3-mi355x-vllm-current-baseline.md) | **中文**

</div>

本文记录如何复现 2026-08-30 确定的 C1、C16 和 C52 配置。GitHub Actions
基线为提交 `654687d7e9e5f9adb4fc50c71694b2752bf669c0` 上的运行
[33324464095](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/33324464095)。

## 基础镜像

三个测试点均从同一不可变镜像开始：

```text
vllm/vllm-openai-rocm:nightly-46638857fdbb30e0c232c9e8f9cb1ff6d6f545c3
sha256:8908b8ab5ba28c3b81f9f42bb72e2421f06a180e001c67c4f10ff7f127c5690b
```

该镜像包含 `amd_aiter 0.1.19`。没有应用 AITER 源码或 site-packages
覆盖层。基准测试会复用持久化的 `k3_aiter_jit_46638857` JIT 缓存，但该
缓存不属于镜像内容。

## 两个已验证的源码变体

已接受的结果没有使用统一的 vLLM 补丁树。C1 使用 online-FP8/KDA 树，
C16 和 C52 共用 overlap/offload 树。在合并后的源码树通过三个测试点的
性能和准确率门禁前，应继续使用独立镜像。

| 测试点 | 补丁 | SHA256 | 源文件数 |
|---|---|---|---:|
| C1 | `vllm_nightly_46638857_k3_c1_current.patch` | `554ec6384b4ae143df42b223af66a8365e2b466c7ea691ed6c5a26a8749a4e6d` | 44 |
| C16、C52 | `vllm_nightly_46638857_k3_c16_c52_current.patch` | `90f975fad15722494366153ec3f32a14c4445bfa88c51ec53043b88eaf64dcc0` | 34 |

两个补丁都是相对于 vLLM 提交
`46638857fdbb30e0c232c9e8f9cb1ff6d6f545c3` 的完整差异。补丁先在固定
镜像中的已安装包上执行 dry-run，再与保留的已接受覆盖层逐字节比较。
C16/C52 补丁有意排除了两个 `.pyc` 文件以及
`manager.py.before-partial-tail-20260830`。

## 构建镜像

从 InferenceX 仓库根目录运行：

```bash
docker build \
  -f benchmarks/single_node/agentic/k3_patches/Dockerfile.kimi-k3-current \
  --build-arg K3_PATCH=benchmarks/single_node/agentic/k3_patches/vllm_nightly_46638857_k3_c1_current.patch \
  --build-arg K3_PATCH_SHA256=554ec6384b4ae143df42b223af66a8365e2b466c7ea691ed6c5a26a8749a4e6d \
  -t kimi-k3-vllm:c1-20260830 .

docker build \
  -f benchmarks/single_node/agentic/k3_patches/Dockerfile.kimi-k3-current \
  --build-arg K3_PATCH=benchmarks/single_node/agentic/k3_patches/vllm_nightly_46638857_k3_c16_c52_current.patch \
  --build-arg K3_PATCH_SHA256=90f975fad15722494366153ec3f32a14c4445bfa88c51ec53043b88eaf64dcc0 \
  -t kimi-k3-vllm:c16-c52-20260830 .
```

发布镜像时，应推送不可变标签，并在包装脚本中用生成的镜像摘要替代运行时
补丁。在合并源码树通过 C1、C16 和 C52 验证前，不要发布统一标签。

Dockerfile 已在 `mi355x-17` 上完成构建测试。本地验证镜像 ID 为：

```text
kimi-k3-vllm:c1-20260830       sha256:b31f5ab0435103f9279d765274654ef130d229982ab2d967a8a8b5757cd78cd7
kimi-k3-vllm:c16-c52-20260830  sha256:aacd27681dbb8717ef544799952ec08ba9c185038f9b931f1aafb56f593c104d
```

这些 ID 是本地构建结果，不是已发布的 Registry 摘要。

## 来自 PR 的源码栈

以下本地快照同时存在于两个变体中：

| 上游工作 | 已验证的本地提交 | 用途 |
|---|---|---|
| vLLM #51705 | `96e0305704`、`af987cdf72`、`4763986cf3`、`97bb6a9c40` | Kimi-K3 DSpark/DCP attention 与验证支持 |
| vLLM #53598 | `d770e4e2f4` | 按组 hybrid-cache 几何结构与 DCP prefix 查找 |
| vLLM #53917 | `0a489b4b8b`、`394bb2fd34`、`4b478c8df5` | Hybrid 几何结构、Mamba replay 边界及失败加载恢复 |
| vLLM #52707 | `f872fdd003` | 防止外部 block 分配数变为负值 |
| vLLM #52494 | `9e08dccddd` | AMD Kimi-K3 包装器中的 MLA q/KV RMSNorm 融合 |
| vLLM #52968 | `208916fb29` | Attention-residual、sigmoid-multiply、QKV-convolution 等融合 |
| vLLM #53166 | `c92234cbce`、`43bd3ac18a`、`7662093dfc` | AITER MLA chunked-context gather 和基于 metadata 的 KV 索引 |
| vLLM #54165 / #54163 | `09438c4eb5`、`1f11b7a933` | 使用 KV connector 时保留 hybrid-Mamba cache hit |

这些哈希标识的是已验证的本地快照。开放 PR 的 head 可能已经 rebase 或
force-push，不能假设其内容逐字节相同。

共同的自定义提交：

- `c7d8e7b8de`：在捕获的 ROCm graph 生命周期内保留打包后的 A2A buffer。
- `6cd48179ca`：限制 SimpleCPU lazy-eviction 扫描，避免对 in-flight store
  反复执行完整 prefix-cache 扫描。

## C1 附加内容

- vLLM #51392，本地提交序列 `0cbbe1491b` 至 `906037cf83`：将 online
  quantization 与预量化 checkpoint 组合。
- vLLM #54248，本地提交 `fe0647fd34`：为 AITER PTPC linear 声明
  per-token FP8 输入。
- vLLM #54254，本地提交 `b1ae0ffc2c`：将 KDA gated RMSNorm 与
  per-token-FP8 `o_proj` 路径融合。
- `quantization/online/fp8.py` 的 import 顺序清理。
- AMD `kda.py` 直接返回 PTPC `o_proj` 结果。

生成 C1 补丁的源码树为
`/home/hyukjlee/working_projects/vllm-k3-online-fp8-clean`，HEAD 为
`b1ae0ffc2c`，并包含上述两个已记录的 working-tree 变更。

## C16/C52 附加内容

- 来自 vLLM #52033 的 ROCm 双 stream shared-expert 机制，以及本地 force
  和量化输入安全控制。已接受的运行显式设置
  `VLLM_ROCM_FORCE_SHARED_EXPERTS_STREAM=0`。
- 来自 vLLM #51437 的 shared all-reduce 与 routed up-projection 重叠。
  本地 `fused_output.shape[0] <= 80` guard 限制融合的 AITER
  all-reduce/RMSNorm 路径。
- AMD KDA/Mamba checkpoint 导出、细粒度 replay 边界、partial checkpoint
  re-key 和 partial-tail 外部缓存交接。
- SimpleCPU request block-table 与 partial-tail 集成。
- SimpleCPU 异步传输生命周期处理：在 DMA 完成前跨 reset 保持 GPU 与 CPU
  引用 pinned，跟踪 outstanding block hash，并按逻辑 hash 去重 eager/lazy
  store。
- ROCm scaled-mm guard：当单行 activation 的表面连续 view 具有更宽 row
  stride 时，在进入 `wvSplitKQ` 前执行 clone。

精确保留的源码覆盖层是
`/home/hyukjlee/k3-offload-overlap-51437-compose-20260830/vllm`。完整
37 文件 manifest 的聚合 SHA256 为
`9f744c8326fe2759d96b460722094cc26eb1136ac484c293f6dfc6b41d7b4130`；
干净补丁仅包含其中 34 个源文件。

本基线中的 SimpleCPU 补丁是已验证的组合，而不是可直接提交上游的变更。
在提出 vLLM PR 前，必须将传输生命周期修复与 partial-tail、speculative
cache 语义分离。

## 服务配置范围

| 测试点 | Speculation | DCP | Offload | MNS | MBT | GMU | Graphs | Async |
|---|---|---:|---|---:|---:|---:|---|---|
| C1 | DSpark K=6，synthetic AL 3.84 | 1 | none | 2 | 8192 | 0.875 | 1-16 | off |
| C16 | none | 8/A2A | none | 80 | 8192 | 0.86 | 1-80 | off |
| C52 | none | 8/A2A | 512 GB `vllm-simple` | 80 | 16384 | 0.90 | 1-80、128-4096 的 2 次幂 | on |

所有测试点都使用 ROCm AITER MLA、prefill-query quantization、FP8 KV
cache、prefix match unit 128、SHA256 prefix hash，以及包装脚本中的相同
custom-op 列表。

## 基线派发

精确的 Workflow 派发命令为：

```bash
gh workflow run .github/workflows/e2e-tests.yml \
  --repo SemiAnalysisAI/InferenceX \
  --ref amd/kimi-k3-current-baseline-20260831 \
  -f ref=amd/kimi-k3-current-baseline-20260831 \
  -f generate-cli-command='test-config --config-files ./configs/amd-master.yaml --config-keys kimik3-fp4-mi355x-vllm-agentic-current-baseline' \
  -f test-name='Kimi K3 current findings baseline C1 C16 C52' \
  -f duration-override=3600 \
  -f fail-fast=false
```

派发前，生成的矩阵必须正好包含 C1、C16 和 C52。

## C16 `torch.compile` 候选

基线之后第一个显著的 C16 提升来自 vLLM
[#52190](https://github.com/vllm-project/vllm/pull/52190)，已验证 head 为
`c70113053761985aa289d5088503731c535dc028`。基线日志显示已启用
`torch.compile`，但 Kimi-K3 不支持，因此已声明的 `mla_dual_rms_norm` 和
`allreduce_rms` fusion pass 从未获得 graph。

post-overlay 补丁为：

```text
vllm_nightly_46638857_k3_compile_52190_delta.patch
SHA256 de1ac272820122281f865c4f81d3f7a87e03c0cb42feb59390d9012b9bb88c00
```

该补丁在 `vllm_nightly_46638857_k3_c16_c52_current.patch` 之后应用，修改
五个文件：

- `vllm/config/compilation.py`
- `vllm/models/kimi_k3/amd/kda.py`
- `vllm/models/kimi_k3/amd/latent_moe_runner.py`
- `vllm/models/kimi_k3/amd/linear.py`
- `vllm/models/kimi_k3/amd/ops/attn_res.py`

该 delta 保留已接受的下游 KDA checkpoint、partial-tail 和 latent-MoE
实现。它启用 Kimi-K3 compile 支持，将原地修改的 KDA core 与
attention-residual launcher 注册为编译器可见的 custom operation，将 KDA
core 加入默认 graph splitting operation，并把一次性 latent-tail 日志移出
compiled forward 路径。

构建干净的 C16 镜像：

```bash
docker build \
  -f benchmarks/single_node/agentic/k3_patches/Dockerfile.kimi-k3-c16-compile52190 \
  -t kimi-k3-vllm:c16-compile52190-20260831 .
```

该 Dockerfile 已在 `mi355x-17` 上完成构建测试。本地镜像 ID 为：

```text
sha256:e2fb0e238e7612f06cc47d656009584f8cc902b90bb1119c2b66e9330d3b3d1b
```

构建后，base 与 delta 补丁涉及的 36 个唯一源码路径全部与已验证运行时
覆盖层逐字节一致。镜像 label 记录基础镜像摘要和 #52190 head。

两个相同的 C16 synthetic screen 首先在无 MTP 条件下隔离 compile 变更。
其余设置为无 offload、DCP8/A2A、MNS80、MBT8192、GMU0.86，以及 graph
capture size 1 至 80：

| 测试点 | Aggregate tok/s | Tok/s/GPU | Mean TPOT | P99 TPOT | Requests |
|---|---:|---:|---:|---:|---:|
| 精确 control | 40,005.53 | 5,000.69 | 34.493 ms | 37.043 ms | 96/96 |
| Compile run 1 | 46,106.25 | 5,763.28 | 31.357 ms | 32.559 ms | 96/96 |
| Compile run 2 | 46,039.66 | 5,754.96 | 31.410 ms | 32.370 ms | 96/96 |

两次运行的平均值为 46,072.96 tok/s aggregate，即 5,759.12 tok/s/GPU，
比精确 control 高 15.17%。Mean TPOT 改善 9.01%，平均 P99 TPOT 改善
12.36%。

`mi355x-17` 上的产物：

```text
/home/hyukjlee/k3-c16-compile-52190-screen-20260831/results/c16_compile52190_nospec_nooffload_noreplay_20260830T172845Z
/home/hyukjlee/k3-c16-compile-52190-repeat2-20260831/results/c16_compile52190_repeat2_nospec_nooffload_noreplay_20260830T173800Z
```

在该编译配置上增加 DSpark K=3、probabilistic drafting 和 synthetic
acceptance length 3.00 后得到：

| Aggregate tok/s | Tok/s/GPU | Mean TPOT | P99 TPOT | Requests |
|---:|---:|---:|---:|---:|
| 63,226.07 | 7,903.26 | 22.71 ms | 23.08 ms | 96/96 |

`mi355x-17` 上的产物：

```text
/home/hyukjlee/k3-c16-compile-52190-mtp3-20260831/results/c16_compile52190_mtp3_k3_synthetic_nooffload_noreplay_20260830T174817Z
```

最终候选仅在性能测试中使用 synthetic rejection。准确率使用相同服务配置
和 DSpark K=3，仅将 `rejection_sample_method` 改为 `block`，并运行 GSM8K
与 `lm-eval --limit 128`。strict match 与 flexible extraction 均为
128/128。真实 block rejection 的平均 acceptance length 约为 3.46，drafted
token acceptance 约为 82%。

`mi355x-17` 上的准确率产物：

```text
/home/hyukjlee/k3-c16-compile-52190-mtp3-accuracy-20260831/results/gsm8k_limit128_c16_compile52190_mtp3_block_20260830T180014Z
```

在将该 delta 视为发布配置前，仍必须完成 3,600 秒 AIPerf 门禁。

## 最终候选派发

最终矩阵保留已接受的 C1 和 C52 配置，并将 C16 改为上述 compiled
DSpark K=3 配置。派发命令为：

```bash
gh workflow run .github/workflows/e2e-tests.yml \
  --repo SemiAnalysisAI/InferenceX \
  --ref main \
  -f ref=amd/kimi-k3-compile52190-final-20260831 \
  -f generate-cli-command='test-config --config-files ./configs/amd-master.yaml --config-keys kimik3-fp4-mi355x-vllm-agentic-compile52190-final' \
  -f test-name='Kimi K3 compile52190 final C1 C16 C52' \
  -f duration-override=3600 \
  -f fail-fast=false
```

派发前确认生成矩阵正好包含 C1、C16 和 C52，且 C16 标记为
`spec-decoding: mtp`。
