# Kimi-K3 MI355X vLLM 镜像复现

<div align="center">

[English](./kimi-k3-mi355x-vllm-image-reproduction.md) | **中文**

</div>

最后更新：2026-08-31

本文档用于重建 C16 CPRR 性能工作所使用的 Kimi-K3 vLLM 运行时。它将
精确验证过的产物与上游 PR head 分开记录，使干净镜像可以在不依赖已修改
容器或 AITER JIT 缓存的情况下重建。

## 基础镜像

```text
vllm/vllm-openai-rocm:nightly-46638857fdbb30e0c232c9e8f9cb1ff6d6f545c3
sha256:8908b8ab5ba28c3b81f9f42bb72e2421f06a180e001c67c4f10ff7f127c5690b
```

该镜像包含 vLLM commit `46638857fd` 和 `amd_aiter 0.1.19`。可复现的
基础是 digest，而不是可变的 nightly tag。

## 分层顺序

必须按以下顺序安装各层：

1. 应用 C16/C52 vLLM overlay：

   ```text
   benchmarks/single_node/agentic/k3_patches/
     vllm_nightly_46638857_k3_c16_c52_current.patch
   SHA256 90f975fad15722494366153ec3f32a14c4445bfa88c51ec53043b88eaf64dcc0
   ```

2. 应用包含五个文件的 vLLM #52190 delta：

   ```text
   benchmarks/single_node/agentic/k3_patches/
     vllm_nightly_46638857_k3_compile_52190_delta.patch
   SHA256 de1ac272820122281f865c4f81d3f7a87e03c0cb42feb59390d9012b9bb88c00
   tested PR head c70113053761985aa289d5088503731c535dc028
   ```

3. 安装完整 CPRR runtime bundle：

   ```text
   benchmarks/single_node/agentic/k3_patches/
     aiter_pr4521_plus_4964_runtime/
   SHA256(SHA256SUMS) cb6f7ab6210d876e674f276cbaacf638936358cc12c1f89622084a611bb1d342
   files covered by manifest: 62
   ```

实现该顺序的 Dockerfile 为：

```text
benchmarks/single_node/agentic/k3_patches/
  Dockerfile.kimi-k3-c16-compile52190-cprr
```

这是选定的 A8W4/CPRR 镜像。vLLM PR #53940 与对应的本地 AITER selector
补丁会把 SiTUv2 MoE 路由到 A4W4，因此单独进行了评估。它们不属于该
Dockerfile：精确的 C16 screen 比选定 A8W4 路径更慢，P90 TPOT 也更差。
详情见下文“未采用的 A4W4 实验”。

## PR 与自定义修复来源

C16/C52 overlay 是一个精确快照，并不表示当前列出的所有 PR head 可以
直接一起 cherry-pick。其上游来源组件如下：

| 工作项 | 已验证的本地 commit | 用途 |
|---|---|---|
| vLLM #51705 | `96e0305704`, `af987cdf72`, `4763986cf3`, `97bb6a9c40` | Kimi-K3 DSpark 与 DCP attention/verification 支持。 |
| vLLM #53598 | `d770e4e2f4` | 分组 hybrid-cache 几何与 DCP prefix-cache lookup。 |
| vLLM #53917 | `0a489b4b8b`, `394bb2fd34`, `4b478c8df5` | Hybrid 几何一致性、Mamba replay 边界与失败 load 恢复。 |
| vLLM #52707 | `f872fdd003` | 防止外部 block allocation 出现负数。 |
| vLLM #52494 | `9e08dccddd` | 在 AMD Kimi-K3 wrapper 中融合 MLA Q/KV RMSNorm。 |
| vLLM #52968 | `208916fb29` | Kimi-K3 attention-residual、sigmoid-multiply 与 QKV-convolution 融合。 |
| vLLM #53166 | `c92234cbce`, `43bd3ac18a`, `7662093dfc` | AITER MLA chunked-context gather 与由 metadata 构造的 KV indices。 |
| vLLM #54165 | `09438c4eb5`, `1f11b7a933` | 在 DFlash/DSpark 与 KV connector 下保留 hybrid-Mamba cache hit。 |
| vLLM #52033 | 保留的本地快照 | ROCm dual-stream shared-expert 机制。 |
| vLLM #51437 | 保留的本地快照 | 将 shared all-reduce 与 routed up-projection 重叠。 |
| vLLM #52190 | `c70113053761985aa289d5088503731c535dc028` | 启用 Kimi-K3 compilation，并声明 KDA 与 attention-residual 的 mutation boundary。 |

Overlay 还包含以下自定义修复：

- `c7d8e7b8de`：在捕获的 ROCm graph 生命周期内保留打包后的 A2A buffer。
- `6cd48179ca`：限制 SimpleCPU lazy-eviction 扫描范围。
- 在 reset 后继续持有进行中的 SimpleCPU store ID、hash、GPU pin 与 CPU
  reference，直到异步 DMA 完成。被放弃的 completion 只释放所有权，不会
  将过期数据发布到 CPU prefix cache。
- 导出并复用 KDA/Mamba 内部 checkpoint 与符合条件的 partial tail。
- 将 SimpleCPU partial-tail transfer specification 集成到 eager/lazy store。
- 将源自 #51437 的 fused all-reduce/RMSNorm 路径限制在不超过 80 token
  的 batch。
- 增加 shared-expert stream 强制开关与 quantized-input 安全条件。选定运行
  保持 `VLLM_ROCM_FORCE_SHARED_EXPERTS_STREAM=0`。
- 在进入 ROCm `wvSplitKQ` 前，对表面 row stride 更宽的单行 activation
  执行 clone。

更广泛的 SimpleCPU/partial-tail 变更必须在任何上游提交前拆分并审阅。
Transfer-lifetime 行为以及该修复与更广泛 cache 变更之间的边界，汇总在
[C16/C52 附加内容](./kimi-k3-mi355x-vllm-current-baseline_zh.md#c16c52-附加内容)。

## 兼容的 AITER 重建

AITER PR #4964 不能直接基于 nightly 镜像内保留的源码快照重建。该源码
早于 AITER PR #4521 已安装的预编译 dispatcher。直接重建会丢失 #4521
的 Python/C++ ABI 与 dispatch 变更，而 vLLM 仍调用新接口，最终会复现
GPU memory-access fault。

兼容源码的重建方式为：

```text
AITER PR #4521 head:
  0cbedbb1bc5b3b254dd12ca4e8d3c7638b86830b

Replay these four AITER PR #4964 commits in order:
  b989492aa7b9baf7fccd46cd137a3d25dec264ef
  1fd8439223328072b189f988e8358be6cacd7893
  d579b1afd70d3fd65580181a0222b541ac3d1075
  5432e1a06ecf67782f553106bf5eca66dd01b789

Resulting compatibility commit:
  3281ad690206e4ab9b08eb3a5eddeaaf57b13f19
```

重建 worktree 为：

```text
/home/hyukjlee/working_projects/aiter-pr4521-plus4964
```

Runtime bundle 包含匹配的 Python dispatcher、C++ interface 与 metadata
源码、完整 gfx950 MLA registry/kernel 目录以及两个预编译 module。关键
hash 如下：

```text
vLLM rocm_aiter_mla.py
20ed4a04e4fb730dd9031eb3be8462e9193fda0c0a8a4daeb202d587c6ac310d

module_mla_asm.so
2cb23f820559cfd60d8bb27ea82fb9fa31cce9115a098097725825e8c20da505

module_mla_metadata.so
2806c1c2d7ddf9bebe72b0a1a1a42fe8035c3a7c299cb653f4d80cc21ad6c556
```

## 构建

在 InferenceX 仓库根目录运行：

```bash
docker build \
  -f benchmarks/single_node/agentic/k3_patches/Dockerfile.kimi-k3-c16-compile52190-cprr \
  -t kimi-k3-vllm:c16-compile52190-cprr-20260831 .
```

构建会在修改 site-packages 前校验两个 vLLM patch hash、runtime manifest
hash，以及 manifest 中列出的每个文件。它使用预编译 gfx950 module，因此
Docker 构建本身不需要 GPU。

该 Dockerfile 已于 2026-08-31 在 `mi355x-17` 上完成构建验证。生成的
本地镜像为：

```text
kimi-k3-vllm:c16-clean-cprr-20260831
sha256:247e2b452c4745cbe119148caae4979f09ed131a2747551748912f19afe2668f
```

构建校验了 CPRR manifest 中全部 62 个文件。已安装的 vLLM compile 文件、
vLLM MLA wrapper、AITER Python 文件以及两个预编译 MLA module 的 hash，
均与此前验证的 runtime 镜像 `kimi-k3-vllm:c16-cprr4521-4964-20260831`
逐字节一致。

## 逐字节校验

构建后应校验已安装文件，而不能只依赖镜像 label：

```bash
docker run --rm --entrypoint bash \
  kimi-k3-vllm:c16-compile52190-cprr-20260831 -lc '
    cd /usr/local/lib/python3.12/dist-packages
    sha256sum \
      vllm/v1/attention/backends/mla/rocm_aiter_mla.py \
      aiter/mla.py \
      aiter/ops/attention.py \
      aiter_meta/csrc/py_itfs_cu/asm_mla.cu \
      aiter_meta/csrc/kernels/mla/metadata/v1_2_device.cuh \
      aiter_meta/hsa/gfx950/mla/mla_asm.csv \
      aiter/jit/module_mla_asm.so \
      aiter/jit/module_mla_metadata.so
  '
```

预期 hash 记录在 bundle 的 `SHA256SUMS` 中。Dockerfile 还记录了基础
digest、vLLM #52190、AITER #4521、AITER #4964、兼容重建 commit 与
manifest hash 的 label。

运行该镜像时，不要把旧的 `k3_aiter_jit_46638857` volume 挂载到
`/usr/local/lib/python3.12/dist-packages/aiter/jit`。该挂载会遮蔽镜像内
构建的两个 CPRR module。一次失败的干净镜像准确率启动已经确认了这一点：
运行时找不到 GQA96、QLen4、LSE+CPRR 的 heuristic kernel，随后 worker
退出。移除该 volume 后，运行时会使用通过 SHA 校验的 module，并保持
镜像与运行时等价。可以在其他路径挂载新的 cache，但不得遮蔽 bundle 的
module 目录。

## GitHub Actions 运行时等价性

Actions 配置仍引用固定的上游 nightly。对于 C16 MTP arm，
`benchmarks/single_node/agentic/kimik3_fp4_mi355x_vllm_mtp.sh` 会选择两个
vLLM patch 与精确的 CPRR runtime bundle。公共 launcher 会校验 bundle，
并在导入 serving engine 前安装同一组文件。因此，对基础镜像之外的文件，
Actions 运行时与干净 Dockerfile 逐字节等价。

## 验证证据

- 聚焦数值测试 `test_dcp_fp8_round_robin_verify_matches_causal_attention`
  通过。
- 测试加载了 `mla_a8w8_qh32_qseqlen4_gqaratio32_lse_cprr_ps`，这是
  Kimi-K3 DCP8 verification 所需的 kernel（`decode_heads=96`，
  `logical_qlen=4`）。
- PIECEWISE 与 FULL graph capture 在 MNS/CG16 下完成，并在无 GPU fault
  的情况下服务 32/32 个请求。
- GMU 0.84 下的 MNS/CG80 已完成 capture，但没有 KV-cache budget。
- GMU 0.84 下的 MNS/CG64 已完成 capture，但报告可用 KV-cache memory
  为 `-2.18 GiB`。这是明确的容量失败，不是 GPU fault。

### C16 CPRR serving screen

除特别说明外，下表全部使用 TP8/DCP8 A2A、DSpark K=3、synthetic
acceptance length 3.0、MBT8192、无 KV offload，并关闭强制 shared-expert
stream。吞吐量为总 token throughput 除以八。

| MNS / graph envelope | GMU | Throughput/GPU | P90 TPOT | 结果 |
|---|---:|---:|---:|---|
| 64 / dense through 64 | 0.86 | 6,960.97 tok/s | 15.44 ms | 96/96 |
| 16 / sparse through 64 | 0.84 | 11,829.58 tok/s | 34.25 ms | 96/96 |
| 8 / `1,2,4,8,16,24,32` | 0.84 | 9,610.86 tok/s | 17.20 ms | 96/96 |
| 6 / `1,2,4,6,8,12,16,20,24` | 0.84 | **8,141.60 tok/s** | **16.02 ms** | 96/96 |
| 8 / `1,2,4,8,16,24,32`, forced stream | 0.84 | 9,155.52 tok/s | 17.94 ms | 96/96 |

选定的 MNS6 结果保存在：

```text
/home/hyukjlee/k3-c16-cprr4964-smoke-20260831/results/gmu0.84_20260830T232533Z
```

它在目标 7,381.5 tok/s/GPU 之上保留约 10% 吞吐余量，同时相对 MNS8
将 P90 TPOT 降低 1.18 ms。更偏向吞吐的 MNS8 结果仍可作为备选，保存在：

```text
/home/hyukjlee/k3-c16-cprr4964-smoke-20260831/results/gmu0.84_20260830T223841Z
```

MNS8 运行报告 13.05 GiB KV-cache memory、4,289,629-token KV capacity、
`VmPin=0`，且运行期间没有 GPU fault。Server log 末尾的 traceback 来自
benchmark 结束后 launcher 主动执行的 `docker stop`。

### 未采用的 A4W4 实验

vLLM PR #53940 的验证 head 为
`47cd3318351d9a62a900529fbe88a1f64b293532`，以四个 runtime source file
的 delta 应用，SHA256 为
`29bbe3bef0e99799c7394870f75e83aedba90d86494c2947de54b140ad1cfba7`。
Nightly AITER dispatcher 不包含对应的 `AITER_SITUV2_A4W4` selector，
因此第一次运行实际静默选择了 `flydsl_moe1_abf16_wfp4_*`，不能作为
A4W4 对比。

随后对 `aiter/fused_moe.py` 应用了最小 selector patch，SHA256 为
`d23e36ff40f28632a4ae2f27b6896e15802a13ee4eed4c1cb62e84a532af13a7`。
Import-time 校验显示 `AITER_SITUV2_A4W4=1`，并移除了旧的
`AITER_SITUV2_A8W4`。Capture log 确认使用
`flydsl_moe1_afp4_wfp4_bf16_*`，并使用与选定路径相同的 CPRR MLA kernel。

有效的 A4W4 screen 完成 96/96 个请求，达到 9,396.09 tok/s/GPU 与
18.10 ms P90 TPOT。由于两个指标都劣于选定的 MNS8 A8W4/CPRR 结果，
#53940 与 AITER selector 被记录为未采用实验，不会安装到干净镜像中。

### 干净镜像准确率

选定的 MNS6/CG24 配置在干净镜像上使用 block rejection 与
`lm-eval --limit 128` 完成 GSM8K five-shot，并通过准确率门禁：

- strict match：128/128
- flexible extraction：128/128
- 实测 mean acceptance length：3.44-3.49
- 实测 drafted-token acceptance：81.2%-83.1%
- 无 worker death、OOM、GPU fault 或 missing-kernel error
- teardown 后八张 GPU 均恢复为 0% use 与 0% allocated VRAM

`mi355x-17` 上的产物：

```text
/home/hyukjlee/k3-c16-clean-cprr-accuracy-20260831/results/gsm8k_limit128_c16_clean_cprr_mns6_cg24_block_20260830T234636Z
```

剩余 release gate 为：

1. 执行一次 3,600 秒 GitHub Actions C1/C16/C52 运行。
2. 审计导出结果、server log、memory growth 与 GPU release。
