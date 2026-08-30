# Kimi-K3 MI355X vLLM current baseline

<div align="center">

**English** | [中文](./kimi-k3-mi355x-vllm-current-baseline_zh.md)

</div>

This document reproduces the C1, C16, and C52 configurations established on
2026-08-30. The GitHub Actions baseline is run
[33324464095](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/33324464095)
at commit `654687d7e9e5f9adb4fc50c71694b2752bf669c0`.

## Base image

All three arms start from the same immutable image:

```text
vllm/vllm-openai-rocm:nightly-46638857fdbb30e0c232c9e8f9cb1ff6d6f545c3
sha256:8908b8ab5ba28c3b81f9f42bb72e2421f06a180e001c67c4f10ff7f127c5690b
```

The image contains `amd_aiter 0.1.19`. No AITER source or site-packages overlay
is applied. The benchmark reuses the persistent `k3_aiter_jit_46638857` JIT
cache, which is not part of the image contents.

## Two tested source variants

The accepted results did not use one universal patched vLLM tree. C1 used the
online-FP8/KDA tree, while C16 and C52 shared the overlap/offload tree. Keep the
two images separate until a combined tree passes all three performance and
accuracy gates.

| Arms | Patch | SHA256 | Source files |
|---|---|---|---:|
| C1 | `vllm_nightly_46638857_k3_c1_current.patch` | `554ec6384b4ae143df42b223af66a8365e2b466c7ea691ed6c5a26a8749a4e6d` | 44 |
| C16, C52 | `vllm_nightly_46638857_k3_c16_c52_current.patch` | `90f975fad15722494366153ec3f32a14c4445bfa88c51ec53043b88eaf64dcc0` | 34 |

Both patches are full diffs from vLLM commit
`46638857fdbb30e0c232c9e8f9cb1ff6d6f545c3`. They were dry-run against the
installed package in the pinned image and then compared byte-for-byte with the
retained accepted overlays. Two `.pyc` files and
`manager.py.before-partial-tail-20260830` were intentionally excluded from the
C16/C52 patch.

## Build the images

Run these commands from the InferenceX repository root:

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

For a release image, push immutable tags and replace the runtime patching in
the wrappers with the resulting image digests. Do not publish a single combined
tag until the combined source tree is validated at C1, C16, and C52.

The Dockerfile was build-tested on `mi355x-17`. The local verification image
IDs were:

```text
kimi-k3-vllm:c1-20260830       sha256:b31f5ab0435103f9279d765274654ef130d229982ab2d967a8a8b5757cd78cd7
kimi-k3-vllm:c16-c52-20260830  sha256:aacd27681dbb8717ef544799952ec08ba9c185038f9b931f1aafb56f593c104d
```

These IDs are local build outputs, not published registry digests.

## PR-derived source stack

The following local snapshots are present in both variants:

| Upstream work | Tested local commit(s) | Purpose |
|---|---|---|
| vLLM #51705 | `96e0305704`, `af987cdf72`, `4763986cf3`, `97bb6a9c40` | Kimi-K3 DSpark/DCP attention and verification support |
| vLLM #53598 | `d770e4e2f4` | Per-group hybrid-cache geometry and DCP prefix lookup |
| vLLM #53917 | `0a489b4b8b`, `394bb2fd34`, `4b478c8df5` | Hybrid geometry, Mamba replay boundaries, and failed-load recovery |
| vLLM #52707 | `f872fdd003` | Prevent negative external block allocation |
| vLLM #52494 | `9e08dccddd` | Fused MLA q/KV RMSNorm in the AMD Kimi-K3 wrapper |
| vLLM #52968 | `208916fb29` | Attention-residual, sigmoid-multiply, QKV-convolution, and related fusions |
| vLLM #53166 | `c92234cbce`, `43bd3ac18a`, `7662093dfc` | AITER MLA chunked-context gather and metadata-built KV indices |
| vLLM #54165 / #54163 | `09438c4eb5`, `1f11b7a933` | Preserve hybrid-Mamba cache hits with a KV connector |

These hashes identify the tested local snapshots. Open PR heads may have been
rebased or force-pushed and must not be assumed byte-identical.

Common custom commits:

- `c7d8e7b8de`: retain packed A2A buffers for the lifetime of captured ROCm
  graphs.
- `6cd48179ca`: bound SimpleCPU lazy-eviction scanning and avoid repeated full
  prefix-cache scans for in-flight stores.

## C1 additions

- vLLM #51392, local series `0cbbe1491b` through `906037cf83`: compose online
  quantization with a pre-quantized checkpoint.
- vLLM #54248, local commit `fe0647fd34`: advertise per-token FP8 input for
  AITER PTPC linears.
- vLLM #54254, local commit `b1ae0ffc2c`: fuse KDA gated RMSNorm with the
  per-token-FP8 `o_proj` path.
- Import-order cleanup in `quantization/online/fp8.py`.
- Direct return of the PTPC `o_proj` result in AMD `kda.py`.

The source tree used to generate the C1 patch was
`/home/hyukjlee/working_projects/vllm-k3-online-fp8-clean` at
`b1ae0ffc2c`, including its two documented working-tree changes.

## C16/C52 additions

- vLLM #52033-derived ROCm dual-stream shared-expert machinery plus local force
  and quantized-input safety controls. The accepted runs explicitly set
  `VLLM_ROCM_FORCE_SHARED_EXPERTS_STREAM=0`.
- vLLM #51437-derived overlap of the shared all-reduce with the routed
  up-projection. A local `fused_output.shape[0] <= 80` guard limits the fused
  AITER all-reduce/RMSNorm path.
- AMD KDA/Mamba checkpoint export, fine-grained replay boundaries, partial
  checkpoint re-keying, and partial-tail external-cache handoff.
- SimpleCPU request block-table and partial-tail integration.
- SimpleCPU asynchronous transfer-lifetime handling: keep GPU and CPU
  references pinned across reset until DMA completion, track outstanding block
  hashes, and deduplicate eager/lazy stores by logical hash.
- ROCm scaled-mm guard that clones a single-row activation when its apparent
  contiguous view has a wider row stride before `wvSplitKQ`.

The exact retained source overlay is
`/home/hyukjlee/k3-offload-overlap-51437-compose-20260830/vllm`. Its complete
37-file manifest has aggregate SHA256
`9f744c8326fe2759d96b460722094cc26eb1136ac484c293f6dfc6b41d7b4130`;
the clean patch contains the 34 source files only.

The SimpleCPU patch in this baseline is a tested composition, not an
upstream-ready change. The transfer-lifetime fix must be separated from the
partial-tail and speculative-cache semantics before proposing a vLLM PR.

## Serving envelopes

| Arm | Speculation | DCP | Offload | MNS | MBT | GMU | Graphs | Async |
|---|---|---:|---|---:|---:|---:|---|---|
| C1 | DSpark K=6, synthetic AL 3.84 | 1 | none | 2 | 8192 | 0.875 | 1-16 | off |
| C16 | none | 8/A2A | none | 80 | 8192 | 0.86 | 1-80 | off |
| C52 | none | 8/A2A | 512 GB `vllm-simple` | 80 | 16384 | 0.90 | 1-80, 128-4096 powers of two | on |

All arms use ROCm AITER MLA, prefill-query quantization, FP8 KV cache, prefix
match unit 128, SHA256 prefix hashes, and the same custom-op list in the
wrapper scripts.

## Baseline dispatch

The exact workflow dispatch command is:

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

The generated matrix must contain exactly C1, C16, and C52 before dispatch.

## C16 `torch.compile` candidate

The first material C16 improvement after the baseline came from vLLM
[#52190](https://github.com/vllm-project/vllm/pull/52190), tested at head
`c70113053761985aa289d5088503731c535dc028`. The baseline logged that
`torch.compile` was enabled but unsupported by Kimi-K3, so the advertised
`mla_dual_rms_norm` and `allreduce_rms` fusion passes never received a graph.

The post-overlay patch is:

```text
vllm_nightly_46638857_k3_compile_52190_delta.patch
SHA256 de1ac272820122281f865c4f81d3f7a87e03c0cb42feb59390d9012b9bb88c00
```

It is applied after
`vllm_nightly_46638857_k3_c16_c52_current.patch` and changes five files:

- `vllm/config/compilation.py`
- `vllm/models/kimi_k3/amd/kda.py`
- `vllm/models/kimi_k3/amd/latent_moe_runner.py`
- `vllm/models/kimi_k3/amd/linear.py`
- `vllm/models/kimi_k3/amd/ops/attn_res.py`

The delta preserves the accepted downstream KDA checkpoint, partial-tail, and
latent-MoE implementation. It adds Kimi-K3 compile support, makes the in-place
KDA core and attention-residual launcher compiler-visible custom operations,
adds the KDA core to the default graph splitting operations, and moves
one-time latent-tail logging out of compiled forward paths.

Build the clean C16 image with:

```bash
docker build \
  -f benchmarks/single_node/agentic/k3_patches/Dockerfile.kimi-k3-c16-compile52190 \
  -t kimi-k3-vllm:c16-compile52190-20260831 .
```

This Dockerfile was build-tested on `mi355x-17`. The local image ID was:

```text
sha256:e2fb0e238e7612f06cc47d656009584f8cc902b90bb1119c2b66e9330d3b3d1b
```

All 36 unique source paths touched by the base and delta patches matched the
tested runtime overlay byte-for-byte after the build. The image labels record
the base digest and #52190 head.

Two identical C16 synthetic screens first isolated the compile change with no
MTP, no offload, DCP8/A2A, MNS80, MBT8192, GMU0.86, and graph capture sizes 1
through 80:

| Arm | Aggregate tok/s | Tok/s/GPU | Mean TPOT | P99 TPOT | Requests |
|---|---:|---:|---:|---:|---:|
| Exact control | 40,005.53 | 5,000.69 | 34.493 ms | 37.043 ms | 96/96 |
| Compile run 1 | 46,106.25 | 5,763.28 | 31.357 ms | 32.559 ms | 96/96 |
| Compile run 2 | 46,039.66 | 5,754.96 | 31.410 ms | 32.370 ms | 96/96 |

The two-run average is 46,072.96 tok/s aggregate, or 5,759.12 tok/s/GPU:
15.17% above the exact control. Mean TPOT improved by 9.01%, and average P99
TPOT improved by 12.36%.

Artifacts on `mi355x-17`:

```text
/home/hyukjlee/k3-c16-compile-52190-screen-20260831/results/c16_compile52190_nospec_nooffload_noreplay_20260830T172845Z
/home/hyukjlee/k3-c16-compile-52190-repeat2-20260831/results/c16_compile52190_repeat2_nospec_nooffload_noreplay_20260830T173800Z
```

Adding DSpark K=3 with probabilistic drafting and synthetic acceptance length
3.00 to that compiled configuration produced:

| Aggregate tok/s | Tok/s/GPU | Mean TPOT | P99 TPOT | Requests |
|---:|---:|---:|---:|---:|
| 63,226.07 | 7,903.26 | 22.71 ms | 23.08 ms | 96/96 |

Artifact on `mi355x-17`:

```text
/home/hyukjlee/k3-c16-compile-52190-mtp3-20260831/results/c16_compile52190_mtp3_k3_synthetic_nooffload_noreplay_20260830T174817Z
```

The final candidate uses synthetic rejection only for performance measurement.
Accuracy is checked separately with the same server envelope and DSpark K=3,
changing only `rejection_sample_method` to `block`, using GSM8K and
`lm-eval --limit 128`. That check passed 128/128 for both strict match and
flexible extraction. Real block rejection reported mean acceptance length
approximately 3.46 and about 82% drafted-token acceptance.

Accuracy artifact on `mi355x-17`:

```text
/home/hyukjlee/k3-c16-compile-52190-mtp3-accuracy-20260831/results/gsm8k_limit128_c16_compile52190_mtp3_block_20260830T180014Z
```

The 3,600-second AIPerf gate remains mandatory before treating this delta as
the release configuration.

## Final candidate dispatch

The final matrix keeps the accepted C1 and C52 configurations and changes C16
to the compiled DSpark K=3 configuration above. Dispatch it with:

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

Before dispatch, confirm that the generated matrix contains exactly C1, C16,
and C52, with C16 marked `spec-decoding: mtp`.
