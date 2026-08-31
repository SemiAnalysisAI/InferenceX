# Kimi-K3 MI355X vLLM image reproduction

<div align="center">

**English** | [中文](./kimi-k3-mi355x-vllm-image-reproduction_zh.md)

</div>

Last updated: 2026-08-31

This document reconstructs the Kimi-K3 vLLM runtime used for the C16 CPRR
performance work. It separates the exact tested artifacts from upstream PR
heads so that a clean image can be rebuilt without relying on a modified
container or an AITER JIT cache.

## Base image

```text
vllm/vllm-openai-rocm:nightly-46638857fdbb30e0c232c9e8f9cb1ff6d6f545c3
sha256:8908b8ab5ba28c3b81f9f42bb72e2421f06a180e001c67c4f10ff7f127c5690b
```

The image contains vLLM commit `46638857fd` and `amd_aiter 0.1.19`. The
digest, rather than the mutable nightly tag, is the reproducible base.

## Layer order

The layers must be installed in this order:

1. Apply the C16/C52 vLLM overlay:

   ```text
   benchmarks/single_node/agentic/k3_patches/
     vllm_nightly_46638857_k3_c16_c52_current.patch
   SHA256 90f975fad15722494366153ec3f32a14c4445bfa88c51ec53043b88eaf64dcc0
   ```

2. Apply the five-file vLLM #52190 delta:

   ```text
   benchmarks/single_node/agentic/k3_patches/
     vllm_nightly_46638857_k3_compile_52190_delta.patch
   SHA256 de1ac272820122281f865c4f81d3f7a87e03c0cb42feb59390d9012b9bb88c00
   tested PR head c70113053761985aa289d5088503731c535dc028
   ```

3. Install the complete CPRR runtime bundle:

   ```text
   benchmarks/single_node/agentic/k3_patches/
     aiter_pr4521_plus_4964_runtime/
   SHA256(SHA256SUMS) cb6f7ab6210d876e674f276cbaacf638936358cc12c1f89622084a611bb1d342
   files covered by manifest: 62
   ```

The Dockerfile implementing this sequence is:

```text
benchmarks/single_node/agentic/k3_patches/
  Dockerfile.kimi-k3-c16-compile52190-cprr
```

This is the selected A8W4/CPRR image. vLLM PR #53940 and the matching local
AITER selector patch were evaluated separately because they route SiTUv2 MoE
to A4W4. They are not part of this Dockerfile: the exact C16 screen was slower
and had worse P90 TPOT than the selected A8W4 path. See "Rejected A4W4
experiment" below.

## PR and custom-fix provenance

The C16/C52 overlay is an exact snapshot, not a promise that the current heads
of all listed PRs cherry-pick together. Its upstream-derived components are:

| Work | Tested local commits | Purpose |
|---|---|---|
| vLLM #51705 | `96e0305704`, `af987cdf72`, `4763986cf3`, `97bb6a9c40` | Kimi-K3 DSpark and DCP attention/verification support. |
| vLLM #53598 | `d770e4e2f4` | Per-group hybrid-cache geometry and DCP prefix-cache lookup. |
| vLLM #53917 | `0a489b4b8b`, `394bb2fd34`, `4b478c8df5` | Hybrid geometry consistency, Mamba replay boundaries, and failed-load recovery. |
| vLLM #52707 | `f872fdd003` | Prevent negative external block allocation. |
| vLLM #52494 | `9e08dccddd` | Fuse MLA Q/KV RMSNorm in the AMD Kimi-K3 wrapper. |
| vLLM #52968 | `208916fb29` | Kimi-K3 attention-residual, sigmoid-multiply, and QKV-convolution fusions. |
| vLLM #53166 | `c92234cbce`, `43bd3ac18a`, `7662093dfc` | AITER MLA chunked-context gather and metadata-built KV indices. |
| vLLM #54165 | `09438c4eb5`, `1f11b7a933` | Preserve hybrid-Mamba cache hits with DFlash/DSpark and a KV connector. |
| vLLM #52033 | retained local snapshot | ROCm dual-stream shared-expert machinery. |
| vLLM #51437 | retained local snapshot | Overlap shared all-reduce with routed up-projection. |
| vLLM #52190 | `c70113053761985aa289d5088503731c535dc028` | Enable Kimi-K3 compilation and declare the KDA and attention-residual mutation boundaries. |

The overlay also contains these custom fixes:

- `c7d8e7b8de`: retain packed A2A buffers for captured ROCm graph lifetime.
- `6cd48179ca`: bound SimpleCPU lazy-eviction scanning.
- Retain in-flight SimpleCPU store IDs, hashes, GPU pins, and CPU references
  across reset until asynchronous DMA completion. Abandoned completions release
  ownership without publishing stale data to the CPU prefix cache.
- Export and reuse KDA/Mamba internal checkpoints and eligible partial tails.
- Integrate SimpleCPU partial-tail transfer specifications with eager/lazy
  stores.
- Restrict the #51437-derived fused all-reduce/RMSNorm path to batches of at
  most 80 tokens.
- Add a shared-expert stream force flag and a quantized-input safety predicate.
  The selected runs leave `VLLM_ROCM_FORCE_SHARED_EXPERTS_STREAM=0`.
- Clone a single-row activation with a wider apparent row stride before
  entering ROCm `wvSplitKQ`.

The broader SimpleCPU/partial-tail changes must be split and reviewed before
any upstream submission. The transfer-lifetime behavior and the boundary
between that fix and the broader cache work are summarized in the
[C16/C52 additions](./kimi-k3-mi355x-vllm-current-baseline.md#c16c52-additions).

## Compatible AITER reconstruction

AITER PR #4964 cannot be rebuilt directly on the source snapshot retained in
the nightly image. That source is older than the prebuilt dispatcher installed
by AITER PR #4521. Rebuilding it discards #4521 Python/C++ ABI and dispatch
changes while vLLM still calls the newer interface; the result reproduced GPU
memory-access faults.

The compatible source reconstruction is:

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

The reconstruction worktree is:

```text
/home/hyukjlee/working_projects/aiter-pr4521-plus4964
```

The runtime bundle includes the matching Python dispatcher, C++ interface and
metadata sources, complete gfx950 MLA registry/kernel directory, and both
prebuilt modules. Important hashes are:

```text
vLLM rocm_aiter_mla.py
20ed4a04e4fb730dd9031eb3be8462e9193fda0c0a8a4daeb202d587c6ac310d

module_mla_asm.so
2cb23f820559cfd60d8bb27ea82fb9fa31cce9115a098097725825e8c20da505

module_mla_metadata.so
2806c1c2d7ddf9bebe72b0a1a1a42fe8035c3a7c299cb653f4d80cc21ad6c556
```

## Build

From the InferenceX repository root:

```bash
docker build \
  -f benchmarks/single_node/agentic/k3_patches/Dockerfile.kimi-k3-c16-compile52190-cprr \
  -t kimi-k3-vllm:c16-compile52190-cprr-20260831 .
```

The build verifies both vLLM patch hashes, the runtime-manifest hash, and every
file listed in the manifest before changing site-packages. It uses prebuilt
gfx950 modules, so the Docker build itself does not need GPU access.

The Dockerfile was build-tested on `mi355x-17` on 2026-08-31. The resulting
local image was:

```text
kimi-k3-vllm:c16-clean-cprr-20260831
sha256:247e2b452c4745cbe119148caae4979f09ed131a2747551748912f19afe2668f
```

The build checked all 62 files in the CPRR manifest. Hashes of the installed
vLLM compile files, vLLM MLA wrapper, AITER Python files, and both prebuilt MLA
modules matched the previously tested runtime image
`kimi-k3-vllm:c16-cprr4521-4964-20260831` byte for byte.

## Byte verification

After the build, verify the installed files rather than relying only on image
labels:

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

The expected hashes are recorded in the bundle's `SHA256SUMS`. The Dockerfile
also records labels for the base digest, vLLM #52190, AITER #4521, AITER #4964,
the compatibility commit, and the manifest hash.

Do not mount the legacy `k3_aiter_jit_46638857` volume over
`/usr/local/lib/python3.12/dist-packages/aiter/jit` when running this image.
Such a mount hides the two CPRR modules baked into the image. This was
confirmed in a failed clean-image accuracy startup: the runtime reported no
heuristic kernel for GQA96, QLen4, LSE+CPRR and then lost a worker. Removing
the volume exposes the SHA-verified modules and preserves image/runtime
equivalence. A fresh cache may be mounted elsewhere, but the bundled module
directory itself must not be shadowed.

## GitHub Actions runtime equivalence

The Actions configuration continues to name the pinned upstream nightly.
For the C16 MTP arm,
`benchmarks/single_node/agentic/kimik3_fp4_mi355x_vllm_mtp.sh` selects the two
vLLM patches and the exact CPRR runtime bundle. The common launcher verifies
the bundle and installs the same files before importing the serving engine.
This makes the Actions runtime byte-equivalent to the clean Dockerfile for the
files that differ from the base image.

## Validation evidence

- Focused numerical test:
  `test_dcp_fp8_round_robin_verify_matches_causal_attention` passed.
- The test loaded
  `mla_a8w8_qh32_qseqlen4_gqaratio32_lse_cprr_ps`, the required kernel for
  Kimi-K3 DCP8 verification (`decode_heads=96`, `logical_qlen=4`).
- PIECEWISE and FULL graph capture completed at MNS/CG16 and served 32/32
  requests without a GPU fault.
- MNS/CG80 at GMU 0.84 completed capture but had no KV-cache budget.
- MNS/CG64 at GMU 0.84 completed capture but reported `-2.18 GiB` available
  KV-cache memory. This was a clean sizing failure, not a GPU fault.

### C16 CPRR serving screens

All rows below use TP8/DCP8 A2A, DSpark K=3, synthetic acceptance length 3.0,
MBT8192, no KV offload, and forced shared-expert streaming disabled unless
stated otherwise. Throughput is total-token throughput divided by eight.

| MNS / graph envelope | GMU | Throughput/GPU | P90 TPOT | Outcome |
|---|---:|---:|---:|---|
| 64 / dense through 64 | 0.86 | 6,960.97 tok/s | 15.44 ms | 96/96 |
| 16 / sparse through 64 | 0.84 | 11,829.58 tok/s | 34.25 ms | 96/96 |
| 8 / `1,2,4,8,16,24,32` | 0.84 | 9,610.86 tok/s | 17.20 ms | 96/96 |
| 6 / `1,2,4,6,8,12,16,20,24` | 0.84 | **8,141.60 tok/s** | **16.02 ms** | 96/96 |
| 8 / `1,2,4,8,16,24,32`, forced stream | 0.84 | 9,155.52 tok/s | 17.94 ms | 96/96 |

The selected MNS6 result is stored at:

```text
/home/hyukjlee/k3-c16-cprr4964-smoke-20260831/results/gmu0.84_20260830T232533Z
```

It keeps roughly 10% throughput headroom over the requested 7,381.5
tok/s/GPU while lowering P90 TPOT by 1.18 ms relative to MNS8. The higher
throughput MNS8 result remains useful as a throughput-biased alternative and
is stored at:

```text
/home/hyukjlee/k3-c16-cprr4964-smoke-20260831/results/gmu0.84_20260830T223841Z
```

The MNS8 run reported 13.05 GiB of KV-cache memory, 4,289,629-token KV capacity,
`VmPin=0`, and no in-run GPU fault. The traceback at the end of the server log
was generated by the launcher's deliberate `docker stop` after the benchmark.

### Rejected A4W4 experiment

vLLM PR #53940 at
`47cd3318351d9a62a900529fbe88a1f64b293532` was applied as the four runtime
source-file delta with SHA256
`29bbe3bef0e99799c7394870f75e83aedba90d86494c2947de54b140ad1cfba7`.
The nightly AITER dispatcher did not contain the corresponding
`AITER_SITUV2_A4W4` selector, so the first run silently used
`flydsl_moe1_abf16_wfp4_*` and was invalid as an A4W4 comparison.

A minimal selector patch with SHA256
`d23e36ff40f28632a4ae2f27b6896e15802a13ee4eed4c1cb62e84a532af13a7`
was then applied to `aiter/fused_moe.py`. Import-time verification showed
`AITER_SITUV2_A4W4=1` and the legacy `AITER_SITUV2_A8W4` removed. Capture logs
confirmed `flydsl_moe1_afp4_wfp4_bf16_*` and the same CPRR MLA kernel used by
the selected path.

The valid A4W4 screen completed 96/96 requests at 9,396.09 tok/s/GPU and
18.10 ms P90 TPOT. Because both metrics were worse than the selected MNS8
A8W4/CPRR result, #53940 and the AITER selector are documented as rejected
experiments and are not installed by the clean image.

### Clean-image accuracy

The selected MNS6/CG24 configuration passed GSM8K five-shot on the clean image
with block rejection and `lm-eval --limit 128`:

- strict match: 128/128
- flexible extraction: 128/128
- observed mean acceptance length: 3.44-3.49
- observed drafted-token acceptance: 81.2%-83.1%
- no worker death, OOM, GPU fault, or missing-kernel error
- all eight GPUs returned to 0% use and 0% allocated VRAM after teardown

Artifact on `mi355x-17`:

```text
/home/hyukjlee/k3-c16-clean-cprr-accuracy-20260831/results/gsm8k_limit128_c16_clean_cprr_mns6_cg24_block_20260830T234636Z
```

The remaining release gates are:

1. A 3,600-second GitHub Actions C1/C16/C52 run.
2. Exported-result, server-log, memory-growth, and GPU-release audits.
