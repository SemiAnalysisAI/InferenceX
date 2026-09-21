# DeepSeek V4.1 ROCm attention backport

**English** | [中文](README_zh.md)

This isolated Python package brings the official preview's compression-ratio 1/2
ROCm attention implementation onto the pinned September 21 nightly image. The
stock nightly fails graph capture with `No indexer pool for compression ratio 4`.
Only the HIP `deepseek_v41` registry path selects this package. Other model and
hardware paths retain the original nightly implementation.

`provenance.json` records both immutable image digests, the original file hashes,
and the adapted file hashes. The preview image history identifies overlay
`f8f290f2`, which is not publicly resolvable; the image digest is authoritative.
The preview and nightly use ROCm 7.2.4, AITER
`4ad99832823dde2315b361cbd3b54b1c5c12acd5`, and Triton
`3.7.0+amd.rocm7.2.0.git89002410`. No preview binaries replace nightly binaries.

Adaptations isolate imports, follow the nightly's moved candidate-indexer and
capture APIs, and read kernel configuration through `get_exec()`. The current
nightly disables fused low-ratio compression on HIP, so its unfused compressor
weight layout is retained. The nightly model applies inverse RoPE after attention;
it does not supply the preview's optional fused inverse-RoPE argument.

`install.py` verifies the original registry and every payload hash before copying
these files into the job's ephemeral container. It rejects an unexpected registry
revision and records the installed registry hash in the result directory. Run it
before importing the attention registry. Installation is idempotent.

Both STP and DSpark retain the quantization defaults shipped with the pinned
nightly. No custom draft weight or projection conversion patch is installed.
The V4.1 gfx950 block-FP8 adapter reuses the nightly's existing Triton group
quantizer and GEMM because the generic UE8M0 wrapper selects a CUDA-only JIT
header. It preserves group size 32, E4M3 range, the 1e-10 absmax floor, and upward
power-of-two scales. The dispatcher is gated to V4.1 on HIP; original paths for
other models remain intact. GPU numerical and graph tests cover the adapter.

Passing startup or a limited eval is not full performance or accuracy
qualification. Source is adapted from SGLang under the accompanying Apache 2.0
license.
