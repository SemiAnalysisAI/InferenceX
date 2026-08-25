# Inference-engine patch waiver - PR #2726

Filed per [`docs/PR_REVIEW_CHECKLIST.md`](../PR_REVIEW_CHECKLIST.md): this
benchmark applies a pinned serving-stack source delta to the public upstream
image before serving. The patch is required to reproduce the validated curve
and is explicitly recorded in each result directory.

## Config covered

- Master config: `minimaxm3-fp4-mi355x-vllm-agentic-mtp`
- Image tag: `vllm/vllm-openai-rocm:v0.27.1`
- Validated image digest: `sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7`
- vLLM source base: `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`
- AITER source base: `545d97cc0aaeef7915e2c6df80b7f63f9d8ad657`
- vLLM patch SHA-256: `662e0d70ccd051225b638bfd1f541f0861e1fbba56502696d65937906dcd1162`
- AITER patch SHA-256: `b3d47fc883288532e92cf026945ce7f7d61fff1a100f7c731710e531c34ca742`
- Precompile helper SHA-256: `cbc30626128b18a917dc6f4145945eb6f109f708675eaf9b191e770bbd6cda5c`
- Entry point: `benchmarks/single_node/agentic/apply_minimaxm3_agentx_patches.sh`

## What is patched

The patch script applies two checked-in, pristine-to-runtime diffs and fails
closed if either source tree differs from the validated base.

- vLLM: MiniMax-M3 AITER sparse-indexer integration, CUDA-graph metadata
  repair, AgentX dynamic-kernel warmup, FP8 query/output handling for unified
  attention, EAGLE3 draft-window support with full KV retention, and the fused
  QK-norm/RoPE/cache-insert plus fused all-reduce/Gemma routing used by the
  accepted runs.
- AITER: the fused MiniMax-M3 QK-norm/RoPE/shuffled-cache insert operation and
  the gfx950 sliding-window 3-D unified-attention path used by the target and
  EAGLE3 draft.

The script verifies the checked-in vLLM patch, AITER patch, and precompile
helper by SHA-256, then writes those hashes together with the image, source
pins, and apply status to `container_patches.txt` beside the benchmark
artifacts. It changes only the ephemeral container filesystem and places JIT
outputs under the runner's cache; it does not write root-owned files into the
GitHub Actions workspace.

## Why the unmodified image cannot reproduce this curve

The stock v0.27.1 image can run the earlier Triton/LMCache recipe, but it does
not contain the optimized AITER indexer, repaired unified-attention graph
metadata, full long-context JIT warmup, or fused cache-insert path used by the
validated result. Removing the delta changes the serving stack and invalidates
the claimed performance provenance; it can also allow first-use JIT work to
enter the measured phase.

## Upstream status

- The public image is upstream vLLM; no custom image is used.
- The exact vLLM and AITER deltas are included in this PR for review and
  reproducibility. Standalone upstream engine PRs are still required.
- Related vLLM MiniMax-M3 AITER work:
  https://github.com/vllm-project/vllm/pull/52849

## Removal plan

Land the engine deltas in vLLM/AITER, move the InferenceX config to the first
upstream `vllm/vllm-openai-rocm` image containing them, rerun the full AgentX
sweep and eval, then remove the patch script, both patch files, and this waiver
in the same image-bump PR.
