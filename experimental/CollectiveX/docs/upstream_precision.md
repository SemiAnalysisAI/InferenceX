# Upstream precision work — review + mapping to CollectiveX (goal P1 "Integrate precision-related upstream work")

Reviews the three precision PRs named in goal.md and maps each onto CollectiveX's precision axes
(`shape.dispatch_dtype`, `shape.quant.combine_input_dtype/combine_quant_mode`, the
`combine_quant_in_timing` reproduction flag, and the `capability.py` / `backends.yaml` `combine_dtypes`
+ `quant_modes` sets). All three are MERGED upstream. CollectiveX now has real runs for the supported
FlashInfer MXFP8/NVFP4 paths and keeps MXFP4 as a reserved-but-gated mode until its scale-factor layout
can be represented honestly in the current A2A payload contract.

## MoRI PR 311 — `feat(EP): FP8 blockwise quantization for IntraNode combine` (ROCm/mori, MERGED)
- **What:** adds `QuantType::Fp8BlockwiseQuant` (Python `fp8_blockwise`) — a quant-aware FP8 combine for
  the IntraNode EP path, replacing MoRI's old direct-cast (which truncated activations above the e4m3
  range and degraded SGLang DeepSeek-R1 accuracy at high concurrency). Per-token per-block max-abs scale
  on the quant side; per-block FMA dequant on recv. Block size = `hidden_dim / scale_dim`.
- **Maps to:** the `combine_quant_mode` axis. CollectiveX's `ep_mori.py` / `capability.py` /
  `backends.yaml` already reserve this ("`+ fp8 when the MoRI quant_type combine path (PR311) lands`").
  The reserved mode id is now concrete: **`fp8_blockwise`** with `combine_input_dtype=fp8`,
  per-block scale layout — exactly the CollectiveX `combine_quant_mode` + `scale_layout` fields.
- **Scope:** AMD/MI355X (MoRI is the AMD backend). Out of scope for *NVIDIA chips*, but it is the
  reference design for the quant-combine contract that the NVIDIA backends will mirror.

## FlashInfer PR 3376 — `feat: add mxfp8 quant to moe a2a combine` (flashinfer-ai/flashinfer, MERGED)
- **What:** `moe_a2a_combine` can directly output **MXFP8** — adds `output_dtype`, `output_scales`,
  `sf_layout`; bumps `kMaxPayloads` for per-token quantization dispatch.
- **Maps to:** `combine_quant_mode=mxfp8`, `combine_output_dtype=mxfp8`, `scale_layout=sf_layout`, and
  `combine_quant_in_timing=true` (the quant is inside the combine kernel). This is the NVIDIA
  quantized-combine path.

## FlashInfer PR 3643 — `feat: add mxfp4/nvfp4 quant to moe a2a combine` (flashinfer-ai/flashinfer, MERGED)
- **What:** follow-up to 3376; adds **MXFP4 / NVFP4** quant to `moe_a2a_combine`, plus
  `output_scalar_scale: float = 1.0`.
- **Maps to:** `combine_quant_mode ∈ {mxfp4, nvfp4}`, `combine_output_dtype ∈ {mxfp4, nvfp4}`. These are
  the goal's "NVFP4 combine" / "MXFP8 combine" precision-matrix rows, and (via the dispatch side of the
  same kernel family) the "NVFP4/MXFP4/MXFP8 dispatch" rows.

## Current NVIDIA run status (see docs/gated.md)
This note was originally written before the FlashInfer adapter landed. The current status is now:
- **FlashInfer dispatch:** BF16, e4m3 FP8 variants, MXFP8, and NVFP4 dispatch have valid runs where
  the backend and architecture support them. NVFP4 is Blackwell-only.
- **FlashInfer quantized combine:** MXFP8 and NVFP4 combine have valid B300 runs through the
  `moe_a2a_combine` output-quant path. H100 was build-budget-limited for the source-build path, not
  architecturally ruled out — and since the nightly wheel gained `output_dtype` the source build is no
  longer needed, so an H100 mxfp8-combine re-run is attainable (subject to the h100 intermittent MNNVL
  deadlock; see docs/gated.md).
- **MXFP4 dispatch/combine:** still gated because the FlashInfer MXFP4 scale-factor layout is
  tile-padded/swizzled rather than a simple per-token tensor that can be moved through the current A2A
  payload list.

DeepEP's own dispatch remains e4m3-fp8-only; the wider MXFP8/NVFP4/MXFP4 matrix belongs to the
FlashInfer MoE all-to-all path.

## What CollectiveX did with this review
- **Capability table:** the mode ids are now named in `capability.py` / `backends.yaml`
  comments (`fp8_blockwise` for mori; `mxfp8`/`mxfp4`/`nvfp4` for the flashinfer combine path). MXFP8
  and NVFP4 are runnable where the backend/architecture supports them; MXFP4 remains rejected by
  `capability.resolve` until the scale-factor layout is movable through the payload list.
- **Schema/labels:** `shape.quant.{combine_input_dtype,combine_quant_mode,combine_output_dtype,
  scale_layout}` + `reproduction.combine_quant_in_timing` already exist (v4 schema), so a quantized-
  combine result is a distinct, correctly-labelled comparison point the moment one is produced.
- **Correctness tests:** the runnable MXFP8/NVFP4 dispatch and B300 quant-combine paths are covered by
  the `reference_ep.py` oracle with explicit tolerance classes. MXFP4 correctness remains deferred
  because no valid MXFP4 payload representation is currently emitted.
