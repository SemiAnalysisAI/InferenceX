# Upstream precision work — review + mapping to CollectiveX (goal P1 "Integrate precision-related upstream work")

Reviews the three precision PRs named in goal.md and maps each onto CollectiveX's precision axes
(`shape.dispatch_dtype`, `shape.quant.combine_input_dtype/combine_quant_mode`, the
`combine_quant_in_timing` reproduction flag, and the `capability.py` / `backends.yaml` `combine_dtypes`
+ `quant_modes` sets). All three are MERGED upstream. CollectiveX already carries the *scaffold* for
them (the combine-path axes default to bf16/none and are validated by `capability.resolve`), so each PR
maps to a concrete, reserved mode id that slots in when the kernel is wired + hardware-available.

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

## Why these are not yet RUN on NVIDIA (see docs/gated.md)
The FlashInfer combine quant (3376/3643) lives in `flashinfer.comm.moe_a2a_*` — the same MoE all-to-all
that needs a **symmetric multi-process MNNVL workspace**. On x86_64 (H100/H200/B200) that needs
`CAP_SYS_PTRACE`/pidfd (not granted in the enroot/pyxis container); on aarch64 (GB200/GB300) it uses
CUDA FABRIC handles (would work; GB300 capacity-limited). So MXFP8/MXFP4/NVFP4 *combine* (and the fp4
*dispatch* in the same family) are reachable on NVIDIA only once that container-capability/hardware
blocker is resolved — they are not silently faked. DeepEP's own dispatch remains e4m3-fp8-only.

## What CollectiveX did with this review
- **Capability table:** the reserved mode ids are now named in `capability.py` / `backends.yaml`
  comments (`fp8_blockwise` for mori; `mxfp8`/`mxfp4`/`nvfp4` for the flashinfer combine path) so a
  future wiring is a one-line capability widening, not a redesign. They remain **rejected** by
  `capability.resolve` today (not runnable → not claimed).
- **Schema/labels:** `shape.quant.{combine_input_dtype,combine_quant_mode,combine_output_dtype,
  scale_layout}` + `reproduction.combine_quant_in_timing` already exist (v4 schema), so a quantized-
  combine result is a distinct, correctly-labelled comparison point the moment one is produced.
- **Correctness tests:** deferred with the kernels — when a quant-combine path is wired, the
  `reference_ep.py` oracle gains a tolerance class per `combine_quant_mode` (looser e4m3/fp4 bound),
  mirroring the existing fp8-dispatch tolerance (1.25e-1 vs bf16 5e-3).
