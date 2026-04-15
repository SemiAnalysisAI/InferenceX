# Gated 1M-class Qwen3.5 preview lane

This directory carries the committed InferenceX-side Qwen3.5 artifacts for a
bounded `1M`-class ISB1 coding replay preview.

## What these files are

- dedicated replay bundles restricted to `qwen3_5_397b_a17b`
- producer cells for standalone `vllm` and standalone `sglang`
- committed bundle coverage for `nvidia:b200_sxm_180gb`, `nvidia:h100_sxm_80gb`, and `nvidia:h200_sxm_141gb`
- restricted to `ulc2_1m_plus`
- restricted to `support_status=reviewed_preview` at the selected export-cell level
- restricted to `benchmark_certification_status=dataset_replay_verified`
- exposed downstream only through the separate manual config
  `.github/configs/isb1-qwen-1m-preview.yaml`
- explicit `max-model-len: 1048576` when the manual config is used

## Current claim boundary

These files are committed preview artifacts plus a gated/manual validation path.
They do **not** imply ordinary runnable ISB1 support in `isb1-master.yaml`.

Safe wording:
- InferenceX carries bounded 1M-class Qwen3.5 replay preview artifacts.
- InferenceX carries a separate gated/manual Qwen3.5 1M validation path.

Unsafe wording:
- native 1M served-lane support
- ordinary/general runnable consumer support
- KV-offload certification

See `manifest.json` for the exact preview boundary and
`.github/configs/isb1-qwen-1m-preview.yaml` for the manual validation surface.
