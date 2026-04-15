# Bounded 500k-class preview lanes

This directory carries the smallest honest InferenceX consumer handoff for bounded
`500k`-class ISB1 coding replay paths.

## What these files are

- dedicated replay bundles derived from committed `131k1k` extension exports
- restricted to `gpt_oss_120b` or `qwen3_5_397b_a17b`
- restricted to `xlc2_384k_512k`
- restricted to standalone `vllm` and standalone `sglang`
- restricted to `nvidia:b200_sxm_180gb`, `nvidia:h100_sxm_80gb`, and `nvidia:h200_sxm_141gb`
- restricted to `support_status=reviewed_preview`
- restricted to `benchmark_certification_status=dataset_replay_verified`
- wired in the consumer with explicit `max-model-len: 524288`

## What these files are not

- not a native InferenceX `500k+` served lane
- not a native InferenceX `1M+` served lane
- not a supported-tier long-context expansion
- not a chat preview lane
- not an offload-depth lane
- not a KV-offload certification claim

## Why the files exist

The existing `extension_131k/*/code_131k1k.json` and model-scoped
`code_131k1k_qwen3.5.json` bundles already contain honest `xlc2_384k_512k`
replay cells, but they are mixed with lower-band cells. The InferenceX workflow
selects rows by runtime, hardware, model, and support tier — not by
`context_band`.

These dedicated files isolate only the `xlc2_384k_512k` rows so InferenceX can
run bounded `500k`-class previews without over-selecting lower-band cells.

## Consumer contract

- `isb1-master.yaml` pins these rows as `reviewed_preview`
- `isb1-master.yaml` pins `max-model-len: 524288`
- current search space is intentionally bounded to single-concurrency preview execution
- result processing preserves `context_bands`, `profile_id`, and the producer handoff claim boundary

See `manifest.json` for the GPT-OSS derivation record and `manifest_qwen3.5.json`
for the Qwen derivation record.
