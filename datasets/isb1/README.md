# ISB1 replay artifacts for InferenceX

This directory is the InferenceX-side consumer package for ISB1 replay.

InferenceX consumes committed file artifacts only:
- replay export JSON bundles under `datasets/isb1/exports/`
- consumer configs in `.github/configs/isb1-*.yaml`
- replay processing through `utils/bench_serving/benchmark_export_replay.py`
- result normalization through `utils/process_result_isb1.py`

InferenceX does **not** import external runtime code and does **not** make
live-serving claims from export-file existence alone.

---

## Why not random data?

Random-data benchmarks show worst-case performance. Real inference workloads
have multi-turn conversations where each turn shares context with previous
turns. This enables:

- **Prefix caching** — 60-95% of each request's tokens are shared with the
  previous turn. Prefix cache hit rates directly affect throughput.
- **KV cache reuse** — the server reuses computed KV cache entries instead of
  recomputing them. This is the biggest performance optimization in production.
- **Realistic offload behavior** — KV cache grows across turns, eventually
  exceeding GPU memory and requiring CPU offload. Random data never reaches
  this point because each request is independent.

These traces stress-test the exact KV cache behaviors that determine real
production performance.

---

## Coverage

Strict audit rule: count only model-architecture-valid cells. Per-model context
limits (DSR1 163,840; GPT-OSS 131,072; Qwen3.5 1,010,000) produce N/A rows
above each model's max.

### Verified coverage

| Model | Chat | Code |
|---|---|---|
| `dsr1` | `8k`, `32k`, `64k`, `131k` | `8k`, `32k`, `64k`, `131k` |
| `gptoss` | `8k`, `32k`, `64k`, `131k` | `8k`, `32k`, `64k`, `131k` |
| `qwen3.5` | `8k`, `32k`, `64k`, `131k`, `500k` | `8k`, `32k`, `64k`, `131k`, `500k` |

### Existing preview artifacts

- `gptoss` `500k` chat/code preview files exist at `reviewed_preview` tier
- `qwen3.5` `1M` chat/code preview files exist at `gated` tier (consumed only
  through `isb1-qwen-1m-preview.yaml`)
- `dsr1` has no `500k` or `1M` lane because the model tops out at `163,840`

---

## Inventory

### Export-file layout (post-flatten)

Bundle files are flat per context-band directory — framework-specific variants
are consolidated into single files whose internal cell rows carry runtime
metadata.

| Subtree | Bundle files | Notes |
|---|---:|---|
| `core/` | 4 | 8K chat/code × {generic, qwen3.5} |
| `extension_32k/` | 4 | 32K chat/code × {generic, qwen3.5} |
| `extension_64k/` | 4 | 64K chat/code × {generic, qwen3.5} |
| `extension_131k/` | 5 | 131K chat/code × {generic, qwen3.5, dsr1 chat} |
| `preview/long_context_500k/` | 4 + 2 manifests | 500K chat/code × {gptoss, qwen3.5} |
| `preview/long_context_1m/` | 2 + 1 manifest | 1M chat/code × qwen3.5 |

All export files are valid JSON and replay-hydratable via
`utils/bench_serving/benchmark_export_replay.py`.

---

## Support-status vocabulary

ISB1 replay surfaces classify under the five-class support vocabulary:

- `supported` — core 8K replay path
- `reviewed_preview` — 32K / 64K / 131K extensions, 500K preview
- `gated` — 1M preview (manual config only)
- `artifact_only` — retained artifacts without live replay
- `unsupported` — not a valid path

No ISB1 surface claims `live_benchmark_certification`; all claims are bounded
to `dataset_replay_verified`.

---

## Claim boundary

Safe claims:
- InferenceX carries the ISB1 replay corpus described above.
- Strict replay-file coverage is **26 valid / 0 failed / 10 N/A** across 36
  (model × band × workload) combinations.
- DSR1 strict coverage stops at `131k`.
- GPT-OSS strict coverage stops at `131k`.
- Qwen3.5 strict coverage reaches `500k`.
- GPT-OSS `500k` and Qwen3.5 `1M` files exist but are excluded from the strict
  pass count (`reviewed_preview` and `gated` tiers, respectively).

Unsafe claims:
- `26/26` valid cells verified (10 N/A due to model `max_position_embeddings`
  limits)
- strict GPT-OSS `500k` coverage
- strict Qwen3.5 `1M` coverage
- turning preview-file existence into live benchmark certification

---

## Related docs

- [`COEXISTENCE_WITH_KV_CACHE_TESTER.md`](COEXISTENCE_WITH_KV_CACHE_TESTER.md) —
  how PR #1032 coexists with PR #993's kv-cache-tester
- [`GMI_EXECUTION_PLAN.md`](GMI_EXECUTION_PLAN.md) — bare-metal execution
  runbook for ISB1 replay on GMI Cloud Hopper and Blackwell
- [`exports/preview/long_context_500k/README.md`](exports/preview/long_context_500k/README.md) —
  500K preview lane claim boundary
- [`exports/preview/long_context_1m/README.md`](exports/preview/long_context_1m/README.md) —
  1M gated preview lane claim boundary
