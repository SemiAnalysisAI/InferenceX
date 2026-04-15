# ISB1 replay artifacts for InferenceX

This directory is the InferenceX-side consumer package for ISB1 replay.

InferenceX consumes committed file artifacts only:
- replay export JSON bundles under `datasets/isb1/exports/`
- consumer configs in `.github/configs/isb1-*.yaml`
- replay processing through `utils/bench_serving/benchmark_export_replay.py`
- result normalization through `utils/process_result_isb1.py`


## Why not random data?

Random data benchmarks show worst-case performance. Real inference workloads
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

InferenceX does **not** import external runtime code and does **not** make live-serving claims from export-file existence alone.

---

## Current ground truth (verified 2026-04-12)

The definitive strict audit found:

- **26 PASSED**
- **0 FAILED**
- **10 N/A**

Strict audit rule: count only model-architecture-valid cells.

### Strict verified coverage

| Model | Chat | Code |
|---|---|---|
| `dsr1` | `8k`, `32k`, `64k`, `131k` | `8k`, `32k`, `64k`, `131k` |
| `gptoss` | `8k`, `32k`, `64k`, `131k` | `8k`, `32k`, `64k`, `131k` |
| `qwen3.5` | `8k`, `32k`, `64k`, `131k`, `500k` | `8k`, `32k`, `64k`, `131k`, `500k` |

### Existing but excluded from the strict pass count

- `gptoss` `500k` chat/code preview files exist, but strict coverage stops at `131k`
- `qwen3.5` `1M` chat/code preview files exist, but were excluded from the strict audit
- `dsr1` has no strict `500k` or `1M` lane because the model tops out at `163840`

---

## Inventory

### Export-file counts

- **50 export files**
- **3 JSON manifests**
- **53 total JSON files** under `datasets/isb1/exports/`
- **888 total cells**
- **5,094 total turns**
- **13 MB actual message content**
- **All export files are valid JSON**

### Export-file breakdown

| Class | Count |
|---|---:|
| Core `8k1k` | 8 |
| Extension `32k1k` | 8 |
| Extension `64k1k` | 8 |
| Extension `131k1k` | 10 |
| Preview `offload_core` | 4 |
| Preview `500k` | 8 |
| Preview `1M` | 4 |
| JSON manifests | 3 |

---

## Claim boundary

Safe claims:
- InferenceX carries the full audited ISB1 replay corpus described above.
- Strict replay-file coverage is **26 passed / 0 failed / 10 N/A**.
- DSR1 strict coverage stops at `131k`.
- GPT-OSS strict coverage stops at `131k`.
- Qwen strict coverage reaches `500k`.
- GPT-OSS `500k` and Qwen `1M` files exist, but are excluded from the strict pass count.

Unsafe claims:
- `26/26` valid cells verified (10 N/A due to model `max_position_embeddings` limits: DSR1=163,840, GPT-OSS=131,072, Qwen3.5=1,010,000)
- strict GPT-OSS `500k` coverage
- strict Qwen `1M` coverage
- turning preview-file existence into live benchmark certification

---

## Key docs

- [`COVERAGE_AUDIT_2026-04-11.md`](COVERAGE_AUDIT_2026-04-11.md) — definitive strict audit, file-path mapping, and N/A rationale
- [`LONG_CONTEXT_TRUTH_MATRIX.md`](LONG_CONTEXT_TRUTH_MATRIX.md) — canonical claim boundary
- [`SUPPORT_MATRIX.md`](SUPPORT_MATRIX.md) — lane-by-lane audited support table
- [`PRODUCER_GAPS.md`](PRODUCER_GAPS.md) — what remains truly open vs no longer applicable
- [`RUNBOOK_EXTERNAL_GMI.md`](RUNBOOK_EXTERNAL_GMI.md) — external operator path
- [`RUNBOOK_INTERNAL_SEMIANALYSIS.md`](RUNBOOK_INTERNAL_SEMIANALYSIS.md) — internal workflow-backed path
- [`INVESTIGATION_KV_CACHE_PROFILING_2026-04-11.md`](INVESTIGATION_KV_CACHE_PROFILING_2026-04-11.md) — what the long-context preview paths actually measure

---

## Export roots

- `datasets/isb1/exports/core/`
- `datasets/isb1/exports/extension_32k/`
- `datasets/isb1/exports/extension_64k/`
- `datasets/isb1/exports/extension_131k/`
- `datasets/isb1/exports/preview/offload_core/`
- `datasets/isb1/exports/preview/long_context_500k/`
- `datasets/isb1/exports/preview/long_context_1m/`

