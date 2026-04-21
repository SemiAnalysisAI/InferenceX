# ISB1 replay artifacts for InferenceX

This directory is the InferenceX-side consumer package for ISB1 replay.

InferenceX consumes committed file artifacts only:
- replay export JSON bundles under `datasets/isb1/exports/`
- conversion to SemiAnalysis's `kv-cache-tester` format via
  [`tools/isb1_to_kvcache_tester.py`](../../tools/isb1_to_kvcache_tester.py)
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

All bundles can also be converted to SemiAnalysis's `kv-cache-tester`
per-conversation trace format via [`tools/isb1_to_kvcache_tester.py`](../../tools/isb1_to_kvcache_tester.py);
see [How to consume](#how-to-consume).

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

## How to consume

Two consumption paths are supported, both fed from the same committed bundles:

1. **InferenceX-internal replay** — `utils/bench_serving/benchmark_export_replay.py`
   directly, with `utils/process_result_isb1.py` for result normalization.
2. **SemiAnalysis `kv-cache-tester`** — convert via
   [`tools/isb1_to_kvcache_tester.py`](../../tools/isb1_to_kvcache_tester.py),
   then feed the resulting directory to
   [`trace_replay_tester.py --trace-directory`](https://github.com/callanjfox/kv-cache-tester)
   (PR #993 submodule).

Path 2 is documented here because it is the path that lets ISB1 traces plug
into SemiAnalysis's existing Slurm benchmarking pipeline without the consumer
having to read our replay code.

### Step 1 — fetch the bundles (LFS)

From a clone of InferenceX:

```bash
git lfs install
git lfs pull --include='datasets/isb1/exports/**/*.json'
```

Or, if the bundles are published to Hugging Face (see
[HF publication](#hf-publication)), download directly:

```bash
huggingface-cli download <org>/<repo> \
    --repo-type dataset \
    --local-dir ./isb1_bundles
```

### Step 2 — convert one bundle to `kv-cache-tester` format

```bash
python tools/isb1_to_kvcache_tester.py \
    --export-file datasets/isb1/exports/core/chat_8k1k_qwen3.5.json \
    --output-dir  traces_isb1/core_chat_qwen/ \
    --runtime-stack-id standalone:vllm \
    --hardware-profile-id nvidia:h200_sxm_141gb \
    --canonical-model-id qwen3_5_397b_a17b \
    --support-status supported
```

This writes one `trace_<conversation_id>.json` per cell that passes the
filters, in the flat layout `trace_replay_tester.py --trace-directory`
expects. The schema matches `kv-cache-tester@main` (`trace.id`, `requests[].t/in/out/hash_ids`),
so `normalize_trace()` accepts it as-is.

To convert every bundle in one shot:

```bash
python tools/isb1_to_kvcache_tester.py \
    --export-root datasets/isb1/exports/ \
    --output-dir  traces_isb1/
```

### Step 3 — replay against a running vLLM / SGLang server

Using PR #993's own recipes (e.g. `benchmarks/single_node/multiturn_fp8_h200_trace_replay.sh`),
set `TRACE_DIR` to the converted directory and let the existing Slurm wiring
pick it up:

```bash
TRACE_DIR=$PWD/traces_isb1/core_chat_qwen \
MODEL=Qwen/Qwen3.5-397B-A17B-FP8 \
TP=8 USERS=8 OFFLOAD_MODE=off TOTAL_CPU_DRAM_GB=0 \
RESULT_DIR=$PWD/results/isb1_smoke \
bash experimental/multiturn/benchmarks/single_node/multiturn_fp8_h200_trace_replay.sh
```

Or, equivalently, invoke Cam's tester directly:

```bash
python $KV_CACHE_TESTER_DIR/trace_replay_tester.py \
    --api-endpoint http://127.0.0.1:8888 \
    --trace-directory $PWD/traces_isb1/core_chat_qwen \
    --output-dir      $PWD/results/isb1_smoke \
    --start-users 2 --max-users 2 --test-duration 60
```

### Step 4 — verify the result

```bash
jq '.' results/isb1_smoke/results.json | head
```

Expected:

- `trace_replay_tester.py` logs show `Loaded N traces (filtered from N)`.
- Cache-hit rate reported during the run is non-zero for multi-turn bundles
  (because each turn's `hash_ids` extends the previous turn's prefix).
- Completed sessions ≥ 1; HTTP error count = 0.

---

## Smoke test

The one-liner below is the binary go/no-go for "does this PR actually help
SemiAnalysis":

```bash
# Assumes a vLLM OpenAI server is up on :8888 serving the model.
python tools/isb1_to_kvcache_tester.py \
    --export-file datasets/isb1/exports/core/chat_8k1k_qwen3.5.json \
    --output-dir  /tmp/isb1_proof/ \
    --canonical-model-id qwen3_5_397b_a17b \
&& python $KV_CACHE_TESTER_DIR/trace_replay_tester.py \
    --api-endpoint http://127.0.0.1:8888 \
    --trace-directory /tmp/isb1_proof/ \
    --output-dir      /tmp/isb1_proof/out \
    --start-users 1 --max-users 1 --test-duration 30
```

Pass criteria:

| Artifact | Threshold |
|---|---|
| Shim exit code | `0` |
| `trace_replay_tester.py` exit code | `0` |
| `Loaded N traces` (N) | `≥ 1` |
| Completed sessions | `≥ 1` |
| HTTP errors | `0` |

Any failure of the above means the PR is not actually plumbed end-to-end for
this bundle and should be reproduced against Cam's `trace_replay_tester.py`
before being claimed as compatible.

---

## HF publication

The `kv-cache-tester` Slurm recipes accept an HF dataset source via the
`hf_<org>--<repo>` prefix convention on `TRACE_DIR` — the wrapper `.sh`
scripts download with `huggingface-cli` and point the tester at the local
mirror.

To publish an HF mirror of these bundles:

1. Create a dataset repo (e.g. `semianalysisai/isb1-core-v0`).
2. Mirror the directory layout of `datasets/isb1/exports/` exactly.
   (Do not copy the inner `datasets/isb1/.gitattributes` — one top-level
   LFS-only `.gitattributes` at the HF repo root is sufficient.)
3. For each published bundle, run
   `tools/isb1_to_kvcache_tester.py --export-root <downloaded dir> --output-dir <out>`
   locally to verify the conversion stays green at the new revision.
4. Pin revisions by HF branch/tag matching the producer's
   `schema_version` (e.g. `v0.2.0`).

Once published, Cam's Slurm scripts can consume a bundle with no code change:

```bash
TRACE_DIR=hf_semianalysisai--isb1-core-v0 \
bash experimental/multiturn/benchmarks/single_node/multiturn_fp8_h200_trace_replay.sh
```

(Where the `.sh` does `huggingface-cli download semianalysisai/isb1-core-v0`
into a scratch dir, then runs the converter shim against it before
invoking `trace_replay_tester.py`.)

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
