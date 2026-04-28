---
pretty_name: ISB-1 KV-Cache-Tester Traces
tags:
  - kv-cache
  - inference-benchmarking
  - vllm
  - long-context
  - multi-turn
license: apache-2.0
task_categories:
  - text-generation
size_categories:
  - 1M<n<10M
configs:
  - config_name: core
    default: true
    data_files: "core/**/*.json"
  - config_name: extension_32k
    data_files: "extension_32k/**/*.json"
  - config_name: extension_64k
    data_files: "extension_64k/**/*.json"
  - config_name: extension_131k
    data_files: "extension_131k/**/*.json"
  - config_name: preview_500k
    data_files: "preview/long_context_500k/**/*.json"
  - config_name: preview_1m
    data_files: "preview/long_context_1m/**/*.json"
---

# ISB-1 KV-Cache-Tester Traces

## Overview

This dataset is a Hugging Face mirror of `datasets/isb1/converted/` from
`SemiAnalysisAI/InferenceX`. It packages the pre-converted ISB-1 multi-turn
trace corpus in the JSON schema already consumed by the kv-cache-tester replay
path, so downstream users can switch from a local directory to
`TRACE_DIR=hf_<org>--<repo>` with no harness changes.

Current corpus summary:

- 179 validated trace JSON files
- 1,226 total requests across the corpus
- ~9.26 MiB on disk including `manifest.json`
- Context bands spanning 8k, 32k, 64k, 131k, 500k preview, and 1m preview
- Model coverage including DeepSeek-R1-0528, GPT-OSS-120B, GLM-5,
  MiniMax-M2.5, and Qwen3.5-397B-A17B variants present in the source corpus

Preferred dataset name:

- `semianalysisai/isb1-cc-traces`

Fallback if SemiAnalysisAI org write access is unavailable:

- `ocwc22/isb1-cc-traces`

The intended consumer is Cam's existing kv-cache-tester replay flow, which
already accepts `hf_<org>--<repo>` as a `TRACE_DIR` source and hydrates the
remote dataset locally before replay.

## Schema

Each JSON file is one replayable trace object with the following top-level
fields:

| Field | Type | Notes |
| --- | --- | --- |
| `id` | `str` | Stable trace/session identifier |
| `models` | `list[str]` | Model family identifiers for the trace |
| `block_size` | `int` | Hash-block size used for `hash_ids`; current corpus uses `64` |
| `hash_id_scope` | `str` | `local` or `global`; validated by the stdlib checker |
| `requests` | `list[object]` | Ordered replay requests |
| `tool_tokens` | `int` | Optional token accounting |
| `system_tokens` | `int` | Optional token accounting |
| `totals` / `isb1` | `object` | Source metadata carried through from conversion |

Each request entry includes the fields kv-cache-tester expects today:

- `type`
- `t`
- `in`
- `out`
- `hash_ids`
- optional `model`, `stop`, `input_types`, `output_types`, `api_time`,
  `think_time`

Before publishing or consuming the corpus, validate it with the bundled
stdlib-only checker:

```bash
python3 tools/validate_kvcache_tester_trace.py datasets/isb1/converted/
```

Exit codes:

- `0` = all trace files valid
- `1` = one or more trace files failed validation
- `2` = usage or path error

## How to use

For zero-friction consumption through the existing replay wrapper, point
`TRACE_DIR` at the HF dataset name in `hf_<org>--<repo>` form:

```bash
TRACE_DIR=hf_semianalysisai--isb1-cc-traces \
bash experimental/multiturn/benchmarks/single_node/multiturn_fp8_h200_trace_replay.sh
```

That is the whole integration contract: the wrapper handles the HF download and
passes the hydrated local mirror into the replay runner. If the preferred org
is unavailable, the fallback form is identical:

```bash
TRACE_DIR=hf_ocwc22--isb1-cc-traces \
bash experimental/multiturn/benchmarks/single_node/multiturn_fp8_h200_trace_replay.sh
```

For local inspection after download, the dataset also includes:

- `manifest.json` — corpus summary and per-trace metadata
- the original directory layout under `core/`, `extension_32k/`,
  `extension_64k/`, `extension_131k/`, and `preview/`

## License

This corpus follows the repository license in `SemiAnalysisAI/InferenceX`,
which is Apache-2.0.

## Citation

If you use this corpus in a benchmark, report, or derivative evaluation, cite
both the InferenceX repository and the dataset name/revision that you consumed.
A lightweight citation template is below.

```bibtex
@misc{isb1_kvcache_tester_traces_2026,
  title        = {ISB-1 KV-Cache-Tester Traces},
  author       = {SemiAnalysisAI},
  year         = {2026},
  howpublished = {Hugging Face dataset repository},
  note         = {Preferred repo: semianalysisai/isb1-cc-traces; fallback: ocwc22/isb1-cc-traces}
}
```
