# HF publication recipe for ISB1 converted traces

Mirror `datasets/isb1/converted/` to Hugging Face so Cam's
`TRACE_DIR=hf_<org>--<repo>` path works immediately with kv-cache-tester.
Recommended target: `semianalysisai/isb1-cc-traces`.

## 1. Target namespace

- Dataset repo: `semianalysisai/isb1-cc-traces`
- Source directory: `datasets/isb1/converted/`
- Consumer contract: Cam's replay scripts interpret `hf_<org>--<repo>` as a
  Hugging Face dataset reference before calling `trace_replay_tester.py`

## 2. Prereqs

- `huggingface-cli >= 0.20`
- `HF_TOKEN` with write scope to the destination org
- Local validation already green:
  `python3 tools/validate_kvcache_tester_trace.py datasets/isb1/converted/`

Authenticate first:

```bash
export HF_TOKEN=hf_xxx
huggingface-cli login --token "$HF_TOKEN"
```

## 3. Dataset card template

Create the HF dataset `README.md` with this content:

```markdown
---
license: apache-2.0
task_categories: [text-generation]
language: [en]
pretty_name: ISB1 Converted kv-cache-tester Traces
tags: [kv-cache, trace-replay, inference-benchmark, semianalysis, isb1]
---

# ISB1 Converted kv-cache-tester Traces

This dataset mirrors `datasets/isb1/converted/` from SemiAnalysisAI/InferenceX
PR #1032 so Cam's kv-cache-tester replay flow from PR #993 can consume ISB1
traces directly through the `hf_<org>--<repo>` `TRACE_DIR` convention.

## Contents

- 179 pre-converted trace JSON files
- 8k / 32k / 64k / 131k / 500k preview / 1m preview coverage
- Kimi K2.5 / DSR1 / GPT-OSS / Qwen3.5 coverage
- `manifest.json` metadata catalog

## Provenance

- Source repo: `SemiAnalysisAI/InferenceX`
- Source PR: `#1032`
- Consumer workflow: `callanjfox/kv-cache-tester` PR `#993`
- License: Apache-2.0
```

## 4. Upload command

```bash
huggingface-cli upload \
  semianalysisai/isb1-cc-traces \
  datasets/isb1/converted/ \
  . \
  --repo-type dataset \
  --revision main
```

If the repo does not exist yet, create it in the HF UI first, then rerun the
upload.

## 5. Cam's Slurm integration

After publication, switch Cam's script from a local directory to the HF path:

```bash
TRACE_DIR=hf_semianalysisai--isb1-cc-traces  # replaces datasets/isb1/converted
```

That triggers the `hf_<org>--<repo>` branch in Cam's PR #993 replay script
(`benchmarks/single_node/multiturn_fp4_b200_trace_replay.sh`, lines 54-58),
which rewrites the value into `--hf-dataset <org>/<repo>` before invoking
`trace_replay_tester.py`.

## 6. Versioning

When new traces land:

1. Regenerate `datasets/isb1/converted/manifest.json`
2. Re-run local validation on the converted directory
3. Upload the updated directory to HF `main`
4. Create a matching HF tag such as `v0.2.0` or `pr1032-r2`
5. Record the InferenceX commit SHA and HF revision together

Consumers who need immutability should pin an HF revision instead of floating
on `main`.

## 7. Verification

```bash
rm -rf /tmp/verify
huggingface-cli download semianalysisai/isb1-cc-traces \
  --repo-type dataset \
  --local-dir /tmp/verify
python3 tools/validate_kvcache_tester_trace.py /tmp/verify
```

Expected result:

- Download succeeds with all trace JSONs present
- Validator reports all converted traces passing
- Cam's replay wrapper accepts
  `TRACE_DIR=hf_semianalysisai--isb1-cc-traces` with no shell-script changes

## Notes

- Publish converted artifacts and metadata only
- Keep the layout compatible with `trace_replay_tester.py`
- If the org name changes, update both the upload command and `TRACE_DIR`
  example together
