# HF publication recipe for ISB1 converted traces

Mirror `datasets/isb1/converted/` to Hugging Face so Cam's
`TRACE_DIR=hf_<org>--<repo>` path works immediately with kv-cache-tester.
Preferred target: `semianalysisai/isb1-cc-traces`.
Fallback if org write access is unavailable: `ocwc22/isb1-cc-traces`.

## 1. What gets published

This publish package is the checked-in trio below:

- `datasets/isb1/converted/` — 179 validated kv-cache-tester trace JSON files
- `datasets/isb1/converted/manifest.json` — corpus metadata (`1226` total requests)
- `datasets/isb1/hf_dataset_card.md` — staged to HF as `README.md`

The consumer contract is unchanged: Cam's replay scripts interpret
`hf_<org>--<repo>` as a Hugging Face dataset source, hydrate it locally, and
then invoke the existing replay path.

## 2. Pre-flight validation

Run the stdlib validator before every publish attempt:

```bash
python3 tools/validate_kvcache_tester_trace.py datasets/isb1/converted/
```

Expected result:

- `✓ 179 files valid | 0 failed`
- Exit code `0`

If validation fails, stop and fix the source corpus before publishing. Do not
push a broken dataset mirror to HF.

## 3. Python version

`tools/publish_hf_dataset.py` imports `huggingface_hub >= 0.24`, which in turn
requires Python 3.10+. On macOS the system `/usr/bin/python3` is 3.9 and does
not ship `huggingface_hub`; do not use it.

Use Python 3.13 explicitly:

```bash
/opt/homebrew/opt/python@3.13/bin/python3.13 -m pip install --user huggingface_hub
/opt/homebrew/opt/python@3.13/bin/python3.13 tools/publish_hf_dataset.py --help
```

Or activate a virtualenv / pyenv shim that resolves to 3.10+ before running any
of the commands below. If you see `ModuleNotFoundError: huggingface_hub`, you
are on 3.9 — switch interpreters first.

## 4. Token setup

Authenticate with a token that has write access to the destination namespace:

```bash
huggingface-cli login
```

If you prefer explicit token injection:

```bash
export HF_TOKEN=hf_xxx
huggingface-cli login --token "$HF_TOKEN"
```

## 5. Dry-run the publish package locally

The uploader script stages the converted corpus plus the dataset card and
prints the exact file list it would upload without making any remote changes.

```bash
python3 tools/publish_hf_dataset.py \
  --source datasets/isb1/converted/ \
  --repo semianalysisai/isb1-cc-traces \
  --private \
  --dry-run
```

Use the fallback namespace instead if needed:

```bash
python3 tools/publish_hf_dataset.py \
  --source datasets/isb1/converted/ \
  --repo ocwc22/isb1-cc-traces \
  --private \
  --dry-run
```

## 6. Publish for real

Once the dry-run output looks correct and HF auth is configured, publish with
one of the exact commands below.

Private-first publish:

```bash
python3 tools/publish_hf_dataset.py \
  --source datasets/isb1/converted/ \
  --repo semianalysisai/isb1-cc-traces \
  --private \
  --commit-message "Publish ISB-1 kv-cache-tester traces"
```

Or make the dataset public at creation time:

```bash
python3 tools/publish_hf_dataset.py \
  --source datasets/isb1/converted/ \
  --repo semianalysisai/isb1-cc-traces \
  --public \
  --commit-message "Publish ISB-1 kv-cache-tester traces"
```

Fallback org:

```bash
python3 tools/publish_hf_dataset.py \
  --source datasets/isb1/converted/ \
  --repo ocwc22/isb1-cc-traces \
  --public \
  --commit-message "Publish ISB-1 kv-cache-tester traces"
```

The script will:

1. Stage `datasets/isb1/converted/` into a temporary upload tree
2. Copy `datasets/isb1/hf_dataset_card.md` into that tree as `README.md`
3. Create the dataset repo if it does not already exist
4. Upload the staged folder with `huggingface_hub`
5. Verify the published snapshot with `snapshot_download` into `/tmp`

## 7. Post-publish verification

### Repository-level verification

Re-download the published dataset and re-run the validator against the hydrated
copy:

```bash
huggingface-cli download semianalysisai/isb1-cc-traces \
  --repo-type dataset \
  --local-dir /tmp/isb1-cc-traces-verify
python3 tools/validate_kvcache_tester_trace.py /tmp/isb1-cc-traces-verify
```

### Harness-level verification

The exact consumer path for Cam is the existing `TRACE_DIR=hf_<org>--<repo>`
contract. In the replay harness checkout, the closest end-to-end verification
command is:

```bash
TRACE_DIR=hf_semianalysisai--isb1-cc-traces \
bash experimental/multiturn/benchmarks/single_node/multiturn_fp8_h200_trace_replay.sh
```

If the SemianalysisAI org is not available, swap in the fallback namespace:

```bash
TRACE_DIR=hf_ocwc22--isb1-cc-traces \
bash experimental/multiturn/benchmarks/single_node/multiturn_fp8_h200_trace_replay.sh
```

## 8. Consumer note for Cam

This is the zero-friction handoff:

```bash
TRACE_DIR=hf_semianalysisai--isb1-cc-traces \
bash experimental/multiturn/benchmarks/single_node/multiturn_fp8_h200_trace_replay.sh
```

No code change is required in Cam's harness. The only user action is publishing
this dataset repo once with valid HF credentials.

## 9. Versioning guidance

When new traces land:

1. Regenerate `datasets/isb1/converted/manifest.json`
2. Re-run `tools/validate_kvcache_tester_trace.py`
3. Re-run the uploader dry-run
4. Publish with a commit message that records the corpus revision
5. Record the InferenceX commit SHA and the HF dataset revision together

Consumers that need immutability should pin an HF revision instead of floating
on `main`.

## Notes

- Publish converted artifacts and metadata only
- Do not modify `datasets/isb1/converted/**` during publication prep
- Keep the uploaded layout compatible with kv-cache-tester's existing
  `TRACE_DIR=hf_<org>--<repo>` convention
