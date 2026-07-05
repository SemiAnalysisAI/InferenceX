---
description: Bump single-node inference-engine image tags (vLLM or SGLang) across recipes, one [Klaud Cold] PR per model+precision+SKU
argument-hint: <vllm|sglang> <target-tag> [model/sku filter]
---

Bump the container image tag for single-node benchmark recipes that use a given
inference engine, opening **one PR per recipe family** with the grouping rules below.

Arguments (`$ARGUMENTS`): `<engine> <target-tag> [filter]`
- `engine` — `vllm` or `sglang`
- `target-tag` — e.g. `v0.22.0` (NVIDIA/CUDA) ; for SGLang the NVIDIA and AMD tag
  strings usually differ (CUDA `…-cu130` vs ROCm `…-rocm720-mi35x-…`), so confirm
  the exact tag per image repo with the user before editing.
- `filter` (optional) — restrict to a model and/or SKU substring (e.g. `kimik2.5`,
  `b300`, `minimaxm2.5 mi355x`). If omitted, all matching recipes are in scope.

## Image repos by engine + vendor

| engine | NVIDIA image | AMD/ROCm image | master config |
|--------|--------------|----------------|---------------|
| vllm   | `vllm/vllm-openai` | `vllm/vllm-openai-rocm` | `configs/nvidia-master.yaml` / `amd-master.yaml` |
| sglang | `lmsysorg/sglang`  | `lmsysorg/sglang` (rocm-suffixed tag) | same two files |

## Grouping rules (NON-NEGOTIABLE)

1. **One PR per `model + precision + SKU` recipe family.** The config-key shape is
   `<model>-<precision>-<sku>-<engine>` (e.g. `kimik2.5-int4-b300-vllm`).
2. **Fold the `-mtp` (and non-mtp) sibling into the SAME PR** as its base recipe.
   This is the *only* thing you may combine.
3. **Never** put two different models, two different precisions, or two different
   SKUs in the same PR. (fp4 vs fp8 vs int4 are different precisions → separate PRs.)
4. Skip `*-agentic` recipes unless the user explicitly opts in — they are
   deliberately diverged/pinned.

## Step 1 — discover candidate recipes

Parse both master YAMLs for top-level keys whose `framework:` matches `engine`, and
record each key's current `image:`. Keep only single-node keys (they carry a SKU like
`b200/b300/h100/h200/mi300x/mi325x/mi355x` and map to `benchmarks/single_node/*`); drop
multi-node/disagg keys. Apply the `filter` if given. Then collapse `-mtp` siblings into
their base family.

## Step 2 — verify the target tag(s) EXIST before bumping

Per standing guidance, never invent a tag. Check each image repo you'll touch:

```bash
for repo in vllm/vllm-openai vllm/vllm-openai-rocm; do   # or lmsysorg/sglang
  code=$(curl -s -o /dev/null -w "%{http_code}" "https://hub.docker.com/v2/repositories/${repo}/tags/<TAG>")
  echo "$repo:<TAG> -> $code"   # want 200
done
```

If a vendor-specific variant 404s (e.g. `…-cu130` for a version that only ships
plain), confirm the correct tag string with the user before proceeding.

## Step 3 — confirm scope with the user (AskUserQuestion)

Before creating anything, present the full recipe list (count + current→target per
family) and confirm:
- **Vendor scope**: NVIDIA, AMD, or both.
- **Agentic**: include `*-agentic` siblings? (default: exclude)
- **Special pins**: call out any recipe currently on a nightly/non-stable/special tag
  (e.g. `nightly-…`, `…-cu129`, a one-off build) and ask whether to override it.

Each PR triggers a full GPU sweep, so surface the total PR count explicitly.

## Step 4 — create one PR per family

Use these helpers (write them to /tmp) for precise, per-config-key edits — a blind
`sed` is unsafe because the same old tag appears under many keys.

`/tmp/edit_image.py`:
```python
#!/usr/bin/env python3
# Usage: edit_image.py <yaml_file> <new_image> <key1> [key2 ...]
import re, sys
f, new_image, keys = sys.argv[1], sys.argv[2], sys.argv[3:]
lines = open(f).read().split('\n')
for key in keys:
    kre = re.compile(r'^' + re.escape(key) + r':\s*$')
    start = next((i for i,l in enumerate(lines) if kre.match(l)), None)
    if start is None: sys.exit(f"ERROR: key not found: {key}")
    img_i = None
    for j in range(start+1, len(lines)):
        if re.match(r'^[A-Za-z0-9._-]+:\s*$', lines[j]): break  # next top-level key
        m = re.match(r'^(\s+)image:\s*(.+?)\s*$', lines[j])
        if m: img_i, indent, old = j, m.group(1), m.group(2); break
    if img_i is None: sys.exit(f"ERROR: no image: line for key {key}")
    if old != new_image: lines[img_i] = f"{indent}image: {new_image}"; print(f"{key}: {old} -> {new_image}")
    else: print(f"{key}: already {new_image} (no change)")
open(f,'w').write('\n'.join(lines))
```

`/tmp/append_changelog.py`:
```python
#!/usr/bin/env python3
# Usage: append_changelog.py <changelog> <description> <key1> [key2 ...]
import sys
f, desc, keys = sys.argv[1], sys.argv[2], sys.argv[3:]
content = open(f).read().rstrip('\n')
block = ["", "- config-keys:"] + [f"    - {k}" for k in keys]
block += ["  description:", f'    - "{desc}"', "  pr-link: PRLINK_PLACEHOLDER"]
open(f,'w').write(content + '\n' + '\n'.join(block) + '\n')
```

For each family (run strictly sequentially — git checkouts can't be parallel):

```bash
git checkout main -q && git reset --hard origin/main -q
branch="klaud-cold/<basekey>-<TAG>"
git checkout -b "$branch" -q
python3 /tmp/edit_image.py <master.yaml> <NEW_IMAGE> <key> [<key>-mtp]
python3 /tmp/append_changelog.py perf-changelog.yaml "<DESC>" <key> [<key>-mtp]
git add -A
git commit -q -m "[Klaud Cold] Update <basekey>[ (+mtp)] <PHRASE> to <TAG>" \
  -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git push -u origin "$branch" -q --force-with-lease
url=$(gh pr create --repo SemiAnalysisAI/InferenceX --base main --head "$branch" \
      --title "[Klaud Cold] Update <basekey>[ (+mtp)] <PHRASE> to <TAG>" \
      --body "<BODY>" --label full-sweep-fail-fast | grep -o 'https://github.com/[^ ]*')
# patch the changelog pr-link with the real URL, then amend + force-push
python3 - perf-changelog.yaml "$url" <<'PY'
import sys; f,u=sys.argv[1],sys.argv[2]
open(f,'w').write(open(f).read().replace("PRLINK_PLACEHOLDER",u,1))
PY
git add perf-changelog.yaml && git commit -q --amend --no-edit && git push -q --force-with-lease
```

Conventions:
- `<PHRASE>` = `vLLM image` / `vLLM ROCm image` / `SGLang image` / `SGLang ROCm image`.
- Title gets `(+mtp)` only when the family has an mtp sibling.
- Every PR carries the **`full-sweep-fail-fast`** label (strongly recommended over `full-sweep-enabled` - a broken image bump burns one job per matrix, not the full fan-out) so CI kicks off.
- `<DESC>` = `Update <PHRASE> from <old-tag> to <TAG>` (note both tags when the
  base/mtp differ, e.g. base already on target).
- PR body:
  ```
  ## Summary
  <DESC>

  Recipes touched: `key1`, `key2`

  ## Test plan
  - [ ] full-sweep-fail-fast sweep passes.

  🤖 Generated with [Claude Code](https://claude.com/claude-code)
  ```

## Step 5 — finish

Return to a clean `main` (`git checkout main && git reset --hard origin/main`).
Report a table of every PR created (number + URL + recipe), flag any special-pin
overrides, and note that each PR's sweep will run via the `full-sweep-fail-fast` label.
