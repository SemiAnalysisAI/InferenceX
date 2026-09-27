---
description: Sequentially run uv run --extra workflows python -m infx.workflows.merge_with_reuse for each PR number
argument-hint: <pr-number> [<pr-number>...]
---

First, ensure local main is up to date:

```bash
git checkout main && git pull origin main
```

Then run `uv run --extra workflows python -m infx.workflows.merge_with_reuse <pr>` once for each PR number in: $ARGUMENTS

Run them strictly sequentially because the script does git checkouts, so they cannot run in parallel. Stop on the first failure and report which PRs were merged (with merge SHA) and which one failed.
