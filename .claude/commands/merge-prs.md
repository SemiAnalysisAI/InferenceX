---
description: Sequentially run utils/merge_with_reuse.sh for each PR number
argument-hint: <pr-number> [<pr-number>...]
---

First, ensure local main is up to date:

```bash
git checkout main && git pull origin main
```

Then run `utils/merge_with_reuse.sh <pr>` once for each PR number in: $ARGUMENTS

Run them strictly sequentially — the script does git checkouts, so they cannot run in parallel. Stop on the first failure and report which PRs were merged (with merge SHA) and which one failed.
