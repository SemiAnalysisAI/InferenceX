---
description: Scan /it-share/gharunners* on the AMD MI355X TW cluster for root-owned files stranded in runner workspaces and delete them via the sudo hop, then rerun the failed sweeps
argument-hint: [failed-run-url ...]
---

Fix `EACCES: permission denied` workspace-cleanup failures on the AMD MI355X TW
GitHub Actions runners (`mi355x-amds_*` / `gharunnerNN`). Symptom in the job log:

```
Deleting the contents of '/it-share/gharunners2/gharunnerNN/actions-runner/_work/InferenceX/InferenceX'
Error: File was unable to be removed Error: EACCES: permission denied, rmdir '.../benchmark_logs/logs/slurm_job-<id>'
```

Root cause: multi-node disagg benchmarks submit slurm jobs whose containers
write logs as **root** into the runner workspace (`benchmark_logs/logs/slurm_job-*`).
Normal teardown cleans them up, but a **cancelled** job skips teardown and
strands root-owned dirs that the runner user cannot delete, breaking workspace
cleanup for every subsequent job on that runner.

## Access path

The jumpbox has **no sudo**; the hop host has **passwordless sudo** on the
shared `/it-share`:

```bash
ssh -A -o BatchMode=yes amd-tw-mi355 "ssh -o BatchMode=yes mia1-vm-amd-prj3-slog-001 '<command>'"
```

## Procedure

1. **Read-only scan first — list every match, do not delete yet.** Scope the
   scan to the runner `_work` workspaces only:

   ```bash
   ssh -A -o BatchMode=yes amd-tw-mi355 "ssh -o BatchMode=yes mia1-vm-amd-prj3-slog-001 \
     'sudo find /it-share/gharunners*/gharunner*/actions-runner/_work -user root 2>/dev/null'"
   ```

2. **Review the full list before deleting.** Every path must be under an
   `actions-runner/_work/` workspace (typically `.../InferenceX/InferenceX/benchmark_logs/...`).
   If anything outside `_work` shows up, STOP and report instead of deleting.

3. **Delete the verified matches** (scoped to the same `_work` glob — never
   `rm -rf` an unscoped `/it-share` path):

   ```bash
   ssh -A -o BatchMode=yes amd-tw-mi355 "ssh -o BatchMode=yes mia1-vm-amd-prj3-slog-001 \
     'sudo find /it-share/gharunners*/gharunner*/actions-runner/_work -user root -print0 2>/dev/null \
      | xargs -0 -r sudo rm -rf'"
   ```

4. **Verify** the same scan now returns zero entries.

5. **Rerun the affected sweeps.** For each failed run in `$ARGUMENTS` (or found
   via `gh run list`): try `gh run rerun <id> --failed`; if GitHub refuses a
   partial rerun (common after cancellation), use a full `gh run rerun <id>`;
   as a last resort remove and re-add the PR's sweep label to force a fresh run.

6. Optional forensics — identify what stranded the files: the dir name
   `slurm_job-<id>` maps to slurm accounting (`sacct -j <id>` on the jumpbox
   shows start/end/state; CANCELLED means teardown was skipped), and the GitHub
   job that submitted it is whichever job ran on that runner at the slurm start
   time (`gh api .../actions/runs/<run>/jobs` → match `runner_name` and
   `started_at` within seconds).
