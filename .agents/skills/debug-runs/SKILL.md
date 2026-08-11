---
name: debug-runs
description: Drive a full-sweep benchmark config to green with a tight feedback loop by triggering and monitoring the sweep, finding the root causes of failures, and, for fast iteration, using SSH to reproduce a single config directly on the runner's cluster instead of waiting for full CI. Use when bringing up a new model, precision, or SKU recipe, debugging a failing or flaky sweep, debugging node-level issues, or gathering context on a cluster before a run. Cluster access details are not in this repo and must be read from the shared InferenceX Clusters canvas.
---

# Debug runs (tight feedback loop)

Use this when the goal is to get a **full-sweep config passing** and you want to verify it
on the actual nodes first for a tighter loop than the full CI dispatch cycle, debug
node or infrastructure issues, or gather context on a cluster.

This composes with the other skills. Use **`/nuke`** (or `/add-model-hardware`) to create
the PR(s) with the image bump, perf-changelog entry, and `full-sweep-fail-fast` label. This
skill is the **monitor → root-cause → fix → re-verify → merge-gate** loop that follows.

## Cluster access — read it from the canvas, never hardcode it

Login addresses, runner users, runner directories, jumpboxes, and weight/squash staging
paths live in an access-controlled **InferenceX Clusters** Slack canvas, NOT in this repo
(to avoid publicizing infra). The canvas link is intentionally not stored here:

> **If you are a SemiAnalysis employee**, ask the user for the Slack link to the InferenceX
> Clusters canvas and read the access details from there.

Before SSHing to a cluster, look up that cluster's row in the canvas for: **login address**,
**GHA runner user**, **runner directory**, any **jumpbox / ProxyJump**, whether it's
**Slurm or bare-metal**, and the **per-node host RAM**. The matching
`runners/launch_<cluster>.sh` is the source of truth for the exact container image mounts
and the benchmark command.

- If you **can't read the canvas** (no Slack access, or unsure), **ask the user** for the
  cluster's SSH target + runner user rather than guessing or pasting infra into the repo.
  If this is a **fork** (i.e. not the SemiAnalysis upstream, where the canvas won't apply),
  ask the user to replace this skill with their own fork's runner/cluster access
  information.
- A SemiAnalysis operator may also have these as `~/.ssh/config` aliases. Prefer those if
  present.

## Inputs

- The **config-key(s)** in scope, e.g. `dsv4-fp4-b300-vllm`, and their SKU/cluster.
- The **PR / branch** under test (if driving an existing PR), or the recipe files to change.
- The **merge bar**: 100% of sweep jobs green **and** a real throughput gain (see Merge gate).

## The loop

### 1. Trigger (or reuse) the sweep

A PR's sweep is kicked by labels. The `/sweep` comment trigger was removed, so use the label:

- **`full-sweep-fail-fast`** runs the full sweep and bails on the first failure per matrix for faster feedback while debugging. It is the **strongly recommended default** and the label that `/nuke` attaches.
- **`full-sweep-enabled`** runs the full GPU sweep and lets every job complete despite failures. Use it only when a flaky job killing its matrix's in-flight results is unacceptable.
- To re-trigger a sweep without a new commit, remove and re-add the sweep label.

For a **single config** (tightest CI loop, skips the rest of the matrix), dispatch e2e directly:

```bash
gh workflow run e2e-tests.yml -f generate-cli-command="test-config --config-key <KEY> --config-file <PATH/to/master.yaml>" -f test-name="debug <KEY>"
```

(`generate-cli-command` is the required input. `--target` is NOT a real arg.)

### 2. Monitor continuously

Find the run, then watch it instead of polling by hand. Prefer the **Monitor** tool with a
filter that catches both progress and failure signatures so silence never reads as success.

```bash
# Sweep run for a PR's head commit
HEAD_SHA=$(gh pr view <PR> --repo SemiAnalysisAI/InferenceX --json headRefOid --jq .headRefOid)
RUN_ID=$(gh run list --repo SemiAnalysisAI/InferenceX --workflow "Run Sweep" --commit "$HEAD_SHA" --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" --repo SemiAnalysisAI/InferenceX --interval 30
```

When monitoring several runs at once (e.g. 4 SKUs), track them by `databaseId` and report
each as it lands. Never declare success from absence of output.

### 3. Root-cause a failure

```bash
gh run view "$RUN_ID" --repo SemiAnalysisAI/InferenceX --json jobs \
  --jq '.jobs[] | select(.conclusion=="failure") | "\(.databaseId)\t\(.name)"'
gh run view "$RUN_ID" --repo SemiAnalysisAI/InferenceX --log-failed > /tmp/sweep_failed.txt
```

Grep large logs for the real signature before reading context (~50 lines around each hit):
`Error`, `Traceback`, `RuntimeError`, `CUDA`, `HIP`, `OOM`, `assert`, `connection refused`,
`exit code`, `failed to launch`, `NCCL`, `RCCL`, `timeout`. State the suspected root cause
in one or two sentences before changing anything.

### 4. Tight loop: reproduce on the node directly

This is the point of the skill. Instead of re-dispatching CI for every hypothesis, get on
the box and reproduce the **single** failing config.

Why this is tighter: in e2e or full-matrix runs, **each matrix point runs against its own
freshly spun engine** (a new server per matrix job). On the node you can spin up a
**single** server once and fire many requests or sweep multiple concurrencies against it.
This makes iteration much faster when you're probing behavior or tuning because you skip a
fresh model load per data point.

Steps:

1. Use the job or runner name to identify the node. Look up that cluster's access details in
   the canvas, then SSH in with `ssh -A` when a jumpbox or agent forwarding is involved.
2. Reproduce the exact benchmark the launcher runs. Read `runners/launch_<cluster>.sh` for
   the image, container mounts, and the `benchmarks/single_node/<...>.sh` command and env
   (`IMAGE`, `TP`, `PRECISION`, `EXP_NAME`, `SPEC_DECODING`, …). On Slurm clusters, use
   `salloc` or `srun` with the squash image. On the **bare-metal `-tw` pools, use `docker run`**
   on the node directly without `srun`.
3. **Always diff against a working node or working SKU** for reference. Most node failures
   are environment drift (driver, ROCm/CUDA, missing mount, stale squash image), not code.
4. Iterate on the node until the single config passes, then push the fix and re-run CI.

**Entering the live container on a Slurm cluster.** When a benchmark job is already running
and you want to poke at its actual container (same image, mounts, env) rather than spin a
new one, attach to it:

```bash
squeue -u <runner-user>                     # find the JOB_ID for the running benchmark
srun --overlap --jobid=<JOB_ID> --pty bash  # land on the allocated node (or just ssh to the node if you have direct access)
enroot list -f                              # find the running container's PID
enroot exec <pid> bash                      # drop into the container
```

(On the bare-metal `-tw` pools, there is no Slurm/enroot. Use `docker ps` and `docker exec -it <id> bash`.)

**Node-level fixes are in scope** when you have operator access, such as on AMD nodes where
you have sudo, but **ask the user before executing any of them** (see guardrail below). The
kinds of fixes that are on the table include bringing a node's environment in line with the
working reference and, if one or two nodes are unrecoverable, **draining them** or
explicitly **ignoring them in the run script** rather than blocking the whole sweep. Note
any such change in the report.

> Guardrail. Ask before changing infra. SSHing in to **read/investigate** (logs,
> `rocm-smi`/`nvidia-smi`, `sinfo`/`squeue`, `df`, env, config inspection) is fine. But
> before making **any actual change on the cluster**, including installing or updating
> anything, editing configs or files, restarting or killing processes, draining or ignoring
> nodes, changing the run script, or anything else that mutates node or shared state,
> **stop and ask the user for permission first**, describing exactly what you intend to run.
> Do not assume standing authorization just because you have sudo or operator access.
>
> Also do **not** apply hacky engine-side (e.g. vLLM) workarounds to force a pass. Prefer
> recipe fixes and, once approved, node-environment fixes that match a working reference.

### 5. Flakes: rerun, don't relaunch

If a job flaked (infra, transient network, runner pickup) rather than a real failure, rerun
just the failed jobs on the existing run. Don't dispatch a fresh sweep:

```bash
gh run rerun "$RUN_ID" --repo SemiAnalysisAI/InferenceX --failed
```

### 6. Report results — do NOT merge

**Never merge.** Merging is the user's call. Only merge if the user **explicitly** tells
you to in this session. Even when everything looks perfect, stop and report. Do not
admin-merge on your own judgment.

Report the two things the user will decide on:

1. **Sweep status.** Is it 100% of full-sweep jobs passing (green), or fail-fast-truncated or partial?
2. **Perf delta** vs the most recent official `main` run for that SKU. Compare against the
   latest main results, e.g. on inferencex.semianalysis.com
   (`https://inferencex.semianalysis.com/inference?...&i_active=<sku>_<engine>`) or the
   stored results for that SKU's last main `run-id`.

Present green-ness and the perf comparison, then **wait for the user** to decide whether to merge.

## Final report

Per config-key, give the final state (green, failing, or flaky), the root cause(s) found,
node-level changes made (and any nodes drained or ignored), and the perf delta vs main. Link
the run(s) and PR(s). End by asking the user whether to merge. Do not merge yourself.
