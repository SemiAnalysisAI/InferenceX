---
name: live-logs
description: Open a live, per-node log viewer in the browser for a running multi-node sweep job, given a PR number, PR URL, workflow run URL or job URL. Resolves the GitHub job to its Slurm job on the cluster, then streams every worker (prefill / decode / agg), srtctl, Dynamo frontend, Mooncake / etcd service and aiperf log into resizable panes with phase tracking, error highlighting and node memory. Use when the user asks to watch, stream, tail or open the logs of a running sweep, canary or AgentX job.
---

# Live logs for a sweep job

GitHub Actions buffers multi-node job output until the job ends. This skill streams the
srt-slurm logs straight from the cluster to a local page instead.

## Run it

```bash
python3 .agents/skills/live-logs/scripts/live_logs.py <PR# | PR URL | run URL | job URL>
```

- **PR number or PR URL:** uses the in-progress `Run Sweep` run on the PR head. If none is running, it falls back to the latest one.
- **Run URL:** uses every in-progress job with a runner.
- **Job URL:** uses just that job.
- **Result:** one local viewer per Slurm job, for example the benchmark job and the eval job of the same run. Each opens in the browser, and each URL is printed.
- **Stop:** `live_logs.py --stop` stops every viewer the script started.
- **Faster start on very large logs:** pass `--history 4000` to fetch only the last 4000 lines of each file.
- **Requirements:** only the Python standard library, plus `gh` (authenticated) and `ssh` to the cluster login node. The viewer binds to `127.0.0.1` only.

Job resolution relies on srtctl naming the Slurm job after the GitHub runner (for example `b300-dsxe_01`). `squeue -n <runner>` finds the job, and `sacct` covers a job that has just finished. `scontrol show job` gives the WorkDir, and the logs are read from `<WorkDir>/outputs/<job>/logs`. Single-node jobs have no such Slurm job, and the script says so.

## Cluster login hosts: never put them in the repo

Login addresses are infra details that stay out of this repo; see `$debug-runs`. The script reads the host for each cluster (the runner-name prefix, for example `b300-dsxe`) from either:

- `~/.config/infx-live-logs/clusters.json`, for example `{"b300-dsxe": {"host": "<ssh alias or user@login>"}}`
- or `INFX_LIVE_LOGS_HOST_B300_DSXE=<ssh target>`, with the cluster name upper-cased and `-` turned into `_`.

If the host is missing, the script names the cluster and exits. Get the login address from the InferenceX Clusters canvas; ask the user for the link, or ask them for the SSH target. Add it to the local config, and never commit it.

## What the page shows

- **Header:**
  - SSH connection state and Slurm state (state, elapsed time, nodes).
  - Per-node free host memory and CPU load, refreshed every 20 s. The memory bar turns red below 200 GiB free, which is useful for Mooncake / EFA memory-registration OOMs.
- **Rows:**
  1. Engine workers (prefill / decode / agg).
  2. srtctl sweep, Dynamo frontend, Mooncake master, etcd.
  3. aiperf / benchmark logs. These appear on their own when the benchmark starts, because new files are discovered every 30 s.
- **Each pane header:**
  - The latest phase: weights loaded → KV cache sized → EFA devices up → Mooncake segment registration → autotune → ✅ healthy.
  - A count of real errors, excluding known noise such as NCCL `ibv_query_port_speed`, the pip resolver notice and the node-exporter TaskProlog message.
  - Buttons: jump to start / end, copy, and maximize (Esc restores).
- **Controls:**
  - Regex filter and "errors only".
  - "Hide Mooncake metrics noise", on by default. It hides the client metric report blocks, throughput and latency summaries, and dynamo HTTP 200 spam.
  - `show`: last 4000 lines (the default) or the entire log. The server keeps the whole file, so switching is instant.
  - **Download all logs:** a zip of every log file (`*.out`, `*.log`, `*.txt`) under the job's `logs/` directory, freshly `tar`-ed on the cluster. Files are always complete, whatever the `show`, filter or `--history` settings. Tachometer metrics are left out.
  - Theme and reset layout.
- **Resizing:**
  - Drag the gutters between panes and rows; double-click a gutter to even out its two panes.
  - Sizes are saved per layout shape.
  - Scrolling up in a pane pauses follow mode (amber outline); scrolling back to the bottom resumes it.

## Notes

- One SSH `tail -F` runs per batch of discovered files. After an SSH drop, those files are re-read from the start, so nothing is lost.
- The viewer only reads logs. It never touches processes or files on shared hosts.
