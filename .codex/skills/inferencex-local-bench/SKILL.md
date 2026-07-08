---
name: inferencex-local-bench
description: Reproduce InferenceX benchmarks locally on Slurm/GPU nodes, especially MI355X multinode Kimi MoRI/Mooncake/LMCache agentic experiments. Use when Codex needs to run local_runner scripts, allocate or reuse Slurm jobs, launch prefill/decode/router/LMCache stacks, run smoke/GSM8K/agentic pressure tests, monitor cache hit and wait metrics, archive logs under durable storage, or compare local reproductions against CI behavior.
---

# InferenceX Local Bench

Use this skill for local or cluster-side reproduction. Keep local replay behavior isolated from CI: prefer `benchmarks/multi_node/local_runner/` wrappers and explicit env overrides instead of baking site-specific paths into CI recipes.

## Quick Workflow

1. Pick the correct worktree and branch.
   - Prefer the clean task worktree under `/mnt/c/users/yiczhu/workspace/06_kimi/mi300_lmcache/`.
   - Avoid dirty shared worktrees unless the user explicitly chose them.
   - Run `git status --short --branch` before editing or launching.

2. Choose the recipe and topology.
   - For SA/CI parity, use config-generated runners and avoid ad hoc env drift.
   - For local debug, use `benchmarks/multi_node/local_runner/*`.
   - Keep CI and local runner differences explicit: model cache path, `SLURM_REUSE_JOBID`, persistent run root, detached execution, and router image overrides.

3. Allocate or reuse nodes deliberately.
   - Do not release jobs until the user asks or the time box ends.
   - If a job must be kept while debugging, use `SLURM_REUSE_JOBID` or a detached launcher.
   - Record node names, IPs, job ID, container IDs, run directory, and exact git SHA.

4. Launch servers once, then run validation workloads without unnecessary restarts.
   - First run smoke/chat completion.
   - Then run GSM8K or eval if correctness is in scope.
   - Then run agentic pressure tests at requested concurrency and duration.
   - Avoid restarting prefill/decode just to change client workload.

5. Monitor the right metrics.
   - Track request success, running/waiting queues, prefix cache hit, external cache hit, GPU/CPU KV usage, LMCache health/metrics, and router/backend errors.
   - For MoRI/RDMA, also validate RDMA reachability and driver/userspace versions when transfer hangs.

6. Archive before cleanup.
   - Use durable storage such as `/it-share/yichaozhu/kimi-agentic` or `/data/yichaozhu`; avoid `/tmp`.
   - Archive server/router/LMCache logs, Slurm stdout/stderr, benchmark JSON, raw agentic traces, profile exports, metrics snapshots, and the launch env.
   - Record conclusions and paths in the experiment roadmap.

## Commands

For launch, monitor, validation, and cleanup templates, read [references/commands.md](references/commands.md).

## Experiment Notes

For Kimi MoRI/LMCache and Mooncake lessons learned, read [references/kimi-disagg-notes.md](references/kimi-disagg-notes.md).

## Safety Rules

- Do not kill existing containers or jobs unless the user says the occupied resources can be killed.
- Do not store the only copy of logs under `/tmp`.
- Do not mix local-only env workarounds into CI recipes.
- When debugging long-context agentic runs, treat slow decode as normal until metrics show a real stall.
