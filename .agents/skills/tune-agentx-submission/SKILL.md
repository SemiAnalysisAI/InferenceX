---
name: tune-agentx-submission
description: Supervise an AgentX performance submission from a bare recipe through a defensible Pareto sweep and accepted artifacts. Use for topology selection, staged tuning, and coordinating focused research and live-debug subskills.
---

# Tune an AgentX submission

Local-only orchestration skill. Do not commit, push, publish, or store private cluster addresses. Keep this workflow short. Put model and hardware evidence in a focused subskill and update it after every material result.

## Supporting skills

Load only the skills that match the current phase:

- `kimi-k3-b200-agentx` for PR #2475 decisions, measurements, concurrency ladders, and the living run record.
- `debug-agentx-runs` for compute-visible AgentX phase and health monitoring.
- `.agents/skills/debug-runs/SKILL.md` for the general CI, cluster reproduction, and merge-gate loop.

Create another model and hardware subskill when the evidence cannot be kept cleanly separate. Do not turn this coordinator into a model-specific notebook.

## Guardrails

1. Work only on the assigned PR and hardware.
2. For aggregate multinode jobs, srt-slurm owns allocation, rank-aware startup, readiness, and logs. Never hand-roll node orchestration.
3. Treat the checked-in recipe as serving-topology truth. Matrix metadata must match it.
4. Start from the official upstream recipe. Deviations need target-hardware evidence.
5. Preserve native context and the selected AgentX dataset.
6. Use synthetic acceptance only for comparable performance. Require real verification for correctness.
7. Read-only cluster inspection is allowed. Without explicit prior authorization, leave cancellation and infrastructure mutation untouched.
8. Run at most one or two small cluster experiments concurrently. Expand only after their live evidence is understood.
9. Success requires accepted artifacts, not healthy-looking logs.

## Supervision pattern

Keep the top-level decision with the lead agent. Delegate bounded slices such as upstream compatibility, model and KV geometry, prior artifacts, or renderer behavior.

Every slice must return:

- observed facts with paths, run IDs, and source links
- estimates with formulas and assumptions
- contradictions or blockers
- the smallest experiment that can resolve uncertainty.

Steer immediately when an agent follows an unsupported topology, treats an estimate as a measurement, or proposes custom orchestration. Cross-check consequential claims before they enter the case subskill.

## Workflow

1. **Ground the case.** Read the PR diff, current recipe, matrix entry, launcher, benchmark path, official recipe, model config, dataset distribution, and closest accepted artifacts.
2. **Design the Pareto profiles.** Cover low latency, balanced throughput, GPU-resident high concurrency, and CPU-offloaded maximum concurrency. Keep logical TP or TEP cache capacity separate from independent DEP pools.
3. **Verify rendering.** Generate the matrix and dry-run srt-slurm. Inspect leader and every non-leader command, especially rank-varying DP arguments.
4. **Run sparse canaries.** Use `agentx-fast` only for bring-up. Run one or two small points at a time, each exercising a distinct topology or mechanism.
5. **Monitor live and adapt.** Use compute-visible logs and metrics. Expand only while output throughput improves without leaving the intended latency frontier.
6. **Run the official sweep.** Remove fast mode, use the appropriate full-sweep label, verify correctness coverage and artifacts, then record the actual Pareto points.

## Expansion and stop rules

Continue only when all workers are active, routing is balanced, request errors are acceptable, and useful output throughput rises.

Stop expanding a branch when KV stays saturated with a growing queue, output throughput flattens while TTFT or TPOT worsens, offload traffic rises without benefit, a worker remains idle, or progress stops across repeated metric samples.

Capture evidence before recommending cancellation. If the user is unavailable and cancellation was not explicitly authorized, continue read-only monitoring and do not cancel.

## Update protocol

After every decision or run:

1. Update the focused case subskill with measured values and links.
2. Replace estimates that the run resolved.
3. Remove superseded topology or concurrency guidance.
4. State the next hypothesis and smallest disconfirming run.
5. Keep private access details outside the repository.
