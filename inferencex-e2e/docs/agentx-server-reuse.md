# Reuse a server for AgentX

`AGENTX_SERVER_REUSE=true` runs `CONC_LIST` points in order on the existing server,
with a local prefix-cache reset between points. Without it, AgentX still runs one
concurrency per deployment. Recipes and generators do not enable reuse automatically.

Use it only for identical single-node vLLM launches with local KV cache and
`agentx-harness==1.0.12`. Direct traffic and one consistent-hash vLLM Router are
supported. Multiple frontends, connectors, offloading, eval-only runs and power
windows are rejected. Launch vLLM with `VLLM_SERVER_DEV_MODE=1` to expose the reset
endpoint.

## Run a group

Keep the normal [AgentX inputs](agentx-standalone.md). From the host, generate a
launch identity for each point using its bound `recipe.yaml` from
`infx.srt_slurm.single_node prepare`:

```bash
python -m infx.srt_slurm.reuse_identity \
  --point 8=c8/recipe.yaml --point 16=c16/recipe.yaml --output launch-identities.json
```

The identities compare the full server recipe, excluding only `name` and
`benchmark`. Model, image, engine arguments, environment and resources must match.
Connector/offload settings are refused even when identical. Launch the server
from that recipe and apply any job overrides consistently: this file describes
the intended launch, not the arguments of a running server.

```bash
export AGENTX_SERVER_REUSE=true CONC_LIST="8 16" CONC=8 KV_OFFLOADING=none
export AGENTX_REUSE_ENGINES="0 1"
export AGENTX_REUSE_LAUNCH_IDENTITIES=/path/to/launch-identities.json
export AGENTX_REUSE_DRAIN_TIMEOUT_SECONDS=600 AGENTX_REUSE_CLIENT_EXIT_TIMEOUT_SECONDS=60
export AIPERF_BENCHMARK_GRACE_PERIOD=1800
python3 -m infx.bench agentic
```

`CONC` must be the first point. Use fresh result paths; existing group or point
outputs are rejected. Give the client enough grace time to finish long streams:
a profiling cutoff followed by cancellation fails the group.

Routed groups require one `SRT_AGG_ENDPOINTS` backend and a router configured with
`policy: consistent_hash` and `disable-retries: true`. Replay uses the router;
reset and metrics use the backend directly. Direct groups can use their replay
API for both. The control address must be an HTTP API root without credentials,
subpaths, query or fragment. If `AIPERF_SERVER_METRICS_URLS` is set, it must match
that backend's single `/metrics` endpoint.

## Between points

Before traffic, `/is_sleeping` must report false and every expected engine must
be idle. Each later point also requires:

1. The previous point passed the normal scorer and its pinned client log shows
   completed warmup and profiling phases with every issued request finished,
   zero errors/cancellations and no forced completion or expired grace period.
2. The client's process group has exited. Survivors are killed after
   `AGENTX_REUSE_CLIENT_EXIT_TIMEOUT_SECONDS`, and the group fails.
3. Two consecutive metrics scrapes show exactly `AGENTX_REUSE_ENGINES`, with zero
   running/waiting requests. Each engine's prefix-query and prefix-hit counters
   must be present and continuous; a decrease indicates a restart. Waiting is
   bounded by `AGENTX_REUSE_DRAIN_TIMEOUT_SECONDS`.
4. One `POST /reset_prefix_cache?reset_running_requests=false&reset_external=false`
   returns HTTP 200 and JSON `{"success": true}` with a boolean value. There is
   no retry or preemption. The backend fans reset out to its internal DP engines.
5. The same idle and counter-continuity checks pass after reset.

Any failure stops the group before the next point. SIGINT, SIGTERM and SIGHUP
stop admission, forward to the active client session, and return `128 + signal`.

Each point keeps `RESULT_DIR/conc_<C>` and `<RESULT_FILENAME>_conc<C>` results.
`RESULT_DIR/server_reuse.json` records progress, launch identity, exit codes,
drain evidence and boundary metrics. SIGKILL leaves the last saved step.

Reset clears local prefix lookup state, not every part of a fresh process.
Prefix counters remain cumulative, so compare deltas between boundaries.
Qualify the target image, router and workload on GPUs before treating reused
measurements as equivalent to fresh-server results. CPU checks establish control
flow, not benchmark accuracy or performance equivalence.
