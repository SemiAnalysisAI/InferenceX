# Reuse One AgentX Server Across Points

By default `python3 -m infx.bench agentic` measures exactly one concurrency per server
deployment, and that stays the default. Setting `AGENTX_SERVER_REUSE=true` instead runs
every `CONC_LIST` point, in order, against the one server that is already up, without
restarting it. Reuse is allowed only for points whose bound recipes launch an identical
vLLM server with a local KV cache. Traffic may use its direct API or one consistent-hash
vLLM Router. No recipe or generator emits a reuse group.

## Admission between points

Before the first point, the group checks two things. First, `GET /is_sleeping` must answer
`{"is_sleeping": false}`. That route is read-only, and vLLM's Python and Rust API servers
mount it under the same `VLLM_SERVER_DEV_MODE` switch as `/reset_prefix_cache`, so a
server launched without dev mode is refused before any traffic. Second, every expected
engine must be idle. The first point then runs on the fresh server exactly as a normal
point. A later point starts only after all of these hold:

1. The previous point exited 0 under the ordinary scorer (aggregation, validation and the
   optional required-metric prefix).
2. Its `aiperf_artifacts/logs/aiperf.log` shows one warmup then one profiling phase, with
   every issued request completed. Nothing was cancelled or errored, no grace period
   expired, and no completion was forced. The ordinary scorer tolerates up to 10% failed
   requests; reuse tolerates none.
3. No process from the client's session is still alive
   `AGENTX_REUSE_CLIENT_EXIT_TIMEOUT_SECONDS` after the client exits. The replay runs in
   its own session, and signals are forwarded to it. Survivors are killed and the group
   stops.
4. Two consecutive `GET /metrics` scrapes report exactly the `AGENTX_REUSE_ENGINES`
   engine indices, all idle, within `AGENTX_REUSE_DRAIN_TIMEOUT_SECONDS`. Idle means
   `vllm:num_requests_running` and `vllm:num_requests_waiting` are 0. Each engine's
   lifetime `vllm:prefix_cache_queries_total` and `vllm:prefix_cache_hits_total` must be
   present and never lower than at the previous scrape. A drop means an engine restarted
   under the same index.
5. `POST /reset_prefix_cache?reset_running_requests=false&reset_external=false` is sent
   once, with no retry. It must answer HTTP 200 with a JSON object whose `success` is the
   boolean `true`. The serving API fans the reset out to its internal DP engines, so there
   is no per-engine reset route.
6. After the reset, the same engines are idle again and their counters are still
   continuous.

Any failure stops the group, and later points never start. Each point keeps its own
`RESULT_DIR/conc_<C>` directory, `<RESULT_FILENAME>_conc<C>` aggregate and AIPerf
profiling-window metrics. Prefix reuse inside a point and normal warmup are unchanged.
Before it starts, the group refuses result paths that already exist, so an earlier run's
log or metrics can't mix into a point.

## Signals

SIGINT, SIGTERM or SIGHUP during a point reaches only that point's client session. The
point exits with 128 + N, and no later point starts. A signal between points (while
draining, resetting or waiting for idle) stops admission at the next check, and the
command exits with 128 + N.

`RESULT_DIR/server_reuse.json` records the group's progress after each step:
`status` is `running`, `passed`, `failed` or `terminated`, and `error` gives the reason
(for example `signal 15 received after c8; no later point admitted`). It also holds the
launch-identity hash, per-point exit codes, drain receipts (log SHA-256 and request
counts), and engine gauges and counters before and after each reset. A SIGKILL leaves
the last saved step.

## Launch identity

`AGENTX_REUSE_LAUNCH_IDENTITIES` names the JSON file that
`python -m infx.srt_slurm.reuse_identity` writes from each point's bound single-node
recipe: the `recipe.yaml` that `infx.srt_slurm.single_node prepare` writes beside its
arguments. Run it on the host, outside the container:

```bash
python -m infx.srt_slurm.reuse_identity \
  --point 8=c8/recipe.yaml --point 16=c16/recipe.yaml --output launch-identities.json
```

The file has this schema:

```json
{"schema": "infx-agentx-server-reuse/v1",
 "points": {"<CONC>": {"image": "<IMAGE>", "model": "<MODEL>", "server": {...}}}}
```

`server` is the complete bound recipe without the two point-specific keys: `benchmark`
(the client's command, `CONC`, output paths and concurrencies) and `name` (the variant
label). It covers model, container, engine, frontend, every role's engine arguments and
environment, setup script, resources and telemetry. The producer refuses a recipe bound
to a different `CONC`. Both the producer and the client then require all of the
following:

- Every point declares an identical `server`. The error names each differing path, such
  as `server.roles.agg.args.max-num-seqs`.
- `model` and `image` match `MODEL` and `IMAGE`, and also `server.model.path` and
  `container`.
- The engine is `vllm` with no `engine.connector`.
- `frontend` has `enable_multiple_frontends: false` and type `vllm` or `vllm-router`.
  A router must use `policy: consistent_hash` and `disable-retries: true`.
- There is exactly one `agg` role with one worker and non-empty engine arguments.
- No key anywhere in `server` names a KV connector, KV transfer, offload, LMCache or
  Mooncake setting. This includes `kv-transfer-config`, `kv-offloading-backend` and
  `LMCACHE_*`.

Runtime `--set` overrides that `single_node prepare` derives from the job (GPU count,
srun options, client paths) apply to the whole job and are not part of the file. The
identity proves what the points' recipes would launch. It cannot read the running
server's arguments, so the job must be launched from that same bound recipe.

## Supported scope

- Client: `agentx-harness==1.0.12` only. The drain barrier parses that release's
  phase-lifecycle log lines. If `infx/bench/agentic/requirements.txt` pins anything
  else, the group is rejected until the parser is revalidated.
- Rejected before any point: `IS_MULTINODE=true`, `EVAL_ONLY=true`, AgentX power
  windows, `KV_OFFLOADING` other than `none`, a non-empty `KV_OFFLOAD_BACKEND`, and any
  identity outside the rules above.
- The reset clears only the local GPU prefix cache. External KV stores and runtime
  scheduler settings are out of scope.
- On current `main`, the DSV4.1-Flash B300 AgentX DEP2 points (c8 and c16 bind identical
  servers, with `max-num-seqs` 16) run behind `frontend.type: vllm-router`. Their bound
  launch identities pass. c16/c32 is refused because sequence and graph settings differ.

## Router and control endpoints

Replay keeps its normal frontend URL. Control uses the one aggregated backend from
`SRT_AGG_ENDPOINTS`. Routed groups require this backend address before any traffic; they
never send reset requests through the router. Direct groups without a separate address
use their replay API for control.

The backend must be an HTTP API root without a subpath, credentials, query or fragment.
When a separate control address is used, each point's server-metrics input is its
`/metrics` endpoint. An explicit `AIPERF_SERVER_METRICS_URLS` must match that single
endpoint, so router and engine counters cannot be mixed. The expected engine set is
still checked on every scrape. One backend API performs the internal-DP reset.

The integrated command was exercised with DeepSeek-V4.1-Flash on two B300s
(TP1/DP2) through vLLM Router 0.1.14. A c8-to-c16 smoke run completed with no
failed or cancelled requests, a successful non-preempting reset between points,
and clean client/server teardown. A post-group probe confirmed cold miss, warm hit,
reset, and first-post-reset miss on both engines. The 60-second profiling windows
qualify these mechanics, not performance equivalence or model accuracy. Other
images, client versions and topologies need their own qualification.

A point whose issued streams are still running at the profiling cutoff needs
`AIPERF_BENCHMARK_GRACE_PERIOD` long enough for them to finish. With the upstream default,
cancelled requests fail the drain barrier and the group stops.

## Example

```bash
export AGENTX_SERVER_REUSE=true CONC_LIST="8 16" CONC=8 KV_OFFLOADING=none
export AGENTX_REUSE_ENGINES="0 1"
export AGENTX_REUSE_LAUNCH_IDENTITIES=/path/to/launch-identities.json
export AGENTX_REUSE_DRAIN_TIMEOUT_SECONDS=600 AGENTX_REUSE_CLIENT_EXIT_TIMEOUT_SECONDS=60
export AIPERF_BENCHMARK_GRACE_PERIOD=1800
# Routed jobs supply SRT_AGG_ENDPOINTS through srt-slurm.
# plus the normal single-node point inputs (RESULT_DIR, MODEL, IMAGE, DURATION, ...)
python3 -m infx.bench agentic
```

Before a reused result is compared with or reported as a fresh-server result, confirm the
group on real GPUs for the target recipe. CPU tests cover the control flow, not drain
timing, router or output behavior.
