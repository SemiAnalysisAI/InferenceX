# llm-d-vllm multi-node SLURM scaffolding

This directory holds the SLURM-side orchestration for the `llm-d-vllm`
benchmark framework. It mirrors the AMD `sglang-disagg` pattern under
`benchmarks/multi_node/amd_utils/` (NOT the Dynamo / srt-slurm pattern):
InferenceX itself owns the SLURM job, no vendor multi-node tool involved.

| File | Role |
|---|---|
| `submit.sh` | sbatch wrapper. Validates env, exports tuning vars, returns `JOB_ID`. May read `slurm.time_limit` from the recipe to override `TIME_LIMIT`. |
| `job.slurm` | sbatch entrypoint. Allocates `PREFILL_NODES + DECODE_NODES` nodes, derives per-node IPs, runs one Docker container per node via `srun`, threads role assignment env into each. |
| `server.sh` | Per-node entry. Reads `NODE_RANK = SLURM_PROCID`, picks role, starts vLLM (with the wide-EP / DeepEP / NIXL flag set from the llm-d wide-EP-lws guide), starts the pd-sidecar on each leader, and on the decode leader additionally writes `endpoints.yaml`, starts EPP + Envoy, runs `benchmark_serving.py`, and `scancel`s the job. |

## Topology

For an `xP` prefill nodes / `yD` decode nodes run, total nodes = `xP + yD`.
There is **no dedicated coordinator node**. The decode leader doubles as
the coordinator (EPP + Envoy + bench), exactly like the AMD path's
"decode rank 0" coordinator role.

| Rank | Role |
|---|---|
| `0` | prefill leader (`LWS_WORKER_INDEX=0`, DP rank 0) + pd-sidecar |
| `1 .. xP-1` | prefill workers |
| `xP` | decode leader + pd-sidecar + EPP + Envoy + benchmark client |
| `xP+1 .. xP+yD-1` | decode workers |

Each instance (prefill or decode) is one vLLM engine spanning multiple
nodes via `--data-parallel-hybrid-lb`. With `xP=2, yD=2,
GPUS_PER_NODE=8` you get DP=16 prefill + DP=16 decode (the wide-EP
reference). Per-rank split: `--data-parallel-size 16
--data-parallel-size-local 8 --data-parallel-start-rank
$((LWS_WORKER_INDEX * 8))`.

## How `endpoints.yaml` is generated (file-discovery contract)

The EPP runs in **no-Kubernetes mode**, using the `file-discovery` plugin
from `llm-d-inference-scheduler` (branch `filediscovery-4`). At startup
it reads `/tmp/endpoints.yaml`; the file lists every backend the EPP can
route to, with role labels.

The file is generated at runtime by `server.sh` on the decode leader
(rank `PREFILL_NODES`). Because all node IPs are only known after
`sbatch` allocates the job, the file cannot be baked into the image and
is not part of the repo.

Generation flow:

1. `submit.sh` calls `sbatch -N (xP+yD)`. `sbatch` allocates nodes.
2. `job.slurm` resolves each node's IP via `srun ip route get 1.1.1.1`,
   slices them into `PREFILL_LEADER_IP` (= IPS[0]) and `DECODE_LEADER_IP`
   (= IPS[PREFILL_NODES]), and passes both into the container as env
   vars.
3. On the decode leader, `server.sh` writes `/tmp/endpoints.yaml`
   inside the container with one entry per leader:

   ```yaml
   endpoints:
     - name: prefill-0
       address: <PREFILL_LEADER_IP>
       port: "8000"            # pd-sidecar port
       labels:
         llm-d.ai/role: prefill
     - name: decode-0
       address: <DECODE_LEADER_IP>
       port: "8000"
       labels:
         llm-d.ai/role: decode
   ```

4. The EPP (started immediately after) loads the file via
   `dataLayer.discovery.pluginRef: file-disc` (see
   `benchmarks/llm-d/epp-config.yaml`). The plugin enumerates the
   endpoints into the EPP datastore before the EPP starts serving
   `ext_proc`, so Envoy never gets a request before discovery is ready.
5. The `disagg-profile-handler` in the EPP config uses `prefill-filter`
   and `decode-filter` to pick the right backend per request phase,
   matching on the `llm-d.ai/role` label.

### Why one entry per *leader* (not per node)

In the wide-EP guide each instance is a single vLLM engine that spans
multiple nodes via `--data-parallel-hybrid-lb`. With hybrid-lb, the
leader pod (`LWS_WORKER_INDEX=0`) accepts external traffic and
distributes it internally across the local DP ranks; in our LWS-free
SLURM mapping, the prefill-leader and decode-leader are the only nodes
addressable from outside. Adding an entry per worker would cause EPP to
route directly to a worker, bypassing the engine's internal load
balancing.

If we later want to expose all pods of an instance (the alternative
hybrid-lb interpretation: external LB across nodes too), we can extend
the loop in `server.sh` to emit one entry per `IPS[i]` in the prefill
range and one per `IPS[i]` in the decode range, all carrying the same
role label. EPP then load-balances across them via `random-picker`.

### Live reload

`watchFile: false` in `epp-config.yaml`. Endpoints are static for the
job lifetime - no reason to pay for `fsnotify` here. Set `watchFile:
true` (and rewrite `/tmp/endpoints.yaml` from the coordinator) only if
you want to drain or add an instance mid-run.

### Validation rules (enforced by the plugin)

- `address` must be a literal IPv4 address (no IPv6, no hostnames).
- `port` is a string in `1..65535`.
- File capped at 1 MiB.
- Names must be unique within their namespace (we use the default
  namespace, so they must be globally unique in the file).

The IPs we collect from `ip route get 1.1.1.1` are always IPv4 on the
H200 / B200 cluster's primary fabric; if you point at a different
interface and it returns an IPv6 address, EPP will reject the file at
startup.

## Recipe files

`benchmarks/multi_node/llm-d-recipes/<name>.yaml` is selected via
`CONFIG_FILE=<name>.yaml` in the master config's `additional-settings`.
Each recipe carries:

- top-level `plugins:` / `schedulingProfiles:` / `dataLayer:` - fed into
  the EPP via `--config-file`. Lets you change routing strategy without
  rebuilding the image.
- `prefill:` / `decode:` blocks with `extra-args` (appended to the vLLM
  launch command on each node of that role) and `env` (exported before
  vLLM starts).
- `slurm.time_limit` - overrides `TIME_LIMIT` for that recipe.

When `CONFIG_FILE` is unset or the file is missing, the EPP falls back
to `/etc/epp/config.yaml` baked into the image, and vLLM runs with no
extra flags beyond the wide-EP common set in `server.sh`.
