# Kimi K3 on Two MI300X Nodes

<div align="center">

**English** | [中文](./kimik3-mi300x-native-multinode_zh.md)

</div>

Operator guide for the opt-in native multi-node path that serves Kimi K3 with
plain vLLM across two 8×MI300X nodes and drives AgentX against it.

> **This launcher never downloads the checkpoint.** Kimi K3 MXFP4 is about
> 1.5 TB. If the weights are missing from a node, the job fails immediately with
> a message naming the missing path. Staging is a separate, deliberate step you
> run before the benchmark.

## What this path supports

| | |
|---|---|
| Scenario | agentic-coding (AgentX trace replay) |
| Framework | plain vLLM, aggregated (no P/D disaggregation) |
| Hardware | 2 nodes × 8 gfx942 MI300X |
| Parallelism | TP8 inside each node, PP2 across the two nodes, EP1 |
| Concurrency | one of 1, 2, 4, or 8 per job |
| KV cache | GPU-resident only |

Anything else is rejected before the job allocates anything. TP8 stays inside a
node so no tensor-parallel collective crosses the network; PP2 is what makes the
model fit, because one node's aggregate HBM leaves no safe headroom for a 1.5 TB
checkpoint.

MI325X, Kubernetes, P/D disaggregation, srt-slurm, and DSpark are all out of
scope here.

## Turning it on

The matrix key is `kimik3-fp4-mi300x-vllm-agentic` in `configs/amd-master.yaml`.
Its prefill worker carries one additional setting:

```yaml
additional-settings:
- "NATIVE_MULTINODE=1"
- "KIMIK3_NODELIST=chi-mi300x-043,chi-mi300x-054"
- "AITER_SITUV2_A8W4=0"
```

`NATIVE_MULTINODE=1` selects the native launcher. `KIMIK3_NODELIST` pins the
allocation to the exact pair carrying the node-local snapshot.
`runners/launch_mi300x-amds.sh` checks the first setting and hands off to
`runners/launch_mi300x-amds-native-multinode.sh`. Every other MI300X config
still runs the unchanged single-node path.

The launcher requires this environment, all of which the multi-node workflow
template already provides:

```text
IS_MULTINODE=true          SCENARIO_TYPE=agentic-coding    IS_AGENTIC=1
FRAMEWORK=vllm             MODEL_PREFIX=kimik3             PRECISION=fp4
SPEC_DECODING=none         DISAGG=false                    KV_OFFLOADING=none
PREFILL_NUM_WORKERS=1      PREFILL_TP=8                    PREFILL_PP_SIZE=2
PREFILL_EP=1               PREFILL_DP_ATTN=false
DECODE_NUM_WORKERS=0       DECODE_TP=8                     DECODE_PP_SIZE=2
DECODE_EP=1                DECODE_DP_ATTN=false
CONC_LIST=<one of 1 2 4 8> KIMIK3_NODELIST=<staged-node-a,staged-node-b>
IMAGE  MODEL  RESULT_FILENAME  RUNNER_NAME
```

Optional knobs, with their defaults:

| Variable | Default | Purpose |
|---|---|---|
| `KIMIK3_MODEL_CACHE_ROOT` | `/raid/hf-hub-cache/models--moonshotai--Kimi-K3` | node-local model cache |
| `KIMIK3_NODELIST` | required; matrix pins `chi-mi300x-043,chi-mi300x-054` | exact pair carrying the staged snapshot |
| `KIMIK3_SQUASH_DIR` | `/raid/hf-hub-cache/inferencex/squash` | node-local image tree |
| `HF_HUB_CACHE_MOUNT` | `/raid/hf-hub-cache/inferencex/agentx-hub` | host side of the client's HF cache mount |
| `HF_HUB_CACHE` | `/hf-hub-cache` | container side; where AgentX caches its trace corpus |
| `KIMIK3_SLURM_TIME_MINUTES` | `480` | allocation wall clock |
| `KIMIK3_STARTUP_TIMEOUT_SECONDS` | `7200` | how long to wait for `/health` |
| `KIMIK3_CLEANUP_TIMEOUT_SECONDS` | `120` | how long cleanup waits for the job to disappear |
| `KIMIK3_PRESCANCEL_TIMEOUT_SECONDS` | `15` | deadline for the one cleanup step that must precede `scancel` |
| `KIMIK3_IMAGE_LOCK_TIMEOUT_SECONDS` | `3600` | wait for the per-node image lock |

## Model layout on each node

`/raid` is node-local, so a snapshot on one node is invisible to the other.
Both allocated nodes need their own copy, at the same revision:

```text
/raid/hf-hub-cache/models--moonshotai--Kimi-K3/
├── refs/main                       # 40-character revision
└── snapshots/<revision>/
    ├── config.json
    ├── model.safetensors.index.json
    └── model-*.safetensors
```

Hugging Face snapshots normally contain symlinks into the sibling `blobs/`
directory. The server therefore mounts the entire
`models--moonshotai--Kimi-K3` cache read-only and resolves the model as
`/models-cache/snapshots/<revision>`; mounting only the snapshot would leave
those weight links dangling.

`runners/mi300x_native_node_preflight.sh` runs on every allocated node and
checks, in order:

1. exactly 8 GPU agents, all `gfx942`;
2. `refs/main` holds a 40-character revision;
3. the snapshot directory and its `config.json` exist;
4. **every** shard named in `model.safetensors.index.json` exists and is
   non-empty.

Step 4 matters most. A partially synced snapshot otherwise looks fine until the
model load fails hours into the allocation. The launcher then compares the
records from both nodes and refuses to start a server unless it sees two
distinct hostnames reporting one shared revision.

## Container image on each node

The image is imported per node into `/raid/hf-hub-cache/inferencex/squash`.
Three paths are deliberately avoided:

- `/home` and `/nvme_home` are the same nearly-full NFS export, and the image is
  about 30.8 GB;
- `/raid/squash` cannot be created by the `gharunner` user.

Import is serialized with a node-local `flock`, so parallel jobs on the same
node wait rather than race. A valid image is reused as-is. An absent or invalid
one is re-imported into a temporary file in the same directory, validated with
`unsquashfs -s`, and only then moved into place. If a run dies mid-import, the
trap removes the temporary file and leaves any previously validated image alone.

Enroot's cache and temp directories are pinned under the same node-local tree so
nothing lands in `$HOME`.

## AITER_SITUV2_A8W4

The image is built with `AITER_ROCM_ARCH=gfx942;gfx950`, but its original
Kimi K3 tuned MoE tables cover gfx950 only. The exact K3-shape gfx942 kernel
gate passed both constant and random numerical checks with the A16W4 path, so
the matrix pins:

```text
AITER_SITUV2_A8W4=0
```

The entrypoint still accepts an explicit `0` or `1` for diagnostic runs and
rejects anything else before allocation. It prints the resolved value at
startup so every run records what it used.

## Verifying locally

None of this needs a GPU, Slurm, a container import, or network access:

```bash
python3 -m pytest utils/matrix_logic/ -q
python3 utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files configs/amd-master.yaml \
  --config-keys kimik3-fp4-mi300x-vllm-agentic \
  --scenario-type agentic-coding \
  --no-evals
bash -n runners/launch_mi300x-amds.sh
bash -n runners/launch_mi300x-amds-native-multinode.sh
bash -n runners/mi300x_native_node_preflight.sh
bash -n benchmarks/multi_node/agentic/kimik3_fp4_mi300x_vllm.sh
```

Matrix generation should print four rows, at concurrency 1, 2, 4, and 8.

To see the exact vLLM command a rank would run, without starting anything:

```bash
KIMIK3_VLLM_DRY_RUN=1 MULTINODE_NODE_RANK=0 MULTINODE_MASTER_ADDR=<head-node> \
  bash benchmarks/multi_node/agentic/kimik3_fp4_mi300x_vllm.sh
```

## Cluster gates, in order

Run these before enabling the sweep. Each one is a stop-the-line check.

1. Single-GPU gfx942 preflight: confirm the AITER MoE path works and record
   whether a16w4 or a8w4 is valid. That result is what finally fixes
   `AITER_SITUV2_A8W4`.
2. Stage the target snapshot on the chosen pair of nodes, without touching
   `/home`.
3. Run the node preflight on both nodes. Expect one shared revision and a valid
   image on each.
4. Bring up plain vLLM and serve one direct request against rank 0.
5. Run AgentX at concurrency 1. Check the result is valid and no Slurm job is
   left behind.
6. Run the full 1/2/4/8 sweep.

DSpark (PR B) starts only after gate 5 is green, so a distributed-launch problem
never gets mistaken for a speculative-decoding problem.

## Troubleshooting

**`missing model revision pointer ...` or `missing model snapshot directory ...`**
The node has no staged snapshot. Stage it and re-run; the launcher will not
download it for you.

**`missing weight shard(s) ...`**
The snapshot is incomplete. Finish the transfer rather than deleting the index.

**`nodes hold different model revisions: ...`**
The two allocated nodes are on different snapshots. Re-stage the older node, or
pin the allocation to a matched pair.

**`imported image ... failed unsquashfs validation`**
The import produced a bad squash. The temporary file is already removed; check
free space under `/raid` and re-run.

**`the vLLM server step exited with code N before becoming healthy`**
A rank died during startup. The last 200 lines of the server log are printed
inline, and the full log is uploaded as `multinode_server_logs.tar.gz`.
`--kill-on-bad-exit=1` means one bad rank takes down both.

**`the vLLM server did not become healthy within Ns`**
Loading 1.5 TB across two nodes is slow. Confirm both ranks are alive in the
log before raising `KIMIK3_STARTUP_TIMEOUT_SECONDS`.

**A Slurm job outlives the run**
Cleanup calls `scancel` and then polls `squeue` on every exit path, including
`INT`, `TERM`, and `HUP`, and the workflow cancels by job name before and after
each run. If one still lingers, `scancel --name="$RUNNER_NAME"` clears it.
