# Kimi K3 MI300X Native Multi-Node AgentX Design

<div align="center">

**English** | [中文](./2026-07-28-kimik3-mi300x-native-multinode-design_zh.md)

</div>

## Summary

Add an opt-in, native Slurm multi-node path for aggregated vLLM Kimi K3 AgentX
on the AMD MI300X cluster. The first vertical slice serves the plain target
model on two 8-GPU nodes with TP8 × PP2 and runs an AgentX canary. A second,
stacked change adds DSpark speculative decoding on the same lifecycle.

This is Kimi K3 support on Slurm. Kubernetes, P/D disaggregation, MI325X, and
changes to vLLM itself are outside this design.

## Current constraints

- Kimi K3 MXFP4 is approximately 1.5–1.56 TB. One 8×MI300X node has about the
  same aggregate HBM capacity and does not have safe runtime headroom.
- The usable MI300X nodes each expose 8 gfx942 GPUs and node-local
  `/raid/hf-hub-cache`.
- The existing `launch_mi300x-amds.sh` path is single-node and conflates total
  tensor parallelism with GPUs requested per node.
- `/raid` is node-local. A model cached on one node is not visible on another.
- Slurm credential creation, a two-node allocation, and a cross-node `srun` are
  now verified working on this cluster. The earlier
  `slurm_cred_create failure, holding job` hold is resolved and is no longer an
  execution gate.
- `/home` resolves to the nearly-full `/nvme_home` NFS export and is forbidden
  for the 30.8 GB container image. `gharunner` also cannot create `/raid/squash`.
  Every allocated node therefore imports and validates its own squash below
  `/raid/hf-hub-cache/inferencex/squash`, which the runner user can write.
- The target `moonshotai/Kimi-K3` weights are absent from the MI300X nodes
  today. Staging them is a later, explicit gate; the launcher fails closed
  rather than downloading them inside a timed benchmark job.
- The current Kimi K3 AMD references are InferenceX PRs #2351 (plain MI355X)
  and #2367 (DSpark MI355X). PR #2353 provides a native multi-node Kimi K3
  lifecycle on H200. The MI300X implementation reuses their proven contracts
  without depending on NVIDIA-only attention backends.
- The upstream Kimi K3 recipe marks only MI350X/MI355X gfx950 as verified.
  However, `vllm/vllm-openai-rocm:kimi-k3` is built with
  `AITER_ROCM_ARCH=gfx942;gfx950`, contains the AMD Kimi K3 implementation, and
  contains Kimi K3 AITER kernels. Its committed Kimi K3 tuned MoE tables are
  gfx950-only, so gfx942 correctness is a required preflight rather than an
  assumption.

## Approaches considered

### 1. Delegate from the existing MI300X launcher to an isolated native path

Add one early, opt-in branch to `launch_mi300x-amds.sh`:

```bash
if [[ "${NATIVE_MULTINODE:-0}" == "1" ]]; then
    exec bash runners/launch_mi300x-amds-native-multinode.sh
fi
```

The new launcher owns allocation, per-node container preparation, server
lifecycle, AgentX, artifacts, and cleanup. The existing single-node path stays
byte-for-byte unchanged below the delegation.

**Decision: selected.** It minimizes regression risk while retaining the
existing runner labels and matrix routing.

### 2. Add a completely new runner type and launcher

This provides maximum isolation but also requires new self-hosted runner
labels, runner-group wiring, and workflow routing. The physical pool already
uses `cluster:mi300x-amds`, so the extra runner identity adds operational work
without improving the benchmark.

**Decision: rejected.**

### 3. Route the experiment through srt-slurm

The B200 K3 work demonstrates this approach, but native multi-node aggregate
vLLM support still depends on upstream srt-slurm work and a patch. MI300X does
not need P/D orchestration for this vertical slice.

**Decision: rejected for this task.** It can be reconsidered if the scope later
expands to disaggregated serving.

## Delivery slices

### K3 PR A: plain vLLM vertical slice

- Opt-in MI300X native multi-node launcher.
- Two-node TP8 × PP2 Kimi K3 server entrypoint.
- Plain AgentX configuration with concurrency `[1, 2, 4, 8]`.
- Static, matrix-generation, lifecycle, and cleanup tests.
- One exact-head two-node concurrency-1 canary, followed by the bounded sweep.

The current matrix generator intentionally emits one multi-node AgentX
concurrency per job. The ladder above therefore creates four independent
Slurm allocations, rather than reusing one server allocation across all four
points.

### K3 PR B: DSpark

- DSpark server entrypoint or a thin mode wrapper over the plain entrypoint.
- `Inferact/Kimi-K3-DSpark`, seven speculative tokens, and the committed
  Kimi K3 golden acceptance-length contract.
- Synthetic acceptance for throughput and real block verification for eval.
- The same `[1, 2, 4, 8]` ladder so plain and DSpark are directly comparable.

PR B starts only after PR A has a green real-cluster canary. This keeps
distributed-launch failures separate from speculative-decoding failures.

## Topology contract

The only initial topology is:

```text
nodes              = 2
GPUs per node      = 8
world size         = 16
tensor parallelism = 8
pipeline stages    = 2
aggregate workers  = 1
decode workers     = 0
```

TP8 stays within each eight-GPU node; PP2 crosses the node boundary. This
matches the working K3 topology used by the H200 and B200 efforts and avoids a
cross-node TP16 collective on every tensor-parallel operation.

The launcher must keep these concepts separate:

```text
MULTINODE_NODE_COUNT=2
MULTINODE_GPUS_PER_NODE=8
PREFILL_TP=8
PREFILL_PP_SIZE=2
WORLD_SIZE=16
```

It must never derive the per-node Slurm GRES request from `TP`.

## Configuration contract

The AMD master configuration uses the existing aggregate multi-node schema:

```yaml
kimik3-fp4-mi300x-vllm-agentic:
  image: vllm/vllm-openai-rocm:kimi-k3
  model: moonshotai/Kimi-K3
  model-prefix: kimik3
  runner: cluster:mi300x-amds
  precision: fp4
  framework: vllm
  multinode: true
  disagg: false
  scenarios:
    agentic-coding:
    - search-space:
      - spec-decoding: none
        kv-offloading: none
        conc-list: [1, 2, 4, 8]
        prefill:
          num-worker: 1
          tp: 8
          pp: 2
          ep: 1
          dp-attn: false
          additional-settings:
          - "NATIVE_MULTINODE=1"
        decode:
          num-worker: 0
          tp: 8
          pp: 2
          ep: 1
          dp-attn: false
```

The runner rejects any native MI300X configuration that is not AgentX, vLLM,
aggregated, two-node, TP8 × PP2, and full-node 8-GPU allocation. This is a
deliberately narrow first contract.

The server entrypoint reuses only the architecture-neutral parts of the Kimi K3
AMD contract already exercised by #2351: the ROCm Kimi K3 image, fast
safetensors, and required parser/serving flags.

`AITER_SITUV2_A8W4` stays a runtime input rather than a frozen configuration
value. The matrix does not set it; the entrypoint accepts only `0` or `1` when
the caller sets it and otherwise preserves the image default. It becomes a
fixed value only after the parent task's exact-shape gfx942 comparison selects
a mode. Memory utilization and batching values remain explicit inputs on the
same grounds; the implementation must not copy gfx950-only tuning from MI355X
without evidence.

## Slurm and process lifecycle

The launcher performs this sequence:

```text
validate environment and topology
→ allocate 2 exclusive nodes, 1 task/node, 8 GPUs/node
→ resolve rank-0 hostname from the Slurm task rank
→ verify target cache on every allocated node
→ import/validate the container squash on every allocated node
→ launch one vLLM rank per node
→ wait for the rank-0 health endpoint
→ launch AgentX client on rank 0 with --overlap
→ persist results and bounded diagnostic artifacts
→ stop server step
→ cancel and reap allocation
```

The server receives:

```text
MULTINODE_NODE_COUNT=2
MULTINODE_GPUS_PER_NODE=8
MULTINODE_NODE_RANK=$SLURM_PROCID
MULTINODE_MASTER_ADDR=<rank-0 hostname>
```

The vLLM command uses:

```text
--tensor-parallel-size 8
--pipeline-parallel-size 2
--nnodes 2
--node-rank <0|1>
--master-addr <rank-0 hostname>
```

Rank 0 owns the OpenAI endpoint. Rank 1 adds `--headless`.

ROCm uses the Kimi K3 AMD environment from PRs #2351/#2367. The MI300X path
must not add NVIDIA-only `FLASHMLA` or `FLASHINFER_MLA` backends. Initial
bring-up uses GPU-resident KV only; host KV offload is a follow-up optimization,
not a loading workaround.

## Model and image staging

Formal benchmark runs are offline with respect to model weights:

- Target: `moonshotai/Kimi-K3`
- DSpark draft: `Inferact/Kimi-K3-DSpark`
- Host model cache: `/raid/hf-hub-cache` (node-local)
- Container image: `/raid/hf-hub-cache/inferencex/squash`, imported
  independently per node and validated by `unsquashfs -s`

The image is never staged under `/home`, `/nvme_home`, or `/raid/squash`: the
first two are the same nearly-full NFS export and the third is not creatable by
`gharunner`. Because `/raid` is node-local, the 30.8 GB import happens on every
allocated node rather than once on shared storage.

For the first canary, stage both selected nodes and pin the allocation to that
pair. Before enabling the CI sweep, stage the target on every eligible MI300X
runner node; stage the draft on the same pool before PR B.

Staging is resumable and serialized with a node-local lock. Each node must pass:

```text
exactly 8 gfx942 GPU agents
target snapshot present, revision-pinned, and complete against its weight index
draft snapshot present for DSpark
container squash present and valid
```

The production launcher fails closed when a required cache is absent. It does
not begin a 1.5 TB download inside a timed benchmark job.

## Logs, artifacts, and cleanup

- Server stdout/stderr goes to a host-owned scratch directory outside the
  GitHub Actions workspace.
- AgentX writes into a mounted scratch result directory. The host copies only
  final result files and bounded diagnostics into the workspace.
- No root-owned files may remain in the workspace, including after
  cancellation.
- `EXIT`, `INT`, and `TERM` cleanup stops the server step, snapshots available
  diagnostics, calls `scancel`, and waits until the allocation disappears.
- A server-rank failure terminates the entire `srun` via `--kill-on-bad-exit=1`.
- Readiness has a bounded 7,200-second deadline and fails early when the server
  step exits.

## Verification

### CPU/static gates

- `bash -n` for all changed shell scripts.
- Matrix generation for only the new config keys.
- Existing matrix suite remains green.
- Launcher tests cover:
  - native mode delegates while default mode does not;
  - total world size and per-node GPU request are not conflated;
  - unsupported topology fails before allocation;
  - both nodes participate in image/cache preflight;
  - rank 0 serves and rank 1 is headless;
  - failure and signal paths cancel the allocation.

### Cluster gates

1. A one-GPU container preflight reports gfx942, imports the AMD Kimi K3
   implementation, and passes a Kimi K3-shaped AITER MoE smoke test. It records
   whether a16w4 or a8w4 is valid, which is what finally fixes
   `AITER_SITUV2_A8W4`; failure of both modes becomes an explicit vLLM/AITER
   dependency rather than an InferenceX launcher workaround.
2. The target snapshot is staged on the selected pair without touching `/home`.
3. Both nodes pass model and container preflight, reporting one shared revision
   and a valid 30.8 GB squash each.
4. Plain vLLM reaches health and serves one direct request.
5. AgentX concurrency 1 produces a valid result and no residual Slurm job.
6. The four-job bounded plain sweep is green.
7. PR B repeats gates 4–6 with DSpark and runs the required eval path.

## Non-goals

- Kubernetes.
- MI325X.
- P/D disaggregation.
- srt-slurm changes.
- Hidden vLLM or Kimi K3 kernel changes inside the InferenceX recipe PR. If
  gfx942 preflight proves that an upstream fix is required, it is tracked as an
  explicit dependency.
- Facility-level power measurement.
- Rack-scale or more-than-two-node tuning.
- CPU/DRAM KV offload in the first plain vertical slice.
