# CollectiveX

Cross-vendor collective / EP-library benchmark (see `plan.md` for the full design).
The core is **MoE expert-parallel dispatch/combine** compared apples-to-apples across
EP libraries and SKUs, plus the surrounding inference collectives (KV-cache transfer,
all-reduce/all-gather, CPU↔GPU offload, copy-engine/SDMA, RL mesh transfer). Every
result is schema-validated (`schemas/ep-result-v4.schema.json`), correctness-gated
against an independent pure-torch oracle (`tests/reference_ep.py`), and carries full
provenance + a `comparison_key` so mismatched workloads are never silently overlaid.

> Experimental: WIP, not an official InferenceMAX result. All logic stays under
> `experimental/CollectiveX/`; the only files outside are the two orchestration-only
> workflows.

## EP backends

| Backend | Adapter | What it is | Coverage |
|---|---|---|---|
| `deepep` | `tests/ep_deepep.py` | bundled DeepEP 1.2.1 (`kernel_gen=v1`) | h100/h200/b200/b300/gb200/gb300 (EP4+EP8 MNNVL) |
| `deepep` + `--deepep-v2` | same (`kernel_gen=v2`) | upstream DeepEP main, built from source | same, incl. rack EP8 (needs `CX_ALLOW_MNNVL=1`) |
| `deepep-hybrid` | `tests/ep_deepep_hybrid.py` | NVIDIA HybridEP branch (`HybridEPBuffer`, TMA-NVLink) | h100/h200/b300/gb300 EP4+EP8 |
| `flashinfer` | `tests/ep_flashinfer.py` | TRT-LLM NVLink one-sided A2A (`MoeAlltoAll`); bf16 + fp8/mxfp8/nvfp4 dispatch, mxfp8/nvfp4 quant-combine | h100/b300/gb200/gb300 (rack EP up to 64); h200 = pidfd cap wall |
| `uccl` | `tests/ep_uccl.py` | UCCL EP via vendored `deep_ep_wrapper` | h100/h200/b200/b300 (x86 only — aarch64 wall) |
| `nccl-ep` | `tests/ep_nccl.py` | portable NCCL/RCCL `all_to_all_single` token-shuffle baseline (the ONLY backend that survives cross-node-over-IB here) | all NVIDIA SKUs + mi355x, incl. 2-node ws16 |
| `mori` | `tests/ep_mori.py` | AMD MoRI EP (bf16 + e4m3fnuz fp8) | mi355x |

Native `NVIDIA/nccl contrib/nccl_ep` is a **separate backend surface, not yet wired**
(do not alias it to DeepEP V2) — see `docs/gated.md`. Per-backend walls (h200
flashinfer pidfd/CAP_SYS_PTRACE, uccl aarch64, NIXL device-EP, MXFP4 scale layout,
h100 flashinfer intermittent MNNVL deadlock + LL fabric hang) are all evidenced in
`docs/gated.md` — judge runs by the artifact data (`correct=`/`status`), not the GHA
job conclusion (single diagnostic-case crashes flip jobs red despite 200+ correct points).

## Run

### CollectiveX Sweep (`.github/workflows/collectivex-sweep.yml`) — the main lane

`workflow_dispatch` → `sweep_matrix.py` resolves `configs/suites.yaml` into shards
(one shard = one GHA job = one slurm allocation sweeping many cases in one container);
an aggregate job collects every shard into `results/aggregate/*.ndjson`. Inputs:
`backend` (`all` = every EP backend in one combined matrix), `suites`, `only_sku`,
`min_nodes`/`max_nodes` (rack-scale EP8 vs single-tray), `max_cases` (chunking;
flashinfer force-chunks at 12 with a 3× per-case retry), `flashinfer_upgrade`.

### CollectiveX Experimental (`.github/workflows/collectivex-experimental.yml`)

- **push** to `experimental/CollectiveX/**` → the MI355X MoRI dispatch/combine sweep.
- **workflow_dispatch** → one `sku` × `benchmark` job: any EP backend above, or
  `nccl` (nccl-/rccl-tests), `flashinfer-combine-fp8|-nvfp4` (quant combine),
  `nixl`, `mori-io`, `nccl-kv`, `mooncake` (KV transfer), `offload`, `copy-engine`,
  `kv-cache`, `rl-mesh`, `allreduce-fw`, `allreduce-fw-vllm`, or `all`.

Both land on the SKU's self-hosted runner and invoke
`launchers/launch_${RUNNER_NAME%%_*}.sh` → `runtime/run_in_container.sh` (enroot/pyxis).
Do not delete ALL runs of the experimental workflow — it lives only on this branch and
would de-register (see `docs/gated.md`, operational note).

### Directly on a cluster login node

```bash
CX_BENCH=deepep bash experimental/CollectiveX/launchers/launch_h100-dgxc-slurm.sh
CX_BENCH=flashinfer CX_NODES=2 bash experimental/CollectiveX/launchers/launch_gb300-nv.sh  # rack EP8
CX_BENCH=mori bash experimental/CollectiveX/launchers/launch_mi355x-amds.sh
```

Key knobs: `CX_BENCH`, `CX_PHASE` (decode|prefill|both), `CX_TOKENS_LADDER`,
`CX_MODE` (normal|ll), `CX_DISPATCH_DTYPE`, `CX_COMBINE_DTYPE`, `CX_NODES`,
`CX_RDZV_FILE` (cross-node FileStore rendezvous), `CX_ALLOW_MNNVL`,
`CX_FLASHINFER_RETRIES`, `CX_TIME`, `CX_IMAGE`, `CX_DRYRUN=1`.

## Pipeline & files

| File | Role |
|---|---|
| `configs/suites.yaml` + `workloads.yaml` + `backends.yaml` + `platforms.yaml` | suite/workload/backend/SKU definitions |
| `sweep_matrix.py` (uses `generate_matrix.py`) | suites → shard matrix for the sweep workflow |
| `tests/run_ep.py` + `tests/ep_harness.py` | EP entrypoint (torchrun) + shared harness: token ladder, separated dispatch/combine/roundtrip timing, correctness gate, doc emission |
| `tests/capability.py` | (sku, backend, mode, dtype, contract) validity — rejects unsupported combos up front |
| `tests/reference_ep.py` | independent pure-torch EP oracle (routing/dispatch/combine ground truth) |
| `tests/routing.py`, `tests/workload.py`, `tests/eplb.py` | routing distributions + canonical workload manifests (`workload_id`, trace signatures) |
| `validate_results.py` | strict v4-schema + comparison-contract validation of every artifact |
| `aggregate_results.py`, `summarize.py`, `regression.py`, `cohort.py`, `repeated_runs.py`, `prune_results.py` | aggregate/report/regress/prune tooling (workflow-invoked) |
| `plot_ep.py` (+ `plot.py`, `analyze_ep.py`) | the 8-tab HTML report (EP, KV-cache, all-reduce, all-gather, RL-mesh, copy-engine, …) with comparison guards |
| `runtime/common.sh`, `runtime/run_in_container.sh`, `runtime/_xnode_net.sh` | image resolve/squash, in-container dispatcher (per-case loop, idempotent from-source builds, flashinfer retry), cross-node net helpers |
| `run_nccl.py` | nccl-/rccl-tests runner + text-table parser |
| `env_capture.py` | Layer-0 environment + topology fingerprint on every result |
| `schemas/` | `ep-result-v4` + `workload-v1` JSON schemas |
| `docs/` | `methodology.md` (timing/correctness/publication contracts), `gated.md` (evidenced walls + open items), `upstream_precision.md` (PR311/3376/3643 review), `references.md` (paper notes) |
| `CONTAINERS.md` | pinned containers + audited library versions |

## Container

One multi-arch image for all NVIDIA SKUs, imported by tag `lmsysorg/sglang:v0.5.11-cu130`
(amd64+arm64; bundles deep_ep 1.2.1 / flashinfer 0.6.8 / NCCL 2.28.9 / torch 2.11).
Container switches per bench where needed (dynamo image for NIXL, vllm/vllm-openai for
`allreduce-fw-vllm`, ROCm MoRI image for MI355X). See `CONTAINERS.md`.

## Status

All P0/P1/P2 goal items are done or evidenced-gated; full EP sweeps exist for
h100 / h200 / b300 / gb300 (+ b200/gb200 spot coverage and mi355x MoRI). The open
items are: the native `contrib/nccl_ep` adapter (only remaining unwired backend),
the h100 flashinfer intermittent-deadlock root-cause (needs live compute-sanitizer),
and an h100 quant-combine re-run on the newer wheel. Details: `docs/gated.md`.
