# CollectiveX — Plan

> **How to read this.** This is the single canonical plan. It is **spike-first** and **scoped to `experimental/CollectiveX/`** on a branch — nothing in the production serving path changes until a promotion decision is made later. Part 1 is background (what CollectiveX is, reconstructed from team discussion). Part 2 is the implementation plan. Where this plan says "now," it means the Milestone 0 spike; "later" items (GitHub workflow, database, app frontend) are deliberately deferred. All repository references (runners, launchers, workflows, matrix logic, the `experimental/` charter) were verified against the live InferenceX repo — see References.

---

# Part 1 — Background

## What it is

CollectiveX is an benchmarking workstream under the InferenceX umbrella. It measures **collective communication** and **MoE dispatch/combine**, and performs **apples-to-apples, cross-vendor comparison of expert-parallel (EP) libraries** across NVIDIA and AMD (TPU later). The intended deliverables are an **OSS benchmark project** and a **public explainer article** — a credible cross-vendor collective benchmark plus the story around it.

## Why

Existing public benchmarks don't offer trustworthy, like-for-like collective/EP comparison across vendors. CollectiveX fills that gap by reusing InferenceX's runner and cluster infrastructure to produce reproducible, provenance-tagged results.

## Current state

- An initial MVP exists: it collected collective and kernel shapes and produced MoE dispatch/combine results on NVIDIA.
- **Normal mode works; low-latency (LL) mode is blocked** on IBGDA enablement — a direct GPU↔NIC data-and-control path over PCIe that removes CPU coordination and simplifies MoE dispatch/combine collectives — which depends on cluster-networking work outside this project.
- The main near-term enabler is NVIDIA networking / IBGDA; the AMD EP stack and AMD networking (Ultra Ethernet) are the cross-vendor counterpart.

---

# Part 2 — Implementation plan

## Implementation status (built)

The Milestone-0 spike ran for real on **both** B200 (8× NVLink island, x86_64) and GB200 (4× NVL72 MNNVL, aarch64) — 4 NCCL primitives, correctness-passed, topology-keyed distinctly (peak bus-bw: B200 all-reduce 835 GB/s; GB200 689 GB/s). Built on top of that:

- **Multi-arch, digest-pinned container** for all NVIDIA SKUs: `lmsysorg/sglang:v0.5.12-cu130@sha256:4219…f356` (amd64 + arm64) — one reference both arches; DeepEP via `rebuild-deepep`. See `CONTAINERS.md`.
- **Per-SKU launch adapters** (`launchers/launch_<sku>.sh`, the InferenceX `launch_${RUNNER_NAME%%_*}.sh` convention) that run **any** benchmark via `CX_BENCH` (nccl|deepep|all) through a shared `launchers/run_in_container.sh`.
- **`on: push` workflow** (`.github/workflows/collectivex-experimental.yml`): push → GB200 NCCL smoke; `workflow_dispatch` → chosen `sku`+`benchmark`. No merge to main; activates when the branch is pushed to GitHub.

This supersedes the Milestone-0 "light single-script launcher" sketch below where they differ — launchers are now thin SKU adapters + a shared dispatcher (still light/experimental).

## Scope and placement

CollectiveX starts as an **experimental project on its own branch**, fully contained under `experimental/CollectiveX/`:

```bash
git switch main
git pull --ff-only
git switch -c collectivex
mkdir -p experimental/CollectiveX
```

This matches the repository's intent: `experimental/` is explicitly non-core ("experimental WIP code that is mostly Claude Code generated… not intended for production use or as part of the official InferenceMAX results").

For the experimental phase, **everything stays inside `experimental/CollectiveX/**`**. Do **not** modify:

```text
benchmarks/
runners/
utils/
.github/configs/
perf-changelog.yaml
InferenceX-app
```

The only eventual exception is a minimal workflow dispatcher under `.github/workflows/` (because executable workflows must live there); all real CollectiveX logic, schemas, launchers, and processing stay under `experimental/CollectiveX/`.

**This supersedes any notion of CollectiveX becoming a top-level InferenceX subsystem or extending the production serving matrix up front.** Promotion — into core InferenceX, into a dedicated repo, or into InferenceX-app's database/frontend — is an explicit *later* decision (Milestone 4), made only after the benchmark contract has stabilized on real hardware.

### What InferenceX already gives us

InferenceX's existing execution model is almost exactly the control plane CollectiveX needs:

1. Generate and strictly validate a matrix on a GitHub-hosted runner.
2. Fan jobs out to named or labelled self-hosted runners.
3. Those listeners submit work to Slurm (or launch Docker locally).
4. Normalize outputs.
5. Upload artifacts.
6. Aggregate and dispatch ingestion to the dashboard.

`e2e-tests.yml` already divides generated configs into job families and invokes reusable single-node and multi-node workflows; `benchmark-tmpl.yml` cleans up resources, checks out the selected ref, **derives the launcher from the runner name**, launches the job, validates outputs, and uploads normalized results. Runner listeners live on cluster login/controller nodes while jobs run on compute nodes via Slurm; runner names/labels are load-bearing — the name prefix selects the launcher and exact names/SKU labels control scheduling.

CollectiveX reuses all of this, but enters through **CollectiveX-specific launchers** rather than threading fake models through the serving launchers (see Cluster reuse).

## Architecture

Four planes, cleanly separated:

- **Control plane:** scheduling, runners, cleanup, artifact movement, workflow metadata (reused from InferenceX).
- **Benchmark plane:** collective semantics, backend invocation, correctness, timing.
- **Data plane:** canonical result records, raw per-rank samples, topology and provenance.
- **Presentation plane:** comparable subsets, charts, history, diagnostics.

Data flow within the experimental directory:

```text
Portable shape definitions
          +
Backend definitions
          +
Target/cluster definitions
          ↓
CollectiveX matrix resolver
          ↓
Resolved shards
          ↓
Existing InferenceX self-hosted runner
          ↓
experimental/CollectiveX/launchers/*
          ↓
Backend adapter  (NCCL / RCCL / DeepEP / AITER / MoRI / …)
          ↓
Versioned result bundle
          ↓
Aggregator + regression checker
          ↓
Static experimental report   →  (later) InferenceX-app ingestion → Postgres → /collectives
```

### Target structure at promotion (Milestone 4)

This packaged layout is the **promotion target**, not the spike. Milestone 0 uses the light layout in the rollout section below (`run_nccl.py` / `run_deepep.py` / `env_capture.py` / `plot.py` + flat `results/`); the structure here is what CollectiveX grows into *if* it is promoted out of `experimental/`.

```text
InferenceX/
├── experimental/
│   ├── README.md
│   └── CollectiveX/
│       ├── README.md
│       ├── DESIGN.md
│       ├── ROADMAP.md
│       ├── pyproject.toml
│       ├── Makefile
│       │
│       ├── src/
│       │   └── collectivex/
│       │       ├── __init__.py
│       │       ├── cli.py
│       │       ├── config/
│       │       │   ├── models.py
│       │       │   ├── loader.py
│       │       │   ├── resolver.py
│       │       │   └── matrix.py
│       │       ├── benchmark/
│       │       │   ├── harness.py
│       │       │   ├── timing.py
│       │       │   ├── correctness.py
│       │       │   ├── routing.py
│       │       │   └── metrics.py
│       │       ├── backends/
│       │       │   ├── base.py
│       │       │   ├── fake.py
│       │       │   ├── nccl_tests.py
│       │       │   ├── rccl_tests.py
│       │       │   ├── deepep.py
│       │       │   └── framework_ep.py
│       │       ├── cluster/
│       │       │   ├── inventory.py
│       │       │   ├── capabilities.py
│       │       │   ├── environment.py
│       │       │   └── launcher.py
│       │       ├── results/
│       │       │   ├── models.py
│       │       │   ├── writer.py
│       │       │   ├── aggregate.py
│       │       │   ├── compare.py
│       │       │   └── redact.py
│       │       └── report/
│       │           ├── build.py
│       │           └── templates/
│       │
│       ├── configs/
│       │   ├── suites/
│       │   │   ├── smoke.yaml
│       │   │   ├── primitives.yaml
│       │   │   ├── moe-decode.yaml
│       │   │   ├── moe-prefill.yaml
│       │   │   └── full.yaml
│       │   ├── shapes/
│       │   │   ├── synthetic/
│       │   │   └── traced/
│       │   ├── backends/
│       │   ├── targets/
│       │   └── clusters.yaml
│       │
│       ├── launchers/
│       │   ├── common.sh
│       │   ├── launch_b200-dgxc.sh         # B200 single node
│       │   ├── launch_b200-dgxc-slurm.sh   # B200 multinode
│       │   └── launch_gb200-nv.sh          # GB200 NVL72
│       │
│       ├── schemas/
│       │   ├── case-v1.schema.json
│       │   ├── result-v1.schema.json
│       │   ├── manifest-v1.schema.json
│       │   └── environment-v1.schema.json
│       │
│       ├── scripts/
│       │   ├── bootstrap.sh
│       │   ├── run_suite.sh
│       │   ├── run_shard.sh
│       │   └── build_report.sh
│       │
│       ├── tests/
│       │   ├── fixtures/
│       │   ├── test_config.py
│       │   ├── test_matrix.py
│       │   ├── test_parsers.py
│       │   ├── test_correctness.py
│       │   └── test_comparability.py
│       │
│       └── docs/
│           ├── BENCHMARK_CONTRACT.md
│           ├── BACKEND_ADAPTER.md
│           ├── SHAPE_REGISTRY.md
│           ├── RESULT_FORMAT.md
│           ├── FRONTEND.md
│           └── PROMOTION_CRITERIA.md
│
└── .github/workflows/
    └── collectivex-experimental.yml   # Added only when cluster CI begins (Milestone 2)
```

> Note: launcher names mirror the real runner-name prefixes. The spike adds the three NVIDIA launchers above; AMD (`launch_mi355x-amds.sh`) and others follow.

## Benchmark model — keep four concepts separate

CollectiveX needs its **own** schema. Do **not** reuse or extend the serving matrix, which is built around model / ISL / OSL / framework / TP / EP / concurrency and lives in `utils/matrix_logic/generate_sweep_configs.py`. Representing collectives with fake model names, `ISL=0`, or overloaded concurrency fields would create permanent technical debt. CollectiveX gets its own matrix logic (in the packaged layout, `src/collectivex/config/matrix.py`) — introduced with the workflow at Milestone 2, not the spike — rather than touching `utils/matrix_logic/generate_sweep_configs.py`.

The model keeps four concepts independent:

**Shape** — the logical communication workload:

```text
operation, message size, tokens per rank, hidden size, top-k,
expert count, routing distribution, dtype, phase
```

**Backend** — the implementation under test:

```text
NCCL, RCCL, DeepEP, AITER, MoRI, framework-native EP, reference implementation
```

**Target** — where and how it runs:

```text
runner type, cluster, nodes, GPUs per node, rank placement,
fabric, container image, transport capabilities
```

**Suite** — a curated selection of shape × backend × target combinations. Keeping these separate prevents copying the same DeepSeek/MiniMax shape into every NVIDIA and AMD configuration.

### Portable definitions

Shape:

```yaml
schema-version: 1
shape-id: moe.decode.h7168.top8.e256.t64.uniform.v1

kind: moe
phase: decode
operation: dispatch-combine

shape:
  tokens-per-rank: 64
  hidden-size: 7168
  top-k: 8
  num-experts: 256
  dispatch-dtype: fp8
  combine-dtype: bf16
  routing:
    distribution: uniform
    seed: 67
  expert-alignment: 16
```

Backend:

```yaml
backend-id: deepep-normal
backend: deepep
mode: normal

source:
  repository: deepseek-ai/DeepEP
  ref: pinned-commit

settings:
  async-overlap: false
  num-comm-sms: standardized
  qp-count: auto
```

Target:

```yaml
target-id: b200-dgxc-4n
runner-type: b200-multinode
cluster-id: b200-dgxc

resources:
  nodes: 4
  gpus-per-node: 8
  exclusive: true

placement:
  ranks-per-node: 8
  rank-order: contiguous

capabilities:
  rdma: true
  ibgda: experimental
  nvshmem: true
```

Suite:

```yaml
suite-id: moe-decode-smoke

shapes:
  - moe.decode.h7168.top8.e256.t64.uniform.v1

backends:
  - deepep-normal
  - deepep-low-latency

targets:
  - b200-dgxc-2n

measurement:
  warmup-iterations: 20
  measured-iterations: 200
  trials: 3
  correctness: full
```

### Case identity

A **case** is one immutable, versioned point: the natural key composes the three concepts —

```text
case-id = <backend-id> __ <shape-id> __ <target-id>
e.g.  deepep-normal__moe.decode.h7168.top8.e256.t64.uniform.v1__b200-dgxc-4n
      nccl__allreduce.fp16.logsweep.v1__b200-dgxc-2n
```

A shape must never silently change; a newly extracted distribution gets a new versioned `shape-id`.

**Required shape fields — primitives:** operation; logical element count; datatype; input/output bytes; in-place vs out-of-place; reduction op (where applicable); world size; rank placement; host-driven vs device-driven launch; blocking/synchronization semantics.

**Required shape fields — MoE (additional):** tokens per rank; hidden size; top-k; number of experts; EP size; dispatch and combine dtypes; routing distribution; expert alignment/padding; capacity constraints; quantization scale representation; cached vs recomputed routing layout; communication-SM count; async-overlap mode. DeepEP shows why these must be first-class — its interface takes tokens/rank, hidden size, top-k, expert count, FP8 mode and comm-SM settings, and exposes async dispatch/combine.

### Shape registry

Two independent shape sources:

**Synthetic** — for continuous curves and hardware characterization (logarithmic byte sweep for primitives; token-count sweep for MoE; EP-scaling sweep; uniform and controlled-skew routing; intranode and internode placements; decode-oriented and prefill-oriented regimes). Don't build every Cartesian combination; define named suites (`primitive-latency-v1`, `primitive-bandwidth-v1`, `moe-decode-v1`, `moe-prefill-v1`, `moe-skew-v1`, `scaleout-v1`).

**Trace-derived** — extracted from real InferenceX runs/profiles:

```text
models/deepseek-v4/decode/<shape-id>
models/minimax-m3/decode/<shape-id>
models/kimi-k2.7/prefill/<shape-id>
```

Each traced shape retains: source workflow run; model/config; phase; layer/layer-group; observed token histogram; routing skew; concurrent collective count; framework version; extraction-tool version. InferenceX already has a targeted profiling workflow (`profile.yml`) with optional MoE debug output and a separate trace-storage path — a natural source for real shapes rather than only guessed synthetic inputs.

## Benchmark layers and comparison classes

| Layer | Purpose | Examples |
|---|---|---|
| **L0 Environment** | Prove the cluster is benchmarkable | topology, NIC/GPU state, peer access, RDMA, IBGDA capability, version capture |
| **L1 Primitive collectives** | Characterize the raw communication substrate | send/recv, all-reduce, all-gather, reduce-scatter, all-to-all, all-to-allv |
| **L2 MoE communication** | Compare real EP libraries | dispatch, combine, dispatch+combine round trip, normal and low-latency modes |
| **L3 Integrated pipelines** | Communication in realistic operator sequences | route → permute → dispatch → grouped GEMM → combine → unpermute |
| **L4 E2E correlation** | Explain InferenceX serving performance | isolated CollectiveX result linked to the corresponding InferenceX run/profile |

The MVP concentrates on **L1 and L2**. L3 overlaps OperatorX and comes after the contracts are stable; L4 is the eventual tie-back to serving.

**L0 — Environment validation** (before measuring anything): GPU count/identity; GPU/NIC topology; CUDA/ROCm version; driver version; NCCL/RCCL version; RDMA device visibility; peer-access matrix; IBGDA/SHMEM capability; container digest; clock/power state; selected network interfaces. A failed probe yields one clear `environment-invalid` result, not dozens of misleading backend failures.

**L1 — Primitives:** send/receive, all-reduce, all-gather, reduce-scatter, all-to-all, all-to-allv. Use vendor test programs where possible rather than rewriting primitives. Measure two regions separately: latency (bytes→low KiB) and bandwidth (MiB→GiB).

**L2 — MoE collectives:** dispatch, combine, dispatch+combine. Dimensions: tokens/rank, hidden size, top-k, expert count, EP size, dispatch dtype, combine dtype, routing skew, normal vs low-latency, comm-SM count, node count.

### Three comparison classes

Every result is tagged with exactly one, and they must never be silently mixed on one chart:

| Class | Meaning |
|---|---|
| `standardized` | Matched logical shape **and** fixed resource budget — same shape, topology, dtype, correctness contract, allowed comm-SMs, and timing boundaries. The main apples-to-apples comparison. |
| `backend-optimized` | Same logical output, but each library uses its recommended comm-SMs / protocols / QP count / buffer sizing / graph capture / tuning. Answers "what is the best each stack can do?" |
| `framework-integrated` | The actual path used by SGLang / vLLM / TensorRT-LLM / Dynamo. Connects to InferenceX; not a pure microbenchmark. |

### Comparability key

Every result gets a machine-generated comparison key; rows with different keys are not connected on the same curve by default:

```text
operation, shape ID, dtype, world size, node count, rank placement,
routing distribution, comparison class, measurement contract version, topology class
```

## Measurement and correctness

### Timing boundaries

Record separately — never report one latency that sometimes includes JIT and sometimes doesn't:

```text
1. communicator creation
2. buffer allocation and registration
3. first invocation / JIT
4. warmed steady-state invocation
5. host launch time
6. GPU completion time
7. optional end-to-end framework-visible time
```

Per measured iteration: synchronize before starting (unless explicitly testing queued execution); use GPU events for device duration and host monotonic time for API/launch duration; retain per-rank measurements; aggregate only after rank-level data is stored; report the **slowest rank** as well as the average.

### Correctness as a hard gate

A result is `valid` only after correctness passes. A fast result that fails correctness stays visible as `invalid` — never silently dropped.

Primitive checks: deterministic input; expected reduction result; guard regions around buffers; in-place and out-of-place checks; dtype-specific tolerances.

MoE checks: token conservation; correct expert assignment; correct routing weights; valid permutation metadata; dispatch output vs reference; combine output vs reference; no padded-token leakage; deterministic routing hash.

Failed results remain in artifacts, e.g.:

```json
{
  "status": "invalid",
  "correctness_passed": false,
  "error": "combine result exceeded bf16 tolerance"
}
```

### Routing distributions

At minimum: uniform; single-hot/worst-case concentration; Zipf-like skew; bounded imbalance; replayed real histogram. Store the routing seed and the generated assignment hash.

### Metrics

| Category | Metrics |
|---|---|
| Latency | p50, p90, p95, p99, min, max |
| Rank behavior | slowest-rank latency, rank spread, coefficient of variation |
| Primitive throughput | algorithm bandwidth, bus bandwidth, effective bytes/s |
| MoE throughput | tokens/s, logical payload GB/s, dispatch and combine separately |
| Efficiency | bandwidth relative to declared topology bottleneck |
| Host overhead | API launch time, CPU utilization where available |
| GPU overhead | communication SM count, GPU active time, optional power |
| Memory | persistent buffer bytes, peak temporary bytes |
| Overlap | standalone comm, standalone compute, overlapped duration, overlap efficiency |
| Reliability | initialization failures, hangs, retries, correctness failures |
| Provenance | all software, image, driver, firmware and topology identifiers |

### Bandwidth definitions

NCCL `algbw`/`busbw` are stored but not treated as universal (NCCL applies operation-specific correction factors). MoE libraries often report **logical bottleneck bandwidth** (may include local-rank traffic or exclude metadata/padding; DeepEP explicitly publishes logical bandwidth). Store separate fields, and use `null` rather than a deceptive inference when a backend can't expose physical bytes:

```text
logical_payload_bytes
allocated_payload_bytes
estimated_link_bytes
metadata_bytes
padding_bytes
```

## Result and artifact format

Each shard emits a versioned bundle:

```text
output/
├── manifest.json
├── cases.json
├── results.jsonl
├── rank-samples.jsonl.gz
├── summary.json
├── environment/
│   ├── gpu.json
│   ├── network.json
│   ├── topology.json
│   └── software.json
├── raw/
│   ├── stdout.log
│   ├── stderr.log
│   └── backend-output/
├── commands/
│   └── reproduce.sh
└── profiles/
```

**Manifest** (invariant run-level metadata): schema version; workflow run + attempt; source SHA/ref; cluster ID; runner; Slurm job ID; node count; topology fingerprint; image digest; backend commit/build; start/end timestamps; redaction version.

**Result row:**

```json
{
  "schema_version": 1,
  "case_id": "deepep-normal__moe.decode.h7168.top8.e256.t64.uniform.v1__b200-dgxc-4n",
  "status": "valid",
  "trial": 1,
  "backend": "deepep",
  "mode": "normal",
  "comparison_class": "standardized",
  "metrics": {
    "latency_us_p50": 0,
    "latency_us_p99": 0,
    "slowest_rank_us_p50": 0,
    "logical_bandwidth_gbps": 0,
    "tokens_per_second": 0,
    "rank_spread_pct": 0,
    "persistent_buffer_bytes": 0
  },
  "correctness": { "passed": true, "max_abs_error": 0, "max_rel_error": 0 }
}
```

Use an explicit `schema_version` from the beginning — do not repeat the app's historical need to infer schema version from whether a field happens to exist.

## Backend adapters

Each adapter implements a small contract:

```python
class CollectiveBackend:
    def probe(self, environment) -> CapabilityReport: ...
    def prepare(self, case, workdir) -> PreparedCommand: ...
    def run(self, prepared, launcher) -> RawRun: ...
    def parse(self, raw_run) -> list[RankSample]: ...
    def validate(self, case, raw_run) -> CorrectnessReport: ...
    def describe(self) -> BackendProvenance: ...
```

**Tier 0 — communication baselines:** NVIDIA `nccl-tests`, ROCm `rccl-tests`, optionally PyTorch distributed as a common-API baseline. Don't rewrite primitives from scratch — `nccl-tests` already supports multi-node, warmups, correctness checking (`-c 1`), per-rank aggregation, device-driven implementations, and separate CPU-time reporting. *(Confirm whether the installed build emits JSON; if not, parse the text table.)*

**Tier 1 — MoE dispatch/combine:** upstream DeepEP, ROCm DeepEP, and the NVIDIA/AMD EP paths already used by the InferenceX serving stacks. **Version pins are first-class.** Upstream DeepEP V2 changed NVSHMEM→NCCL, unified high-throughput and low-latency APIs, changed buffer behavior, and removed a previous zero-SM LL mode; ROCm's port has different maturity, NIC variants, rocSHMEM dependencies. DeepEP is **built at job setup** (via `rebuild-deepep.sh`, resolved by srt-slurm), not shipped in the image — its build time and `aarch64` (GB200) feasibility are tracked spike risks. A chart labelled only "DeepEP" is therefore ambiguous — store:

```text
backend name, upstream/fork, git commit, API generation,
transport backend, build flags, runtime library versions, container digest
```

**Tier 2 — additional optimized stacks (later):** MSCCL++, AITER comm/fusion paths, MoRI/Pollara, NVSHMEM/rocSHMEM microbenchmarks, framework-native fused collectives.

## Rollout — spike-first

**Spike-first.** No schema, Pydantic model, or comparison contract is frozen until one real, correctness-gated number exists on real hardware. The first milestone is a single end-to-end spike on **two NVIDIA topologies, B200 and GB200**, chosen because they exercise the two transport regimes that matter: B200 is an 8-GPU NVLink island with CX-7 InfiniBand between nodes; GB200 is an NVL72 multi-node-NVLink (MNNVL) domain. Running the same collective across both is itself the first headline result, and it forces the provenance and comparison-class machinery to be real from line one. The schema is the spike's *output*, extracted from the artifacts it produces — not its input. AMD and all platform work (workflow, DB, frontend) follow.

### Milestone 0 — NVIDIA B200 + GB200 spike

One milestone, NVIDIA-only, end to end. This collapses the former "design contract," "CPU framework," "primitive NVIDIA baseline," and the NVIDIA half of "MoE MVP" into a single vertical slice that produces real numbers on real fabric.

Scaffolding — deliberately light, matching `experimental/` convention (bare scripts + flat JSON + a plot; no package / Pydantic / JSON-schemas yet — those arrive at the contract freeze):

```text
experimental/CollectiveX/
  README.md
  run_nccl.py        # argparse; run stock nccl-tests, parse its text table (do NOT assume JSON)
  run_deepep.py      # one dispatch+combine shape, normal mode
  env_capture.py     # Layer-0 env + topology fingerprint (torch.cuda.* + nvidia-smi topo) → json
  plot.py            # matplotlib, like token_position_decode_slo/*/plot_*.py
  launchers/
    common.sh
    launch_b200-dgxc.sh         # B200 single node  (b200-dgxc runner → 8-GPU NVLink island, x86_64)
    launch_b200-dgxc-slurm.sh   # B200 multinode    (b200-multinode runner → CX-7 IB spine)
    launch_gb200-nv.sh          # GB200             (gb200 runner → NVL72 MNNVL, aarch64, 4 GPU/node)
  results/*.json     # flat, hand-verifiable
```

Reuse existing patterns rather than reinventing: `experimental/dsv32/bench.py` for `torch.cuda.Event` timing and stdout environment capture, and `experimental/token_position_decode_slo/glm-5/{bmk_*_sbatch.sh,plot_sla_frontier.py}` for Slurm orchestration + plotting. Mirror the runner→launcher routing convention (`bash ./launchers/launch_${RUNNER_NAME%%_*}.sh`) so the runner name selects the CollectiveX launcher as the serving path does.

**DeepEP is not prebuilt in any image.** The serving recipes build it at job setup via `setup_script: rebuild-deepep.sh` (resolved by srt-slurm; see `benchmarks/multi_node/srt-slurm-recipes/sglang/qwen3.5/gb200-fp8/`). The spike reuses that same rebuild path — on B200 (x86_64) first. Pin images by digest from `.github/configs/nvidia-master.yaml`: B200 `lmsysorg/sglang:deepseek-v4-blackwell@sha256:df18bfc4aa9ecf59451002b49ba00cae58042de9e2a96378bbd21b404dd62c7b`; GB200 `lmsysorg/sglang:nightly-dev-cu13-20260608-303757cc` (an unpinned nightly today — capture its digest before relying on it).

What it measures:

```text
Primitives (stock nccl-tests, -c 1 for correctness) — on BOTH B200 and GB200:
  all-reduce, all-gather, reduce-scatter, all-to-all
  latency regime (bytes→KiB) and bandwidth regime (MiB→GiB)
  B200  : 8 GPU/node (x86_64); 1 node (NVLink island) and 2 nodes (cross CX-7 IB)
  GB200 : 4 GPU/node (aarch64); 1 node and 2+ nodes — all still inside the NVL72 NVLink (MNNVL) domain

MoE (DeepEP, normal mode only — LL mode is the known-broken/blocked path, out of scope):
  one decode-shaped dispatch+combine: tokens-per-rank=64, hidden=7168,
  top-k=8, experts=256, dispatch fp8
  correctness: token conservation + combine vs a reference implementation
  B200 (x86_64) first; GB200 DeepEP is a fast-follow once the aarch64 rebuild-deepep path is proven
```

The headline is the **same NCCL primitive shape on both topologies**: B200's 2-node path crosses CX-7 InfiniBand, while GB200's stays on NVL72 NVLink (MNNVL). That IB-vs-MNNVL contrast at a matched logical shape is the result worth publishing. (nccl-tests and DeepEP must be built for `aarch64` on GB200 — the reason DeepEP is B200-first.)

Provenance captured on every row from the first run — non-negotiable even in a spike, because it is what makes the B200-vs-GB200 number defensible:

```text
topology-class       b200-nvlink-island(+cx7-ib)  |  gb200-nvl72-mnnvl
transport actually used   (NVLink / IB / NVSHMEM-IBGDA), derived from flags + measured behavior
transport env set/recorded:
  B200  : NCCL_CUMEM_ENABLE=1
  GB200 : NCCL_CUMEM_ENABLE=1, NCCL_MNNVL_ENABLE=1, MC_FORCE_MNNVL=1
  (also seen in serving: NCCL_P2P_LEVEL=NVL, SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK)
comm-SM count, QP count where applicable
backend commit + API generation + build flags
container digest, CUDA / driver / NCCL versions
comparison-class tag (standardized where shape, dtype and SM budget match)
```

These flags come from validated GB200 serving recipes (`…/srt-slurm-recipes/sglang/qwen3.5/gb200-fp8/`); MNNVL is GB200/GB300-only, which is exactly what makes the transport differ from B200.

Output: a result bundle on disk (`manifest.json`, `results.jsonl`, `environment/`, `raw/`, `commands/reproduce.sh`). Hand-verify the first rows; do not build a generated Pydantic contract yet.

Exit criteria:

* real NCCL latency + bandwidth curves on **both** B200 and GB200, correctness-passed (the headline)
* one DeepEP dispatch+combine number (normal mode) on **B200**, correctness-passed; GB200 DeepEP as the immediate fast-follow
* every row carries topology-class, transport, comparison-class and full provenance
* a B200-vs-GB200 side-by-side that the comparison key permits **and labels as topology-class-differing** — that labeled comparison is the intended result, not an accident
* **only now** freeze the schema (`CollectiveCase` / `CollectiveResult` / manifest), extracted from these artifacts

Explicitly out of scope for the spike: AMD, IBGDA low-latency mode, GitHub Actions, database, frontend, trace-derived shapes, and the fake backend as a deliverable (keep a trivial one only if it speeds offline tests).

### Milestone 1 — AMD parity

Bring the AMD side up against the schema the spike froze — not in parallel with it:

```text
RCCL-tests adapter (mirror the nccl-tests text-table parser)
one AMD launcher (launch_mi355x-amds.sh)
one AMD MoE dispatch/combine backend (DeepEP ROCm / AITER / MoRI)
equivalent shapes + identical result contract
first cross-vendor (NVIDIA vs AMD) comparison
```

Record the AMD transport stack (rocSHMEM, MoRI-IO / Pollara, NIC variant) with the same provenance rigor the spike established. An unlabeled "DeepEP" row compared across vendors is meaningless.

### Milestone 2 — GitHub workflow

Add (orchestration only; see GitHub workflow design below):

```text
collectivex-experimental.yml
preflight
canary
matrix sharding
artifact collection
regression comparison
static report artifact
```

Do not connect it to `perf-changelog.yaml`.

### Milestone 3 — Trace-derived shapes

Extract representative shapes from InferenceX profiles (DeepSeek V4, MiniMax M3, Kimi). Every traced shape must retain: source workflow run; source configuration; framework version; model phase; extraction-tool version; routing-histogram hash.

### Milestone 4 — Promotion decision

Only then decide whether to: keep CollectiveX permanently experimental; move it into core InferenceX; extract it into a dedicated repository; or integrate its data into InferenceX-app (database + `/collectives` frontend).

### First PRs (the spike)

The spike lands as a few small PRs, each producing something runnable — not a docs-and-schema PR:

```text
1. Scaffold + NCCL on B200 single node
   run_nccl.py (text-table parser), env_capture.py, plot.py,
   launchers/launch_b200-dgxc.sh, results/*.json
   → lands when it emits a real all-reduce curve with provenance from an 8-GPU B200

2. B200 multinode + GB200
   launchers/launch_b200-dgxc-slurm.sh, launchers/launch_gb200-nv.sh
   → lands when the same primitive runs on 2-node B200 (cross-IB) and on GB200 NVL72 (MNNVL),
     each tagged with topology-class and transport (aarch64 build for GB200)

3. DeepEP dispatch+combine — B200 first
   run_deepep.py, routing generator + reference combine for correctness,
   reusing rebuild-deepep at job setup
   → one decode shape, normal mode, on B200; GB200 DeepEP fast-follow

4. Freeze the contract
   extract the case / result / manifest schema from the bundles produced in 1–3;
   add fixtures captured from real output — this is where the packaged structure begins
```

The first objective is a real, provenance-tagged, correctness-gated number on two NVIDIA topologies — the contract is the spike's output, not its foundation.

## Cluster reuse and capability inventory

### What to reuse

Existing self-hosted runner registrations; exact runner labels; Slurm access from runner hosts; checkout and artifact patterns; resource-cleanup strategy; repository secrets; container caches where appropriate. The runner inventory (`.github/configs/runners.yaml`) already enumerates H100, H200, B200, B300, GB200, GB300, MI300X, MI325X, MI355X fleets and groups such as `h200-multinode`, `b200-multinode`, individual nodes, etc. CollectiveX **reads** this file rather than duplicating runner names.

### What not to reuse directly

Do not call the serving launchers (`runners/launch_${RUNNER_NAME%%_*}.sh`) — they carry model-serving assumptions (model paths, framework setup, result naming). Mirror the **selection convention** with CollectiveX launchers instead:

```bash
bash experimental/CollectiveX/launchers/launch_${RUNNER_NAME%%_*}.sh
```

Each CollectiveX launcher handles only: Slurm allocation; container image; mounts; network environment; rank launch; result copy-back; cleanup. There are **two launch paths**, mirroring the serving side: **single-node** B200 mirrors the `salloc … --gres=gpu:N --exclusive … && srun --container-image=<enroot squash>` pattern in `runners/launch_b200-dgxc.sh`; **multi-node** B200/GB200 drives **srt-slurm** (`srtctl apply -f <recipe>`), which already knows how to rebuild DeepEP and set the MNNVL env — so the CollectiveX GB200 launcher is a thin wrapper handing srt-slurm a CollectiveX recipe, not a from-scratch sbatch. (Later, common Slurm/container functions can be factored into a shared lib used by both systems.)

> Runner-name subtlety to handle in `inventory.py`: one physical cluster can appear under multiple prefixes — `b200-dgxc_NN` routes to `launch_b200-dgxc.sh` (single-node) while `b200-dgxc-slurm_N` (label `b200-multinode`) routes to `launch_b200-dgxc-slurm.sh`. One fabric domain can therefore span several runner labels.

### Capability overlay

`inventory.py` loads `../../../.github/configs/runners.yaml` and combines it with a CollectiveX capability overlay — one source of truth for runner names, CollectiveX metadata kept isolated:

```yaml
b200-multinode:
  launcher: b200-dgxc-slurm
  vendor: nvidia
  hardware: b200
  topology-class: b200-nvlink-cx7
  fabric-domain: b200-dgxc-main
  gpus-per-node: 8
  arch: x86_64
  max-nodes: 16
  scheduler: slurm
  container-runtime: enroot-pyxis
  capabilities:
    nccl: true
    deepep: true                # built at job setup via rebuild-deepep, not prebuilt
    rdma: true
    nvshmem: true
    ibgda: experimental         # capability present ≠ currently validated
  scheduling:
    exclusive-nodes: true
    max-parallel-shards: 1

gb200:
  launcher: gb200-nv
  vendor: nvidia
  hardware: gb200
  topology-class: gb200-nvl72-mnnvl
  gpus-per-node: 4              # NVL72 compute tray
  arch: aarch64                 # nccl-tests + DeepEP must build for aarch64
  scheduler: srt-slurm
  transport-env: { NCCL_CUMEM_ENABLE: 1, NCCL_MNNVL_ENABLE: 1, MC_FORCE_MNNVL: 1 }
  capabilities:
    nccl: true
    deepep: true                # rebuilt at setup; aarch64 path is a tracked risk
    mnnvl: true                 # GB200/GB300 only
    ibgda: experimental
```

`fabric-domain` is essential: two jobs on separate compute nodes may still contend for the same leaf/spine network, so **GitHub concurrency is keyed by fabric domain, not GPU SKU**. The inventory distinguishes hardware capability, software currently installed, and feature state (known-good vs experimental vs temporarily broken) — IBGDA support and "IBGDA low-latency currently validated" are different properties.

**Operational coexistence with the serving sweep.** `b200-multinode` is only three runners (`b200-dgxc-slurm_7/8/9`), **shared with the production serving sweeps**, and srt-slurm allocations are long. Exclusive nodes + `max-parallel-shards: 1` + fabric-domain serialization means CollectiveX and the serving sweep contend for the same scarce runners. Decide the scheduling/coexistence policy (off-hours windows? a dedicated runner?) before enabling any recurring CollectiveX suite, rather than discovering the contention in CI.

## GitHub workflow design (Milestone 2)

When cluster CI begins, add one small orchestration-only file — `.github/workflows/collectivex-experimental.yml` — with no benchmarking logic:

```text
validate → resolve matrix → preflight canaries → benchmark shards
→ aggregate → compare against baseline → build static report → upload artifacts
```

Triggers while on the branch:

```yaml
on:
  push:
    branches: [ collectivex ]
    paths:
      - experimental/CollectiveX/**
      - .github/workflows/collectivex-experimental.yml
  pull_request:
    paths:
      - experimental/CollectiveX/**
      - .github/workflows/collectivex-experimental.yml
```

Later, after a minimal dispatcher exists on `main`, add `workflow_dispatch` with inputs: `ref, suite, target, backend, shape, profile` (and comparison class / normal-LL-both / dry-run).

Jobs:

1. **Validate** — install the package; validate all suite/shape/backend/cluster YAML; confirm runner references exist in `runners.yaml`; reject unknown fields; emit the resolved run plan as an artifact. (Match InferenceX's strict Pydantic practice — models reject extra fields.)
2. **Compile and shard** — **do not** generate one job per benchmark point. Group cases by `cluster, node count, GPU placement, container image, backend build, transport mode, fabric domain, profiler requirement`. A shard runs many compatible points under one Slurm allocation (avoids thousands of matrix jobs, repeated communicator init, queue latency, repeated container import). Bounded runtime; record per-case failures unless the cluster itself is unhealthy.
3. **Preflight** — confirm GPU count; validate peer access; enumerate NICs; test RDMA/device visibility; verify backend libraries; run a tiny correctness case; capture topology/software. A failed preflight marks the whole shard `environment-invalid` rather than manufacturing dozens of backend failures.
4. **Canary** — for each `(cluster, backend, mode)` group, run one small representative case; launch the larger matrix only after it passes (mirrors InferenceX's canary-before-full-sweep).
5. **Benchmark** (`collectivex-benchmark-tmpl.yml`) — run on the resolved runner label; unique Slurm job name from workflow/attempt/shard; exclusive nodes; serialize/limit by `fabric-domain`; call the CollectiveX launcher; upload results even on partial failure; always upload environment+logs; fail the job only after artifact creation.
6. **Aggregate and regress** — validate every result against JSON schema; reject duplicate natural keys; merge rank samples and summaries; compute trial aggregates; compare against the most recent compatible baseline; publish a step summary; upload one `results_collectivex` bundle.
7. **Dispatch ingestion** (only once promoted to feed the app) — repository-dispatch the InferenceX-app repo with `{ "benchmark-family": "collectivex", "run-id": "...", "run-attempt": "..." }`.

Use a separate `collectivex-changelog.yaml`: a CollectiveX backend change must not trigger the expensive serving sweep through `perf-changelog.yaml`, and a serving change must not launch every collective suite.

## Regression policy (Milestone 2+)

A compatible baseline requires exact matches on: case ID; cluster ID; topology fingerprint (or approved topology class); backend; comparison class; normal/LL mode; node and rank placement; dtype and shape; measurement-contract version. **Do not compare "same GPU SKU" across materially different fabrics.**

```text
regression if:
  correctness changed pass → fail
  OR median latency degradation exceeds max(fixed floor, cluster noise threshold)
  OR bandwidth degradation exceeds max(fixed floor, cluster noise threshold)
```

Derive each cluster's noise threshold from repeated baseline measurements via median absolute deviation — don't hard-code a universal 3% before knowing each fabric's noise. Retain failed, timed-out, and invalid results; reliability is part of the benchmark.

## Reporting, database, and frontend

**Now (spike / Milestone 2): a static, artifact-driven report.** Do not begin by changing InferenceX-app.

```bash
python -m collectivex.report --results output/aggregate.json --output output/report/
```

```text
report/
├── index.html
├── data.json
├── assets/
└── runs/
    └── <case-id>.html
```

Report views: **Overview** (supported clusters/backends, latest run, correctness failures, recent regressions, coverage matrix); **Primitive explorer** (latency / algbw / busbw / rank-spread vs payload size; single-node vs multinode); **MoE explorer** (dispatch & combine latency vs tokens/rank; tokens/s vs EP size; uniform vs skewed; normal vs LL; comm-SMs vs performance); **Case details** (exact shape, backend commit, container digest, topology fingerprint, environment, command, correctness report, rank-level distribution, raw logs). A **comparison warning** must visibly reject invalid comparisons:

```text
Not directly comparable:
- different routing distribution
- different topology class
- different communication-SM budget
- standardized versus backend-optimized mode
```

**Later (Milestone 4 / promotion into InferenceX-app):** add `/collectives` to the app (Next.js, React Query, raw API rows, client-side transforms, D3 charts; tab metadata/routing are centralized). Avoid a single global "CollectiveX score" at launch. Port the report views, plus Library Comparison, Scale-and-topology, and Historical-regression views, and a run-detail drawer. The frontend computes the `comparison-key` and refuses to connect rows with differing keys by default — **this guard matters more than any individual chart.**

API routes (app):

```text
/api/v1/collectives
/api/v1/collectives/availability
/api/v1/collectives/history
/api/v1/collectives/runs/:id
/api/v1/collectives/artifacts/:id
```

Continue the app convention: API returns raw DB rows; the frontend does chart-specific transforms.

**Database (app, later).** Do not put CollectiveX rows in `benchmark_results` (its identity is serving configs + ISL/OSL/concurrency). Reuse `workflow_runs`, then add:

```sql
collective_workloads(id, case_id, schema_version, family, operation, shape jsonb)
collective_environments(id, cluster_id, hardware, topology_class, topology_hash, software jsonb, capabilities jsonb)
collective_configs(id, workload_id, environment_id, backend, backend_version, comparison_class, mode, nodes, gpus_per_node, world_size, settings jsonb)
collective_results(id, workflow_run_id, config_id, trial, date, status, metrics jsonb,
                   latency_p50_us, latency_p99_us, logical_bandwidth_gbps, bus_bandwidth_gbps,
                   tokens_per_second, rank_skew_pct, error)
collective_artifacts(result_id, artifact_type, storage_url, metadata jsonb)
collective_availability(date, hardware, cluster_id, backend, family, operation, mode)
```

Follow the app's hybrid design (JSONB for evolving metrics; indexed "hot" columns for common filters; idempotent ingestion; natural unique keys; denormalized date; latest-results materialized view). Keep raw per-rank samples in artifacts/object storage, not in Postgres.

## Future expansions

The spike de-risks the path to the actual deliverable — a public OSS collective benchmark and an explainer article. Expansion axes, roughly near → far, with dependencies:

**Hardware breadth.** B300 / GB300 next (GB300 is also MNNVL, with known disagg KV-transfer wins) → H100 / H200 as a cheaper, more-available **InfiniBand baseline** ideal for characterizing per-fabric noise → AMD MI300X / MI325X / MI355X (this is Milestone 1) → TPU (far; a separate stack and toolchain).

**Backend breadth.** Framework-native EP (the `framework-integrated` class — ties numbers back to the SGLang/vLLM serving paths) → MSCCL++, NVSHMEM / rocSHMEM microbenchmarks, AITER comm/fusion, MoRI / Pollara (AMD).

**IBGDA low-latency mode.** The recurring strategic blocker and the original "LL is broken" story; gated on the NVIDIA SRE maintenance window for B200/B300. Highest narrative value — add as an experimental suite the moment it unblocks.

**Scale-out.** 2 → 4 → 8 → 16 nodes; on GB200, intra-NVL72 vs cross-rack scaling-efficiency curves (where MNNVL ends and the inter-rack fabric begins).

**L3 integrated operator path.** route → permute → dispatch → grouped-GEMM → combine → unpermute — the bridge to OperatorX.

**L4 e2e correlation.** Link an isolated dispatch/combine number to the same shape's cost inside a real serving run via `profile.yml` traces — the "explain serving performance" payoff and the tie-back to the core product.

**Trace-derived shapes (Milestone 3).** DeepSeek V4 / MiniMax M3 / Kimi token-histogram and routing-skew extraction, so the synthetic shapes are anchored to real workloads.

**AMD Ultra Ethernet (UEC).** The AMD networking path; pairs with the MoRI / Pollara backends.

**Productization (north star).** Static report → public OSS benchmark site + the explainer article; promotion into InferenceX-app (`/collectives` + Postgres + nightly suite + regression alerts) at Milestone 2 / 4.

## Continuous benchmark — vision & scope

Goal: a continuous benchmark that reproduces the spike automatically and grows into a credible cross-vendor EP/collective comparison. **Start with balanced DeepSeek shapes, intranode EP**, then venture to advanced cases. Target **≥1 EP library per platform** first — DeepEP on NVIDIA, MoRI on AMD.

### EP library landscape
- MoRI (AMD) — https://github.com/ROCm/mori
- DeepEP / DeepEPv2 / Hybrid-EP — https://github.com/deepseek-ai/DeepEP (hybrid: https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep)
- NVIDIA NCCL EP — https://github.com/NVIDIA/nccl/tree/master/contrib/nccl_ep
- UCCL — https://github.com/uccl-project/uccl
- NVLink One-Sided AllToAll EP (mainly NVL72) — TensorRT-LLM blog18 (Optimizing MoE Communication with One-Sided AllToAll over NVLink)
- NIXL EP — https://github.com/ai-dynamo/nixl/tree/main/examples/device/ep

### Shapes & axes
- **Classic DeepSeek V3:** hidden 7168, top-8, 256 routable experts.
- **Prefill vs decode** (# tokens).
- **Normal EP vs low-latency (LL) EP.**
- **Dispatch precision:** NVFP4, MXFP4, MXFP8, BF16.
- **Combine precision:** MXFP8, direct-cast FP8, BF16, NVFP4 — see MoRI #311, flashinfer #3643 / #3376.
- **Balanced vs unbalanced vs EPLB.**
- **Realistic shapes from InferenceX models** — collect hidden sizes / routing (Qwen3.5 has an unusual top-k).

### Other inference collectives (later)
- KV-cache transfer: MoRI-IO, NIXL, Mooncake; CPU↔GPU offload — `experimental/kvcache_transfer_DtoH_HtoD/benchmark.py`.
- Low-latency one-shot / two-shot all-reduce (SGLang & vLLM in-tree kernels + AITER / FlashInfer variants) — e.g. sglang `sgl-kernel/csrc/allreduce/quick_all_reduce.cuh`.

### Reference benchmark scripts to draw from
- flashinfer PR #3000; ROCm/mori `tests/python/ops`; DeepEP `tests/legacy`.

### Learning resources
- arXiv 2511.15076, 2603.13606, 2512.19849, 2412.19437.

## Things not to do

* Do not add collective fields to the existing serving matrix.
* Do not make one GitHub Actions job per payload size.
* Do not call all logical-bandwidth figures "bus bandwidth."
* Do not compare different topology fingerprints as though GPU SKU were sufficient.
* Do not silently discard failed or incorrect results.
* Do not let a backend choose undocumented tuning parameters (in `standardized` mode).
* Do not make low-latency mode the only reported result.
* Do not publish one overall ranking before coverage and comparison contracts are stable.
* Do not start with every EP library, TPU, UEC, and every model shape.
* Do not store full raw rank samples indefinitely in Postgres.
* Do not expose internal hostnames, paths, NIC GUIDs, IP addresses, or private image references in public artifacts.
* Do not freeze the schema before the spike has produced a real artifact to freeze it from.

## References (verified against the live InferenceX repo)

- `experimental/README.md` — the non-core / "not official results" charter this project lives under.
- `.github/configs/runners.yaml` — runner labels and exact names (H100…GB300, AMD MI3xx).
- `.github/workflows/benchmark-tmpl.yml`, `benchmark-multinode-tmpl.yml`, `profile.yml`, `speedbench-al.yml` — the `bash ./runners/launch_${RUNNER_NAME%%_*}.sh` selection convention.
- `runners/launch_*.sh` — existing per-cluster launchers (`launch_b200-dgxc.sh`, `launch_b200-dgxc-slurm.sh`, `launch_gb200-nv.sh`, `launch_mi355x-amds.sh`, …).
- `utils/matrix_logic/generate_sweep_configs.py`, `validation.py` — the serving matrix CollectiveX must **not** extend.
- `.github/workflows/e2e-tests.yml`, `collect-results.yml` — the validate → fan-out → collect control plane being reused.
- `perf-changelog.yaml` — the additions-only serving gate CollectiveX must **not** trigger.
- NVIDIA Magnum IO NVSHMEM + GPUDirect Async (IBGDA): `https://developer.nvidia.com/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-io-nvshmem-and-gpudirect-async/`
