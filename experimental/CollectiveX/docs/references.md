# CollectiveX — learning / resource notes

> Status: experimental (goal P2, "Add learning/resource notes"). These four arXiv papers are the
> learning resources listed in `plan.md`. Each summary below was fetched from `arxiv.org/abs/<id>`
> (titles/authors/dates taken from the live abstract page) and is then **mapped to the specific
> CollectiveX benchmark dimensions it informs** — the metric, contract, capability axis, or
> comparison the paper bears on.

**Retrieval status (fetched 2026-06):**

| arXiv ID | Title | Retrieved? | Note |
|---|---|---|---|
| 2511.15076 | GPU-Initiated Networking for NCCL | yes | clean fetch |
| 2603.13606 | NCCL EP: Towards a Unified Expert Parallel Communication API for NCCL | yes | **ID looked future-dated (year "26"); verify.** The page resolved to real content (submitted 13 Mar 2026 per the page), not a not-found error — recorded as retrieved, flagged for a sanity check of the ID/date before citing. |
| 2512.19849 | UCCL-EP: Portable Expert-Parallel Communication | yes | clean fetch |
| 2412.19437 | DeepSeek-V3 Technical Report | yes | clean fetch |

All four resolved to genuine abstract pages. 2603.13606 is the only one flagged: its identifier
(and the page's stated 13 March 2026 submission date) is forward-dated relative to when it was
assigned in the plan, so although the fetch returned coherent NCCL-EP content, the ID should be
double-checked against arXiv directly before it is used as a hard citation. Nothing below is
fabricated; the one uncertainty is called out here.

---

## Summarize arXiv 2511.15076

**GPU-Initiated Networking for NCCL** — Hamidouche, Bachan, Markthub, Gootzen, Agostini, Jeaugey,
Shafi, Theodorakis, Gorentla Venkata (NVIDIA). Submitted 19 Nov 2025 (v2 24 Nov 2025).

Describes NCCL 2.28's new **Device API**, focused on the **GPU-Initiated Networking (GIN)**
component for network RDMA. The motivation is fine-grained, low-latency GPU-to-GPU communication
for tightly coupled compute-communication workloads — explicitly Mixture-of-Experts — where the
traditional host-initiated model's CPU coordination is overhead. GIN is a three-layer architecture:
host-side setup APIs, device-side remote-memory operations callable from inside CUDA kernels, and a
network plugin with dual semantics (GPUDirect Async Kernel-Initiated and a Proxy backend). The paper
demonstrates GIN by integrating it with **DeepEP** and reports benchmark results, positioning GIN as
combining low-latency device-initiated ops with NCCL's collective algorithms and production
infrastructure.

## Summarize arXiv 2603.13606

> **Flagged ID — see retrieval table.** The arXiv identifier is forward-dated; the fetch returned
> the content below (an NCCL-EP paper), but verify the ID/date before citing as authoritative.

**NCCL EP: Towards a Unified Expert Parallel Communication API for NCCL** — Goldman, Boker,
Sheraizin, Admoni, Polyakov, Bhattacharya, Yu, Sun, Theodorakis, Yin, Gootzen, Shafi, Ravid,
Di Girolamo, Dinan, Li, Gorentla Venkata, Bloch (NVIDIA). Page states submitted 13 Mar 2026
(v3 2 Apr 2026); 13 pages, 8 figures, 7 tables; cs.DC.

Introduces **NCCL EP**, an MoE communication library built on NCCL's Device API (the GIN work
above), offering unified `ncclEpDispatch` / `ncclEpCombine` primitives with **C and Python**
interfaces. It has two modes: a **Low-Latency (LL)** mode for inference decode targeting small
batches (the page quotes "1–128 tokens") over all-to-all RDMA+NVLink, and a **High-Throughput (HT)**
mode for training and inference prefill targeting large batches ("4096+ tokens") using hierarchical
communication that aggregates within NVLink domains before inter-node RDMA. It situates itself
alongside DeepEP and Hybrid-EP, evaluates on an H100 cluster across multi-node configs (LL kernel
results + end-to-end with vLLM), and aims to be a supported EP path on current and emerging NVIDIA
platforms.

## Summarize arXiv 2512.19849

**UCCL-EP: Portable Expert-Parallel Communication** — Mao, Zhang, Cui, Huang, You, Chen, Xu, Gu,
Shenker, Raiciu, Zhou, Stoica. Submitted 22 Dec 2025 (v2 22 Jan 2026).

Targets the **portability** problem in EP: systems like DeepEP perform well but require tight
GPU↔NIC coupling for GPU-initiated RDMA, so they don't run everywhere. **UCCL-EP** instead routes
compact token commands through a **GPU–CPU control channel** where multithreaded CPU proxies issue
the RDMA operations, and it **emulates ordering semantics using RDMA immediate data** for NICs that
lack native support (e.g. AWS EFA). Implemented on **both NVIDIA and AMD** GPUs with EFA and
Broadcom NICs, it reports up to **2.1× dispatch/combine throughput on EFA**, up to **40% higher
SGLang token throughput**, and up to **45% higher DeepSeek-V3 training throughput on a 16-node
AMD+Broadcom platform**.

## Summarize arXiv 2412.19437

**DeepSeek-V3 Technical Report** — DeepSeek-AI et al. (~200 authors). Submitted 27 Dec 2024
(v2 18 Feb 2025).

Describes **DeepSeek-V3**, a **Mixture-of-Experts** LLM with **671B total / 37B activated per
token**, using **Multi-head Latent Attention (MLA)** and **DeepSeekMoE**, an **auxiliary-loss-free
load-balancing** strategy, and a **multi-token-prediction** objective. Pre-trained on 14.8T tokens
then SFT + RL; reported comparable to leading closed-source models at **2.788M H800 GPU-hours**, with
stable training (no irrecoverable loss spikes / rollbacks) and public checkpoints. For CollectiveX
the load-bearing details are the **MoE shape and the load-balancing approach**, not the end-to-end
quality numbers.

---

## Map each paper to CollectiveX benchmark dimensions

Each paper informs specific, concrete axes of the harness (`tests/ep_harness.py`,
`tests/ep_deepep.py`, `configs/backends.yaml`, `schemas/ep-result-v4.schema.json`). The mapping:

### 2511.15076 (GIN / NCCL Device API) → the DeepEP **kernel-generation axis** and the **runtime-visible** boundary
- **`shape.kernel_gen` (v1 NVSHMEM vs v2 NCCL-GIN).** The harness already records DeepEP's kernel
  generation as part of line identity (`kernel_gen` derived from `deepep_version`, folded into
  `comparison_key`) precisely because DeepEP V2 moved its transport from NVSHMEM to the NCCL Device
  API. This paper *is* the NCCL device-side RDMA (GIN) that the V2 path builds on — it is the
  primary-source explanation for why a "DeepEPv2" run must never be conflated with a "DeepEP V1" run
  (goal P1, "DeepEP version matrix"). Informs the `kernel_gen` field and the version-as-first-class-
  axis requirement.
- **`runtime-visible-v1` measurement contract.** GIN's thesis is removing CPU coordination so comm
  is launched/issued from inside the kernel. That is exactly the cost-surface `runtime-visible-v1`
  tries to capture (cast + layout + comm + recv-dequant inside the timed window). The paper
  motivates why a serving-realistic boundary, not just comm-only, is worth measuring.
- **`transport` axis** (`nvlink`/`mnnvl`/`rdma` in `backends.yaml`) — GIN is the RDMA device-path
  whose latency the EP transports record.

### 2603.13606 (NCCL EP) → the planned **NVIDIA NCCL EP adapter**, the **dispatch/combine API contract**, and **phase = decode/prefill**
- **The open "NVIDIA NCCL EP" backend** (goal P1: *"Add adapter for `NVIDIA/nccl/contrib/nccl_ep`"*)
  — this paper is the design of that very library (`ncclEpDispatch` / `ncclEpCombine`). It is the
  reference for adding an `nccl-ep` entry to `configs/backends.yaml` and a third adapter beside
  DeepEP and MoRI, to be compared against DeepEP normal/LL under `layout-and-dispatch-v1`.
- **`mode` axis (normal vs ll) and `phase` (decode vs prefill).** NCCL EP's split into **LL
  (1–128 tokens, decode)** and **HT (4096+ tokens, prefill/training)** lines up directly with the
  harness's `DECODE_LADDER = [1..128]` / `PREFILL_LADDER = [128..4096]` and the `mode = ll|normal`
  axis. It corroborates the decode/prefill token-regime modelling and the LL decode cap.
- **`comparison_key` design.** NCCL EP, DeepEP, and Hybrid-EP being distinct libraries with the same
  `dispatch`/`combine` surface is exactly the situation the `backend` field + provenance
  (`backend name, fork, commit, API generation`) exist to disambiguate.

### 2512.19849 (UCCL-EP) → **cross-vendor portability**, the planned **UCCL adapter**, and the **transport / resource axes**
- **The open "UCCL EP" backend** (goal P1: *"Add UCCL backend adapter … Add cross-platform result
  class"*) — this paper is that backend. It is the reference for a UCCL `backends.yaml` entry and a
  capability declaration spanning **both NVIDIA and AMD** (the only paper here that is natively
  cross-vendor, like CollectiveX itself).
- **The whole cross-vendor comparison thesis.** UCCL-EP exists because DeepEP's GPU↔NIC coupling
  isn't portable. CollectiveX's reason for being is comparing such EP libraries fairly *across
  vendors* — and its mechanism (one deterministic shared routing trace, `layout-and-dispatch-v1` as
  the common contract, topology-class in the `comparison_key` so NVIDIA and AMD are never silently
  overlaid) is the apparatus needed to evaluate exactly this paper's portability-vs-performance
  trade-off.
- **`transport` axis + the CPU-proxy resource story.** UCCL-EP's CPU-proxy / RDMA-immediate-data
  design adds transports (EFA, Broadcom) beyond `nvlink/xgmi`, and its CPU-side issue model is a
  data point for the `resource_profile` vocabulary (comm units / where the work runs), which today
  models SM/CU fractions.

### 2412.19437 (DeepSeek-V3) → the **default benchmark shape**, **EPLB / routing-skew axis**, and **fp8 dispatch**
- **The headline shape itself.** The harness defaults — `hidden = 7168`, `topk = 8`,
  `experts = 256` (`add_common_args`), and the goal's "Default to DeepSeek V3 shape / EP8 / uniform
  / BF16" — *are* DeepSeek-V3's MoE configuration. This paper is the source of the canonical shape
  every official curve is reported at, and of the `deepseek-v3-v1` / `deepseek-v4-v1` workload
  manifests (goal P1).
- **EPLB and the routing-distribution axis.** DeepSeek-V3's **auxiliary-loss-free load balancing**
  is the real-world counterpart to (a) the `--routing` skew distributions (`zipf*`, `hotspot-*`) the
  harness stresses and (b) the **EPLB** expert-replication transform (`tests/eplb.py`,
  `--eplb`/`--num-redundant-experts`) offered as the remedy for skew. The paper motivates *why*
  load imbalance and its mitigation are first-class benchmark dimensions (`expert_load_cv`,
  `rank_load_cv`, `hotspot_ratio`, the EPLB `imbalance_before/after` + `mapping_hash`).
- **fp8 throughout.** DeepSeek-V3's fp8 training/inference underpins the `dispatch_dtype = fp8`
  axis and the per-token block-128 fp8 scale convention in `ep_deepep.py`.
- **Per-token activation rate.** "37B activated per token" is the MoE sparsity that makes
  tokens-per-rank (not model size) the meaningful x-axis for a dispatch/combine micro-benchmark.
