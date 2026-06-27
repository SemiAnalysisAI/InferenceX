# CollectiveX EP benchmark — methodology mapping

> Status: experimental (goal P2, "Methodology/reference docs"). This document explains
> what the CollectiveX EP dispatch/combine harness reused from upstream test code, what it
> deliberately changed, and the exact contracts a result must satisfy to be published. It is
> grounded in the code as it stands: `tests/ep_harness.py`, `tests/ep_deepep.py`,
> `tests/ep_mori.py`, `tests/reference_ep.py`, `tests/run_ep.py`, `validate_results.py`, and
> `schemas/ep-result-v4.schema.json`. Where a claim cannot be verified from the repo it is
> flagged inline rather than asserted.

The shared design constraint behind everything below is the *fair-comparison contract* stated at
the top of `ep_harness.py`: a single deterministic routing trace is generated once from a fixed
seed over the **global** batch and is identical on every SKU; each rank materializes only its
slice (`routing.rank_slice` / the `my_off:my_off+my_cnt` slice in `run_sweep`). Adapters never
roll their own RNG. So "what was reused vs changed" always means: *reused the library's API call,
changed the workload and the timing boundary so every backend runs the same problem under a named,
machine-checkable measurement contract.*

---

## DeepEP tests/legacy: what was reused

The DeepEP adapter (`tests/ep_deepep.py`) reuses DeepEP's **documented normal-mode and
low-latency Python API directly**, the same surface its own intranode/internode test code drives:

- **The buffer + dispatch/combine call sequence.** Normal mode constructs a single
  `deep_ep.Buffer(group, num_nvl_bytes, 0)`, calls `buffer.get_dispatch_layout(topk_idx, experts)`,
  then `buffer.dispatch(...)` and `buffer.combine(...)`. Low-latency mode uses
  `Buffer(..., low_latency_mode=True, num_qps_per_rank=…)`, `low_latency_dispatch`, and
  `low_latency_combine`. These are DeepEP's own entrypoints, not reimplementations.
- **The correctness identity from DeepEP's intranode test.** A pure dispatch→combine round trip
  with *no expert compute* reconstructs `x` scaled by the number of destination ranks each token
  was sent to. The adapter's `expected()` encodes exactly this: `ref * ranks_per_token`, where
  `ranks_per_token = is_token_in_rank.sum(dim=1)` (see the module docstring and `expected()`).
  This is the same invariant DeepEP's `test_intranode` relies on.
- **DeepEP's own comm-only timing boundary** is preserved as one of the offered contracts:
  `cached-layout-comm-only-v1` hoists `get_dispatch_layout` out of the timed region (computed once
  in `make_problem`, stored on `p.layout`), so the timed `dispatch()` is pure communication —
  matching the boundary DeepEP's own benchmark uses.
- **The fp8 per-token block-128 cast convention.** `deep_ep` 1.2.x ships no helper for this (its
  `utils` is empty), so `_per_token_cast_to_fp8` / `_per_block_dequant` implement the exact
  convention DeepEP's kernels expect (scales `[T, H//128]` float32, e4m3, `448.0` as e4m3 max).
  This is faithful reuse of the kernel's data contract, not a new scheme.
- **The LL QP convention** (one QP per local expert: `num_qps = experts // world_size`) and the
  fixed `num_max_dispatch_tokens_per_rank` decode shape follow DeepEP's LL usage.

## DeepEP tests/legacy: what was changed

- **Workload: synthetic per-rank uniform random routing → one deterministic global trace.**
  DeepEP's tests generate routing per rank locally. CollectiveX generates the routing **once over
  the global batch** from a fixed seed (`routing.build_global_routing`) and hands each rank its
  slice via `make_problem`, so DeepEP and MoRI provably run the *same* routed problem
  (`make_problem` does no RNG — see the docstring: "materializes the harness-provided rank slice").
- **Workload axes DeepEP's test does not sweep.** The harness drives a tokens-per-rank ladder
  (decode `1..128`, prefill `128..4096`), and adds routing-distribution control (`uniform`,
  `zipf*`, `hotspot-*`, `alternating-groups`, `balanced*`), temporal snapshots (`--routing-step`),
  uneven per-rank source-token allocation (`--uneven-tokens`), EPLB replication
  (`tests/eplb.py`), and structured placement metadata. None of these exist in the upstream test.
- **Timing boundary made explicit and named.** DeepEP's bench implicitly measures comm-only;
  CollectiveX requires the adapter to *declare* `SUPPORTED_CONTRACTS` and conform to whichever the
  run requests — `layout-and-dispatch-v1` (layout timed *inside* dispatch),
  `cached-layout-comm-only-v1` (DeepEP's own boundary), or `runtime-visible-v1` (fp8 cast +
  recv-dequant moved *inside* the timed window). `run_ep.py` rejects an unsupported contract
  rather than letting the backend silently pick one.
- **Statistics.** Instead of a single timed loop, the harness pools `iters × trials`
  (default `200 × 3 = 600`) samples with per-trial token-order shuffling, reduces **cross-rank MAX
  per iteration before percentiling** (`median_i(max_r)`, not `max_r(median_i)`), and reports
  p50/p90/p95/p99 with p99 as the headline. It also adds a separately *measured* round trip
  (dispatch→stage→combine in one timed region) distinct from the `isolated_sum` of the two medians.
- **Correctness oracle is independent.** DeepEP's test validates DeepEP against DeepEP's own
  expected formula; CollectiveX additionally carries a backend-free oracle (`reference_ep.py`,
  see below) so correctness is not "backend vs itself."
- **Resource normalization.** The adapter can be restricted to a device-SM *fraction*
  (`set_num_sms(round(sm_fraction · device_sms))`) so DeepEP and MoRI run at a comparable comm-unit
  budget — an axis the upstream test does not model.

> Note on "DeepEP `tests/legacy`": the plan references upstream DeepEP `tests/legacy` and a
> "DeepEP legacy test parity" item (goal P1, still open). The current adapter follows DeepEP's
> *documented normal/LL API*; a dedicated `tests/legacy` parity adapter is not yet implemented in
> this repo, so claims here describe the API surface reuse, not a line-for-line legacy port.

---

## MoRI tests/python/ops: what was reused

The MoRI adapter (`tests/ep_mori.py`) follows the upstream `ROCm/mori` `tests`/`examples`
dispatch+combine path:

- **The op construction and call sequence.** It builds `mori.ops.EpDispatchCombineConfig(...)` and
  `mori.ops.EpDispatchCombineOp(config)`, then calls `op.dispatch(x, weights, scales, indices, …)`
  and `op.combine(...)` — MoRI's own ops, with `block_num` / `warp_per_block` launch parameters as
  in its examples.
- **The shmem bring-up.** It registers the torch process group as `"default"` and calls
  `mori.shmem.shmem_torch_process_group_init("default")`, mirroring MoRI's reference test setup
  (`cpu:gloo,cuda:nccl` group with an explicit `device_id`, set up in `run_ep.py`).
- **The zero-copy registered-combine-input buffer path.**
  `op.get_registered_combine_input_buffer(...)` is filled in `stage()` — the same zero-copy path
  the upstream example uses to place "expert outputs" before combine.
- **The combine correctness identity.** MoRI's combine sums one copy per destination **rank**, so
  with no expert compute `combined[i] ≈ x[i] × (#unique destination ranks among the token's topk
  experts)`. `expected()` computes exactly this (`unique_pes` per token). This is the upstream
  example's `expected = input × #unique-destination-ranks` reused verbatim in intent.
- **int32 expert ids / the scale-tensor shape.** MoRI expects int32 indices and a real `(T, 0)`
  fp8 scale tensor (because `scale_dim == 0`); the adapter honors both.

## MoRI tests/python/ops: what was changed

- **Workload: always-uniform → the shared global trace.** The reference test routes uniformly.
  The adapter's `make_problem` now materializes the **harness-provided** rank slice, so MoRI honors
  the requested routing distribution and runs the identical workload to the NVIDIA SKUs (docstring:
  "it no longer always-uniform").
- **Heap held at 2 GiB instead of the reference's hardcoded 6 GiB.** MoRI registers the *entire*
  symmetric heap as one RDMA MR at init. On the MI355X ionic_rdma NICs a 6 GiB MR fails
  (`RegisterRdmaMemoryRegion … EINVAL`); 2 GiB registers. The adapter sets
  `MORI_SHMEM_HEAP_SIZE` (default `2G`) **before** `import mori`. The reference's 6 GiB is "exactly
  why it can't run as-is here" (CONTAINERS.md).
- **Bounded `max_num_inp_token_per_rank` → a real `buffer_cap`.** Capped at 512 tokens/rank at
  hidden 7168 so dispatch/combine buffers fit the 2 GiB heap. The harness clamps the ladder to this
  cap and **reports dropped points** rather than silently truncating (`token_ladder` returns
  `dropped`).
- **`combine_needs_redispatch = True`.** MoRI's `combine()` resets `recv_num`, so `total_recv`
  must be read **before** combine, and the harness re-dispatches (untimed) before *each* timed
  combine sample (`time_us(..., pre=prep)`). DeepEP reuses its handle, so it sets this `False`.
- **Gradual cold-start ramp.** MoRI wedges on a cold dispatch that jumps straight to a large T, so
  `needs_gradual_ramp = True` makes the harness approach max-T via a geometric ramp from 1 and
  *not* shuffle token order. It also opts out of the Blackwell warm-burst (`wants_warm_burst =
  False`) because a sustained burst wedges it.
- **Hard-exit teardown.** MoRI's post-`shmem_finalize()` teardown asserts (`CheckStatusValid` →
  SIGABRT). The adapter's `finalize()` flushes results and `os._exit()`s past it instead of
  returning cleanly the way DeepEP does.
- **Contract restriction.** MoRI computes its routing layout **inside** the dispatch kernel and it
  cannot be hoisted, so it declares only `layout-and-dispatch-v1`. This is *why* cross-vendor
  comparisons must use `layout-and-dispatch-v1` — it is the one contract both backends can honor.
- **Resource budget floored, not normalized down.** MoRI deadlocks at T≥32 when `block_num` is
  reduced to the normalized target (validated: 46 wedges, 80 completes), so the adapter floors
  `block_num` at a functional minimum and **records that the target fraction was not reached**
  (`block_num_floored = True`, `tuned_source = "normalized-floored"`). The harness reads this and
  marks the result resource-nonconforming → demoted to `diagnostic` (see publication contract).

> Note on the exact upstream path name: CONTAINERS.md and the plan refer to `ROCm/mori`
> `tests`/`examples` and `tests/python/ops`. The adapter reproduces that dispatch+combine path's
> API and expected-value formula; the precise upstream file/commit is captured at runtime via
> `MORI_COMMIT` (else the image tag) into provenance rather than pinned in this doc.

---

## FlashInfer PR 3000 benchmark inspiration

The project plan lists, under "Reference benchmark scripts to draw from": *"flashinfer PR #3000;
ROCm/mori `tests/python/ops`; DeepEP `tests/legacy`."* (`plan.md`). FlashInfer PR #3000 is named
there as **methodological inspiration for the EP dispatch/combine benchmark shape** — i.e. one of
the reference benchmark scripts whose structure informed how CollectiveX measures a single MoE
dispatch+combine pair — alongside the MoRI and DeepEP test code described above.

**What is verifiable from this repo:** PR #3000 is cited only as a reference script in `plan.md`.
There is no FlashInfer adapter, import, or copied benchmark code in the tree today (a "FlashInfer
EP paths" item remains open in goal.md P1, and FlashInfer is otherwise referenced only for combine
precision via PRs #3643 / #3376). 

**What this doc does not assert:** I have **not** independently verified the contents of FlashInfer
PR #3000 (its exact title, the kernel it benchmarks, or which specific measurement choices were
borrowed) against the FlashInfer repository — that verification is outside what the CollectiveX
codebase contains, and the PR number is recorded here as-cited. Treat the specific influence as
"named as inspiration in the plan," not as a line-level provenance claim. If precise attribution is
needed, confirm against `flashinfer-ai/flashinfer` PR #3000 directly before publishing.

What CollectiveX's EP methodology demonstrably shares with a good EP micro-benchmark (whatever its
origin): dispatch and combine are timed **separately**, each point is **one MoE layer / one step /
one dispatch+combine collective pair** (not a whole model), the token-count is the swept x-axis,
and percentiles come from many pooled iterations rather than a single timed loop.

---

## Why CollectiveX timing boundaries differ

DeepEP's and MoRI's own benchmarks each measure *their* natural boundary, which makes their numbers
non-comparable: DeepEP can hoist layout computation out of the timed region; MoRI computes layout
*inside* its kernel and cannot. If each backend simply reported "dispatch latency" under its own
convention, a DeepEP comm-only number would be compared against a MoRI layout-and-dispatch number
as if they measured the same thing. CollectiveX therefore makes the boundary an **explicit, named,
machine-checked contract** (review #3 in `ep_harness.py`): adapters declare `SUPPORTED_CONTRACTS`
and `run_ep.py` rejects an unsupported request. There are three contracts.

### `layout-and-dispatch-v1` — the cross-vendor common boundary
Dispatch timing **includes** routing-layout generation. For DeepEP, `get_dispatch_layout` runs
*inside* the timed `dispatch()` (`p.layout is None`). For MoRI, layout is computed inside the
kernel and **cannot** be hoisted — so this is *the only contract MoRI can honor*, and hence the one
both vendors share. The fp8 cast/dequant stays **outside** the timed window (cast in
`make_problem`, dequant in `stage`), modelling a producer that hands the dispatcher already-quantized
activations. **Use this for any DeepEP-vs-MoRI comparison.**

### `cached-layout-comm-only-v1` — DeepEP's own boundary (DeepEP only, normal mode)
Layout is computed **once, untimed** (in `make_problem`, stored on `p.layout`) so the timed
`dispatch()` is **pure communication**. This reproduces DeepEP's own benchmark boundary and is
useful for "how fast is the comm kernel alone," but it is **not** comparable to MoRI (which can't
hoist layout) and is rejected for LL mode (low-latency dispatch computes layout internally —
nothing to hoist; `run_ep.py` rejects this combo).

### `runtime-visible-v1` — the serving-realistic boundary (DeepEP only today)
Dispatch starts from **what the runtime has right after routing** and **includes everything needed
to make expert input consumable**: the per-token block-128 **fp8 cast moves inside** the timed
window, plus layout, comm, and the recv-side **dequant to bf16** (`_per_block_dequant` inside
`dispatch()`, after which `stage()` no-ops). Combine starts from bf16 expert outputs and ends when
token outputs are consumable. This answers "what does the serving path actually pay," and the
adapter records the boundary honestly via `fp8_in_timing` (true only under this contract for fp8).
LL is runtime-visible *by construction* (its single kernel already times cast+layout+comm), so the
flag only changes normal mode.

### Boundaries shared across all three
- **Combine excludes staging in every contract.** Placement of expert outputs (`stage()`) is
  untimed for every backend — it stands in for the expert FFN write, which is not part of the
  collective being measured.
- **`isolated_sum` is a diagnostic, not a measurement.** It is the arithmetic SUM of the isolated
  dispatch and combine percentiles. It **cannot** reveal shared sync, launch amortization, or
  dispatch/combine overlap, so it must not be used for throughput or SLO capacity. The **measured
  round trip** (`roundtrip`, one timed region over dispatch→stage→combine) is the real chained
  latency, and it is the only basis for `roundtrip_tokens_per_second`.
- **Cross-rank reduction order.** A collective finishes with its slowest rank, so each iteration's
  latency is reduced **MAX across ranks first**, then percentiled.

The contract name is part of the `comparison_key` and the schema enum, so two rows under different
contracts are labelled distinct and never silently overlaid.

---

## Correctness contract definition

"Correct" in CollectiveX has two layers: the **independent oracle** that defines the semantics, and
the **runtime gate** that every sweep point must pass.

### The independent oracle (`tests/reference_ep.py`)
A from-scratch numpy model of MoE dispatch + combine, written **without** DeepEP or MoRI, used only
for untimed validation — so the benchmark is never "validated against itself." Its model:

- **Layout:** expert `e` lives on rank `e // experts_per_rank`.
- **Dispatch:** token `t` selected for expert `e` contributes one copy of `x[t]` to
  `(rank e//epr, expert e)`. `dispatch_plan()` enumerates every routed copy exactly once and
  `validate_dispatch()` asserts each `(token, selected-expert)` maps to the **correct rank and
  expert, exactly once** (duplicate `(token,expert)` pairs and out-of-range ranks are errors).
- **Expert transform:** a deterministic per-expert factor `f_e = 1 + e/E`, **distinct per expert**,
  so a copy routed to the *wrong* expert produces a wrong value (identity would hide mis-routing —
  the self-test corrupts one expert id and asserts the oracle output changes).
- **Combine:** `y[t] = Σ_k weights[t,k] · f_e(x[t])`, reduced over the token's selected experts,
  output in **source-token order**. `validate_combine()` recomputes this two independent ways
  (vectorizable reduction vs explicit per-copy accumulation) and asserts they agree — exercising
  the reduction, the **gate-weighting**, the **source ordering**, and the
  **multiple-experts-on-one-rank** case.
- **Edge cases** (goal P3): empty rank, repeated destination rank, single-rank hotspot (all topk on
  rank 0) are covered in the self-test; non-divisible global token counts are handled by callers.

So the oracle's definition of correct is **exact destination rank/expert/token mapping (each routed
copy once), plus the combine reduction with correct gate weights in correct source order.**

### The runtime gate (in `ep_harness.run_sweep`)
Per ladder point, each backend's `combine` output is compared to its `expected()` reference
(DeepEP: `x · #destination-ranks`; MoRI: `x · #unique-destination-ranks`). The gate computes
`max_rel = max_abs_error / max|expected|` and passes the point when `max_rel < tolerance`
(bf16 `5e-2`; fp8 `1.25e-1`, looser because e4m3's 3 mantissa bits cap round-trip error — the
tolerance is **recorded in the artifact** so the looser fp8 gate is explicit). A point is `correct`
only if the local gate passes on **every** rank (MIN-reduced `local_ok`) **and** non-zero tokens
were actually received (`recv_total > 0`) — so a silent no-op cannot pass.

The artifact is honest about scope: `correctness.scope = "roundtrip-reconstruction-smoke-v1"` — it
is a round-trip reconstruction plus non-silent-recv check at runtime, **not** a full per-token
routing/ordering/padding proof at runtime (that exhaustive proof is what `reference_ep.py` provides
off the hot path).

### Workload identity (part of "did everyone run the same correct thing")
Beyond per-point correctness, the sweep proves all ranks built the **same** global routing: each
rank hashes its per-T routing hashes into a `trace_signature` and the harness MIN/MAX-reduces it;
`workload_identity = "consistent-across-ranks"` only if all ranks agree. A mismatch means NVIDIA and
AMD did **not** run identical routing, which (see below) makes the result `invalid`.

---

## Publication contract definition

`publication_status` is **machine-derived** from a multi-dimensional `validity` record — no caller
may hand-label a result `official`. The derivation lives in `ep_harness._derive_publication_status`
and is **mirrored** in `validate_results.py:derive_publication_status`; the validator's core job is
to confirm the recorded status equals this re-derivation (a mismatch = "validity tampered or
stale", a hard error). The five tiers and their gates:

### `failed`
`execution_status != "complete"` — the sweep produced no rows. Nothing else is evaluated.

### `invalid`
Execution completed but a **fundamental soundness gate failed**: `semantic_correctness != "pass"`
(a point failed the correctness gate), **or** `measurement_conformance != "conformant"`, **or**
`workload_identity == "inconsistent"` (ranks did not run the same routing). An invalid result is
not a usable measurement of anything.

### `diagnostic`
Measurement is **sound** (correct + consistent workload + conformant contract) but it is **not a
fair cross-platform point**, for one of:
- **Resource-nonconforming** — `resource_conformance` ends in `"nonconforming"` (e.g. MoRI's
  floored `block_num`: it needed *more* comm units than the normalized target, so it isn't an
  apples-to-apples resource point). Fixed-kernel paths (DeepEP LL: `low_latency_mode`) are
  classified `not-applicable`, **not** a conformance failure, and are simply excluded from the
  resource-Pareto comparison.
- **A flagged timing anomaly** — `anomaly_free == false`. The harness flags
  `roundtrip_gt_isolated_sum` (measured RT p99 > `threshold ×` isolated-sum p99, default 3×; the
  open LL-FP8 case) and `roundtrip_lt_component_floor` (RT p50 < 0.95 × max(dispatch, combine) p50,
  which violates chained-op sync semantics). Either demotes to `diagnostic` **unless explicitly
  waived** via `--waive-anomaly` (which sets `anomaly_free = true`) *after* the cause is understood
  and documented.
- It is also the fallback for an otherwise-sound result that does not meet the higher bars.

### `comparable-experimental`
Measurement is sound (`semantic_correctness == pass`, `workload_identity` starts with
`"consistent"`, `measurement_conformance == conformant`), resource-conforming, and anomaly-free —
but it is **missing a publication requirement** (e.g. incomplete provenance, or a seeded-runtime
workload rather than a canonical serialized one). This is the normal tier for a clean development or
cross-vendor run that hasn't cleared the full official bar. It is comparable, just not "official."

### `official`
Everything `comparable-experimental` requires **plus both**:
- `provenance_complete == true` — no `"unknown"` backend provenance, **and** a non-empty image
  digest, **and** a GitHub run record with `run_id` + `source_sha` (assembled in `run_ep.py` from
  `GITHUB_*` / `COLLECTIVEX_*` env). A bare local run can never be official.
- `workload_source == "canonical-serialized"` — the run consumed pre-generated, checksum-verified
  trace bytes (`--workload-dir`, `tests/workload.py`), so it is **provably** the same workload as
  any other run consuming the same files (not just a same-seed regeneration).

`validate_results.py` enforces additional **official-grade** gates on top of the derivation: a
non-null `workload_id` and `trace_signature`, no unwaived anomalies, every point `correct`, and a
minimum of `100` pooled samples per point (`MIN_SAMPLES_OFFICIAL`). It exits non-zero if any doc
claims `official` but fails a gate, and (with `--require-official`) if any non-legacy doc is not
official.

### Cross-run identity (validator-only)
Within a `comparison_key` (further grouped by `routing_step` and `uneven_tokens`, which change the
realized workload but live in `reproduction`, not the key), the validator checks **per-T
`routing_hash` agreement**: two runs at the same config and same T but **different routing bytes**
are flagged as "not the same workload." It deliberately keys on per-T hashes (not the whole
`trace_signature`) so a capped cross-vendor sweep (e.g. `1..16`) and a full headline sweep
(`1..128`) of the same config are **not** falsely flagged — only a genuine same-T conflict is.

### Other record types the validator preserves
- **Legacy (v3, no `publication_status`)** docs load as `legacy-experimental` and are reported, not
  failed.
- **Preserved failed-case** records (`record_type == "failed-case"`, emitted by the runner on a
  wedge/timeout/crash) are reported as preserved cases, **not** validation errors — the project
  rule is "do not silently discard failed or incorrect results."
