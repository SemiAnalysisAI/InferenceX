# E2E serving correlation study — does EP microbench p99 predict tok/s?

Status: **design** (nothing measured yet). This answers the sharpest external critique of
CollectiveX: *"you time dispatch/combine in isolation; real serving overlaps A2A with GEMM
and batches differently — show me the microbench predicts anything."* The deliverable is a
measured answer (correlation or falsification), not an assumption either way.

## 1. Claim under test

CollectiveX's EP tab implies: **backend ranking by `roundtrip p99` at matched
(shape, EP, T) predicts serving-throughput ranking when only the A2A backend changes.**

Two testable forms, weak → strong:

- **H1 (rank agreement)**: for a fixed (sku, model, concurrency), ordering backends by
  microbench `roundtrip p99` at the matched T equals their ordering by measured decode
  tok/s/gpu (Spearman ρ, exact agreement for the 2–3-backend case).
- **H2 (magnitude)**: per-token decode latency (ITL) deltas between backends are explained
  by `n_moe_layers × Δroundtrip(T)` within a fitted in-situ inflation factor
  (regression `ITL = a + b·n_layers·roundtrip(T)`; report R² and b — b≈1 means the
  isolated microbench transfers, b<1 means serving hides comm behind overlap).

Falsification is a publishable result: if LL-vs-normal crossovers in serving don't match
the microbench crossover (the Decision tab's headline claim), the Decision tab must say so.

## 2. Why this is cheap here

The serving fleet already flips the exact kernels CollectiveX times:

- `benchmarks/multi_node/srt-slurm-recipes/.../1k1k_stp_hightpt_0.yaml:134-136` serves with
  `moe-a2a-backend: deepep` + `deepep-mode: low_latency` — the same DeepEP LL path as
  `tests/ep_deepep.py` mode=ll.
- The CollectiveX NVIDIA container **is** the serving container
  (`lmsysorg/sglang:v0.5.11-cu130`), so kernel/library versions match by construction —
  the microbench point and the serving run share `deep_ep 1.2.1 / flashinfer 0.6.8 /
  NCCL 2.28.9` provenance.
- MI355X serving on SGLang exists (dsr1/qwen3.5/glm5 recipes), giving the AMD leg.

So the study is a **controlled A/B on an existing recipe** (vary ONE key), not new infra.

## 3. Design

**Vary (the treatment):** the A2A backend only.
- NVIDIA: `moe-a2a-backend deepep` × `deepep-mode {normal, low_latency, auto}` vs
  `moe-a2a-backend none` (sglang's non-EP/TP fallback = the "no specialized A2A" control).
- AMD: the MoRI-EP path vs the default (aiter/RCCL) path in the ROCm sglang image.
- Step 0 (verify-first): `python -m sglang.launch_server --help | grep -iE "a2a|deepep"`
  in the pinned container to enumerate what THIS sglang actually switches; the study
  covers exactly the backends the serving stack can run (that's the decision users face).
  DO NOT claim uccl/deepep-hybrid/flashinfer coverage unless a real sglang flag drives them.

**Hold fixed (everything else):** model + quant, container digest, TP/DP/EP layout,
kv-cache config, batch composition, node, clock/power state (record `nvidia-smi -q -d
CLOCK,POWER` before/after — env_capture already fingerprints this).

**Model/SKU matrix (small — it's a study, not a sweep):**

| leg | sku | model (existing recipe base) | EP shape exercised |
|---|---|---|---|
| NV-1 | h200 | DSR1-fp8 (fixed_seq_len recipe) | 7168/8/256 — the ds-like-ref headline shape |
| NV-2 | b300 | DSR1-fp4 (`dsr1_fp4_b300.sh`) | same shape, Blackwell |
| AMD | mi355x | DSR1-fp8 (`dsr1_fp8_mi355x*.sh`) | same shape, MoRI leg |

One SKU (h200) first; the other two only after the method holds there.

**Concurrency ↔ T mapping (the join key):** decode tokens/rank/step ≈ running requests
per attention-DP rank. Pick serving concurrencies so per-rank T lands on microbench ladder
points **{8, 32, 128}** (e.g. EP8 + dp-attention 8 → concurrency 64 ⇒ T≈8/rank). Record
the *realized* per-step batch from sglang metrics — don't trust the target. 1k1k
fixed-seq-len workload (existing generator) so decode dominates and prefill contamination
is bounded; 3 repeats per cell, fresh server process each.

**Cell count:** 3 backends × 3 T × 3 repeats = 27 serving runs per SKU leg, ~10 min each
≈ one evening of one node. Microbench counterpart points already exist in the sweep data.

## 4. What to measure

Per serving run:
1. **tok/s/gpu + ITL p50/p99** — from the existing bench client (the InferenceMAX
   serving-bench output the recipes already emit).
2. **In-situ A2A time** — a 30 s `torch.profiler` window (or sglang's kernel-timing env if
   the container exposes it) mid-steady-state: sum of dispatch/combine kernel time per
   decode step. This is the number the microbench claims to approximate; the ratio
   `insitu / (n_moe_layers × microbench_roundtrip(T))` is the **inflation factor** —
   >1 means contention the microbench misses, <1 means overlap hides comm.
   If the profiler perturbs tok/s >2%, run it as a separate 4th repeat, not inside the
   timed repeats.
3. **Realized routing skew** — expert-load CV from sglang's expert-distribution metrics if
   exposed; otherwise note as ungated. Joins to the microbench zipf-sensitivity view and
   feeds the trace-replay backlog item (a captured serving routing trace is the natural
   `basis: replayed` workload the headline still lacks).

## 5. Artifact + join contract

New family `e2e-correlation`, one doc per serving run (extends the ep-result-v4 pattern;
new schema `e2e-correlation-v1.schema.json`, stdlib-validated like the others):

```
{ family: "e2e-correlation", schema_version: 1,
  serving: { stack: "sglang", version, model, quant, flags{moe_a2a_backend, deepep_mode,...},
             concurrency, realized_tokens_per_rank, tokps_per_gpu, itl_p50_ms, itl_p99_ms,
             insitu_a2a_us_per_step | null, expert_load_cv | null },
  microbench_ref: { comparison_key, backend, mode, T, roundtrip_p99_us, source_run_id },
  joined: { n_moe_layers, predicted_a2a_us_per_step, inflation, notes },
  environment / reproduction / provenance: as in ep-result-v4 }
```

Join rule: microbench point must match (sku, backend+mode, shape, EP, contract=
`cached-layout-comm-only-v1` — serving reuses layouts, so the cached contract is the
honest counterpart, NOT layout-and-dispatch) and T within one ladder step. Mismatched
joins are refused, same doctrine as `comparison_key`.

Analysis output (one script, `analyze_correlation.py`): rank-agreement table + ITL
regression + inflation factors per (sku, T) → a "Does the microbench predict serving?"
section in the report/app. Publication tier: `study` (never mixed into official EP rows).

## 6. Companion contract: overlapped-with-compute (closes the isolation critique directly)

Independent of serving, add measurement contract **`overlapped-gemm-v1`** to the EP
harness: run the timed dispatch/combine loop while a second stream runs the expert-shaped
GEMM victim that `copy_engine_bench.py` already implements (matmul 2048³ pattern — reuse
that code, don't reinvent). Record (a) comm percentiles under compute contention and
(b) GEMM slowdown vs its solo baseline (= SM-stealing signal, the copy-engine bench's
`sm_slowdown` metric applied to EP). This is ~a day of harness work: new contract enum in
schema + capability + harness stream logic. It measures exactly what tuned-SM backends
(DeepEP num_sms) trade away, and gives the microbench an overlap-aware column *without*
needing the full serving study. Run it in the same sweep lanes; it becomes a per-backend
line, not a study.

## 7. Risks / expected walls (pre-registered, judge-by-data)

- **sglang flag coverage**: if v0.5.11 can't switch some backend, the study scope shrinks
  to what it CAN switch — that's still the real user decision. Evidence the flag list in
  the artifact.
- **DSR1 memory fit at bf16**: use the fp8/fp4 recipes as-is; quant differs from the
  microbench's bf16 headline — join against the matching-dtype microbench points
  (fp8 dispatch exists for deepep/flashinfer/mori).
- **`none` backend confound**: `moe-a2a-backend none` changes more than comm (different
  MoE execution path). Treat it as a secondary control; the primary contrast is
  deepep-normal vs deepep-LL (identical everything except kernels — also directly tests
  the Decision tab's LL-crossover claim).
- **Noise**: ITL jitter from scheduler/kv events can swamp µs-scale comm deltas at low T.
  That's a finding, not a failure: "below T=X the A2A backend choice is not observable in
  serving" is Decision-tab content.
- **MNNVL/rack legs**: out of scope v1; single-node EP8 only (matches the headline view).

## 8. Execution checklist

1. [ ] Step-0 capability probe on h200: enumerate sglang A2A flags in the pinned container.
2. [ ] Serving A/B harness: wrap ONE existing dsr1 recipe with backend/mode + concurrency
       envs; emit the `e2e-correlation` doc per run (launcher lane `CX_BENCH=e2e-correlation`).
3. [ ] Profiler probe: verify dispatch/combine kernels are visible + <2% overhead.
4. [ ] h200 matrix (27 runs) + `analyze_correlation.py` → rank table, R², inflation.
5. [ ] Decision gate: method sound on h200? → b300 + mi355x legs; else document why.
6. [ ] `overlapped-gemm-v1` contract in the EP harness (independent track, can start now).
7. [ ] Report/app: "microbench→serving" section + study-tier publication contract.
