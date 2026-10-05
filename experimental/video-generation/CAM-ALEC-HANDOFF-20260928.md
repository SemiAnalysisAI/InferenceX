# Cam / Alec — H3 interface handoff (2026-09-28)

Decision-oriented status after reconciling InferenceX [#2916](https://github.com/SemiAnalysisAI/InferenceX/pull/2916) to current `main` and checking App [#1193](https://github.com/SemiAnalysisAI/InferenceX-app/pull/1193) against that contract. No third product PR.

## SHAs

| Side | Branch | Head | Base |
| --- | --- | --- | --- |
| Backend #2916 | `feat/h3-cross-hardware` (repair also on `cursor/h3-main-reconcile-7116`) | `1e648b146fc1dede7e4758392787fd8b1f1de30c` | `main` @ `e0127239f073f3cd64f0732acdff62354af7b49a` |
| App #1193 | `feat/videogenx-hardware-dashboard` | `ac73782b325d3b5a7764243d6a8aac2f7434663a` | `master` (unchanged this pass) |

#2916 is **MERGEABLE** vs current main (was CONFLICTING). Still draft / blocked on reviews and required checks — not GPU-qualified.

## Contract matrix (what is compatible now)

| Surface | Backend #2916 after reconcile | App #1193 consumer | Verdict |
| --- | --- | --- | --- |
| **YAML / planner** | Empty `config-keys` + `workflow-dispatch: h3-video.yml`; planner validates workflow then selects **no** LLM jobs | N/A (App does not plan sweeps) | Compatible |
| **Runner / paths** | Workflows, `PYTHONPATH`, and staged power packaging read `inferencex-e2e/`; staged package still exposes `infx/...` | Local demo still serves extracted CI bundles (`scripts/serve-h3-artifact.py`) | Compatible; path fix was backend-only |
| **Results** | `result.json` `schema_version: "1.0.0"`, `bundle_type: h3_benchmark_result` ([RESULTS.md](./RESULTS.md), [result.schema.json](./result.schema.json)) | `bundle.ts` requires exactly `1.0.0` / `h3_benchmark_result`; history projection is additive nullable | Compatible — no App edit required for this repair |
| **Media** | Original media + SHA256 inventory retained; export/fidelity workflows unchanged in behavior | Compare / `h3-media` publish path read published blobs; chart uses history projection, not raw bundles | Compatible; publication/warm path not re-run here |
| **Async completion** | Per-request outcomes (`completed` / `invalid_media` / timeouts / etc.), job IDs, submit→terminal timings | History keeps scheduled / completed / failed / valid counts separate; nulls stay null | Compatible |
| **AMD** | Prep/inventory/serving CI paths + Python 3.10 staged power; generation still gated | Roster keeps MI355X as **Not measured** | Compatible as “prep-only / not measured”; no AMD generation claim |

## What still needs qualification (do not claim)

1. **GPU execution on the reconciled head** — no new allocation this pass; prior CPU green at old head does not qualify this tip.
2. **AMD generation smoke** — staging/prep only; do not treat MI355X as measured.
3. **Healthy H100 replacement / A/A noise floor / AMD monitor trust** — still unproven for article claims.
4. **Native H3 SRT adapter** — separate migration slice; not in #2916/#1193 closure.
5. **App remote Unit/E2E** — #1193 still draft / blocked; checks were skipped, not passed.

## Recovered plateau → deployment-scaling story (do not re-open as “unknown”)

Later evidence already supersedes “first discover deployment tradeoffs”:

- Retained batch-one / single-replica cells: C1/C2/C4 throughput within about ±1.5% while latency grows with concurrency → concurrency mostly measures queueing on that layout ([App videogenx-dashboard.md](https://github.com/SemiAnalysisAI/InferenceX-app/blob/feat/videogenx-hardware-dashboard/docs/videogenx-dashboard.md)).
- App already keys the curve by **deployment** (`participating GPUs` × `tp_size` × `ulysses_degree`), treats `concurrency > replicas` as queueing, and keeps queued cells out of the chart.
- Fixture still has **one** measured deployment per measured SKU (4 GPUs / TP2×Ulysses2). Sealed 2-GPU and H200 4/8-GPU campaign evidence exists in the Sept 26 work package; the remaining engineering sweep is the authorized GPUs-per-video matrix on a new backend head — not another plateau hunt.

## App #1193 compatibility verdict

**Code + fixture read: compatible with repaired #2916.** No mandatory App patch for the `inferencex-e2e/` path realignment or the preserved `result.json` 1.0.0 contract.

Unverified here (needs App owner / browser when UI changes): live blob publication, Cypress E2E against non-fixture backends, and any UI polish still open on #1193. Prefer App-owner fixes inside #1193 over a third PR.

## Next concrete decision (pick one)

1. **Merge-ready backend review of #2916** at `1e648b14` (CPU contract green; still draft; GPU still outstanding), then App #1193 review/merge sequence; **or**
2. **Authorize one GPU qualification on this head** — prefer a single H200 C1 (or one GPUs-per-video cell) before any broad campaign; **or**
3. **Defer GPU** and only land the planner/workflow foundation, keeping article claims inside already retained deployment-scaling evidence.

Recommendation: (1) if the goal is to unblock review of the existing drafts; schedule (2) only when someone will own the exact cell and acceptance receipt.

## Validation run this pass (CPU only)

```text
# H3 harness (matches test-h3-video.yml)
cd experimental/video-generation
PYTHONPATH=$PWD/../../inferencex-e2e
uv run --no-project --python 3.12 \
  --with 'av==16.1.0' --with 'numpy==2.3.5' \
  --with 'pytest>=8,<9' --with 'jsonschema>=4,<5' python -m pytest -q
# → 661 passed

# Staged power on retained runtime Python
uv run --no-project --python 3.10 --with 'pytest==8.4.2' python -m pytest -q \
  experimental/video-generation/tests/test_ci.py::test_staged_package_recomputes_power_without_the_repository
# → 1 passed

# Planner workflow-dispatch contract
cd inferencex-e2e && uv run pytest -q \
  infx/tests/matrix/test_process_changelog.py -k 'workflow_dispatch or manual' \
  infx/tests/matrix/test_validation.py -k 'workflow_dispatch or manual' \
  infx/tests/test_installed_package.py -k 'workflow_dispatch or manual'
# → 18 passed
```

## Owner notes

- Repair commits authored/committed as Wenyao Gao `<wenyao.gao28@gmail.com>`.
- Sibling repair branch `cursor/h3-main-reconcile-7116` matches `feat/h3-cross-hardware`; do not open a third H3 product PR.
- No Slack/email/public review comments; no GPU dispatch; no claim of GPU qualification.
