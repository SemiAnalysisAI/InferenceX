# CollectiveX

CollectiveX is an experimental expert-parallel communication benchmark for comparing EP libraries
on one platform and matched EP latency/effective logical payload bandwidth across platforms.

> Publication hold: existing schema 3-5 artifacts are historical diagnostics. They cannot drive a
> ranking, recommendation, regression baseline, or CollectiveX v1 dataset.

## v1 Target

The namespaced `collectivex.ep.v1` product covers H100, H200, B200, B300, GB200, GB300, MI325X, and
MI355X with explicit topology. Headline points use the same BF16 workload, 512 observations, and
three independent allocations. The final dataset provides:

- measured roundtrip p50/p99 and independently available component latency;
- effective logical payload GB/s, kept separate from bus or wire metrics;
- within-chip library, portable-reference, identical-stack, and best-conforming comparisons;
- complete accepted/failed/unsupported coverage, provenance, and repeat stability;
- immutable locally hosted artifacts with an atomic development channel.

`goal.md` is the local `/goal` execution checklist. [docs/methodology.md](docs/methodology.md) is the
tracked technical contract and artifact architecture. `notes.md` is a local evidence ledger.

## EP Backends

| Backend | v1 status |
|---|---|
| Legacy DeepEP | Adapter uses `deep_ep.Buffer` |
| DeepEP PR #605 V2 | Future: needs a dedicated `ElasticBuffer`/NCCL-Gin adapter |
| DeepEP Hybrid | Adapter exists; exact API/build/timing identity required |
| FlashInfer EP | Paired roundtrip; isolated components may be unavailable |
| UCCL EP | Adapter exists; native build and provenance required |
| NCCL/RCCL A2A | Portable `all_to_all_single` reference |
| MoRI | AMD adapter exists; timing/correctness and launcher fixes remain |

Historical `--deepep-v2` runs instantiated legacy `Buffer` and are not PR #605 V2 evidence. V2 is
excluded from every workflow and promoted suite until the real adapter exists. Native NCCL EP and
AITER EP are follow-on adapters, not aliases for the portable reference.

## Workflows

`.github/workflows/collectivex-sweep.yml` resolves named suites into self-hosted shard jobs and
aggregates uploaded results. It has exactly two promoted suites:

- `ep-core-v1`: 78 uniform cases and 390 token points;
- `ep-routing-v1`: 154 Zipf/EPLB cases and 228 token points.

The combined run is 39 shard cells, 232 cases, and 618 token points. Every case is normal-mode BF16
under `layout-and-dispatch-v1`. Cached-layout (`[cl]`), runtime-visible (`[rv]`), LL, FP8, extra
routing distributions, model envelopes, placement labels, and temporal/uneven scenarios are not v1
sweep dimensions. Their adapter paths remain available only for explicit manual diagnostics and
historical display.

Once the real PR #605 adapter exists, its eight cells add 48 cases and 128 token points, making the
final v1 target 47 cells, 280 cases, and 746 points.

`.github/workflows/collectivex-experimental.yml` is manual bring-up. Both workflows stop at GitHub
artifacts; neither updates the frontend or any external store. Results remain diagnostic until v1
validation, exact coverage, repeat stability, and local promotion gates land.

Workflows map public SKU labels to launchers explicitly and never persist the physical runner name.
Container images and digests live in `runtime/common.sh`; the public GHA SKU and build capability
table lives in `capability.py`. Private host inventory is never part of generation.

## Runner Configuration

Each self-hosted runner sources one operator-owned shell file outside the checkout. The default is
`$XDG_CONFIG_HOME/inferencex/collectivex.env` (or `~/.config/inferencex/collectivex.env`); set
`COLLECTIVEX_OPERATOR_CONFIG` to use another location. Required exported variables are:

| Public SKU | Required variables |
|---|---|
| `h100-dgxc`, `b200-dgxc` | `CX_PARTITION`, `CX_ACCOUNT`, `CX_SQUASH_DIR` |
| `h200-dgxc` | `CX_PARTITION`, `CX_SQUASH_DIR` |
| `b300`, `gb200` | `CX_PARTITION`, `CX_ACCOUNT`, `CX_SQUASH_DIR`, `CX_STAGE_DIR` |
| `gb300` | `CX_PARTITION`, `CX_ACCOUNT`, `CX_SQUASH_DIR`, `CX_STAGE_DIR`, `CX_ENROOT_CACHE_PATH` |
| `mi325x`, `mi355x` | `CX_PARTITION`, `CX_SQUASH_DIR` |

`CX_EXCLUDE_NODES`, `CX_NODELIST`, `CX_ACCOUNT` (where optional), `CX_STAGE_DIR` (where optional),
`CX_LOCK_DIR`, and `CX_IMAGE` are deployment overrides. The config file and `env_*.json` captures are
never uploaded as workflow artifacts.

## Local Checks

```bash
python3 -m unittest discover experimental/CollectiveX/tests -p 'test_*.py'
python3 experimental/CollectiveX/sweep_matrix.py \
  --suites ep-core-v1 --backends deepep,nccl-ep --only-sku h100-dgxc \
  --out /tmp/collectivex-matrix.json >/dev/null
bash -n experimental/CollectiveX/runtime/*.sh experimental/CollectiveX/launchers/*.sh
```

These exercise the current implementation; they do not promote data.

## Main Files

| Path | Role |
|---|---|
| `capability.py`, `configs/` | Public backend/platform capabilities and workload/suite registries |
| `sweep_matrix.py`, `generate_matrix.py` | Suite and shard resolution |
| `tests/ep_harness.py`, `tests/run_ep.py` | Shared EP execution |
| `tests/ep_*.py` | Backend adapters; the independent v1 oracle is not yet wired |
| `validate_results.py` | Strict result validation |
| `aggregate_results.py` | Per-run outcome projection; the private attempt ledger is still pending |
| `make_bundle.py` | Bundle construction; authoritative publisher still pending |
| `docs/methodology.md` | v1 contract, comparability, evidence, and isolated storage |

## Isolated Storage

Development storage is one self-hosted persistent filesystem. GitHub artifacts are transient input;
there is no Vercel, GCP, Neon, managed database, or managed object store. Private run bundles and
sanitized public datasets are immutable and content-addressed; only a validated `dev-latest` pointer
is updated atomically.

## Current Status

Fixed-512 scheduling is present. The v1 schema/identity, backend correctness fixes, exact coverage,
three-allocation stability, local publisher, and frontend channel ingestion remain active work. No
current row is approved for a public library or chip ranking.
