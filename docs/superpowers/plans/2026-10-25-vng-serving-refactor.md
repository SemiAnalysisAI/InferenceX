# VNG Serving Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Each task is one ticket: run it in a clean context, read only the task, the Global Constraints and the spec.

**Goal:** Run every VNG serving benchmark (AgentX, fixed ISL/OSL, ISL/OSL search, lock-pinned file datasets) on current upstream InferenceX through a small `serving` launch path, with six base scenarios and per-node Slurm cluster records.

**Architecture:** A serving YAML is validated and expanded by `infx/serving` into matrix points. `.github/workflows/serving.yml` sends each point to upstream `benchmark-tmpl.yml`. There `python -m infx.launch run` routes `SERVING_JSON` to `infx/launch/drivers/serving.py`, which stages images and `sbatch`es `benchmarks/single_node/serving/job.sh` with the cluster's right-sized resources. `job.sh` starts the DCGM, storage and router sidecars and the engine step (`bench.sh`). `bench.sh` starts the engine replicas and runs `python3 -m infx.serving.client`, which replays the rendered scenario with the upstream `agentx-harness` and writes the result JSON through upstream `infx.results` CLIs.

**Tech Stack:** Python 3.12 (pydantic 2, PyYAML), bash, Slurm + Pyxis/enroot, GitHub Actions, `agentx-harness` (AIPerf) from PyPI.

**Spec:** `docs/superpowers/specs/2026-10-25-vng-serving-refactor-design.md`

## Global Constraints

- Branch `refactor/vng-benchmark`, worktree `/home/thanglq5/greennode-code/InferenceOptimization/InferenceX-refactor`. Never touch `vng-benchmark`.
- Every VNG file has `serving` in its path. Upstream files may only gain the additions listed in Task 3 (`configs/runners.yaml`) and Task 8 (`benchmark-tmpl.yml`, `infx/launch/policy.py`, `infx/launch/drivers/__init__.py`).
- Never `--exclusive`. A job asks for its GPUs, 16 CPUs per GPU and a per-GPU memory share (`salloc-args`), nothing more.
- Ports inside a job are picked free on the node at run time, from 20000–32767 (below the Linux ephemeral range).
- Keep `scenario: inferencex-agentx-mvp` and `unsafeOverride: true` in the five trace scenarios.
- LiveKit default dataset: `livekit-111868-v1.1-warm`. No `livekit_ttft.py`, no `*-plan` lock entries.
- Upstream code is used through public entry points: `SlurmBackend.stage_image`, `SlurmBackend.submit_batch`, `stream_logs`, `state`, the `infx.bench wait` CLI, `infx.bench.agentic.venv`, and the `infx.results.*` CLIs. The one exception is `sweep_result.py`, which imports four private helpers from `infx.results.agentic` (marked `ponytail:`).
- Git: stage files by name (no `git add -A`/`.`), author is the repo config (`Ly Quang Thang <thanglq5@greennode.ai>`). **Never push, never open a PR, never register or remove runners** without the user's explicit OK.
- Test command, from `inferencex-e2e/`: `uv run --no-project --python 3.12 --with pytest --with pydantic --with pyyaml python -m pytest <paths> -q` (written as `PYTEST <paths>` below).

## Deviations from the spec (decided while planning, from reading the code)

1. `submit_batch` adds no CPU or memory flags (only `salloc` adds `cpu_directives`). Right-sizing therefore lives in the cluster record's `salloc-args` (`--cpus-per-gpu`, `--mem-per-gpu`). `submit_batch` passes those to `sbatch` too.
2. Squash records use `import: submit-host` with `single-node-import: true`. The runner lives on its GPU node, so the import lands where the job runs. A `compute` import step could land on the sibling node.
3. hgx engine replicas run in **one** engine `srun` step, as VNG does today (`bench.sh` splits `CUDA_VISIBLE_DEVICES`). Router and storage are separate `--overlap` steps.
4. Exp-names keep the old rule with the label in place of the file name: `<name>_c<conc>`, `<name>_search`, `<name>-<label>` for grids. The aggregate JSON's `scenario` field carries the legacy label.
5. LiveKit smoke gets the label `custom-agentic-livekit111868-v2-smoke` (it was `custom-agentic-livekit111868-smoke`).
6. `benchmark-tmpl.yml` also needs its "Validate agentic result" step skipped for search points (no top-level `profile_export_aiperf.json`). This is a fifth upstream touch, one condition.
7. `configs/serving/` receives seven curated examples, one per pattern. The other 46 legacy YAMLs stay on `vng-benchmark`; the skill's migration table covers them.
8. hgx-h200-01 declares only 4 GPUs in Slurm gres today, so `clusters.hcm-hgx01.gpus-per-node` is 4 until infra fixes it.
9. The VNG fix that counts Mooncake `storage_*` hits as external cache hits in `infx/results/agentic/backends/sglang.py` is **not** ported. Aggregates of L3 runs under-report `server_external_cache_hit_rate`; `result-report` computes tiers from raw metrics and is unaffected. Candidate upstream PR.

## File map

| File | Task | Responsibility |
|---|---|---|
| `inferencex-e2e/benchmarks/single_node/serving/datasets/lock.yaml` | 1 | pinned file datasets (+ `entries`) |
| `inferencex-e2e/infx/serving/__init__.py`, `datasets.py` | 1 | read the lock, fetch + sha256-verify one dataset |
| `inferencex-e2e/infx/serving/slo.py` | 2 | parse `metric:stat:op:threshold;...` (ported) |
| `inferencex-e2e/benchmarks/single_node/serving/scenarios/*.yaml` (6) | 2 | base AIPerf configs |
| `inferencex-e2e/infx/serving/scenarios.py` | 2 | base + overrides/point/search/slo → final config + legacy label |
| `inferencex-e2e/configs/runners.yaml` | 3 | `hcm-*` labels and cluster records |
| `inferencex-e2e/infx/serving/schema.py`, `render.py` | 4 | serving YAML model + validation; command rendering (ported) |
| `inferencex-e2e/infx/serving/to_matrix.py`, `utils/serving/dispatch.sh`, `.github/workflows/serving.yml` | 5 | points for the workflow; local validate + dispatch |
| `inferencex-e2e/infx/serving/client.py`, `sweep_result.py` | 6 | in-container client; search result writer |
| `inferencex-e2e/benchmarks/single_node/serving/{job,bench,sidecar}.sh` | 7 | allocation steps; engine step; router/storage sidecar |
| `inferencex-e2e/infx/launch/drivers/serving.py` + 3 upstream hooks | 8 | the launch path |
| `inferencex-e2e/configs/serving/*.yaml` (7) | 9 | curated examples |
| `.agents/skills/{serving-bench,serving-runner-setup,result-report}` + `.claude/skills` links | 10 | skills |
| `inferencex-e2e/docs/serving/ARCHITECTURE.md` | 11 | the flow, for people |
| — | 12 | live verification and runner cutover (manual, gated) |

Tests live in `inferencex-e2e/infx/tests/serving/` (`infx/tests/pytest.ini` puts `inferencex-e2e` on the path).

---

### Task 1: Dataset lock and fetcher

**Files:**
- Create: `inferencex-e2e/benchmarks/single_node/serving/datasets/lock.yaml`
- Create: `inferencex-e2e/infx/serving/__init__.py`, `inferencex-e2e/infx/serving/datasets.py`
- Test: `inferencex-e2e/infx/tests/serving/__init__.py`, `inferencex-e2e/infx/tests/serving/test_datasets.py`

**Interfaces:**
- Produces: `LOCK: Path`; `LockEntry(name, repo, revision, file, sha256, entries: int | None)`; `DatasetError(ValueError)`; `load_lock(path: Path = LOCK) -> dict[str, LockEntry]`; `fetch(name: str, lock: dict[str, LockEntry] | None = None) -> Path`; CLI `python -m infx.serving.datasets <name>` prints the local path, or one `ERROR ...` line and exits 1.

- [ ] **Step 1: Write the lock file**

`inferencex-e2e/benchmarks/single_node/serving/datasets/lock.yaml`:

```yaml
# File datasets for the serving scenarios, keyed by the scenario's datasets[0].name.
# Each entry pins one file at one HF commit; infx/serving/datasets.py downloads it
# (token: GREENLAB_HF_TOKEN if set) and verifies sha256. `entries` is the dataset's
# session count, copied into the scenario when a serving YAML picks this dataset.
tau2-cs:
  repo: Green-Lab/customer-service-agent-traces
  revision: 2c793380d8ba45bfde79f9a5c7b5991ed40ab2ea
  file: trace.tau2.jsonl
  sha256: 598ab95c793326182175d56599c743dc84d8c6d46623359327813099df05c47e
  entries: 1576
# E2 shared-prefix trace: HiCache L2 vs Mooncake L3 A/B.
shared-prefix-e2:
  repo: Green-Lab/shared-prefix-e2
  revision: 02692baf2e3e93baae625a1ee093b7a1679f0952
  file: trace.shared-prefix-e2.jsonl
  sha256: e4dbb42007fd6349530b6f16a26e39d2f1465f8bb76691b32db0591567c50c7e
  entries: 2400
# E3 hidden-prefix trace: shared tool catalogs the router cannot see.
hidden-prefix-e3:
  repo: Green-Lab/hidden-prefix-e3
  revision: 479ae539a2307ec831a05a416987bd94fdd7f20e
  file: trace.hidden-prefix-e3.jsonl
  sha256: e902b741fc20b9396f628431fd4aa9982ce16caa23516ca151f0030616da1cc1
  entries: 2400
livekit-111868-v1:
  repo: Green-Lab/2026-10-16-111868-agentic-traces
  revision: 002c982432295d1603dcaafd68a09cdf5c37e4ac
  file: trace.v1-append-ideal.jsonl
  sha256: 684dc5cb9e845b5e5a01b7e925aabf715869ea7c3f77c97850438f3066dca8f5
  entries: 1217
livekit-111868-v2:
  repo: Green-Lab/2026-10-16-111868-agentic-traces
  revision: 7e36287967360851a1b05ba05730ff1d256b60c7
  file: trace.v2-observed.jsonl
  sha256: 190af677563ea782d060456bbc44f92e802e1d63e4322b147ef0290a49efd4f3
  entries: 1217
# V1.1 = V1 with a dial warm-up before each room's first turn, restricted to the 671 rooms
# with >= 4 real requests. aiperf TTFT includes the 601 warm-up requests (max_tokens 1).
livekit-111868-v1.1-warm:
  repo: Green-Lab/2026-10-16-111868-agentic-traces
  revision: 6e0934c957565a42d52cafac096ec758a4302355
  file: trace.v1.1-append-ideal-warm.jsonl
  sha256: e26542392a202764849c05018f2f8ea3f9574277ffa3400615f0b4df23b0eb2f
  entries: 671
```

- [ ] **Step 2: Write the failing tests**

`inferencex-e2e/infx/tests/serving/__init__.py`: empty file.

`inferencex-e2e/infx/tests/serving/test_datasets.py`:

```python
import hashlib
import sys
import types

import pytest

from infx.serving.datasets import DatasetError, LockEntry, fetch, load_lock


def test_lock_pins_six_datasets_with_entries():
    lock = load_lock()
    assert set(lock) == {
        "tau2-cs", "shared-prefix-e2", "hidden-prefix-e3",
        "livekit-111868-v1", "livekit-111868-v2", "livekit-111868-v1.1-warm",
    }
    assert lock["livekit-111868-v1.1-warm"].entries == 671
    assert lock["livekit-111868-v1"].entries == 1217


@pytest.mark.parametrize("body, message", [
    ("x: {repo: a/b}\n", "x needs exactly repo, revision, file, sha256"),
    ("x: {repo: a/b, revision: abc, file: f, sha256: " + "b" * 64 + "}\n", "x.revision"),
    # all-digit hashes would load as YAML ints, hence a/b
    ("x: {repo: a/b, revision: " + "a" * 40 + ", file: f, sha256: " + "b" * 64 + ", entries: 0}\n",
     "x.entries must be a positive integer"),
])
def test_bad_lock_is_one_line_error(tmp_path, body, message):
    path = tmp_path / "lock.yaml"
    path.write_text(body)
    with pytest.raises(DatasetError, match=message):
        load_lock(path)


def _fake_hub(monkeypatch, path):
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda *args, **kwargs: str(path)
    errors = types.ModuleType("huggingface_hub.errors")
    errors.HfHubHTTPError = type("HfHubHTTPError", (Exception,), {})
    errors.RepositoryNotFoundError = type("RepositoryNotFoundError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)


def test_fetch_verifies_sha256_and_records_the_dataset(tmp_path, monkeypatch):
    data = tmp_path / "trace.jsonl"
    data.write_text("{}\n")
    digest = hashlib.sha256(b"{}\n").hexdigest()
    _fake_hub(monkeypatch, data)
    monkeypatch.setenv("RESULT_DIR", str(tmp_path))
    lock = {"d": LockEntry("d", "o/r", "0" * 40, "trace.jsonl", digest, 3)}
    assert fetch("d", lock) == data
    assert (tmp_path / "dataset.txt").read_text().startswith("d o/r@")


def test_fetch_rejects_a_changed_file(tmp_path, monkeypatch):
    data = tmp_path / "trace.jsonl"
    data.write_text("tampered\n")
    _fake_hub(monkeypatch, data)
    lock = {"d": LockEntry("d", "o/r", "0" * 40, "trace.jsonl", "0" * 64, 3)}
    with pytest.raises(DatasetError, match="sha256 mismatch"):
        fetch("d", lock)


def test_unknown_dataset(monkeypatch):
    with pytest.raises(DatasetError, match="dataset nope: not in lock.yaml"):
        fetch("nope", {})
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `PYTEST infx/tests/serving/test_datasets.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'infx.serving'`.

- [ ] **Step 4: Implement**

`inferencex-e2e/infx/serving/__init__.py`:

```python
"""VNG serving-YAML benchmarks: see docs/serving/ARCHITECTURE.md."""
```

`inferencex-e2e/infx/serving/datasets.py`:

```python
"""Lock-pinned file datasets of the serving scenarios: read the lock, fetch and verify one file.

    python -m infx.serving.datasets <name>   # local path on stdout, or one ERROR line, exit 1

Token: GREENLAB_HF_TOKEN when set; unset means anonymous (never the runner's HF_TOKEN).
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

LOCK = Path(__file__).resolve().parents[2] / "benchmarks/single_node/serving/datasets/lock.yaml"
FIELDS = {
    "repo": r"[^/\s]+/[^/\s]+",
    "revision": r"[0-9a-f]{40}",
    "file": r"\S+",
    "sha256": r"[0-9a-f]{64}",
}


class DatasetError(ValueError):
    """A lock or download problem, as one line."""


@dataclass(frozen=True)
class LockEntry:
    name: str
    repo: str
    revision: str
    file: str
    sha256: str
    entries: int | None = None


def load_lock(path: Path = LOCK) -> dict[str, LockEntry]:
    """Every pinned dataset by name; raise ``DatasetError`` on the first malformed entry."""
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as error:
        raise DatasetError(f"lock: cannot read {path.name} ({error})".replace("\n", " ")) from None
    if not isinstance(raw, dict):
        raise DatasetError("lock: must map dataset names to entries")
    lock = {}
    for name, entry in raw.items():
        keys = set(entry) if isinstance(entry, dict) else set()
        if not set(FIELDS) <= keys or keys - {*FIELDS, "entries"}:
            raise DatasetError(f"lock: {name} needs exactly {', '.join(FIELDS)} (and optional entries)")
        for field, pattern in FIELDS.items():
            if not re.fullmatch(pattern, str(entry[field])):
                raise DatasetError(f"lock: {name}.{field} {entry[field]!r} does not match {pattern}")
        entries = entry.get("entries")
        if entries is not None and (isinstance(entries, bool) or not isinstance(entries, int) or entries < 1):
            raise DatasetError(f"lock: {name}.entries must be a positive integer")
        lock[name] = LockEntry(name, *(str(entry[field]) for field in FIELDS), entries)
    return lock


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name: str, lock: dict[str, LockEntry] | None = None) -> Path:
    """Download dataset ``name`` into the HF cache, check its sha256 and return its path."""
    entry = (load_lock() if lock is None else lock).get(name)
    if entry is None:
        raise DatasetError(f"dataset {name}: not in {LOCK.name}")
    token = os.environ.get("GREENLAB_HF_TOKEN") or None
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    where = f"{entry.repo}@{entry.revision}"
    unreadable = f"dataset {name}: {where} not readable; GREENLAB_HF_TOKEN is {'set' if token else 'unset'}"
    try:
        path = Path(hf_hub_download(entry.repo, entry.file, revision=entry.revision,
                                    repo_type="dataset", token=token or False))
    except RepositoryNotFoundError:  # includes GatedRepoError
        raise DatasetError(unreadable) from None
    except HfHubHTTPError as error:
        if getattr(getattr(error, "response", None), "status_code", None) in (401, 403):
            raise DatasetError(unreadable) from None
        raise
    actual = sha256(path)
    if actual != entry.sha256:
        raise DatasetError(f"dataset {name}: sha256 mismatch for {where}/{entry.file}: "
                           f"expected {entry.sha256}, got {actual}")
    if os.environ.get("RESULT_DIR"):
        Path(os.environ["RESULT_DIR"], "dataset.txt").write_text(
            f"{name} {where} file={entry.file} sha256={actual} path={path}\n")
    return path


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m infx.serving.datasets <name>", file=sys.stderr)
        return 2
    try:
        print(fetch(argv[0]))
    except DatasetError as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTEST infx/tests/serving/test_datasets.py`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add inferencex-e2e/benchmarks/single_node/serving/datasets/lock.yaml inferencex-e2e/infx/serving/__init__.py inferencex-e2e/infx/serving/datasets.py inferencex-e2e/infx/tests/serving/__init__.py inferencex-e2e/infx/tests/serving/test_datasets.py
git commit -m "feat(serving): dataset lock with entries and the v1.1-warm LiveKit trace"
```

---

### Task 2: Six base scenarios and the renderer

**Files:**
- Create: `inferencex-e2e/benchmarks/single_node/serving/scenarios/{agentx,tau2,livekit,shared-prefix-e2,hidden-prefix-e3,synthetic}.yaml`
- Create: `inferencex-e2e/infx/serving/slo.py` (port), `inferencex-e2e/infx/serving/scenarios.py`
- Create: `inferencex-e2e/infx/tests/serving/legacy_scenarios/*.yaml` (20 legacy files, test fixtures)
- Test: `inferencex-e2e/infx/tests/serving/test_scenarios.py`, `inferencex-e2e/infx/tests/serving/test_slo.py` (port)

**Interfaces:**
- Consumes: `infx.serving.datasets.load_lock`, `LockEntry` (Task 1).
- Produces:
  - `FAMILIES = ("agentx", "tau2", "livekit", "shared-prefix-e2", "hidden-prefix-e3", "synthetic")`
  - `OVERRIDE_KEYS = ("duration", "dataset", "max_context_length", "generation", "trace_idle_gap_cap")`
  - `ISL = {"8k": 8192, "32k": 32768, "64k": 65536, "128k": 131072}`, `OSL = {"512": 512, "1k": 1024}`, `CACHE = (0, 50, 90)`
  - `ScenarioError(ValueError)`
  - `Point(isl: str, osl: str, cache: int)`
  - `Rendered(doc: dict, label: str, kind: str, dataset: str, search: bool, duration: float)` with `.yaml() -> str`
  - `render(family, overrides=None, point=None, search=False, slo=None, lock=None) -> Rendered`
  - `slo.parse_slo(text) -> list[SLO]`, `SLOError` (unchanged from the port).

- [ ] **Step 1: Copy the legacy fixtures and the base files from git**

Run from the worktree root:

```bash
E=inferencex-e2e; L=$E/infx/tests/serving/legacy_scenarios; S=$E/benchmarks/single_node/serving/scenarios
mkdir -p "$L" "$S"
for f in agentx-cc-062126 agentx-cc-062126-smoke agentx-cc-062126-256k agentx-cc-062126-1m agentx-cc-062126-topp95 \
         custom-agentic-tau2 custom-agentic-tau2-smoke custom-agentic-livekit111868-v1 custom-agentic-livekit111868-v2 \
         custom-agentic-livekit111868-smoke agentx-shared-prefix-e2 agentx-hidden-prefix-e3 agentx-hidden-prefix-e3-smoke \
         sweep-isl8k-osl512-cache50 sweep-isl8k-osl1k-cache90 sweep-isl128k-osl512-cache0 sweep-isl128k-osl1k-cache0 \
         fixed-isl8k-osl1k-cache50 fixed-isl8k-osl1k-cache0; do
  git show "vng-benchmark:scenarios/$f.yaml" > "$L/$f.yaml"
done
git show "bench/gemma4-livekit-dyn-swa-arms:scenarios/custom-agentic-livekit111868-v1.1-warm.yaml" > "$L/custom-agentic-livekit111868-v1.1-warm.yaml"
git show vng-benchmark:scenarios/agentx-cc-062126.yaml > "$S/agentx.yaml"
git show vng-benchmark:scenarios/custom-agentic-tau2.yaml > "$S/tau2.yaml"
cp "$L/custom-agentic-livekit111868-v1.1-warm.yaml" "$S/livekit.yaml"
git show vng-benchmark:scenarios/agentx-shared-prefix-e2.yaml > "$S/shared-prefix-e2.yaml"
git show vng-benchmark:scenarios/agentx-hidden-prefix-e3.yaml > "$S/hidden-prefix-e3.yaml"
git show vng-benchmark:scenarios/fixed-isl8k-osl1k-cache0.yaml > "$S/synthetic.yaml"
git show vng-benchmark:infx/serving/slo.py > $E/infx/serving/slo.py
git show vng-benchmark:utils/test_serving_slo.py > $E/infx/tests/serving/test_slo.py
ls "$L" | wc -l   # expect 20
```

Then edit the header comment (the leading `#` lines only) of each base file so it names the base file, not the legacy one:
- `agentx.yaml`: `# AgentX replay of the SemiAnalysis CC traces (062126 corpus). Variants (smoke, 256k, 1m, top_p) are serving-YAML overrides; see infx/serving/scenarios.py.`
- `tau2.yaml`, `livekit.yaml`, `shared-prefix-e2.yaml`, `hidden-prefix-e3.yaml`: keep the existing description lines; replace any path that points at `scenarios/`, `utils/serving/` or `benchmark_lib.sh` with `infx/serving/`; replace `serve_generic.sh` with `infx/serving/client.py`.
- `synthetic.yaml`: `# Synthetic ISL/OSL/prefix-cache workload. infx/serving/scenarios.py fills isl, osl and prefix_prompts from workload: and adds the adaptive_search sweep: block when workload.search is true.`

- [ ] **Step 2: Write the failing tests**

`inferencex-e2e/infx/tests/serving/test_scenarios.py`:

```python
from pathlib import Path

import pytest
import yaml

from infx.serving.scenarios import FAMILIES, Point, ScenarioError, render

LEGACY = Path(__file__).parent / "legacy_scenarios"
GEN = {"top_p": 0.95, "temperature": 1.0}

# (legacy file, family, overrides, point, search, expected label)
CASES = [
    ("agentx-cc-062126", "agentx", {}, None, False, "agentx-cc-062126"),
    ("agentx-cc-062126-smoke", "agentx", {"duration": 180}, None, False, "agentx-cc-062126-smoke"),
    ("agentx-cc-062126-256k", "agentx", {"dataset": "semianalysis_cc_traces_weka_062126_256k"}, None, False,
     "agentx-cc-062126-256k"),
    ("agentx-cc-062126-1m", "agentx", {"max_context_length": None}, None, False, "agentx-cc-062126-1m"),
    ("agentx-cc-062126-topp95", "agentx", {"generation": GEN}, None, False, "agentx-cc-062126-topp95"),
    ("custom-agentic-tau2", "tau2", {}, None, False, "custom-agentic-tau2"),
    ("custom-agentic-tau2-smoke", "tau2", {"duration": 180}, None, False, "custom-agentic-tau2-smoke"),
    ("custom-agentic-livekit111868-v1.1-warm", "livekit", {}, None, False,
     "custom-agentic-livekit111868-v1.1-warm"),
    ("custom-agentic-livekit111868-v1", "livekit", {"dataset": "livekit-111868-v1"}, None, False,
     "custom-agentic-livekit111868-v1"),
    ("custom-agentic-livekit111868-v2", "livekit", {"dataset": "livekit-111868-v2"}, None, False,
     "custom-agentic-livekit111868-v2"),
    ("custom-agentic-livekit111868-smoke", "livekit", {"dataset": "livekit-111868-v2", "duration": 180}, None,
     False, "custom-agentic-livekit111868-v2-smoke"),
    ("agentx-shared-prefix-e2", "shared-prefix-e2", {}, None, False, "agentx-shared-prefix-e2"),
    ("agentx-hidden-prefix-e3", "hidden-prefix-e3", {}, None, False, "agentx-hidden-prefix-e3"),
    ("agentx-hidden-prefix-e3-smoke", "hidden-prefix-e3", {"duration": 180}, None, False,
     "agentx-hidden-prefix-e3-smoke"),
    ("sweep-isl8k-osl512-cache50", "synthetic", {}, Point("8k", "512", 50), True, "sweep-isl8k-osl512-cache50"),
    ("sweep-isl8k-osl1k-cache90", "synthetic", {}, Point("8k", "1k", 90), True, "sweep-isl8k-osl1k-cache90"),
    ("sweep-isl128k-osl512-cache0", "synthetic", {}, Point("128k", "512", 0), True, "sweep-isl128k-osl512-cache0"),
    ("sweep-isl128k-osl1k-cache0", "synthetic", {}, Point("128k", "1k", 0), True, "sweep-isl128k-osl1k-cache0"),
    ("fixed-isl8k-osl1k-cache50", "synthetic", {}, Point("8k", "1k", 50), False, "fixed-isl8k-osl1k-cache50"),
    ("fixed-isl8k-osl1k-cache0", "synthetic", {}, Point("8k", "1k", 0), False, "fixed-isl8k-osl1k-cache0"),
]


@pytest.mark.parametrize("legacy, family, overrides, point, search, label", CASES, ids=[c[0] for c in CASES])
def test_base_plus_variant_equals_the_legacy_file(legacy, family, overrides, point, search, label):
    rendered = render(family, overrides, point, search)
    assert rendered.doc == yaml.safe_load((LEGACY / f"{legacy}.yaml").read_text())
    assert rendered.label == label
    assert yaml.safe_load(rendered.yaml()) == rendered.doc


def test_every_legacy_fixture_is_covered():
    assert {path.stem for path in LEGACY.glob("*.yaml")} == {case[0] for case in CASES}


def test_kinds_and_datasets():
    assert render("agentx").kind == "public"
    tau2 = render("tau2")
    assert (tau2.kind, tau2.dataset, tau2.duration) == ("file", "tau2-cs", 1200.0)
    assert render("livekit").dataset == "livekit-111868-v1.1-warm"
    sweep = render("synthetic", point=Point("8k", "1k", 0), search=True)
    assert (sweep.kind, sweep.search) == ("synthetic", True)


def test_slo_replaces_the_sweep_filters():
    doc = render("synthetic", point=Point("8k", "1k", 0), search=True,
                 slo="time_to_first_token:p95:lt:500;inter_token_latency:p50:lt:50").doc
    assert doc["sweep"]["sla_filters"] == [
        {"metric_tag": "time_to_first_token", "stat": "p95", "op": "lt", "threshold": 500.0},
        {"metric_tag": "inter_token_latency", "stat": "p50", "op": "lt", "threshold": 50.0},
    ]


@pytest.mark.parametrize("args, message", [
    (("agentx", {"warmup": 1}), "overrides: unknown key"),
    (("agentx", {"dataset": "tau2-cs"}), "overrides.dataset: agentx takes a public"),
    (("tau2", {"dataset": "semianalysis_cc_traces_weka_062126"}), "is not in the dataset lock"),
    (("agentx", {"duration": 0}), "overrides.duration"),
    (("agentx", {"generation": {"top_k": 5}}), "overrides.generation"),
    (("synthetic", {}), "workload: synthetic needs"),
    (("synthetic", {"dataset": "x"}, Point("8k", "1k", 0)), "overrides: synthetic only takes duration"),
    (("synthetic", {}, Point("16k", "1k", 0)), "workload.isl"),
    (("synthetic", {}, Point("8k", "1k", 0), False, "time_to_first_token:avg:lt:1"), "slo: only applies"),
    (("agentx", {}, Point("8k", "1k", 0)), "workload/slo: only for scenario synthetic"),
    (("nope",), "scenario: 'nope' is not one of"),
])
def test_bad_variants_fail_with_one_line(args, message):
    with pytest.raises(ScenarioError, match=message):
        render(*args)


def test_families_have_base_files():
    from infx.serving.scenarios import SCENARIOS_DIR
    assert {path.stem for path in SCENARIOS_DIR.glob("*.yaml")} == set(FAMILIES)
```

In the ported `test_slo.py`, change the import line to `from infx.serving.slo import SLOError, parse_slo` if it differs (the module path is the same; only check).

- [ ] **Step 3: Run the tests to verify they fail**

Run: `PYTEST infx/tests/serving/test_scenarios.py infx/tests/serving/test_slo.py`
Expected: `test_slo.py` passes; `test_scenarios.py` FAILS with `ModuleNotFoundError: No module named 'infx.serving.scenarios'`.

- [ ] **Step 4: Implement `scenarios.py`**

`inferencex-e2e/infx/serving/scenarios.py`:

```python
"""Serving scenarios: six base AIPerf configs plus the variant a serving YAML asks for.

render() returns the final AIPerf config and the legacy label its results are filed under
(agentx-cc-062126-smoke, sweep-isl8k-osl1k-cache50, ...), so old and new runs line up.
Every error is one line "<field>: <problem>".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from infx.serving.datasets import LockEntry, load_lock
from infx.serving.slo import SLOError, parse_slo

SCENARIOS_DIR = Path(__file__).resolve().parents[2] / "benchmarks/single_node/serving/scenarios"
FAMILIES = ("agentx", "tau2", "livekit", "shared-prefix-e2", "hidden-prefix-e3", "synthetic")
OVERRIDE_KEYS = ("duration", "dataset", "max_context_length", "generation", "trace_idle_gap_cap")
SYNTHETIC_OVERRIDES = ("duration",)
GENERATION_KEYS = ("top_p", "temperature")
ISL = {"8k": 8192, "32k": 32768, "64k": 65536, "128k": 131072}
OSL = {"512": 512, "1k": 1024}
CACHE = (0, 50, 90)
DEFAULT_SLO = "time_to_first_token:avg:lt:8000;inter_token_latency:p90:lt:50"
PUBLIC_DATASET = "semianalysis_cc_traces_weka_062126"
LABELS = {
    "agentx": "agentx-cc-062126",
    "tau2": "custom-agentic-tau2",
    "shared-prefix-e2": "agentx-shared-prefix-e2",
    "hidden-prefix-e3": "agentx-hidden-prefix-e3",
}
LIVEKIT_LABEL = "custom-agentic-livekit111868-"


class ScenarioError(ValueError):
    """A scenario variant problem, as one line."""


@dataclass(frozen=True)
class Point:
    """One synthetic grid point."""

    isl: str
    osl: str
    cache: int

    @property
    def label(self) -> str:
        return f"isl{self.isl}-osl{self.osl}-cache{self.cache}"


@dataclass(frozen=True)
class Rendered:
    """The AIPerf config one benchmark job replays, and what it is filed under."""

    doc: dict[str, Any]
    label: str
    kind: str  # public | file | synthetic
    dataset: str  # public loader alias, lock key, or "synthetic"
    search: bool
    duration: float

    def yaml(self) -> str:
        return yaml.safe_dump(self.doc, sort_keys=False)


def load_base(family: str) -> dict[str, Any]:
    if family not in FAMILIES:
        raise ScenarioError(f"scenario: {family!r} is not one of {', '.join(FAMILIES)}")
    return yaml.safe_load((SCENARIOS_DIR / f"{family}.yaml").read_text())


def _profiling(doc: dict[str, Any]) -> dict[str, Any]:
    [phase] = [p for p in doc["benchmark"]["phases"] if p.get("kind") == "profiling"]
    return phase


def _positive(field: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ScenarioError(f"{field}: must be a positive number, got {value!r}")
    return float(value)


def _apply(doc: dict[str, Any], family: str, overrides: dict[str, Any],
           lock: dict[str, LockEntry]) -> list[str]:
    """Apply ``overrides`` to ``doc`` in place; return the label suffixes, in a fixed order."""
    if unknown := sorted(set(overrides) - set(OVERRIDE_KEYS)):
        raise ScenarioError(f"overrides: unknown key(s) {', '.join(unknown)}; allowed: {', '.join(OVERRIDE_KEYS)}")
    dataset = doc["benchmark"]["datasets"][0]
    suffixes: list[str] = []
    if "dataset" in overrides:
        value = overrides["dataset"]
        if dataset["type"] == "public":
            if not (isinstance(value, str) and value.startswith(PUBLIC_DATASET)):
                raise ScenarioError(f"overrides.dataset: {family} takes a public {PUBLIC_DATASET}* dataset, got {value!r}")
            dataset["dataset"] = value
            if value != PUBLIC_DATASET:
                suffixes.append("-" + value.removeprefix(PUBLIC_DATASET).lstrip("_"))
        else:
            entry = lock.get(value) if isinstance(value, str) else None
            if entry is None:
                raise ScenarioError(f"overrides.dataset: {value!r} is not in the dataset lock; available: {', '.join(lock)}")
            if entry.entries is None:
                raise ScenarioError(f"overrides.dataset: lock entry {value} has no entries count")
            dataset["name"], dataset["entries"] = value, entry.entries
    if "max_context_length" in overrides:
        value = overrides["max_context_length"]
        if value is None:
            dataset.pop("maxContextLength", None)
            suffixes.append("-1m")
        else:
            dataset["maxContextLength"] = int(_positive("overrides.max_context_length", value))
            suffixes.append(f"-ctx{dataset['maxContextLength']}")
    if "generation" in overrides:
        value = overrides["generation"]
        if not isinstance(value, dict) or not value or set(value) - set(GENERATION_KEYS):
            raise ScenarioError(f"overrides.generation: a mapping of {', '.join(GENERATION_KEYS)}")
        extra = doc["benchmark"]["endpoint"].setdefault("extra", {})
        extra.update({key: float(item) for key, item in value.items()})
        suffixes.append("-topp95" if value == {"top_p": 0.95, "temperature": 1.0} else "-gen")
    if "trace_idle_gap_cap" in overrides:
        value = overrides["trace_idle_gap_cap"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ScenarioError(f"overrides.trace_idle_gap_cap: seconds >= 0, got {value!r}")
        dataset["traceIdleGapCapSeconds"] = float(value)
        suffixes.append("-nothink" if value == 0 else f"-idle{value:g}")
    if "duration" in overrides:
        value = _positive("overrides.duration", overrides["duration"])
        _profiling(doc)["duration"] = value
        suffixes.append("-smoke" if value == 180 else f"-d{int(value)}")
    return suffixes


def _synthetic(point: Point, search: bool, slo: str | None) -> dict[str, Any]:
    if point.isl not in ISL:
        raise ScenarioError(f"workload.isl: {point.isl!r} is not one of {', '.join(ISL)}")
    if point.osl not in OSL:
        raise ScenarioError(f"workload.osl: {point.osl!r} is not one of {', '.join(OSL)}")
    if point.cache not in CACHE:
        raise ScenarioError(f"workload.cache: {point.cache!r} is not one of {', '.join(map(str, CACHE))}")
    if slo and not search:
        raise ScenarioError("slo: only applies with workload.search: true")
    isl = ISL[point.isl]
    prefix = round(isl * point.cache / 100)  # added on top of isl, so isl - prefix keeps the total
    doc = load_base("synthetic")
    dataset = doc["benchmark"]["datasets"][0]
    dataset["isl"], dataset["osl"] = isl - prefix, OSL[point.osl]
    dataset.pop("prefix_prompts", None)
    if prefix:
        dataset["prefix_prompts"] = {"shared_system_length": prefix}
    if search:
        try:
            filters = parse_slo(slo or DEFAULT_SLO)
        except SLOError as error:
            raise ScenarioError(str(error)) from None
        doc["sweep"] = {
            "type": "adaptive_search",
            "planner": "smooth_isotonic",
            "objectives": [{"metric": "output_token_throughput", "stat": "avg", "direction": "maximize"}],
            "search_space": [{"path": "phases.profiling.concurrency", "lo": 1, "hi": 512, "kind": "int"}],
            "sla_filters": [
                {"metric_tag": s.metric, "stat": s.stat, "op": s.op, "threshold": s.threshold} for s in filters
            ],
            "max_iterations": 16,
            "sla_precision": "normal",  # 1000 requests/probe cap
        }
    return doc


def render(family: str, overrides: dict[str, Any] | None = None, point: Point | None = None,
           search: bool = False, slo: str | None = None,
           lock: dict[str, LockEntry] | None = None) -> Rendered:
    """The final AIPerf config of ``family`` with its variant applied."""
    overrides = dict(overrides or {})
    lock = load_lock() if lock is None else lock
    if family == "synthetic":
        if point is None:
            raise ScenarioError("workload: synthetic needs isl, osl and cache")
        if extra := sorted(set(overrides) - set(SYNTHETIC_OVERRIDES)):
            raise ScenarioError(f"overrides: synthetic only takes {', '.join(SYNTHETIC_OVERRIDES)}, not {', '.join(extra)}")
        doc = _synthetic(point, search, slo)
        suffixes = _apply(doc, family, overrides, lock)
        label = f"{'sweep' if search else 'fixed'}-{point.label}{''.join(suffixes)}"
        return Rendered(doc, label, "synthetic", "synthetic", search, _profiling(doc)["duration"])
    if point is not None or search or slo:
        raise ScenarioError(f"workload/slo: only for scenario synthetic, not {family}")
    doc = load_base(family)
    suffixes = _apply(doc, family, overrides, lock)
    dataset = doc["benchmark"]["datasets"][0]
    if dataset["type"] == "public":
        kind, name = "public", dataset["dataset"]
    else:
        kind, name = "file", dataset["name"]
        if name not in lock:
            raise ScenarioError(f"scenario: dataset {name} of {family} is not in the dataset lock")
    base = LABELS.get(family) or LIVEKIT_LABEL + name.removeprefix("livekit-111868-")
    return Rendered(doc, base + "".join(suffixes), kind, name, False, _profiling(doc)["duration"])
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTEST infx/tests/serving/test_scenarios.py infx/tests/serving/test_slo.py`
Expected: all pass. If a parity case fails, print `yaml.safe_dump` of both docs and fix the base file or `_apply`, never the legacy fixture.

- [ ] **Step 6: Commit**

```bash
git add inferencex-e2e/benchmarks/single_node/serving/scenarios inferencex-e2e/infx/serving/slo.py inferencex-e2e/infx/serving/scenarios.py inferencex-e2e/infx/tests/serving/legacy_scenarios inferencex-e2e/infx/tests/serving/test_scenarios.py inferencex-e2e/infx/tests/serving/test_slo.py
git commit -m "feat(serving): six base scenarios plus overrides render the legacy scenario files"
```

---

### Task 3: VNG cluster records and runner labels

**Files:**
- Modify: `inferencex-e2e/configs/runners.yaml` (append to `labels:` and to `clusters:`)
- Test: `inferencex-e2e/infx/tests/serving/test_clusters.py`

**Interfaces:**
- Produces: clusters `hcm-hgx01`, `hcm-hgx02`, `hcm-rtx01`, `hcm-rtx02`; runner slots `hcm-hgx0N_00..03`, `hcm-rtx0N_00..01`; pool labels `h200-hcm`, `rtx6000pro-hcm`. Every VNG cluster id starts with `hcm-`.

- [ ] **Step 1: Write the failing test**

`inferencex-e2e/infx/tests/serving/test_clusters.py`:

```python
import pytest

from infx.clusters import load_inventory

NODES = {
    "hcm-hgx01": ("hgx-h200-01", 4, "gpu:nvidia_h200:{gpus}", 4),
    "hcm-hgx02": ("hgx-h200-02", 8, "gpu:nvidia_h200:{gpus}", 4),
    "hcm-rtx01": ("hoanq3-rtx6000x2-slurm-1", 2, "gpu:nvidia_rtx6000pro:{gpus}", 2),
    "hcm-rtx02": ("hoanq3-rtx6000x2-slurm-2", 2, "gpu:nvidia_rtx6000pro:{gpus}", 2),
}


@pytest.mark.parametrize("cluster_id", NODES)
def test_vng_node_is_one_cluster_pinned_and_shared(cluster_id):
    node, gpus, gres, slots = NODES[cluster_id]
    inventory = load_inventory()
    runners = inventory.labels[f"cluster:{cluster_id}"]
    assert runners == [f"{cluster_id}_{index:02d}" for index in range(slots)]
    cluster = inventory.cluster_for(runners[0])
    settings = cluster.scheduler_settings
    assert cluster.gpus_per_node == gpus
    assert settings.exclusive is False
    assert settings.gres == gres
    assert f"--nodelist={node}" in settings.salloc_args
    assert "--cpus-per-gpu=16" in settings.salloc_args
    assert any(arg.startswith("--mem-per-gpu=") for arg in settings.salloc_args)
    assert "hf-hub-cache" in settings.volumes
    assert settings.squash.import_mode == "submit-host" and settings.squash.single_node_import


def test_pool_labels_list_every_slot():
    labels = load_inventory().labels
    assert labels["h200-hcm"] == labels["cluster:hcm-hgx01"] + labels["cluster:hcm-hgx02"]
    assert labels["rtx6000pro-hcm"] == labels["cluster:hcm-rtx01"] + labels["cluster:hcm-rtx02"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTEST infx/tests/serving/test_clusters.py`
Expected: FAIL with `KeyError: 'cluster:hcm-hgx01'`.

- [ ] **Step 3: Add the labels**

Append under `labels:` in `inferencex-e2e/configs/runners.yaml` (block style, like the existing entries):

```yaml
  # VNG on-node runners (stackops, systemd), one cluster per node: the runner lives on its GPU
  # node, and every host path below is node-local. See .agents/skills/serving-runner-setup.
  cluster:hcm-hgx01:
    - hcm-hgx01_00
    - hcm-hgx01_01
    - hcm-hgx01_02
    - hcm-hgx01_03
  cluster:hcm-hgx02:
    - hcm-hgx02_00
    - hcm-hgx02_01
    - hcm-hgx02_02
    - hcm-hgx02_03
  cluster:hcm-rtx01:
    - hcm-rtx01_00
    - hcm-rtx01_01
  cluster:hcm-rtx02:
    - hcm-rtx02_00
    - hcm-rtx02_01
  h200-hcm:
    - hcm-hgx01_00
    - hcm-hgx01_01
    - hcm-hgx01_02
    - hcm-hgx01_03
    - hcm-hgx02_00
    - hcm-hgx02_01
    - hcm-hgx02_02
    - hcm-hgx02_03
  rtx6000pro-hcm:
    - hcm-rtx01_00
    - hcm-rtx01_01
    - hcm-rtx02_00
    - hcm-rtx02_01
```

- [ ] **Step 4: Add the cluster records**

Append under `clusters:`:

```yaml
  # VNG research Slurm cluster (test-cluster): cons_tres + CR_CORE_MEMORY, nodes shared with the
  # team, never --exclusive. salloc-args right-size every job (sbatch gets them too) and pin
  # it to the runner's own node, whose local disk holds the HF cache and squash files.
  hcm-hgx01:
    gpus-per-node: 4  # Slurm gres declares 4 of the 8 H200 (2026-10-25); raise when infra fixes it
    available-cpu-dram-mib: 2_050_000
    arch: x86_64
    scheduler: slurm
    slurm:
      partition: hardware-hcm
      account: dev
      exclusive: false
      gres: "gpu:nvidia_h200:{gpus}"
      salloc-args: [--nodelist=hgx-h200-01, --cpus-per-gpu=16, --mem-per-gpu=240000M]
      volumes:
        hf-hub-cache: {path: /mnt/hf_hub_cache}
      squash:
        dir: /mnt/containers
        visibility: shared
        import: submit-host
        single-node-import: true
  hcm-hgx02:
    gpus-per-node: 8
    available-cpu-dram-mib: 2_050_000
    arch: x86_64
    scheduler: slurm
    slurm:
      partition: hardware-hcm
      account: dev
      exclusive: false
      gres: "gpu:nvidia_h200:{gpus}"
      salloc-args: [--nodelist=hgx-h200-02, --cpus-per-gpu=16, --mem-per-gpu=240000M]
      volumes:
        hf-hub-cache: {path: /mnt/hf_hub_cache}
      squash:
        dir: /mnt/containers
        visibility: shared
        import: submit-host
        single-node-import: true
  hcm-rtx01:
    gpus-per-node: 2
    available-cpu-dram-mib: 241_000
    arch: x86_64
    scheduler: slurm
    slurm:
      partition: rtx6000pro-hcm
      account: dev
      exclusive: false
      gres: "gpu:nvidia_rtx6000pro:{gpus}"
      salloc-args: [--nodelist=hoanq3-rtx6000x2-slurm-1, --cpus-per-gpu=16, --mem-per-gpu=110000M]
      volumes:
        hf-hub-cache: {path: /data/hf_hub_cache}
      squash:
        dir: /data/containers
        visibility: shared
        import: submit-host
        single-node-import: true
  hcm-rtx02:
    gpus-per-node: 2
    available-cpu-dram-mib: 241_000
    arch: x86_64
    scheduler: slurm
    slurm:
      partition: rtx6000pro-hcm
      account: dev
      exclusive: false
      gres: "gpu:nvidia_rtx6000pro:{gpus}"
      salloc-args: [--nodelist=hoanq3-rtx6000x2-slurm-2, --cpus-per-gpu=16, --mem-per-gpu=110000M]
      volumes:
        hf-hub-cache: {path: /data/hf_hub_cache}
      squash:
        dir: /data/containers
        visibility: shared
        import: submit-host
        single-node-import: true
```

- [ ] **Step 5: Run the new test and the upstream inventory tests**

Run: `PYTEST infx/tests/serving/test_clusters.py infx/tests/clusters infx/tests/launch/test_launch_policy.py`
Expected: all pass. If an upstream test enumerates every cluster and needs a field the VNG records lack (for example an `srt-slurm:` block), add the smallest field that satisfies it to the four records and note it in the commit message. Do not edit the upstream test.

- [ ] **Step 6: Commit**

```bash
git add inferencex-e2e/configs/runners.yaml inferencex-e2e/infx/tests/serving/test_clusters.py
git commit -m "infra(runners): one cluster per VNG node, right-sized and never exclusive"
```

---

### Task 4: Serving YAML schema and command rendering

**Files:**
- Create: `inferencex-e2e/infx/serving/schema.py`
- Create: `inferencex-e2e/infx/serving/render.py` (port, unchanged)
- Test: `inferencex-e2e/infx/tests/serving/test_schema.py`, `inferencex-e2e/infx/tests/serving/test_render.py` (port)

**Interfaces:**
- Consumes: `scenarios.render`, `Point`, `Rendered`, `ScenarioError` (Task 2); `slo.SLOError` (Task 2); `infx.clusters.load_inventory`, `RunnerInventory` (upstream; records from Task 3).
- Produces:
  - `ServingError(ValueError)`
  - `ServingSpec` with fields `name, runner, image, model, gpus, replicas, precision, env, serve, router, storage, scenario, overrides, workload, slo, conc`
  - `Workload(isl: list[str], osl: list[str], cache: list[int], search: bool)` with `.points() -> list[Point]`
  - `RouterSpec`, `StorageSpec` (unchanged shape)
  - `runner_gpus(runner: str, inventory: RunnerInventory) -> int`
  - `variants(spec: ServingSpec) -> list[Rendered]`
  - `load_serving(text: str, inventory: RunnerInventory | None = None) -> ServingSpec`
  - `render.render(spec) -> dict[str, str]` with keys `framework, serve-cmd, serve-env, served-model-name, router-image, router-cmd, router-env, router-profile, router-metrics-port, storage-image, storage-cmd, storage-env` (unchanged).

- [ ] **Step 1: Port `render.py` and its tests**

```bash
E=inferencex-e2e
git show vng-benchmark:infx/serving/render.py > $E/infx/serving/render.py
git show vng-benchmark:utils/test_serving_render.py > $E/infx/tests/serving/test_render.py
sed -i 's/runner="h200-greennode-slurm-1x_01"/runner="cluster:hcm-hgx02"/; s/runner="hardware-hcm-8x_02"/runner="cluster:hcm-hgx02"/g; s/scenario="agentx-cc-062126-smoke"/scenario="agentx"/' $E/infx/tests/serving/test_render.py
```

`test_render.py` only builds `ServingSpec.model_validate(...)` and calls `render`, `split_serve` and `infer_framework`. After these three replacements it needs nothing else.

- [ ] **Step 2: Write the failing schema tests**

`inferencex-e2e/infx/tests/serving/test_schema.py`:

```python
import textwrap

import pytest

from infx.serving.schema import ServingError, load_serving, runner_gpus, variants
from infx.clusters import load_inventory

BASE = """\
name: qwen-tp1
runner: cluster:hcm-hgx02
image: vllm/vllm-openai:v0.29.0
model: Qwen/Qwen3.8-27B
gpus: 1
serve: vllm serve Qwen/Qwen3.8-27B --max-model-len 262144
"""


def yaml_of(extra: str) -> str:
    return BASE + textwrap.dedent(extra)


def errors(text: str) -> str:
    with pytest.raises(ServingError) as caught:
        load_serving(text)
    return str(caught.value)


def test_engine_only_agentx_with_smoke_override():
    spec = load_serving(yaml_of("scenario: agentx\noverrides: {duration: 180}\nconc: [8, 16]\n"))
    assert spec.conc == [8, 16]
    [variant] = variants(spec)
    assert variant.label == "agentx-cc-062126-smoke"


def test_synthetic_grid_expands_to_points():
    spec = load_serving(yaml_of(
        "scenario: synthetic\nworkload: {isl: [8k, 32k], osl: 1k, cache: [0, 50], search: true}\nconc: 1\n"))
    assert [v.label for v in variants(spec)] == [
        "sweep-isl8k-osl1k-cache0", "sweep-isl8k-osl1k-cache50",
        "sweep-isl32k-osl1k-cache0", "sweep-isl32k-osl1k-cache50",
    ]


def test_osl_numbers_read_as_labels():
    spec = load_serving(yaml_of("scenario: synthetic\nworkload: {isl: 8k, osl: 512, cache: 0}\nconc: [4]\n"))
    assert variants(spec)[0].label == "fixed-isl8k-osl512-cache0"


@pytest.mark.parametrize("extra, message", [
    ("scenario: agentx\noverrides: {warmup: 2}\nconc: 8\n", "overrides: unknown key(s) warmup"),
    ("scenario: synthetic\nconc: 1\n", "workload: synthetic needs"),
    ("scenario: agentx\nworkload: {isl: 8k, osl: 1k, cache: 0}\nconc: 1\n", "workload/slo: only for scenario synthetic"),
    ("scenario: synthetic\nworkload: {isl: 8k, osl: 1k, cache: 0, search: true}\nconc: [8, 16]\n",
     "conc: workload.search finds the concurrency itself"),
    ("scenario: synthetic\nworkload: {isl: [8k, 32k], osl: 1k, cache: 0}\nconc: [8, 16]\n",
     "conc: a workload grid runs one job per point"),
    ("scenario: agentx\nconc: 8\nenv: {PORT: '1'}\n", "env.PORT: reserved"),
    ("scenario: agentx\nconc: 8\nduration: 180\n", "duration: set it with overrides"),
    ("scenario: agentx\nconc: 8\nreplicas: 2\n", "replicas: gpus 1 is not a multiple of replicas 2"),
    ("scenario: agentx-cc-062126\nconc: 8\n", "scenario:"),
])
def test_bad_yaml_names_field_and_fix(extra, message):
    assert message in errors(yaml_of(extra))


def test_runner_must_be_a_vng_cluster_slot_or_pool():
    inventory = load_inventory()
    assert runner_gpus("cluster:hcm-hgx02", inventory) == 8
    assert runner_gpus("hcm-hgx02_01", inventory) == 8
    assert runner_gpus("h200-hcm", inventory) == 4  # the smallest node of the pool
    with pytest.raises(ServingError, match="runner: h100-cw_00 is not a VNG serving runner"):
        runner_gpus("h100-cw_00", inventory)


def test_gpus_above_the_node_fail():
    text = BASE.replace("runner: cluster:hcm-hgx02", "runner: cluster:hcm-hgx01").replace("gpus: 1", "gpus: 8")
    assert "gpus: 8 > 4 GPUs per node" in errors(text + "scenario: agentx\nconc: 8\n")
```

- [ ] **Step 3: Run them to verify they fail**

Run: `PYTEST infx/tests/serving/test_schema.py infx/tests/serving/test_render.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'infx.serving.schema'`.

- [ ] **Step 4: Implement `schema.py`**

`inferencex-e2e/infx/serving/schema.py`:

```python
"""Serving YAML: the one file a team member writes to benchmark a serving config.

Engine command (serve:), optional router and storage sidecars, replicas, and the scenario
with its variant (overrides / workload / slo). Every error is one line
"<field>: <problem and fix>" so a person or an agent can act on it.
Spec: docs/superpowers/specs/2026-10-25-vng-serving-refactor-design.md
"""

from __future__ import annotations

import itertools
import re
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from infx.clusters import RunnerInventory, load_inventory
from infx.serving import scenarios
from infx.serving.scenarios import Point, Rendered, ScenarioError

NAME_PATTERN = r"^[a-z0-9][a-z0-9.-]*$"
LOCAL_PATH_MARKERS = ("/models/", "/mnt/", "/data/")
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# image is spliced into srun arguments on the GPU node; model is passed to the client argv.
IMAGE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@-]*")
MODEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+")
VNG_CLUSTER_PREFIX = "hcm-"
# Variables job.sh / bench.sh / sidecar.sh / client.py set themselves.
RESERVED_ENV = (
    "PORT", "MODEL", "CONC", "RESULT_DIR", "HF_HUB_CACHE", "FRAMEWORK",
    "SERVE_CMD", "SERVE_ENV", "SERVED_MODEL_NAME", "SERVING_YAML", "SERVING_SCENARIO_YAML",
    "SERVING_SCENARIO_LABEL", "SERVING_DATASET", "SERVING_DATASET_KIND", "SERVING_SEARCH",
    "SERVING_ENGINE_IMAGE", "SERVING_DCGM_IMAGE", "SERVING_HF_CACHE_HOST", "IX_CUSTOM_DATASET_FILE",
    "AIPERF_SERVER_METRICS_URLS", "AIPERF_GPU_TELEMETRY_URL",
    "DYN_DISCOVERY_BACKEND", "DYN_FILE_KV", "DYN_SYSTEM_ENABLED", "DYN_SYSTEM_PORT",
    "ROUTER_IMAGE", "ROUTER_CMD", "ROUTER_ENV", "ROUTER_PROFILE", "ROUTER_METRICS_PORT",
    "ENGINE_PORT", "ENGINE_PORTS", "REPLICAS",
    "STORAGE_IMAGE", "STORAGE_CMD", "STORAGE_ENV", "STORAGE_MASTER_PORT", "STORAGE_META_PORT",
)  # fmt: skip
# {{name}} placeholders sidecar.sh / bench.sh expand at run time; {{ x }} (spaces) is inline Jinja.
ROUTER_PLACEHOLDERS = ("router_port", "engine_url", "engine_port", "engine_urls",
                       "storage_master_port", "storage_meta_port")
ENV_PLACEHOLDERS = ("storage_master_port", "storage_meta_port")
STORAGE_PLACEHOLDERS = ("storage_master_port", "storage_meta_port")
PLACEHOLDER = re.compile(r"\{\{([a-z_]+)\}\}")


class ServingError(ValueError):
    """Validation failure; each line names a field and the fix."""


def _coerce_env(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    out = {}
    for key, item in value.items():
        if isinstance(item, bool):
            raise ValueError(f'{key}: quote boolean values, e.g. "1" or "true"')
        out[key] = str(item) if isinstance(item, (int, float)) else item
    return out


def _listed(value: Any) -> Any:
    return value if isinstance(value, list) else [value]


class RouterSpec(BaseModel):
    """A router sidecar in front of the engine(s)."""

    model_config = ConfigDict(extra="forbid")

    image: str = Field(min_length=1)
    serve: str = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)
    metrics_port: int | None = Field(default=None, ge=1, le=65535)

    @field_validator("env", mode="before")
    @classmethod
    def _env_strings(cls, value: Any) -> Any:
        return _coerce_env(value)


class StorageSpec(BaseModel):
    """A non-GPU storage sidecar started before the engine (e.g. Mooncake master)."""

    model_config = ConfigDict(extra="forbid")

    image: str = Field(min_length=1)
    serve: str = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("env", mode="before")
    @classmethod
    def _env_strings(cls, value: Any) -> Any:
        return _coerce_env(value)


class Workload(BaseModel):
    """The synthetic grid: one job per (isl, osl, cache) point."""

    model_config = ConfigDict(extra="forbid")

    isl: list[str] = Field(min_length=1)
    osl: list[str] = Field(min_length=1)
    cache: list[int] = Field(min_length=1)
    search: bool = False

    @field_validator("isl", "osl", mode="before")
    @classmethod
    def _labels(cls, value: Any) -> Any:
        return [str(item) for item in _listed(value)]

    @field_validator("cache", mode="before")
    @classmethod
    def _cache(cls, value: Any) -> Any:
        return _listed(value)

    def points(self) -> list[Point]:
        return [Point(i, o, c) for i, o, c in itertools.product(self.isl, self.osl, self.cache)]


class ServingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=NAME_PATTERN)
    runner: str = Field(min_length=1)
    image: str = Field(min_length=1)
    model: str = Field(min_length=1)
    gpus: int = Field(gt=0)
    replicas: int = Field(default=1, ge=1)
    precision: str = Field(default="bf16", pattern=r"^[a-z0-9]+$")
    env: dict[str, str] = Field(default_factory=dict)
    serve: str = Field(min_length=1)
    router: RouterSpec | None = None
    storage: StorageSpec | None = None
    scenario: Literal["agentx", "tau2", "livekit", "shared-prefix-e2", "hidden-prefix-e3", "synthetic"]
    overrides: dict[str, Any] = Field(default_factory=dict)
    workload: Workload | None = None
    slo: str | None = None
    conc: list[Annotated[int, Field(gt=0)]] = Field(min_length=1)

    @field_validator("conc", mode="before")
    @classmethod
    def _scalar_conc(cls, value: Any) -> Any:
        return [value] if isinstance(value, int) and not isinstance(value, bool) else value

    @field_validator("env", mode="before")
    @classmethod
    def _env_strings(cls, value: Any) -> Any:
        return _coerce_env(value)


def _format_validation_error(error: ValidationError) -> str:
    models = {"router": RouterSpec, "storage": StorageSpec, "workload": Workload}
    lines = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "serving"
        message = item["msg"]
        if item["type"] == "extra_forbidden":
            model = models.get(str(item["loc"][0]), ServingSpec) if len(item["loc"]) > 1 else ServingSpec
            message = "unknown key (allowed: " + ", ".join(model.model_fields) + ")"
        elif item["loc"] == ("scenario",):
            message = f"must be one of {', '.join(scenarios.FAMILIES)} (variants go in overrides:)"
        lines.append(f"{location}: {message}")
    return "\n".join(lines)


def runner_gpus(runner: str, inventory: RunnerInventory) -> int:
    """GPUs per node behind ``runner`` (a cluster label, a pool label or a slot name)."""
    vng = sorted(cid for cid in inventory.clusters if cid.startswith(VNG_CLUSTER_PREFIX))
    hint = f"runner: {runner} is not a VNG serving runner; use cluster:<id> with <id> in {', '.join(vng)}, or h200-hcm / rtx6000pro-hcm"
    names = inventory.labels.get(runner, [runner])
    gpus = set()
    for name in names:
        try:
            cluster = inventory.cluster_for(name)
        except ValueError:
            raise ServingError(hint) from None
        if not cluster.id.startswith(VNG_CLUSTER_PREFIX):
            raise ServingError(hint)
        gpus.add(cluster.gpus_per_node)
    return min(gpus)


def variants(spec: ServingSpec) -> list[Rendered]:
    """Every scenario variant the YAML asks for (one per workload point, or exactly one)."""
    try:
        if spec.workload is None:
            return [scenarios.render(spec.scenario, spec.overrides, slo=spec.slo)]
        return [
            scenarios.render(spec.scenario, spec.overrides, point, spec.workload.search, spec.slo)
            for point in spec.workload.points()
        ]
    except ScenarioError as error:
        raise ServingError(str(error)) from None


def _command_problems(spec: ServingSpec) -> list[str]:
    problems = []
    commands = [("serve", spec.serve)]
    commands += [("router.serve", spec.router.serve)] if spec.router else []
    commands += [("storage.serve", spec.storage.serve)] if spec.storage else []
    for field, command in commands:
        marker = next((m for m in LOCAL_PATH_MARKERS if m in command), None)
        if marker:
            problems.append(f"{field}: local path containing {marker!r}; replace it with the HF repo id from model")
        if "$" in command:
            problems.append(f"{field}: shell variables ($...) are not expanded; write literal values")
        if "--numa-node" in command:
            problems.append(f"{field}: drop --numa-node (benchmark convention)")
    envs = [("env", spec.env)]
    envs += [("router.env", spec.router.env)] if spec.router else []
    envs += [("storage.env", spec.storage.env)] if spec.storage else []
    for prefix, env in envs:
        for key, value in env.items():
            if not ENV_NAME.fullmatch(key):
                problems.append(f"{prefix}.{key}: not a valid environment variable name")
            elif key in RESERVED_ENV:
                problems.append(f"{prefix}.{key}: reserved, the benchmark sets it; remove it from env")
            if "\n" in value:
                problems.append(f"{prefix}.{key}: value must be a single line")
    return problems


def _placeholder_problems(spec: ServingSpec) -> list[str]:
    problems = []
    blocks = []
    if spec.router:
        if not IMAGE_PATTERN.fullmatch(spec.router.image):
            problems.append("router.image: only letters, digits and ._/:@- are allowed")
        blocks.append(("router", spec.router, ROUTER_PLACEHOLDERS))
    if spec.storage:
        if not IMAGE_PATTERN.fullmatch(spec.storage.image):
            problems.append("storage.image: only letters, digits and ._/:@- are allowed")
        blocks.append(("storage", spec.storage, STORAGE_PLACEHOLDERS))
    for name, block, allowed in blocks:
        texts = [(f"{name}.serve", block.serve)] + [(f"{name}.env.{k}", v) for k, v in block.env.items()]
        for field, body in texts:
            problems.extend(
                f"{field}: unknown placeholder {{{{{found}}}}}; allowed: {', '.join(allowed)}"
                for found in PLACEHOLDER.findall(body) if found not in allowed
            )
    allowed_env = ENV_PLACEHOLDERS if spec.storage else ()
    for key, value in spec.env.items():
        for found in PLACEHOLDER.findall(value):
            if found not in allowed_env:
                hint = (f"allowed: {', '.join(ENV_PLACEHOLDERS)}" if spec.storage
                        else "no placeholders are allowed in env: without a storage block")
                problems.append(f"env.{key}: unknown placeholder {{{{{found}}}}}; {hint}")
    return problems


def load_serving(text: str, inventory: RunnerInventory | None = None) -> ServingSpec:
    """Validate a serving YAML; raise ``ServingError`` listing every problem."""
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ServingError(f"serving: not valid YAML ({error})".replace("\n", " ")) from error
    if not isinstance(raw, dict):
        raise ServingError("serving: the top level must be a mapping of keys")
    if "duration" in raw:
        raise ServingError("duration: set it with overrides: {duration: 180} (180 s is the smoke run)")
    try:
        spec = ServingSpec.model_validate(raw)
    except ValidationError as error:
        raise ServingError(_format_validation_error(error)) from error

    problems: list[str] = []
    if not IMAGE_PATTERN.fullmatch(spec.image):
        problems.append("image: only letters, digits and ._/:@- are allowed (e.g. vllm/vllm-openai:v0.29.0)")
    if spec.model.startswith("/"):
        problems.append("model: use the HF repo id (e.g. Qwen/Qwen3.8-27B), not a local path")
    elif not MODEL_PATTERN.fullmatch(spec.model):
        problems.append("model: must be an HF repo id <org>/<name> using letters, digits and ._-")
    problems += _command_problems(spec)
    problems += _placeholder_problems(spec)
    if len(set(spec.conc)) != len(spec.conc):
        problems.append("conc: duplicate values")
    if spec.gpus % spec.replicas:
        problems.append(f"replicas: gpus {spec.gpus} is not a multiple of replicas {spec.replicas}; "
                        "each replica gets gpus/replicas GPUs")
    if spec.replicas > 1:
        if spec.router is None:
            problems.append("replicas: more than 1 replica needs a router: block whose serve lists {{engine_urls}}")
        elif "dynamo.frontend" in spec.router.serve:
            problems.append("replicas: the Dynamo frontend discovers its engines itself; use replicas: 1")
    try:
        rendered = variants(spec)
    except ServingError as error:
        problems.append(str(error))
        rendered = []
    if any(variant.search for variant in rendered) and spec.conc != [1]:
        problems.append(f"conc: workload.search finds the concurrency itself; set conc: 1 (got {spec.conc})")
    elif len(rendered) > 1 and len(spec.conc) > 1:
        problems.append(f"conc: a workload grid runs one job per point; set exactly one conc (got {len(spec.conc)})")
    try:
        per_node = runner_gpus(spec.runner, inventory or load_inventory())
        if spec.gpus > per_node:
            problems.append(f"gpus: {spec.gpus} > {per_node} GPUs per node on runner {spec.runner}; pick a bigger node")
    except ServingError as error:
        problems.append(str(error))
    if problems:
        raise ServingError("\n".join(problems))
    return spec
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTEST infx/tests/serving/test_schema.py infx/tests/serving/test_render.py`
Expected: all pass. `render.py` imports `ServingError, ServingSpec` from `infx.serving.schema`, and both still exist.

- [ ] **Step 6: Commit**

```bash
git add inferencex-e2e/infx/serving/schema.py inferencex-e2e/infx/serving/render.py inferencex-e2e/infx/tests/serving/test_schema.py inferencex-e2e/infx/tests/serving/test_render.py
git commit -m "feat(serving): serving YAML schema with scenario variants and VNG cluster runners"
```

---

### Task 5: Matrix points, the serving workflow and dispatch

**Files:**
- Create: `inferencex-e2e/infx/serving/to_matrix.py`
- Create: `.github/workflows/serving.yml`
- Create: `inferencex-e2e/utils/serving/dispatch.sh`
- Test: `inferencex-e2e/infx/tests/serving/test_to_matrix.py`

**Interfaces:**
- Consumes: `schema.load_serving`, `schema.variants` (Task 4); `render.render` (Task 4).
- Produces:
  - `build(text: str, inventory=None) -> {"points": [{"config": row, "serving": str}], "title": str}`. `row` has exactly the keys of `_row` below. `serving` is the JSON string Task 8's `Serving` model reads, with keys `framework, serve_cmd, serve_env, served_model_name, router_image, router_cmd, router_env, router_profile, router_metrics_port, replicas, storage_image, storage_cmd, storage_env, scenario_yaml, scenario_label, dataset_kind, dataset, search, yaml`.
  - CLI: `python -P -m infx.serving.to_matrix <file>` prints `json.dumps(points)`; `--check` prints the title; errors print `ERROR <line>` per problem and exit 2.
  - Workflow `serving.yml` with inputs `serving` (YAML text) and `test-name`; run name `serving - <test-name>`.

- [ ] **Step 1: Write the failing tests**

`inferencex-e2e/infx/tests/serving/test_to_matrix.py`:

```python
import json

import yaml

from infx.serving import scenarios
from infx.serving.to_matrix import build, main

ENGINE = """\
name: qwen-tp1
runner: cluster:hcm-hgx02
image: vllm/vllm-openai:v0.29.0
model: Qwen/Qwen3.8-27B
gpus: 1
serve: vllm serve Qwen/Qwen3.8-27B --max-model-len 262144 --port 9000
"""
ROW_KEYS = {
    "image", "model", "model-prefix", "precision", "framework", "runner", "tp", "pp", "dcp-size",
    "pcp-size", "ep", "dp-attn", "spec-decoding", "conc", "kv-offloading", "total-cpu-dram-gb",
    "duration", "exp-name", "scenario-type",
}


def test_one_point_per_conc_with_the_rendered_scenario():
    result = build(ENGINE + "scenario: agentx\noverrides: {duration: 180}\nconc: [8, 16]\n")
    points = result["points"]
    assert [p["config"]["exp-name"] for p in points] == ["qwen-tp1_c8", "qwen-tp1_c16"]
    assert set(points[0]["config"]) == ROW_KEYS
    assert points[0]["config"]["duration"] == 180 and points[0]["config"]["tp"] == 1
    serving = json.loads(points[0]["serving"])
    assert serving["scenario_label"] == "agentx-cc-062126-smoke"
    assert serving["search"] is False and serving["dataset_kind"] == "public"
    assert yaml.safe_load(serving["scenario_yaml"]) == scenarios.render("agentx", {"duration": 180}).doc
    assert "--port" not in serving["serve_cmd"]
    assert result["title"] == "serving qwen-tp1 agentx-cc-062126-smoke (180s) c8,16 cluster:hcm-hgx02"


def test_grid_is_one_job_per_point():
    points = build(ENGINE + "scenario: synthetic\nworkload: {isl: [8k, 32k], osl: 1k, cache: 50, search: true}\nconc: 1\n")["points"]
    assert [p["config"]["exp-name"] for p in points] == [
        "qwen-tp1-sweep-isl8k-osl1k-cache50", "qwen-tp1-sweep-isl32k-osl1k-cache50"]
    assert all(json.loads(p["serving"])["search"] for p in points)


def test_single_search_point_keeps_the_search_name():
    [point] = build(ENGINE + "scenario: synthetic\nworkload: {isl: 8k, osl: 1k, cache: 0, search: true}\nconc: 1\n")["points"]
    assert point["config"]["exp-name"] == "qwen-tp1_search"


def test_cli_check_and_errors(tmp_path, capsys):
    good = tmp_path / "good.yaml"
    good.write_text(ENGINE + "scenario: tau2\nconc: 4\n")
    assert main(["--check", str(good)]) == 0
    assert capsys.readouterr().out.startswith("serving qwen-tp1 custom-agentic-tau2")
    bad = tmp_path / "bad.yaml"
    bad.write_text(ENGINE + "scenario: tau2\nconc: 4\noverrides: {nope: 1}\n")
    assert main([str(bad)]) == 2
    assert "ERROR overrides: unknown key(s) nope" in capsys.readouterr().err
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PYTEST infx/tests/serving/test_to_matrix.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'infx.serving.to_matrix'`.

- [ ] **Step 3: Implement `to_matrix.py`**

`inferencex-e2e/infx/serving/to_matrix.py`:

```python
"""Serving YAML -> the points .github/workflows/serving.yml runs.

    python -P -m infx.serving.to_matrix serving.yaml          # [{"config": row, "serving": json}, ...]
    python -P -m infx.serving.to_matrix --check serving.yaml  # the run title only

``config`` is the benchmark-tmpl.yml row; ``serving`` is SERVING_JSON for
infx/launch/drivers/serving.py, with the rendered scenario inside.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from infx.serving.render import render
from infx.serving.scenarios import Rendered
from infx.serving.schema import ServingError, ServingSpec, load_serving, variants


def _row(spec: ServingSpec, framework: str, conc: int, duration: int, exp_name: str) -> dict:
    return {
        "image": spec.image, "model": spec.model, "model-prefix": spec.name,
        "precision": spec.precision, "framework": framework, "runner": spec.runner,
        "tp": spec.gpus, "pp": 1, "dcp-size": 1, "pcp-size": 1, "ep": 1, "dp-attn": False,
        "spec-decoding": "none", "conc": conc, "kv-offloading": "none", "total-cpu-dram-gb": 0,
        "duration": duration, "exp-name": exp_name, "scenario-type": "agentic-coding",
    }  # fmt: skip


def _serving(spec: ServingSpec, rendered: dict[str, str], variant: Rendered, text: str) -> str:
    return json.dumps({
        "framework": rendered["framework"],
        "serve_cmd": rendered["serve-cmd"],
        "serve_env": rendered["serve-env"],
        "served_model_name": rendered["served-model-name"],
        "router_image": rendered["router-image"],
        "router_cmd": rendered["router-cmd"],
        "router_env": rendered["router-env"],
        "router_profile": rendered["router-profile"],
        "router_metrics_port": rendered["router-metrics-port"],
        "replicas": spec.replicas,
        "storage_image": rendered["storage-image"],
        "storage_cmd": rendered["storage-cmd"],
        "storage_env": rendered["storage-env"],
        "scenario_yaml": variant.yaml(),
        "scenario_label": variant.label,
        "dataset_kind": variant.kind,
        "dataset": variant.dataset,
        "search": variant.search,
        "yaml": text,
    })


def build(text: str, inventory=None) -> dict:
    spec = load_serving(text, inventory)
    rendered = render(spec)
    chosen = variants(spec)
    multi = len(chosen) > 1
    points = []
    for variant in chosen:
        for conc in spec.conc:
            if multi:
                exp_name = f"{spec.name}-{variant.label}"
            else:
                exp_name = f"{spec.name}_search" if variant.search else f"{spec.name}_c{conc}"
            row = _row(spec, rendered["framework"], conc, int(variant.duration), exp_name)
            points.append({"config": row, "serving": _serving(spec, rendered, variant, text)})
    what = f"{len(chosen)} points" if multi else chosen[0].label
    durations = {int(variant.duration) for variant in chosen}
    seconds = f"{durations.pop()}s" if len(durations) == 1 else "mixed"
    concs = ",".join(str(conc) for conc in spec.conc)
    return {"points": points, "title": f"serving {spec.name} {what} ({seconds}) c{concs} {spec.runner}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path)
    parser.add_argument("--check", action="store_true", help="validate and print the run title only")
    args = parser.parse_args(argv)
    try:
        try:
            text = args.path.read_text()
        except OSError as error:
            raise ServingError(f"serving: cannot read {args.path} ({error.strerror})") from error
        result = build(text)
    except ServingError as error:
        for line in str(error).splitlines():
            print(f"ERROR {line}", file=sys.stderr)
        return 2
    print(result["title"] if args.check else json.dumps(result["points"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTEST infx/tests/serving/test_to_matrix.py`
Expected: 4 passed.

- [ ] **Step 5: Write the workflow**

`.github/workflows/serving.yml`. Copy the exact `actions/checkout` and `astral-sh/setup-uv` pinned SHAs from `.github/workflows/benchmark-tmpl.yml`; the ones below are the pins on `cfb7bdd0c`:

```yaml
name: Serving
run-name: serving - ${{ inputs.test-name }}
# VNG serving-YAML benchmarks (inferencex-e2e/docs/serving/ARCHITECTURE.md).
# Dispatch with inferencex-e2e/utils/serving/dispatch.sh, which validates first.
on:
    workflow_dispatch:
        inputs:
            serving:
                description: "Serving YAML text (inferencex-e2e/configs/serving/*.yaml)"
                required: true
                type: string
            test-name:
                description: "Run title, printed by dispatch.sh"
                required: true
                type: string

permissions:
    contents: read

jobs:
    plan:
        runs-on: ubuntu-latest
        outputs:
            points: ${{ steps.plan.outputs.points }}
        steps:
            - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
              with:
                  persist-credentials: false
            - uses: astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d # v10.0.1
            - id: plan
              working-directory: inferencex-e2e
              env:
                  SERVING_YAML: ${{ inputs.serving }}
                  PYTHONPATH: ${{ github.workspace }}/inferencex-e2e
              run: |
                  printf '%s\n' "$SERVING_YAML" > "$RUNNER_TEMP/serving.yaml"
                  points=$(uv run --no-project --python 3.12 --with pydantic --with pyyaml \
                    python -P -m infx.serving.to_matrix "$RUNNER_TEMP/serving.yaml")
                  echo "points=$points" >> "$GITHUB_OUTPUT"

    bench:
        needs: plan
        uses: $/.github/workflows/benchmark-tmpl.yml
        name: serving /
        strategy:
            fail-fast: false
            matrix:
                include: ${{ fromJSON(needs.plan.outputs.points) }}
        secrets:
            INFERENCEX_OFFICIAL_RO_HF_TOKEN: ${{ secrets.INFERENCEX_OFFICIAL_RO_HF_TOKEN }}
        with:
            config: ${{ toJSON(matrix.config) }}
            serving: ${{ matrix.serving }}
            runner: ${{ matrix.config.runner }}
            priority: "0"
            queue-token: serving
            dp-attn: false
            run-eval: false
            scenario-type: agentic-coding
            duration: ${{ matrix.config.duration }}
            test-name: ${{ inputs.test-name }}
```

The `serving` input of `benchmark-tmpl.yml` is added in Task 8. Until then this file references an input that does not exist yet; that is expected and is checked in Task 8, Step 7.

- [ ] **Step 6: Write `dispatch.sh`**

`inferencex-e2e/utils/serving/dispatch.sh` (then `chmod +x`):

```bash
#!/usr/bin/env bash
# Dispatch one serving YAML on vngcloud/InferenceX:
#   inferencex-e2e/utils/serving/dispatch.sh inferencex-e2e/configs/serving/<name>.yaml [--ref <branch>]
# Validates locally with the code CI runs, starts serving.yml with the YAML as input,
# and prints the run URL.
set -eo pipefail

E2E="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GH_REPO="vngcloud/InferenceX"

usage() {
    echo "usage: utils/serving/dispatch.sh <serving.yaml> [--ref <branch>]" >&2
    exit 2
}
[[ $# -ge 1 ]] || usage
yaml_file="$(realpath "$1")"
shift
# The workflow input is the file content, not HEAD: warn when the run would differ from git.
if [ -n "$(git -C "$E2E" status --porcelain -- "$yaml_file")" ]; then
    echo "WARNING: $yaml_file differs from HEAD; the run dispatches the working-tree content." >&2
fi
ref=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref) [[ $# -ge 2 && -n "$2" ]] || usage; ref="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

title=$(cd "$E2E" && PYTHONPATH="$E2E" uv run -q --no-project --python 3.12 --with pydantic --with pyyaml \
    python -P -m infx.serving.to_matrix --check "$yaml_file") || exit 2

if [[ -z "$ref" ]]; then
    ref="$(git -C "$E2E" rev-parse --abbrev-ref HEAD)"
    remote_sha="$(git -C "$E2E" ls-remote origin "refs/heads/$ref" | cut -f1)"
    if [[ -z "$remote_sha" ]]; then
        echo "ERROR branch $ref is not on origin; push it first (CI runs the pushed code)" >&2
        exit 2
    fi
    if [[ "$remote_sha" != "$(git -C "$E2E" rev-parse HEAD)" ]]; then
        echo "ERROR local HEAD differs from origin/$ref; push or pull first (CI runs the pushed code)" >&2
        exit 2
    fi
fi

started=$(( $(date +%s) - 60 ))  # 60 s early: tolerate a local clock ahead of GitHub's
gh workflow run serving.yml --repo "$GH_REPO" --ref "$ref" -f serving="$(cat "$yaml_file")" -f test-name="$title"
for _ in $(seq 1 30); do
    sleep 4
    run_id=$(gh run list --repo "$GH_REPO" -w serving.yml -L 20 --json databaseId,displayTitle,createdAt \
        --jq "[.[] | select(.displayTitle == \"serving - $title\" and (.createdAt | fromdateiso8601) >= $started)][0].databaseId // empty")
    if [[ -n "$run_id" ]]; then
        echo "https://github.com/$GH_REPO/actions/runs/$run_id"
        exit 0
    fi
done
echo "ERROR dispatched, but the run did not show up within 2 minutes; do not redispatch, check: gh run list --repo $GH_REPO -w serving.yml -L 5" >&2
exit 1
```

- [ ] **Step 7: Smoke the validation path locally**

Run from the worktree root:

```bash
printf 'name: x\nrunner: cluster:hcm-hgx02\nimage: vllm/vllm-openai:v0.29.0\nmodel: Qwen/Qwen3.8-27B\ngpus: 1\nserve: vllm serve Qwen/Qwen3.8-27B\nscenario: agentx\noverrides: {duration: 180}\nconc: 8\n' > /tmp/x.yaml
(cd inferencex-e2e && PYTHONPATH=$PWD uv run -q --no-project --python 3.12 --with pydantic --with pyyaml python -P -m infx.serving.to_matrix --check /tmp/x.yaml)
bash -n inferencex-e2e/utils/serving/dispatch.sh
```

Expected: `serving x agentx-cc-062126-smoke (180s) c8 cluster:hcm-hgx02`, and no syntax error.

- [ ] **Step 8: Commit**

```bash
git add inferencex-e2e/infx/serving/to_matrix.py inferencex-e2e/infx/tests/serving/test_to_matrix.py .github/workflows/serving.yml inferencex-e2e/utils/serving/dispatch.sh
git commit -m "feat(serving): matrix points, serving.yml workflow and one-command dispatch"
```

---

### Task 6: In-container client and the search result writer

**Files:**
- Create: `inferencex-e2e/infx/serving/client.py`
- Create: `inferencex-e2e/infx/serving/sweep_result.py` (port of `vng-benchmark:infx/results/sweep.py`)
- Create: `inferencex-e2e/infx/tests/serving/fixtures/sweep/` (port of `vng-benchmark:utils/fixtures/sweep/`)
- Test: `inferencex-e2e/infx/tests/serving/test_client.py`, `inferencex-e2e/infx/tests/serving/test_sweep_result.py` (port)

**Interfaces:**
- Consumes: `infx.serving.datasets.fetch`, `DatasetError` (Task 1); upstream `infx.bench.agentic.venv.Runtime`, `bootstrap`; upstream `infx.bench.proc.call(argv, env)`, `proc.tee(argv, log, env)`, `proc.pythonpath(environ)`; upstream CLIs `infx.results.agentic.process_agentic_result`, `analyze_benchmark_distributions`, `validate_agentic_result`.
- Env it reads (set by `job.sh` / `bench.sh` / `benchmark-tmpl.yml`): `RESULT_DIR, RESULT_FILENAME, AGENTIC_OUTPUT_DIR, MODEL, SERVED_MODEL_NAME, CONC, PORT, SERVING_SCENARIO_YAML, SERVING_SCENARIO_LABEL, SERVING_DATASET_KIND, SERVING_DATASET, SERVING_SEARCH, AIPERF_SERVER_METRICS_URLS, AIPERF_GPU_TELEMETRY_URL`.
- Produces: CLI `python3 -m infx.serving.client prepare|run`; `agentx_argv(agentx: str, scenario: Path, env: Mapping[str, str]) -> list[str]`; `label_result(path: Path, label: str) -> None`; CLI `python -m infx.serving.sweep_result` (writes `$AGENTIC_OUTPUT_DIR/$RESULT_FILENAME.json`).

- [ ] **Step 1: Port the sweep writer and its tests**

```bash
E=inferencex-e2e
git show vng-benchmark:infx/results/sweep.py > $E/infx/serving/sweep_result.py
git show vng-benchmark:utils/test_sweep_result.py > $E/infx/tests/serving/test_sweep_result.py
mkdir -p $E/infx/tests/serving/fixtures
git archive vng-benchmark utils/fixtures/sweep | tar -x -C /tmp && cp -r /tmp/utils/fixtures/sweep $E/infx/tests/serving/fixtures/
sed -i 's/from infx.results.sweep import/from infx.serving.sweep_result import/; s/"SERVING_SCENARIO"/"SERVING_SCENARIO_LABEL"/g; s/SERVING_SCENARIO=/SERVING_SCENARIO_LABEL=/g' $E/infx/tests/serving/test_sweep_result.py
```

Edit `infx/serving/sweep_result.py`:

1. Replace `from infx.results.agentic import _validate_kv_offload_env, identity_fields` with:

```python
# ponytail: private helpers of infx.results.agentic; a rename upstream breaks test_sweep_result.py,
# copy them here if that happens.
from infx.results.agentic import (
    _env_bool,
    _env_int,
    _gpu_shape,
    _optional_component_metadata,
    _validate_kv_offload_env,
)
```

and add this function above `summarize_search`:

```python
def identity_fields(env: Mapping[str, str]) -> dict:
    """The env-derived identity block of an agentic result, without kv-offload and request counts."""
    multinode_fields, num_gpus, tp, ep, dp_attention = _gpu_shape(env)
    fields = {
        "hw": env.get("RUNNER_TYPE", ""),
        "conc": int(env.get("CONC", "0")),
        "image": env.get("IMAGE", ""),
        "recipe_fingerprint": env.get("RECIPE_FINGERPRINT", ""),
        "model": env.get("MODEL", ""),
        "infmax_model_prefix": env.get("MODEL_PREFIX", ""),
        "framework": env.get("FRAMEWORK", ""),
        "precision": env.get("PRECISION", ""),
        "spec_decoding": env.get("SPEC_DECODING", "none"),
        "disagg": _env_bool(env, "DISAGG"),
        "scenario_type": "agentic-coding",
        "is_multinode": _env_bool(env, "IS_MULTINODE"),
        "num_gpus": num_gpus,
        "tp": tp,
        "ep": ep,
        "dp_attention": dp_attention,
        "allocated_cpu_dram_gb": _env_int(env, "TOTAL_CPU_DRAM_GB"),
        **multinode_fields,
    }
    if (router := _optional_component_metadata(env, "ROUTER_METADATA")) is not None:
        fields["router"] = router
    return fields
```

2. In `_nominal_cache_pct`, read `env.get("SERVING_SCENARIO_LABEL", "")`.
3. In `build_sweep_result`, set `result["scenario"] = env.get("SERVING_SCENARIO_LABEL", "")`.
4. Update the module docstring's first line to `"""Turn an AIPerf adaptive-search run of a serving point into one InferenceX result JSON.`

Run: `PYTEST infx/tests/serving/test_sweep_result.py`
Expected: all pass. A failing `_gpu_shape` call means its upstream signature changed: read `infx/results/agentic/__init__.py:_gpu_shape` and adapt the unpacking.

- [ ] **Step 2: Write the failing client tests**

`inferencex-e2e/infx/tests/serving/test_client.py`:

```python
import json
from pathlib import Path

from infx.serving.client import agentx_argv, label_result, online

ENV = {
    "PORT": "21000", "MODEL": "Qwen/Qwen3.8-27B", "SERVED_MODEL_NAME": "", "CONC": "8",
    "RESULT_DIR": "/workspace/results", "SERVING_SEARCH": "0",
    "AIPERF_GPU_TELEMETRY_URL": "http://localhost:21001/metrics",
    "AIPERF_SERVER_METRICS_URLS": "http://localhost:21000/metrics,http://localhost:21002/metrics",
}


def test_fixed_concurrency_argv():
    argv = agentx_argv("/venv/bin/agentx", Path("/workspace/results/scenario.yaml"), ENV)
    assert argv[:4] == ["/venv/bin/agentx", "profile", "--config", "/workspace/results/scenario.yaml"]
    assert argv[argv.index("--url") + 1] == "http://localhost:21000"
    assert argv[argv.index("--model") + 1] == "Qwen/Qwen3.8-27B"
    assert argv[argv.index("--concurrency") + 1] == "8"
    assert argv[argv.index("--gpu-telemetry") + 1] == "http://localhost:21001/metrics"
    metrics = argv.index("--server-metrics")
    assert argv[metrics + 1: metrics + 3] == ["http://localhost:21000/metrics", "http://localhost:21002/metrics"]
    assert argv[-2:] == ["--output-artifact-dir", "/workspace/results/aiperf_artifacts"]


def test_search_lets_aiperf_pick_the_concurrency_and_served_name_wins():
    env = {**ENV, "SERVING_SEARCH": "1", "SERVED_MODEL_NAME": "qwen", "AIPERF_GPU_TELEMETRY_URL": ""}
    argv = agentx_argv("agentx", Path("s.yaml"), env)
    assert "--concurrency" not in argv and "--no-gpu-telemetry" in argv
    assert argv[argv.index("--model") + 1] == "qwen"
    assert argv[argv.index("--tokenizer") + 1] == "Qwen/Qwen3.8-27B"


def test_label_result_files_the_aggregate_under_the_legacy_label(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"conc": 8}))
    label_result(path, "agentx-cc-062126-smoke")
    assert json.loads(path.read_text()) == {"conc": 8, "scenario": "agentx-cc-062126-smoke"}
    label_result(tmp_path / "missing.json", "x")  # a failed aggregation leaves nothing to label


def test_online_drops_the_engine_offline_switches():
    assert online({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "A": "b"}) == {"A": "b"}
```

- [ ] **Step 3: Run them to verify they fail**

Run: `PYTEST infx/tests/serving/test_client.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'infx.serving.client'`.

- [ ] **Step 4: Implement `client.py`**

`inferencex-e2e/infx/serving/client.py`:

```python
"""Benchmark client of one serving-YAML point, run in the engine container by bench.sh.

    python3 -m infx.serving.client prepare   # AgentX venv, file dataset, model weights
    python3 -m infx.serving.client run       # replay the scenario, write the result JSON

Both re-exec into the job's AgentX venv (upstream infx.bench.agentic.venv); the first call
builds it. Result JSONs come from the upstream infx.results CLIs, in the order
infx/bench/agentic/run.py runs them, or from infx.serving.sweep_result for a search.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from collections.abc import Mapping
from pathlib import Path

from infx.bench import proc
from infx.bench.agentic.venv import Runtime, bootstrap

OFFLINE = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
DATASET_PATH = ".dataset_path"
FAILED_REQUEST_THRESHOLD = "0.10"


def online(env: Mapping[str, str]) -> dict[str, str]:
    """``env`` without the offline switches a serving YAML may set for its engine."""
    return {key: value for key, value in env.items() if key not in OFFLINE}


def agentx_argv(agentx: str, scenario: Path, env: Mapping[str, str]) -> list[str]:
    argv = [agentx, "profile", "--config", str(scenario)]
    argv += ["--url", f"http://localhost:{env['PORT']}"]
    argv += ["--model", env.get("SERVED_MODEL_NAME") or env["MODEL"], "--tokenizer", env["MODEL"]]
    if env.get("SERVING_SEARCH") != "1":
        argv += ["--concurrency", env["CONC"]]
    if telemetry := env.get("AIPERF_GPU_TELEMETRY_URL"):
        argv += ["--gpu-telemetry", telemetry]
    else:
        argv.append("--no-gpu-telemetry")
    if urls := env.get("AIPERF_SERVER_METRICS_URLS"):
        argv += ["--server-metrics", *urls.split(",")]
    argv += ["--output-artifact-dir", f"{env['RESULT_DIR']}/aiperf_artifacts"]
    return argv


def label_result(path: Path, label: str) -> None:
    """File the aggregate under the legacy scenario label result-report reads first."""
    if not path.exists():
        return
    result = json.loads(path.read_text())
    result["scenario"] = label
    path.write_text(json.dumps(result, indent=2))


def prepare(env: Mapping[str, str], runtime: Runtime) -> int:
    result_dir = Path(env["RESULT_DIR"])
    result_dir.mkdir(parents=True, exist_ok=True)
    if env.get("SERVING_DATASET_KIND") == "file":
        from infx.serving.datasets import DatasetError, fetch

        try:
            (result_dir / DATASET_PATH).write_text(str(fetch(env["SERVING_DATASET"])))
        except DatasetError as error:
            print(f"ERROR {error}", file=sys.stderr)
            return 1
    return proc.call([str(runtime.hf), "download", env["MODEL"]], online(env))


def run(env: Mapping[str, str], runtime: Runtime) -> int:
    result_dir = Path(env["RESULT_DIR"])
    artifacts = result_dir / "aiperf_artifacts"
    output_dir = Path(env.get("AGENTIC_OUTPUT_DIR") or "/workspace")
    scenario = result_dir / "scenario.yaml"
    scenario.write_text(env["SERVING_SCENARIO_YAML"])
    child = online(env) | {
        "AIPERF_DATASET_CONFIGURATION_TIMEOUT": "1800",
        "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT": "1800",
        "RESULT_DIR": str(result_dir),
        "AGENTIC_OUTPUT_DIR": str(output_dir),
        "PYTHONPATH": proc.pythonpath(env),
    }
    if (marker := result_dir / DATASET_PATH).exists():
        child["IX_CUSTOM_DATASET_FILE"] = marker.read_text().strip()
    argv = agentx_argv(str(runtime.agentx), scenario, env)
    (result_dir / "benchmark_command.txt").write_text(shlex.join(argv) + "\n")
    replay_rc = proc.tee(argv, result_dir / "benchmark.log", child)
    python = str(runtime.python)
    if env.get("SERVING_SEARCH") == "1":
        return replay_rc or proc.call([python, "-m", "infx.serving.sweep_result"], child)
    aggregate_rc = proc.call([python, "-m", "infx.results.agentic.process_agentic_result"], child)
    proc.call([python, "-m", "infx.results.agentic.analyze_benchmark_distributions",
               str(artifacts), "-o", str(result_dir)], child)
    validate_rc = proc.call([python, "-m", "infx.results.agentic.validate_agentic_result", str(artifacts),
                             "--failed-request-threshold", FAILED_REQUEST_THRESHOLD], child)
    label_result(output_dir / f"{env['RESULT_FILENAME']}.json", env["SERVING_SCENARIO_LABEL"])
    return replay_rc or aggregate_rc or validate_rc


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv not in (["prepare"], ["run"]):
        print("usage: python3 -m infx.serving.client prepare|run", file=sys.stderr)
        return 2
    runtime = Runtime.for_job(os.environ)
    if not runtime.active():
        if not runtime.python.exists():
            print(f"Preparing the AgentX runtime in {runtime.venv}", flush=True)
            if rc := bootstrap(runtime):
                return rc
        environ = {**os.environ, "PYTHONPATH": proc.pythonpath(), "PYTHONSAFEPATH": "1"}
        runtime.exec_python(["-m", "infx.serving.client", *argv], environ)
    return prepare(os.environ, runtime) if argv == ["prepare"] else run(os.environ, runtime)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PYTEST infx/tests/serving/test_client.py infx/tests/serving/test_sweep_result.py`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add inferencex-e2e/infx/serving/client.py inferencex-e2e/infx/serving/sweep_result.py inferencex-e2e/infx/tests/serving/test_client.py inferencex-e2e/infx/tests/serving/test_sweep_result.py inferencex-e2e/infx/tests/serving/fixtures
git commit -m "feat(serving): in-container client on the upstream AgentX harness and result CLIs"
```

---

### Task 7: The allocation script, the engine step and the sidecar step

**Files:**
- Create: `inferencex-e2e/benchmarks/single_node/serving/job.sh`, `bench.sh`, `sidecar.sh` (all `chmod +x`)
- Test: `inferencex-e2e/infx/tests/serving/test_sidecar.py`

**Interfaces:**
- Consumes: env from the driver (Task 8: `SERVE_CMD, SERVE_ENV, SERVED_MODEL_NAME, ROUTER_*, REPLICAS, STORAGE_*, SERVING_ENGINE_IMAGE, SERVING_DCGM_IMAGE, SERVING_HF_CACHE_HOST, SERVING_*`) and from `benchmark-tmpl.yml` (`MODEL, CONC, FRAMEWORK, RESULT_DIR=/workspace/results, RESULT_FILENAME, HF_HUB_CACHE=/mnt/hf_hub_cache/`); `python3 -m infx.serving.client` (Task 6); upstream `python3 -m infx.bench wait --url --pid --log`.
- Produces: `job.sh` runs in the allocation with cwd = workspace. `bench.sh` runs in the engine container with cwd `/workspace`. `sidecar.sh router|storage` runs in the router/storage container. Files in `results/`: `serving.yaml, server[.N].log, server_command[.N].txt, router.log, router_command.txt, storage.log, storage_command.txt, dcgm.log`, plus the client's.

- [ ] **Step 1: Write the failing sidecar test**

`inferencex-e2e/infx/tests/serving/test_sidecar.py`:

```python
import os
import shlex
import subprocess
from pathlib import Path

SIDECAR = Path(__file__).resolve().parents[3] / "benchmarks/single_node/serving/sidecar.sh"


def run(mode: str, tmp_path: Path, **env: str) -> subprocess.CompletedProcess:
    full = {"PATH": os.environ["PATH"], "RESULT_DIR": str(tmp_path), **env}
    return subprocess.run(["bash", str(SIDECAR), mode], env=full, capture_output=True, text=True, check=False)


def test_generic_router_expands_engine_and_router_ports(tmp_path):
    command = shlex.join(["echo", "--worker-urls", "{{engine_urls}}", "--port={{router_port}}", "{{ keep }}"])
    out = run("router", tmp_path, ROUTER_CMD=command, ROUTER_PROFILE="generic", PORT="21000",
              ENGINE_PORT="21001", ENGINE_PORTS="21001 21002", ROUTER_ENV="TOKEN={{engine_port}}")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "--worker-urls http://127.0.0.1:21001 http://127.0.0.1:21002 --port=21000 {{ keep }}"
    assert "router env: TOKEN" in out.stderr and "21001" not in out.stderr  # names only, never values
    assert (tmp_path / "router_command.txt").read_text().startswith("echo ")


def test_dynamo_router_gets_the_http_flags(tmp_path):
    out = run("router", tmp_path, ROUTER_CMD="echo frontend", ROUTER_PROFILE="dynamo", PORT="21000")
    assert out.stdout.strip() == "frontend --http-host 0.0.0.0 --http-port 21000"


def test_storage_expands_its_own_ports(tmp_path):
    out = run("storage", tmp_path, STORAGE_CMD=shlex.join(["echo", "--port={{storage_master_port}}"]),
              STORAGE_MASTER_PORT="22000", STORAGE_META_PORT="22001")
    assert out.stdout.strip() == "--port=22000"


def test_shell_syntax():
    for name in ("job.sh", "bench.sh", "sidecar.sh"):
        assert subprocess.run(["bash", "-n", str(SIDECAR.with_name(name))], check=False).returncode == 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTEST infx/tests/serving/test_sidecar.py`
Expected: FAIL (`sidecar.sh` missing: non-zero return codes, `bash -n` fails).

- [ ] **Step 3: Write `sidecar.sh`**

```bash
#!/usr/bin/env bash
# Router or storage sidecar of one serving-YAML point, started by job.sh as an --overlap step:
# expand the {{...}} port placeholders, export the env lines, exec the command.
#   sidecar.sh router   ROUTER_CMD / ROUTER_ENV / ROUTER_PROFILE (dynamo | generic)
#   sidecar.sh storage  STORAGE_CMD / STORAGE_ENV
set -eo pipefail
mode="${1:-}"

expand() {
    local text="$1" urls="" port
    text="${text//\{\{router_port\}\}/${PORT:-}}"
    text="${text//\{\{engine_port\}\}/${ENGINE_PORT:-}}"
    text="${text//\{\{engine_url\}\}/http://127.0.0.1:${ENGINE_PORT:-}}"
    for port in ${ENGINE_PORTS:-}; do
        urls+="${urls:+ }http://127.0.0.1:$port"
    done
    text="${text//\{\{engine_urls\}\}/$urls}"
    [[ -n "${STORAGE_MASTER_PORT:-}" ]] && text="${text//\{\{storage_master_port\}\}/$STORAGE_MASTER_PORT}"
    [[ -n "${STORAGE_META_PORT:-}" ]] && text="${text//\{\{storage_meta_port\}\}/$STORAGE_META_PORT}"
    printf '%s' "$text"
}

case "$mode" in
    router) env_lines="${ROUTER_ENV:-}"; command="${ROUTER_CMD:?}" ;;
    storage) env_lines="${STORAGE_ENV:-}"; command="${STORAGE_CMD:?}" ;;
    *) echo "usage: sidecar.sh router|storage" >&2; exit 2 ;;
esac
dynamo=""
[[ "$mode" == router && "${ROUTER_PROFILE:-}" == dynamo ]] && dynamo=1

# Env values may hold tokens: export them without echoing the values.
while IFS= read -r assignment; do
    [[ -z "$assignment" ]] && continue
    [[ -z "$dynamo" ]] && assignment="$(expand "$assignment")"
    export "$assignment"
    echo "$mode env: ${assignment%%=*}" >&2
done <<< "$env_lines"

if [[ -n "$dynamo" ]]; then
    eval "set -- $command"
    set -- "$@" --http-host 0.0.0.0 --http-port "$PORT"
else
    eval "set -- $(expand "$command")"
fi
mkdir -p "$RESULT_DIR"
printf '%q ' "$@" > "$RESULT_DIR/${mode}_command.txt"
printf '\n' >> "$RESULT_DIR/${mode}_command.txt"
exec "$@"
```

- [ ] **Step 4: Write `job.sh`**

```bash
#!/usr/bin/env bash
# One serving-YAML benchmark point inside its Slurm allocation. infx/launch/drivers/serving.py
# sbatches this file with --chdir=<workspace>; every input is env (see the driver's job_env).
# Steps: DCGM, then the optional storage and router sidecars as --overlap steps, then the engine
# step (bench.sh: engine replicas + client) in the foreground. Ports are picked free on this node
# at run time: pyxis shares the host network with other users' jobs, and nodes are never exclusive.
set -euo pipefail

: "${SERVING_ENGINE_IMAGE:?}" "${SERVING_DCGM_IMAGE:?}" "${SERVING_HF_CACHE_HOST:?}"
WS="$PWD"
RESULTS="$WS/results"
mkdir -p "$RESULTS"
rm -rf "$RESULTS/.dynkv" "$RESULTS/.router_exited" "$RESULTS/.storage_exited"

PICKED=()
pick() {  # pick <var>: a free TCP port in 20000-32767 (below the ephemeral range), not picked before
    local _free  # not "port": a caller's `pick port` must set its own variable
    _free=$(python3 - "${PICKED[@]}" <<'PY'
import random, socket, sys
taken = set(sys.argv[1:])
for port in random.sample(range(20000, 32768), 2000):
    if str(port) in taken:
        continue
    with socket.socket() as probe:
        try:
            probe.bind(("", port))
        except OSError:
            continue
    print(port)
    break
PY
)
    [[ -n "$_free" ]] || { echo "ERROR: no free port in 20000-32767" >&2; exit 1; }
    PICKED+=("$_free")
    printf -v "$1" '%s' "$_free"
}

HF_TARGET="${HF_HUB_CACHE:-/mnt/hf_hub_cache}"
CONTAINER=(
    --container-mounts="$WS:/workspace,$SERVING_HF_CACHE_HOST:${HF_TARGET%/}"
    --no-container-mount-home --container-remap-root --container-writable
    --container-workdir=/workspace --no-container-entrypoint
)
SIDECARS=()
cleanup() {
    kill "${SIDECARS[@]}" 2>/dev/null || true
    rm -rf "$RESULTS/.dynkv"
}
trap cleanup EXIT

pick DCGM_PORT
srun --overlap --container-image="$SERVING_DCGM_IMAGE" "${CONTAINER[@]}" \
    dcgm-exporter -a ":$DCGM_PORT" > "$RESULTS/dcgm.log" 2>&1 &
SIDECARS+=("$!")
export AIPERF_GPU_TELEMETRY_URL="http://localhost:$DCGM_PORT/metrics"

if [[ -n "${STORAGE_CMD:-}" ]]; then
    pick STORAGE_MASTER_PORT
    pick STORAGE_META_PORT
    export STORAGE_MASTER_PORT STORAGE_META_PORT
    (
        rc=0
        NVIDIA_VISIBLE_DEVICES=none srun --overlap --container-image="$STORAGE_IMAGE" "${CONTAINER[@]}" \
            bash benchmarks/single_node/serving/sidecar.sh storage > "$RESULTS/storage.log" 2>&1 || rc=$?
        echo "exit $rc" > "$RESULTS/.storage_exited"
    ) &
    SIDECARS+=("$!")
    SECONDS=0
    until (exec 9<>"/dev/tcp/127.0.0.1/$STORAGE_MASTER_PORT") 2>/dev/null; do
        if [[ -f "$RESULTS/.storage_exited" ]]; then
            echo "ERROR: storage sidecar exited before readiness; see results/storage.log" >&2
            exit 1
        fi
        if (( SECONDS > 300 )); then
            echo "ERROR: storage sidecar did not open port $STORAGE_MASTER_PORT within 300 s" >&2
            exit 1
        fi
        sleep 2
    done
fi

pick PORT
export PORT
if [[ -n "${ROUTER_CMD:-}" ]]; then
    if [[ "${ROUTER_PROFILE:-}" == dynamo ]]; then
        export DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/workspace/results/.dynkv DYN_SYSTEM_ENABLED=true
        pick DYN_SYSTEM_PORT  # Dynamo parses it as i16: 20000-32767 fits
    else
        ENGINE_PORTS=""
        for ((replica = 0; replica < ${REPLICAS:-1}; replica++)); do
            pick port
            ENGINE_PORTS+="${ENGINE_PORTS:+ }$port"
        done
        export ENGINE_PORTS ENGINE_PORT="${ENGINE_PORTS%% *}"
    fi
    (
        rc=0
        srun --overlap --container-image="$ROUTER_IMAGE" "${CONTAINER[@]}" \
            bash benchmarks/single_node/serving/sidecar.sh router > "$RESULTS/router.log" 2>&1 || rc=$?
        echo "exit $rc" > "$RESULTS/.router_exited"
    ) &
    SIDECARS+=("$!")
    # exported after the router step forked, so the frontend does not inherit the engine's port
    if [[ "${ROUTER_PROFILE:-}" == dynamo ]]; then
        export DYN_SYSTEM_PORT
    fi
fi

srun --container-image="$SERVING_ENGINE_IMAGE" "${CONTAINER[@]}" bash benchmarks/single_node/serving/bench.sh
```

- [ ] **Step 5: Write `bench.sh`**

```bash
#!/usr/bin/env bash
# Engine step of one serving-YAML point (started by job.sh in the engine container, cwd /workspace):
# prepare the client, start the engine replica(s), wait until they (and the router) serve, run the client.
set -eo pipefail
: "${RESULT_DIR:?}" "${SERVE_CMD:?}" "${PORT:?}" "${MODEL:?}"
export PYTHONPATH=/workspace AGENTIC_OUTPUT_DIR=/workspace
mkdir -p "$RESULT_DIR"
printf '%s' "${SERVING_YAML:-}" > "$RESULT_DIR/serving.yaml"

expand_storage() {
    local text="$1"
    [[ -n "${STORAGE_MASTER_PORT:-}" ]] && text="${text//\{\{storage_master_port\}\}/$STORAGE_MASTER_PORT}"
    [[ -n "${STORAGE_META_PORT:-}" ]] && text="${text//\{\{storage_meta_port\}\}/$STORAGE_META_PORT}"
    printf '%s' "$text"
}
# SERVE_ENV values may hold tokens: export them without echoing the values.
while IFS= read -r assignment; do
    [[ -z "$assignment" ]] && continue
    assignment="$(expand_storage "$assignment")"
    export "$assignment"
    echo "serve env: ${assignment%%=*}"
done <<< "${SERVE_ENV:-}"

nvidia-smi || true
python3 -m infx.serving.client prepare

dynamo=""
[[ -n "${ROUTER_CMD:-}" && "${ROUTER_PROFILE:-}" == dynamo ]] && dynamo=1
if [[ -n "$dynamo" ]]; then
    ports=("$PORT")  # one engine without an HTTP port; the frontend serves $PORT
    export AIPERF_SERVER_METRICS_URLS="http://localhost:$PORT/metrics,http://localhost:$DYN_SYSTEM_PORT/metrics"
elif [[ -n "${ROUTER_CMD:-}" ]]; then
    read -r -a ports <<< "$ENGINE_PORTS"
    urls="http://localhost:${ROUTER_METRICS_PORT:-$PORT}/metrics"
    for port in "${ports[@]}"; do urls+=",http://localhost:$port/metrics"; done
    export AIPERF_SERVER_METRICS_URLS="$urls"
else
    ports=("$PORT")
    export AIPERF_SERVER_METRICS_URLS="http://localhost:$PORT/metrics"
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=, read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
replicas=${#ports[@]}
if (( ${#gpus[@]} == 0 || ${#gpus[@]} % replicas )); then
    echo "ERROR: ${#gpus[@]} GPUs do not split into $replicas replicas" >&2
    exit 1
fi
per=$(( ${#gpus[@]} / replicas ))

eval "set -- $SERVE_CMD"
pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT
for ((replica = 0; replica < replicas; replica++)); do
    suffix=""
    (( replica == 0 )) || suffix=".$replica"
    if [[ -n "$dynamo" ]]; then
        command=("$@")
    else
        command=("$@" --host 0.0.0.0 --port "${ports[replica]}")
    fi
    printf '%q ' "${command[@]}" > "$RESULT_DIR/server_command$suffix.txt"
    printf '\n' >> "$RESULT_DIR/server_command$suffix.txt"
    CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${gpus[*]:replica*per:per}")" \
        "${command[@]}" > "$RESULT_DIR/server$suffix.log" 2>&1 &
    pids+=("$!")
done

if [[ -z "$dynamo" ]]; then
    for ((replica = 0; replica < replicas; replica++)); do
        suffix=""
        (( replica == 0 )) || suffix=".$replica"
        python3 -m infx.bench wait --url "http://localhost:${ports[replica]}/health" \
            --pid "${pids[replica]}" --log "$RESULT_DIR/server$suffix.log"
    done
fi
if [[ -n "${ROUTER_CMD:-}" ]]; then
    # The router is ready once it lists the model (a Dynamo frontend only does after the engine registered).
    model="${SERVED_MODEL_NAME:-$MODEL}"
    until python3 - "$PORT" "$model" <<'PY' 2>/dev/null
import json, sys, urllib.request
port, model = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=5) as response:
    ids = [entry.get("id") for entry in json.load(response).get("data", [])]
sys.exit(0 if model in ids else 1)
PY
    do
        if ! kill -0 "${pids[0]}" 2>/dev/null; then
            echo "ERROR: engine exited before the router listed $model; last lines of server.log:" >&2
            tail -n 50 "$RESULT_DIR/server.log" >&2
            exit 1
        fi
        if [[ -f "$RESULT_DIR/.router_exited" ]]; then
            echo "ERROR: router step exited ($(cat "$RESULT_DIR/.router_exited")) before it listed $model; see results/router.log" >&2
            exit 1
        fi
        sleep 5
    done
fi

# the engine keeps its offline switches; the client must reach the Hub for datasets and the tokenizer
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
python3 -m infx.serving.client run
```

`bench.sh` has no unit test. It needs a real engine, so the live smoke in Task 12 covers it.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
chmod +x inferencex-e2e/benchmarks/single_node/serving/{job,bench,sidecar}.sh
```

Run: `PYTEST infx/tests/serving/test_sidecar.py`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add inferencex-e2e/benchmarks/single_node/serving/job.sh inferencex-e2e/benchmarks/single_node/serving/bench.sh inferencex-e2e/benchmarks/single_node/serving/sidecar.sh inferencex-e2e/infx/tests/serving/test_sidecar.py
git commit -m "feat(serving): allocation script, engine step and router/storage sidecar"
```

---

### Task 8: The `serving` launch path

**Files:**
- Create: `inferencex-e2e/infx/launch/drivers/serving.py`
- Modify: `inferencex-e2e/infx/launch/policy.py` (`LaunchPath` enum and `launch_path`)
- Modify: `inferencex-e2e/infx/launch/drivers/__init__.py` (import + one `ROUTES` entry)
- Modify: `.github/workflows/benchmark-tmpl.yml` (input `serving`, env `SERVING_JSON`, one condition)
- Test: `inferencex-e2e/infx/tests/serving/test_driver.py`

**Interfaces:**
- Consumes: the `SERVING_JSON` keys of Task 5; `job.sh` (Task 7); upstream `SlurmBackend.stage_image(image, *, single_node=True) -> Image`, `submit_batch(script, *, gpus, time_min, log) -> SlurmJob`, `stream_logs(job)`, `state(job) -> JobStatus`; `policy.salloc_time_limit(cluster_id, request)`; test helpers in `infx/tests/launch/fake_slurm.py`.
- Produces: `LaunchPath.SERVING = "serving"`; `drivers.serving.run(launch: Launch) -> int`; `Serving` (pydantic model of `SERVING_JSON`); `job_env(serving, images, hf_cache) -> dict[str, str]`.

- [ ] **Step 1: Write the failing tests**

`inferencex-e2e/infx/tests/serving/test_driver.py`:

```python
import json
import shutil
from types import SimpleNamespace

import pytest

from infx.launch.policy import LaunchPath, launch_path
from infx.launch.request import LaunchRequest
from infx.tests.launch.fake_slurm import (
    ROOT,
    base_env,
    install_fakes,
    launch,
    lines,
    runner_for,
    sandbox_runner_config,
)

SRUN_LOGGER = '#!/bin/bash\nprintf "%s\\n" "$*" >> "$FAKE_LOG_DIR/srun.log"\nexit 0\n'


def serving_json(**extra) -> str:
    return json.dumps({
        "framework": "sglang", "serve_cmd": "python3 -m sglang.launch_server --model-path org/model",
        "scenario_yaml": "schemaVersion: '2.0'\n", "scenario_label": "agentx-cc-062126-smoke",
        "dataset_kind": "public", "dataset": "semianalysis_cc_traces_weka_062126", "search": False,
        "yaml": "name: x\n", **extra,
    })


@pytest.fixture
def harness(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    workspace = tmp_path / "ws"
    serving = workspace / "benchmarks/single_node/serving"
    serving.mkdir(parents=True)
    shutil.copy(ROOT / "benchmarks/single_node/serving/job.sh", serving / "job.sh")
    fakes = install_fakes(tmp_path / "bin")
    (fakes / "srun").write_text(SRUN_LOGGER)
    logs = tmp_path / "logs"
    env = base_env(fakes=fakes, logs=logs, workspace=workspace, sandbox=sandbox)
    env.update(
        RUNNER_NAME=runner_for("hcm-hgx02"), IMAGE="lmsysorg/sglang:v0.5.22", MODEL="org/model",
        GPU_COUNT="2", SALLOC_TIME_LIMIT="30", ENROOT_IMPORT_TIME_LIMIT="10",
        HF_HUB_CACHE="/mnt/hf_hub_cache/", SERVING_JSON=serving_json(),
    )
    return SimpleNamespace(config=sandbox_runner_config(sandbox), workspace=workspace, logs=logs, env=env)


def test_policy_routes_a_serving_point():
    request = LaunchRequest.from_env({"RUNNER_NAME": "hcm-hgx02_00", "SERVING_JSON": "{}"})
    assert launch_path("hcm-hgx02", request) == LaunchPath.SERVING


def test_point_is_one_right_sized_shared_batch_job_on_its_node(harness):
    result = launch(harness.env, harness.config, harness.workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    [submit] = lines(harness.logs, "sbatch")
    flags = set(submit.split())
    assert {"--gres=gpu:nvidia_h200:2", "--nodelist=hgx-h200-02", "--cpus-per-gpu=16",
            "--mem-per-gpu=240000M", f"--chdir={harness.workspace}"} <= flags
    assert "--exclusive" not in flags
    steps = lines(harness.logs, "srun")
    assert len(steps) == 2  # DCGM sidecar + engine step
    assert "bench.sh" in steps[-1] and "--overlap" not in steps[-1]


def test_router_runs_as_an_overlap_sidecar(harness):
    env = {**harness.env, "SERVING_JSON": serving_json(
        router_image="lmsysorg/sglang-router:v0.2", router_cmd="echo router", router_profile="generic", replicas=2)}
    result = launch(env, harness.config, harness.workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    [router] = [step for step in lines(harness.logs, "srun") if "sidecar.sh router" in step]
    assert "--overlap" in router


def test_bad_serving_json_fails_before_any_slurm_call(harness):
    result = launch({**harness.env, "SERVING_JSON": "{}"}, harness.config, harness.workspace)
    assert result.returncode == 1
    assert "SERVING_JSON" in result.stdout + result.stderr
    assert lines(harness.logs, "sbatch") == []
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PYTEST infx/tests/serving/test_driver.py`
Expected: FAIL. `test_policy_routes_a_serving_point` errors with `AttributeError: SERVING`; the others fail because the launch takes the `srt-single` path.

- [ ] **Step 3: Add the policy hook**

In `inferencex-e2e/infx/launch/policy.py`, add to `class LaunchPath(StrEnum)`:

```python
    SERVING = "serving"
```

and in `launch_path`, directly after the `if request.is_multinode:` block (before `wrapped = ...`):

```python
    if request.env.get("SERVING_JSON"):  # VNG serving-YAML point (infx/launch/drivers/serving.py)
        return LaunchPath.SERVING
```

- [ ] **Step 4: Add the route**

In `inferencex-e2e/infx/launch/drivers/__init__.py` change `from infx.launch.drivers import script, srt` to `from infx.launch.drivers import script, serving, srt`, and add to `ROUTES`:

```python
    LaunchPath.SERVING: Route("slurm", serving.run),
```

- [ ] **Step 5: Implement the driver**

`inferencex-e2e/infx/launch/drivers/serving.py`:

```python
"""The serving driver: one VNG serving-YAML point (infx/serving) as one Slurm batch job.

benchmarks/single_node/serving/job.sh runs inside the allocation and starts the DCGM, storage
and router sidecars and the engine step. This driver stages their images, hands the job its
environment, submits it with the cluster's right-sized resources and follows it to the end.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from infx.launch import policy
from infx.launch.backends.base import BackendError
from infx.launch.backends.slurm import SlurmBackend
from infx.launch.context import Launch, LaunchError
from infx.launch.request import LaunchRequest, RequestError

JOB_SCRIPT = Path("benchmarks/single_node/serving/job.sh")
DCGM_IMAGE = "nvcr.io/nvidia/k8s/dcgm-exporter:4.2.3-4.1.3-ubuntu22.04"
HF_VOLUME = "hf-hub-cache"


class ServingRequest(LaunchRequest):
    """What a serving point reads from the workflow environment."""

    github_workspace: Path = Field(alias="GITHUB_WORKSPACE")
    image: str = Field(alias="IMAGE")
    gpu_count: int = Field(alias="GPU_COUNT")
    salloc_time_limit: int = Field(alias="SALLOC_TIME_LIMIT")
    serving_json: str = Field(alias="SERVING_JSON")


class Serving(BaseModel):
    """SERVING_JSON, as infx/serving/to_matrix.py writes it."""

    model_config = ConfigDict(extra="forbid")

    framework: str
    serve_cmd: str
    serve_env: str = ""
    served_model_name: str = ""
    router_image: str = ""
    router_cmd: str = ""
    router_env: str = ""
    router_profile: str = ""
    router_metrics_port: str = ""
    replicas: int = 1
    storage_image: str = ""
    storage_cmd: str = ""
    storage_env: str = ""
    scenario_yaml: str
    scenario_label: str
    dataset_kind: str
    dataset: str
    search: bool
    yaml: str


def job_env(serving: Serving, images: dict[str, str], hf_cache: Path) -> dict[str, str]:
    """The environment job.sh, bench.sh, sidecar.sh and the client read."""
    return {
        "SERVE_CMD": serving.serve_cmd,
        "SERVE_ENV": serving.serve_env,
        "SERVED_MODEL_NAME": serving.served_model_name,
        "ROUTER_IMAGE": images.get("router", ""),
        "ROUTER_CMD": serving.router_cmd,
        "ROUTER_ENV": serving.router_env,
        "ROUTER_PROFILE": serving.router_profile,
        "ROUTER_METRICS_PORT": serving.router_metrics_port,
        "REPLICAS": str(serving.replicas),
        "STORAGE_IMAGE": images.get("storage", ""),
        "STORAGE_CMD": serving.storage_cmd,
        "STORAGE_ENV": serving.storage_env,
        "SERVING_ENGINE_IMAGE": images["engine"],
        "SERVING_DCGM_IMAGE": images["dcgm"],
        "SERVING_HF_CACHE_HOST": str(hf_cache),
        "SERVING_SCENARIO_YAML": serving.scenario_yaml,
        "SERVING_SCENARIO_LABEL": serving.scenario_label,
        "SERVING_DATASET_KIND": serving.dataset_kind,
        "SERVING_DATASET": serving.dataset,
        "SERVING_SEARCH": "1" if serving.search else "0",
        "SERVING_YAML": serving.yaml,
    }


def run(launch: Launch) -> int:
    """Stage the images, sbatch job.sh, follow it, and return its exit code."""
    event = launch.life.event
    request = ServingRequest.from_env(launch.request.env)
    try:
        serving = Serving.model_validate_json(request.serving_json)
    except ValidationError as error:
        fields = ", ".join(".".join(map(str, item["loc"])) or "serving" for item in error.errors())
        raise RequestError(f"SERVING_JSON is not a serving point (bad or missing: {fields})") from None
    backend = launch.backend
    if not isinstance(backend, SlurmBackend):
        raise LaunchError(f"the serving path needs a slurm cluster, not {launch.cluster.scheduler!r}")
    volume = backend.settings.volumes.get(HF_VOLUME)
    if volume is None:
        raise LaunchError(f"cluster {launch.cluster.id!r} has no {HF_VOLUME!r} volume")
    minutes = policy.salloc_time_limit(launch.cluster.id, request)
    wanted = {"engine": request.image, "dcgm": DCGM_IMAGE}
    if serving.router_image:
        wanted["router"] = serving.router_image
    if serving.storage_image:
        wanted["storage"] = serving.storage_image
    with event.stage("prepare"):
        images = {role: backend.stage_image(image, single_node=True).reference for role, image in wanted.items()}
    os.environ.update(job_env(serving, images, volume.path))  # sbatch --export=ALL hands them to job.sh
    log = Path(request.env.get("RUNNER_TEMP") or request.workspace) / f"serving-{os.getpid()}.log"
    with event.stage("submit"):
        try:
            job = backend.submit_batch(request.workspace / JOB_SCRIPT, gpus=request.gpu_count,
                                       time_min=minutes, log=log)
        except BackendError as error:
            event.fail(type(error).__name__, f"batch allocation unavailable: {error}", retriable=error.retriable)
            return 1
    print(f"Serving job {job.id}; log: {log}", flush=True)
    try:
        backend.stream_logs(job)
    except BackendError:
        return 1
    status = backend.state(job)
    return 0 if status.succeeded else status.exit_code or 1
```

- [ ] **Step 6: Run the tests**

The upstream srt tests import the `srt-slurm` submodule; check it out once: `git submodule update --init inferencex-e2e/utils/srt-slurm`.

Run: `PYTEST infx/tests/serving/test_driver.py infx/tests/launch` (add `--with pytest-xdist ... -n 8` to the uv line for speed)
Expected: all pass, the upstream launch tests included. If `stage_image` fails under the fakes, read `ensure_image` in `infx/launch/backends/slurm/squash.py` for the `submit-host` branch and add the missing fake behaviour **in the test** (for example a pre-created squash file in the sandboxed squash dir). Do not change upstream code.

- [ ] **Step 7: Wire `benchmark-tmpl.yml`**

In `.github/workflows/benchmark-tmpl.yml`:

1. Under `on.workflow_call.inputs`, add:

```yaml
      serving:
        description: "VNG serving point (SERVING_JSON from infx/serving/to_matrix.py); empty for every other caller"
        required: false
        type: string
        default: ""
```

2. Under the top-level `env:`, add:

```yaml
  SERVING_JSON: ${{ inputs.serving }}
```

3. In the step `Validate agentic result`, change the condition to:

```yaml
        if: ${{ !inputs.eval-only && inputs.scenario-type == 'agentic-coding' && !(inputs.serving != '' && fromJSON(inputs.serving).search) }}
```

Then check both workflows parse: `uv run --no-project --with pyyaml python -c "import yaml,sys; [yaml.safe_load(open(p)) for p in sys.argv[1:]]" .github/workflows/benchmark-tmpl.yml .github/workflows/serving.yml`. Expected: no error. If the repo has a workflow lint step (`zizmor`, `actionlint` in `.github/workflows/zizmor.yml`), run it on both files the way that workflow does.

- [ ] **Step 8: Commit**

```bash
git add inferencex-e2e/infx/launch/drivers/serving.py inferencex-e2e/infx/launch/policy.py inferencex-e2e/infx/launch/drivers/__init__.py .github/workflows/benchmark-tmpl.yml inferencex-e2e/infx/tests/serving/test_driver.py
git commit -m "feat(launch): serving path, one right-sized batch job per VNG serving point"
```

---

### Task 9: Curated serving YAMLs

**Files:**
- Create: `inferencex-e2e/configs/serving/{qwen38-27b-tau2-smoke,glm53-w4afp8-sglang-router,glm53-w4afp8-dynamo,glm53-w4afp8-router-mooncake-e2,gemma4-livekit-2rep-cacheaware,gemma4-sweep-grid,gemma4-hicache-fixed-isl8k}.yaml`
- Test: `inferencex-e2e/infx/tests/serving/test_configs.py`

**Interfaces:**
- Consumes: `to_matrix.build` (Task 5).

- [ ] **Step 1: Write the failing test**

`inferencex-e2e/infx/tests/serving/test_configs.py`:

```python
from pathlib import Path

import pytest

from infx.serving.to_matrix import build

CONFIGS = sorted((Path(__file__).resolve().parents[3] / "configs/serving").glob("*.yaml"))


def test_seven_examples_one_per_pattern():
    assert len(CONFIGS) == 7


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_every_serving_config_validates(path):
    assert build(path.read_text())["points"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTEST infx/tests/serving/test_configs.py`
Expected: FAIL (`assert 0 == 7`).

- [ ] **Step 3: Convert the seven legacy YAMLs**

For each pair below, run `git show vng-benchmark:serving/<legacy>.yaml > inferencex-e2e/configs/serving/<new>.yaml`, then apply the conversion rules:

| Legacy (`vng-benchmark:serving/…`) | New (`configs/serving/…`) | Pattern |
|---|---|---|
| `qwen38-27b-baseline-tau2smoke.yaml` | `qwen38-27b-tau2-smoke.yaml` | engine only, file dataset |
| `glm53-w4afp8-sglang-router.yaml` | `glm53-w4afp8-sglang-router.yaml` | sglang-router + engine |
| `glm53-w4afp8-dynamo.yaml` | `glm53-w4afp8-dynamo.yaml` | Dynamo frontend + engine |
| `glm53-w4afp8-router-mooncake-e2.yaml` | `glm53-w4afp8-router-mooncake-e2.yaml` | storage sidecar |
| `gemma4-livekit-2rep-cacheaware-v1.yaml` | `gemma4-livekit-2rep-cacheaware.yaml` | replicas + router, LiveKit |
| `gemma4-sweep-smoke-1xh200.yaml` | `gemma4-sweep-grid.yaml` | synthetic search grid |
| `gemma4-hicache-1gpu-isl8k-osl1k-cache50.yaml` | `gemma4-hicache-fixed-isl8k.yaml` | synthetic fixed conc |

Conversion rules (also the migration table of Task 10's skill):

- `runner:`
  - `hardware-hcm-*_01` → `cluster:hcm-hgx01`;
  - `hardware-hcm-*_02` → `cluster:hcm-hgx02`;
  - `hcm-rtx6000pro_01` / `_02` → `cluster:hcm-rtx01` / `cluster:hcm-rtx02`;
  - `h200-greennode-slurm-1x_*` → `cluster:hcm-hgx02` (the HAN pool is out of service).
- `scenario:`
  - `agentx-cc-062126` → `scenario: agentx`;
  - `agentx-cc-062126-smoke` → `scenario: agentx` + `overrides: {duration: 180}`;
  - `agentx-cc-062126-256k` → `scenario: agentx` + `overrides: {dataset: semianalysis_cc_traces_weka_062126_256k}`;
  - `agentx-cc-062126-1m` → `scenario: agentx` + `overrides: {max_context_length: null}`;
  - `agentx-cc-062126-topp95` → `scenario: agentx` + `overrides: {generation: {top_p: 0.95, temperature: 1.0}}`;
  - `custom-agentic-tau2[-smoke]` → `scenario: tau2` (smoke: + `overrides: {duration: 180}`);
  - `custom-agentic-livekit111868-v1|v2` → `scenario: livekit` + `overrides: {dataset: livekit-111868-v1|v2}`; to move to the warm trace, drop the dataset override;
  - `custom-agentic-livekit111868-smoke` → `scenario: livekit` + `overrides: {dataset: livekit-111868-v2, duration: 180}`;
  - `agentx-shared-prefix-e2` → `scenario: shared-prefix-e2`;
  - `agentx-hidden-prefix-e3[-smoke]` → `scenario: hidden-prefix-e3` (smoke: + `overrides: {duration: 180}`);
  - `sweep-isl<I>-osl<O>-cache<C>` (one or a list) → `scenario: synthetic` + `workload: {isl: [...], osl: [...], cache: [...], search: true}` + `conc: 1`;
  - `fixed-isl<I>-osl<O>-cache<C>` → the same `workload` with `search: false` and the original `conc` list.
- Drop any `IX_SLO` from `env:` and write it as `slo: "<same tuples>"`.
- Rename `name:` to the new file stem.
- Rewrite the header comment: one or two lines saying which pattern this file shows.

For the LiveKit example, also drop the `dataset` override so it uses `livekit-111868-v1.1-warm`, and say so in the header.

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTEST infx/tests/serving/test_configs.py`
Expected: 8 passed. A failing file prints `ServingError` lines: fix the YAML, never the schema.

- [ ] **Step 5: Commit**

```bash
git add inferencex-e2e/configs/serving inferencex-e2e/infx/tests/serving/test_configs.py
git commit -m "feat(serving): seven example serving YAMLs, one per pattern"
```

---

### Task 10: Skills

**Files:**
- Create: `.agents/skills/serving-bench/SKILL.md`, `.claude/skills/serving-bench` (symlink)
- Create: `.agents/skills/serving-runner-setup/SKILL.md`, `.claude/skills/serving-runner-setup` (symlink)
- Create: `.agents/skills/result-report/**` (port), `.claude/skills/result-report` (symlink)
- Test: the result-report suite plus one new fixture test

- [ ] **Step 1: Port `result-report` and run its tests**

```bash
git archive vng-benchmark .agents/skills/result-report | tar -x
ln -s ../../.agents/skills/result-report .claude/skills/result-report
cd .agents/skills/result-report/scripts && uv run --no-project --python 3.12 --with pytest python -m pytest -q; cd -
```

Expected: the suite passes unchanged.

Then add this test to `.agents/skills/result-report/scripts/test_extract.py` (adapt the helper names to the ones that file already uses for building a point from an aggregate; the assertion is what matters):

```python
def test_new_style_run_takes_the_scenario_from_the_aggregate():
    # serving-path runs (refactor/vng-benchmark) run aiperf with --config results/scenario.yaml
    # and serving.yaml says only the family; the label lives in the aggregate.
    agg = {"scenario": "agentx-cc-062126-smoke", "conc": 8}
    config = {"serving.yaml": "name: x\nscenario: agentx\n",
              "benchmark_command.txt": "agentx profile --config /workspace/results/scenario.yaml"}
    assert scenario_of(agg, config) == "agentx-cc-062126-smoke"
```

If `extract.py` has no function that takes `(agg, config)`, extract the existing scenario expression at `extract.py:346` into `def scenario_of(agg: dict, config: dict) -> str | None` (same expression, no behaviour change), use it at line 346, and import it in the test. Run the suite again: expected all pass.

- [ ] **Step 2: Write `serving-bench`**

`.agents/skills/serving-bench/SKILL.md`:

````markdown
---
name: serving-bench
description: Benchmark an LLM serving config on a VNG GPU node with the AgentX coding scenario, the tau2 or LiveKit agent traces, the E2/E3 prefix traces, or the synthetic ISL/OSL/cache grid (fixed concurrency or SLO search), starting from a vllm/sglang launch command, a Dynamo frontend + engine pair, a docker-compose file or a k8s manifest. Use when the user says "benchmark agentx với config…", "chạy agentx/tau2/livekit cho config này", "bench sweep", "tìm max CCU", or pastes a serve command and asks for a benchmark. Writes one serving YAML and dispatches it with one command.
---

# serving-bench

You turn the user's serving config into **one YAML file** and run **one command**. Do not write recipes, edit `configs/runners.yaml`, or call `gh workflow run` yourself. How it runs end to end: `inferencex-e2e/docs/serving/ARCHITECTURE.md`.

## 1. Write `inferencex-e2e/configs/serving/<name>.yaml`

```yaml
name: qwen38-bf16-tp1               # [a-z0-9.-], no "_"; prefixes the artifact names
runner: cluster:hcm-hgx02           # see "Runner" below
image: vllm/vllm-openai:v0.29.0     # required
model: Qwen/Qwen3.8-27B             # HF repo id, never a local path
gpus: 1                             # GPUs the command needs
replicas: 1                         # optional: N engine copies, gpus/N each; N > 1 needs router: with {{engine_urls}}
precision: bf16                     # optional (bf16 default), naming only
env:                                # optional, the engine's env vars; quote booleans
  PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
serve: |                            # the launch command, as given
  vllm serve Qwen/Qwen3.8-27B --tensor-parallel-size 1 --max-model-len 262144
router:                             # optional: any router in front of the engine(s)
  image: <router image>
  env: {}
  metrics_port: 18888               # optional: the router's /metrics port; default = the router port
  serve: |                          # {{router_port}} = the port the benchmark calls, {{engine_url(s)}} = the engine(s)
    python3 -m sglang_router.launch_router --host=0.0.0.0 --port={{router_port}} --worker-urls={{engine_url}}
storage:                            # optional: a non-GPU sidecar started first (e.g. Mooncake master)
  image: <image>
  serve: mooncake_master --port={{storage_master_port}} --http_metadata_server_port={{storage_meta_port}}
scenario: agentx                    # agentx | tau2 | livekit | shared-prefix-e2 | hidden-prefix-e3 | synthetic
overrides: {duration: 180}          # optional variant, see the table below
conc: [4, 8]                        # one benchmark job per value
```

Working examples, one per pattern: `inferencex-e2e/configs/serving/*.yaml`. They are format references: never dispatch them or copy their values for a user's request.

### Scenarios

| `scenario:` | Workload | Default length |
|---|---|---|
| `agentx` | SemiAnalysis coding traces (weka 062126), needs 262144 context | 3600 s |
| `tau2` | tau2 customer-service agent traces | 1200 s |
| `livekit` | LiveKit voice-agent traces, V1.1 warm (dial warm-ups count in TTFT) | 1200 s |
| `shared-prefix-e2` / `hidden-prefix-e3` | prefix-cache A/B traces (HiCache L2 vs Mooncake L3) | 1800 s |
| `synthetic` | synthetic ISL/OSL with a shared-prefix cache hit, needs `workload:` | 300 s per point |

Variants (`overrides:`, only these keys):

| You want | Write |
|---|---|
| a smoke run | `overrides: {duration: 180}` |
| the 256k AgentX corpus | `overrides: {dataset: semianalysis_cc_traces_weka_062126_256k}` |
| AgentX without the 262144 cap | `overrides: {max_context_length: null}` |
| sampling instead of greedy | `overrides: {generation: {top_p: 0.95, temperature: 1.0}}` |
| LiveKit V1 / V2 trace | `overrides: {dataset: livekit-111868-v1}` (or `-v2`) |
| a different trace idle-gap cap | `overrides: {trace_idle_gap_cap: 1}` |

Synthetic grid, one job per point, all in one run:

```yaml
scenario: synthetic
workload: {isl: [8k, 32k, 64k, 128k], osl: [512, 1k], cache: [0, 50, 90], search: true}
slo: "time_to_first_token:avg:lt:8000;inter_token_latency:p90:lt:50"   # optional; this is the default
conc: 1                             # search: true finds the concurrency itself
```

`search: false` runs each point at every `conc` value instead (then only one point, or one `conc`).

Old scenario names (`agentx-cc-062126-smoke`, `custom-agentic-tau2`, `sweep-isl8k-osl1k-cache50`, …) map as in the table of `docs/serving/ARCHITECTURE.md` § Scenarios. Results keep the old names.

### Translation rules

- **Model path → HF id.** `--model-path /models/<org>/<model>` becomes `--model-path <org>/<model>` and `model: <org>/<model>`. If you cannot tell the HF id from the path, ask.
- **Host/port:** leave them or drop them; the benchmark picks its own free ports.
- **Router:** the router part goes under `router:` with its own `image`; the engine stays in `serve`. Engine env → `env:`, router env → `router.env:`.
  - **Replicas** (e.g. 2× TP1): `replicas: N`, `gpus` = N × per-engine GPUs, the router lists `{{engine_urls}}`. Not with the Dynamo frontend.
  - **Dynamo** (`python3 -m dynamo.frontend` + `python3 -m dynamo.sglang`): leave out `DYN_*`, `ETCD_ENDPOINTS`, NATS and `--http-host/--http-port`; the benchmark wires them.
  - **Any other router:** keep the user's flags except the wiring. Listen flags → `--host=0.0.0.0 --port={{router_port}}`. A worker URL → `{{engine_url}}` (or `{{engine_port}}`). A literal metrics port flag → also `router.metrics_port: <port>`.
  - **Router command missing** (only flags + image): infer it from the flag family and say so. `--worker-urls`, `--policy=cache_aware`, `--prometheus-port` → `python3 -m sglang_router.launch_router`. Unknown family → ask. Slim images without python3 run a binary path.
  - Placeholders exist only in `router.*`, `storage.*` and (storage ports only) `env:`. No `$VAR` anywhere.
- **Env vars** (`NCCL_P2P_LEVEL=NVL`, compose `environment:`, k8s `env:`) go under `env:`. No secrets: the YAML is a run input and an artifact. Dataset access uses the runner's `GREENLAB_HF_TOKEN`.
- **Images:** public ones and `vcr.vngcloud.vn` ones the runner can read need nothing. Another private registry fails at `enroot import`: tell the user it needs infra (credentials for `stackops` or a pre-imported squash).
- **gpus:** tensor-parallel × pipeline-parallel (× replicas). If unclear, ask.
- **Runner** (each VNG node is one cluster; the job takes only its GPUs, 16 CPUs and a memory share per GPU, the rest of the node stays usable by the team):

  | Node | `runner:` | GPUs |
  |---|---|---|
  | hgx-h200-01 | `cluster:hcm-hgx01` | 4 (until infra declares all 8) |
  | hgx-h200-02 | `cluster:hcm-hgx02` | 8 |
  | either hgx | `h200-hcm` | up to 4 |
  | RTX PRO 6000 node 1 / 2 | `cluster:hcm-rtx01` / `cluster:hcm-rtx02` | 2 |
  | either RTX node | `rtx6000pro-hcm` | up to 2 |

  Check load first: `ssh slurm-client squeue -p hardware-hcm` (or `-p rtx6000pro-hcm`). If the node is busy, tell the user the job waits in the Slurm queue. Never cancel another user's job.
- **docker-compose / k8s:** `image` from `image:`; `serve` from `command:` + `args:`; `env` from `environment:` / `env:`.
- **Drop `--numa-node …`.**
- **DeepEP on hgx** (`--moe-a2a-backend deepep`): add `NVSHMEM_REMOTE_TRANSPORT: none` to `env:`.
- **JSON in a flag must be single-quoted:** `--kv-events-config='{"publisher":"zmq"}'`.
- **Context length:** `agentx` needs 262144 (`--max-model-len 262144` / `--context-length 262144`); ask if the config cannot. Trace scenarios are fine with the model default unless it is below 32k.

## 2. Ask the user (do not guess) when

Asking means: stop, write nothing, dispatch nothing, and end your reply with the question.

- No image and no version → ask for the exact tag. Never take it from `configs/serving/*.yaml` or past runs.
- A **version** but no image → resolve it: SGLang → `lmsysorg/sglang:v<ver>`, vLLM → `vllm/vllm-openai:v<ver>`. Verify the tag exists: `curl -s "https://hub.docker.com/v2/repositories/<repo>/tags?name=<tag>&page_size=50"`. Not found → ask. Router images are never resolved this way.
- No scenario → list the six with their lengths and ask.
- `gpus` fits no node in the runner table → ask.
- A Dynamo router with a non-Dynamo engine, or the reverse → the validator rejects it; explain and ask.

## 3. Dispatch

```bash
inferencex-e2e/utils/serving/dispatch.sh inferencex-e2e/configs/serving/<name>.yaml
```

It validates first. On `ERROR <field>: <fix>` lines, fix the YAML and rerun; nothing was dispatched. On success it prints the run URL. The branch must be pushed (the script says so); do not push without the user's OK.

## 4. Results

- Status: `gh run view <run-id> --repo vngcloud/InferenceX`. One job per point (`conc` value or grid point). Expect about 20 min of image import, model load and warmup on top of the scenario length.
- Artifacts: `agentic_<exp-name>_…` (`results/`: `serving.yaml`, `scenario.yaml`, `server_command.txt`, `server.log`, `router.log`, `storage.log`, `benchmark_command.txt`, `benchmark.log`, `aiperf_artifacts/`) and `bmk_agentic_<exp-name>_…` (the aggregate JSON). For a report, use the `result-report` skill.
- A search point's aggregate carries `state` (`ok`, `cliff`, `no_pass_in_range`, `no_failure_in_range`, `max_iterations`, `no_data`), `max_conc` and the metrics at `max_conc`. Every state is a result, not a failure.
- A failed job: `server.log` first, then `benchmark.log`; `router.log` / `storage.log` when it fails before aiperf starts. `gh run view <id> --repo vngcloud/InferenceX --log-failed | tail -100` shows the Slurm/enroot side.
- If the engine crashes, report the error and the config as given. Do not change flags or redispatch a modified config on your own.
- Report: run URL, per-point success/failure, where the artifacts are.
````

Then: `ln -s ../../.agents/skills/serving-bench .claude/skills/serving-bench`.

- [ ] **Step 3: Write `serving-runner-setup`**

`.agents/skills/serving-runner-setup/SKILL.md`:

````markdown
---
name: serving-runner-setup
description: Bring a VNG Slurm GPU node into the InferenceX runner fleet the standard way (on-node listeners as stackops under systemd, one cluster record per node in configs/runners.yaml), or check, move or retire existing VNG runners. Use when the user says "setup runner", "thêm node vào InferenceX", "đăng ký runner", "runner offline", "chuyển runner sang stackops", or a serving job lands on the wrong node or cannot start a container.
---

# serving-runner-setup

A thin layer over upstream `inferencex-e2e/utils/runner_setup/RUNNER_SETUP.md`: read that first. VNG differs in one way: there is no shared filesystem big enough, so each runner lives **on its GPU node** and each node is **one cluster** pinned by `salloc-args --nodelist`.

Every step that registers, removes or restarts a runner, or changes a node, needs the user's explicit OK first. Nodes are shared with the team: never `--exclusive`, never cancel someone else's job.

## Convention (all pools)

1. User `stackops`. Its home holds `~/.config/enroot/.credentials` (VCR pull) and the runner `.env` (`GREENLAB_HF_TOKEN=…`).
2. Name `<cluster-id>_<NN>` (`hcm-hgx02_00`), exactly one `cluster:<cluster-id>` label.
3. Directory `<large-local-disk>/gharunners/<name>/` (`/mnt/gharunners` on hgx, `/data/gharunners` on rtx).
4. systemd: `sudo ./svc.sh install stackops && sudo ./svc.sh start` in each runner directory. No tmux, no hand-run `run.sh`.
5. Slots per node = maximum concurrent jobs (hgx 4, rtx 2).

## 1. Preflight the node

Run from `slurm-client` as yourself; each probe is a 1-CPU job of a few seconds:

```bash
NODE=hgx-h200-02; PART=hardware-hcm
srun -p $PART -w $NODE -n1 -c1 --mem=512M -t 2 bash -c '
  hostname; getent passwd stackops || echo "MISSING passwd entry for stackops"
  grep -v "^#" /etc/enroot/enroot.conf | grep -v "^$"
  df -h / /mnt /data 2>/dev/null; nvidia-smi -L | wc -l'
srun -p $PART -w $NODE -n1 -c1 --mem=512M -t 5 --container-image=ubuntu:24.04 true && echo "pyxis OK"
scontrol show node $NODE | grep -E "NodeHostName|Gres|RealMemory|CPUTot"
```

Pass criteria:
- `stackops` exists.
- Pyxis starts a container.
- The enroot paths are on a disk with > 100 GB free.
- The GPU count in `Gres` equals `nvidia-smi -L`.
- `NodeHostName` matches the node name.

Fix anything else on the infra side before registering. Known state (2026-10-25):
- HAN `test` partition: `/` at 99%, enroot paths on `/`, slurmd name mismatch on `hoanq3-h200-1x-han-3`. The pool stays out until fixed.
- `hgx-h200-01`: gres declares 4 of 8 GPUs.
- `/shared` NFS at 96%.

## 2. Register the listeners

As `stackops` on the node, with a registration token from https://github.com/vngcloud/InferenceX/settings/actions/runners/new:

```bash
mkdir -p /mnt/gharunners && cd /mnt/gharunners
curl -fsSLO https://raw.githubusercontent.com/vngcloud/InferenceX/refactor/vng-benchmark/inferencex-e2e/utils/runner_setup/setup.sh
bash setup.sh <TOKEN> <RUNNER_TARBALL_URL> 0 3 /mnt/gharunners hcm-hgx02 slurm,h200,cluster:hcm-hgx02 https://github.com/vngcloud/InferenceX
for d in /mnt/gharunners/gharunner0*/actions-runner; do (cd "$d" && sudo ./svc.sh install stackops && sudo ./svc.sh start); done
```

Put the runner `.env` (`GREENLAB_HF_TOKEN=…`) in each `actions-runner/` directory before starting.

## 3. Add the cluster record

In `inferencex-e2e/configs/runners.yaml`, add the `cluster:<id>` label with the slot names, add the slots to the pool label (`h200-hcm` / `rtx6000pro-hcm`), and copy the nearest `hcm-*` record under `clusters:`. Change:
- the `--nodelist` in `salloc-args`;
- `gpus-per-node`, from `Gres`;
- `available-cpu-dram-mib`, from `RealMemory`;
- `--mem-per-gpu`: about 95% of `RealMemory` / GPUs, in M;
- the volume and squash paths.

Keep `exclusive: false` and `--cpus-per-gpu=16`. Run `PYTEST infx/tests/serving/test_clusters.py` (extend its `NODES` table) and commit.

## 4. Smoke

Dispatch an `agentx` + `overrides: {duration: 180}` serving YAML on `cluster:<id>` with the `serving-bench` skill. Watch it get scheduled with `squeue -w <node>`.

## Move or retire a runner

- **Move to `stackops`** (for example the RTX runners that ran as `thanglq5` in `/data/thanglq5/runner-rtx0N`):
  1. Stop the process.
  2. `./config.sh remove --token <removal-token>`.
  3. Register again under the convention.
  4. Move the HF cache and squash files: `mv /data/thanglq5/hf_hub_cache /data/hf_hub_cache`, `mv /data/thanglq5/containers /data/containers`, then `chown -R stackops`.
- **Retire**:
  1. `sudo ./svc.sh stop && sudo ./svc.sh uninstall`.
  2. `./config.sh remove --token <removal-token>`.
  3. Delete the slot from `configs/runners.yaml`.
````

Then: `ln -s ../../.agents/skills/serving-runner-setup .claude/skills/serving-runner-setup`.

The `setup.sh` URL in step 2 points at this branch. Task 12 updates it if the branch is renamed.

- [ ] **Step 4: Commit**

```bash
git add .agents/skills/serving-bench .agents/skills/serving-runner-setup .agents/skills/result-report .claude/skills/serving-bench .claude/skills/serving-runner-setup .claude/skills/result-report
git commit -m "docs(skills): serving-bench and serving-runner-setup for the serving path; port result-report"
```

---

### Task 11: ARCHITECTURE.md

**Files:**
- Create: `inferencex-e2e/docs/serving/ARCHITECTURE.md`

- [ ] **Step 1: Write the document**

Use the outline below. Every section is required. Take facts only from the spec, this plan and the code as it now exists; give file:line references for the flow steps (look the lines up in the code you wrote in Tasks 1–8).

1. **What this is.** One paragraph: upstream InferenceX plus one launch path, `serving`, for VNG serving-YAML benchmarks on VNG Slurm nodes. Link the spec.
2. **One benchmark point, end to end.** The flow block from the spec section "Flow of one benchmark point", updated to the real names (`job.sh`, `bench.sh`, `sidecar.sh`, `client.py prepare/run`). Add one line per step with its file:line.
3. **Where VNG code lives.** The rule (`serving` in every path; `git grep -l serving`). The file table from this plan's "File map". The five upstream touch points with the exact lines added.
4. **VNG Slurm.**
   - One cluster per node and why (no shared FS; measured facts).
   - The runner convention.
   - Right-sizing through `salloc-args` (and why: `submit_batch` adds no CPU/memory flags).
   - Free ports at run time (20000–32767) and why (pyxis shares the host network; never exclusive).
   - Squash `submit-host`.
5. **Scenarios.**
   - The six families.
   - The overrides whitelist.
   - The synthetic grid.
   - The old → new mapping table from Task 9 Step 3.
   - The parity test (`infx/tests/serving/test_scenarios.py`).
   - The LiveKit warm-up TTFT note.
6. **Results.**
   - Artifact names.
   - `results/` contents.
   - The aggregate `scenario` label.
   - Search `state` values.
   - `result-report`.
7. **Syncing upstream.**
   - `git fetch upstream && git merge upstream/main`.
   - Conflicts can only be in the five touch points.
   - After a merge, rerun `PYTEST infx/tests/serving infx/tests/launch`.
   - Check that `infx.bench.agentic.venv`, `infx.bench wait`, `SlurmBackend.stage_image/submit_batch` and the `infx.results.agentic` private helpers in `sweep_result.py` still exist.
8. **Not done here (later).** The list from the spec's "Out of scope", plus deviation 9 of this plan.

- [ ] **Step 2: Humanize**

Run the `humanizer` skill on the file (user rule for written reports) and apply its edits. Keep every fact, path and number.

- [ ] **Step 3: Commit**

```bash
git add inferencex-e2e/docs/serving/ARCHITECTURE.md
git commit -m "docs(serving): ARCHITECTURE of the serving path"
```

---

### Task 12: Live verification and runner cutover (manual, gated)

Nothing in this task runs without the user's explicit OK, step by step. It touches shared GPU nodes and GitHub. Every OK is for that step only.

- [ ] **Step 1: Full local suite**

Run: `PYTEST infx/tests/serving infx/tests/launch infx/tests/clusters`
Expected: all pass. Paste the summary line in the report to the user.

- [ ] **Step 2 (needs OK): Push the branch**

`git push -u origin refactor/vng-benchmark`. The existing runners do not serve the new labels yet, so nothing runs.

- [ ] **Step 3 (needs OK): Register the new slots next to the old runners**

Follow `serving-runner-setup` §1–2 for `hgx-h200-02` first (4 slots), then `hgx-h200-01`, then the two RTX nodes. The RTX nodes get the "move to stackops" procedure. Old runners keep running: Slurm shares the GPUs.

- [ ] **Step 4 (needs OK): Live checks**

Dispatch each of the following with `dispatch.sh` (temporary YAMLs, do not commit them) and record run URL and outcome:
1. `configs/serving/qwen38-27b-tau2-smoke.yaml` on `cluster:hcm-hgx02`: file dataset, engine only.
2. `agentx` + `overrides: {duration: 180}` on `cluster:hcm-rtx01`.
3. `configs/serving/gemma4-livekit-2rep-cacheaware.yaml` with `overrides: {duration: 180}`: router + 2 replicas.
4. `configs/serving/glm53-w4afp8-router-mooncake-e2.yaml` with `overrides: {duration: 180}`: storage sidecar.
5. A synthetic search point (`workload: {isl: 8k, osl: 1k, cache: 50, search: true}`): aggregate has a `state`.
6. Ports: two jobs at once on `cluster:hcm-hgx02` (1 GPU each); both succeed. `squeue -w hgx-h200-02` shows both running, and neither is exclusive.
7. Parity: the same engine-only serving YAML with `agentx` smoke on `vng-benchmark` (old syntax) and on this branch. Compare throughput and TTFT p50 with `result-report`; expect within run-to-run noise. Report the numbers.

- [ ] **Step 5 (needs OK): Team switch and retirement**

1. Tell the team to dispatch from this branch.
2. Running `bench/*` branches finish on `vng-benchmark`.
3. Retire the old runners (`hardware-hcm-*`, `hcm-rtx6000pro_*` under `thanglq5`, the offline `h200-greennode-slurm-1x_*`, `bench-client_*`) with `serving-runner-setup` "Retire".
4. Renaming the branch to the main VNG working branch is a separate decision for the user. If it happens, update the `setup.sh` URL in `serving-runner-setup`.
