"""File discovery and loading for the AgentX result and power CLIs."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any


def resolve_artifact_dir(result_dir: Path) -> Path:
    """Find the dir containing aiperf's profile_export* files."""
    base = result_dir / "aiperf_artifacts"
    if (base / "profile_export.jsonl").is_file():
        return base
    if base.is_dir():
        for child in sorted(base.iterdir()):
            if child.is_dir() and (child / "profile_export.jsonl").is_file():
                return child
    return base


def load_aggregate(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_records(path: Path) -> list[dict[str, Any]]:
    records, _ = load_records_with_accounting(path)
    return records


def load_records_with_accounting(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load profiling records from profile_export.jsonl.

    Warmup rows are diagnostics only. Older artifacts did not have
    metadata.benchmark_phase, so missing phase is treated as profiling.
    """
    records: list[dict[str, Any]] = []
    accounting: dict[str, Any] = {
        "records_total": 0,
        "records_profiled": 0,
        "records_dropped_total": 0,
        "records_warmup_dropped": 0,
        "records_error_dropped": 0,
        "error_categories": {},
    }
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            accounting["records_total"] += 1
            phase = obj.get("metadata", {}).get("benchmark_phase")
            is_warmup = phase is not None and phase != "profiling"
            error = obj.get("error")
            if is_warmup:
                accounting["records_warmup_dropped"] += 1
            if error:
                accounting["records_error_dropped"] += 1
                category = _error_category(error)
                categories = accounting["error_categories"]
                categories[category] = categories.get(category, 0) + 1
            if error or is_warmup:
                continue
            records.append(obj)
    accounting["records_profiled"] = len(records)
    accounting["records_dropped_total"] = accounting["records_total"] - len(records)
    return records, accounting


def _error_category(error: Any) -> str:
    if isinstance(error, dict):
        for key in ("type", "error_type", "code", "class", "status"):
            value = error.get(key)
            if value not in (None, ""):
                return str(value)
        message = error.get("message") or error.get("error")
    else:
        message = error

    if not message:
        return "unknown"
    first_line = str(message).strip().splitlines()[0]
    return (first_line.split(":", 1)[0] or "unknown")[:120]


def load_server_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_server_log_head(path: Path, max_bytes: int = 64 * 1024 * 1024) -> str | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = f.read(max_bytes)
    return data.decode("utf-8", errors="replace").replace("\x00", "")


def find_server_log_paths(result_dir: Path) -> list[Path]:
    paths: list[Path] = []
    direct = result_dir / "server.log"
    if direct.is_file():
        paths.append(direct)

    for root in (result_dir, *result_dir.parents[:3]):
        if not root.is_dir():
            continue
        paths.extend(sorted(root.glob("watchtower-*.out")))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def _hf_traces_dir(hf_dataset_name: str | None, env: Mapping[str, str]) -> Path | None:
    if not hf_dataset_name:
        return None

    hub_cache = env.get("HF_HUB_CACHE") or env.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        cache_root = Path(hub_cache)
    else:
        home = env.get("HF_HOME")
        cache_root = Path(home) / "hub" if home else Path.home() / ".cache" / "huggingface" / "hub"

    snap_root = cache_root / f"datasets--{hf_dataset_name.replace('/', '--')}" / "snapshots"
    if not snap_root.is_dir():
        return None

    # The export has no resolved revision; multiple snapshots are ambiguous.
    snapshots = [path for path in snap_root.iterdir() if path.is_dir()]
    return snapshots[0] if len(snapshots) == 1 else None


def _iter_trace_blobs(traces_dir: Path) -> Iterator[dict[str, Any]]:
    for path in sorted(traces_dir.glob("*.jsonl")):
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue

    for path in sorted(traces_dir.glob("*.json")):
        try:
            with open(path) as f:
                yield json.load(f)
        except (json.JSONDecodeError, OSError):
            continue


def iter_trace_blobs(
    aggregate: Mapping[str, Any], env: Mapping[str, str],
) -> Iterator[dict[str, Any]]:
    """Read the declared dataset only when request processing consumes traces."""
    metadata = aggregate.get("metadata")
    dataset = metadata.get("dataset") if isinstance(metadata, dict) else None
    hf_dataset_name = dataset.get("hf_dataset_name") if isinstance(dataset, dict) else None
    if not isinstance(hf_dataset_name, str):
        return
    traces_dir = _hf_traces_dir(hf_dataset_name, env)
    if traces_dir is not None:
        yield from _iter_trace_blobs(traces_dir)
