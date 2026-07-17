from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from operatorx.core.op import Op
from operatorx.core.run import RunInfo, from_dict as _run_from_dict, to_dict as _run_to_dict


SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class Result:
    op: Op
    metrics: Mapping[str, float] = field(default_factory=dict)
    status: str = "ok"            # "ok" | "unsupported" | "error"
    message: str | None = None    # only set when status != "ok"
    testlist: str | None = None   # which testlist file produced this op


def to_dict(r: Result) -> dict:
    op_dict: dict = {
        "type": r.op.type,
        "args": dict(r.op.args),
        "backend": r.op.backend,
    }
    if r.op.name is not None:
        op_dict["name"] = r.op.name
    out: dict = {
        "op": op_dict,
        "metrics": dict(r.metrics),
        "status": r.status,
    }
    if r.message is not None:
        out["message"] = r.message
    if r.testlist is not None:
        out["testlist"] = r.testlist
    return out


def _result_from_dict(d: dict) -> Result:
    op = Op(
        type=d["op"]["type"],
        args=d["op"]["args"],
        backend=d["op"]["backend"],
        name=d["op"].get("name"),
    )
    return Result(
        op=op,
        metrics=d.get("metrics", {}),
        status=d.get("status", "ok"),
        message=d.get("message"),
        testlist=d.get("testlist"),
    )


def write_run_result(path: Path | str, run: RunInfo, results: Iterable[Result]) -> None:
    """Write one run's worth of results to ``path`` in the run-result wrapper shape.

    Body: ``{"schema_version", "run": {...RunInfo...}, "rows": [{op, metrics}, ...]}``.
    Parent directories are created. Existing files are overwritten — one file = one run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "schema_version": SCHEMA_VERSION,
        "run": _run_to_dict(run),
        "rows": [to_dict(r) for r in results],
    }
    path.write_text(json.dumps(body, indent=2))


def read_run_result(path: Path | str) -> tuple[RunInfo, list[Result]]:
    """Inverse of :func:`write_run_result`. Returns ``(run, rows)``."""
    path = Path(path)
    body = json.loads(path.read_text())
    run = _run_from_dict(body["run"])
    rows = [_result_from_dict(d) for d in body["rows"]]
    return run, rows
