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
    message: str | None = None    # only when status != "ok"
    testlist: str | None = None   # testlist file that produced this op


def to_dict(r: Result) -> dict:
    op_dict: dict = {
        "type": r.op.type,
        "args": dict(r.op.args),
        "backend": r.op.backend,
    }
    op_dict["sources"] = list(r.op.sources)
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
        sources=d["op"].get("sources", ()),
    )
    return Result(
        op=op,
        metrics=d.get("metrics", {}),
        status=d.get("status", "ok"),
        message=d.get("message"),
        testlist=d.get("testlist"),
    )


def write_run_result(path: Path | str, run: RunInfo, results: Iterable[Result]) -> None:
    """{"schema_version", "run", "rows"}; one file = one run, replaced atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "schema_version": SCHEMA_VERSION,
        "run": _run_to_dict(run),
        "rows": [to_dict(r) for r in results],
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(body, indent=2))
    temporary.replace(path)


def read_run_result(path: Path | str) -> tuple[RunInfo, list[Result]]:
    path = Path(path)
    body = json.loads(path.read_text())
    run = _run_from_dict(body["run"])
    rows = [_result_from_dict(d) for d in body["rows"]]
    return run, rows
