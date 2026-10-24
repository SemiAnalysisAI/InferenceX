from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

WORKERS = {
    "unquantized": "operatorx_ascend_worker",
    "fp8": "operatorx_fp8_block_worker",
}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def normalize(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = {}
    for entry in entries:
        op = entry.get("opType", entry.get("type"))
        if not isinstance(op, str) or not isinstance(entry.get("args"), dict):
            raise ValueError("Each entry needs type/opType and an args object")
        if "opType" in entry and "type" in entry and entry["opType"] != entry["type"]:
            raise ValueError("Conflicting operation types")
        semantic = {"opType": op, "args": entry["args"]}
        identity = canonical(semantic)
        cid = op + "-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
        if "case_id" in entry and entry["case_id"] != cid:
            raise ValueError("case_id does not match complete arguments")
        sources = entry.get("sources", [])
        if not isinstance(sources, list):
            raise ValueError("sources must be a list")
        if identity not in cases:
            cases[identity] = {**semantic, "case_id": cid, "sources": []}
        for source in sources:
            if source not in cases[identity]["sources"]:
                cases[identity]["sources"].append(source)
    return list(cases.values())


def make_plan(entries: list[dict[str, Any]], kind: str, budget: int) -> dict[str, Any]:
    if type(budget) is not int or budget <= 0:
        raise ValueError("Memory budget must be a positive integer")
    worker = importlib.import_module(WORKERS[kind])
    config = {
        "cache_write_bytes": 256 * 1024**2,
        "workspace_reserve_bytes": 2 * 1024**3,
        "memory_budget_bytes": budget,
    }
    cases = []
    for case in normalize(entries):
        prepared = worker.plan_case(
            case,
            cache_write_bytes=config["cache_write_bytes"],
            workspace_reserve_bytes=config["workspace_reserve_bytes"],
        )
        case["support"] = prepared["support"]
        case["phase"] = "implementation_required"
        if prepared["support"]["supported"]:
            case["memory"] = prepared["memory_estimate"]
            case["phase"] = (
                "new" if case["memory"]["required_bytes"] <= budget else "memory_review"
            )
        cases.append(case)
    cases.sort(
        key=lambda c: (
            c["phase"],
            c.get("memory", {}).get("required_bytes", 0),
            c["case_id"],
        )
    )
    groups = {}
    for case in cases:
        if case["phase"] == "new":
            spec = {
                k: v
                for k, v in case["support"]["normalized"].items()
                if k not in {"m", "n", "k"}
            }
            groups.setdefault(canonical(spec), []).append(case)
    smoke = [
        rows[i]["case_id"]
        for rows in groups.values()
        for i in sorted({0, len(rows) // 2, len(rows) - 1})
    ]
    return {
        "schema_version": 1,
        "config": config,
        "cases": cases,
        "smoke_case_ids": smoke,
        "counts": dict(Counter(c["phase"] for c in cases)),
        "kind": kind,
        "worker_sha256": hashlib.sha256(Path(worker.__file__).read_bytes()).hexdigest(),
        "dependency_hashes": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("operatorx_queued_timing.py", "operatorx_ascend_worker.py")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--kind", choices=WORKERS, required=True)
    parser.add_argument("--memory-budget-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = []
    for path in args.input:
        value = json.loads(path.read_text())
        value = value["cases"] if isinstance(value, dict) else value
        if not isinstance(value, list):
            raise ValueError("Input must be a testlist array or a cases object")
        entries.extend(value)
    plan = make_plan(entries, args.kind, args.memory_budget_bytes)
    with args.output.open("x") as target:
        json.dump(plan, target, indent=2, allow_nan=False)
        target.write("\n")
    print(json.dumps(plan["counts"]))


if __name__ == "__main__":
    main()
