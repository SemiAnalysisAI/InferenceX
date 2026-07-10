#!/usr/bin/env python3
"""Resolve CollectiveX suites and extract execution shards.

Mode changes measurement semantics and therefore participates in case identity.
Dispatch and combine are fixed BF16 benchmark facts, so a case's coordinates are
suite/workload/backend/topology only; the matrix schedules, it never ranks.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "bench"))

try:  # Shard extraction on GPU runners is intentionally stdlib-only.
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised by the workflow environment
    yaml = None

import capability as cap  # noqa: E402
import ep_harness  # noqa: E402


EP_TIMING_PROFILE = (
    f"{ep_harness.TIMED_ITERS_PER_TRIAL}:{ep_harness.TRIALS_PER_POINT}:"
    f"{ep_harness.WARMUP_ITERS_PER_TRIAL}"
)
TOPOLOGY_FIELDS = (
    "nodes", "gpus_per_node", "scale_up_domain", "scope", "scale_up_transport",
    "scale_out_transport", "transport", "topology_class",
)


if yaml is not None:
    class _UniqueKeyLoader(yaml.SafeLoader):
        pass

    def _unique_mapping(loader: Any, node: Any, deep: bool = False) -> dict[Any, Any]:
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise SystemExit(f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    _UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
    )


def _load(name: str) -> dict[str, Any]:
    if yaml is None:
        raise SystemExit("matrix generation requires PyYAML; shard extraction does not")
    try:
        with (HERE / "configs" / name).open() as fh:
            document = yaml.load(fh, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise SystemExit(f"configs/{name} is not valid YAML: {exc}") from exc
    if not isinstance(document, dict):
        raise SystemExit(f"configs/{name} must contain a YAML object")
    return document


def _dims(workloads: dict[str, Any], name: str) -> tuple[int, int, int]:
    config = workloads[name]
    return config["hidden"], config["topk"], config["routed_experts"]


def _ladder(suite: dict[str, Any], phase: str) -> str:
    points = suite.get(f"token_points_{phase}")
    if points is None:
        points = ep_harness.DECODE_LADDER if phase == "decode" else ep_harness.PREFILL_LADDER
    if (not isinstance(points, list) or not points
            or any(isinstance(point, bool) or not isinstance(point, int) or point <= 0
                   for point in points)
            or points != sorted(set(points))):
        raise SystemExit(f"invalid {phase} token ladder: {points!r}")
    return " ".join(map(str, points))


def _semantic_points(sku: str, case: dict[str, Any]) -> list[str]:
    execution = {
        key: value for key, value in case.items()
        if key not in {"canonical", "case_id", "ladder", "suite", "workload"}
    }
    return [
        json.dumps(
            {"sku": sku, "tokens_per_rank": int(point), **execution},
            sort_keys=True,
            separators=(",", ":"),
        )
        for point in case["ladder"].split()
    ]


def _select_backends(backend: str, backends: str) -> list[str]:
    available = list(cap.SWEEP_BACKENDS)
    if backend and backends:
        raise SystemExit("--backend and --backends are mutually exclusive")
    if backends:
        names = available if backends == "all" else [
            value.strip() for value in backends.split(",") if value.strip()
        ]
    else:
        names = [backend or "deepep-v2"]
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise SystemExit(f"unknown backend values {unknown}; have {available}")
    if len(names) != len(set(names)):
        raise SystemExit("backend selection contains duplicates")
    return names


def resolve_matrix(
    suites: str = "all",
    backend: str = "",
    backends: str = "",
    only_sku: str = "",
    exclude_skus: str = "",
    ep_sizes: str = "",
    max_cases: int = 128,
) -> dict[str, Any]:
    """Resolve suite configuration into allocation-sized workflow shards."""
    if max_cases <= 0:
        raise SystemExit("--max-cases must be positive")
    # --ep-sizes narrows the matrix to specific expert-parallel degrees at dispatch
    # time: "8" keeps every EP8 shard and drops EP16, so a comprehensive run can
    # co-schedule the 8-GPU SKUs' single-node EP8 with the GB SKUs' two-node EP8
    # without dispatching any EP16 leg. Blank keeps every degree. The resulting
    # matrix is a partial subset that only omits cases; it never reclassifies them.
    selected_eps: set[int] = set()
    for value in (part.strip() for part in ep_sizes.split(",")):
        if not value:
            continue
        if not value.isdigit() or int(value) <= 0:
            raise SystemExit(f"invalid --ep-sizes {ep_sizes!r}; expected positive integers")
        selected_eps.add(int(value))
    if only_sku and only_sku not in cap.PLATFORMS:
        raise SystemExit(f"unknown --only-sku {only_sku!r}; have {sorted(cap.PLATFORMS)}")
    # --exclude-skus narrows the matrix to a subset by dropping whole runner pools
    # — e.g. exclude a SKU whose cluster is unavailable. It only omits cases.
    excluded = {value.strip() for value in exclude_skus.split(",") if value.strip()}
    unknown_excluded = sorted(excluded - set(cap.PLATFORMS))
    if unknown_excluded:
        raise SystemExit(
            f"unknown --exclude-skus {unknown_excluded}; have {sorted(cap.PLATFORMS)}"
        )
    if only_sku and only_sku in excluded:
        raise SystemExit("--only-sku and --exclude-skus select disjoint pools")

    suites_document = _load("suites.yaml")
    workloads = suites_document["workloads"]
    registry = suites_document["suites"]
    select_all = suites == "all"
    names = (
        list(registry)
        if select_all
        else [value.strip() for value in suites.split(",") if value.strip()]
    )
    if not names or len(names) != len(set(names)):
        raise SystemExit("suite selection must be non-empty and unique")
    unknown = sorted(set(names) - set(registry))
    if unknown:
        raise SystemExit(f"unknown suites {unknown}; have {sorted(registry)}")
    targets = _select_backends(backend, backends)

    shards: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    requested_cases: list[dict[str, Any]] = []
    scheduled: set[str] = set()
    for suite_name in names:
        suite = registry[suite_name]
        mode = suite["mode"]
        phases = suite["phases"]
        routings = suite["routings"]
        suite_backends = set(suite.get("backends", cap.SWEEP_BACKENDS))
        suite_targets = [target for target in targets if target in suite_backends]
        if not suite_targets:
            continue
        for platform_name in suite["platforms"]:
            if only_sku and platform_name != only_sku:
                continue
            if platform_name in excluded:
                continue
            ep_degrees = suite["ep_degrees"]
            for workload, ep, phase, routing, target in itertools.product(
                suite["workloads"], ep_degrees, phases, routings,
                suite_targets,
            ):
                if selected_eps and ep not in selected_eps:
                    continue
                topology = cap.topology_for(platform_name, ep)
                if topology is None:
                    raise SystemExit(
                        f"suite {suite_name}: {platform_name} EP{ep} is not registered"
                    )
                nodes = int(topology["nodes"])
                capability_disposition, capability_detail = cap.resolve_disposition(
                    platform_name,
                    target,
                    ep=ep,
                    nodes=nodes,
                    routing=routing,
                    mode=mode,
                )
                hidden, topk, experts = _dims(workloads, workload)

                def add_case(
                    case_ladder: str,
                    disposition: str,
                    reason: str | None,
                    detail: str | None,
                ) -> None:
                    case: dict[str, Any] = {
                        "suite": suite_name,
                        "workload": workload,
                        "backend": target,
                        "routing": routing,
                        "phase": phase,
                        "ep": ep,
                        "hidden": hidden,
                        "topk": topk,
                        "experts": experts,
                        "samples_per_point": ep_harness.TIMED_SAMPLES_PER_POINT,
                        "warmup_semantics": ep_harness.WARMUP_SEMANTICS,
                        "ladder": case_ladder,
                        "mode": mode,
                        "timing": EP_TIMING_PROFILE,
                        "canonical": True,
                        **{field: topology[field] for field in TOPOLOGY_FIELDS},
                    }
                    for signature in _semantic_points(platform_name, case):
                        if signature in scheduled:
                            raise SystemExit(
                                f"suite {suite_name}: duplicate semantic point for {platform_name}"
                            )
                        scheduled.add(signature)
                    # Same function the harness recomputes at run time — a scheduled
                    # case ID can never drift from its realized factors.
                    case["case_id"] = ep_harness.case_id(platform_name, case)
                    requested_cases.append(
                        {
                            "sku": platform_name,
                            "case": case,
                            "disposition": disposition,
                            "reason": reason,
                            "detail": detail,
                        }
                    )
                    if disposition == "runnable":
                        shards.setdefault((platform_name, target, nodes), []).append(case)

                requested_ladder = _ladder(suite, phase)
                if capability_disposition == "unsupported":
                    add_case(
                        requested_ladder,
                        "unsupported",
                        "backend-platform-unsupported",
                        capability_detail,
                    )
                    continue
                add_case(requested_ladder, "runnable", None, None)

    shards_by_sku: dict[str, list[dict[str, Any]]] = {}
    for (sku, target, nodes), cases in sorted(shards.items()):
        chunk_size = max_cases
        for offset in range(0, len(cases), chunk_size):
            chunk = cases[offset:offset + chunk_size]
            part = offset // chunk_size
            shard_id = f"{sku}-{target}-n{nodes}"
            if len(cases) > chunk_size:
                shard_id += f"-p{part}"
            shards_by_sku.setdefault(sku, []).append({
                "id": shard_id,
                "sku": sku,
                "backend": target,
                "launcher": cap.PLATFORMS[sku]["launcher"],
                **{field: chunk[0][field] for field in TOPOLOGY_FIELDS},
                "n": len(chunk),
                "execution_weight": execution_weight(chunk),
                "case_ids": [case["case_id"] for case in chunk],
                "cases": chunk,
            })
    include = [
        shards_by_sku[sku][round_index]
        for round_index in range(max(map(len, shards_by_sku.values()), default=0))
        for sku in sorted(shards_by_sku)
        if round_index < len(shards_by_sku[sku])
    ]
    return {
        "version": suites_document["version"],
        "requested_cases": requested_cases,
        "include": include,
    }


def execution_weight(cases: list[dict[str, Any]]) -> int:
    """Return GPU-point work used to bound workflow parallelism."""
    return sum(int(case["ep"]) * len(case["ladder"].split()) for case in cases)


def extract_shard(matrix_path: str, shard_id: str, output_path: str) -> dict[str, Any]:
    """Select one generator-produced shard and write its execution document."""
    with open(matrix_path) as fh:
        document = json.load(fh)
    matches = [item for item in document["include"] if item["id"] == shard_id]
    if len(matches) != 1:
        raise SystemExit(f"expected one shard {shard_id!r}, found {len(matches)}")
    source = matches[0]
    control = {
        key: source[key]
        for key in ("id", "sku", "backend", "nodes", "n", "execution_weight", "cases")
    }
    control["version"] = document["version"]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(control, sort_keys=True, separators=(",", ":")) + "\n")
    return control


def main() -> int:
    parser = argparse.ArgumentParser(description="CollectiveX matrix resolver")
    parser.add_argument("--suites", default="all", help="'all' or comma-list of suites")
    parser.add_argument("--backend", default="", help="select one EP backend")
    parser.add_argument("--backends", default="", help="'all' or comma-list of EP backends")
    parser.add_argument("--only-sku", default="")
    parser.add_argument(
        "--exclude-skus",
        default="",
        help="comma-list of runner pools to drop (partial matrix); disjoint from --only-sku",
    )
    parser.add_argument(
        "--ep-sizes",
        default="",
        help="comma-list of expert-parallel degrees to keep (e.g. 8 drops EP16); blank = all",
    )
    parser.add_argument("--max-cases", type=int, default=128)
    parser.add_argument("--extract-from", default="", metavar="MATRIX")
    parser.add_argument("--shard-id", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if args.extract_from:
        if not all((args.shard_id, args.out)):
            parser.error("shard extraction requires --shard-id and --out")
        control = extract_shard(args.extract_from, args.shard_id, args.out)
        print(f"extracted {control['id']}: {control['n']} cases", file=sys.stderr)
        print(json.dumps(control, separators=(",", ":")))
        return 0

    matrix = resolve_matrix(
        suites=args.suites,
        backend=args.backend,
        backends=args.backends,
        only_sku=args.only_sku,
        exclude_skus=args.exclude_skus,
        ep_sizes=args.ep_sizes,
        max_cases=args.max_cases,
    )
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(matrix, fh, sort_keys=True, separators=(",", ":"))
            fh.write("\n")
    runnable = sum(
        item["disposition"] == "runnable" for item in matrix["requested_cases"]
    )
    unsupported = len(matrix["requested_cases"]) - runnable
    print(
        f"resolved {len(matrix['include'])} shard-cells, "
        f"{runnable} runnable and {unsupported} unsupported cases",
        file=sys.stderr,
    )
    print(json.dumps(matrix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
