#!/usr/bin/env python3
"""Build the CollectiveX sweep matrix and extract execution shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "bench"))

import ep_harness  # noqa: E402


TOPOLOGY_FIELDS = (
    "nodes", "gpus_per_node", "scale_up_domain", "scope", "scale_up_transport",
    "scale_out_transport", "transport", "topology_class",
)


def _load_config(name: str) -> dict[str, Any]:
    return json.loads((HERE / "configs" / name).read_text(encoding="utf-8"))


SWEEP = _load_config("sweep.json")
PLATFORMS = _load_config("platform_config.json")["platforms"]
SWEEP_BACKENDS = tuple(dict.fromkeys(
    backend for platform in PLATFORMS.values() for backend in platform["backends"]
))
# Dispatch precisions each backend realizes. BF16 is the universal control; FP8 is a
# caller-prequantized dispatch (DeepEP blockwise e4m3fn; MoRI per-SKU e4m3 tensor cast).
# A precision the backend does not list is gated out at generation (no case emitted).
BACKEND_PRECISIONS = {
    "deepep-v2": ("bf16", "fp8"),
    "mori": ("bf16", "fp8"),
    "uccl-ep": ("bf16", "fp8"),
}
# Short shard-ID slug per non-normal mode. Normal-mode shard IDs carry no mode
# segment so existing references stay valid; a low-latency shard adds "-ll".
_MODE_SLUG = {"low-latency": "ll"}


def _ll_runnable(platform: dict[str, Any], backend: str, ep: int) -> bool:
    """Whether this SKU/backend/EP cell can run the low-latency decode kernels.

    Capability is data-driven from the platform registry's optional ``ll_backends``
    map (backend -> runnable EP degrees), mirroring ``backends`` for normal mode. A SKU
    with no ``ll_backends`` entry emits no low-latency cases. This is scope, not a wall:
    the low-latency DeepEP path mandates NVSHMEM/IBGDA (and thus gdrdrv) even for
    single-node EP8, so the LL-runnable set is narrower than and distinct from the
    normal-mode set and is enabled per SKU as bring-up confirms it.
    """
    return ep in platform.get("ll_backends", {}).get(backend, [])


def _topology(platform: dict[str, Any], ep: int) -> dict[str, Any]:
    gpus_per_node = platform["gpus_per_node"]
    if ep % gpus_per_node:
        raise SystemExit(f"EP{ep} is not divisible by {gpus_per_node} GPUs per node")
    product = platform["product"]
    domain = platform["scale_up_domain"]
    scale_up = platform["scale_up_transport"]
    scale_out = ep > domain
    if scale_up == "mnnvl":
        scale_up_class = f"{product}-nvl{domain}-mnnvl"
    elif scale_up == "xgmi":
        scale_up_class = f"{product}-xgmi"
    else:
        scale_up_class = f"{product}-{scale_up}-island"
    return {
        "nodes": ep // gpus_per_node,
        "gpus_per_node": gpus_per_node,
        "scale_up_domain": domain,
        "scope": "scale-out" if scale_out else "scale-up",
        "scale_up_transport": scale_up,
        "scale_out_transport": "rdma" if scale_out else None,
        "transport": f"{scale_up}-rdma" if scale_out else scale_up,
        "topology_class": f"{product}-{scale_up}-rdma" if scale_out else scale_up_class,
    }


def _selected_backends(backend: str) -> list[str]:
    if backend == "all":
        return list(SWEEP_BACKENDS)
    if backend not in SWEEP_BACKENDS:
        raise SystemExit(f"unknown --backend {backend!r}; have {list(SWEEP_BACKENDS)}")
    return [backend]


def resolve_matrix(
    backend: str = "all",
    only_sku: str = "",
    exclude_skus: str = "",
    ep_sizes: str = "",
    precisions: str = "",
    modes: str = "",
) -> dict[str, Any]:
    """Resolve the fixed sweep into allocation-sized workflow shards."""
    selected_eps: set[int] = set()
    for value in filter(None, (part.strip() for part in ep_sizes.split(","))):
        if not value.isdigit() or int(value) <= 0:
            raise SystemExit(f"invalid --ep-sizes {ep_sizes!r}; expected positive integers")
        selected_eps.add(int(value))
    known_precisions = set(SWEEP["precisions"])
    selected_precisions = {value.strip() for value in precisions.split(",") if value.strip()}
    unknown_precisions = sorted(selected_precisions - known_precisions)
    if unknown_precisions:
        raise SystemExit(
            f"unknown --precisions {unknown_precisions}; have {sorted(known_precisions)}"
        )
    known_modes = set(SWEEP["modes"])
    selected_modes = {value.strip() for value in modes.split(",") if value.strip()}
    unknown_modes = sorted(selected_modes - known_modes)
    if unknown_modes:
        raise SystemExit(f"unknown --modes {unknown_modes}; have {sorted(known_modes)}")

    if only_sku and only_sku not in PLATFORMS:
        raise SystemExit(f"unknown --only-sku {only_sku!r}; have {sorted(PLATFORMS)}")
    excluded = {value.strip() for value in exclude_skus.split(",") if value.strip()}
    unknown = sorted(excluded - set(PLATFORMS))
    if unknown:
        raise SystemExit(f"unknown --exclude-skus {unknown}; have {sorted(PLATFORMS)}")
    if only_sku in excluded:
        raise SystemExit("--only-sku and --exclude-skus select disjoint pools")

    timing = SWEEP["timing"]
    timing_profile = ":".join(str(timing[key]) for key in (
        "iters_per_trial", "trials_per_point", "warmup_iters_per_trial",
    ))
    workload = SWEEP["workload"]
    targets = _selected_backends(backend)
    requested_cases: list[dict[str, Any]] = []
    shards: dict[tuple[str, str, str, int, str], list[dict[str, Any]]] = {}

    for sku in sorted(PLATFORMS):
        if (only_sku and sku != only_sku) or sku in excluded:
            continue
        platform = PLATFORMS[sku]
        for ep in SWEEP["ep_degrees"]:
            if selected_eps and ep not in selected_eps:
                continue
            topology = _topology(platform, ep)
            for phase, ladder in workload["token_ladders"].items():
                for target in targets:
                    runnable_eps = platform["backends"].get(target)
                    if runnable_eps is None:
                        continue
                    runnable = ep in runnable_eps
                    backend_precisions = BACKEND_PRECISIONS.get(target, ("bf16",))
                    supported = [
                        precision for precision in SWEEP["precisions"]
                        if precision in backend_precisions
                    ]
                    if runnable:
                        # A runnable cell fans out over the modes it realizes at this
                        # phase (normal everywhere; low-latency only where the mode is
                        # decode-scoped in sweep.json AND the cell is capability-gated in),
                        # crossed with the selected precisions.
                        cell_modes = [
                            mode for mode in SWEEP["modes"]
                            if phase in SWEEP["modes"][mode]
                            and (not selected_modes or mode in selected_modes)
                            and (mode == "normal" or _ll_runnable(platform, target, ep))
                        ]
                        emit_precisions = [
                            precision for precision in supported
                            if not selected_precisions or precision in selected_precisions
                        ]
                    else:
                        # An ep-unsupported cell records once with a placeholder mode and
                        # precision (never dispatched). It is independent of BOTH the --modes
                        # and --precisions filters (so a subset stays a strict subset of the
                        # full matrix) and of the sweep.json mode/precision ORDER (so
                        # reordering the config never silently renames unsupported case_ids):
                        # normal mode plus sorted() pin it to the supported control (bf16
                        # sorts first). Low-latency adds only runnable cells, never
                        # unsupported rows.
                        cell_modes = ["normal"]
                        emit_precisions = sorted(supported)[:1]
                    for mode in cell_modes:
                        for precision in emit_precisions:
                            case = {
                                "suite": SWEEP["suite"],
                                "workload": workload["name"],
                                "backend": target,
                                "routing": SWEEP["routing"],
                                "precision": precision,
                                "phase": phase,
                                "ep": ep,
                                "hidden": workload["hidden"],
                                "topk": workload["topk"],
                                "experts": workload["routed_experts"],
                                "seed": workload["seed"],
                                "ladder": " ".join(map(str, ladder)),
                                "mode": mode,
                                "timing": timing_profile,
                                **{field: topology[field] for field in TOPOLOGY_FIELDS},
                            }
                            case["case_id"] = ep_harness.case_id(sku, case)
                            requested_cases.append({
                                "sku": sku,
                                "case": case,
                                "disposition": "runnable" if runnable else "unsupported",
                                "reason": None if runnable else "backend-platform-unsupported",
                                "detail": None,
                            })
                            if runnable:
                                shards.setdefault(
                                    (sku, target, mode, topology["nodes"], precision), []
                                ).append(case)

    shards_by_sku: dict[str, list[dict[str, Any]]] = {}
    for (sku, target, mode, nodes, precision), cases in sorted(shards.items()):
        first = cases[0]
        # Normal-mode shard IDs are unchanged (no mode segment) so existing references
        # stay valid; a non-normal mode inserts a short slug (low-latency -> "ll").
        mode_segment = "" if mode == "normal" else f"-{_MODE_SLUG[mode]}"
        shards_by_sku.setdefault(sku, []).append({
            "id": f"{sku}-{target}{mode_segment}-{precision}-n{nodes}",
            "sku": sku,
            "backend": target,
            "mode": mode,
            "launcher": PLATFORMS[sku]["launcher"],
            "nodes": nodes,
            "gpus_per_node": first["gpus_per_node"],
            "scale_up_domain": first["scale_up_domain"],
            "cases": cases,
        })
    include = [
        shards_by_sku[sku][index]
        for index in range(max(map(len, shards_by_sku.values()), default=0))
        for sku in sorted(shards_by_sku)
        if index < len(shards_by_sku[sku])
    ]
    return {
        "version": SWEEP["version"],
        "requested_cases": requested_cases,
        "include": include,
    }


def extract_shard(matrix_path: str, shard_id: str, output_path: str) -> dict[str, Any]:
    """Write one generator-produced shard as a runner control document."""
    document = json.loads(Path(matrix_path).read_text(encoding="utf-8"))
    matches = [item for item in document["include"] if item["id"] == shard_id]
    if len(matches) != 1:
        raise SystemExit(f"expected one shard {shard_id!r}, found {len(matches)}")
    source = matches[0]
    control = {key: source[key] for key in ("id", "sku", "backend", "nodes", "cases")}
    control["version"] = document["version"]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(control, sort_keys=True, separators=(",", ":")) + "\n")
    return control


def main() -> int:
    parser = argparse.ArgumentParser(description="CollectiveX matrix resolver")
    parser.add_argument("--backend", default="all")
    parser.add_argument("--only-sku", default="")
    parser.add_argument("--exclude-skus", default="")
    parser.add_argument("--ep-sizes", default="")
    parser.add_argument("--precisions", default="",
                        help="comma-separated subset of configs/sweep.json precisions")
    parser.add_argument("--modes", default="",
                        help="comma-separated subset of configs/sweep.json modes "
                             "(normal, low-latency); blank = all")
    parser.add_argument("--extract-from", default="", metavar="MATRIX")
    parser.add_argument("--shard-id", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if args.extract_from:
        if not all((args.shard_id, args.out)):
            parser.error("shard extraction requires --shard-id and --out")
        control = extract_shard(args.extract_from, args.shard_id, args.out)
        print(f"extracted {control['id']}: {len(control['cases'])} cases", file=sys.stderr)
        print(json.dumps(control, separators=(",", ":")))
        return 0

    matrix = resolve_matrix(
        backend=args.backend,
        only_sku=args.only_sku,
        exclude_skus=args.exclude_skus,
        ep_sizes=args.ep_sizes,
        precisions=args.precisions,
        modes=args.modes,
    )
    if args.out:
        Path(args.out).write_text(
            json.dumps(matrix, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    runnable = sum(item["disposition"] == "runnable" for item in matrix["requested_cases"])
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
