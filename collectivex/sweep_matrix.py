#!/usr/bin/env python3
"""Build the CollectiveX sweep matrix and extract execution shards.

Each suite resolves into the same shard shape (one allocation, a list of cases), runs through
the pool's own launcher, and reaches its benchmark through the case codec in runtime/config.py.
"""
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
SWAP_SWEEP = _load_config("swap_sweep.json")
KV_SWEEP = _load_config("kv_sweep.json")
PLATFORMS = _load_config("platform_config.json")["platforms"]
SUITES = ("ep", "swap-blocks", "kv-transfer")


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
    # NCCL EP is BF16-only now; NCCL EP v0.2 supports FP8 dispatch, but the integration is
    # pending (see bench/ep_nccl.py).
    "nccl-ep": ("bf16",),
    # FlashInfer FP8 is dispatch-side only (scales as a fourth payload, combine stays BF16),
    # and uses the same per-128-block e4m3 recipe as deepep-v2/uccl-ep so the axis is
    # comparable. Realizable, but off every deployed path -- see OFF_PATH_PRECISIONS.
    "flashinfer-ep": ("bf16", "fp8"),
}
# Precisions a backend REALIZES but that no serving engine can select on that transport. Kept
# out of the default matrix so a production sweep measures deployable configurations, and still
# reachable by naming the precision explicitly (`--precisions fp8`) for transport comparison.
# vLLM accepts only nvfp4/mxfp8/bf16 on FlashInfer's one-sided all-to-all, so its FP8 row
# measures the collective off any path an engine selects. Declared here rather than on the
# adapter because this generator must resolve the matrix with no vendor imports.
OFF_PATH_PRECISIONS = {"flashinfer-ep": ("fp8",)}
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


def _comma_subset(flag: str, value: str, known) -> set[str]:
    selected = {part.strip() for part in value.split(",") if part.strip()}
    unknown = sorted(selected - set(known))
    if unknown:
        raise SystemExit(f"unknown --{flag} {unknown}; have {sorted(known)}")
    return selected


def _shard(sku: str, shard_id: str, backend: str, cases: list[dict[str, Any]],
           **extra: Any) -> dict[str, Any]:
    """One allocation-sized shard; its placement is its cases' (they share one allocation)."""
    first = cases[0]
    return {
        "id": shard_id,
        "sku": sku,
        # runs-on label: the SKU unless the registry names the pool's runners.
        "runner": PLATFORMS[sku].get("runner_label", sku),
        "backend": backend,
        "launcher": PLATFORMS[sku]["launcher"],
        **{field: first[field] for field in ("nodes", "gpus_per_node", "scale_up_domain")},
        **extra,
        "cases": cases,
    }


def _runnable(sku: str, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"sku": sku, "case": case, "disposition": "runnable", "reason": None, "detail": None}
            for case in cases]


def _vendor(platform: dict[str, Any]) -> str:
    return "amd" if platform["arch"].startswith("gfx") else "nvidia"


def _swap_shard(sku: str, profile_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One single-GPU shard per pool: a case per layout over the profile's block grid."""
    profile = SWAP_SWEEP["profiles"][profile_name]
    pinned = SWAP_SWEEP["sku_images"].get(sku, {})
    cases = []
    for layout in SWAP_SWEEP["layouts"]:
        case = {
            "suite": SWAP_SWEEP["suite"],
            "backend": "swap-blocks",
            "profile": profile_name,
            "layout": layout,
            "directions": " ".join(SWAP_SWEEP["directions"]),
            "block_bytes": " ".join(map(str, profile["block_bytes"])),
            "num_blocks": " ".join(map(str, profile["num_blocks"])),
            "max_payload_bytes": profile["max_payload_bytes"],
            "warmup": profile["warmup"],
            "iterations": profile["iterations"],
            "seed": SWAP_SWEEP["seed"],
            "device": SWAP_SWEEP["device"],
            "nodes": 1,
            "gpus_per_node": 1,
            "scale_up_domain": 1,
        }
        case["case_id"] = ep_harness.slug_id((sku, "swap-blocks", profile_name, layout))
        cases.append(case)
    staged = {key: pinned[key] for key in ("staged_image_dir",) if key in pinned}
    shard = _shard(
        sku, f"{sku}-swap-blocks", "swap-blocks", cases, suite=SWAP_SWEEP["suite"],
        image=pinned.get("image", SWAP_SWEEP["images"][_vendor(PLATFORMS[sku])]),
        allocation_minutes=SWAP_SWEEP["allocation_minutes"], **staged,
    )
    return _runnable(sku, cases), shard


def _kv_backend_spec(value: Any) -> dict[str, Any]:
    """A registry kv_backends value is a fabric list (the library runs the full sweep) or an
    object that restricts it: `ops` (a backend that cannot serve one direction on a fabric, e.g.
    mooncake on Pollara, where ionic RDMA READ is broken upstream), `image` (a backend shipped
    only inside a specific image, e.g. AMD's atom-dev mooncake build), `device` (an engine NIC
    filter template; `{gpu}` expands to the physical GPU index at runtime), and `pool_budget`
    (bytes, where the engine cannot register the default pool on that pool's NICs)."""
    if isinstance(value, list):
        value = {"fabrics": value}
    return {
        "fabrics": value["fabrics"],
        "ops": value.get("ops") or " ".join(KV_SWEEP["ops"]),
        "image": value.get("image"),
        "device": value.get("device") or "",
        "pool_budget": value.get("pool_budget"),
    }


def _kv_shards(sku: str, selected_precisions: set[str]) -> tuple[list, list[dict[str, Any]]]:
    """The kv-transfer shards one pool runs: one per (backend, fabric) its registry enables.

    A KV leg is 2 nodes x 1 GPU: the per-worker transfer pair an engine actually forms, not an
    allocation-wide collective. A pool with no ``kv_backends`` entry emits nothing -- absence is
    not-yet-enabled, mirroring ll_backends. The allocation and the per-case hang guard come from
    kv_sweep.json's scheduling block: KV grids run for hours, and gb300 paces ~1.8x gb200 at the
    top ISLs over mnnvl, so both are sized per pool rather than fleet-wide.
    """
    platform = PLATFORMS[sku]
    timing = KV_SWEEP["timing"]
    scheduling = KV_SWEEP["scheduling"].get(sku, KV_SWEEP["scheduling"]["default"])
    requested, shards = [], []
    for backend, raw in sorted(platform.get("kv_backends", {}).items()):
        spec = _kv_backend_spec(raw)
        for fabric in spec["fabrics"]:
            cases = []
            # A workload's dtype mix can be architectural (dsv4's fp8 slots), so the sweep config
            # maps each workload to its precisions; a test pins the map to kv_workload's PRESETS.
            for workload, workload_precisions in KV_SWEEP["workloads"].items():
                for precision in workload_precisions:
                    if selected_precisions and precision not in selected_precisions:
                        continue
                    case = {
                        "suite": KV_SWEEP["suite"],
                        "workload": workload,
                        "backend": backend,
                        "routing": "paged",
                        "precision": precision,
                        "phase": "xfer",
                        "ep": 2,
                        "mode": fabric,
                        "isl_ladder": " ".join(map(str, KV_SWEEP["isl_ladder"])),
                        "page_tokens": " ".join(map(str, KV_SWEEP["page_tokens"])),
                        "batch_sizes": " ".join(map(str, KV_SWEEP["batch_sizes"])),
                        "ops": spec["ops"],
                        "kv_device": spec["device"],
                        "pool_slack": KV_SWEEP["pool_slack"],
                        "seed": KV_SWEEP["seed"],
                        "warmup": timing["warmup_per_trial"],
                        "reps": timing["reps_per_trial"],
                        "trials": timing["trials_per_point"],
                        "nodes": 2,
                        "gpus_per_node": 1,
                        "scale_up_domain": platform["scale_up_domain"],
                        "scale_up_transport": platform["scale_up_transport"],
                        "topology_class": f"{platform['product']}-kv-{fabric}",
                    }
                    if spec["pool_budget"]:
                        case["pool_budget"] = spec["pool_budget"]
                    case["case_id"] = ep_harness.case_id(sku, case)
                    cases.append(case)
            if not cases:
                continue
            requested += _runnable(sku, cases)
            shards.append(_shard(
                sku, f"{sku}-kv-{backend}-{fabric}", backend, cases,
                suite=KV_SWEEP["suite"], mode=fabric, fabric=fabric,
                allocation_minutes=scheduling["allocation_minutes"],
                run_timeout=scheduling["run_timeout"],
                # The GitHub job must outlive the allocation, or it cancels a healthy shard
                # before the launcher's own guards act.
                job_timeout_minutes=scheduling["allocation_minutes"] + 30,
                **({"image": spec["image"]} if spec["image"] else {}),
            ))
    return requested, shards


def resolve_matrix(
    backend: str = "all",
    only_sku: str = "",
    exclude_skus: str = "",
    ep_sizes: str = "",
    precisions: str = "",
    modes: str = "",
    suites: str = "ep",
    swap_profile: str = "smoke",
) -> dict[str, Any]:
    """Resolve the selected suites into allocation-sized workflow shards."""
    selected_suites = _comma_subset("suites", suites, SUITES)
    if not selected_suites:
        raise SystemExit("--suites selects no suite")
    if "ep" not in selected_suites and (backend != "all" or ep_sizes or modes):
        raise SystemExit("EP filters need the ep suite in --suites")
    if not selected_suites & {"ep", "kv-transfer"} and precisions:
        raise SystemExit("--precisions needs the ep or kv-transfer suite in --suites")
    if swap_profile not in SWAP_SWEEP["profiles"]:
        raise SystemExit(
            f"unknown --swap-profile {swap_profile!r}; have {sorted(SWAP_SWEEP['profiles'])}"
        )
    selected_eps: set[int] = set()
    for value in filter(None, (part.strip() for part in ep_sizes.split(","))):
        if not value.isdigit() or int(value) <= 0:
            raise SystemExit(f"invalid --ep-sizes {ep_sizes!r}; expected positive integers")
        selected_eps.add(int(value))
    selected_precisions = _comma_subset("precisions", precisions, SWEEP["precisions"])
    selected_modes = _comma_subset("modes", modes, SWEEP["modes"])

    if only_sku and only_sku not in PLATFORMS:
        raise SystemExit(f"unknown --only-sku {only_sku!r}; have {sorted(PLATFORMS)}")
    excluded = _comma_subset("exclude-skus", exclude_skus, PLATFORMS)
    if only_sku in excluded:
        raise SystemExit("--only-sku and --exclude-skus select disjoint pools")

    timing = SWEEP["timing"]
    # Passed through as an object keyed by the sweep.json names. runtime/config.py maps each
    # key to its run_ep flag and holds the legacy colon-string decode in one migration function.
    timing_profile = {key: int(timing[key]) for key in (
        "iters_per_trial", "trials_per_point", "warmup_iters_per_trial",
        "chain_iters_per_trial", "chain_trials_per_point", "chain_drop",
    )}
    workload = SWEEP["workload"]
    targets = _selected_backends(backend)
    # Fail closed on a backend with no declared precisions: defaulting to BF16 would drop its
    # fp8 cases from the matrix entirely, and a case that never ran is invisible to every
    # downstream gate (run_sweep's non-bf16-dispatch guard only sees cases that did run).
    undeclared = [target for target in targets if target not in BACKEND_PRECISIONS]
    if undeclared:
        raise SystemExit(
            f"backends {undeclared} have no BACKEND_PRECISIONS entry; declare their dispatch "
            "precisions rather than silently defaulting to bf16"
        )
    requested_cases: list[dict[str, Any]] = []
    shards: dict[tuple[str, str, str, int, str], list[dict[str, Any]]] = {}

    selected_skus = [
        sku for sku in sorted(PLATFORMS)
        if not (only_sku and sku != only_sku) and sku not in excluded
    ]
    for sku in selected_skus if "ep" in selected_suites else ():
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
                    backend_precisions = BACKEND_PRECISIONS[target]
                    # Off-path precisions are dropped unless the caller named the precision
                    # explicitly, so the default matrix carries only deployable configurations.
                    off_path = OFF_PATH_PRECISIONS.get(target, ())
                    supported = [
                        precision for precision in SWEEP["precisions"]
                        if precision in backend_precisions
                        and (precision not in off_path or precision in selected_precisions)
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
        # Normal-mode shard IDs are unchanged (no mode segment) so existing references
        # stay valid; a non-normal mode inserts a short slug (low-latency -> "ll").
        mode_segment = "" if mode == "normal" else f"-{_MODE_SLUG[mode]}"
        shards_by_sku.setdefault(sku, []).append(_shard(
            sku, f"{sku}-{target}{mode_segment}-{precision}-n{nodes}", target, cases, mode=mode,
        ))
    if "swap-blocks" in selected_suites:
        for sku in selected_skus:
            requested, shard = _swap_shard(sku, swap_profile)
            requested_cases += requested
            shards_by_sku.setdefault(sku, []).append(shard)
    if "kv-transfer" in selected_suites:
        for sku in selected_skus:
            requested, kv_shards = _kv_shards(sku, selected_precisions)
            requested_cases += requested
            if kv_shards:
                shards_by_sku.setdefault(sku, []).extend(kv_shards)
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
    parser.add_argument("--suites", default="ep",
                        help=f"comma-separated subset of {', '.join(SUITES)}")
    parser.add_argument("--swap-profile", default="smoke",
                        help="configs/swap_sweep.json profile for the swap-blocks suite")
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
        suites=args.suites,
        swap_profile=args.swap_profile,
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
