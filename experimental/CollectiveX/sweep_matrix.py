#!/usr/bin/env python3
"""CollectiveX — sweep matrix resolver (the `setup` job of collectivex-sweep.yml).

Resolves the requested suites into the GHA matrix of shards. A shard is one allocation that sweeps
many cases sharing (sku, backend, node count). Large shards are chunked. Each case is enriched with
model dims (hidden/topk/experts from workloads.yaml) + token ladder + canonical flag, so the in-
container shard loop (run_in_container.sh SHARD mode) needs no further config lookup.

Knobs: --backends sweeps every EP library in one matrix; --backend remaps the DeepEP matrix onto a
single other library (capability-filtered). Emits a JSON matrix for ``fromJSON`` in the workflow.

  python3 sweep_matrix.py --suites all --out matrix.json
  python3 sweep_matrix.py --suites all --backend uccl --max-cases 12 --out matrix.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))
import yaml  # noqa: E402
import generate_matrix as gm  # noqa: E402
import capability as cap  # noqa: E402
import ep_harness  # noqa: E402

EP_TIMING_PROFILE = (f"{ep_harness.TIMED_ITERS_PER_TRIAL}:"
                     f"{ep_harness.TRIALS_PER_POINT}:"
                     f"{ep_harness.WARMUP_ITERS_PER_TRIAL}")


def _dims(wl_cfg, name):
    for sec in ("synthetic", "model_derived"):
        m = (wl_cfg.get(sec) or {}).get(name)
        if m:
            return m.get("hidden"), m.get("topk"), m.get("experts", m.get("routed_experts"))
    return None, None, None


def _union_ladder(a, b):
    """Union two token-point ladders; '' means the harness phase-default FULL ladder (a superset
    of every suite's token_points), so union with '' is ''."""
    if a == "" or b == "":
        return ""
    return " ".join(map(str, sorted({int(x) for x in (a.split() + b.split())})))


def _ladder(suite_cfg, phase):
    if phase == "decode" and suite_cfg.get("token_points_decode"):
        return " ".join(map(str, suite_cfg["token_points_decode"]))
    if phase == "prefill" and suite_cfg.get("token_points_prefill"):
        return " ".join(map(str, suite_cfg["token_points_prefill"]))
    if suite_cfg.get("token_points"):
        return " ".join(map(str, suite_cfg["token_points"]))
    return ""


def _resolved_ladder(ladder, phase, backend, routing, platform):
    """Apply backend/platform limits after expansion without capping the portable reference."""
    if backend != "mori":
        return ladder
    if (platform == "mi355x" and phase == "prefill"
            and routing not in {"uniform", "balanced", "balanced-rank-local"}):
        return None
    defaults = ep_harness.DECODE_LADDER if phase == "decode" else ep_harness.PREFILL_LADDER
    points = [int(x) for x in ladder.split()] if ladder else list(defaults)
    capped = [point for point in points if point <= 512]
    return " ".join(map(str, capped)) if capped else None


def _case_id(sku, case):
    """Stable scheduled-case identity, including the scored token ladder."""
    payload = json.dumps({"sku": sku, **case}, sort_keys=True, separators=(",", ":"))
    return f"cxv1-{hashlib.sha256(payload.encode()).hexdigest()[:20]}"


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX sweep matrix resolver")
    ap.add_argument("--suites", default="all", help="'all' or comma-list of suite names")
    backend_names = ",".join(cap.SWEEP_BACKENDS)
    ap.add_argument("--backend", default="",
                    help=f"select exactly one EP backend ({backend_names})")
    ap.add_argument("--backends", default="",
                    help=f"combined matrix: 'all' or a comma-list ({backend_names}); "
                         "capability-filtered and overrides --backend")
    ap.add_argument("--only-sku", default="", help="restrict to one workflow sku value")
    ap.add_argument("--min-nodes", type=int, default=0,
                    help="keep only shards whose tray count (nodes, blank=1) is >= this; "
                         "e.g. 2 = rack-scale EP8 only (skip the single-tray EP4 cells)")
    ap.add_argument("--max-nodes", type=int, default=0,
                    help="keep only shards whose tray count (nodes, blank=1) is <= this; "
                         "e.g. 1 = single-tray EP4 only (skip the rack-scale EP8 cells)")
    ap.add_argument("--max-cases", type=int, default=128, help="chunk shards larger than this into sub-cells (128 = effectively no chunking for current suites; each shard's cases run consecutively in ONE allocation, amortizing runner/enroot/build startup)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    wl_cfg = yaml.safe_load(open(os.path.join(HERE, "configs", "workloads.yaml")))
    suites_cfg = yaml.safe_load(open(os.path.join(HERE, "configs", "suites.yaml")))["suites"]
    suite_names = list(suites_cfg) if a.suites == "all" else [s.strip() for s in a.suites.split(",")]

    # --backends "all"|comma-list emits every requested implementation in one matrix.
    all_backends = list(cap.SWEEP_BACKENDS)
    if a.backends:
        names = all_backends if a.backends == "all" else [x.strip() for x in a.backends.split(",") if x.strip()]
        unknown = sorted(set(names) - set(all_backends))
        if unknown:
            raise SystemExit(f"unknown --backends values {unknown}; have {all_backends}")
        targets = names
    else:
        target = a.backend or "deepep"
        if target not in all_backends:
            raise SystemExit(f"unknown --backend value {target!r}; have {all_backends}")
        targets = [target]

    # collect enriched cases, deduped globally (a config shared by several suites appears once)
    seen = {}
    shards: dict = {}
    for sname in suite_names:
        scfg = suites_cfg[sname]
        for c in gm.generate(sname)["cases"]:
            if int(c["samples_per_point"]) != ep_harness.TIMED_SAMPLES_PER_POINT:
                raise SystemExit(f"case from {sname} violates fixed-512-v1: {c['samples_per_point']}")
            if c.get("timing") != EP_TIMING_PROFILE:
                raise SystemExit(f"case from {sname} has timing={c.get('timing')!r}; "
                                 f"fixed-512-v1 requires {EP_TIMING_PROFILE}")
            if c.get("warmup_semantics") != ep_harness.WARMUP_SEMANTICS:
                raise SystemExit(f"case from {sname} has warmup_semantics="
                                 f"{c.get('warmup_semantics')!r}; expected "
                                 f"{ep_harness.WARMUP_SEMANTICS!r}")
            plat = c["platform"]
            beng0 = c["backend"]
            if beng0 not in ("deepep", "mori"):
                continue
            sku = plat
            if a.only_sku and sku != a.only_sku:
                continue
            phase = c["phase"]
            rmode = c["resource_mode"]
            lad = _ladder(scfg, phase)
            h, t, e = _dims(wl_cfg, c["workload"])
            # Derive physical topology from the public platform contract. Keep nodes explicit in
            # every matrix cell even though manual launchers default a blank value to one node.
            gpus_per_node = int(cap.PLATFORMS[plat]["gpus_per_node"])
            scale_up_domain = int(cap.PLATFORMS[plat]["scale_up_domain"])
            nodes = str(max(1, (int(c.get("ep") or gpus_per_node) + gpus_per_node - 1)
                            // gpus_per_node))
            # The base registry uses DeepEP to enumerate NVIDIA shapes and MoRI for AMD shapes.
            # Apply the requested backend filter here; the portable NCCL/RCCL reference spans both.
            if beng0 == "mori":
                case_targets = [name for name in targets if name in ("mori", "nccl-ep")]
            else:
                case_targets = [name for name in targets if name != "mori"]
            for beng in case_targets:
                ok, _r = cap.resolve(
                    plat, beng, mode=c["mode"], dtype=c["dtype"], contract=c["contract"],
                    combine_quant_mode=c.get("combine_quant_mode", "none"), routing=c["routing"],
                    eplb=bool(c.get("eplb")),
                    activation_profile=c.get("activation_profile", "normal"),
                )
                if not ok:
                    continue
                lad_i = _resolved_ladder(lad, phase, beng, c["routing"], plat)
                if lad_i is None:
                    continue
                case = {
                    "suite": c["suite"], "workload": c["workload"],
                    "required_publication": c.get("required_publication"),
                    "backend": beng, "mode": c["mode"],
                    "dtype": c["dtype"], "contract": c["contract"], "routing": c["routing"],
                    "phase": phase, "ep": int(c["ep"]), "eplb": bool(c.get("eplb")),
                    "combine_quant_mode": c.get("combine_quant_mode", "none"),
                    "resource_mode": rmode,
                    "activation_profile": c.get("activation_profile", "normal"),
                    "placement": c.get("placement", "packed"),
                    "routing_step": str(c.get("routing_step", 0)),
                    "uneven_tokens": c.get("uneven_tokens", "none"),
                    "hidden": "" if h in (None, 7168) else str(h),
                    "topk": "" if t in (None, 8) else str(t),
                    "experts": "" if e in (None, 256) else str(e),
                    "samples_per_point": int(c["samples_per_point"]),
                    "warmup_semantics": c["warmup_semantics"], "ladder": lad_i,
                    "timing": c["timing"], "canonical": bool(c.get("canonical")), "nodes": nodes,
                    "gpus_per_node": gpus_per_node, "scale_up_domain": scale_up_domain,
                }
                case["case_id"] = _case_id(sku, case)
                sig = (
                    sku, case["suite"], case["workload"], beng, c["mode"], c["dtype"],
                    c["contract"], c["routing"], phase, case["ep"], case["eplb"],
                    case["combine_quant_mode"], rmode, case["activation_profile"],
                    case["placement"], case["routing_step"], case["uneven_tokens"],
                    case["hidden"], case["topk"], case["experts"],
                    case["samples_per_point"], case["warmup_semantics"], nodes,
                    gpus_per_node, scale_up_domain, c["timing"],
                )
                if sig in seen:
                    seen[sig]["ladder"] = _union_ladder(seen[sig]["ladder"], lad_i)
                    continue
                seen[sig] = case
                # One allocation/build per (SKU, backend, tray count).
                key = (sku, beng, nodes)
                shards.setdefault(key, []).append(case)

    # Per-backend chunk size. Fast backends run a whole build group
    # in ONE allocation (max_cases, ~no chunking). flashinfer is SLOW (~3.2 min/case, heavy per-case MNNVL
    # workspace setup) and intermittently hits `CUDA error: unspecified launch failure` under rapid
    # back-to-back cases — so chunk it small: bounded, PARALLEL jobs, fewer successive setups per
    # allocation. UCCL is not chunked because its current promoted shard fits comfortably.
    SLOW_MAX_CASES = {"flashinfer": 12}   # 12 (not 16): flashinfer cases retry up to 3x for the intermittent
                                          # MNNVL-barrier deadlock, so smaller chunks keep a chunk within --time.
    include = []
    for (sku, beng, nodes), cases in sorted(shards.items()):
        if a.min_nodes and max(1, int(nodes or 1)) < a.min_nodes:
            continue   # --min-nodes: skip single-tray (EP4) shards, keep only rack-scale (EP8+)
        if a.max_nodes and max(1, int(nodes or 1)) > a.max_nodes:
            continue   # --max-nodes: skip rack-scale (EP8+) shards, keep only single-tray (EP4)
        mc = min(a.max_cases, SLOW_MAX_CASES.get(beng, a.max_cases))
        for ci in range(0, len(cases), mc):
            chunk = cases[ci:ci + mc]
            part = ci // mc
            sid = f"{sku}-{beng}" + (f"-n{nodes}" if nodes else "") + (f"-p{part}" if len(cases) > mc else "")
            include.append({
                "id": sid, "sku": sku, "backend": beng,
                "launcher": cap.PLATFORMS[sku]["launcher"],
                "gpus_per_node": cap.PLATFORMS[sku]["gpus_per_node"],
                "scale_up_domain": cap.PLATFORMS[sku]["scale_up_domain"],
                "nodes": nodes, "n": len(chunk), "cases": chunk,
            })

    n_cells = len(include)
    n_cases = sum(x["n"] for x in include)
    matrix = {"include": include}
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(matrix, fh)
    print(f"resolved {n_cells} shard-cells, {n_cases} cases "
          f"(suites={len(suite_names)} backends={a.backends or a.backend or 'deepep'})",
          file=sys.stderr)
    # stdout = the matrix JSON (for `$(...)` capture in the workflow)
    print(json.dumps(matrix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
