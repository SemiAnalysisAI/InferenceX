#!/usr/bin/env python3
"""CollectiveX — sweep matrix resolver (the `setup` job of collectivex-sweep.yml).

Resolves the requested suites into the GHA matrix of SHARDS. A shard = one allocation that sweeps
many cases sharing (sku, backend, mode, resource_mode) — generate_matrix's own grouping. Big shards
are CHUNKED so no single matrix cell exceeds the GHA 6h job budget. Each case is enriched with its
model dims (hidden/topk/experts from workloads.yaml) + token ladder + canonical flag, so the in-
container shard loop (run_in_container.sh SHARD mode) needs no further config lookup.

Knobs mirror _gha_suite.sh: --backend remaps the deepep matrix onto another EP library (capability-
filtered), --deepep-v2 threads kernel_gen=v2. Emits a JSON matrix object for `fromJSON` in the
workflow: {"include": [ {id, sku, backend, mode, resource, deepep_v2, n, cases:[...]}, ... ]}.

  python3 sweep_matrix.py --suites all --out matrix.json
  python3 sweep_matrix.py --suites all --backend uccl --max-cases 12 --out matrix.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))
import yaml  # noqa: E402
import generate_matrix as gm  # noqa: E402
import capability as cap  # noqa: E402

# platform key -> workflow `sku` input value (must match the workflow's sku choices + runner label)
SKU = {"h100": "h100-dgxc", "h200": "h200", "b300": "b300", "b200": "b200-dgxc",
       "mi355x": "mi355x", "gb300": "gb300", "gb200": "gb200"}


def _dims(wl_cfg, name):
    for sec in ("synthetic", "model_derived"):
        m = (wl_cfg.get(sec) or {}).get(name)
        if m:
            return m.get("hidden"), m.get("topk"), m.get("experts", m.get("routed_experts"))
    return None, None, None


def _ladder(suite_cfg, phase):
    if phase == "decode" and suite_cfg.get("token_points_decode"):
        return " ".join(map(str, suite_cfg["token_points_decode"]))
    if phase == "prefill" and suite_cfg.get("token_points_prefill"):
        return " ".join(map(str, suite_cfg["token_points_prefill"]))
    if suite_cfg.get("token_points"):
        return " ".join(map(str, suite_cfg["token_points"]))
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX sweep matrix resolver")
    ap.add_argument("--suites", default="all", help="'all' or comma-list of suite names")
    ap.add_argument("--backend", default="", help="remap deepep cases onto this EP lib (uccl/flashinfer/deepep-hybrid/nccl-ep)")
    ap.add_argument("--deepep-v2", action="store_true")
    ap.add_argument("--only-sku", default="", help="restrict to one workflow sku value")
    ap.add_argument("--max-cases", type=int, default=14, help="chunk shards larger than this into sub-cells")
    ap.add_argument("--out", default="")
    ap.add_argument("--slim", action="store_true",
                    help="emit matrix WITHOUT the per-cell cases list (fits the GHA output size cap); "
                         "cells re-resolve their own cases via --emit-shard")
    ap.add_argument("--emit-shard", default="",
                    help="write just this shard id's {cases:[...]} (the CX_SHARD_FILE for run_in_container)")
    ap.add_argument("--shard-out", default="results/.shard.json")
    a = ap.parse_args()

    wl_cfg = yaml.safe_load(open(os.path.join(HERE, "configs", "workloads.yaml")))
    suites_cfg = yaml.safe_load(open(os.path.join(HERE, "configs", "suites.yaml")))["suites"]
    suite_names = list(suites_cfg) if a.suites == "all" else [s.strip() for s in a.suites.split(",")]

    # collect enriched cases, deduped globally (a config shared by several suites appears once)
    seen = set()
    shards: dict = {}
    for sname in suite_names:
        scfg = suites_cfg[sname]
        for c in gm.generate(sname)["cases"]:
            plat = c["platform"]
            beng = c["backend"]
            if beng not in ("deepep", "mori"):
                continue
            if a.backend and beng == "deepep":
                beng = a.backend
            ok, _r = cap.resolve(plat, beng, mode=c["mode"], dtype=c["dtype"], contract=c["contract"],
                                 routing=c["routing"], eplb=bool(c.get("eplb")),
                                 activation_profile=c.get("activation_profile", "normal"))
            if not ok:
                continue
            sku = SKU.get(plat, plat)
            if a.only_sku and sku != a.only_sku:
                continue
            phase = c["phase"]
            rmode = c["resource_mode"]
            lad = _ladder(scfg, phase)
            h, t, e = _dims(wl_cfg, c["workload"])
            # MoRI envelope guard (mirrors _gha_suite.sh): decode-only, capped ladder, tuned.
            if sku == "mi355x":
                if phase == "prefill":
                    continue
                lad, rmode = "1 2 4 8 16", "tuned"
            # rack-scale tray->nodes (gb200/gb300 = 4 GPU/tray): EP8 = 2 trays. Recorded for the cell.
            nodes = ""
            if plat in ("gb200", "gb300"):
                nd = max(1, int(c.get("ep") or 8) // 4)
                if nd > 1:
                    nodes = str(nd)
            canonical = (c.get("uneven_tokens", "none") == "none" and int(c.get("routing_step", 0)) == 0)
            case = {
                "backend": beng, "mode": c["mode"], "dtype": c["dtype"], "contract": c["contract"],
                "routing": c["routing"], "phase": phase, "eplb": bool(c.get("eplb")),
                "resource_mode": rmode, "activation_profile": c.get("activation_profile", "normal"),
                "placement": c.get("placement", "packed"), "routing_step": str(c.get("routing_step", 0)),
                "uneven_tokens": c.get("uneven_tokens", "none"),
                "hidden": "" if h in (None, 7168) else str(h),
                "topk": "" if t in (None, 8) else str(t),
                "experts": "" if e in (None, 256) else str(e),
                "ladder": lad, "canonical": canonical, "nodes": nodes,
            }
            sig = (sku, beng, c["mode"], c["dtype"], c["contract"], c["routing"], phase,
                   case["eplb"], rmode, case["activation_profile"], case["placement"],
                   case["routing_step"], case["uneven_tokens"], case["hidden"], case["topk"],
                   case["experts"], nodes)
            if sig in seen:
                continue
            seen.add(sig)
            # shard key: same allocation reuse -> (sku, backend, mode, resource, nodes)
            key = (sku, beng, c["mode"], rmode, nodes)
            shards.setdefault(key, []).append(case)

    # build matrix include, chunking oversized shards
    include = []
    for (sku, beng, mode, rmode, nodes), cases in sorted(shards.items()):
        for ci in range(0, len(cases), a.max_cases):
            chunk = cases[ci:ci + a.max_cases]
            part = ci // a.max_cases
            sid = f"{sku}-{beng}-{mode}-{rmode}" + (f"-n{nodes}" if nodes else "") + (f"-p{part}" if len(cases) > a.max_cases else "")
            include.append({
                "id": sid, "sku": sku, "backend": beng, "mode": mode, "resource_mode": rmode,
                "nodes": nodes, "deepep_v2": bool(a.deepep_v2 and beng == "deepep"),
                "n": len(chunk), "cases": chunk,
            })

    # --emit-shard: write just one shard's cases (the per-cell CX_SHARD_FILE) and exit.
    if a.emit_shard:
        match = next((x for x in include if x["id"] == a.emit_shard), None)
        if match is None:
            print(f"ERROR: shard id '{a.emit_shard}' not found among {len(include)} cells", file=sys.stderr)
            return 2
        os.makedirs(os.path.dirname(a.shard_out) or ".", exist_ok=True)
        with open(a.shard_out, "w") as fh:
            json.dump({"id": match["id"], "sku": match["sku"], "backend": match["backend"],
                       "nodes": match["nodes"], "deepep_v2": match["deepep_v2"],
                       "cases": match["cases"]}, fh)
        print(f"wrote shard {a.emit_shard} ({match['n']} cases) -> {a.shard_out}", file=sys.stderr)
        return 0

    n_cells = len(include)
    n_cases = sum(x["n"] for x in include)
    # slim: drop the heavy `cases` from each cell so the matrix fits the GHA job-output size cap;
    # each cell re-derives its cases with --emit-shard <id>.
    out_include = ([{k: v for k, v in x.items() if k != "cases"} for x in include]
                   if a.slim else include)
    matrix = {"include": out_include}
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(matrix, fh)
    print(f"resolved {n_cells} shard-cells, {n_cases} cases "
          f"(suites={len(suite_names)} backend-override={a.backend or 'deepep'} v2={a.deepep_v2})",
          file=sys.stderr)
    # stdout = the matrix JSON (for `$(...)` capture in the workflow)
    print(json.dumps(matrix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
