#!/usr/bin/env python3
"""CollectiveX — sweep matrix resolver (the `setup` job of collectivex-sweep.yml).

Resolves the requested suites into the GHA matrix of SHARDS. A shard = one allocation that sweeps
many cases sharing (sku, backend, mode, resource_mode) — generate_matrix's own grouping. Big shards
are CHUNKED so no single matrix cell exceeds the GHA 6h job budget. Each case is enriched with its
model dims (hidden/topk/experts from workloads.yaml) + token ladder + canonical flag, so the in-
container shard loop (run_in_container.sh SHARD mode) needs no further config lookup.

Knobs: --backends sweeps every EP library in ONE matrix; --backend remaps the deepep matrix onto a
single other library (capability-filtered); --deepep-v2 threads kernel_gen=v2. Emits a JSON matrix for `fromJSON` in the
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
    ap.add_argument("--backend", default="", help="remap deepep cases onto ONE EP lib (uccl/flashinfer/deepep-hybrid/nccl-ep)")
    ap.add_argument("--backends", default="",
                    help="combined multi-backend matrix in ONE run: 'all' or a comma-list "
                         "(deepep,deepep-v2,uccl,flashinfer,deepep-hybrid,nccl-ep). Each deepep-origin "
                         "case is emitted once per backend (capability-filtered); mori stays AMD-native. "
                         "Supersedes per-backend dispatches. Overrides --backend/--deepep-v2 when set.")
    ap.add_argument("--deepep-v2", action="store_true")
    ap.add_argument("--only-sku", default="", help="restrict to one workflow sku value")
    ap.add_argument("--min-nodes", type=int, default=0,
                    help="keep only shards whose tray count (nodes, blank=1) is >= this; "
                         "e.g. 2 = rack-scale EP8 only (skip the single-tray EP4 cells)")
    ap.add_argument("--max-nodes", type=int, default=0,
                    help="keep only shards whose tray count (nodes, blank=1) is <= this; "
                         "e.g. 1 = single-tray EP4 only (skip the rack-scale EP8 cells)")
    ap.add_argument("--max-cases", type=int, default=128, help="chunk shards larger than this into sub-cells (128 = effectively no chunking for current suites; each shard's cases run consecutively in ONE allocation, amortizing runner/enroot/build startup)")
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

    # Backend expansion targets for a deepep-origin case, as (backend, deepep_v2) pairs:
    #  --backends "all"|comma-list -> COMBINED matrix (every backend in ONE run; supersedes the
    #    per-backend dispatches). 'deepep-v2' is the from-source V2 kernel = deepep + v2 flag.
    #  else -> the legacy single --backend (+ --deepep-v2) behavior.
    NV_EP_ALL = ["deepep", "deepep-v2", "uccl", "flashinfer", "deepep-hybrid", "nccl-ep"]
    if a.backends:
        names = NV_EP_ALL if a.backends == "all" else [x.strip() for x in a.backends.split(",") if x.strip()]
        targets = [("deepep", True) if n == "deepep-v2" else (n, False) for n in names]
    else:
        targets = [(a.backend or "deepep", a.deepep_v2)]

    # collect enriched cases, deduped globally (a config shared by several suites appears once)
    seen = set()
    shards: dict = {}
    for sname in suite_names:
        scfg = suites_cfg[sname]
        for c in gm.generate(sname)["cases"]:
            plat = c["platform"]
            beng0 = c["backend"]
            if beng0 not in ("deepep", "mori"):
                continue
            sku = SKU.get(plat, plat)
            if a.only_sku and sku != a.only_sku:
                continue
            phase = c["phase"]
            rmode = c["resource_mode"]
            lad = _ladder(scfg, phase)
            h, t, e = _dims(wl_cfg, c["workload"])
            # MoRI envelope guard: capped ladder (T=1..16) + tuned for BOTH phases. MoRI prefill IS
            # supported (MORI-EP does intra+inter-node, both modes — ROCm/mori); prefill at the capped
            # ladder is validated 5/5 (run 28461798511). It was an UNCAPPED ladder to T=128 that timed
            # out, not prefill itself — so prefill is capped here, NOT skipped (correcting an earlier
            # decode-only assumption).
            if sku == "mi355x":
                lad, rmode = "1 2 4 8 16", "tuned"
            # rack-scale tray->nodes (gb200/gb300 = 4 GPU/tray): EP4 = 1 tray, EP8 = 2 trays. ALWAYS
            # set an EXPLICIT count: the gb300 launcher does NODES="${CX_NODES:-2}", so an EMPTY
            # CX_NODES coerces to 2 (EP8) — an EP4 cell with nodes="" silently ran EP8 (the rack
            # multi-srun, which bypasses cx_build_deepep_v2 / cx_build_flashinfer_latest). nodes="1"
            # makes EP4 actually run EP4 (run_in_container, which builds V2/quant-combine).
            nodes = ""
            if plat in ("gb200", "gb300"):
                nodes = str(max(1, int(c.get("ep") or 8) // 4))
            # The broad sweep runs SEEDED-runtime (comparable-experimental), NOT pre-staged canonical:
            # a fixed seed + identical params already yields the same cross-SKU trace for a fair
            # comparison, without the per-case canonical-manifest staging (overhead + a fragility — the
            # official cohort is a separate targeted run). run_in_container also re-stages per case if
            # canonical is ever re-enabled (the CX_WORKLOAD_DIR unset fix).
            canonical = False
            # mori cases stay AMD-native; deepep-origin cases expand across the requested backend set.
            case_targets = [("mori", False)] if beng0 == "mori" else targets
            for (beng, v2) in case_targets:
                ok, _r = cap.resolve(plat, beng, mode=c["mode"], dtype=c["dtype"], contract=c["contract"],
                                     routing=c["routing"], eplb=bool(c.get("eplb")),
                                     activation_profile=c.get("activation_profile", "normal"))
                if not ok:
                    continue
                # DeepEP V2 (from-source kernel_gen=v2) is genuine on aarch64 gb200/gb300 at BOTH EP4
                # (single-tray, gb300 run 28429220764) AND EP8 rack (2-tray MNNVL, gb300 run 28434764062
                # -> kernel_gen=v2/ws8/correct). The EP8 rack path builds V2 once-per-node into a persistent
                # container (CX_BUILD_ONLY) and the harness passes allow_mnnvl=True (CX_ALLOW_MNNVL) so the
                # NVL buffer spans trays — so v2 is now allowed on gb200/gb300 at every EP degree.
                case = {
                    "backend": beng, "deepep_v2": v2, "mode": c["mode"], "dtype": c["dtype"],
                    "contract": c["contract"], "routing": c["routing"], "phase": phase,
                    "eplb": bool(c.get("eplb")), "resource_mode": rmode,
                    "activation_profile": c.get("activation_profile", "normal"),
                    "placement": c.get("placement", "packed"), "routing_step": str(c.get("routing_step", 0)),
                    "uneven_tokens": c.get("uneven_tokens", "none"),
                    "hidden": "" if h in (None, 7168) else str(h),
                    "topk": "" if t in (None, 8) else str(t),
                    "experts": "" if e in (None, 256) else str(e),
                    "ladder": lad, "canonical": canonical, "nodes": nodes,
                }
                sig = (sku, beng, v2, c["mode"], c["dtype"], c["contract"], c["routing"], phase,
                       case["eplb"], rmode, case["activation_profile"], case["placement"],
                       case["routing_step"], case["uneven_tokens"], case["hidden"], case["topk"],
                       case["experts"], nodes)
                if sig in seen:
                    continue
                seen.add(sig)
                # shard key = the CONTAINER/allocation-determining fields only: (sku, backend, v2, nodes).
                # mode + resource_mode are per-case runtime knobs (run_in_container reads CX_MODE/
                # CX_RESOURCE_MODE per case), so they do NOT split shards — all modes/rmodes of one
                # (sku,backend,v2,nodes) run consecutively in ONE allocation, paying the enroot import +
                # from-source build ONCE (not once per mode).
                key = (sku, beng, v2, nodes)
                shards.setdefault(key, []).append(case)

    # PER-BACKEND chunk size. Fast backends (deepep*/nccl-ep/mori/deepep-hybrid) run a whole build-group
    # in ONE allocation (max_cases, ~no chunking). flashinfer is SLOW (~3.2 min/case, heavy per-case MNNVL
    # workspace setup) and intermittently hits `CUDA error: unspecified launch failure` under rapid
    # back-to-back cases — so chunk it small: bounded, PARALLEL jobs, fewer successive setups per
    # allocation. (uccl is NOT chunked: it fit a 74-case allocation cleanly; its only misses were a few
    # ll-mode per-case timeouts that chunking wouldn't change.)
    SLOW_MAX_CASES = {"flashinfer": 12}   # 12 (not 16): flashinfer cases retry up to 3x for the intermittent
                                          # MNNVL-barrier deadlock, so smaller chunks keep a chunk within --time.
    include = []
    for (sku, beng, v2, nodes), cases in sorted(shards.items()):
        if a.min_nodes and max(1, int(nodes or 1)) < a.min_nodes:
            continue   # --min-nodes: skip single-tray (EP4) shards, keep only rack-scale (EP8+)
        if a.max_nodes and max(1, int(nodes or 1)) > a.max_nodes:
            continue   # --max-nodes: skip rack-scale (EP8+) shards, keep only single-tray (EP4)
        tag = beng + ("-v2" if v2 else "")   # distinct shard id/runner for the V2 kernel variant
        mc = min(a.max_cases, SLOW_MAX_CASES.get(beng, a.max_cases))
        for ci in range(0, len(cases), mc):
            chunk = cases[ci:ci + mc]
            part = ci // mc
            sid = f"{sku}-{tag}" + (f"-n{nodes}" if nodes else "") + (f"-p{part}" if len(cases) > mc else "")
            include.append({
                "id": sid, "sku": sku, "backend": beng,
                "nodes": nodes, "deepep_v2": v2,
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
          f"(suites={len(suite_names)} backends={a.backends or a.backend or 'deepep'} v2={a.deepep_v2})",
          file=sys.stderr)
    # stdout = the matrix JSON (for `$(...)` capture in the workflow)
    print(json.dumps(matrix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
