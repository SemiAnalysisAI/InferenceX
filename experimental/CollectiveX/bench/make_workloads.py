#!/usr/bin/env python3
"""Generate canonical serialized workloads. Runs the stdlib counter generator for
each (routing, global_tokens) in a ladder and writes <workload_id>.npz + .manifest.json into a
dir that runs then consume via `run_ep.py --workload-dir`. One trace is emitted per global-token
count because global token count is part of workload identity.

  python3 bench/make_workloads.py --out-dir /path/to/cx_workloads \\
      --routing uniform --ep 8 --hidden 7168 --topk 8 --experts 256 --seed 67 \\
      --tokens-ladder "1 2 4 8 16 32 64 128 256 512"

Or by the named v1 workload in configs/workloads.yaml. Explicit dimension flags still override it:

  python3 bench/make_workloads.py --out-dir /path/to/cx_workloads --workload deepseek-v3-v1 --routing uniform --ep 8

--id-only prints the content-bound workload_id per ladder point without torch/numpy:

  python3 bench/make_workloads.py --workload deepseek-v3-v1 --ep 8 --id-only

Generate every routing the suites need by running once per --routing. Idempotent (same id => same
file). The dir is the cross-hardware artifact: copy it to each cluster so all consume identical bytes.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import workload as wl   # noqa: E402

# Repo root holds configs/ (this file is in tests/). Used only for --workload name resolution.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_manifest(name):
    """Look a workload name up in configs/workloads.yaml and return (hidden, topk, experts).
    Searches synthetic + model_derived; expert count = `experts` or (for model-derived) `routed_experts`.
    Raises SystemExit with the known names if the manifest is absent. Pure PyYAML + stdlib."""
    import yaml
    path = os.path.join(_REPO, "configs", "workloads.yaml")
    with open(path) as handle:
        cfg = yaml.safe_load(handle)
    known = []
    for section in ("synthetic", "model_derived"):
        sec = cfg.get(section) or {}
        known += list(sec)
        m = sec.get(name)
        if m is None:
            continue
        experts = m.get("experts", m.get("routed_experts"))
        if m.get("hidden") is None or m.get("topk") is None or experts is None:
            raise SystemExit(f"workload '{name}' is missing hidden/topk/experts in {path}")
        return int(m["hidden"]), int(m["topk"]), int(experts)
    raise SystemExit(f"unknown --workload '{name}'; known: {sorted(known)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate canonical CollectiveX workloads")
    ap.add_argument("--out-dir", help="required unless --id-only")
    ap.add_argument("--workload", help="named manifest in configs/workloads.yaml (sets hidden/topk/experts)")
    ap.add_argument("--routing", default="uniform", choices=["uniform"])
    ap.add_argument("--ep", type=int, required=True, help="ep_size (global_tokens = T * ep)")
    ap.add_argument("--hidden", type=int, help="override (default 7168, or the --workload's hidden)")
    ap.add_argument("--topk", type=int, help="override (default 8, or the --workload's topk)")
    ap.add_argument("--experts", type=int, help="override (default 256, or the --workload's experts)")
    ap.add_argument("--seed", type=int, default=67)
    ap.add_argument("--tokens-ladder", default="1 2 4 8 16 32 64 128 256 512")
    ap.add_argument("--id-only", action="store_true",
                    help="print content-bound workload_id per point without torch/numpy")
    a = ap.parse_args()

    # Resolve dims: a named --workload supplies defaults; explicit --hidden/--topk/--experts override
    # per field. With neither, fall back to the v1 DeepSeek dimensions (7168/8/256).
    base_h, base_t, base_e = (7168, 8, 256)
    if a.workload:
        base_h, base_t, base_e = resolve_manifest(a.workload)
    hidden = a.hidden if a.hidden is not None else base_h
    topk = a.topk if a.topk is not None else base_t
    experts = a.experts if a.experts is not None else base_e

    if not a.id_only and not a.out_dir:
        ap.error("--out-dir is required unless --id-only")

    raw_ladder = [int(token) for token in a.tokens_ladder.replace(",", " ").split()]
    if (a.ep <= 0 or min(hidden, topk, experts) <= 0 or topk > experts or experts % a.ep
            or not raw_ladder or any(token <= 0 for token in raw_ladder)
            or len(raw_ladder) != len(set(raw_ladder))):
        ap.error("shape, EP, and token ladder must be positive, divisible, and unique")
    ladder = sorted(raw_ladder)
    epr = experts // a.ep
    label = f"workload={a.workload} " if a.workload else ""

    if a.id_only:
        # The stdlib counter generator derives the same content-bound ID on every runtime.
        made = []
        for T in ladder:
            gt = T * a.ep
            wid = wl.compute_workload_id(a.routing, hidden, topk, experts, a.ep, gt, a.seed)
            made.append((T, gt, wid))
            print(f"  T={T:<5} gt={gt:<6} routing={a.routing} -> {wid}")
        print(f"{label}id-only: {len(made)} workload_id(s) "
              f"(hidden={hidden} topk={topk} experts={experts} ep={a.ep} routing={a.routing} seed={a.seed})")
        return 0

    os.makedirs(a.out_dir, exist_ok=True)
    made = []
    for T in ladder:
        gt = T * a.ep
        idx, w, man = wl.build_workload(hidden, topk, experts, a.routing, gt, a.seed, epr)
        wid = wl.save_workload(a.out_dir, idx, w, man)
        made.append((T, gt, wid))
        print(f"  T={T:<5} gt={gt:<6} routing={a.routing} -> {wid}  "
              f"(trace sha {man['checksums']['trace'][:12]})")
    print(f"{label}wrote {len(made)} canonical workloads to {a.out_dir} (routing={a.routing}, ep={a.ep})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
