#!/usr/bin/env python3
"""Keep the newest GOOD result per config; archive the rest (immediate cleanup: 'delete old runs').

After a full-suite re-run, results/ holds several runs of the same config across SHAs (the fresh
campaign + older campaigns + canonical-incompatible failures superseded by seeded re-runs). This
keeps ONE doc per config — the most recent that is not failed/invalid (prefer canonical-official) —
and moves the rest to _superseded/ (outside the results glob). Failed-case records whose config now
has a good result are archived too; a config that ONLY ever failed keeps its newest failed-case so
the failure is still preserved (goal P2).

config key = (sku, backend, dtype, mode, contract, routing+eplb, ep, phase, activation_profile,
              combine_quant_mode, uneven_tokens, routing_step) — i.e. everything but the SHA/run/ts.

  python3 launchers/_keep_newest.py            # archive superseded; keep newest-good per config
  python3 launchers/_keep_newest.py --dry      # report only
"""
import glob, json, os, sys, shutil

DRY = "--dry" in sys.argv
RES = "results"
ARCH = "_superseded"


def cfg_key(d):
    sh = d.get("shape") or {}
    q = sh.get("quant") or {}
    e = d.get("eplb") or {}
    rp = d.get("reproduction") or {}
    prof = d.get("resource_profile") or {}
    sku = (d.get("runner") or "?").split("_")[0].split("-")[0]
    # include the WORKLOAD DIMS (hidden/topk/experts) — model-derived workloads differ only here —
    # AND the RESOURCE axis (resource_mode + normalized comm-fraction): normalized@0.10 vs @0.35 vs
    # tuned are distinct operating points (the resource-Pareto ladder + the tuned official cohort);
    # omitting them would collapse the ladder and merge tuned with normalized.
    # trace_signature distinguishes the T-LADDER: re-runs of the same config+ladder share it
    # (dedup to newest), but a capped cross-vendor cohort run (T<=16) keeps its own identity vs the
    # full-ladder per-GPU run (T<=128) — so both survive (per-GPU completeness AND the matched cohort).
    wl = d.get("workload") or {}
    return (sku, d.get("backend"), sh.get("hidden"), sh.get("topk"), sh.get("experts"),
            sh.get("dispatch_dtype"), d.get("mode"), d.get("measurement_contract"),
            f"{sh.get('routing')}{'+eplb' if e.get('enabled') else ''}",
            d.get("ep_size"), d.get("phase"), sh.get("activation_profile", "normal"),
            q.get("combine_quant_mode", "none"),
            rp.get("uneven_tokens", "none"), rp.get("routing_step", 0),
            d.get("resource_mode"), prof.get("requested_fraction"), wl.get("trace_signature"))


def rank(d):
    """sort key: prefer NOT-failed, then official>comparable>diagnostic, then newest."""
    pub = d.get("publication_status") or "legacy"
    failed = (d.get("record_type") == "failed-case") or (d.get("status") == "failed") or not d.get("rows")
    order = {"official": 4, "comparable-experimental": 3, "diagnostic": 2, "legacy": 1,
             "invalid": 0, "failed": 0}.get(pub, 0)
    return (0 if failed else 1, order, d.get("generated_at") or "")


def main():
    docs = {}
    for f in glob.glob(os.path.join(RES, "*.json")):
        b = os.path.basename(f)
        if "deepep" not in b and "mori" not in b and not b.startswith("failed_"):
            continue
        try:
            d = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("family") != "moe":
            continue
        docs.setdefault(cfg_key(d), []).append((f, d))
    os.makedirs(ARCH, exist_ok=True)
    kept = moved = 0
    for k, lst in docs.items():
        lst.sort(key=lambda fd: rank(fd[1]), reverse=True)
        kept += 1                                  # keep lst[0] (best/newest)
        for f, d in lst[1:]:                       # archive the rest
            moved += 1
            if not DRY:
                shutil.move(f, os.path.join(ARCH, os.path.basename(f)))
    print(f"{'(dry) ' if DRY else ''}configs={len(docs)} kept={kept} archived={moved} -> {ARCH}/")


if __name__ == "__main__":
    main()
