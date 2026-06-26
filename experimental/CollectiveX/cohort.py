#!/usr/bin/env python3
"""CollectiveX publication-cohort builder + validator (goal Part 1: publication cohort manifests,
official-cohort validation, source-SHA pinning; goal Part 2: EPLB mapping identity).

A *publication cohort* is the set of result artifacts that are meant to be compared on ONE chart —
e.g. the same workload + measurement contract + config across SKUs/backends. Unlike `comparison_key`
(which gates a single curve and so INCLUDES topology/sku), a cohort deliberately lets sku / backend /
topology VARY (those are the independent variable) while requiring everything that must be identical
for the comparison to be fair to actually match:

    cohort_key = (mode, phase, ep_size, resource_mode, comparison_class, measurement_contract,
                  dispatch_dtype, activation_profile, combine_quant_mode, trace_signature)

For each cohort this tool emits a MANIFEST listing every member with its identity fingerprint
(source SHA, workload id, image digest, backend version, schema version) and decides whether the
cohort is OFFICIAL-eligible. A cohort is official only when every member is itself measurement-sound
and the dimensions that MUST match across hardware do:

  * one benchmark source SHA           (goal P1 "same benchmark source SHA"; --pin-sha enforces)
  * non-null + identical workload_id   (goal P1 "non-null workload identity")
  * identical trace_signature          (same realized routing bytes — by cohort_key construction)
  * identical EPLB mapping_hash         (goal P2 "matching EPLB mapping identity") when EPLB is on
  * no unresolved timing anomalies      (goal P1 anomaly gate)
  * complete provenance per member      (image digest + git run)

Rejected members are recorded WITH machine-readable reasons (goal P1 "store rejected artifacts with
explicit rejection reasons") rather than silently dropped.

  python3 cohort.py --results-dir results                      # summarize all cohorts
  python3 cohort.py --results-dir results --require-official    # exit 3 unless an official cohort exists
  python3 cohort.py --results-dir results --pin-sha --out results/cohorts.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

MIN_SAMPLES_OFFICIAL = 100


def _backend_version(doc: dict) -> str:
    p = doc.get("backend_provenance", {}) or {}
    return (p.get("deepep_commit") or p.get("deepep_version")
            or p.get("mori_commit") or "unknown")


def fingerprint(doc: dict, path: str) -> dict:
    """Per-artifact identity used to detect cohort mismatches + build the cohort id."""
    sh = doc.get("shape", {}) or {}
    q = sh.get("quant", {}) or {}
    wl = doc.get("workload", {}) or {}
    repro = doc.get("reproduction", {}) or {}
    gr = repro.get("git_run") or {}
    eplb = doc.get("eplb") or {}
    v = doc.get("validity", {}) or {}
    return {
        "file": os.path.basename(path),
        "sku": (doc.get("runner") or "?").split("_")[0].split("-")[0],
        "backend": doc.get("backend"), "mode": doc.get("mode"), "phase": doc.get("phase"),
        "ep_size": doc.get("ep_size"), "resource_mode": doc.get("resource_mode"),
        "comparison_class": doc.get("comparison_class"),
        "measurement_contract": doc.get("measurement_contract"),
        "dispatch_dtype": sh.get("dispatch_dtype"),
        "activation_profile": sh.get("activation_profile", "normal"),
        "combine_quant_mode": q.get("combine_quant_mode", "none"),
        "trace_signature": wl.get("trace_signature") or (doc.get("routing_identity") or {}).get("trace_signature"),
        "workload_id": wl.get("workload_id"),
        "workload_source": wl.get("source"),
        "source_sha": (gr.get("source_sha") or ""),
        "image_digest": (repro.get("image_digest") or ""),
        "backend_version": _backend_version(doc),
        "schema_version": doc.get("schema_version"),
        "publication_status": doc.get("publication_status") or "legacy",
        "anomaly_free": v.get("anomaly_free", True),
        "provenance_complete": v.get("provenance_complete", False),
        "eplb_enabled": bool(eplb.get("enabled")),
        "eplb_mapping_hash": eplb.get("mapping_hash"),
        "min_samples": min((r.get("samples_pooled", 0) for r in doc.get("rows", [])), default=0),
        "correct": all(r.get("correct") for r in doc.get("rows", [])) if doc.get("rows") else False,
    }


def cohort_key(fp: dict) -> tuple:
    """Identity a cohort's members must share. sku/backend/topology deliberately EXCLUDED — those
    are what a cross-hardware chart compares."""
    return (fp["mode"], fp["phase"], fp["ep_size"], fp["resource_mode"], fp["comparison_class"],
            fp["measurement_contract"], fp["dispatch_dtype"], fp["activation_profile"],
            fp["combine_quant_mode"], fp["trace_signature"])


def cohort_id(members: list) -> str:
    """Stable content hash of the cohort: encodes every member's (source SHA, workload id, image
    digest, backend version, schema version) — goal P1 'cohort IDs that encode ...'."""
    parts = sorted(f"{m['sku']}|{m['backend']}|{m['source_sha']}|{m['workload_id']}|"
                   f"{m['image_digest']}|{m['backend_version']}|{m['schema_version']}" for m in members)
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def evaluate_cohort(members: list, pin_sha: bool) -> dict:
    """Split members into the OFFICIAL subset (accepted) + the rest (rejected, with reasons).
    A non-canonical (wid=null / seeded-runtime) member is REJECTED from the official cohort but
    does NOT block it — that is the point of recording rejections. official_eligible then depends
    on the ACCEPTED subset being mutually consistent (one source SHA under --pin-sha, one workload_id,
    one EPLB mapping), NOT on there being zero rejected members. A seeded run of the same config
    shares the deterministic trace_signature, so it lands in the same cohort and is simply excluded."""
    rejected, accepted = [], []
    for m in members:
        reasons = []                                  # PER-MEMBER gates only
        # publication_status is machine-derived from ALL validity dims (correctness, workload
        # identity, measurement + RESOURCE conformance, provenance, anomalies). Only an 'official'
        # member belongs in an official cohort — this is the authoritative gate; the granular
        # checks below just enrich the rejection reason (e.g. a resource-nonconforming MoRI run is
        # 'diagnostic' and excluded here even though it is correct + canonical + provenance-complete).
        if m["publication_status"] != "official":
            reasons.append(f"publication_status={m['publication_status']} (official cohort needs 'official')")
        if not m["correct"]:
            reasons.append("a point failed correctness")
        if not m["anomaly_free"]:
            reasons.append("unresolved timing anomaly (not waived)")
        if not m["workload_id"]:
            reasons.append("workload_id is null (not canonical-serialized) — comparable-experimental, not official")
        if m["workload_source"] != "canonical-serialized":
            reasons.append(f"workload_source={m['workload_source']} (official needs canonical-serialized)")
        if not m["provenance_complete"]:
            reasons.append("provenance incomplete (image digest / git run missing)")
        if m["min_samples"] < MIN_SAMPLES_OFFICIAL:
            reasons.append(f"a point has <{MIN_SAMPLES_OFFICIAL} pooled samples")
        (rejected if reasons else accepted).append({**m, "rejection_reasons": reasons})
    # cross-member consistency over the ACCEPTED (would-be-official) subset.
    a_shas = {m["source_sha"] for m in accepted if m["source_sha"]}
    a_wids = {m["workload_id"] for m in accepted if m["workload_id"]}
    a_maps = {m["eplb_mapping_hash"] for m in accepted if m["eplb_enabled"]}
    a_eplb = any(m["eplb_enabled"] for m in accepted)
    incoherent = []
    if pin_sha and len(a_shas) > 1:
        incoherent.append(f"accepted members span {len(a_shas)} source SHAs (--pin-sha requires one)")
    if len(a_wids) > 1:
        incoherent.append(f"accepted members span {len(a_wids)} workload_ids")
    if a_eplb and len(a_maps) > 1:
        incoherent.append(f"accepted members span {len(a_maps)} EPLB mapping_hashes")
    official_eligible = len(accepted) >= 1 and not incoherent
    return {
        "cohort_id": cohort_id(members), "n_members": len(members),
        "skus": sorted({m["sku"] for m in members}),
        "official_skus": sorted({m["sku"] for m in accepted}),
        "backends": sorted({m["backend"] for m in members if m["backend"]}),
        "source_shas": sorted({m["source_sha"] for m in members if m["source_sha"]}),
        "workload_ids": sorted({m["workload_id"] for m in members if m["workload_id"]}),
        "official_source_shas": sorted(a_shas), "official_workload_ids": sorted(a_wids),
        "eplb_mapping_hashes": sorted(a_maps), "any_eplb": a_eplb,
        "official_eligible": official_eligible, "incoherent": incoherent,
        "accepted": accepted, "rejected": rejected,
    }


def build(results_dir: str, pin_sha: bool) -> dict:
    cohorts = {}
    for f in sorted(glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)):
        if os.path.basename(f).startswith("env_"):
            continue
        try:
            doc = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        if doc.get("family") != "moe" or not doc.get("rows"):
            continue
        if "publication_status" not in doc:
            continue                                   # legacy v3 — not cohort-eligible
        fp = fingerprint(doc, f)
        cohorts.setdefault(cohort_key(fp), []).append(fp)
    out = []
    for ck, members in cohorts.items():
        ev = evaluate_cohort(members, pin_sha)
        ev["key"] = {"mode": ck[0], "phase": ck[1], "ep_size": ck[2], "resource_mode": ck[3],
                     "comparison_class": ck[4], "measurement_contract": ck[5],
                     "dispatch_dtype": ck[6], "activation_profile": ck[7],
                     "combine_quant_mode": ck[8], "trace_signature": ck[9]}
        out.append(ev)
    out.sort(key=lambda c: (not c["official_eligible"], -c["n_members"]))
    return {"results_dir": results_dir, "pin_sha": pin_sha, "n_cohorts": len(out),
            "n_official_eligible": sum(1 for c in out if c["official_eligible"]),
            "cohorts": out}


def to_markdown(report: dict) -> str:
    h = (f"### Publication cohorts ({report['n_cohorts']} cohorts, "
         f"{report['n_official_eligible']} official-eligible; pin_sha={report['pin_sha']})\n\n"
         "| cohort | contract | dtype·act·cq | EP | SKUs | backends | members | official | top rejection |\n"
         "|---|---|---|---|---|---|---|---|---|\n")
    for c in report["cohorts"]:
        k = c["key"]
        cfg = f"{k['dispatch_dtype']}·{k['activation_profile']}·{k['combine_quant_mode']}"
        rej = ""
        if c["rejected"]:
            rs = c["rejected"][0]["rejection_reasons"]
            rej = (rs[0] if rs else "")[:48]
        h += (f"| `{c['cohort_id']}` | {(k['measurement_contract'] or '').replace('-v1','')} | {cfg} | "
              f"{k['ep_size']} | {','.join(c['skus'])} | {','.join(c['backends'])} | "
              f"{len(c['accepted'])}✓/{len(c['rejected'])}✗ | {'YES' if c['official_eligible'] else '—'} | {rej} |\n")
    return h


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX publication-cohort builder/validator")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--pin-sha", action="store_true",
                    help="require all members of an official cohort to share one source SHA")
    ap.add_argument("--require-official", action="store_true",
                    help="exit 3 unless at least one cohort is official-eligible")
    ap.add_argument("--out", help="write the full cohort manifest JSON here")
    a = ap.parse_args()
    report = build(a.results_dir, a.pin_sha)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(report, open(a.out, "w"), indent=2, sort_keys=True)
        print(f"wrote {a.out}")
    print(to_markdown(report))
    if a.require_official and report["n_official_eligible"] == 0:
        print("FAIL: no official-eligible cohort (see rejection reasons above)")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
