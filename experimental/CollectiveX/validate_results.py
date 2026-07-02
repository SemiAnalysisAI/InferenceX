#!/usr/bin/env python3
"""CollectiveX result validator (goal Part 1: schema + validation tooling).

Validates EP result JSON docs against ep-result-v4 and the project's semantic gates:
schema shape, provenance completeness, workload identity (incl. cross-run trace-signature
agreement within a comparison_key), measurement-contract membership, byte-contract presence,
sample counts, and — crucially — that `publication_status` is the MACHINE-DERIVED function of
`validity` (no doc may hand-label itself official). Exits non-zero when any doc claims
`official` but fails a gate (or, with --require-official, when any doc isn't official).

Pure stdlib; uses `jsonschema` if importable, else a built-in required-key/type/enum check.
v3 docs (no publication_status) load as legacy/experimental and are reported, not failed.

  python3 validate_results.py results/*.json
  python3 validate_results.py --require-official --schema schemas/ep-result-v4.schema.json results/
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

MIN_SAMPLES_OFFICIAL = 100
# Must stay in sync with the measurement_contract enum in schemas/ep-result-v4.schema.json
# (mori-quant-combine-v1 is reserved for the MoRI PR311 quant-combine axis; no emitter yet).
KNOWN_CONTRACTS = {"layout-and-dispatch-v1", "cached-layout-comm-only-v1", "runtime-visible-v1",
                   "mori-quant-combine-v1"}
PUB_STATES = {"official", "comparable-experimental", "diagnostic", "invalid", "failed"}


def derive_publication_status(v: dict) -> str:
    """MUST mirror ep_harness._derive_publication_status — the validator's job is to confirm the
    recorded status equals this derivation."""
    if v.get("execution_status") != "complete":
        return "failed"
    if (v.get("semantic_correctness") != "pass" or v.get("measurement_conformance") != "conformant"
            or v.get("workload_identity") == "inconsistent"):
        return "invalid"
    sound = (v.get("semantic_correctness") == "pass"
             and str(v.get("workload_identity", "")).startswith("consistent")
             and v.get("measurement_conformance") == "conformant")
    if str(v.get("resource_conformance", "")).endswith("nonconforming"):
        return "diagnostic"
    # contract-level anomaly (goal P1-e/f): demotes to diagnostic unless waived (anomaly_free).
    if not v.get("anomaly_free", True):
        return "diagnostic"
    if sound and v.get("provenance_complete") and v.get("workload_source") == "canonical-serialized":
        return "official"
    if sound:
        return "comparable-experimental"
    return "diagnostic"


def _schema_check(doc, schema):
    """jsonschema if available; else a pragmatic required-keys/enum check of the top level + rows."""
    try:
        import jsonschema
        jsonschema.validate(doc, schema)
        return []
    except ImportError:
        errs = []
        for k in schema.get("required", []):
            if k not in doc:
                errs.append(f"missing required field '{k}'")
        # enum spot-checks the built-in path can do cheaply
        ms = doc.get("measurement_contract")
        if ms is not None and ms not in KNOWN_CONTRACTS:
            errs.append(f"unknown measurement_contract '{ms}'")
        ps = doc.get("publication_status")
        if ps is not None and ps not in PUB_STATES:
            errs.append(f"unknown publication_status '{ps}'")
        if not doc.get("rows"):
            errs.append("no rows")
        return errs
    except Exception as exc:   # jsonschema.ValidationError
        return [f"schema: {exc.message if hasattr(exc, 'message') else exc}"]


def validate_doc(doc, schema, path):
    errs, warns = [], []
    legacy = "publication_status" not in doc
    if legacy:
        warns.append("legacy (v3, no publication_status) — loads as experimental, not comparable as official")
        return errs, warns, "legacy-experimental"
    if doc.get("record_type") == "failed-case":
        # Intentionally preserved failure skeleton (judge-by-data doctrine): validate the
        # skeleton contract only — the full-sweep gates below do not apply.
        if doc.get("publication_status") != "failed":
            errs.append(f"failed-case record with publication_status '{doc.get('publication_status')}' (must be 'failed')")
        if doc.get("rows"):
            errs.append("failed-case record must have empty rows")
        fail = doc.get("failure") or {}
        if not fail.get("failure_mode") or "return_code" not in fail:
            errs.append("failed-case record missing failure evidence (failure_mode/return_code)")
        return errs, warns, "failed"
    errs += _schema_check(doc, schema) if schema else []
    v = doc.get("validity", {})
    recorded = doc.get("publication_status")
    derived = derive_publication_status(v)
    if recorded != derived:
        errs.append(f"publication_status '{recorded}' != machine-derived '{derived}' (validity tampered or stale)")
    # byte + contract + sample gates
    if doc.get("measurement_contract") not in KNOWN_CONTRACTS:
        errs.append(f"unknown measurement_contract {doc.get('measurement_contract')}")
    rows = doc.get("rows", [])
    for r in rows:
        if "byte_contracts" not in r:
            errs.append(f"T={r.get('tokens_per_rank')}: missing byte_contracts"); break
        for op in ("dispatch", "combine", "roundtrip"):
            if op not in r or "p99" not in r.get(op, {}):
                errs.append(f"T={r.get('tokens_per_rank')}: missing {op} percentiles"); break
    # anomaly self-consistency (goal P1-e): validity.anomaly_free must equal (no anomalies or waived).
    anoms = doc.get("anomalies") or []
    waived = (doc.get("anomaly_summary") or {}).get("waived", False)
    expect_anomaly_free = (len(anoms) == 0) or bool(waived)
    if v.get("anomaly_free", True) != expect_anomaly_free:
        errs.append(f"validity.anomaly_free={v.get('anomaly_free')} but {len(anoms)} anomalies "
                    f"(waived={waived}) imply {expect_anomaly_free}")
    if anoms and not waived and recorded not in ("diagnostic", "invalid", "failed"):
        errs.append(f"{len(anoms)} unwaived timing anomaly(ies) but status={recorded} (must be diagnostic)")
    # official-grade gates
    if recorded == "official":
        if not v.get("provenance_complete"):
            errs.append("official but provenance_complete=false")
        if v.get("workload_source") != "canonical-serialized":
            errs.append("official but workload not canonical-serialized")
        # goal P1: official requires NON-NULL workload identity (id + signature).
        wl = doc.get("workload") or {}
        if not wl.get("workload_id"):
            errs.append("official but workload_id is null (non-null workload identity required)")
        if not wl.get("trace_signature"):
            errs.append("official but trace_signature is null")
        if anoms and not waived:
            errs.append("official but has unwaived timing anomalies")
        if rows and min((r.get("samples_pooled", 0) for r in rows)) < MIN_SAMPLES_OFFICIAL:
            errs.append(f"official but a point has <{MIN_SAMPLES_OFFICIAL} pooled samples")
        if not all(r.get("correct") for r in rows):
            errs.append("official but a point failed correctness")
    return errs, warns, recorded


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX EP result validator")
    ap.add_argument("paths", nargs="+", help="result JSON files or dirs")
    ap.add_argument("--schema", default=os.path.join(os.path.dirname(__file__), "schemas", "ep-result-v4.schema.json"))
    ap.add_argument("--require-official", action="store_true",
                    help="fail if any non-legacy doc is not 'official'")
    ap.add_argument("--regression", action="store_true",
                    help="also run threshold-based performance-regression detection (regression.py) "
                         "over the same files and fail if any hard regression (outside run-to-run "
                         "noise) is found, so one CI step gates on validity AND performance")
    ap.add_argument("--regression-metric", default="roundtrip", help="regression op (default roundtrip)")
    ap.add_argument("--regression-pct", default="p99", help="regression percentile (default p99)")
    ap.add_argument("--regression-threshold", type=float, default=0.10,
                    help="regression fractional threshold (default 0.10)")
    a = ap.parse_args()
    schema = None
    if a.schema and os.path.exists(a.schema):
        schema = json.load(open(a.schema))
    files = []
    for p in a.paths:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "**", "*.json"), recursive=True)
        else:
            files.append(p)
    files = sorted(f for f in files if not os.path.basename(f).startswith("env_"))

    # cross-run workload identity: within a comparison_key, the realized routing must be the SAME
    # workload. We check PER-TOKEN routing_hash agreement (not the whole trace_signature) so two
    # runs of the same config at DIFFERENT ladders (e.g. a capped cross-vendor sweep 1..16 vs a full
    # 1..128 headline) are NOT falsely flagged — only a genuine conflict (same T, different routing
    # bytes) is a different workload.
    by_ck = {}   # ck -> {T: {routing_hash: [files]}}
    bad = 0
    for f in files:
        try:
            doc = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        if doc.get("family") != "moe":
            continue
        # preserved failed-case record (goal immediate P2): a classified failure (run_in_container
        # emitted it on a wedge/timeout/crash). Report it as a preserved case, NOT a validation error.
        if doc.get("record_type") == "failed-case":
            fm = (doc.get("failure") or {}).get("failure_mode", "?")
            print(f"[FAILED-CASE] {os.path.basename(f):68s} mode={fm}  (preserved, not a validation error)")
            continue
        errs, warns, status = validate_doc(doc, schema, f)
        ck = doc.get("comparison_key")
        # routing_step (temporal) + uneven_tokens change the realized workload but are NOT in the
        # comparison_key (they live in reproduction) — include them in the cross-run grouping so a
        # moving-hotspot step / uneven-allocation variant isn't falsely flagged as a conflicting
        # same-config workload.
        repro = doc.get("reproduction") or {}
        gk = (ck, repro.get("routing_step", 0), repro.get("uneven_tokens", "none")) if ck else None
        if gk:
            for r in doc.get("rows", []):
                T, rh = r.get("tokens_per_rank"), r.get("routing_hash")
                if T is not None and rh:
                    by_ck.setdefault(gk, {}).setdefault(T, {}).setdefault(rh, []).append(os.path.basename(f))
        tag = "OK" if not errs else "FAIL"
        if errs:
            bad += 1
        if a.require_official and status not in ("official",) and not errs:
            tag = "FAIL"; bad += 1; errs = [f"not official (status={status})"]
        print(f"[{tag}] {os.path.basename(f):70s} status={status}")
        for e in errs:
            print(f"        ERROR: {e}")
        for w in warns:
            print(f"        note: {w}")
    # report cross-run identity CONFLICTS: same comparison_key + same T but DIFFERENT routing bytes
    # (a genuine "not the same workload" — different hardware ran different routing for one point).
    for gk, perT in by_ck.items():
        ck = gk[0]
        conflicts = {T: hs for T, hs in perT.items() if len(hs) > 1}
        if conflicts:
            bad += 1
            print(f"[FAIL] comparison_key {ck[:12]} (step={gk[1]},uneven={gk[2]}): per-T routing-hash CONFLICT — not the same workload:")
            for T, hs in sorted(conflicts.items()):
                print(f"        T={T}: " + "; ".join(f"{h[:10]}=[{', '.join(fs)}]" for h, fs in hs.items()))
    print(f"\n{'FAILED' if bad else 'PASS'}: {len(files)} files, {bad} problem(s)")

    # Optional performance-regression gate (goal P1 "Add regression thresholds"). Imported lazily so
    # validation carries no new dependency/behavior unless --regression is passed. A hard regression
    # (a >threshold slowdown outside this point's run-to-run noise) folds into the non-zero exit.
    if a.regression:
        import regression as _reg
        rep = _reg.analyze(a.paths, metric=a.regression_metric, pct=a.regression_pct,
                           threshold=a.regression_threshold)
        print()
        print(_reg.to_markdown(rep))
        if rep["hard_regressions"]:
            bad += rep["hard_regressions"]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
