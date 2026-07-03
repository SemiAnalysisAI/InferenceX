#!/usr/bin/env python3
"""CollectiveX result validator (goal Part 1: schema + validation tooling).

Validates EP result JSON docs against their versioned schema (v4 historical, v5 current) and the
project's semantic gates:
schema shape, provenance completeness, workload identity (incl. cross-run trace-signature
agreement within a comparison_key), measurement-contract membership, byte-contract presence,
the fixed-512-v1 sample contract, and — crucially — that `publication_status` is the
MACHINE-DERIVED function of `validity` (no doc may hand-label itself official). Exits non-zero when any doc claims
`official` but fails a gate (or, with --require-official, when any doc isn't official).

Requires `jsonschema`; validation never falls back to a partial structural check.
v3 docs (no publication_status) load as legacy/experimental and are reported, not failed.

  python3 validate_results.py results/*.json
  python3 validate_results.py --require-official results/
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import jsonschema

import capability

SAMPLING_CONTRACT = "fixed-512-v1"
TIMED_SAMPLES_PER_POINT = 512
TIMED_ITERS_PER_TRIAL = 8
TRIALS_PER_POINT = 64
WARMUP_ITERS_PER_TRIAL = 32
WARMUP_SEMANTICS = "full-roundtrip-per-trial-point-v1"
HISTORICAL_V4_MIN_SAMPLES_OFFICIAL = 100
CURRENT_SCHEMA_VERSION = 5
HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATHS = {
    3: os.path.join(HERE, "schemas", "ep-result-v4.schema.json"),
    4: os.path.join(HERE, "schemas", "ep-result-v4.schema.json"),
    5: os.path.join(HERE, "schemas", "ep-result-v5.schema.json"),
}
# Must stay in sync with the measurement_contract enum in the versioned result schemas.
# (mori-quant-combine-v1 is reserved for the MoRI PR311 quant-combine axis; no emitter yet).
KNOWN_CONTRACTS = {"layout-and-dispatch-v1", "cached-layout-comm-only-v1", "runtime-visible-v1",
                   "mori-quant-combine-v1"}
PUB_STATES = {"official", "comparable-experimental", "diagnostic", "invalid", "failed"}
REQUIRED_BACKEND_PROVENANCE = {
    "deepep": ("deepep_version", "deepep_commit"),
    "deepep-hybrid": ("deepep_commit", "branch"),
    "flashinfer": ("flashinfer_version", "flashinfer_commit", "flashinfer_stack"),
    "uccl": ("uccl_version", "uccl_commit"),
    "mori": ("mori_commit",),
    "nccl-ep": ("nccl_version",),
}


def _resolved_provenance_value(field: str, value) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    if not text or text in {"unknown", "none", "null", "n/a", "?", "capture-failed"}:
        return False
    if "capture-failed" in text:
        return False
    if field.endswith("_commit"):
        if text in {"main", "hybrid-ep", "uccl", "pkg-uccl"}:
            return False
        if text.endswith(("-unknown", "-none", "-main", "-hybrid-ep")):
            return False
    return True


def backend_provenance_issues(doc: dict) -> list[str]:
    provenance = doc.get("backend_provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    return [field for field in REQUIRED_BACKEND_PROVENANCE.get(doc.get("backend"), ())
            if not _resolved_provenance_value(field, provenance.get(field))]


def _normalized_sku(value) -> str | None:
    value = str(value or "").lower()
    return next((sku for sku in sorted(capability.PLATFORMS, key=len, reverse=True)
                 if value == sku or value.startswith(f"{sku}-") or value.startswith(f"{sku}_")),
                None)


def topology_issues(doc: dict) -> list[str]:
    sku = _normalized_sku(doc.get("runner"))
    try:
        current = int(doc.get("schema_version") or 0) >= CURRENT_SCHEMA_VERSION
    except (TypeError, ValueError):
        current = False
    if not sku or not current:
        return []
    placement = doc.get("placement")
    if not isinstance(placement, dict):
        placement = {}
    issues = []
    for field in ("gpus_per_node", "scale_up_domain"):
        expected = int(capability.PLATFORMS[sku][field])
        if placement.get(field) != expected:
            issues.append(f"placement.{field}={placement.get(field)!r}, expected {expected} for {sku}")
    return issues


def derive_publication_status(v: dict, require_sampling: bool = True) -> str:
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
    if require_sampling and v.get("sampling_conformance") != "conformant":
        return "diagnostic"
    # contract-level anomaly (goal P1-e/f): demotes to diagnostic unless waived (anomaly_free).
    if not v.get("anomaly_free", True):
        return "diagnostic"
    if sound and v.get("provenance_complete") and v.get("workload_source") == "canonical-serialized":
        return "official"
    if sound:
        return "comparable-experimental"
    return "diagnostic"


def load_schema_registry() -> dict[int, dict]:
    """Load every supported EP schema keyed by the document's schema_version."""
    schemas, loaded = {}, {}
    for version, path in SCHEMA_PATHS.items():
        if path not in loaded:
            with open(path) as fh:
                loaded[path] = json.load(fh)
        schemas[version] = loaded[path]
    return schemas


def _schema_for_doc(doc: dict, schema_or_registry) -> tuple[dict | None, list[str]]:
    if schema_or_registry is None:
        return None, []
    # Backward-compatible programmatic/CLI override: a raw JSON schema applies to every input doc.
    if "$schema" in schema_or_registry:
        return schema_or_registry, []
    version = doc.get("schema_version")
    schema = schema_or_registry.get(version)
    if schema is None:
        return None, [f"unsupported schema_version {version!r}; supported={sorted(schema_or_registry)}"]
    return schema, []


def _schema_check(doc, schema):
    """Validate a document with the required JSON Schema implementation."""
    try:
        jsonschema.validate(doc, schema)
        return []
    except jsonschema.ValidationError as exc:
        return [f"schema: {exc.message}"]
    except jsonschema.SchemaError as exc:
        return [f"invalid schema: {exc.message}"]


def _sampling_contract_issues(doc: dict) -> list[str]:
    """Verify the fixed sample basis from configuration through stored histograms."""
    issues = []
    repro = doc.get("reproduction") or {}
    if repro.get("sampling_contract") != SAMPLING_CONTRACT:
        issues.append(f"sampling_contract must be '{SAMPLING_CONTRACT}'")
    iters, trials, warmup = repro.get("iters"), repro.get("trials"), repro.get("warmup")
    expected = (TIMED_ITERS_PER_TRIAL, TRIALS_PER_POINT, WARMUP_ITERS_PER_TRIAL)
    if (iters, trials, warmup) != expected:
        issues.append(f"iters:trials:warmup={iters}:{trials}:{warmup}, expected "
                      f"{expected[0]}:{expected[1]}:{expected[2]}")
    if repro.get("warmup_semantics") != WARMUP_SEMANTICS:
        issues.append(f"warmup_semantics must be '{WARMUP_SEMANTICS}'")
    if repro.get("samples_per_point") != TIMED_SAMPLES_PER_POINT:
        issues.append(f"reproduction.samples_per_point must equal {TIMED_SAMPLES_PER_POINT}")
    for row in doc.get("rows", []):
        t = row.get("tokens_per_rank")
        if row.get("samples_pooled") != TIMED_SAMPLES_PER_POINT:
            issues.append(f"T={t}: samples_pooled={row.get('samples_pooled')}, "
                          f"expected {TIMED_SAMPLES_PER_POINT}")
        if isinstance(trials, int) and row.get("trials") != trials:
            issues.append(f"T={t}: row trials={row.get('trials')}, reproduction trials={trials}")
        raw = row.get("raw_samples") or {}
        for op in ("dispatch", "combine", "roundtrip"):
            hist = raw.get(op) or {}
            if hist.get("n") != TIMED_SAMPLES_PER_POINT:
                issues.append(f"T={t}: raw_samples.{op}.n={hist.get('n')}, "
                              f"expected {TIMED_SAMPLES_PER_POINT}")
            counts = hist.get("counts")
            if not isinstance(counts, list):
                issues.append(f"T={t}: raw_samples.{op}.counts is missing")
            elif sum(counts) != TIMED_SAMPLES_PER_POINT:
                issues.append(f"T={t}: raw_samples.{op}.counts sum to {sum(counts)}, "
                              f"expected {TIMED_SAMPLES_PER_POINT}")
    return issues


def validate_doc(doc, schema, path):
    errs, warns = [], []
    legacy = "publication_status" not in doc
    try:
        declared_version = int(doc.get("schema_version") or 0)
    except (TypeError, ValueError):
        declared_version = 0
    if legacy and declared_version <= 3:
        warns.append("legacy (v3, no publication_status) — loads as experimental, not comparable as official")
        return errs, warns, "legacy-experimental"
    selected_schema, schema_errors = _schema_for_doc(doc, schema)
    errs += schema_errors
    errs += _schema_check(doc, selected_schema) if selected_schema else []
    scheduled = bool(doc.get("suite") or doc.get("required_publication"))
    if scheduled:
        for field in ("case_id", "suite", "workload_name", "required_publication", "phase",
                      "ep_size", "mode", "measurement_contract"):
            if doc.get(field) in (None, ""):
                errs.append(f"scheduled result missing {field}")
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
        if scheduled:
            case = fail.get("case") or {}
            for field in ("case_id", "suite", "workload", "required_publication", "backend",
                          "phase", "ep", "dispatch_dtype", "mode", "contract", "routing",
                          "eplb", "combine_quant_mode", "resource_mode", "tokens_ladder",
                          "gpus_per_node", "scale_up_domain",
                          "sampling_contract", "samples_per_point", "iters", "trials", "warmup",
                          "warmup_semantics"):
                if field not in case or (field != "tokens_ladder" and case[field] in (None, "")):
                    errs.append(f"scheduled failed-case missing failure.case.{field}")
        return errs, warns, "failed"
    v = doc.get("validity", {})
    recorded = doc.get("publication_status")
    schema_version = declared_version
    require_sampling = schema_version >= CURRENT_SCHEMA_VERSION
    sampling_issues = _sampling_contract_issues(doc) if require_sampling else []
    if require_sampling:
        observed_sampling = "conformant" if not sampling_issues else "nonconformant"
        recorded_sampling = v.get("sampling_conformance")
        if recorded_sampling != observed_sampling:
            errs.append(f"validity.sampling_conformance={recorded_sampling!r}, but artifact is "
                        f"{observed_sampling} under {SAMPLING_CONTRACT}")
    provenance_issues = backend_provenance_issues(doc)
    if v.get("provenance_complete") and provenance_issues:
        errs.append("validity.provenance_complete=true with unresolved backend identity: "
                    + ", ".join(provenance_issues))
    errs.extend(topology_issues(doc))
    derived = derive_publication_status(v, require_sampling=require_sampling)
    if recorded != derived:
        errs.append(f"publication_status '{recorded}' != machine-derived '{derived}' (validity tampered or stale)")
    # byte + contract + sample gates
    if doc.get("measurement_contract") not in KNOWN_CONTRACTS:
        errs.append(f"unknown measurement_contract {doc.get('measurement_contract')}")
    rows = doc.get("rows", [])
    for r in rows:
        if "byte_contracts" not in r:
            errs.append(f"T={r.get('tokens_per_rank')}: missing byte_contracts")
            break
        for op in ("dispatch", "combine", "roundtrip"):
            if op not in r or "p99" not in r.get(op, {}):
                errs.append(f"T={r.get('tokens_per_rank')}: missing {op} percentiles")
                break
    # anomaly self-consistency (goal P1-e): validity.anomaly_free must equal (no anomalies or waived).
    anoms = doc.get("anomalies") or []
    waived = (doc.get("anomaly_summary") or {}).get("waived", False)
    expect_anomaly_free = (len(anoms) == 0) or bool(waived)
    if v.get("anomaly_free", True) != expect_anomaly_free:
        errs.append(f"validity.anomaly_free={v.get('anomaly_free')} but {len(anoms)} anomalies "
                    f"(waived={waived}) imply {expect_anomaly_free}")
    if anoms and not waived and recorded not in ("diagnostic", "invalid", "failed"):
        errs.append(f"{len(anoms)} unwaived timing anomaly(ies) but status={recorded} (must be diagnostic)")
    if sampling_issues:
        if recorded in ("official", "comparable-experimental"):
            errs.extend(f"comparison-grade sampling violation: {issue}" for issue in sampling_issues)
        else:
            warns.extend(f"sampling diagnostic: {issue}" for issue in sampling_issues)
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
        if require_sampling:
            if rows and any(r.get("samples_pooled") != TIMED_SAMPLES_PER_POINT for r in rows):
                errs.append(f"official but a point does not have exactly {TIMED_SAMPLES_PER_POINT} pooled samples")
        elif rows and min((r.get("samples_pooled", 0) for r in rows)) < HISTORICAL_V4_MIN_SAMPLES_OFFICIAL:
            errs.append(f"v4 official but a point has <{HISTORICAL_V4_MIN_SAMPLES_OFFICIAL} pooled samples")
        if not all(r.get("correct") for r in rows):
            errs.append("official but a point failed correctness")
    return errs, warns, recorded


def cross_document_workload_issues(docs: list[dict]) -> list[str]:
    """Find canonical same-workload cells whose realized per-T identity differs."""
    observed: dict[tuple, dict[int, set[tuple]]] = {}
    for doc in docs:
        if doc.get("family") != "moe" or doc.get("record_type") == "failed-case":
            continue
        workload = doc.get("workload") or {}
        if workload.get("source") != "canonical-serialized":
            continue
        shape = doc.get("shape") or {}
        reproduction = doc.get("reproduction") or {}
        eplb = doc.get("eplb") or {}
        key = (
            doc.get("suite"), doc.get("workload_name"), doc.get("phase"), doc.get("ep_size"),
            shape.get("hidden"), shape.get("topk"), shape.get("experts"),
            shape.get("dispatch_dtype"), shape.get("routing"), bool(eplb.get("enabled")),
            reproduction.get("routing_step", 0), reproduction.get("uneven_tokens", "none"),
            shape.get("activation_profile", "normal"),
        )
        activation_identity = workload.get("activation_identity")
        mapping_hash = eplb.get("mapping_hash") if eplb.get("enabled") else None
        for row in doc.get("rows", []):
            tokens, routing_hash = row.get("tokens_per_rank"), row.get("routing_hash")
            if tokens is None or not routing_hash:
                continue
            identity = (str(routing_hash), activation_identity, mapping_hash)
            observed.setdefault(key, {}).setdefault(int(tokens), set()).add(identity)

    issues = []
    for key, per_token in observed.items():
        for tokens, identities in per_token.items():
            if len(identities) > 1:
                issues.append(
                    f"canonical workload identity conflict for suite={key[0]!r} "
                    f"workload={key[1]!r} phase={key[2]!r} ep={key[3]!r} T={tokens}"
                )
    return issues


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX EP result validator")
    ap.add_argument("paths", nargs="+", help="result JSON files or dirs")
    ap.add_argument("--schema", default="",
                    help="override with one schema for all docs; blank selects v3-v5 by schema_version")
    ap.add_argument("--require-official", action="store_true",
                    help="fail if any non-legacy doc is not 'official'")
    a = ap.parse_args()
    schema = json.load(open(a.schema)) if a.schema else load_schema_registry()
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
    validated_docs = []
    bad = 0
    for f in files:
        try:
            doc = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        if doc.get("family") != "moe":
            continue
        validated_docs.append(doc)
        errs, warns, status = validate_doc(doc, schema, f)
        # A well-formed failed-case is preserved evidence, not a benchmark validation failure. Its
        # versioned schema and failure fields are still validated before this reporting shortcut.
        if doc.get("record_type") == "failed-case":
            fm = (doc.get("failure") or {}).get("failure_mode", "?")
            if errs:
                bad += 1
                print(f"[FAIL] {os.path.basename(f):70s} status=failed")
                for e in errs:
                    print(f"        ERROR: {e}")
            else:
                print(f"[FAILED-CASE] {os.path.basename(f):68s} mode={fm}  (preserved, schema-valid evidence)")
            continue
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
            tag = "FAIL"
            bad += 1
            errs = [f"not official (status={status})"]
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
    for issue in cross_document_workload_issues(validated_docs):
        bad += 1
        print(f"[FAIL] {issue}")
    print(f"\n{'FAILED' if bad else 'PASS'}: {len(files)} files, {bad} problem(s)")

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
