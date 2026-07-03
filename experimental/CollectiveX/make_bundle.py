#!/usr/bin/env python3
"""CollectiveX publication bundle generator (goal P1: continuous benchmark infrastructure).

Turns a validated aggregate into ONE self-contained, citable directory:

    bundle/
      manifest.json      bundle format, source run provenance, coverage + validation counts
      <aggregate>.ndjson the schema-validated dataset (copied verbatim)
      SHA256SUMS         checksums of every file above

Fail-loud doctrine: every doc in the aggregate is validated (versioned EP result schema +
validate_results semantic gates) BEFORE anything is written; any schema error or
publication_status tamper aborts the bundle with a non-zero exit. A bundle therefore
certifies its own dataset — nothing lands in it that the validator has not passed.

  python3 make_bundle.py --aggregate results/aggregate/collectivex_all_123.ndjson \
      --out-dir results/bundle --source-run-id 123 --source-sha abc --source-run-url https://...
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import validate_results as vr  # noqa: E402
import capability as cap  # noqa: E402
from artifact_safety import assert_publication_safe  # noqa: E402

BUNDLE_FORMAT = 1
PUBLICATION_RANK = {
    "failed": 0,
    "invalid": 0,
    "diagnostic": 1,
    "valid": 1,
    "comparable-experimental": 2,
    "official": 3,
}
PHASE_TOKEN_DEFAULTS = {
    "decode": (1, 2, 4, 8, 16, 32, 64, 128),
    "prefill": (128, 256, 512, 1024, 2048, 4096),
}
SKU_PREFIXES = (
    "h100-dgxc", "h200-dgxc", "b200-dgxc", "mi355x", "mi325x", "gb300", "gb200", "b300",
)


def _sku_of(doc: dict) -> str:
    """SKU token from the runner name: 'h100-dgxc-slurm_19' -> 'h100', 'gb300-8x' -> 'gb300'."""
    runner = str(doc.get("runner") or "unknown")
    return runner.split("_")[0].split("-")[0] or "unknown"


def _normalized_sku(value) -> str:
    """Map runner names and matrix labels onto the v1 scheduled SKU vocabulary."""
    value = str(value or "unknown").lower()
    return next((sku for sku in SKU_PREFIXES if value == sku
                 or value.startswith(f"{sku}-") or value.startswith(f"{sku}_")), "unknown")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_ndjson(path: str) -> list[dict]:
    docs = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"bundle: {path}:{i + 1} is not JSON ({exc}) — refusing to bundle")
            if not isinstance(value, dict):
                raise SystemExit(
                    f"bundle: {path}:{i + 1} is not a JSON object — refusing to bundle"
                )
            docs.append(value)
    return docs


def validate(docs: list[dict], schema: dict | None) -> dict:
    """Validate every EP doc and reject every other family."""
    assert_publication_safe(docs)
    by_status: dict[str, int] = {}
    by_family: dict[str, int] = {}
    n_err = 0
    for i, doc in enumerate(docs):
        fam = doc.get("family") or "unknown"
        by_family[fam] = by_family.get(fam, 0) + 1
        if fam != "moe":
            raise SystemExit(
                f"bundle: doc[{i}] has unsupported family {fam!r}; CollectiveX v1 is EP-only"
            )
        errs, _warns, status = vr.validate_doc(doc, schema, f"doc[{i}]")
        by_status[status] = by_status.get(status, 0) + 1
        for e in errs:
            n_err += 1
            print(f"bundle: INVALID doc[{i}] ({doc.get('backend')}/{doc.get('runner')}): {e}",
                  file=sys.stderr)
    if n_err:
        raise SystemExit(f"bundle: {n_err} validation error(s) — refusing to publish a tainted bundle")
    identity_issues = vr.cross_document_workload_issues(docs)
    if identity_issues:
        raise SystemExit(
            "bundle: cross-document workload identity failed: " + "; ".join(identity_issues[:8])
        )
    return {"by_publication_status": by_status, "by_family": by_family, "errors": 0}


def coverage(docs: list[dict]) -> dict:
    skus, backends, ws, contracts, versions = set(), set(), set(), set(), set()
    newest = ""
    for d in docs:
        skus.add(_sku_of(d))
        if d.get("backend"):
            backends.add(d["backend"])
        if d.get("world_size"):
            ws.add(int(d["world_size"]))
        if d.get("measurement_contract"):
            contracts.add(d["measurement_contract"])
        if d.get("schema_version") is not None:
            versions.add(int(d["schema_version"]))
        newest = max(newest, str(d.get("generated_at") or ""))
    return {"skus": sorted(skus), "backends": sorted(backends), "world_sizes": sorted(ws),
            "measurement_contracts": sorted(contracts), "schema_versions": sorted(versions),
            "newest_result_at": newest or None}


def _tokens(value, phase: str) -> tuple[int, ...]:
    """Normalize a matrix/result ladder; blank means the v1 default for that phase."""
    if value in (None, ""):
        return PHASE_TOKEN_DEFAULTS.get(str(phase), ())
    if isinstance(value, str):
        values = value.replace(",", " ").split()
    else:
        values = value
    return tuple(sorted(int(token) for token in values))


def _expected_case_identity(case: dict) -> dict:
    """Normalize every scheduled field that a v1 result can prove."""
    identity = {}
    for field in ("suite", "workload", "required_publication", "backend", "mode", "dtype",
                  "contract", "routing", "phase", "combine_quant_mode", "resource_mode",
                  "activation_profile", "placement", "uneven_tokens", "warmup_semantics"):
        if field in case:
            identity[field] = str(case[field])
    for field in ("eplb", "canonical"):
        if field in case:
            identity[field] = bool(case[field])
    for field in ("ep", "samples_per_point", "gpus_per_node", "scale_up_domain"):
        if field in case:
            identity[field] = int(case[field])
    for field, default in (("hidden", 7168), ("topk", 8), ("experts", 256)):
        if field in case:
            identity[field] = int(case[field] or default)
    if "routing_step" in case:
        identity["routing_step"] = int(case["routing_step"] or 0)
    if "nodes" in case:
        identity["nodes"] = int(case["nodes"] or 1)
    if "timing" in case:
        identity["timing"] = tuple(int(value) for value in str(case["timing"]).split(":"))
    if "ladder" in case:
        identity["tokens"] = _tokens(case["ladder"], str(case.get("phase") or ""))
    if "_sku" in case:
        identity["sku"] = _normalized_sku(case["_sku"])
    return identity


def _actual_case_identity(doc: dict) -> dict:
    """Project a result onto the same v1 identity as its scheduled matrix case."""
    if doc.get("record_type") == "failed-case":
        failure = doc.get("failure") if isinstance(doc.get("failure"), dict) else {}
        raw = failure.get("case") if isinstance(failure.get("case"), dict) else {}
        case = dict(raw)
        aliases = {"dispatch_dtype": "dtype", "tokens_ladder": "ladder"}
        for source, target in aliases.items():
            if source in case:
                case[target] = case[source]
        case["_sku"] = doc.get("runner")
        if all(field in case for field in ("iters", "trials", "warmup")):
            case["timing"] = f"{case['iters']}:{case['trials']}:{case['warmup']}"
        return _expected_case_identity(case)

    shape = doc.get("shape") if isinstance(doc.get("shape"), dict) else {}
    quant = shape.get("quant") if isinstance(shape.get("quant"), dict) else {}
    reproduction = (doc.get("reproduction")
                    if isinstance(doc.get("reproduction"), dict) else {})
    placement = doc.get("placement") if isinstance(doc.get("placement"), dict) else {}
    workload = doc.get("workload") if isinstance(doc.get("workload"), dict) else {}
    logical_experts = shape.get("num_logical_experts") or shape.get("experts")
    return {
        "suite": doc.get("suite"),
        "workload": doc.get("workload_name"),
        "required_publication": doc.get("required_publication"),
        "backend": doc.get("backend"),
        "mode": doc.get("mode"),
        "dtype": shape.get("dispatch_dtype", reproduction.get("dispatch_dtype")),
        "contract": doc.get("measurement_contract"),
        "routing": shape.get("routing"),
        "phase": doc.get("phase"),
        "ep": doc.get("ep_size"),
        "eplb": bool(shape.get("eplb", False)),
        "combine_quant_mode": quant.get(
            "combine_quant_mode", reproduction.get("combine_quant_mode", "none")),
        "resource_mode": doc.get("resource_mode"),
        "activation_profile": shape.get(
            "activation_profile", reproduction.get("activation_profile", "normal")),
        "placement": placement.get("kind", "packed"),
        "routing_step": int(shape.get("routing_step", reproduction.get("routing_step", 0))),
        "uneven_tokens": shape.get(
            "uneven_tokens", reproduction.get("uneven_tokens", "none")),
        "hidden": shape.get("hidden"),
        "topk": shape.get("topk"),
        "experts": logical_experts,
        "samples_per_point": reproduction.get("samples_per_point"),
        "warmup_semantics": reproduction.get("warmup_semantics"),
        "timing": tuple(reproduction.get(field) for field in ("iters", "trials", "warmup")),
        "canonical": workload.get("source") == "canonical-serialized",
        "nodes": int(doc.get("nodes") or placement.get("nodes") or 1),
        "gpus_per_node": placement.get("gpus_per_node"),
        "scale_up_domain": placement.get("scale_up_domain"),
        "tokens": tuple(sorted(
            int(row["tokens_per_rank"]) for row in doc.get("rows", [])
            if row.get("tokens_per_rank") is not None
        )),
        "sku": _normalized_sku(doc.get("runner")),
    }


def _identity_differences(expected: dict, doc: dict) -> list[str]:
    expected_identity = _expected_case_identity(expected)
    actual_identity = _actual_case_identity(doc)
    return [
        f"{field}={actual_identity.get(field)!r}!={value!r}"
        for field, value in expected_identity.items()
        if actual_identity.get(field) != value
    ]


def validate_expected_coverage(docs: list[dict], matrix: dict) -> dict:
    """Require one semantically matching, sufficiently published result per scheduled case."""
    expected: dict[str, dict] = {}
    for shard in matrix.get("include", []):
        sku = _normalized_sku(shard.get("sku"))
        platform = cap.PLATFORMS.get(sku)
        if shard.get("sku") and platform is None:
            raise SystemExit(f"bundle: unknown matrix SKU {shard.get('sku')!r}")
        if platform:
            for field in ("gpus_per_node", "scale_up_domain"):
                if int(shard.get(field) or 0) != int(platform[field]):
                    raise SystemExit(
                        f"bundle: shard {shard.get('id')!r} has {field}={shard.get(field)!r}, "
                        f"expected {platform[field]} for {sku}"
                    )
        for case in shard.get("cases", []):
            if platform:
                for field in ("gpus_per_node", "scale_up_domain"):
                    if int(case.get(field) or 0) != int(platform[field]):
                        raise SystemExit(
                            f"bundle: case {case.get('case_id')!r} has {field}="
                            f"{case.get(field)!r}, expected {platform[field]} for {sku}"
                        )
            case_id = case.get("case_id")
            if not case_id:
                raise SystemExit("bundle: expected matrix case is missing case_id")
            if case_id in expected:
                raise SystemExit(f"bundle: duplicate expected case_id {case_id}")
            expected[case_id] = {**case, **({"_sku": shard["sku"]} if shard.get("sku") else {})}

    actual: dict[str, list[dict]] = {}
    missing_identity = 0
    identity_mismatch = []
    for doc in docs:
        if doc.get("family") != "moe":
            continue
        case_id = doc.get("case_id")
        if not case_id:
            missing_identity += 1
            continue
        case_id = str(case_id)
        if case_id in expected:
            differences = _identity_differences(expected[case_id], doc)
            if differences:
                identity_mismatch.append(f"{case_id}:" + ",".join(differences))
                continue
        actual.setdefault(case_id, []).append(doc)

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    duplicates = sorted(case_id for case_id, values in actual.items() if len(values) != 1)
    under_tier = []
    for case_id in sorted(set(expected) & set(actual)):
        if len(actual[case_id]) != 1:
            continue
        required = expected[case_id].get("required_publication") or "diagnostic"
        observed = actual[case_id][0].get("publication_status") or "invalid"
        if PUBLICATION_RANK.get(str(observed), -1) < PUBLICATION_RANK.get(str(required), 99):
            under_tier.append(f"{case_id}:{observed}<{required}")

    if missing_identity or missing or extra or duplicates or under_tier or identity_mismatch:
        details = (
            f"missing_identity={missing_identity} missing={missing[:8]} extra={extra[:8]} "
            f"duplicates={duplicates[:8]} under_tier={under_tier[:8]} "
            f"identity_mismatch={identity_mismatch[:8]}"
        )
        raise SystemExit(f"bundle: expected-matrix coverage failed ({details})")
    return {"expected": len(expected), "observed": len(actual), "complete": True}


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX publication bundle generator")
    ap.add_argument("--aggregate", nargs="+", required=True, help="aggregate .ndjson file(s)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results", "bundle"))
    ap.add_argument("--schema", default="",
                    help="override with one schema for all EP docs; blank selects v3-v5 per doc")
    ap.add_argument("--source-run-id", default=os.environ.get("GITHUB_RUN_ID", ""))
    ap.add_argument("--source-sha", default=os.environ.get("GITHUB_SHA", ""))
    ap.add_argument("--source-run-url", default="")
    ap.add_argument("--source-workflow", default=os.environ.get("GITHUB_WORKFLOW", ""))
    ap.add_argument("--matrix", default="", help="resolved matrix_full.json for exact case coverage")
    a = ap.parse_args()

    schema = json.load(open(a.schema)) if a.schema else vr.load_schema_registry()
    docs: list[dict] = []
    for path in a.aggregate:
        if not os.path.exists(path):
            raise SystemExit(f"bundle: aggregate not found: {path}")
        docs.extend(_load_ndjson(path))
    if not docs:
        raise SystemExit("bundle: aggregate is empty — nothing to publish")

    validation = validate(docs, schema)
    matrix_coverage = None
    if a.matrix:
        with open(a.matrix) as fh:
            matrix_coverage = validate_expected_coverage(docs, json.load(fh))

    os.makedirs(a.out_dir, exist_ok=True)
    files: list[str] = []
    for path in a.aggregate:
        dst = os.path.join(a.out_dir, os.path.basename(path))
        shutil.copyfile(path, dst)
        files.append(dst)

    manifest = {
        "bundle_format": BUNDLE_FORMAT,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {"run_id": a.source_run_id or None, "sha": a.source_sha or None,
                   "run_url": a.source_run_url or None, "workflow": a.source_workflow or None},
        "docs": len(docs),
        "validation": validation,
        "coverage": {**coverage(docs), **({"matrix": matrix_coverage} if matrix_coverage else {})},
        "files": {os.path.basename(p): {"sha256": _sha256(p), "bytes": os.path.getsize(p)}
                  for p in files},
    }
    mpath = os.path.join(a.out_dir, "manifest.json")
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2)
    files.append(mpath)

    with open(os.path.join(a.out_dir, "SHA256SUMS"), "w") as fh:
        for p in files:
            fh.write(f"{_sha256(p)}  {os.path.basename(p)}\n")

    print(f"bundle: {len(docs)} docs -> {a.out_dir} "
          f"({', '.join(sorted(os.path.basename(p) for p in files))}, SHA256SUMS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
