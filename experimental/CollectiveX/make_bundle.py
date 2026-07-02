#!/usr/bin/env python3
"""CollectiveX publication bundle generator (goal P1: continuous benchmark infrastructure).

Turns a validated aggregate into ONE self-contained, citable directory:

    bundle/
      manifest.json      bundle format, source run provenance, coverage + validation counts
      <aggregate>.ndjson the schema-validated dataset (copied verbatim)
      report.html        the 8-tab plot_ep.py report rendered from exactly this dataset
      SUMMARY.md         summarize.py markdown over exactly this dataset
      SHA256SUMS         checksums of every file above

Fail-loud doctrine: every doc in the aggregate is validated (ep-result-v4 schema +
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
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import validate_results as vr  # noqa: E402

BUNDLE_FORMAT = 1


def _sku_of(doc: dict) -> str:
    """SKU token from the runner name: 'h100-dgxc-slurm_19' -> 'h100', 'gb300-8x' -> 'gb300'."""
    runner = str(doc.get("runner") or "unknown")
    return runner.split("_")[0].split("-")[0] or "unknown"


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
                docs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"bundle: {path}:{i + 1} is not JSON ({exc}) — refusing to bundle")
    return docs


def validate(docs: list[dict], schema: dict | None) -> dict:
    """Validate every doc; return counts. Aborts (SystemExit) on any error — a bundle
    must certify its dataset. Non-moe families (kv-cache, nccl, ...) carry their own
    v1 schemas and are counted but not gated here."""
    by_status: dict[str, int] = {}
    by_family: dict[str, int] = {}
    n_err = 0
    for i, doc in enumerate(docs):
        fam = doc.get("family") or "unknown"
        by_family[fam] = by_family.get(fam, 0) + 1
        if fam != "moe":
            continue
        errs, _warns, status = vr.validate_doc(doc, schema, f"doc[{i}]")
        by_status[status] = by_status.get(status, 0) + 1
        for e in errs:
            n_err += 1
            print(f"bundle: INVALID doc[{i}] ({doc.get('backend')}/{doc.get('runner')}): {e}",
                  file=sys.stderr)
    if n_err:
        raise SystemExit(f"bundle: {n_err} validation error(s) — refusing to publish a tainted bundle")
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


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX publication bundle generator")
    ap.add_argument("--aggregate", nargs="+", required=True, help="aggregate .ndjson file(s)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results", "bundle"))
    ap.add_argument("--schema", default=os.path.join(HERE, "schemas", "ep-result-v4.schema.json"))
    ap.add_argument("--source-run-id", default=os.environ.get("GITHUB_RUN_ID", ""))
    ap.add_argument("--source-sha", default=os.environ.get("GITHUB_SHA", ""))
    ap.add_argument("--source-run-url", default="")
    ap.add_argument("--source-workflow", default=os.environ.get("GITHUB_WORKFLOW", ""))
    ap.add_argument("--skip-report", action="store_true",
                    help="skip report.html/SUMMARY.md (dataset + manifest only)")
    a = ap.parse_args()

    schema = json.load(open(a.schema)) if os.path.exists(a.schema) else None
    docs: list[dict] = []
    for path in a.aggregate:
        if not os.path.exists(path):
            raise SystemExit(f"bundle: aggregate not found: {path}")
        docs.extend(_load_ndjson(path))
    if not docs:
        raise SystemExit("bundle: aggregate is empty — nothing to publish")

    validation = validate(docs, schema)

    os.makedirs(a.out_dir, exist_ok=True)
    files: list[str] = []
    for path in a.aggregate:
        dst = os.path.join(a.out_dir, os.path.basename(path))
        shutil.copyfile(path, dst)
        files.append(dst)

    if not a.skip_report:
        # plot_ep reads ndjson directly; summarize needs per-doc JSON (aggregate --explode).
        with tempfile.TemporaryDirectory() as tmp:
            for path in a.aggregate:
                shutil.copyfile(path, os.path.join(tmp, os.path.basename(path)))
            subprocess.run([sys.executable, os.path.join(HERE, "aggregate_results.py"),
                            "--in-dir", tmp, "--explode", tmp], check=True, cwd=HERE)
            report = os.path.join(a.out_dir, "report.html")
            subprocess.run([sys.executable, os.path.join(HERE, "plot_ep.py"),
                            "--results-dir", tmp, "--out", report], check=True, cwd=HERE)
            files.append(report)
            summary_md = subprocess.run([sys.executable, os.path.join(HERE, "summarize.py"),
                                         "--results-dir", tmp, "--markdown"],
                                        check=True, cwd=HERE, capture_output=True, text=True).stdout
            summary = os.path.join(a.out_dir, "SUMMARY.md")
            with open(summary, "w") as fh:
                fh.write(summary_md)
            files.append(summary)

    manifest = {
        "bundle_format": BUNDLE_FORMAT,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {"run_id": a.source_run_id or None, "sha": a.source_sha or None,
                   "run_url": a.source_run_url or None, "workflow": a.source_workflow or None},
        "docs": len(docs),
        "validation": validation,
        "coverage": coverage(docs),
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
