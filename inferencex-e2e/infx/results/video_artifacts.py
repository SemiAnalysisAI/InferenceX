"""Validate every selected native video deployment in a downloaded PR sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

VIDEO_ARTIFACT = re.compile(r"h3-results-([0-9]+)-([0-9]+)-([0-9a-f]{64})")


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: expected an object")
    return value


def _inventory(root: Path) -> dict[str, str]:
    expected = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match or match[2] in expected:
            raise ValueError(f"{root.name}: malformed or duplicate checksum entry")
        expected[match[2]] = match[1]
    actual = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"{root.name}: symlink in bundle")
        if path.is_file() and path != root / "SHA256SUMS":
            with path.open("rb") as stream:
                actual[path.relative_to(root).as_posix()] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    if actual != expected:
        raise ValueError(f"{root.name}: checksum mismatch or incomplete inventory")
    return actual


def validate_video_artifacts(
    root: Path, *, run_id: str | None = None, head_sha: str | None = None
) -> int:
    """Return the deployment count; reject missing rows, cells, provenance or sealed files.

    The workflow supplies trusted run/head identity. ``git_commit`` remains the
    actual executed checkout, which can be the PR merge commit rather than its head.
    Older attempts remain archived but do not fill holes in the selected attempt.
    """
    sweep = _read(root / "video-sweep-manifest/sweep_manifest.json")
    source_run, attempt, head = str(sweep["run-id"]), str(sweep["run-attempt"]), sweep["head"]
    if (run_id is not None and run_id != source_run) or (head_sha is not None and head_sha != head):
        raise ValueError("video sweep manifest disagrees with trusted run/head identity")
    rows = sweep["matrix"].get("single_node", {}).get("video", [])
    if not rows:
        raise ValueError("video sweep manifest has no selected video deployments")
    expected = {}
    for row in rows:
        fingerprint = row.get("recipe-fingerprint", "")
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise ValueError("selected video deployment is missing its recipe fingerprint")
        name = f"h3-results-{source_run}-{attempt}-{fingerprint}"
        if name in expected:
            raise ValueError("duplicate selected video deployment")
        expected[name] = row
    actual = {
        path.name for path in root.glob(f"h3-results-{source_run}-{attempt}-*") if path.is_dir()
    }
    if actual != expected.keys():
        raise ValueError(
            f"video deployment coverage mismatch: missing={sorted(expected.keys() - actual)}, extra={sorted(actual - expected.keys())}"
        )
    for name, row in expected.items():
        bundle = root / name
        inventory = _inventory(bundle)
        manifest, ci, matrix = (
            _read(bundle / filename)
            for filename in ("manifest.json", "ci.json", "serving-smoke.json")
        )
        if any(
            doc.get("artifact_name") != name
            or str(doc.get("run_id")) != source_run
            or str(doc.get("run_attempt")) != attempt
            or doc.get("ci", {}).get("head_sha") != head
            for doc in (manifest, ci)
        ):
            raise ValueError(f"{name}: bundle CI identity mismatch")
        if not manifest.get("git_commit") or manifest["git_commit"] != ci.get("source_sha"):
            raise ValueError(f"{name}: executed checkout mismatch")
        if manifest.get("recipe_fingerprint") != row["recipe-fingerprint"]:
            raise ValueError(f"{name}: recipe fingerprint mismatch")
        if any(
            inventory.get(path) != digest for path, digest in manifest.get("evidence", {}).items()
        ):
            raise ValueError(f"{name}: manifest evidence hash mismatch")
        if matrix.get("status") != "complete" or matrix.get("power_verified") is not True:
            raise ValueError(f"{name}: incomplete measured video deployment")
        cells = matrix.get("cells", [])
        if sorted(cell["concurrency"] for cell in cells) != sorted(row["conc-list"]):
            raise ValueError(f"{name}: concurrency coverage mismatch")
        for cell in cells:
            if cell.get("status") != "complete" or cell.get("verified") is not True:
                raise ValueError(f"{name}: incomplete video cell")
            for field in ("run", "receipt", "power"):
                reference = cell[field]
                if inventory.get(reference["path"]) != reference["sha256"]:
                    raise ValueError(f"{name}: {field} reference mismatch")
            run = _read(bundle / cell["run"]["path"])
            summary = run.get("summary", {})
            if (
                run.get("status") != "complete"
                or not summary.get("scheduled")
                or summary.get("valid") != summary["scheduled"]
            ):
                raise ValueError(f"{name}: incomplete measured request population")
            configuration = run.get("configuration", {})
            engine = {
                "name": configuration.get("runtime"),
                "revision": configuration.get("runtime_revision"),
            }
            if (
                not run.get("plan_sha256")
                or not all(engine.values())
                or any(
                    doc.get("plan_sha256") != run["plan_sha256"] or doc.get("engine") != engine
                    for doc in (manifest, matrix)
                )
            ):
                raise ValueError(f"{name}: workload or engine identity mismatch")
            if (
                engine["name"] != row["framework"]
                or len(matrix.get("gpu_uuids", [])) != row["gpus"]
            ):
                raise ValueError(f"{name}: selected engine or GPU count mismatch")
    return len(expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--head-sha", required=True)
    args = parser.parse_args()
    try:
        count = validate_video_artifacts(
            args.artifacts_dir, run_id=args.run_id, head_sha=args.head_sha
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"Video sweep artifact validation failed: {error}\n")
    print(f"Validated {count} complete native video deployment(s).")


if __name__ == "__main__":
    main()
