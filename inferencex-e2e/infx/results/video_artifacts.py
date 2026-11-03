"""Validate every selected native video deployment in a downloaded PR sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

VIDEO_ARTIFACT = re.compile(r"h3-results-([0-9]+)-([0-9]+)-([0-9a-f]{64})")


def _object(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("video evidence: expected an object")
    return value


def _objects(value: object) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError("video evidence: expected a list of objects")
    return [_object(item) for item in value]


def _read(path: Path) -> dict:
    return _object(json.loads(path.read_text()))


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


def _reference(inventory: dict[str, str], reference: dict, *, prefix: str = "") -> str:
    if not isinstance(reference, dict):
        raise ValueError("invalid video evidence reference")
    path, digest = reference.get("path"), reference.get("sha256")
    if (
        not isinstance(path, str)
        or not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
    ):
        raise ValueError("missing path or SHA256 in video evidence reference")
    relative = PurePosixPath(prefix, path).as_posix()
    if relative not in inventory or inventory[relative] != digest:
        raise ValueError(f"video evidence reference mismatch: {relative}")
    return relative


def validate_video_artifacts(
    root: Path,
    *,
    run_id: str | None = None,
    run_attempt: str | None = None,
    head_sha: str | None = None,
) -> int:
    """Return the deployment count; reject missing rows, cells, provenance or sealed files.

    The workflow supplies trusted run/head identity. ``git_commit`` remains the
    actual executed checkout, which can be the PR merge commit rather than its head.
    Older attempts remain archived but do not fill holes in the selected attempt.
    """
    sweep = _read(root / "video-sweep-manifest/sweep_manifest.json")
    source_run, attempt, head = str(sweep["run-id"]), str(sweep["run-attempt"]), sweep["head"]
    if (
        (run_id is not None and run_id != source_run)
        or (run_attempt is not None and run_attempt != attempt)
        or (head_sha is not None and head_sha != head)
    ):
        raise ValueError("video sweep manifest disagrees with trusted run/head/attempt identity")
    if sweep.get("video-full-sweep") is not True:
        raise ValueError("native video reuse requires a full video sweep")
    rows = _objects(_object(_object(sweep["matrix"]).get("single_node", {})).get("video", []))
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
            or _object(doc.get("ci", {})).get("head_sha") != head
            for doc in (manifest, ci)
        ):
            raise ValueError(f"{name}: bundle CI identity mismatch")
        if not manifest.get("git_commit") or manifest["git_commit"] != ci.get("source_sha"):
            raise ValueError(f"{name}: executed checkout mismatch")
        if manifest.get("recipe_fingerprint") != row["recipe-fingerprint"]:
            raise ValueError(f"{name}: recipe fingerprint mismatch")
        if any(
            inventory.get(path) != digest
            for path, digest in _object(manifest.get("evidence", {})).items()
        ):
            raise ValueError(f"{name}: manifest evidence hash mismatch")
        if matrix.get("status") != "complete" or matrix.get("power_verified") is not True:
            raise ValueError(f"{name}: incomplete measured video deployment")
        cells = _objects(matrix.get("cells", []))
        if sorted(cell["concurrency"] for cell in cells) != sorted(row["conc-list"]):
            raise ValueError(f"{name}: concurrency coverage mismatch")
        references = set()
        for cell in cells:
            if cell.get("status") != "complete" or cell.get("verified") is not True:
                raise ValueError(f"{name}: incomplete video cell")
            for field in ("run", "receipt", "power"):
                reference = _reference(inventory, cell[field])
                if reference in references:
                    raise ValueError(f"{name}: reused video cell evidence")
                references.add(reference)
            run = _read(bundle / cell["run"]["path"])
            receipt = _read(bundle / cell["receipt"]["path"])
            power = _read(bundle / cell["power"]["path"])
            if (
                power.get("valid") is not True
                or power.get("status") != "valid"
                or receipt.get("measurement_verified") is not True
                or receipt.get("power_verified") is not True
                or receipt.get("status") != "complete"
            ):
                raise ValueError(f"{name}: unverified native measurement or power receipt")
            plan = run.get("plan")
            plan_sha = hashlib.sha256(
                json.dumps(
                    plan,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            if (
                not isinstance(plan, dict)
                or run.get("plan_sha256") != plan_sha
                or manifest.get("workload_plan") != plan
                or matrix.get("plan") != plan
                or plan.get("model_id") != row["model"]
            ):
                raise ValueError(f"{name}: selected workload plan mismatch")
            repetitions, cases = plan.get("repetitions"), _objects(plan.get("cases"))
            if (
                not isinstance(repetitions, int)
                or repetitions <= 0
                or not isinstance(cases, list)
                or not cases
            ):
                raise ValueError(f"{name}: invalid measured workload population")
            expected_slots = {
                f"measurement-r{repetition:03d}-c{index:03d}": {**case, "repetition": repetition}
                for repetition in range(1, repetitions + 1)
                for index, case in enumerate(cases, 1)
            }
            summary = _object(run.get("summary", {}))
            measured = [
                r for r in _objects(run.get("records", [])) if r.get("phase") == "measurement"
            ]
            if (
                run.get("status") != "complete"
                or summary.get("scheduled") != len(expected_slots)
                or summary.get("valid") != len(expected_slots)
                or len(measured) != len(expected_slots)
                or {r.get("slot_id") for r in measured} != expected_slots.keys()
            ):
                raise ValueError(f"{name}: incomplete measured request population")
            media_paths = set()
            for record in measured:
                expected_slot = expected_slots[record["slot_id"]]
                if (
                    any(
                        record.get(key) != expected_slot[key]
                        for key in ("case_id", "prompt", "seed", "repetition")
                    )
                    or record.get("status") != "succeeded"
                    or _object(record.get("media", {})).get("valid") is not True
                ):
                    raise ValueError(f"{name}: invalid measured workload slot")
                path = _reference(
                    inventory,
                    {"path": record.get("artifact_path"), "sha256": record.get("sha256")},
                    prefix=PurePosixPath(cell["run"]["path"]).parent.as_posix(),
                )
                if path in media_paths:
                    raise ValueError(f"{name}: duplicate measured media reference")
                media_paths.add(path)
            configuration = _object(run.get("configuration", {}))
            concurrency = cell["concurrency"]
            if (
                type(concurrency) is not int
                or concurrency <= 0
                or _object(run.get("measurement", {})).get("concurrency") != concurrency
                or _object(configuration.get("serving", {})).get("concurrency") != concurrency
            ):
                raise ValueError(f"{name}: measured concurrency mismatch")
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
                or len(set(matrix.get("gpu_uuids", []))) != row["gpus"]
                or len(matrix.get("gpu_uuids", [])) != row["gpus"]
            ):
                raise ValueError(f"{name}: selected engine or GPU count mismatch")
    return len(expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-attempt", required=True)
    parser.add_argument("--head-sha", required=True)
    args = parser.parse_args()
    try:
        count = validate_video_artifacts(
            args.artifacts_dir,
            run_id=args.run_id,
            run_attempt=args.run_attempt,
            head_sha=args.head_sha,
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"Video sweep artifact validation failed: {error}\n")
    print(f"Validated {count} complete native video deployment(s).")


if __name__ == "__main__":
    main()
