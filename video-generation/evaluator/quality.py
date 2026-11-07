from __future__ import annotations

import hashlib
import json
import re
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from . import media

PLAN_FIELDS = ("model_id", "model_revision", "generation", "fairness")
CASE_FIELDS = ("case_id", "prompt", "seed", "requires_motion", "requires_sound")


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _checksums(root: Path) -> dict[str, str]:
    sums: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match or match[2] in sums:
            raise ValueError(f"{root.name}: malformed or duplicate checksum entry")
        sums[match[2]] = match[1]
    return sums


def _references(descriptor: dict, directory: Path) -> dict[str, tuple[dict, Path]]:
    references = {}
    for case in descriptor["cases"]:
        path = directory / case["asset"]
        if (
            not path.is_file()
            or path.stat().st_size != case["bytes"]
            or _digest(path) != case["sha256"]
        ):
            raise ValueError(f"reference asset {case['asset']} does not match the descriptor")
        references[case["case_id"]] = (case, path)
    return references


def _measured(
    root: Path, sums: dict[str, str], run_id: str, artifact_id: int
) -> Iterator[tuple[dict, dict, dict]]:
    manifest_sha = _digest(root / "manifest.json")
    if sums.get("manifest.json") != manifest_sha:
        raise ValueError(f"artifact {artifact_id}: manifest.json does not match SHA256SUMS")
    source_id = json.loads((root / "manifest.json").read_bytes())["run_id"]
    for cell in json.loads((root / "serving-smoke.json").read_bytes())["cells"]:
        cell_id = f"c{cell['concurrency']}"
        base = PurePosixPath("gpu", cell_id, "baseline")
        if not (root / base / "run.json").is_file():
            continue
        run = json.loads((root / base / "run.json").read_bytes())
        for record in run.get("records") or []:
            if record.get("phase") != "measurement":
                continue
            relative = PurePosixPath(record.get("artifact_path") or "")
            safe = relative.parts and not relative.is_absolute() and ".." not in relative.parts
            yield (
                run["plan"],
                record,
                {
                    "run_id": run_id,
                    "artifact_id": artifact_id,
                    "source_id": source_id,
                    "manifest_sha256": manifest_sha,
                    "cell_id": cell_id,
                    "slot_id": record.get("slot_id"),
                    "case_id": record.get("case_id"),
                    "prompt": record.get("prompt"),
                    "seed": record.get("seed"),
                    "media_path": (base / relative).as_posix() if safe else None,
                    "media_sha256": record.get("sha256"),
                },
            )


def _unscored_reason(
    root: Path,
    sums: dict[str, str],
    plan: dict,
    record: dict,
    candidate: dict,
    descriptor: dict,
    references: dict[str, tuple[dict, Path]],
) -> tuple[str, list[str]] | None:
    path = candidate["media_path"]
    if (
        record.get("status") != "succeeded"
        or (record.get("media") or {}).get("valid") is not True
        or path is None
        or not (root / path).is_file()
    ):
        return "invalid_candidate", []
    if not (candidate["media_sha256"] == sums.get(path) == _digest(root / path)):
        return "checksum_mismatch", []
    if candidate["case_id"] not in references:
        return "no_reference", []
    reference = references[candidate["case_id"]][0]
    planned = [
        case for case in plan.get("cases") or [] if case.get("case_id") == candidate["case_id"]
    ]
    recorded = {
        **(planned[0] if len(planned) == 1 else {}),
        **{field: candidate[field] for field in ("case_id", "prompt", "seed")},
    }
    fields = [field for field in PLAN_FIELDS if plan.get(field) != descriptor[field]]
    fields += [field for field in CASE_FIELDS if recorded.get(field) != reference[field]]
    return ("settings_mismatch", fields) if fields else None


def score_quality(
    *,
    bundles: dict[int, Path],
    references: Path,
    descriptor: Path,
    run_id: str,
    source_revision: str,
    output: Path,
) -> dict[str, Any]:
    """Score every measured clip against its public reference; unscorable clips stay listed."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("source revision must be a 40-character git commit")
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    raw = descriptor.read_bytes()
    spec = json.loads(raw)
    known = _references(spec, references)
    pairs, unscored = [], []
    with tempfile.TemporaryDirectory() as scratch:
        for artifact_id, path in sorted(bundles.items()):
            root = path
            if path.is_file():
                root = Path(scratch) / str(artifact_id)
                with zipfile.ZipFile(path) as archive:
                    archive.extractall(root)
            sums = _checksums(root)
            for plan, record, candidate in _measured(root, sums, run_id, artifact_id):
                failure = _unscored_reason(root, sums, plan, record, candidate, spec, known)
                if failure is not None:
                    reason, fields = failure
                    unscored.append({"candidate": candidate, "reason": reason, "fields": fields})
                    continue
                case, reference = known[candidate["case_id"]]
                pairs.append(
                    {
                        "candidate": candidate,
                        "reference": {
                            "receipt_sha256": case["receipt_sha256"],
                            "case_id": case["case_id"],
                            "media_path": case["asset"],
                            "media_sha256": case["sha256"],
                        },
                        "comparison": media.compare_media(
                            reference, root / candidate["media_path"]
                        ),
                    }
                )
    attachment = {
        "schema_version": 1,
        "quality_status": "unjudged",
        "release_qualified": False,
        "scorer": {
            "name": "evaluator.media.compare_media",
            "method": "decoded-rgb24-all-frames",
            "source_revision": source_revision,
            "sha256": _digest(Path(media.__file__)),
        },
        "references": {
            "descriptor": descriptor.name,
            "descriptor_sha256": hashlib.sha256(raw).hexdigest(),
            "release": spec["release"],
        },
        "pairs": pairs,
        "unscored": unscored,
    }
    output.mkdir(parents=True)
    (output / descriptor.name).write_bytes(raw)
    (output / "quality.json").write_text(
        json.dumps(attachment, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{_digest(output / name)}  {name}\n"
            for name in sorted([descriptor.name, "quality.json"])
        )
    )
    return {
        "output": str(output),
        "pairs": len(pairs),
        "compatible": sum(pair["comparison"]["compatible"] for pair in pairs),
        "unscored": dict(sorted(Counter(item["reason"] for item in unscored).items())),
    }
