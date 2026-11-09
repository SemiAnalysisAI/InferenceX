from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, NamedTuple

from . import media

PLAN_FIELDS = ("model_id", "model_revision", "generation", "fairness")
CASE_FIELDS = ("case_id", "prompt", "seed", "requires_motion", "requires_sound")
READ_ERRORS = (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile)
# spec.json server keys a declared optimization sets, so they may differ from the lossless twin.
OPTIMIZATION_SERVER_KEYS = {"subblock-sparse-attn": {"attention_backend"}}
# Wall time of SSIM and LPIPS on one 124-frame 1344x768 clip on a GitHub-hosted runner.
# Byte-identical clips skip both and are not counted.
SECONDS_PER_DIFFERING_CLIP = 55


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


def _unpack(path: Path, scratch: Path) -> Path:
    if path.is_file():
        with zipfile.ZipFile(path) as archive:
            archive.extractall(scratch)
        return scratch
    return path


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


def _media_problem(root: Path, sums: dict[str, str], record: dict, clip: dict) -> str | None:
    path = clip["media_path"]
    if (
        record.get("status") != "succeeded"
        or (record.get("media") or {}).get("valid") is not True
        or path is None
        or not (root / path).is_file()
    ):
        return "invalid_candidate"
    if not (clip["media_sha256"] == sums.get(path) == _digest(root / path)):
        return "checksum_mismatch"
    return None


def _unscored_reason(
    root: Path,
    sums: dict[str, str],
    plan: dict,
    record: dict,
    candidate: dict,
    descriptor: dict,
    references: dict[str, tuple[dict, Path]],
) -> tuple[str, list[str]] | None:
    problem = _media_problem(root, sums, record, candidate)
    if problem is not None:
        return problem, []
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


def _error(error: Exception) -> str:
    if isinstance(error, OSError) and error.filename:
        return f"{type(error).__name__}: {Path(error.filename).name}"
    return f"{type(error).__name__}: {error}"


def _score_bundle(
    artifact_id: int,
    path: Path,
    scratch: Path,
    run_id: str,
    descriptor: dict,
    references: dict[str, tuple[dict, Path]],
) -> tuple[list[dict], list[dict]]:
    root = _unpack(path, scratch / str(artifact_id))
    sums = _checksums(root)
    pairs, unscored = [], []
    for plan, record, candidate in _measured(root, sums, run_id, artifact_id):
        failure = _unscored_reason(root, sums, plan, record, candidate, descriptor, references)
        if failure is not None:
            reason, fields = failure
            unscored.append({"candidate": candidate, "reason": reason, "fields": fields})
            continue
        case, reference = references[candidate["case_id"]]
        pairs.append(
            {
                "candidate": candidate,
                "reference": {
                    "receipt_sha256": case["receipt_sha256"],
                    "case_id": case["case_id"],
                    "media_path": case["asset"],
                    "media_sha256": case["sha256"],
                },
                "comparison": media.compare_media(reference, root / candidate["media_path"]),
            }
        )
    return pairs, unscored


def _scorer(source_revision: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("source revision must be a 40-character git commit")
    return {
        "name": "evaluator.media.compare_media",
        "method": "decoded-rgb24-all-frames",
        "source_revision": source_revision,
        "sha256": _digest(Path(media.__file__)),
    }


def _write(output: Path, attachment: dict, copies: dict[str, bytes]) -> None:
    output.mkdir(parents=True)
    for name, raw in copies.items():
        (output / name).write_bytes(raw)
    (output / "quality.json").write_text(
        json.dumps(attachment, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (output / "SHA256SUMS").write_text(
        "".join(f"{_digest(output / name)}  {name}\n" for name in sorted([*copies, "quality.json"]))
    )


def score_quality(
    *,
    bundles: dict[int, Path],
    references: Path,
    descriptor: Path,
    run_id: str,
    source_revision: str,
    output: Path,
) -> dict[str, Any]:
    """Score every measured clip against its public reference; unscorable clips and bundles stay listed."""
    scorer = _scorer(source_revision)
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    raw = descriptor.read_bytes()
    spec = json.loads(raw)
    known = _references(spec, references)
    pairs, unscored, unreadable = [], [], []
    with tempfile.TemporaryDirectory() as scratch:
        for artifact_id, path in sorted(bundles.items()):
            try:
                scored, skipped = _score_bundle(
                    artifact_id, path, Path(scratch), run_id, spec, known
                )
            except READ_ERRORS as error:
                unreadable.append({"artifact_id": artifact_id, "error": _error(error)})
                continue
            pairs += scored
            unscored += skipped
    attachment = {
        "schema_version": 1,
        "quality_status": "unjudged",
        "release_qualified": False,
        "scorer": scorer,
        "references": {
            "descriptor": descriptor.name,
            "descriptor_sha256": hashlib.sha256(raw).hexdigest(),
            "release": spec["release"],
        },
        "pairs": pairs,
        "unscored": unscored,
        "unreadable": unreadable,
    }
    _write(output, attachment, {descriptor.name: raw})
    return {
        "output": str(output),
        "pairs": len(pairs),
        "compatible": sum(pair["comparison"]["compatible"] for pair in pairs),
        "unscored": dict(sorted(Counter(item["reason"] for item in unscored).items())),
        "unreadable": len(unreadable),
    }


class _Clip(NamedTuple):
    plan: dict
    repetition: Any
    entry: dict
    media: Path | None
    # Why the clip cannot be compared, or None when its media and request are sealed and valid.
    problem: str | None


class _Deployment(NamedTuple):
    run_id: str
    artifact_id: int
    cell_id: str
    setup: dict
    optimizations: list[str]
    clips: list[_Clip]


def _deployments(root: Path, run_id: str, artifact_id: int) -> list[_Deployment]:
    """Each cell of a bundle with the serving setup a lossy clip must share with its lossless pair."""
    sums = _checksums(root)
    cells: dict[str, _Deployment] = {}
    for plan, record, entry in _measured(root, sums, run_id, artifact_id):
        cell_id = entry["cell_id"]
        if cell_id not in cells:
            cell = root / "gpu" / cell_id
            aggregate = json.loads((cell / "aggregate.json").read_bytes())
            configuration = json.loads((cell / "baseline" / "run.json").read_bytes())[
                "configuration"
            ]
            setup = {
                "hardware": aggregate["cluster"].partition("-")[0],
                "engine": configuration["runtime"],
                "image_digest": configuration["numerical_identity"]["image_digest"],
                "runtime_revision": configuration["runtime_revision"],
                "gpu_count": aggregate["num_gpus"],
                "precision": aggregate["precision"],
                "concurrency": configuration["serving"]["concurrency"],
                # Every deployment flag the producer records, including the parallel layout.
                "server": json.loads((cell / "spec.json").read_bytes())["server"],
            }
            optimizations = sorted(configuration.get("optimizations") or [])
            cells[cell_id] = _Deployment(run_id, artifact_id, cell_id, setup, optimizations, [])
        slot = entry["slot_id"]
        request = f"gpu/{cell_id}/baseline/requests/{slot}.json"
        sealed = (
            isinstance(slot, str)
            and re.fullmatch(r"[\w-]+", slot) is not None
            and (root / request).is_file()
            and sums.get(request) == _digest(root / request)
        )
        entry = {**entry, "request_sha256": sums[request] if sealed else None}
        problem = _media_problem(root, sums, record, entry) or (
            None if sealed else "invalid_candidate"
        )
        media_path = root / entry["media_path"] if problem is None else None
        cells[cell_id].clips.append(
            _Clip(plan, record.get("repetition"), entry, media_path, problem)
        )
    return list(cells.values())


def _matches(deployment: _Deployment, lossless: list[_Deployment]) -> list[_Deployment]:
    """Other lossless deployments whose setup differs only in the server keys the optimizations set."""
    source = (deployment.run_id, deployment.artifact_id, deployment.cell_id)
    owned = set().union(
        *(OPTIMIZATION_SERVER_KEYS.get(name, ()) for name in deployment.optimizations)
    )

    def key(setup: dict) -> dict:
        return {**setup, "server": {k: v for k, v in setup["server"].items() if k not in owned}}

    return [
        match
        for match in lossless
        if key(match.setup) == key(deployment.setup)
        and (match.run_id, match.artifact_id, match.cell_id) != source
    ]


def _pair(clip: _Clip, lossless: list[_Deployment]) -> tuple[_Clip | None, str | None, list[str]]:
    """The lossless clip a candidate clip is compared with, or why it has none."""
    if clip.problem is not None:
        return None, clip.problem, []
    if len(lossless) != 1:
        return None, "ambiguous_baseline" if lossless else "no_baseline", []
    same_case = [
        other
        for other in lossless[0].clips
        if (other.entry["case_id"], other.repetition) == (clip.entry["case_id"], clip.repetition)
    ]
    if len(same_case) != 1:
        return None, "no_baseline", []
    baseline = same_case[0]
    if baseline.problem is not None:
        return None, "invalid_baseline", []
    fields = [field for field in PLAN_FIELDS if clip.plan.get(field) != baseline.plan.get(field)]
    fields += [
        field
        for field in ("prompt", "seed", "request_sha256")
        if clip.entry[field] != baseline.entry[field]
    ]
    return baseline, ("settings_mismatch" if fields else None), fields


def _summary(comparisons: list[dict]) -> dict[str, Any]:
    """Median and worst clip over the compatible pairs; an identical clip's PSNR ranks as best."""
    metrics = [comparison["metrics"] for comparison in comparisons if comparison["compatible"]]
    psnr = [math.inf if m["video_psnr_db"] is None else m["video_psnr_db"] for m in metrics]
    ssim = [m["video_ssim"] for m in metrics]
    lpips = [m["video_lpips"] for m in metrics]

    def finite(value: float | None) -> float | None:
        return None if value is None or math.isinf(value) else value

    return {
        "pairs": len(comparisons),
        "compatible": len(metrics),
        "identical": sum(m["video_identical"] for m in metrics),
        "video_psnr_db": {
            "median": finite(statistics.median(psnr)) if psnr else None,
            "worst": finite(min(psnr, default=None)),
        },
        "video_ssim": {
            "median": statistics.median(ssim) if ssim else None,
            "worst": min(ssim, default=None),
        },
        "video_lpips": {
            "median": statistics.median(lpips) if lpips else None,
            "worst": max(lpips, default=None),
        },
    }


def _read(
    bundles: dict[int, Path], run_id: str, scratch: Path
) -> tuple[list[_Deployment], list[dict]]:
    deployments, unreadable = [], []
    for artifact_id, path in sorted(bundles.items()):
        try:
            root = _unpack(path, scratch / str(artifact_id))
            deployments += _deployments(root, run_id, artifact_id)
        except READ_ERRORS as error:
            unreadable.append({"artifact_id": artifact_id, "error": _error(error)})
    return deployments, unreadable


def score_paired(
    *,
    bundles: dict[int, Path],
    baselines: dict[int, Path],
    run_id: str,
    baseline_run_id: str,
    source_revision: str,
    output: Path,
    budget_minutes: float | None = None,
) -> dict[str, Any]:
    """Score each clip against the same case from the lossless deployment with the same setup.

    Refuses, before scoring, a dispatch whose differing clips need longer than ``budget_minutes``.
    """
    scorer = _scorer(source_revision)
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    pairs, unscored, deployments = [], [], []
    with tempfile.TemporaryDirectory() as scratch:
        found, baseline_unreadable = _read(baselines, baseline_run_id, Path(scratch, "baseline"))
        lossless = [deployment for deployment in found if not deployment.optimizations]
        candidates, unreadable = _read(bundles, run_id, Path(scratch, "candidate"))
        matched = []
        for deployment in candidates:
            matches = _matches(deployment, lossless)
            clips = [(clip, *_pair(clip, matches)) for clip in deployment.clips]
            matched.append((deployment, matches, clips))
        differing = sum(
            reason is None and clip.entry["media_sha256"] != baseline.entry["media_sha256"]
            for _, _, clips in matched
            for clip, baseline, reason, _ in clips
        )
        needed = differing * SECONDS_PER_DIFFERING_CLIP / 60
        if budget_minutes is not None and needed > budget_minutes:
            raise ValueError(
                f"{differing} clips differ from their lossless pair and need about {needed:.0f} min "
                f"of SSIM and LPIPS at {SECONDS_PER_DIFFERING_CLIP} s each, over the "
                f"{budget_minutes:g}-min budget; score fewer bundles per job"
            )
        for deployment, matches, clips in matched:
            comparisons = []
            for clip, baseline, reason, fields in clips:
                if reason is not None:
                    unscored.append({"candidate": clip.entry, "reason": reason, "fields": fields})
                    continue
                comparison = media.compare_media(baseline.media, clip.media, per_frame=True)
                comparisons.append(comparison)
                pairs.append(
                    {"candidate": clip.entry, "baseline": baseline.entry, "comparison": comparison}
                )
            deployments.append(
                {
                    "candidate": {
                        "artifact_id": deployment.artifact_id,
                        "cell_id": deployment.cell_id,
                        "optimizations": deployment.optimizations,
                    },
                    "baseline": [
                        {"artifact_id": match.artifact_id, "cell_id": match.cell_id}
                        for match in matches
                    ],
                    "setup": deployment.setup,
                    "summary": _summary(comparisons),
                }
            )
    attachment = {
        "schema_version": 1,
        "mode": "paired_lossless",
        "quality_status": "unjudged",
        "release_qualified": False,
        "scorer": {
            **scorer,
            "lpips": {
                "net": "alex",
                "version": "0.1",
                "weights_sha256": media.LPIPS_WEIGHTS_SHA256,
            },
        },
        "baseline": {
            "run_id": baseline_run_id,
            "artifacts": sorted(baselines),
            "unreadable": baseline_unreadable,
        },
        "deployments": deployments,
        "pairs": pairs,
        "unscored": unscored,
        "unreadable": unreadable,
    }
    _write(output, attachment, {})
    return {
        "output": str(output),
        "deployments": [
            {**item["candidate"], **item["summary"]} for item in deployments if item["baseline"]
        ],
        "unscored": dict(sorted(Counter(item["reason"] for item in unscored).items())),
        "unreadable": len(unreadable),
        "baseline_unreadable": len(baseline_unreadable),
    }
