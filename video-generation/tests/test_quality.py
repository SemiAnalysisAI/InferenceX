import hashlib
import json
import shutil
from pathlib import Path

import pytest

from evaluator import cli
from test_media import encode_media

REVISION = "a" * 40
CASES = [
    {
        "case_id": name,
        "prompt": f"{name} prompt",
        "seed": seed,
        "requires_motion": True,
        "requires_sound": True,
    }
    for seed, name in enumerate(("same", "changed", "reseeded"), 1)
]
SETTINGS = {
    "model_id": "MiniMaxAI/MiniMax-H3",
    "model_revision": "b" * 40,
    "generation": {"frame_count": 8, "fps": 10},
    "fairness": {"encoding": {"crf": 18}},
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _setup(root: Path) -> tuple[Path, Path, Path]:
    references = root / "references"
    references.mkdir()
    for case in CASES:
        encode_media(references / f"{case['case_id']}.mp4")
    descriptor = root / "h3-v2.json"
    descriptor.write_text(
        json.dumps(
            {
                **SETTINGS,
                "release": {"repository": "SemiAnalysisAI/InferenceX", "tag": "refs-v1"},
                "cases": [
                    {
                        **case,
                        "asset": f"{case['case_id']}.mp4",
                        "sha256": _sha(references / f"{case['case_id']}.mp4"),
                        "bytes": (references / f"{case['case_id']}.mp4").stat().st_size,
                        "receipt_sha256": "c" * 64,
                        "plan_sha256": "d" * 64,
                    }
                    for case in CASES
                ],
            }
        )
    )
    bundle = root / "bundle"
    artifacts = bundle / "gpu/c1/baseline/artifacts"
    artifacts.mkdir(parents=True)
    shutil.copyfile(references / "same.mp4", artifacts / "measurement-r001-c001.mp4")
    encode_media(artifacts / "measurement-r001-c002.mp4", video_change=True)
    shutil.copyfile(references / "reseeded.mp4", artifacts / "measurement-r001-c003.mp4")
    records = [
        {
            "phase": "measurement",
            "slot_id": f"measurement-r001-c{index:03d}",
            "case_id": case["case_id"],
            "prompt": case["prompt"],
            "seed": 99 if case["case_id"] == "reseeded" else case["seed"],
            "status": "succeeded",
            "media": {"valid": True},
            "artifact_path": f"artifacts/measurement-r001-c{index:03d}.mp4",
            "sha256": _sha(artifacts / f"measurement-r001-c{index:03d}.mp4"),
        }
        for index, case in enumerate(CASES, 1)
    ]
    run = {"plan": {**SETTINGS, "cases": CASES}, "records": records}
    (bundle / "gpu/c1/baseline/run.json").write_text(json.dumps(run))
    (bundle / "serving-smoke.json").write_text(json.dumps({"cells": [{"concurrency": 1}]}))
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "41"}))
    (bundle / "SHA256SUMS").write_text(
        "".join(
            f"{_sha(path)}  {path.relative_to(bundle).as_posix()}\n"
            for path in sorted(bundle.rglob("*"))
            if path.is_file()
        )
    )
    shutil.make_archive(str(root / "bundle"), "zip", bundle)
    return descriptor, references, bundle


def _score(descriptor: Path, references: Path, output: Path, *bundles: str) -> int:
    return cli.main(
        [
            "score-quality",
            *bundles,
            "--descriptor",
            str(descriptor),
            "--references",
            str(references),
            "--run-id",
            "41",
            "--source-revision",
            REVISION,
            "--output",
            str(output),
        ]
    )


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    root = tmp_path_factory.mktemp("quality")
    descriptor, references, bundle = _setup(root)
    shutil.copytree(bundle, root / "partial")
    (root / "partial/manifest.json").unlink()
    zipped, directory, partial = f"7={root / 'bundle.zip'}", f"8={bundle}", f"9={root / 'partial'}"
    assert _score(descriptor, references, root / "first", zipped, partial, directory) == 0
    assert _score(descriptor, references, root / "second", directory, zipped, partial) == 0
    return root


def test_scores_each_measured_clip_against_its_own_reference(scored):
    quality = json.loads((scored / "first/quality.json").read_bytes())
    assert (scored / "first/quality.json").read_bytes() == (
        scored / "second/quality.json"
    ).read_bytes()
    assert (
        f"{_sha(scored / 'first/quality.json')}  quality.json"
        in (scored / "first/SHA256SUMS").read_text().splitlines()
    )
    pairs = {
        (p["candidate"]["artifact_id"], p["candidate"]["case_id"]): p for p in quality["pairs"]
    }
    assert sorted(pairs) == [(7, "changed"), (7, "same"), (8, "changed"), (8, "same")]
    for artifact in (7, 8):
        same, changed = pairs[artifact, "same"], pairs[artifact, "changed"]
        assert same["reference"]["media_path"] == "same.mp4"
        assert same["comparison"]["metrics"]["video_identical"] is True
        assert changed["candidate"]["media_path"] == (
            "gpu/c1/baseline/artifacts/measurement-r001-c002.mp4"
        )
        assert changed["comparison"]["compatible"] is True
        assert 0 < changed["comparison"]["metrics"]["video_psnr_db"] < 15


def test_reports_a_clip_generated_with_other_settings_instead_of_scoring_it(scored):
    quality = json.loads((scored / "first/quality.json").read_bytes())
    assert [
        (
            item["candidate"]["artifact_id"],
            item["candidate"]["case_id"],
            item["reason"],
            item["fields"],
        )
        for item in quality["unscored"]
    ] == [
        (7, "reseeded", "settings_mismatch", ["seed"]),
        (8, "reseeded", "settings_mismatch", ["seed"]),
    ]


def test_lists_a_partial_bundle_as_unreadable_and_still_scores_the_others(scored):
    quality = json.loads((scored / "first/quality.json").read_bytes())
    assert quality["unreadable"] == [
        {"artifact_id": 9, "error": "FileNotFoundError: manifest.json"}
    ]
    assert {pair["candidate"]["artifact_id"] for pair in quality["pairs"]} == {7, 8}


def test_refuses_a_reference_asset_that_differs_from_the_descriptor(tmp_path):
    descriptor, references, bundle = _setup(tmp_path)
    tampered = bytearray((references / "changed.mp4").read_bytes())
    tampered[-1] ^= 1
    (references / "changed.mp4").write_bytes(tampered)
    assert _score(descriptor, references, tmp_path / "out", f"7={bundle}") == 2
    assert not (tmp_path / "out").exists()
