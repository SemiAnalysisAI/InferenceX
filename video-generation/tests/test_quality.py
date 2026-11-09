import hashlib
import json
import shutil
from pathlib import Path

import pytest

from evaluator import cli, media
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


def _bundle(
    bundle: Path,
    media: list[Path],
    *,
    seeds: dict[str, int] | None = None,
    server: dict | None = None,
    precision: str = "bf16",
    optimizations: list[str] | None = None,
) -> Path:
    """An h3-results bundle whose c1 cell measured CASES in order with these media files."""
    cell = bundle / "gpu/c1"
    (cell / "baseline/artifacts").mkdir(parents=True)
    (cell / "baseline/requests").mkdir()
    records = []
    for index, (case, source) in enumerate(zip(CASES, media, strict=True), 1):
        slot = f"measurement-r001-c{index:03d}"
        seed = (seeds or {}).get(case["case_id"], case["seed"])
        shutil.copyfile(source, cell / f"baseline/artifacts/{slot}.mp4")
        (cell / f"baseline/requests/{slot}.json").write_text(
            json.dumps({"prompt": case["prompt"], "seed": seed})
        )
        records.append(
            {
                "phase": "measurement",
                "slot_id": slot,
                "repetition": 1,
                "case_id": case["case_id"],
                "prompt": case["prompt"],
                "seed": seed,
                "status": "succeeded",
                "media": {"valid": True},
                "artifact_path": f"artifacts/{slot}.mp4",
                "sha256": _sha(source),
            }
        )
    configuration = {
        "runtime": "sglang",
        "runtime_revision": "e" * 40,
        "numerical_identity": {"image_digest": "f" * 64},
        "serving": {"concurrency": 1},
        "optimizations": optimizations or [],
    }
    run = {"plan": {**SETTINGS, "cases": CASES}, "records": records, "configuration": configuration}
    (cell / "baseline/run.json").write_text(json.dumps(run))
    aggregate = {"cluster": "h100-dsxe", "num_gpus": 4, "precision": precision}
    (cell / "aggregate.json").write_text(json.dumps(aggregate))
    layout = {
        "tp_size": 2,
        "ulysses_degree": 2,
        "encoder_parallel": "auto",
        "dit_cpu_offload": False,
    }
    (cell / "spec.json").write_text(json.dumps({"server": {**layout, **(server or {})}}))
    (bundle / "serving-smoke.json").write_text(json.dumps({"cells": [{"concurrency": 1}]}))
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "41"}))
    (bundle / "SHA256SUMS").write_text(
        "".join(
            f"{_sha(path)}  {path.relative_to(bundle).as_posix()}\n"
            for path in sorted(bundle.rglob("*"))
            if path.is_file()
        )
    )
    return bundle


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
    changed = encode_media(root / "changed.mp4", video_change=True)
    bundle = _bundle(
        root / "bundle",
        [references / "same.mp4", changed, references / "reseeded.mp4"],
        seeds={"reseeded": 99},
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


@pytest.fixture(scope="module", autouse=True)
def _fixed_similarity():
    """Stands in for SSIM and the LPIPS network, whose weights torchvision downloads."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(media, "_frame_similarity", lambda: lambda x, y: (0.5, 0.25))
        yield


@pytest.fixture(scope="module")
def paired(tmp_path_factory):
    root = tmp_path_factory.mktemp("paired")
    same, other = encode_media(root / "same.mp4"), encode_media(root / "other.mp4")
    changed = encode_media(root / "changed.mp4", video_change=True)
    lossless = _bundle(root / "lossless", [same, other, same])
    # A lossy deployment with the same setup in the baseline run is not a baseline.
    lossy_peer = _bundle(root / "lossy-peer", [other, other, other], optimizations=["teacache"])
    lossy = _bundle(
        root / "lossy",
        [same, changed, same],
        seeds={"reseeded": 99},
        server={"attention_backend": "subblock_sparse_attn"},
        optimizations=["subblock-sparse-attn"],
    )
    offload = _bundle(root / "offload", [same, other, same], server={"dit_cpu_offload": True})
    fp8 = _bundle(root / "fp8", [same, changed, same], precision="fp8", optimizations=["cache-dit"])
    shutil.make_archive(str(root / "lossless"), "zip", lossless)
    arguments = ["score-quality", f"7={lossy}", f"8={offload}", f"9={fp8}", "--baseline"]
    arguments += [f"3={root / 'lossless.zip'}", "--baseline", f"4={lossy_peer}"]
    arguments += ["--baseline-run-id", "40", "--run-id", "41", "--source-revision", REVISION]
    assert cli.main([*arguments, "--output", str(root / "out")]) == 0
    return json.loads((root / "out/quality.json").read_bytes())


def test_pairs_each_clip_with_the_same_case_of_the_lossless_deployment(paired):
    pairs = {pair["candidate"]["case_id"]: pair for pair in paired["pairs"]}
    assert paired["mode"] == "paired_lossless"
    assert sorted(pairs) == ["changed", "same"]
    assert {pair["candidate"]["artifact_id"] for pair in paired["pairs"]} == {7}
    assert {pair["baseline"]["artifact_id"] for pair in paired["pairs"]} == {3}
    same = pairs["same"]["comparison"]["metrics"]
    assert same["video_identical"] is True
    assert (same["video_psnr_db"], same["video_ssim"], same["video_lpips"]) == (None, 1.0, 0.0)
    changed = pairs["changed"]["comparison"]["metrics"]
    assert 0 < changed["video_psnr_db"] < 15
    assert (changed["video_ssim"], changed["video_lpips"]) == (0.5, 0.25)
    assert len(changed["video_ssim_frames"]) == len(changed["video_lpips_frames"]) == 8
    summary = next(item for item in paired["deployments"] if item["candidate"]["artifact_id"] == 7)
    assert summary["candidate"]["optimizations"] == ["subblock-sparse-attn"]
    assert summary["baseline"] == [{"artifact_id": 3, "cell_id": "c1"}]
    assert summary["summary"]["identical"] == 1
    assert summary["summary"]["video_psnr_db"]["worst"] == changed["video_psnr_db"]
    assert summary["summary"]["video_ssim"]["worst"] == changed["video_ssim"]
    assert summary["summary"]["video_lpips"]["worst"] == changed["video_lpips"]


def test_lists_clips_without_a_matched_lossless_clip_instead_of_scoring_them(paired):
    assert [
        (item["candidate"]["artifact_id"], item["candidate"]["case_id"], item["reason"])
        + (item["fields"],)
        for item in paired["unscored"]
    ] == [
        (7, "reseeded", "settings_mismatch", ["seed", "request_sha256"]),
        (8, "same", "no_baseline", []),
        (8, "changed", "no_baseline", []),
        (8, "reseeded", "no_baseline", []),
        (9, "same", "no_baseline", []),
        (9, "changed", "no_baseline", []),
        (9, "reseeded", "no_baseline", []),
    ]


def test_a_same_run_dispatch_pairs_two_deployments_but_never_one_with_itself(tmp_path):
    media = [encode_media(tmp_path / "same.mp4")] * 3
    bundles = [f"{artifact}={_bundle(tmp_path / str(artifact), media)}" for artifact in (3, 4)]
    arguments = ["score-quality", *bundles, "--baseline", bundles[0], "--baseline", bundles[1]]
    arguments += ["--run-id", "41", "--baseline-run-id", "41", "--source-revision", REVISION]
    assert cli.main([*arguments, "--output", str(tmp_path / "out")]) == 0
    quality = json.loads((tmp_path / "out/quality.json").read_bytes())
    assert [
        (pair["candidate"]["artifact_id"], pair["baseline"]["artifact_id"])
        for pair in quality["pairs"]
    ] == [(3, 4)] * 3 + [(4, 3)] * 3
