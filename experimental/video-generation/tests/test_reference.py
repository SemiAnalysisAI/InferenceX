import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Never

import av
import numpy as np
import pytest
from evaluator import reference
from evaluator.reference import digest, save_media, select_cases
from specs import h3_spec


def test_reference_selection_preserves_seeds_and_separates_warmup() -> None:
    spec = h3_spec.load_spec(Path(__file__).parents[1] / "specs/h3-v0.1.yaml")
    plan = h3_spec.plan_for(spec, "release-v1")
    selected = select_cases(plan, ["all"])
    assert len(selected) == 21
    assert selected[0]["role"] == "warmup"
    assert len({case["prompt"] for case in selected}) == 21
    case = selected[1]
    assert select_cases(plan, [case["case_id"]]) == [case]
    assert case["seed"] == plan["cases"][0]["seed"]
    for invalid in (["missing"], ["all", case["case_id"]], [case["case_id"]] * 2):
        with pytest.raises(ValueError, match="distinct case IDs"):
            select_cases(plan, invalid)


def test_reference_retains_float_media_and_encodes_explicit_streams(tmp_path: Path) -> None:
    video = np.random.default_rng(7).random((12, 3, 32, 32), dtype=np.float32)
    audio = np.zeros((2, 16000), dtype=np.float32)
    generation = {"frame_count": 12, "width": 32, "height": 32, "fps": 24}
    encoding = {
        "video_codec": "libx264",
        "preset": "ultrafast",
        "crf": 18,
        "pixel_format": "yuv420p",
        "audio_codec": "aac",
        "audio_sample_rate_hz": 32000,
    }
    target = tmp_path / "clip"
    hashes = save_media(target, video, audio, generation, encoding, threads=1, audio_bitrate=128000)
    assert set(hashes) == {"video.npy", "audio.npy", "reference.mp4"}
    np.testing.assert_array_equal(np.load(target / "video.npy", allow_pickle=False), video)
    np.testing.assert_array_equal(np.load(target / "audio.npy", allow_pickle=False), audio)
    assert all(digest(target / name) == expected for name, expected in hashes.items())
    with av.open(str(target / "reference.mp4")) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert stream.pix_fmt == "yuv420p"
        assert stream.average_rate == 24
        assert len(list(container.decode(video=0))) == 12
    with av.open(str(target / "reference.mp4")) as container:
        assert container.streams.audio[0].codec_context.name == "aac"
        assert container.streams.audio[0].rate == 32000
        assert list(container.decode(audio=0))
    with pytest.raises(FileExistsError):
        save_media(target, video, audio, generation, encoding, threads=1, audio_bitrate=128000)


@pytest.mark.parametrize("invalid", ["shape", "nan", "range", "empty_audio"])
def test_reference_rejects_invalid_raw_outputs_before_creating_files(
    tmp_path: Path, invalid: str
) -> None:
    video = np.zeros((1, 3, 32, 32), dtype=np.float32)
    audio = np.zeros((2, 10), dtype=np.float32)
    if invalid == "shape":
        video = video[:, :, :, :-1]
    elif invalid == "nan":
        video[0, 0, 0, 0] = np.nan
    elif invalid == "range":
        video[0, 0, 0, 0] = 1.1
    else:
        audio = audio[:, :0]
    with pytest.raises(ValueError, match="reference"):
        save_media(
            tmp_path / "bad",
            video,
            audio,
            {"frame_count": 1, "height": 32, "width": 32},
            {},
            threads=1,
            audio_bitrate=128000,
        )
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    ("error", "status"),
    [(KeyboardInterrupt(), "interrupted"), (RuntimeError("load failed"), "failed")],
)
def test_reference_retains_startup_failures_without_attesting_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException, status: str
) -> None:
    plan = h3_spec.plan_for(
        h3_spec.load_spec(Path(__file__).parents[1] / "specs/h3-v0.1.yaml"), "release-v1"
    )
    snapshot = tmp_path / plan["model_revision"]
    snapshot.mkdir()
    (snapshot / "modular_model_index.json").write_text("{}")
    for name in reference.COMPONENTS:
        (snapshot / name).mkdir()

    def fail_load(*_args: object, **_kwargs: object) -> Never:
        raise error

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            ComponentsManager=lambda: None,
            ModularPipeline=SimpleNamespace(from_pretrained=fail_load),
        ),
    )
    monkeypatch.setattr(reference, "schedules", lambda _plan: {})
    args = SimpleNamespace(
        snapshot=snapshot,
        output=tmp_path / "output",
        dtype="bfloat16",
        device="cuda:0",
        offload_reserve="12GB",
        attention_backend="native",
        encoder_threads=1,
        audio_bitrate=128000,
    )
    with pytest.raises(type(error)):
        reference.generate(args, plan, select_cases(plan, ["ceramic-wheel"]), {})
    record = json.loads((args.output / "reference.json").read_text())
    assert record["status"] == status
    assert record["cases"] == []
    assert record["quality_status"] == "unjudged"
    assert record["model_snapshot"]["weights_verified_by_helper"] is False
    assert record["model_snapshot"]["metadata_sha256"]["modular_model_index.json"] == digest(
        snapshot / "modular_model_index.json"
    )
    assert (
        args.output / "SHA256SUMS"
    ).read_text() == f"{digest(args.output / 'reference.json')}  reference.json\n"
