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
from specs import video_spec


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


def test_reference_rejects_invalid_raw_outputs_before_creating_files(tmp_path: Path) -> None:
    video = np.zeros((1, 3, 32, 32), dtype=np.float32)
    audio = np.zeros((2, 10), dtype=np.float32)
    video[0, 0, 0, 0] = np.nan
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


@pytest.mark.parametrize("selection", [["all"], ["../escape"]])
def test_every_selection_rejects_case_ids_that_leave_the_output(selection) -> None:
    plan = video_spec.plan_for(
        video_spec.load_spec(Path(__file__).parents[1] / "specs/h3-v2.yaml"), "release-v1"
    )
    plan["cases"][0]["case_id"] = "../escape"
    with pytest.raises(ValueError, match="plain directory names"):
        select_cases(plan, selection)


def test_reference_retains_startup_failures_without_attesting_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = video_spec.plan_for(
        video_spec.load_spec(Path(__file__).parents[1] / "specs/h3-v2.yaml"), "release-v1"
    )
    snapshot = tmp_path / plan["model_revision"]
    snapshot.mkdir()
    (snapshot / "modular_model_index.json").write_text("{}")
    for name in reference.COMPONENTS:
        (snapshot / name).mkdir()

    def fail_load(*_args: object, **_kwargs: object) -> Never:
        raise RuntimeError("load failed")

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
    with pytest.raises(RuntimeError):
        reference.generate(args, plan, select_cases(plan, ["ceramic-wheel"]), {})
    record = json.loads((args.output / "reference.json").read_text())
    assert record["status"] == "failed"
    assert record["cases"] == []
    assert record["quality_status"] == "unjudged"
    assert record["model_snapshot"]["weights_verified_by_helper"] is False
    assert record["model_snapshot"]["metadata_sha256"]["modular_model_index.json"] == digest(
        snapshot / "modular_model_index.json"
    )
    assert (
        args.output / "SHA256SUMS"
    ).read_text() == f"{digest(args.output / 'reference.json')}  reference.json\n"
