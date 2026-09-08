"""Real encoded fixtures: no decoder, filesystem, or signal-analysis mocks."""

from fractions import Fraction
import hashlib
import json
from pathlib import Path

import pytest

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")

from evaluator.mvp_media import analyze_media, compare_media


def encode_media(
    path: Path,
    *,
    width: int = 64,
    height: int = 48,
    fps: int = 10,
    frame_count: int = 8,
    audio_mode: str = "stereo",
    video_change: bool = False,
    frozen: bool = False,
    blank: bool = False,
    timestamp_offset_seconds: float = 0.0,
    audio_offset_seconds: float = 0.0,
    audio_duration_scale: float = 1.0,
    freeze_after_frame: int | None = None,
) -> Path:
    """Encode exact RGB+PCM in MKV, or browser-playable H.264+AAC in MP4.

    All media is artificial diagnostic material, never model output. MP4 is
    intentionally lossy; assertions requiring sample identity should use MKV.
    """
    is_mp4 = path.suffix.lower() == ".mp4"
    sample_rate = 24000
    samples = round(frame_count / fps * sample_rate * audio_duration_scale)
    phase = np.arange(samples, dtype=np.float64) / sample_rate
    left = 0.35 * np.sin(2 * np.pi * 440 * phase)
    right = 0.25 * np.sin(2 * np.pi * 730 * phase)
    if audio_mode == "silent":
        left[:], right[:] = 0, 0
    elif audio_mode == "silent_right":
        right[:] = 0
    elif audio_mode == "collapsed":
        right = left.copy()
    elif audio_mode == "loud":
        left[:], right[:] = 1, -1
    signal = np.stack([left, right])
    with av.open(str(path), "w") as container:
        video = container.add_stream("libx264" if is_mp4 else "ffv1", rate=fps)
        video.width, video.height = width, height
        video.pix_fmt = "yuv420p" if is_mp4 else "bgr0"
        if is_mp4:
            video.options = {"preset": "ultrafast", "crf": "18"}
        audio = None
        if audio_mode != "absent":
            audio = container.add_stream("aac" if is_mp4 else "pcm_s16le", rate=sample_rate)
            audio.layout = "stereo"
        for index in range(frame_count):
            pixels = np.zeros((height, width, 3), dtype=np.uint8)
            if not blank:
                pixels[:, :, 0] = 40
                pixels[:, :, 1] = np.arange(width, dtype=np.uint8)[None, :] * 3
                moving_index = min(index, freeze_after_frame) if freeze_after_frame is not None else index
                x = 4 if frozen else (4 + moving_index * 3) % (width - 12)
                pixels[8:24, x : x + 12] = [220, 90, 20]
                if video_change:
                    pixels[:, :, 2] = 160
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index + round(timestamp_offset_seconds * fps)
            frame.time_base = Fraction(1, fps)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        if audio is not None:
            for start in range(0, samples, 1024):
                chunk = signal[:, start : start + 1024]
                if is_mp4:
                    values, fmt = chunk.astype(np.float32), "fltp"
                else:
                    values = np.round(np.clip(chunk, -1, 32767 / 32768) * 32768).astype(np.int16).T.reshape(1, -1)
                    fmt = "s16"
                frame = av.AudioFrame.from_ndarray(values, format=fmt, layout="stereo")
                frame.sample_rate = sample_rate
                frame.pts = start + round((timestamp_offset_seconds + audio_offset_seconds) * sample_rate)
                frame.time_base = Fraction(1, sample_rate)
                for packet in audio.encode(frame):
                    container.mux(packet)
            for packet in audio.encode():
                container.mux(packet)
    return path


def expectation() -> dict:
    return {
        "width": 64,
        "height": 48,
        "frame_count": 8,
        "fps": 10,
        "duration_seconds": 0.8,
        "duration_tolerance_seconds": 0.005,
        "audio_required": True,
        "audio_sample_rate_hz": 24000,
        "audio_channels": 2,
        "requires_motion": True,
        "requires_sound": True,
    }


def test_full_decode_contract_and_channel_statistics(tmp_path):
    path = encode_media(tmp_path / "real.mkv")
    result = analyze_media(path, expectation())
    assert result["valid"], result
    assert result["decode_ok"]
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["byte_size"] == path.stat().st_size
    assert result["video"]["frame_count"] == 8
    assert result["video"]["fps"] == pytest.approx(10)
    assert result["video"]["duration_seconds"] == pytest.approx(0.8)
    assert result["audio"]["sample_count"] == 19200
    assert result["audio"]["channels"] == 2
    assert result["audio"]["rms_channels"] == pytest.approx([0.35 / 2**0.5, 0.25 / 2**0.5], abs=5e-5)
    assert result["audio"]["silent_channels"] == []
    assert result["metrics"]["av_start_skew_seconds"] == pytest.approx(0.0)
    assert result["metrics"]["av_end_skew_seconds"] == pytest.approx(0.0, abs=0.001)
    json.dumps(result, allow_nan=False)


def test_corrupt_file_fails_without_pretending_it_is_a_clip(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a media container\x00" * 20)
    result = analyze_media(path, expectation())
    assert not result["valid"]
    assert not result["decode_ok"]
    assert result["errors"]
    assert result["sha256"]
    assert result["video"]["present"] is False
    json.dumps(result, allow_nan=False)


def test_wrong_media_contract_fails(tmp_path):
    path = encode_media(tmp_path / "mismatch.mkv")
    expected = expectation() | {"width": 128, "frame_count": 9, "fps": 24, "duration_seconds": 5, "audio_channels": 1, "audio_sample_rate_hz": 48000}
    result = analyze_media(path, expected)
    assert result["decode_ok"]
    assert not result["valid"]
    failed = {item["name"] for item in result["checks"] if item["status"] == "failed"}
    assert {"video.width", "video.frame_count", "video.fps", "video.duration_seconds", "audio.channels", "audio.sample_rate_hz"} <= failed


def test_silence_freeze_and_blank_are_detected_without_quality_claims(tmp_path):
    path = encode_media(tmp_path / "silent-frozen.mkv", audio_mode="silent", blank=True)
    measured = analyze_media(path)
    assert measured["valid"]  # Silence/static scenes are not universal failures.
    assert measured["audio"]["rms_channels"] == [0.0, 0.0]
    assert measured["audio"]["rms_dbfs_channels"] == [None, None]
    assert measured["audio"]["silent_channels"] == [0, 1]
    assert measured["video"]["blank_fraction"] == 1.0
    assert measured["video"]["duplicate_fraction"] == 1.0
    assert measured["video"]["frozen_fraction"] == 1.0
    result = analyze_media(path, expectation())
    failed = {item["name"] for item in result["checks"] if item["status"] == "failed"}
    assert {"video.motion_presence", "audio.sound_presence"} <= failed
    json.dumps(result, allow_nan=False)


def test_channel_collapse_and_full_scale_are_reported_per_channel(tmp_path):
    collapsed = analyze_media(encode_media(tmp_path / "collapsed.mkv", audio_mode="collapsed"))
    assert collapsed["audio"]["identical_channel_pairs"] == [[0, 1]]
    partial_silence = analyze_media(encode_media(tmp_path / "right-silent.mkv", audio_mode="silent_right"))
    assert partial_silence["audio"]["silent_channels"] == [1]
    assert partial_silence["audio"]["rms_channels"][0] > 0.2
    loud = analyze_media(encode_media(tmp_path / "loud.mkv", audio_mode="loud"))
    assert loud["audio"]["clipping_fraction_channels"] == [1.0, 1.0]


def test_identical_pair_has_complete_coverage_and_no_infinity(tmp_path):
    baseline = encode_media(tmp_path / "baseline.mkv")
    candidate = tmp_path / "candidate.mkv"
    candidate.write_bytes(baseline.read_bytes())
    result = compare_media(baseline, candidate)
    assert result["compatible"], result
    metrics = result["metrics"]
    assert metrics["video_mae"] == 0.0
    assert metrics["video_identical"] is True
    assert metrics["video_psnr_db"] is None
    assert metrics["video_compared_frames"] == 8
    assert metrics["video_sample_coverage_fraction"] == 1.0
    assert metrics["audio_identical"] is True
    assert metrics["audio_waveform_mae"] == 0.0
    assert metrics["audio_rms_ratio_channels"] == [1.0, 1.0]
    assert metrics["audio_spectral_cosine_channels"] == pytest.approx([1.0, 1.0])
    assert metrics["audio_sample_coverage_fraction"] == 1.0
    json.dumps(result, allow_nan=False)


def test_changed_visual_content_and_silent_channel_are_measured(tmp_path):
    baseline = encode_media(tmp_path / "baseline.mkv")
    candidate = encode_media(tmp_path / "changed.mkv", video_change=True, audio_mode="silent_right")
    result = compare_media(baseline, candidate)
    assert result["compatible"], result
    metrics = result["metrics"]
    assert metrics["video_mae"] > 0.1
    assert metrics["video_psnr_db"] < 15
    assert metrics["video_identical"] is False
    assert metrics["audio_rms_ratio_channels"] == [1.0, 0.0]
    assert metrics["audio_spectral_cosine_channels"][0] == pytest.approx(1.0)
    assert metrics["audio_spectral_cosine_channels"][1] is None
    assert metrics["audio_spectral_cosine"] is None
    assert metrics["audio_newly_silent_channels"] == [1]
    assert metrics["audio_waveform_mae_channels"][1] > 0.1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("change", [{"width": 80}, {"frame_count": 7}, {"fps": 20}, {"timestamp_offset_seconds": 0.2}, {"audio_mode": "absent"}])
def test_incompatible_media_is_not_resized_trimmed_or_time_shifted(tmp_path, change):
    baseline = encode_media(tmp_path / "baseline.mkv")
    candidate = encode_media(tmp_path / "incompatible.mkv", **change)
    result = compare_media(baseline, candidate)
    assert not result["compatible"], result
    assert result["metrics"]["video_mae"] is None
    assert any(check["status"] == "failed" for check in result["checks"])


def test_audio_optional_and_identically_silent_pairs_are_explicit(tmp_path):
    absent = encode_media(tmp_path / "absent.mkv", audio_mode="absent")
    result = analyze_media(absent)
    assert result["valid"]
    assert result["audio"]["present"] is False
    assert result["metrics"]["av_end_skew_seconds"] is None
    assert not analyze_media(absent, {"audio_required": True})["valid"]
    no_audio_pair = compare_media(absent, absent)
    assert no_audio_pair["compatible"]
    assert no_audio_pair["metrics"]["audio_spectral_cosine"] is None
    silent = encode_media(tmp_path / "silent.mkv", audio_mode="silent")
    silent_pair = compare_media(silent, silent)
    assert silent_pair["compatible"]
    assert silent_pair["metrics"]["audio_identical"] is True
    assert silent_pair["metrics"]["audio_rms_ratio"] is None
    assert silent_pair["metrics"]["audio_spectral_cosine"] is None
    json.dumps(silent_pair, allow_nan=False)


def test_browser_playable_mp4_is_decoded(tmp_path):
    path = encode_media(tmp_path / "browser.mp4")
    result = analyze_media(path, expectation())
    assert result["valid"], result
    assert result["video"]["codec"] == "h264"
    assert result["audio"]["codec"] == "aac"


def test_tiny_analysis_deadline_fails_without_partial_success(tmp_path):
    path = encode_media(tmp_path / "deadline.mkv")
    result = analyze_media(path, {"timeout_seconds": 1e-12})
    assert not result["valid"]
    assert not result["decode_ok"]
    assert "TimeoutError" in result["errors"][0]


def test_optional_av_boundary_and_partial_freeze_gates_are_explicit(tmp_path):
    offset = encode_media(tmp_path / "offset-audio.mkv", audio_offset_seconds=0.2)
    unconstrained = analyze_media(offset)
    assert unconstrained["valid"]
    assert unconstrained["metrics"]["av_start_skew_seconds"] == pytest.approx(0.2)
    constrained = analyze_media(offset, {"max_av_start_skew_seconds": 0.05})
    assert not constrained["valid"]
    assert any(item["name"] == "av.start_skew_seconds" and item["status"] == "failed" for item in constrained["checks"])
    short_audio = encode_media(tmp_path / "short-audio.mkv", audio_duration_scale=0.5)
    assert analyze_media(short_audio)["valid"]
    constrained = analyze_media(short_audio, {"max_av_end_skew_seconds": 0.05})
    assert not constrained["valid"]
    assert constrained["metrics"]["av_end_skew_seconds"] == pytest.approx(-0.4, abs=0.001)
    partial_freeze = encode_media(tmp_path / "partial-freeze.mkv", freeze_after_frame=2)
    unconstrained = analyze_media(partial_freeze, {"requires_motion": True})
    assert unconstrained["valid"]
    assert unconstrained["video"]["frozen_fraction"] == pytest.approx(5 / 7)
    assert not analyze_media(partial_freeze, {"max_frozen_fraction": 0.2})["valid"]


@pytest.mark.parametrize("expected", [{"fps": float("nan")}, {"timeout_seconds": "oops"}, {"duration_tolerance_seconds": -1}, {"max_av_end_skew_seconds": float("inf")}, {"max_frozen_fraction": 1.1}, {"width": 4.5}, {"audio_required": "false"}])
def test_invalid_expectations_never_produce_nonfinite_or_accepted_results(tmp_path, expected):
    path = encode_media(tmp_path / "valid.mkv")
    report = analyze_media(path, expected)
    assert not report["valid"]
    assert not report["decode_ok"]
    assert report["errors"]
    json.dumps(report, allow_nan=False)
