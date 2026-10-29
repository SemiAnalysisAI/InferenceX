"""Bounded-memory, full-stream media integrity and paired signal measurements.

These are decoded-signal measurements, not generative quality, semantic accuracy,
or lip-sync metrics. No resizing, resampling, channel mixing, or time shifting is
performed. PyAV and NumPy are optional and imported only when a measurement runs.
"""

from __future__ import annotations

import hashlib
import math
import time
from itertools import zip_longest
from pathlib import Path
from typing import Any


IMPLEMENTATION_VERSION = "1.0.0"
SILENCE_AMPLITUDE = 1e-4  # -80 dBFS sample threshold, also used for RMS presence.
CLIPPING_AMPLITUDE = 1.0 - 1.0 / 32768.0
FROZEN_MAE = 1.0 / 1024.0
SPECTRAL_WINDOW_SAMPLES = 1024


def _libraries() -> tuple[Any, Any]:
    try:
        import av
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Media measurements require the 'mvp' extra: uv sync --extra mvp") from exc
    return av, np


def _check(name: str, passed: bool | None, observed: Any, expected: Any, detail: str) -> dict:
    return {
        "name": name,
        "status": "not_applicable" if passed is None else "passed" if passed else "failed",
        "observed": observed,
        "expected": expected,
        "detail": detail,
    }


def _deadline_check(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() > deadline:
        raise TimeoutError("media analysis exceeded its declared timeout_seconds")


def _time(frame: Any) -> float | None:
    if frame.pts is None or frame.time_base is None:
        return None
    return float(frame.pts * frame.time_base)


def _positive_number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


def _validate_expected(expected: dict) -> None:
    integer_keys = {"width", "height", "frame_count", "audio_sample_rate_hz", "audio_channels"}
    positive_keys = integer_keys | {"fps", "duration_seconds", "timeout_seconds"}
    nonnegative_keys = {"duration_tolerance_seconds", "max_av_start_skew_seconds", "max_av_end_skew_seconds", "max_frozen_fraction"}
    for key in (positive_keys | nonnegative_keys) & expected.keys():
        value = expected[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"expected.{key} must be a finite number, got {value!r}")
        outside_domain = value <= 0 if key in positive_keys else value < 0
        if outside_domain:
            raise ValueError(f"expected.{key} is outside its nonnegative/positive domain")
        if key in integer_keys and int(value) != value:
            raise ValueError(f"expected.{key} must be an integer")
        if key == "max_frozen_fraction" and value > 1:
            raise ValueError("expected.max_frozen_fraction must be <=1")
    for key in {"audio_required", "requires_motion", "requires_sound"} & expected.keys():
        if not isinstance(expected[key], bool):
            raise ValueError(f"expected.{key} must be a boolean")


def _audio_array(frame: Any, np: Any) -> Any:
    """Preserve channel order/rate; only unpack and normalize decoded PCM."""
    values = frame.to_ndarray()
    channels = len(frame.layout.channels)
    if frame.format.is_planar:
        values = values.reshape(channels, frame.samples)
    else:
        values = values.reshape(frame.samples, channels).T
    kind, bits = values.dtype.kind, values.dtype.itemsize * 8
    normalized = values.astype(np.float64)
    if kind == "u":
        midpoint = float(2 ** (bits - 1))
        normalized = (normalized - midpoint) / midpoint
    elif kind == "i":
        normalized /= float(2 ** (bits - 1))
    elif kind != "f":
        raise ValueError(f"unsupported decoded audio sample type: {values.dtype}")
    if not np.isfinite(normalized).all():
        raise ValueError("decoded audio contains non-finite samples")
    return normalized


class _Timing:
    def __init__(self) -> None:
        self.first: float | None = None
        self.last: float | None = None
        self.last_duration: float | None = None
        self.missing = 0
        self.nonincreasing = 0
        self.min_step: float | None = None
        self.max_step: float | None = None
        self.tick = 0.0
        self.hash = hashlib.sha256()

    def add(self, frame: Any, duration: float | None) -> None:
        timestamp = _time(frame)
        self.tick = max(self.tick, float(frame.time_base or 0))
        self.hash.update(f"{frame.pts}@{frame.time_base};{duration};".encode())
        if timestamp is None:
            self.missing += 1
            return
        if self.first is None:
            self.first = timestamp
        if self.last is not None:
            step = timestamp - self.last
            if step <= 0:
                self.nonincreasing += 1
            else:
                self.min_step = step if self.min_step is None else min(self.min_step, step)
                self.max_step = step if self.max_step is None else max(self.max_step, step)
        self.last, self.last_duration = timestamp, duration


def _video_analysis(container: Any, stream: Any, np: Any, deadline: float | None) -> dict:
    timing = _Timing()
    count = blank = black = white = duplicates = frozen = corrupt = 0
    previous = None
    first_shape = None
    geometry_stable = True
    adjacent_mae_sum = 0.0
    nominal_fps = _positive_number(stream.average_rate)
    last_duration_source = None
    pixel_formats: set[str] = set()
    for frame in container.decode(stream):
        _deadline_check(deadline)
        pixels = frame.to_ndarray(format="rgb24")
        pixel_formats.add(frame.format.name)
        shape = pixels.shape
        if first_shape is None:
            first_shape = shape
        geometry_stable = geometry_stable and shape == first_shape
        count += 1
        corrupt += int(bool(getattr(frame, "is_corrupt", False)))
        frame_duration = _positive_number(getattr(frame, "duration", None))
        duration = frame_duration * float(frame.time_base) if frame_duration and frame.time_base else None
        last_duration_source = "decoded_frame_duration" if duration else None
        timing.add(frame, duration)
        maximum, minimum = int(pixels.max()), int(pixels.min())
        black += int(maximum <= 3)
        white += int(minimum >= 252)
        blank += int(maximum <= 3 or minimum >= 252)
        if previous is not None and previous.shape == shape:
            exact = bool(np.array_equal(previous, pixels))
            duplicates += int(exact)
            absolute_sum = 0.0
            for row in range(0, shape[0], 64):
                difference = pixels[row : row + 64].astype(np.int16) - previous[row : row + 64]
                absolute_sum += float(np.abs(difference).sum(dtype=np.float64))
            mae = absolute_sum / (pixels.size * 255.0)
            adjacent_mae_sum += mae
            frozen += int(mae <= FROZEN_MAE)
        previous = pixels
    tail_duration = timing.last_duration
    if tail_duration is None and timing.min_step is not None and timing.max_step is not None:
        # A timestamp does not encode the final frame's display duration. Explicitly
        # label the estimate instead of substituting container duration silently.
        tail_duration = (timing.max_step + timing.min_step) / 2.0
        last_duration_source = "observed_frame_interval_estimate"
    if tail_duration is None and nominal_fps is not None:
        tail_duration = 1.0 / nominal_fps
        last_duration_source = "nominal_rate_tail_estimate"
    end = timing.last + tail_duration if timing.last is not None and tail_duration else None
    duration = end - timing.first if end is not None and timing.first is not None else None
    fps = None
    fps_source = None
    if count > 1 and timing.first is not None and timing.last is not None and timing.last > timing.first:
        fps = (count - 1) / (timing.last - timing.first)
        fps_source = "decoded_frame_timestamps"
    elif count == 1 and tail_duration:
        fps, fps_source = 1.0 / tail_duration, last_duration_source
    return {
        "present": True,
        "codec": stream.codec_context.name,
        "pixel_formats": sorted(pixel_formats),
        "width": first_shape[1] if first_shape else None,
        "height": first_shape[0] if first_shape else None,
        "geometry_stable": geometry_stable,
        "frame_count": count,
        "fps": fps,
        "fps_source": fps_source,
        "nominal_fps": nominal_fps,
        "duration_seconds": duration,
        "duration_source": last_duration_source,
        "start_time_seconds": timing.first,
        "end_time_seconds": end,
        "time_base_seconds": timing.tick,
        "missing_timestamps": timing.missing,
        "nonincreasing_timestamps": timing.nonincreasing,
        "min_frame_interval_seconds": timing.min_step,
        "max_frame_interval_seconds": timing.max_step,
        "timestamp_sha256": timing.hash.hexdigest(),
        "corrupt_frame_count": corrupt,
        "blank_fraction": blank / count if count else None,
        "black_fraction": black / count if count else None,
        "white_fraction": white / count if count else None,
        "duplicate_fraction": duplicates / (count - 1) if count > 1 else None,
        "frozen_fraction": frozen / (count - 1) if count > 1 else None,
        "mean_adjacent_frame_mae": adjacent_mae_sum / (count - 1) if count > 1 else None,
    }


def _audio_analysis(container: Any, stream: Any, np: Any, deadline: float | None) -> dict:
    timing = _Timing()
    sample_count = frame_count = corrupt = 0
    rate = channels = None
    names: list[str] = []
    sum_squares = sums = peaks = silent = clipped = cross = None
    stable = True
    previous_end = None
    max_gap = 0.0
    for frame in container.decode(stream):
        _deadline_check(deadline)
        values = _audio_array(frame, np)
        this_rate, this_channels = frame.sample_rate, values.shape[0]
        if not this_rate or frame.samples <= 0:
            raise ValueError("audio frame has no samples or sample rate")
        if rate is None:
            rate, channels = this_rate, this_channels
            names = [channel.name for channel in frame.layout.channels]
            sum_squares = np.zeros(channels, dtype=np.float64)
            sums = np.zeros(channels, dtype=np.float64)
            peaks = np.zeros(channels, dtype=np.float64)
            silent = np.zeros(channels, dtype=np.int64)
            clipped = np.zeros(channels, dtype=np.int64)
            cross = np.zeros((channels, channels), dtype=np.float64)
        if this_rate != rate or this_channels != channels or names != [channel.name for channel in frame.layout.channels]:
            stable = False
            raise ValueError("audio sample rate or channel layout changes inside the stream")
        timestamp = _time(frame)
        if timestamp is not None and previous_end is not None:
            max_gap = max(max_gap, abs(timestamp - previous_end))
        duration = frame.samples / rate
        previous_end = timestamp + duration if timestamp is not None else None
        timing.add(frame, duration)
        frame_count += 1
        sample_count += frame.samples
        corrupt += int(bool(getattr(frame, "is_corrupt", False)))
        absolute = np.abs(values)
        sums += values.sum(axis=1)
        sum_squares += (values * values).sum(axis=1)
        peaks = np.maximum(peaks, absolute.max(axis=1))
        silent += (absolute <= SILENCE_AMPLITUDE).sum(axis=1)
        clipped += (absolute >= CLIPPING_AMPLITUDE).sum(axis=1)
        cross += values @ values.T
        if not np.isfinite(sum_squares).all() or not np.isfinite(sums).all() or not np.isfinite(cross).all():
            raise ValueError("decoded audio overflows finite signal-statistic accumulation")
    rms = np.sqrt(sum_squares / sample_count) if sample_count else np.array([])
    identical_pairs = []
    if sample_count:
        for left in range(channels):
            for right in range(left + 1, channels):
                residual = max(0.0, float(sum_squares[left] + sum_squares[right] - 2 * cross[left, right]))
                if residual / sample_count <= 1e-16:
                    identical_pairs.append([left, right])
    end = timing.last + timing.last_duration if timing.last is not None and timing.last_duration else None
    return {
        "present": True,
        "codec": stream.codec_context.name,
        "sample_rate_hz": rate,
        "channels": channels,
        "channel_names": names,
        "format_stable": stable,
        "frame_count": frame_count,
        "sample_count": sample_count,
        "duration_seconds": sample_count / rate if rate else None,
        "timeline_duration_seconds": end - timing.first if end is not None and timing.first is not None else None,
        "start_time_seconds": timing.first,
        "end_time_seconds": end,
        "time_base_seconds": timing.tick,
        "missing_timestamps": timing.missing,
        "nonincreasing_timestamps": timing.nonincreasing,
        "max_timestamp_gap_seconds": max_gap,
        "timestamp_sha256": timing.hash.hexdigest(),
        "corrupt_frame_count": corrupt,
        "rms_channels": rms.tolist(),
        "rms_dbfs_channels": [20 * math.log10(float(value)) if value > 0 else None for value in rms],
        "peak_channels": peaks.tolist() if sample_count else [],
        "dc_offset_channels": (sums / sample_count).tolist() if sample_count else [],
        "clipping_fraction_channels": (clipped / sample_count).tolist() if sample_count else [],
        "silence_fraction_channels": (silent / sample_count).tolist() if sample_count else [],
        "silent_channels": [index for index, value in enumerate(rms) if value <= SILENCE_AMPLITUDE],
        "identical_channel_pairs": identical_pairs,
    }


def analyze_media(path: Path, expected: dict | None = None) -> dict:
    """Decode a local clip fully and validate only explicitly supplied expectations.

    ``duration_seconds`` is expected *media* duration, not the provider's rounded
    request duration. A caller must resolve those differences explicitly. Motion
    and sound presence are technical proxies only. A timeout is checked between
    native decode calls, not a hard interrupt of one native FFmpeg operation.
    """
    av, np = _libraries()
    expected = dict(expected or {})
    path = Path(path).expanduser().resolve()
    deadline = None
    report: dict[str, Any] = {
        "path": str(path), "sha256": None, "byte_size": None, "decode_ok": False,
        "video": {"present": False}, "audio": {"present": False},
        "checks": [], "valid": False, "errors": [], "metrics": {},
        "implementation": {
            "name": "vgbench.cpu_media", "version": IMPLEMENTATION_VERSION,
            "pyav_version": av.__version__, "numpy_version": np.__version__,
            "ffmpeg_libraries": {key: ".".join(map(str, value)) for key, value in av.library_versions.items()},
            "coverage": "all decoded video frames and all decoded audio samples",
            "video_comparison_color_space": "decoded RGB24; no spatial resizing",
            "silence_amplitude_threshold": SILENCE_AMPLITUDE,
            "clipping_amplitude_threshold": CLIPPING_AMPLITUDE,
            "frozen_adjacent_rgb_mae_threshold": FROZEN_MAE,
            "blank_definition": "all decoded RGB samples <=3 or all >=252 on the 0..255 scale",
            "timestamp_policy": "use decoded PTS; final video duration fallback is explicitly labeled",
            "timeout_policy": "monotonic checks between hash chunks and decoded frames; not a native-call interrupt",
            "non_claims": ["generative quality", "semantic accuracy", "perceptual fidelity", "lip synchronization"],
        },
    }
    try:
        _validate_expected(expected)
        timeout = expected.get("timeout_seconds")
        deadline = time.monotonic() + timeout if timeout else None
        if not path.is_file():
            raise ValueError("media path is not a regular file")
        initial = path.stat()
        report["byte_size"] = initial.st_size
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                _deadline_check(deadline)
                digest.update(chunk)
        report["sha256"] = digest.hexdigest()
        with av.open(str(path), mode="r") as container:
            video_streams = list(container.streams.video)
            audio_streams = list(container.streams.audio)
            if len(video_streams) != 1 or len(audio_streams) > 1:
                raise ValueError("MVP requires exactly one video stream and at most one audio stream; ambiguous tracks are not silently selected")
            report["video"] = _video_analysis(container, video_streams[0], np, deadline)
            has_audio = bool(audio_streams)
        # Reopen, not seek: video decoding consumed the demuxer to EOF.
        if has_audio:
            with av.open(str(path), mode="r") as container:
                report["audio"] = _audio_analysis(container, container.streams.audio[0], np, deadline)
        final = path.stat()
        if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
            raise ValueError("media file changed while it was being analyzed")
        report["decode_ok"] = True
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        # Invalid expectations must never reach numerical checks or serialize
        # non-finite caller values into a report that looks like a valid run.
        if report["byte_size"] is None:
            report["checks"].append(_check("media.preflight", False, report["errors"][-1], "valid contract and local file", "No media measurement was accepted."))
            return report

    checks = report["checks"]
    video, audio = report["video"], report["audio"]
    checks.append(_check("media.decode", report["decode_ok"], report["decode_ok"], True, "Full-stream local decoding completed without an exception."))
    checks.append(_check("video.present", video.get("frame_count", 0) > 0, video.get("frame_count", 0), ">0", "At least one video frame must be decoded."))
    if video.get("frame_count", 0):
        checks.append(_check("video.geometry_stable", video["geometry_stable"], video["geometry_stable"], True, "All decoded frames retain the original geometry."))
        timestamp_ok = video["missing_timestamps"] == 0 and video["nonincreasing_timestamps"] == 0
        checks.append(_check("video.timestamps", timestamp_ok, {"missing": video["missing_timestamps"], "nonincreasing": video["nonincreasing_timestamps"]}, {"missing": 0, "nonincreasing": 0}, "Every decoded frame has strictly increasing PTS."))
        checks.append(_check("video.corrupt_frames", video["corrupt_frame_count"] == 0, video["corrupt_frame_count"], 0, "Decoder corruption flags are not ignored."))
    for name, key in (("width", "width"), ("height", "height"), ("frame_count", "frame_count")):
        if key in expected:
            checks.append(_check(f"video.{name}", video.get(name) == expected[key], video.get(name), expected[key], "Exact expected decoded media contract."))
    if "fps" in expected:
        target = float(expected["fps"])
        actual = video.get("fps")
        tolerance = max(0.001, abs(target) * 0.001)
        checks.append(_check("video.fps", actual is not None and abs(actual - target) <= tolerance, actual, {"value": target, "absolute_tolerance": tolerance}, "FPS is derived from decoded frame timestamps; the tolerance covers container timestamp quantization."))
        if video.get("frame_count", 0) > 1:
            interval = 1.0 / target
            cadence_tolerance = max(video.get("time_base_seconds", 0), interval * 0.001) + 1e-9
            intervals = [video.get("min_frame_interval_seconds"), video.get("max_frame_interval_seconds")]
            cadence_ok = all(value is not None and abs(value - interval) <= cadence_tolerance for value in intervals)
            checks.append(_check("video.cadence", cadence_ok, {"min_seconds": intervals[0], "max_seconds": intervals[1]}, {"interval_seconds": interval, "absolute_tolerance_seconds": cadence_tolerance}, "When a fixed FPS is requested, average FPS alone must not conceal internal gaps or bursty timestamps."))
    if "duration_seconds" in expected:
        target = float(expected["duration_seconds"])
        tolerance = float(expected.get("duration_tolerance_seconds", 1.0 / video["fps"] if video.get("fps") else 0.05))
        actual = video.get("duration_seconds")
        checks.append(_check("video.duration_seconds", actual is not None and tolerance >= 0 and abs(actual - target) <= tolerance + 1e-9, actual, {"value": target, "absolute_tolerance": tolerance}, "Caller supplies expected media duration, with any request-to-output rounding already resolved."))
    if "audio_required" in expected:
        required = bool(expected["audio_required"])
        checks.append(_check("audio.required", bool(audio.get("present")) if required else None, audio.get("present", False), required, "False means audio is optional, not prohibited."))
    if audio.get("present"):
        checks.append(_check("audio.samples", audio.get("sample_count", 0) > 0, audio.get("sample_count", 0), ">0", "An advertised audio stream must contain decoded samples."))
        timestamp_ok = audio["missing_timestamps"] == 0 and audio["nonincreasing_timestamps"] == 0
        checks.append(_check("audio.timestamps", timestamp_ok, {"missing": audio["missing_timestamps"], "nonincreasing": audio["nonincreasing_timestamps"]}, {"missing": 0, "nonincreasing": 0}, "Every audio frame has strictly increasing PTS."))
        tolerance = max(audio.get("time_base_seconds", 0), 1.0 / audio["sample_rate_hz"]) + 1e-9 if audio.get("sample_rate_hz") else 0
        checks.append(_check("audio.continuity", audio["max_timestamp_gap_seconds"] <= tolerance, audio["max_timestamp_gap_seconds"], {"maximum_seconds": tolerance}, "PCM sample continuity allows one container timestamp tick, without filling gaps or removing overlaps."))
        checks.append(_check("audio.corrupt_frames", audio["corrupt_frame_count"] == 0, audio["corrupt_frame_count"], 0, "Decoder corruption flags are not ignored."))
    for key, field in (("audio_sample_rate_hz", "sample_rate_hz"), ("audio_channels", "channels")):
        if key in expected:
            checks.append(_check(f"audio.{field}", audio.get(field) == expected[key], audio.get(field), expected[key], "Original decoded audio contract; no resampling or channel conversion."))
    if expected.get("requires_motion"):
        actual = video.get("mean_adjacent_frame_mae")
        checks.append(_check("video.motion_presence", actual is not None and actual > FROZEN_MAE, actual, {"greater_than": FROZEN_MAE}, "Adjacent RGB change is a technical motion-presence proxy, not action understanding or physical correctness."))
    if "max_frozen_fraction" in expected:
        actual = video.get("frozen_fraction")
        target = expected["max_frozen_fraction"]
        checks.append(_check("video.frozen_fraction", actual is not None and actual <= target, actual, {"maximum": target}, "Explicit caller-supplied limit on near-identical adjacent-frame transitions; no universal freeze threshold is assumed."))
    if expected.get("requires_sound"):
        actual = max(audio.get("rms_channels", []) or [0.0])
        checks.append(_check("audio.sound_presence", actual > SILENCE_AMPLITUDE, actual, {"greater_than": SILENCE_AMPLITUDE}, "At least one channel has non-silent RMS; intentionally silent channels are reported separately."))
    metrics = report["metrics"]
    for key in ("blank_fraction", "duplicate_fraction", "frozen_fraction", "mean_adjacent_frame_mae"):
        metrics[f"video_{key}"] = video.get(key)
    metrics["audio_rms_channels"] = audio.get("rms_channels", [])
    metrics["audio_silent_channel_count"] = len(audio.get("silent_channels", []))
    for boundary in ("start", "end"):
        video_time, audio_time = video.get(f"{boundary}_time_seconds"), audio.get(f"{boundary}_time_seconds")
        metrics[f"av_{boundary}_skew_seconds"] = audio_time - video_time if video_time is not None and audio_time is not None else None
        expectation_key = f"max_av_{boundary}_skew_seconds"
        if expectation_key in expected:
            actual = metrics[f"av_{boundary}_skew_seconds"]
            checks.append(_check(f"av.{boundary}_skew_seconds", actual is not None and abs(actual) <= expected[expectation_key] + 1e-9, actual, {"maximum_absolute_seconds": expected[expectation_key]}, "Explicit caller-supplied media-boundary limit, not semantic or lip synchronization."))
    report["valid"] = report["decode_ok"] and all(check["status"] != "failed" for check in checks)
    return report


def _decoded_video(path: Path, av: Any):
    with av.open(str(path), mode="r") as container:
        yield from container.decode(container.streams.video[0])


def _decoded_audio(path: Path, av: Any, np: Any):
    with av.open(str(path), mode="r") as container:
        for frame in container.decode(container.streams.audio[0]):
            yield _audio_array(frame, np), _time(frame), float(frame.time_base or 0)


def _compare_audio(baseline: Path, candidate: Path, av: Any, np: Any, channels: int, rate: int) -> dict:
    """Compare all samples despite harmless differences in codec frame chunking."""
    left_iter, right_iter = iter(_decoded_audio(baseline, av, np)), iter(_decoded_audio(candidate, av, np))
    left = right = None
    left_offset = right_offset = 0
    absolute = np.zeros(channels)
    squared_error = np.zeros(channels)
    energy_left = np.zeros(channels)
    energy_right = np.zeros(channels)
    spectral_dot = np.zeros(channels)
    spectral_left = np.zeros(channels)
    spectral_right = np.zeros(channels)
    pending_left = np.empty((channels, 0))
    pending_right = np.empty((channels, 0))
    window = np.hanning(SPECTRAL_WINDOW_SAMPLES)
    count = windows = 0
    max_timestamp_delta = 0.0

    def spectrum(a: Any, b: Any) -> None:
        nonlocal windows
        if a.shape[1] < SPECTRAL_WINDOW_SAMPLES:
            padding = ((0, 0), (0, SPECTRAL_WINDOW_SAMPLES - a.shape[1]))
            a, b = np.pad(a, padding), np.pad(b, padding)
        fa = np.abs(np.fft.rfft(a * window, axis=1))
        fb = np.abs(np.fft.rfft(b * window, axis=1))
        spectral_dot[:] += (fa * fb).sum(axis=1)
        spectral_left[:] += (fa * fa).sum(axis=1)
        spectral_right[:] += (fb * fb).sum(axis=1)
        windows += 1

    while True:
        if left is None or left_offset == left[0].shape[1]:
            left, left_offset = next(left_iter, None), 0
        if right is None or right_offset == right[0].shape[1]:
            right, right_offset = next(right_iter, None), 0
        if left is None or right is None:
            if left is not None or right is not None:
                raise ValueError("audio sample counts differ; no truncation is allowed")
            break
        if left[1] is None or right[1] is None:
            raise ValueError("audio timestamps are unavailable")
        delta = abs((left[1] + left_offset / rate) - (right[1] + right_offset / rate))
        max_timestamp_delta = max(max_timestamp_delta, delta)
        if delta > max(left[2], right[2], 1.0 / rate) + 1e-9:
            raise ValueError("paired audio samples are not timestamp-aligned")
        length = min(left[0].shape[1] - left_offset, right[0].shape[1] - right_offset, 65536)
        a = left[0][:, left_offset : left_offset + length]
        b = right[0][:, right_offset : right_offset + length]
        difference = a - b
        absolute += np.abs(difference).sum(axis=1)
        squared_error += (difference * difference).sum(axis=1)
        energy_left += (a * a).sum(axis=1)
        energy_right += (b * b).sum(axis=1)
        count += length
        left_offset += length
        right_offset += length
        pending_left = np.concatenate((pending_left, a), axis=1)
        pending_right = np.concatenate((pending_right, b), axis=1)
        cursor = 0
        while pending_left.shape[1] - cursor >= SPECTRAL_WINDOW_SAMPLES:
            spectrum(pending_left[:, cursor : cursor + SPECTRAL_WINDOW_SAMPLES], pending_right[:, cursor : cursor + SPECTRAL_WINDOW_SAMPLES])
            cursor += SPECTRAL_WINDOW_SAMPLES
        pending_left, pending_right = pending_left[:, cursor:].copy(), pending_right[:, cursor:].copy()
    if pending_left.shape[1]:
        spectrum(pending_left, pending_right)
    if count == 0:
        raise ValueError("paired audio contains no samples")
    ratios = [math.sqrt(float(b / a)) if a > 0 else None for a, b in zip(energy_left, energy_right)]
    cosines = [min(1.0, max(0.0, float(dot / math.sqrt(a * b)))) if a > 0 and b > 0 else None for dot, a, b in zip(spectral_dot, spectral_left, spectral_right)]
    return {
        "audio_channel_count": channels,
        "audio_compared_samples_per_channel": count,
        "audio_sample_coverage_fraction": 1.0,
        "audio_max_alignment_delta_seconds": max_timestamp_delta,
        "audio_waveform_mae": float(absolute.sum() / (count * channels)),
        "audio_waveform_mae_channels": (absolute / count).tolist(),
        "audio_waveform_rmse_channels": np.sqrt(squared_error / count).tolist(),
        "audio_identical": bool(np.all(squared_error == 0)),
        "audio_rms_ratio": math.sqrt(float(energy_right.sum() / energy_left.sum())) if energy_left.sum() > 0 else None,
        "audio_rms_ratio_channels": ratios,
        "audio_spectral_cosine": min(cosines) if all(value is not None for value in cosines) else None,
        "audio_spectral_cosine_channels": cosines,
        "audio_spectral_windows": windows,
        "audio_newly_silent_channels": [index for index, (a, b) in enumerate(zip(energy_left, energy_right)) if math.sqrt(float(a / count)) > SILENCE_AMPLITUDE and math.sqrt(float(b / count)) <= SILENCE_AMPLITUDE],
    }


def compare_media(baseline: Path, candidate: Path) -> dict:
    """Measure full-stream decoded fidelity only when geometry/timing agree.

    The caller chooses calibrated decision thresholds. Compatible means the
    measurements share shape and timing, not that the candidate is good quality.
    Any incompatibility leaves *all* quality-comparison metrics unavailable.
    """
    av, np = _libraries()
    baseline, candidate = Path(baseline), Path(candidate)
    result: dict[str, Any] = {
        "compatible": False,
        "metrics": {
            "video_mae": None, "video_psnr_db": None, "video_identical": None,
            "video_compared_frames": 0, "video_total_frames": 0, "video_sample_coverage_fraction": 0.0,
            "audio_waveform_mae": None, "audio_spectral_cosine": None, "audio_rms_ratio": None,
            "audio_rms_ratio_channels": [], "audio_spectral_cosine_channels": [], "audio_waveform_mae_channels": [],
            "audio_channel_count": 0,
        },
        "checks": [],
        "notes": [
            "Technical paired fidelity only: not semantic accuracy, generative quality, perceptual quality, or lip sync.",
            "All decoded RGB pixels/frames and PCM samples are compared; no resizing, resampling, mixing, trimming, or best-offset alignment.",
            "Video MAE uses RGB values divided by 255. PSNR uses peak=1 and full-stream MSE; exact identity gives null PSNR plus video_identical=true.",
            "Audio spectra use aligned non-overlapping 1024-sample Hann windows, zero-padding only the final partial window. Channel cosine is pooled over all window magnitudes; aggregate is the worst channel, or null if any channel has zero spectral energy.",
            "Audio RMS ratios are candidate/baseline. A zero baseline RMS has undefined ratio, reported null, never infinity.",
            "Timestamp alignment permits at most one native container tick (or one audio sample); the actual maximum delta is reported.",
        ],
    }
    left, right = analyze_media(baseline), analyze_media(candidate)
    checks = result["checks"]
    checks.append(_check("pair.valid_media", left["valid"] and right["valid"], {"baseline": left["valid"], "candidate": right["valid"]}, {"baseline": True, "candidate": True}, "Both original files must decode fully with valid media timing."))
    if not left["valid"] or not right["valid"]:
        result["notes"].extend(left["errors"] + right["errors"])
        return result
    for field in ("width", "height", "frame_count"):
        a, b = left["video"][field], right["video"][field]
        checks.append(_check(f"pair.video.{field}", a == b, {"baseline": a, "candidate": b}, "equal", "Original dimensions and complete decoded frame counts must match."))
    video_tick = max(left["video"]["time_base_seconds"], right["video"]["time_base_seconds"], 1e-9)
    for field in ("start_time_seconds", "end_time_seconds"):
        a, b = left["video"][field], right["video"][field]
        checks.append(_check(f"pair.video.{field}", a is not None and b is not None and abs(a - b) <= video_tick + 1e-9, {"baseline": a, "candidate": b}, {"maximum_delta_seconds": video_tick}, "No timestamp offset or duration trimming is allowed."))
    left_audio, right_audio = left["audio"]["present"], right["audio"]["present"]
    checks.append(_check("pair.audio.presence", left_audio == right_audio if left_audio or right_audio else None, {"baseline": left_audio, "candidate": right_audio}, "equal", "Both clips may omit audio; an audio stream may not disappear in only one clip."))
    if left_audio and right_audio:
        for field in ("sample_rate_hz", "channels", "channel_names", "sample_count"):
            a, b = left["audio"][field], right["audio"][field]
            checks.append(_check(f"pair.audio.{field}", a == b, {"baseline": a, "candidate": b}, "equal", "PCM channel order, sample rate, and complete sample counts must match."))
    if any(check["status"] == "failed" for check in checks):
        return result
    metrics = dict(result["metrics"])
    try:
        count = samples = 0
        absolute_sum = squared_sum = max_alignment_delta = 0.0
        for a, b in zip_longest(_decoded_video(baseline, av), _decoded_video(candidate, av)):
            if a is None or b is None:
                raise ValueError("video frame counts changed during comparison")
            delta = abs(_time(a) - _time(b))
            max_alignment_delta = max(max_alignment_delta, delta)
            if delta > video_tick + 1e-9:
                raise ValueError("video frame timestamps do not align; frame-index matching alone is insufficient")
            x, y = a.to_ndarray(format="rgb24"), b.to_ndarray(format="rgb24")
            if x.shape != y.shape:
                raise ValueError("video geometry changed during comparison")
            for row in range(0, x.shape[0], 64):
                difference = x[row : row + 64].astype(np.float64) - y[row : row + 64]
                absolute_sum += float(np.abs(difference).sum())
                squared_sum += float((difference * difference).sum())
            samples += x.size
            count += 1
        mse = squared_sum / (samples * 255.0 * 255.0)
        metrics.update({
            "video_mae": absolute_sum / (samples * 255.0),
            "video_psnr_db": -10.0 * math.log10(mse) if mse > 0 else None,
            "video_identical": squared_sum == 0,
            "video_compared_frames": count,
            "video_total_frames": left["video"]["frame_count"],
            "video_sample_coverage_fraction": 1.0,
            "video_max_alignment_delta_seconds": max_alignment_delta,
        })
        if left_audio and right_audio:
            metrics.update(_compare_audio(baseline, candidate, av, np, left["audio"]["channels"], left["audio"]["sample_rate_hz"]))
        checks.append(_check("pair.full_stream_alignment", True, True, True, "Every compared frame/sample met temporal compatibility; all streams were exhausted."))
        result["metrics"] = metrics
        result["compatible"] = True
    except Exception as exc:
        checks.append(_check("pair.full_stream_alignment", False, str(exc), "complete aligned streams", "Partial comparison metrics are discarded on any decoding, shape, or timing failure."))
        result["notes"].append(f"{type(exc).__name__}: {exc}")
    return result
