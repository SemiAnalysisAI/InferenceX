"""Test-only encoded media and synthetic timing fixtures.

The videos are real encoded media; generation latencies are explicit synthetic
controls. No model, API, GPU, or measured H3 result is involved.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any


FIXTURE_POLICY: dict[str, Any] = {
    "policy_id": "fixture-control-v1",
    "calibration_status": "fixture_control",
    "max_latency_increase_fraction": 0.10,
    "min_video_psnr_db": 40.0,
    "min_audio_spectral_cosine": 0.99,
    "max_audio_rms_ratio_error": 0.05,
}


def _canonical_digest(document: dict[str, Any]) -> str:
    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, document: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _encode_fixture(path: Path, *, case_index: int, defect: str | None = None) -> None:
    """Encode 2 seconds of moving geometry and independent stereo tones."""
    import av
    import numpy as np

    fps, frame_count, width, height, sample_rate = 24, 48, 320, 180, 32000
    if path.exists():
        raise FileExistsError(path)
    if defect == "corrupt":
        with path.open("xb") as handle:
            handle.write(b"VGBENCH DELIBERATELY CORRUPT MEDIA FIXTURE\n")
        return
    with av.open(str(path), mode="w", format="mp4") as container:
        video = container.add_stream("libx264", rate=fps)
        video.width, video.height, video.pix_fmt = width, height, "yuv420p"
        video.options = {"crf": "18", "preset": "ultrafast", "threads": "1"}
        audio = container.add_stream("aac", rate=sample_rate)
        audio.layout = "stereo"
        audio.bit_rate = 192000
        for index in range(frame_count):
            phase = 0 if defect == "frozen" else index
            rgb = np.empty((height, width, 3), dtype=np.uint8)
            rgb[:, :, 0] = np.linspace(20, 70, width, dtype=np.uint8)
            rgb[:, :, 1] = 26 + case_index * 8
            rgb[:, :, 2] = 48
            left = 12 + (phase * 5 + case_index * 17) % (width - 70)
            top = 50 + int(25 * math.sin(phase / 7))
            rgb[top:top + 46, left:left + 46] = (245, 185, 60)
            rgb[height - 16:height - 10, 8:8 + 6 * (phase + 1)] = (90, 180, 215)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, fps)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)

        sample_count = sample_rate * frame_count // fps
        time_axis = np.arange(sample_count, dtype=np.float64) / sample_rate
        # Different frequencies prevent stereo content collapsing to mono.
        left_pcm = 0.2 * np.sin(2 * math.pi * (440 + 40 * case_index) * time_axis)
        right_pcm = 0.2 * np.sin(2 * math.pi * (660 + 60 * case_index) * time_axis)
        pcm = np.stack((left_pcm, right_pcm)).astype(np.float32)
        if defect == "muted":
            pcm[:] = 0
        for offset in range(0, sample_count, 1024):
            frame = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(pcm[:, offset:offset + 1024]),
                format="fltp", layout="stereo",
            )
            frame.sample_rate = sample_rate
            frame.pts, frame.time_base = offset, Fraction(1, sample_rate)
            for packet in audio.encode(frame):
                container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)


def _fixture_plan() -> dict[str, Any]:
    return {
        "plan_id": "encoded-media-fixture-v1",
        "model_id": "fixture/no-model",
        "model_revision": "fixture-v1",
        "generation": {
            "duration_seconds": 2, "aspect_ratio": "16:9",
            "width": 320, "height": 180, "frame_count": 48, "fps": 24,
            "audio_sample_rate_hz": 32000, "audio_channels": 2,
        },
        "cases": [
            {
                "case_id": f"fixture-{index}",
                "prompt": f"Synthetic moving-square and stereo-tone control {index}; not model-generated.",
                "seed": index, "requires_motion": True, "requires_sound": True,
            }
            for index in (1, 2)
        ],
        "repetitions": 1, "warmup_runs": 0,
    }


def _fixture_run(
    output_dir: Path, name: str, plan: dict[str, Any], *,
    baseline_dir: Path | None = None, defect: str | None = None,
    latency_factor: float = 1.0,
) -> dict[str, Any]:
    from evaluator.mvp_media import analyze_media

    output_dir.mkdir(parents=False, exist_ok=False)
    media_dir = output_dir / "media"
    media_dir.mkdir()
    records: list[dict[str, Any]] = []
    expected = {
        **plan["generation"], "audio_required": True,
        "duration_tolerance_seconds": 0.08,
        "requires_motion": True, "requires_sound": True,
    }
    for index, case in enumerate(plan["cases"], 1):
        slot_id = f"measurement-r001-c{index:03d}"
        relative = Path("media") / f"{slot_id}.mp4"
        artifact = output_dir / relative
        if baseline_dir is not None and (defect is None or index != 1):
            shutil.copyfile(baseline_dir / relative, artifact)
        else:
            _encode_fixture(artifact, case_index=index, defect=defect)
        media = analyze_media(artifact, expected=expected)
        records.append({
            "slot_id": slot_id, "case_id": case["case_id"],
            "prompt": case["prompt"], "seed": case["seed"], "repetition": 1,
            "phase": "measurement", "status": "succeeded",
            "artifact_path": relative.as_posix(),
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "latency_seconds": (8.0 + 2.0 * index) * latency_factor,
            "media": media, "error": None,
        })
    configuration = {
        "runtime": "fixture-encoder", "runtime_revision": f"fixture-{name}",
        "hardware_label": "synthetic-timing-control-not-a-GPU",
        "model_id": "fixture/no-model", "model_revision": "fixture-v1",
        "endpoint": "fixture://no-network",
        "identity_verification": "fixture_control_not_a_model",
    }
    valid = sum(record["media"]["valid"] for record in records)
    wall_seconds = sum(record["latency_seconds"] for record in records)
    epoch = datetime(2000, 1, 1, tzinfo=timezone.utc)
    bundle = {
        "bundle_version": "0.1.0", "bundle_type": "mvp_run",
        "run_id": f"fixture-{name}", "plan_id": plan["plan_id"],
        "plan_sha256": _canonical_digest(plan), "plan": plan,
        "configuration": configuration,
        "configuration_sha256": _canonical_digest(configuration),
        "evidence_kind": "fixture",
        "started_at": epoch.isoformat(),
        "finished_at": (epoch + timedelta(seconds=wall_seconds)).isoformat(),
        "status": "complete" if valid == len(records) else "partial",
        "measurement": {
            "boundary": "submit_to_validated_media", "concurrency": 1,
            "warmup_runs": 0, "wall_seconds": wall_seconds,
            "timing_evidence": "synthetic_fixture_control_not_measured_generation",
        },
        "records": records,
        "summary": {
            "scheduled": len(records), "completed": len(records), "valid": valid,
            "failed": len(records) - valid, "failed_attempts": 0,
            "invalid_completed": len(records) - valid,
            "technical_success_rate": valid / len(records),
            "latency_median_seconds": wall_seconds / len(records),
            "valid_clips_per_second": valid / wall_seconds,
        },
        "limitations": [
            "Real encoded synthetic media; not generated by H3 or any other model.",
            "All latency values and timestamps are synthetic controls, not benchmark measurements.",
        ],
    }
    _write_json(output_dir / "plan.json", plan)
    _write_json(output_dir / "run.json", bundle)
    return bundle
