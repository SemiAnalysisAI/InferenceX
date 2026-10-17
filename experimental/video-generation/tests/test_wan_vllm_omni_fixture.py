"""Wan on vLLM-Omni: the runner's request against what the pinned server accepted and returned."""

import json
import sys
from pathlib import Path

from evaluator import mvp_runner

SPECS = Path(__file__).resolve().parents[1] / "specs"
sys.path.insert(0, str(SPECS))
import h3_spec

FIXTURE = Path(__file__).parent / "fixtures/wan-vllm-omni-a8576ccb"
SPEC_PATH = SPECS / "wan.yaml"


def _shared_plan():
    return h3_spec.plan_for(h3_spec.load_spec(SPEC_PATH), "shared-v1")


def test_runner_builds_the_request_the_pinned_server_completed():
    recorded = json.loads((FIXTURE / "request.json").read_text())
    record = json.loads((FIXTURE / "server-record.json").read_text())
    plan = _shared_plan()
    assert (plan["plan_id"], h3_spec.plan_sha256(plan)) == (
        recorded["plan_id"],
        recorded["spec_plan_sha256"],
    )
    slots = mvp_runner.preview_plan(plan, runtime="vllm-omni")["slots"]
    request = next(
        s["request"]
        for s in slots
        if s["phase"] == "measurement" and s["case_id"] == recorded["case_id"]
    )
    assert request == recorded["request"]
    assert record["status"] == "completed"
    assert record["error"] is None
    assert record["model"] == request["model"]
    assert record["prompt"] == request["prompt"]
    assert record["num_frames"] == request["num_frames"] == 81
    assert record["fps"] == request["fps"] == 16
    assert record["duration_s"] == request["num_frames"] / request["fps"] == 5.0625


def test_completed_clip_matches_the_spec_geometry():
    generation = _shared_plan()["generation"]
    probe = json.loads((FIXTURE / "ffprobe.json").read_text())
    video = [s for s in probe["streams"] if s["codec_type"] == "video"]
    assert len(video) == 1
    assert not [s for s in probe["streams"] if s["codec_type"] == "audio"]
    assert (video[0]["width"], video[0]["height"]) == (
        generation["width"],
        generation["height"],
    )
    assert int(video[0]["nb_read_frames"]) == generation["frame_count"]
    assert video[0]["r_frame_rate"] == f"{generation['fps']}/1"
    expected = generation["frame_count"] / generation["fps"]
    assert abs(float(probe["format"]["duration"]) - expected) < 1 / generation["fps"]
