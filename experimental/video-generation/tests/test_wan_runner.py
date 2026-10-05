"""CPU transport/media acceptance; encoded geometry is not Wan generation."""

import copy
import json
from pathlib import Path

import pytest

from evaluator import mvp_runner as runner
from test_mvp_runner import fixture_server  # noqa: F401
from test_mvp_media import encode_media


@pytest.fixture
def wan_plan():
    return json.loads((Path(__file__).parents[1] / "mvp/wan22-serving.plan.json").read_text())


def test_wan_preview_preserves_explicit_request_controls(wan_plan):
    preview = runner.preview_plan(wan_plan)
    assert preview["measurement_count"] == 4
    assert preview["warmup_count"] == 1
    request = preview["slots"][0]["request"]
    assert request == {
        "model": runner.WAN_MODEL_ID, "prompt": wan_plan["cases"][0]["prompt"],
        "seed": 11, "width": 832, "height": 480, "num_frames": 81, "fps": 16,
        "num_inference_steps": 40, "guidance_scale": 4.0, "guidance_scale_2": 3.0,
        "flow_shift": 12.0, "negative_prompt": "",
    }


@pytest.mark.parametrize("change", [
    {"width": 833}, {"frame_count": 80}, {"guidance_scale_2": None},
    {"audio_flow_shift": 3}, {"duration_seconds": 5}, {"negative_prompt": None}, {"guidance_scale": 1}, {"flow_shift": 5},
])
def test_wan_rejects_ignored_or_rounded_controls(wan_plan, change):
    wan_plan["generation"].update(change)
    with pytest.raises(ValueError):
        runner.preview_plan(wan_plan)


def test_wan_rejects_audio_requirement_and_unreviewed_transport(wan_plan, tmp_path):
    with pytest.raises(ValueError, match="SGLang"):
        runner.preview_plan(wan_plan, runtime="vllm-omni")
    with pytest.raises(ValueError, match="SGLang"):
        runner.run_plan(wan_plan, tmp_path / "run", endpoint="http://127.0.0.1:1",
                        runtime="vllm-omni", runtime_revision="a" * 40,
                        hardware_label="CPU fixture", model_revision=wan_plan["model_revision"])
    assert not (tmp_path / "run").exists()
    wan_plan["cases"][0]["requires_sound"] = True
    with pytest.raises(ValueError, match="video-only"):
        runner.preview_plan(wan_plan)


@pytest.mark.parametrize("mode,audio_mode,valid", [("success", "absent", 1), ("success", "stereo", 0), ("job_failed", "absent", 0)])
def test_wan_http_delivery_uses_real_video_only_decode(wan_plan, tmp_path, monkeypatch, fixture_server, mode, audio_mode, valid):
    import test_mvp_runner

    plan = copy.deepcopy(wan_plan)
    plan["cases"] = plan["cases"][:1]
    plan["warmup_runs"] = 0
    plan["generation"].update(width=64, height=48)
    media = encode_media(tmp_path / "cpu-fixture.mp4", width=64, height=48,
                         fps=16, frame_count=81, audio_mode=audio_mode)
    monkeypatch.setattr(test_mvp_runner, "FIXTURE_BYTES", media.read_bytes())
    endpoint, state = fixture_server(mode=mode)
    result = runner.run_plan(plan, tmp_path / "run", endpoint=endpoint,
                             runtime_revision="a" * 40, hardware_label="CPU HTTP fixture; no GPU",
                             model_revision=plan["model_revision"], timeout_seconds=10,
                             serving_concurrency=1)
    assert result["summary"]["scheduled"] == 1
    assert result["summary"]["valid"] == valid
    assert result["configuration"]["model_id"] == runner.WAN_MODEL_ID
    assert result["configuration"]["capabilities"]["audio_generation"] == "unsupported"
    assert result["configuration"]["protocol_source"]["revision"] == "71de97b264b04dcd514cf904003028aefe9775c8"
    assert state["posts"][0]["guidance_scale_2"] == 3
    record = result["records"][0]
    assert record["expected_media"]["audio_required"] is False
    assert "audio_sample_rate_hz" not in record["expected_media"]
    if valid:
        assert record["media"]["valid"] is True
        assert record["media"]["audio"]["present"] is False
        assert record["sha256"]
    elif mode == "job_failed":
        assert record["artifact_path"] is None
    else:
        assert record["outcome"] == "invalid_media"
        assert any(check["name"] == "audio.absent" and check["status"] == "failed"
                   for check in record["media"]["checks"])
