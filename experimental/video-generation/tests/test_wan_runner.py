"""CPU transport/media acceptance; encoded geometry is not Wan generation."""

import copy

import pytest
import test_mvp_runner
from test_mvp_media import encode_media

from evaluator import mvp_runner as runner

fixture_server = test_mvp_runner.fixture_server


@pytest.fixture
def wan_plan():
    return {
        "plan_id": "controlled-wan-request",
        "model_id": runner.WAN_MODEL_ID,
        "model_revision": "b" * 40,
        "generation": {
            "width": 64,
            "height": 48,
            "frame_count": 81,
            "fps": 16,
            "num_inference_steps": 40,
            "guidance_scale": 4.0,
            "guidance_scale_2": 3.0,
            "flow_shift": 12.0,
            "negative_prompt": "",
        },
        "cases": [
            {
                "case_id": f"motion-{index}",
                "prompt": "A person walks across a room.",
                "seed": 11 + index,
                "requires_motion": True,
                "requires_sound": False,
            }
            for index in range(4)
        ],
        "repetitions": 1,
        "warmup_runs": 1,
    }


@pytest.mark.parametrize("runtime", ["sglang", "vllm-omni"])
def test_wan_preview_preserves_explicit_request_controls(wan_plan, runtime):
    preview = runner.preview_plan(wan_plan, runtime=runtime)
    assert preview["measurement_count"] == 4
    assert preview["warmup_count"] == 1
    assert preview["slots"][0]["request"] == {
        "model": runner.WAN_MODEL_ID,
        "prompt": "A person walks across a room.",
        "seed": 11,
        "width": 64,
        "height": 48,
        "num_frames": 81,
        "fps": 16,
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
        "guidance_scale_2": 3.0,
        "flow_shift": 12.0,
        "negative_prompt": "",
    }


@pytest.mark.parametrize(
    "change",
    [
        {"width": 833},
        {"frame_count": 80},
        {"guidance_scale_2": None},
        {"audio_flow_shift": 3},
        {"duration_seconds": 5},
        {"negative_prompt": None},
        {"guidance_scale": 1},
        {"flow_shift": 5},
    ],
)
def test_wan_rejects_ignored_or_rounded_controls(wan_plan, change):
    wan_plan["generation"].update(change)
    with pytest.raises(ValueError):
        runner.preview_plan(wan_plan)


def test_wan_rejects_audio_requirement(wan_plan):
    wan_plan["cases"][0]["requires_sound"] = True
    with pytest.raises(ValueError, match="video-only"):
        runner.preview_plan(wan_plan)


def test_unknown_runtime_fails_before_creating_a_run(wan_plan, tmp_path):
    with pytest.raises(ValueError, match="sglang or vllm-omni"):
        runner.preview_plan(wan_plan, runtime="other")
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="sglang or vllm-omni"):
        runner.run_plan(
            wan_plan,
            output,
            endpoint="http://127.0.0.1:1",
            runtime="other",
            runtime_revision="a" * 40,
            hardware_label="CPU fixture",
            model_revision=wan_plan["model_revision"],
        )
    assert not output.exists()


@pytest.mark.parametrize("runtime", ["sglang", "vllm-omni"])
@pytest.mark.parametrize(
    "mode,audio_mode,valid",
    [("success", "absent", 1), ("success", "stereo", 0), ("job_failed", "absent", 0)],
)
def test_wan_http_delivery_uses_real_video_only_decode(
    wan_plan, tmp_path, monkeypatch, fixture_server, runtime, mode, audio_mode, valid
):
    plan = copy.deepcopy(wan_plan)
    plan["cases"] = plan["cases"][:1]
    plan["warmup_runs"] = 0
    media = encode_media(
        tmp_path / "cpu-fixture.mp4",
        width=64,
        height=48,
        fps=16,
        frame_count=81,
        audio_mode=audio_mode,
    )
    monkeypatch.setattr(test_mvp_runner, "FIXTURE_BYTES", media.read_bytes())
    endpoint, state = fixture_server(mode=mode)
    result = runner.run_plan(
        plan,
        tmp_path / "run",
        endpoint=endpoint,
        runtime=runtime,
        runtime_revision="a" * 40,
        hardware_label="CPU HTTP fixture; no GPU",
        model_revision=plan["model_revision"],
        timeout_seconds=10,
        serving_concurrency=1,
    )
    assert result["summary"]["scheduled"] == 1
    assert result["summary"]["valid"] == valid
    assert result["configuration"]["model_id"] == runner.WAN_MODEL_ID
    assert result["configuration"]["capabilities"]["audio_generation"] == "unsupported"
    assert result["configuration"]["protocol_source"]["repository"] == (
        "https://github.com/sgl-project/sglang"
        if runtime == "sglang"
        else "https://github.com/vllm-project/vllm-omni"
    )
    assert state["requests"][:2] == [
        ("POST", "/v1/videos"),
        ("GET", "/v1/videos/fixture-1"),
    ]
    expected_post = {
        "model": runner.WAN_MODEL_ID,
        "prompt": "A person walks across a room.",
        "seed": 11,
        "width": 64,
        "height": 48,
        "num_frames": 81,
        "fps": 16,
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
        "guidance_scale_2": 3.0,
        "flow_shift": 12.0,
        "negative_prompt": "",
    }
    if runtime == "vllm-omni":
        expected_post = {key: str(value) for key, value in expected_post.items()}
    assert state["posts"] == [expected_post]
    record = result["records"][0]
    assert record["expected_media"]["audio_required"] is False
    assert "audio_sample_rate_hz" not in record["expected_media"]
    if valid:
        assert state["requests"][-1] == ("GET", "/v1/videos/fixture-1/content")
        assert record["media"]["valid"] is True
        assert record["media"]["audio"]["present"] is False
        assert record["sha256"]
    elif mode == "job_failed":
        assert record["artifact_path"] is None
        assert all(not path.endswith("/content") for _, path in state["requests"])
    else:
        assert record["outcome"] == "invalid_media"
        assert any(
            check["name"] == "audio.absent" and check["status"] == "failed"
            for check in record["media"]["checks"]
        )
