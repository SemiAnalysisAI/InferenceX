"""CPU transport/media acceptance; encoded geometry is not Wan generation."""

import sys
from pathlib import Path

import pytest
import test_mvp_runner
from test_mvp_media import encode_media

from evaluator import mvp_runner as runner

SPECS = Path(__file__).resolve().parents[1] / "specs"
sys.path.insert(0, str(SPECS))
import h3_spec

fixture_server = test_mvp_runner.fixture_server
plan = test_mvp_runner.plan
mocked_media = test_mvp_runner.mocked_media

SPEC_PATH = SPECS / "wan.yaml"
DRUM_PROMPT = "A single continuous medium shot of a drummer tapping a snare drum three times in a rehearsal room. The drumsticks visibly contact the drumhead on each tap. The camera remains still. No subtitles or scene cuts."
WAN_REQUEST = {
    "model": runner.WAN_MODEL_ID,
    "prompt": DRUM_PROMPT,
    "seed": 11,
    "width": 832,
    "height": 480,
    "num_frames": 81,
    "fps": 16,
    "num_inference_steps": 40,
    "guidance_scale": 4.0,
    "guidance_scale_2": 3.0,
    "flow_shift": 12.0,
    "negative_prompt": "",
}
SGLANG_SOURCE = {
    "repository": "https://github.com/sgl-project/sglang",
    "revision": "71de97b264b04dcd514cf904003028aefe9775c8",
    "files": [
        "python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py",
        "python/sglang/multimodal_gen/configs/sample/wan.py",
    ],
}
VLLM_OMNI_SOURCE = {
    "repository": "https://github.com/vllm-project/vllm-omni",
    "revision": "a8576ccb725c4e21cd13c3eb5f9a546b21149d2b",
    "files": [
        "vllm_omni/entrypoints/openai/api_server.py",
        "vllm_omni/entrypoints/openai/protocol/videos.py",
        "vllm_omni/entrypoints/openai/video/generation/helpers.py",
        "vllm_omni/entrypoints/openai/serving_video.py",
        "vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py",
    ],
}
# The CPU fixture shrinks the frame to 64x48; the sglang body is typed JSON, the vllm-omni body is multipart text.
FIXTURE_POSTS = {
    "sglang": {**WAN_REQUEST, "width": 64, "height": 48},
    "vllm-omni": {
        "model": runner.WAN_MODEL_ID,
        "prompt": DRUM_PROMPT,
        "seed": "11",
        "width": "64",
        "height": "48",
        "num_frames": "81",
        "fps": "16",
        "num_inference_steps": "40",
        "guidance_scale": "4.0",
        "guidance_scale_2": "3.0",
        "flow_shift": "12.0",
        "negative_prompt": "",
    },
}
PROTOCOL_SOURCES = {"sglang": SGLANG_SOURCE, "vllm-omni": VLLM_OMNI_SOURCE}
# Each checked_on_clip generation field and the clip check it produces. requires_motion adds
# video.motion_presence and the video-only model contract adds audio.absent; neither is a generation field.
CLIP_CHECKS = {
    "width": "video.width",
    "height": "video.height",
    "frame_count": "video.frame_count",
    "fps": "video.fps",
}
H3_PROTOCOL_REVISIONS = {
    "sglang": "253020450290328e9deb307eece1e402fa17f35e",
    "vllm-omni": "eb11446b7f2e30ca582f8aff3afe12e9a2e66f6c",
}


@pytest.fixture(scope="module")
def wan_spec():
    return h3_spec.load_spec(SPEC_PATH)


@pytest.fixture
def wan_plan(wan_spec):
    return h3_spec.plan_for(wan_spec, "shared-v1")


def run_wan(plan, output_dir, endpoint, runtime):
    return runner.run_plan(
        plan,
        output_dir,
        endpoint=endpoint,
        runtime=runtime,
        runtime_revision="a" * 40,
        hardware_label="CPU HTTP fixture; no GPU",
        model_revision=plan["model_revision"],
        timeout_seconds=10,
        serving_concurrency=1,
    )


def test_wan_sglang_request_is_unchanged_from_3733(wan_plan):
    preview = runner.preview_plan(wan_plan)
    assert preview["measurement_count"] == 8
    assert preview["warmup_count"] == 1
    assert preview["slots"][0]["request"] == WAN_REQUEST


@pytest.mark.parametrize(
    "change,match",
    [
        ({"width": 833}, "multiples of 16"),
        ({"height": 488}, "multiples of 16"),
        ({"frame_count": 80}, r"4n\+1"),
        ({"guidance_scale_2": None}, "guidance_scale_2"),
        ({"audio_flow_shift": 3}, "unsupported fields"),
        ({"negative_prompt": None}, "negative_prompt"),
        ({"guidance_scale": 1}, "exceed 1"),
        ({"flow_shift": 5}, "flow_shift 12"),
    ],
)
def test_wan_rejects_ignored_or_rounded_controls(wan_plan, change, match):
    wan_plan["generation"].update(change)
    with pytest.raises(ValueError, match=match):
        runner.preview_plan(wan_plan)


def test_wan_rejects_sound_requirement(wan_plan):
    wan_plan["cases"][0]["requires_sound"] = True
    with pytest.raises(ValueError, match="video-only"):
        runner.preview_plan(wan_plan)


@pytest.mark.parametrize("runtime", ["trt-llm", ""])
def test_unknown_runtimes_are_still_rejected(wan_plan, tmp_path, runtime):
    with pytest.raises(ValueError, match="sglang or vllm-omni"):
        runner.preview_plan(wan_plan, runtime=runtime)
    with pytest.raises(ValueError, match="sglang or vllm-omni"):
        run_wan(wan_plan, tmp_path / "run", "http://127.0.0.1:1", runtime)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "runtime,mode,audio_mode,valid",
    [
        ("sglang", "success", "absent", 1),
        ("vllm-omni", "success", "absent", 1),
        ("vllm-omni", "success", "stereo", 0),
        ("vllm-omni", "job_failed", "absent", 0),
    ],
)
def test_wan_http_delivery_uses_real_video_only_decode(
    wan_spec,
    wan_plan,
    tmp_path,
    monkeypatch,
    fixture_server,
    runtime,
    mode,
    audio_mode,
    valid,
):
    wan_plan["cases"] = wan_plan["cases"][:1]
    wan_plan["warmup_runs"] = 0
    wan_plan["generation"].update(width=64, height=48)
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
    result = run_wan(wan_plan, tmp_path / "run", endpoint, runtime)
    assert result["summary"]["scheduled"] == 1
    assert result["summary"]["valid"] == valid
    assert result["configuration"]["runtime"] == runtime
    assert result["configuration"]["model_id"] == runner.WAN_MODEL_ID
    assert result["configuration"]["capabilities"]["audio_generation"] == "unsupported"
    assert result["configuration"]["protocol_source"] == PROTOCOL_SOURCES[runtime]
    assert state["posts"] == [FIXTURE_POSTS[runtime]]
    record = result["records"][0]
    assert record["expected_media"]["audio_required"] is False
    assert "audio_sample_rate_hz" not in record["expected_media"]
    if valid:
        checked = {
            name
            for name, field in wan_spec["generation"].items()
            if field["checked_on_clip"]
        }
        assert checked == set(CLIP_CHECKS)
        passed = {
            check["name"]
            for check in record["media"]["checks"]
            if check["status"] == "passed"
        }
        assert (
            set(CLIP_CHECKS.values()) | {"video.motion_presence", "audio.absent"}
            <= passed
        )
        assert record["media"]["valid"] is True
        assert record["media"]["audio"]["present"] is False
        assert record["sha256"]
    elif mode == "job_failed":
        assert record["artifact_path"] is None
    else:
        assert record["outcome"] == "invalid_media"
        assert any(
            check["name"] == "audio.absent" and check["status"] == "failed"
            for check in record["media"]["checks"]
        )


@pytest.mark.parametrize("runtime", ["sglang", "vllm-omni"])
def test_h3_protocol_sources_survive_the_transport_table(
    plan, tmp_path, mocked_media, fixture_server, runtime
):
    endpoint, _ = fixture_server()
    run = test_mvp_runner.execute(plan, tmp_path / "run", endpoint, runtime=runtime)
    assert run["status"] == "complete"
    assert (
        run["configuration"]["protocol_source"]["revision"]
        == H3_PROTOCOL_REVISIONS[runtime]
    )
