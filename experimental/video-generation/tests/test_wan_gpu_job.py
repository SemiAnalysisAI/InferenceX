"""CPU-only Wan launch and evidence checks; fixture bytes are not model output."""

import time

import pytest

from evaluator import mvp_gpu_evidence as evidence
from evaluator import mvp_gpu_job as gpu
from test_mvp_gpu_job import spec  # noqa: F401
from test_mvp_serving_smoke import saved_single


@pytest.fixture
def wan_spec(spec):
    spec["plan"].update(
        model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        generation={
            "width": 832, "height": 480, "frame_count": 81, "fps": 16,
            "num_inference_steps": 40, "guidance_scale": 4.0,
            "guidance_scale_2": 3.0, "flow_shift": 12.0, "negative_prompt": "",
        },
    )
    spec["plan"]["cases"] = [{
        "case_id": "walking", "prompt": "A person walks across a park.", "seed": 11,
        "requires_motion": True, "requires_sound": False,
    }]
    spec["plan"]["repetitions"] = 4
    spec["serving"] = {"concurrency": 1}
    spec["policy"] = {"policy_id": "wan-unqualified-smoke", "calibration_status": "uncalibrated"}
    spec["server"] = {
        "ulysses_degree": 1, "tp_size": 1, "dit_cpu_offload": False, "dit_layerwise_offload": False,
        "text_encoder_cpu_offload": True, "vae_cpu_offload": False,
    }
    return spec


def test_wan_launch_uses_video_model_and_explicit_offload_controls(wan_spec, tmp_path):
    output = str(tmp_path / "job/supervisor/baseline/generated")
    command = gpu._server_argv(wan_spec, "baseline", output_path=output)
    assert command[3:] == [
        "serve", "--model-type", "diffusion", "--model-path", wan_spec["model"]["path"],
        "--model-id", "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "--revision", wan_spec["model"]["revision"], "--num-gpus", "1",
        "--ulysses-degree", "1", "--tp-size", "1", "--ring-degree", "1",
        "--enable-cfg-parallel", "false", "--host", "127.0.0.1",
        "--port", "30280", "--enable-torch-compile", "false", "--dit-cpu-offload", "false",
        "--dit-layerwise-offload", "false",
        "--text-encoder-cpu-offload", "true", "--vae-cpu-offload", "false",
        "--warmup-mode", "off", "--output-path", output,
    ]


def test_wan_preview_is_single_runtime_without_qualification(wan_spec):
    result = gpu.preview_gpu_job(wan_spec)
    assert set(result["commands"]) == {"baseline"}
    assert result["sequence"] == [
        "verify pinned files", "acquire UUID locks", "verify idle",
        "baseline startup/warmup/measure/cleanup",
    ]
    assert result["workload"]["total_requests"] == 5
    assert result["evidence_kind"] == "gpu_job_preview_no_execution"


@pytest.mark.parametrize("field,value", [
    ("calibration_status", "operator_calibrated"), ("policy_id", "  "),
    ("min_audio_spectral_cosine", 0.95),
])
def test_wan_does_not_accept_a_paired_quality_policy(wan_spec, field, value):
    wan_spec["policy"][field] = value
    with pytest.raises(ValueError, match="Wan.*policy"):
        gpu.validate_gpu_job(wan_spec)


@pytest.mark.parametrize("field,value", [
    ("encoder_parallel", "auto"), ("performance_mode", "speed"),
    ("layerwise_offload", {"components": ["dit"], "prefetch_size": 1, "resident_layers": 20}),
    ("attention_backend", "aiter"), ("vae_cpu_offload", "false"), ("dit_layerwise_offload", True),
])
def test_wan_server_rejects_unsupported_or_implicit_controls(wan_spec, field, value):
    wan_spec["server"][field] = value
    with pytest.raises(ValueError, match="server"):
        gpu.validate_gpu_job(wan_spec)


@pytest.mark.parametrize("mutation", ["no-serving", "amd", "h3-timing"])
def test_wan_rejects_unqualified_runtime_paths(wan_spec, mutation):
    if mutation == "no-serving":
        wan_spec.pop("serving")
    elif mutation == "amd":
        wan_spec["gpu_vendor"] = "amd"
        wan_spec["gpu_uuids"] = [value.removeprefix("GPU-") for value in wan_spec["gpu_uuids"]]
    else:
        wan_spec["server_timing"] = True
    with pytest.raises(ValueError, match="Wan"):
        gpu.validate_gpu_job(wan_spec)


@pytest.mark.parametrize("count,tp", [(3, 1), (2, 2)])
def test_wan_rejects_unsupported_attention_layouts(wan_spec, count, tp):
    wan_spec["gpu_uuids"] = [f"GPU-{index:08x}-abcd-abcd-abcd-123456789abc" for index in range(count)]
    wan_spec["server"].update(ulysses_degree=count, tp_size=tp)
    with pytest.raises(ValueError, match="Wan.*TP1"):
        gpu.validate_gpu_job(wan_spec)


def test_wan_paired_execution_is_rejected_before_creating_evidence(wan_spec, tmp_path):
    output = tmp_path / "paired"
    with pytest.raises(ValueError, match="Wan.*serving"):
        gpu.run_gpu_job(wan_spec, output)
    assert not output.exists()


def test_wan_execution_retains_authorization_barrier(wan_spec, tmp_path):
    wan_spec["authorization"]["compute_approved"] = False
    output = tmp_path / "unapproved"
    with pytest.raises(ValueError, match="approval"):
        gpu.run_gpu_job(wan_spec, output, serving_smoke=True)
    assert not output.exists()


def test_wan_evidence_is_model_bound_and_cannot_qualify_paired_jobs(wan_spec, tmp_path):
    directory = tmp_path / "saved-wan"
    receipt = saved_single(wan_spec, directory)
    receipt["evidence_kind"] = "controlled_video_gpu"
    run_path = directory / "baseline/run.json"
    run = gpu._read(run_path)
    run["evidence_kind"] = "live_video"
    gpu._write(run_path, run)
    receipt["roles"]["baseline"]["run_sha256"] = gpu._hash(run_path)
    gpu._write(directory / "gpu-job.json", receipt)
    verified = evidence.verify_measurement_job(directory, deadline=time.monotonic() + 5, serving_smoke=True)
    assert set(verified["runs"]) == {"baseline"}
    assert verified["comparison"] is None
    assert verified["receipt"]["ci_accepted"] is False
    assert verified["receipt"]["release_qualified"] is False
    with pytest.raises(ValueError, match="Wan.*serving"):
        evidence.verify_measurement_job(directory, deadline=time.monotonic() + 5)
    run["evidence_kind"] = "live_h3"
    gpu._write(run_path, run)
    receipt["roles"]["baseline"]["run_sha256"] = gpu._hash(run_path)
    gpu._write(directory / "gpu-job.json", receipt)
    with pytest.raises(ValueError, match="unsupported run evidence"):
        evidence.verify_measurement_job(directory, deadline=time.monotonic() + 5, serving_smoke=True)
