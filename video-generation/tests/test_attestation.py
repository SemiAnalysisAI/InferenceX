import hashlib
import json
from pathlib import Path

import pytest

from evaluator.attestation import attest

# A sanitized excerpt of a real B200 SGLang H3 server log: startup, the client warmup and
# the first measured request, with denoise bars split by interleaved access-log lines.
LOG = Path(__file__).parent / "fixtures/sglang-h3-server.log"
PLAN = {
    "generation": {
        "width": 1344,
        "height": 768,
        "frame_count": 124,
        "fps": 24,
        "num_inference_steps": 49,
        "flow_shift": 12.0,
    }
}
RECORDS = [
    {
        "slot_id": "warmup-001",
        "job_id": "6b0e85dd-c34b-448e-8f9e-f66139721dd3",
        "attempted": True,
        "seed": 11,
    },
    {
        "slot_id": "measurement-r001-c001",
        "job_id": "5f8fc4dc-811a-480f-a533-ba1007fa8ccc",
        "attempted": True,
        "seed": 11,
    },
]


def _failed(document):
    checks = document["server"]["checks"] + [
        check for request in document["requests"] for check in request["checks"]
    ]
    return {check["name"] for check in checks if check["status"] == "failed"}


def test_sglang_log_attests_that_every_request_did_the_declared_work():
    document = attest("sglang", LOG, PLAN, RECORDS, "bf16")

    assert (document["valid"], _failed(document)) == (True, set())
    assert document["server_log"]["sha256"] == hashlib.sha256(LOG.read_bytes()).hexdigest()
    assert [request["slot_id"] for request in document["requests"]] == [
        "warmup-001",
        "measurement-r001-c001",
    ]
    for request in document["requests"]:
        denoise = {check["name"]: check for check in request["checks"]}["request.denoise_steps"]
        assert denoise["observed"] == {"completed": 49, "totals": [49]}
    observed = document["server"]["observed"]
    assert observed["performance_mode"] == "speed" and observed["reduced_precision_numerics"]
    assert "203.0.113.1" not in json.dumps(document)


@pytest.mark.parametrize(
    "fault,failed",
    [
        ("step_cache", {"server.quality_features_off"}),
        ("sparse_attention", {"server.attention_backends"}),
        ("fewer_steps", {"request.parameters"}),
        ("stopped_early", {"request.denoise_steps"}),
        (
            "unserved_request",
            {"request.server_record", "request.parameters", "request.denoise_steps"},
        ),
        ("unknown_runtime", {"runtime.supported"}),
    ],
)
def test_attestation_fails_when_the_log_does_not_prove_the_declared_work(tmp_path, fault, failed):
    text = LOG.read_bytes().decode("utf-8")
    runtime, records = "sglang", RECORDS
    if fault == "step_cache":
        text = text.replace('"cache_dit_config": null', '"cache_dit_config": {"enabled": true}')
    elif fault == "sparse_attention":
        text = text.replace(
            "Attention backends for video_vae: torch_sdpa",
            "Attention backends for video_vae: self=video_sparse_attn_h3 (self-attention override)",
        )
    elif fault == "fewer_steps":
        text = text.replace("infer_steps: 49", "infer_steps: 25")
    elif fault == "stopped_early":
        text = text.replace("49/49 [", "48/49 [")
    elif fault == "unserved_request":
        records = [*RECORDS, {**RECORDS[1], "slot_id": "measurement-r001-c002", "job_id": "x"}]
    else:
        runtime = "tensorrt-llm"
    log = tmp_path / "server.out"
    log.write_bytes(text.encode("utf-8"))

    document = attest(runtime, log, PLAN, records, "bf16")

    assert (document["valid"], _failed(document)) == (False, failed)


# A sanitized real vLLM-Omni v0.30.0 H3 server log from 2x H100: startup and one measured
# frozen-c124 request (107 frames, 50 steps).
VLLM_LOG = Path(__file__).parent / "fixtures/vllm-omni-h3-server.log"
VLLM_PLAN = {"generation": {**PLAN["generation"], "frame_count": 107, "num_inference_steps": 50}}
VLLM_RECORDS = [
    {
        "slot_id": "measurement-r001-c001",
        "job_id": "video_gen_89c18f73f8304628b4a6482bc741ef7d",
        "attempted": True,
        "seed": 11,
    }
]


def _failed_server(document):
    return {check["name"] for check in document["server"]["checks"] if check["status"] == "failed"}


def test_vllm_omni_log_attests_its_server_but_ties_no_request_parameters_to_the_job():
    document = attest("vllm-omni", VLLM_LOG, VLLM_PLAN, VLLM_RECORDS, "bf16")

    assert (document["valid"], _failed(document)) == (
        False,
        {"request.parameters", "request.denoise_steps"},
    )
    assert document["server"]["observed"] == {"enforce_eager": True, "tensor_parallel_size": 2}
    [request] = document["requests"]
    checks = {check["name"]: check for check in request["checks"]}
    assert checks["request.server_record"]["status"] == "passed"
    assert checks["request.parameters"]["observed"] is None
    assert checks["request.denoise_steps"]["observed"] == {"completed": None, "totals": []}
    assert "<model-snapshot>" not in json.dumps(document)


@pytest.mark.parametrize(
    "fault,old,new,failed",
    [
        (
            "fp8_weights",
            "'tensor_parallel_size': 2}",
            "'tensor_parallel_size': 2, 'quantization': 'fp8'}",
            "server.quality_features_off",
        ),
        (
            "configured_attention",
            "for role='self' via platform default",
            "for role='self' via attention_config.default",
            "server.attention_backends",
        ),
        (
            "startup_lora",
            "static_lora_path=None",
            "static_lora_path=<adapter>",
            "server.no_lora",
        ),
        (
            "step_cache",
            "[RequestBatch] engine init",
            "Cache-dit enabled successfully on MiniMaxH3ModularPipeline\n[RequestBatch] engine init",
            "server.no_cache_or_quantization",
        ),
    ],
)
def test_vllm_omni_server_checks_fail_when_the_log_shows_changed_work(
    tmp_path, fault, old, new, failed
):
    text = VLLM_LOG.read_bytes().decode("utf-8")
    assert old in text, fault
    log = tmp_path / "server.out"
    log.write_bytes(text.replace(old, new, 1).encode("utf-8"))

    document = attest("vllm-omni", log, VLLM_PLAN, VLLM_RECORDS, "bf16")

    assert _failed_server(document) == {failed}
