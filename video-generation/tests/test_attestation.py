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


# Each optimization's lines in the format sgl-project/sglang@297e54fd logs them.
OPTIMIZATION_LINES = {
    "cache-dit": (
        "[11-04 11:00:02] Enabling cache-dit on transformer with config: Fn=1, Bn=0, W=4, "
        "R=0.04, MC=1, DMD=False (history=6, rank=0, svd=medium), TaylorSeer=False (order=1), "
        "steps=49\n"
    ),
    "subblock-sparse-attn": (
        "[11-04 11:00:02] Using subblock_sparse_attn attention backend\n"
        "[11-04 11:00:09] SubBlock sparse attention active: Sq=40320 Sk=40320 heads=24 "
        "keeping 74/295 key blocks per query block (sparsity 0.7492)\n"
    ),
}


def _failed(document):
    checks = document["server"]["checks"] + [
        check for request in document["requests"] for check in request["checks"]
    ]
    return {check["name"] for check in checks if check["status"] == "failed"}


def test_sglang_log_attests_that_every_request_did_the_declared_work():
    document = attest("sglang", LOG, PLAN, RECORDS, "bf16", [])

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
        ("undeclared_cache_dit", {"server.optimizations"}),
        ("declared_but_not_logged", {"server.optimizations"}),
    ],
)
def test_attestation_fails_when_the_log_does_not_prove_the_declared_work(tmp_path, fault, failed):
    text = LOG.read_bytes().decode("utf-8")
    runtime, records, optimizations = "sglang", RECORDS, []
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
    elif fault == "undeclared_cache_dit":
        text += OPTIMIZATION_LINES["cache-dit"]
    elif fault == "declared_but_not_logged":
        optimizations = ["subblock-sparse-attn"]
    else:
        runtime = "vllm-omni"
    log = tmp_path / "server.out"
    log.write_bytes(text.encode("utf-8"))

    document = attest(runtime, log, PLAN, records, "bf16", optimizations)

    assert (document["valid"], _failed(document)) == (False, failed)


@pytest.mark.parametrize("optimization", sorted(OPTIMIZATION_LINES))
def test_a_declared_optimization_attests_when_the_log_shows_it_running(tmp_path, optimization):
    log = tmp_path / "server.out"
    log.write_bytes(LOG.read_bytes() + OPTIMIZATION_LINES[optimization].encode("utf-8"))

    document = attest("sglang", log, PLAN, RECORDS, "bf16", [optimization])

    assert (document["valid"], _failed(document)) == (True, set())
