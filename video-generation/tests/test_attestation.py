import hashlib
import json
from pathlib import Path

import pytest

from evaluator.attestation import attest

# Sanitized excerpts of real SGLang server logs: startup, the client warmup and the first
# measured request, with denoise bars split by interleaved access-log lines.
LOG = Path(__file__).parent / "fixtures/sglang-h3-server.log"  # H3 on 4 B200
WAN_LOG = Path(__file__).parent / "fixtures/sglang-wan22-server.log"  # Wan2.2 on 4 H100
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
WAN_PLAN = {
    "generation": {
        "width": 832,
        "height": 480,
        "frame_count": 81,
        "fps": 16,
        "num_inference_steps": 40,
        "flow_shift": 12.0,
    }
}


def _records(warmup_job, measured_job):
    return [
        {"slot_id": "warmup-001", "job_id": warmup_job, "attempted": True, "seed": 11},
        {"slot_id": "measurement-r001-c001", "job_id": measured_job, "attempted": True, "seed": 11},
    ]


RECORDS = _records("6b0e85dd-c34b-448e-8f9e-f66139721dd3", "5f8fc4dc-811a-480f-a533-ba1007fa8ccc")
WAN_RECORDS = _records(
    "42abb7ff-66b0-4ba4-a989-18bdcc19208e", "f15ae171-54c3-4156-bd11-8005030cfa61"
)


def _failed(document):
    checks = document["server"]["checks"] + [
        check for request in document["requests"] for check in request["checks"]
    ]
    return {check["name"] for check in checks if check["status"] == "failed"}


@pytest.mark.parametrize(
    "log,plan,records",
    [(LOG, PLAN, RECORDS), (WAN_LOG, WAN_PLAN, WAN_RECORDS)],
    ids=["h3", "wan22"],
)
def test_sglang_log_attests_that_every_request_did_the_declared_work(log, plan, records):
    document = attest("sglang", log, plan, records, "bf16")

    assert (document["valid"], _failed(document)) == (True, set())
    assert document["server_log"]["sha256"] == hashlib.sha256(log.read_bytes()).hexdigest()
    assert [request["slot_id"] for request in document["requests"]] == [
        "warmup-001",
        "measurement-r001-c001",
    ]
    steps = plan["generation"]["num_inference_steps"]
    for request in document["requests"]:
        denoise = {check["name"]: check for check in request["checks"]}["request.denoise_steps"]
        assert denoise["observed"] == {"completed": steps, "totals": [steps]}
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
    ],
)
def test_attestation_fails_when_the_log_does_not_prove_the_declared_work(tmp_path, fault, failed):
    text = LOG.read_bytes().decode("utf-8")
    records = RECORDS
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
    log = tmp_path / "server.out"
    log.write_bytes(text.encode("utf-8"))

    document = attest("sglang", log, PLAN, records, "bf16")

    assert (document["valid"], _failed(document)) == (False, failed)


# Lines in the format sgl-project/sglang@297e54fd logs them when the optimization runs. Wan2.2
# has two transformers, and H3 masks its first segment's query blocks.
@pytest.mark.parametrize(
    "clean,optimization,line",
    [
        (
            (LOG, PLAN, RECORDS),
            "cache-dit",
            "Enabling cache-dit on transformer with config: Fn=1, Bn=0, W=4, R=0.04, MC=1, "
            "DMD=False (history=6, rank=0, svd=medium), TaylorSeer=False (order=1), steps=49",
        ),
        (
            (WAN_LOG, WAN_PLAN, WAN_RECORDS),
            "cache-dit",
            "Enabling cache-dit on wan2.2 dual transformers with BlockAdapter",
        ),
        (
            (LOG, PLAN, RECORDS),
            "subblock-sparse-attn",
            "SubBlock sparse attention active: Sq=40320 Sk=40320 heads=24 keeping 74/295 key "
            "blocks per query block (sparsity 0.7492)",
        ),
        (
            (LOG, PLAN, RECORDS),
            "subblock-sparse-attn",
            "SubBlock heterogeneous BSA active: Sq=40320 Sk=40320 heads=24; selected query "
            "blocks keep 74/295 key blocks and unselected query blocks are dense",
        ),
    ],
    ids=["h3-cache-dit", "wan22-dual-cache-dit", "subblock", "subblock-heterogeneous"],
)
def test_attestation_fails_when_the_log_shows_a_lossy_optimization(
    tmp_path, clean, optimization, line
):
    source, plan, records = clean
    log = tmp_path / "server.out"
    log.write_bytes(source.read_bytes() + f"[11-04 11:00:02] {line}\n".encode())

    document = attest("sglang", log, plan, records, "bf16")

    assert (document["valid"], _failed(document)) == (False, {"server.optimizations"})
    [check] = [c for c in document["server"]["checks"] if c["name"] == "server.optimizations"]
    assert check["observed"] == [optimization]
