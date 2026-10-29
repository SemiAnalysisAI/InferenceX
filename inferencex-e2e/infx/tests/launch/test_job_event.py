"""The job record: what a launch's environment, results and evidence log make of it."""

import json

import pytest

from infx.launch import event as job_event
from infx.launch.evidence import error_detail
from infx.launch.event import JobError, JobEvent, JobEventBuilder, Topology, annotation
from infx.results.agentic import build_result


def test_annotation_escapes_workflow_command_data_and_title():
    error = JobError(
        stage="batch:run,2", type="JobFailed", message="100% done\r\nthen: died, twice",
        exit_code=1, retriable=False, detail="OSError: disk 100% full\n--- last 1 lines ---\nbye",
    )  # fmt: skip
    assert annotation(error) == (
        "::error title=batch%3Arun%2C2::JobFailed: 100%25 done%0D%0Athen: died, twice"
        "%0AOSError: disk 100%25 full"
    )


def test_an_agentic_point_records_no_sequence_lengths_and_its_pinned_versions(tmp_path, monkeypatch):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("aiperf==0.13.0.dev20261027\ndatasets>=4.7.0\n")
    monkeypatch.setattr(job_event, "REQUIREMENTS", requirements)
    digest = "sha256:" + "ab" * 32
    env = {
        "SCENARIO_TYPE": "agentic-coding", "ISL": "0", "OSL": "0", "IS_MULTINODE": "true",
        "IMAGE": f"lmsysorg/sglang:dev@{digest}", "NODE_COUNT": "3", "PREFILL_NUM_WORKERS": "2",
        "PREFILL_TP": "4", "PREFILL_EP": "4", "PREFILL_DP_ATTN": "true", "DECODE_NUM_WORKERS": "1",
        "DECODE_TP": "8", "DECODE_EP": "8", "DECODE_DP_ATTN": "false", "PR_NUMBER": "3807",
        "GITHUB_EVENT_NAME": "pull_request",
    }  # fmt: skip
    record = JobEventBuilder.begin(env).event
    assert (record.isl, record.osl) == (None, None)
    assert (record.versions.agentx_harness, record.versions.image_digest) == (
        "0.13.0.dev20261027",
        digest,
    )
    assert record.topology.model_dump() == {
        "tp": None, "ep": None, "pp": None, "dp_attn": None, "gpu_count": None, "node_count": 3,
        "prefill": {"num_workers": 2, "tp": 4, "ep": 4, "dp_attn": True},
        "decode": {"num_workers": 1, "tp": 8, "ep": 8, "dp_attn": False},
    }  # fmt: skip
    assert (record.github.pr_number, record.github.event_name) == (3807, "pull_request")

    fixed = JobEventBuilder.begin({**env, "SCENARIO_TYPE": "fixed-seq-len", "ISL": "8192"}).event
    assert (fixed.isl, fixed.versions.agentx_harness) == (8192, None)


FIXED = {
    "max_concurrency": 4, "completed": 98, "num_prompts": 100, "request_throughput": 2.5,
    "output_throughput": 1000.0, "total_token_throughput": 1600.0, "median_ttft_ms": 120.0,
    "p99_ttft_ms": 480.0, "median_tpot_ms": 12.5,
}  # fmt: skip
MULTINODE = "pt_sweep_isl_1024_osl_1024_conc4_gpus_16_ctx_8_gen_8"


def agentx_aggregate() -> dict:
    """The AgentX aggregate of two requests on a TP2 deployment, with one error dropped."""

    def request(start: int, end: int, ttft: float, itl: float, isl: int, osl: int) -> dict:
        metrics = {
            "input_sequence_length": {"value": isl}, "output_sequence_length": {"value": osl},
            "time_to_first_token": {"value": ttft, "unit": "ms"},
            "request_latency": {"value": 900.0, "unit": "ms"},
            "inter_token_latency": {"value": itl, "unit": "ms"},
        }  # fmt: skip
        return {"metadata": {"request_start_ns": start, "request_end_ns": end}, "metrics": metrics}

    records = [request(1_000_000_000, 2_000_000_000, 30, 18, 100, 50),
               request(1_500_000_000, 3_000_000_000, 40, 20, 200, 150)]  # fmt: skip
    accounting = {
        "records_total": 3, "records_profiled": 3, "records_dropped_total": 1,
        "records_warmup_dropped": 0, "records_error_dropped": 1, "error_categories": {},
    }  # fmt: skip
    env = {"TP": "2", "CONC": "4", "KV_OFFLOADING": "none", "FRAMEWORK": "vllm"}
    return build_result(records, {}, {}, env, request_accounting=accounting)


@pytest.mark.parametrize(("files", "gpu_count", "expected"), [
    ({"pt.json": FIXED}, 8, dict(
        source="pt.json", conc=4, requests_completed=98, requests_failed=2, request_throughput=2.5,
        output_throughput=1000.0, total_throughput_per_gpu=200.0, ttft_p50_ms=120.0,
        ttft_p99_ms=480.0, tpot_p50_ms=12.5)),
    ({f"{MULTINODE}.json": {**FIXED, "benchmark_outcome": {"failed": 5}}, f"{MULTINODE}.pytorch.json": []},
     None, dict(source=f"{MULTINODE}.json", conc=4, requests_completed=98, requests_failed=5,
                request_throughput=2.5, output_throughput=1000.0, total_throughput_per_gpu=100.0,
                ttft_p50_ms=120.0, ttft_p99_ms=480.0, tpot_p50_ms=12.5)),
    ({"pt_conc4.json": agentx_aggregate()}, None, dict(
        source="pt_conc4.json", conc=4, requests_completed=2, requests_failed=1,
        request_throughput=1.0, output_throughput=100.0, total_throughput_per_gpu=125.0,
        ttft_p50_ms=35.0, ttft_p99_ms=None, tpot_p50_ms=19.0)),
])  # fmt: skip
def test_the_record_reports_each_benchmark_point_the_launch_produced(tmp_path, files, gpu_count, expected):
    (tmp_path / "pt_stale.json").write_text(json.dumps(FIXED))
    builder = JobEventBuilder(JobEvent(result_filename="pt", topology=Topology(gpu_count=gpu_count)), tmp_path)
    for name, content in files.items():
        (tmp_path / name).write_text(json.dumps(content))
    (tmp_path / "results_gsm8k.json").write_text(json.dumps(FIXED))

    assert [point.model_dump() for point in builder.finish(0).metrics] == [expected]


def test_error_detail_leads_with_the_most_specific_cause_and_ends_with_the_log_tail(tmp_path):
    log = tmp_path / "worker.log"
    log.write_text(
        "ERROR: health check failed, retrying\nUserWarning: TypedStorage is deprecated\n"
        "RuntimeError: worker exited\n\x1b[1mtorch.OutOfMemoryError: CUDA out of memory.\x1b[0m\n"
        + "".join(f"step {index}\n" for index in range(1, 400))
    )
    cause, heading, *tail = error_detail(log).split("\n")
    assert (cause, heading) == ("torch.OutOfMemoryError: CUDA out of memory.", "--- last 20 lines ---")
    assert tail == [f"step {index}" for index in range(380, 400)]
    log.write_text("benchmark complete\n")
    assert error_detail(log) is None
    assert error_detail(tmp_path / "missing.log") is None
