"""A sweep's job records join into one JSON Lines file, reconciled with the run's jobs."""

import json

from infx.launch.event import JobEvent
from infx.results.collect_events import main


def test_every_readable_job_record_becomes_one_line_in_artifact_order(tmp_path, capsys):
    artifacts = tmp_path / "job_events"
    records = {
        "job_event_b": '{\n  "result_filename": "b",\n  "outcome": "failure"\n}\n',
        "job_event_a": '{"result_filename": "a", "outcome": "success"}',
        "job_event_c": '{"result_filename": "c", "outco',
        "job_event_d": '["not", "a", "record"]',
    }
    for artifact, text in records.items():
        (artifacts / artifact).mkdir(parents=True)
        (artifacts / artifact / "job_event.json").write_text(text)
    (artifacts / "job_event_a" / "point.json").write_text('{"not": "a job record"}')
    output = tmp_path / "events.jsonl"

    assert main([str(artifacts), str(output)]) == 0

    assert output.read_text() == (
        '{"result_filename":"a","outcome":"success"}\n'
        '{"result_filename":"b","outcome":"failure"}\n'
    )
    warnings = capsys.readouterr().err
    assert "job_event_c" in warnings and "job_event_d" in warnings


RUN = "https://github.com/o/r/actions/runs/77"


def job(job_id: int, name: str, conclusion: str, steps: list[tuple[str, str]], **fields) -> dict:
    return {
        "id": job_id, "run_id": 77, "run_attempt": 1, "name": name, "conclusion": conclusion,
        "html_url": f"{RUN}/job/{job_id}", "started_at": "2026-10-16T10:00:00Z",
        "completed_at": "2026-10-16T10:05:00Z", "labels": ["self-hosted", "cluster:h200-dgxc"],
        "steps": [{"name": step, "conclusion": result} for step, result in steps], **fields,
    }  # fmt: skip


def test_jobs_without_a_record_get_one_and_recorded_jobs_gain_their_conclusion(tmp_path, monkeypatch):
    jobs = [
        job(1, "get-jobs", "success", [("Set up job", "success")]),
        job(2, "single-node 1k1k / (img...) / p5.000 | dsr1 fp8 cluster:h200-dgxc sgl TP8    c4",
            "failure", [("Set up job", "success"), ("Launch job script", "failure"),
                        ("Process result", "failure")], runner_name="h200-dgxc_01"),
        job(3, "multi-node 8k1k / (img...) / klaud | p1.392 | dsr1 fp4 gb200 dynamo-trt 1P (TP8) "
            "x 3D (TP8) mtp   c6x12", "failure",
            [("Set up job", "success"), ("Run actions/checkout@3d3c", "failure"),
             ("Retry checkout", "failure")], runner_name="gb200-nv_02"),
        job(4, "eval / (img...) / p1.050 | dsr1 fp8 mi355x atom TP8    c64 | eval-only", "cancelled",
            [], runner_name=None),
        job(5, "agentic / (img...) / p2.000 | kimik3 fp4 b200 vllm TP8   c16", "skipped", []),
    ]  # fmt: skip
    payload = tmp_path / "jobs.json"
    payload.write_text(json.dumps([{"total_count": len(jobs), "jobs": jobs}]))
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "gh").write_text(f"#!/bin/bash\ncat {payload}\n")
    (binaries / "gh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{binaries}:/usr/bin:/bin")
    monkeypatch.setenv("GH_TOKEN", "fixture-token")
    artifacts = tmp_path / "job_events" / "job_event_point"
    artifacts.mkdir(parents=True)
    recorded = {"result_filename": "point", "outcome": "failure", "github": {"job_id": 2}}
    (artifacts / "job_event.json").write_text(json.dumps(recorded))
    output = tmp_path / "events.jsonl"

    assert main([str(tmp_path / "job_events"), str(output), "--run", "https://github.com", "o/r", "77"]) == 0

    point, checkout, cancelled = [json.loads(line) for line in output.read_text().splitlines()]
    assert point["github"] == {
        "job_id": 2, "job_url": f"{RUN}/job/2", "job_name": jobs[1]["name"], "conclusion": "failure",
        "failed_step": "Launch job script",
    }  # fmt: skip
    assert {key: checkout[key] for key in ("model_prefix", "precision", "multinode", "conc_list", "runner",
                                           "cluster", "outcome", "duration_s")} == {
        "model_prefix": "dsr1", "precision": "fp4", "multinode": True, "conc_list": [6, 12],
        "runner": "gb200-nv_02", "cluster": None, "outcome": "failure", "duration_s": 300.0,
    }  # fmt: skip
    assert checkout["error"] == {
        "stage": "Run actions/checkout@3d3c", "type": "NoLauncherRecord",
        "message": "job concluded failure at step 'Run actions/checkout@3d3c' and left no job_event.json",
        "exit_code": None, "retriable": None, "evidence": f"{RUN}/job/3",
    }  # fmt: skip
    assert (cancelled["conc"], cancelled["eval_only"], cancelled["error"]["stage"]) == (64, True, "runner")
    for record in (checkout, cancelled):
        assert JobEvent.model_validate(record).github.run_url == RUN
