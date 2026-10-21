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


def serve_jobs(tmp_path, monkeypatch, jobs: list[dict]) -> None:
    """Answer the run's job listing with ``jobs`` through a stub ``gh``."""
    payload = tmp_path / "jobs.json"
    payload.write_text(json.dumps([{"total_count": len(jobs), "jobs": jobs}]))
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "gh").write_text(f"#!/bin/bash\ncat {payload}\n")
    (binaries / "gh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{binaries}:/usr/bin:/bin")
    monkeypatch.setenv("GH_TOKEN", "fixture-token")


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
    serve_jobs(tmp_path, monkeypatch, jobs)
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


def test_a_job_that_fails_after_its_launcher_succeeded_fails_its_record_and_summary_row(tmp_path, monkeypatch):
    jobs = [
        job(21, "eval / (img...) / p5.000 | dsr1 fp8 cluster:h200-dgxc sgl TP8    c4 | eval-only", "failure",
            [("Launch job script", "success"), ("Verify eval scores", "failure")]),
        job(22, "single-node 1k1k / (img...) / p5.000 | dsr1 fp8 cluster:h200-dgxc sgl TP8    c8", "cancelled",
            [("Launch job script", "success"), ("Process result", "cancelled")]),
        job(23, "single-node 1k1k / (img...) / p5.000 | dsr1 fp8 cluster:h200-dgxc sgl TP8    c16", "timed_out",
            [("Launch job script", "success")]),
    ]  # fmt: skip
    serve_jobs(tmp_path, monkeypatch, jobs)
    kept = {"stage": "collect", "type": "ArtifactError", "message": "kept", "exit_code": None, "retriable": False}
    for name, job_id, error in (("a", 21, None), ("b", 22, kept), ("c", 23, None)):
        artifact = tmp_path / "job_events" / f"job_event_{name}"
        artifact.mkdir(parents=True)
        record = {"outcome": "success", "cluster": "h200-dgxc", "error": error, "github": {"job_id": job_id}}
        (artifact / "job_event.json").write_text(json.dumps(record))
    output, summary = tmp_path / "events.jsonl", tmp_path / "summary.md"

    assert main([str(tmp_path / "job_events"), str(output), "--run", "https://github.com", "o/r", "77",
                 "--summary", str(summary)]) == 0  # fmt: skip

    eval_scores, cancelled, timed_out = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["outcome"] for record in (eval_scores, cancelled, timed_out)] == ["failure"] * 3
    assert eval_scores["error"] == {
        "stage": "Verify eval scores", "type": "StepFailed",
        "message": "job concluded failure at step 'Verify eval scores' after the launcher succeeded",
        "exit_code": None, "retriable": False, "evidence": f"{RUN}/job/21",
    }  # fmt: skip
    assert cancelled["error"] == kept
    assert (timed_out["error"]["stage"], timed_out["error"]["message"]) == (
        "runner", "job concluded timed_out after the launcher succeeded")  # fmt: skip
    assert JobEvent.model_validate(timed_out).error.type == "StepFailed"
    assert summary.read_text(encoding="utf-8").splitlines() == [
        "**Job events:** 3 total, 0 succeeded, 3 failed, 0 synthesized",
        "",
        "| Cluster | Job | Stage | Error | Retriable | Cause | Link |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| h200-dgxc | dsr1 fp8 cluster:h200-dgxc sgl TP8    c4 \\| eval-only | Verify eval scores | StepFailed | no | "
        f"job concluded failure at step 'Verify eval scores' after the launcher succeeded | [job]({RUN}/job/21) |",
        f"| h200-dgxc | dsr1 fp8 cluster:h200-dgxc sgl TP8    c8 | collect | ArtifactError | no | kept | [job]({RUN}/job/22) |",
        f"| h200-dgxc | dsr1 fp8 cluster:h200-dgxc sgl TP8    c16 | runner | StepFailed | no | "
        f"job concluded timed_out after the launcher succeeded | [job]({RUN}/job/23) |",
    ]  # fmt: skip


def test_a_record_its_launcher_never_finished_fails_with_its_cancelled_timed_out_or_failed_job(tmp_path, monkeypatch):
    name = "single-node 1k1k / (img...) / p5.000 | dsr1 fp8 cluster:h200-dgxc sgl TP8    c{}"
    jobs = [
        job(31, name.format(1), "cancelled", [("Launch job script", "cancelled")]),
        job(32, name.format(2), "timed_out", [("Launch job script", "cancelled")]),
        job(33, name.format(4), "failure", [("Launch job script", "failure")]),
        job(34, name.format(8), "success", [("Launch job script", "success")]),
    ]  # fmt: skip
    serve_jobs(tmp_path, monkeypatch, jobs)
    recorded = {"stage": "run", "type": "JobNodeFail", "message": "lost", "exit_code": None, "retriable": True}
    for artifact_name, job_id, error in (("a", 31, None), ("b", 32, recorded), ("c", 33, None), ("d", 34, None)):
        artifact = tmp_path / "job_events" / f"job_event_{artifact_name}"
        artifact.mkdir(parents=True)
        record = {"outcome": None, "error": error, "stages": {"prepare": 4.5, "run": 0.0}, "github": {"job_id": job_id}}
        (artifact / "job_event.json").write_text(json.dumps(record))
    output = tmp_path / "events.jsonl"

    assert main([str(tmp_path / "job_events"), str(output), "--run", "https://github.com", "o/r", "77"]) == 0

    cancelled, timed_out, failed, succeeded = [json.loads(line) for line in output.read_text().splitlines()]
    assert (cancelled["outcome"], cancelled["stages"]) == ("failure", {"prepare": 4.5, "run": 0.0})
    assert cancelled["error"] == {
        "stage": "Launch job script", "type": "Interrupted",
        "message": "job concluded cancelled at step 'Launch job script' before the launcher finished",
        "exit_code": None, "retriable": True, "evidence": f"{RUN}/job/31",
    }  # fmt: skip
    assert (timed_out["outcome"], timed_out["error"]) == ("failure", recorded)
    assert (failed["outcome"], failed["error"]["type"], failed["error"]["retriable"]) == ("failure", "Interrupted", None)
    assert (succeeded["outcome"], succeeded["error"]) == (None, None)
    assert JobEvent.model_validate(cancelled).error.retriable is True


def test_summary_counts_the_jobs_and_lists_failures_with_retriable_ones_last(tmp_path, monkeypatch):
    jobs = [
        job(11, "single-node 1k1k / (img...) / p5.000 | dsr1 fp8 cluster:h200-dgxc sgl TP8    c4", "success", []),
        job(12, "single-node 8k1k / (img...) / p5.000 | dsr1 fp8 cluster:mi355x-amds atom TP8    c8", "failure", []),
        job(13, "eval / (img...) / klaud | p1.050 | kimik3 fp4 cluster:b200-nscale vllm TP8    c64 | eval-only",
            "failure", []),
        job(14, "single-node 1k1k / (img...) / p5.000 | qwen3.5 fp8 cluster:h100-dsxe sgl TP8/EP8    c20", "failure",
            []),
        job(15, "multi-node 8k1k / (img...) / p1.392 | dsr1 fp4 gb200 dynamo-trt 1P (TP8) x 3D (TP8) mtp   c6x12",
            "failure", [("Set up job", "success"), ("Run actions/checkout@3d3c", "failure")]),
    ]  # fmt: skip
    serve_jobs(tmp_path, monkeypatch, jobs)
    records = {
        "a": (11, {"outcome": "success", "cluster": "h200-dgxc", "error": None}),
        "b": (12, {"outcome": "failure", "cluster": "mi355x-amds", "error": {
            "stage": "run", "type": "JobNodeFail", "message": "Slurm job 5 ended with state=NODE_FAIL exit_code=0:0",
            "retriable": True, "detail": None}}),
        "c": (13, {"outcome": "failure", "cluster": "b200-nscale", "error": {
            "stage": "eval", "type": "EvalFailed", "message": "eval did not succeed", "retriable": False,
            "detail": "ValueError: a|b " + "x" * 300 + "\n--- last 1 lines ---\ntail"}}),
        "d": (14, {"outcome": "failure", "cluster": "h100-dsxe", "error": {
            "stage": "prepare", "type": "SetupFailed", "message": "make setup exited 2\nsee <logs>", "retriable": False,
            "detail": None}}),
    }  # fmt: skip
    for name, (job_id, record) in records.items():
        artifact = tmp_path / "job_events" / f"job_event_{name}"
        artifact.mkdir(parents=True)
        (artifact / "job_event.json").write_text(json.dumps({**record, "github": {"job_id": job_id}}))
    summary = tmp_path / "summary.md"

    assert main([str(tmp_path / "job_events"), str(tmp_path / "events.jsonl"), "--run", "https://github.com", "o/r",
                 "77", "--summary", str(summary)]) == 0  # fmt: skip

    assert summary.read_text(encoding="utf-8").splitlines() == [
        "**Job events:** 5 total, 1 succeeded, 4 failed, 1 synthesized",
        "",
        "| Cluster | Job | Stage | Error | Retriable | Cause | Link |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| b200-nscale | kimik3 fp4 cluster:b200-nscale vllm TP8    c64 \\| eval-only | eval | EvalFailed | no | "
        f"ValueError: a\\|b {'x' * 183}… | [job]({RUN}/job/13) |",
        f"| h100-dsxe | qwen3.5 fp8 cluster:h100-dsxe sgl TP8/EP8    c20 | prepare | SetupFailed | no | "
        f"make setup exited 2<br>see &lt;logs&gt; | [job]({RUN}/job/14) |",
        f"|  | dsr1 fp4 gb200 dynamo-trt 1P (TP8) x 3D (TP8) mtp   c6x12 | Run actions/checkout@3d3c | NoLauncherRecord "
        f"| ? | job concluded failure at step 'Run actions/checkout@3d3c' and left no job_event.json | [job]({RUN}/job/15) |",
        f"| mi355x-amds | dsr1 fp8 cluster:mi355x-amds atom TP8    c8 | run | JobNodeFail | yes | "
        f"Slurm job 5 ended with state=NODE_FAIL exit_code=0:0 | [job]({RUN}/job/12) |",
    ]  # fmt: skip


def test_summary_of_a_run_without_failures_appends_only_its_counts(tmp_path):
    artifact = tmp_path / "job_events" / "job_event_a"
    artifact.mkdir(parents=True)
    (artifact / "job_event.json").write_text('{"outcome": "success", "error": null}')
    summary = tmp_path / "summary.md"
    summary.write_text("## Revisions\n")

    assert main([str(tmp_path / "job_events"), str(tmp_path / "events.jsonl"), "--summary", str(summary)]) == 0

    assert summary.read_text() == "## Revisions\n**Job events:** 1 total, 1 succeeded, 0 failed, 0 synthesized\n"


def test_a_summary_that_cannot_be_written_leaves_the_records_and_warns(tmp_path, capsys):
    artifact = tmp_path / "job_events" / "job_event_a"
    artifact.mkdir(parents=True)
    (artifact / "job_event.json").write_text('{"outcome": "failure"}')
    output = tmp_path / "events.jsonl"

    assert main([str(tmp_path / "job_events"), str(output), "--summary", str(tmp_path)]) == 0

    assert output.read_text() == '{"outcome":"failure"}\n'
    assert f"WARNING: could not write {tmp_path}" in capsys.readouterr().err
