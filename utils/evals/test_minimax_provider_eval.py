from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import minimax_provider_eval as mpe


def _native_outputs(
    output_dir: Path,
    *,
    model: str = "MiniMax-M3",
    status: str = "success",
    match_rate: float = 1.0,
    schema_rate: float = 1.0,
    reasoning_error_rate: float = 0.0,
    tool_call_total: int = 10,
) -> None:
    (output_dir / mpe.NATIVE_RESULTS_FILENAME).write_text(
        json.dumps({"data_index": 1, "status": status}) + "\n",
        encoding="utf-8",
    )
    schema_errors = round((1.0 - schema_rate) * tool_call_total)
    (output_dir / mpe.NATIVE_REPORT_FILENAME).write_text(
        json.dumps(
            {
                "model": model,
                "success_count": 1 if status == "success" else 0,
                "failure_count": 0 if status == "success" else 1,
                "tool_calls_match_rate": match_rate,
                "tool_calls_schema_validation_error_count": schema_errors,
                "tool_calls_total_count": tool_call_total,
                "error_only_reasoning_rate": reasoning_error_rate,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _compatibility(output_dir: Path) -> dict[str, Any]:
    paths = list(output_dir.glob(mpe.COMPATIBILITY_GLOB))
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding="utf-8"))


def test_prepare_smoke_input_preserves_fixture_row(tmp_path: Path) -> None:
    destination = tmp_path / "smoke.jsonl"

    mpe.prepare_smoke_input(
        fixture_path=mpe.DEFAULT_FIXTURE_PATH,
        destination=destination,
    )

    rows = json.loads(mpe.DEFAULT_FIXTURE_PATH.read_text(encoding="utf-8"))["rows"]
    assert [json.loads(line) for line in destination.read_text().splitlines()] == rows


def test_build_command_invokes_stock_verifier_without_source_changes(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    sample_path = tmp_path / "smoke.jsonl"

    command = mpe.build_verifier_command(
        python=Path("/venv/bin/python"),
        source_dir=source_dir,
        sample_path=sample_path,
        base_url="http://127.0.0.1:8000/v1/",
        model="MiniMax-M3",
        output_dir=output_dir,
    )

    assert command[:3] == [
        "/venv/bin/python",
        str(source_dir / "verify.py"),
        str(sample_path),
    ]
    assert command[command.index("--base-url") + 1] == "http://127.0.0.1:8000/v1"
    assert command[command.index("--concurrency") + 1] == "1"
    assert command[command.index("--timeout") + 1] == "600"
    assert command[command.index("--retries") + 1] == "3"
    assert json.loads(command[command.index("--extra-body") + 1]) == {
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 40960,
    }


def test_run_uses_verified_stock_source_and_projects_native_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    dependency_dir = tmp_path / "deps"
    source_dir.mkdir()
    dependency_dir.mkdir()
    verified: list[Path] = []
    invocation: dict[str, Any] = {}

    monkeypatch.setattr(mpe, "verify_source_tree", verified.append)

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        invocation["command"] = command
        invocation.update(kwargs)
        _native_outputs(
            output_dir,
            match_rate=0.9,
            schema_rate=0.8,
            reasoning_error_rate=0.1,
        )
        return subprocess.CompletedProcess(command, 0)

    passed = mpe.run_evaluation(
        python=Path("/venv/bin/python"),
        source_dir=source_dir,
        dependency_dir=dependency_dir,
        base_url="http://127.0.0.1:8000/v1",
        model="MiniMax-M3",
        output_dir=output_dir,
        runner=runner,
    )

    assert passed is True
    assert verified == [source_dir]
    assert invocation["check"] is False
    assert invocation["timeout"] == mpe.UPSTREAM_TIMEOUT_SECONDS
    assert invocation["env"]["PYTHONPATH"] == f"{source_dir}:{dependency_dir}"
    assert invocation["env"]["PYTHONNOUSERSITE"] == "1"
    assert not (output_dir / "minimax_vendor_smoke_input.jsonl").exists()
    compatibility = _compatibility(output_dir)
    assert compatibility["results"][mpe.TASK_NAME]["exact_match,strict-match"] == 0.8
    assert compatibility["n-samples"][mpe.TASK_NAME]["effective"] == 1
    assert "integration_error" not in compatibility


def test_completed_model_failure_is_not_reclassified_as_integration_error(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    dependency_dir = tmp_path / "deps"
    source_dir.mkdir()
    dependency_dir.mkdir()
    monkeypatch.setattr(mpe, "verify_source_tree", lambda _: None)

    def runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        _native_outputs(output_dir, match_rate=0.0, schema_rate=0.0)
        return subprocess.CompletedProcess(command, 0)

    passed = mpe.run_evaluation(
        python=Path("python"),
        source_dir=source_dir,
        dependency_dir=dependency_dir,
        base_url="http://127.0.0.1:8000/v1",
        model="MiniMax-M3",
        output_dir=output_dir,
        runner=runner,
    )

    assert passed is True
    compatibility = _compatibility(output_dir)
    assert compatibility["results"][mpe.TASK_NAME]["exact_match,strict-match"] == 0.0
    assert compatibility["n-samples"][mpe.TASK_NAME]["effective"] == 1
    assert "integration_error" not in compatibility


def test_completed_zero_tool_call_result_is_model_failure(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    dependency_dir = tmp_path / "deps"
    source_dir.mkdir()
    dependency_dir.mkdir()
    monkeypatch.setattr(mpe, "verify_source_tree", lambda _: None)

    def runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        _native_outputs(output_dir, match_rate=0.0, tool_call_total=0)
        return subprocess.CompletedProcess(command, 0)

    passed = mpe.run_evaluation(
        python=Path("python"),
        source_dir=source_dir,
        dependency_dir=dependency_dir,
        base_url="http://127.0.0.1:8000/v1",
        model="MiniMax-M3",
        output_dir=output_dir,
        runner=runner,
    )

    assert passed is True
    compatibility = _compatibility(output_dir)
    assert compatibility["results"][mpe.TASK_NAME]["exact_match,strict-match"] == 0.0
    assert compatibility["n-samples"][mpe.TASK_NAME]["effective"] == 1
    assert "integration_error" not in compatibility


def test_request_failure_is_reported_as_integration_error(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "output"
    source_dir = tmp_path / "source"
    dependency_dir = tmp_path / "deps"
    source_dir.mkdir()
    dependency_dir.mkdir()
    monkeypatch.setattr(mpe, "verify_source_tree", lambda _: None)

    def runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        _native_outputs(output_dir, status="failed")
        return subprocess.CompletedProcess(command, 0)

    passed = mpe.run_evaluation(
        python=Path("python"),
        source_dir=source_dir,
        dependency_dir=dependency_dir,
        base_url="http://127.0.0.1:8000/v1",
        model="MiniMax-M3",
        output_dir=output_dir,
        runner=runner,
    )

    assert passed is False
    compatibility = _compatibility(output_dir)
    assert compatibility["n-samples"][mpe.TASK_NAME]["effective"] == 0
    assert compatibility["integration_error"]["type"] == "SmokeSuiteError"
    assert "request failure" in compatibility["integration_error"]["message"]
    native = json.loads((output_dir / mpe.NATIVE_REPORT_FILENAME).read_text())
    assert native["failure_count"] == 1
    assert "integration_error" not in native


def test_publish_failure_does_not_replace_existing_native_report(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    native_path = output_dir / mpe.NATIVE_REPORT_FILENAME
    native_path.write_text('{"stock": true}\n', encoding="utf-8")

    mpe.publish_failure(
        output_dir=output_dir,
        model="MiniMax-M3",
        error=RuntimeError("transport failed"),
    )

    assert json.loads(native_path.read_text()) == {"stock": True}
    compatibility = _compatibility(output_dir)
    assert compatibility["integration_error"] == {
        "type": "RuntimeError",
        "message": "transport failed",
    }


def test_failure_cli_is_stdlib_only(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"

    assert (
        mpe.main(
            [
                "failure",
                "--model",
                "MiniMax-M3",
                "--output-dir",
                str(output_dir),
                "--message",
                "setup failed",
            ]
        )
        == 0
    )
    compatibility = _compatibility(output_dir)
    assert compatibility["integration_error"]["message"] == "setup failed"
