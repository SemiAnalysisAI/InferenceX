import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kimi_vendor_eval as kve


def _report(stream_status: str = "passed") -> dict[str, Any]:
    statuses = ["passed", stream_status]
    by_status = {status: statuses.count(status) for status in set(statuses)}
    return {
        "summary": {"total": 2, "by_status": by_status},
        "results": [
            {"mode": "non-stream", "status": "passed"},
            {"mode": "stream", "status": stream_status},
        ],
    }


def _report_with_inconsistent_counts() -> dict[str, Any]:
    report = _report("failed")
    report["summary"]["by_status"]["failed"] = 2
    return report


def _result(output_dir: Path) -> dict[str, Any]:
    paths = list(output_dir.glob(kve.COMPATIBILITY_GLOB))
    assert len(paths) == 1
    assert re.fullmatch(
        r"results_kimi_vendor_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d{6}\.json",
        paths[0].name,
    )
    return json.loads(paths[0].read_text())


def _score(output_dir: Path) -> float:
    return _result(output_dir)["results"][kve.TASK_NAME]["exact_match,strict-match"]


def _n_eff(output_dir: Path) -> int:
    return _result(output_dir)["n-samples"][kve.TASK_NAME]["effective"]


def test_builds_fixed_upstream_pytest_command(tmp_path: Path) -> None:
    report = tmp_path / kve.NATIVE_REPORT_FILENAME

    assert kve.build_pytest_command(
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        model="test-model",
        report_path=report,
    ) == [
        sys.executable,
        "-m",
        "pytest",
        "tests/tool_call_json_schema/test_tool_call_json_schema.py",
        "--reruns",
        "6",
        "--reruns-delay",
        "3",
        "--only-rerun",
        (
            r"(?i)(Error code: (404|429|5[0-9]{2})|APIConnectionError|"
            r"APITimeoutError|Connection error|timed out)"
        ),
        "--base-url",
        "http://127.0.0.1:8000/v1",
        "--api-key",
        "EMPTY",
        "--smoke-model",
        "test-model",
        "--think-mode",
        "none",
        "--selection",
        "object",
        "--max-cases",
        "1",
        "--case-dir",
        "testdata/walle_validator_cases/validator_cases",
        "--max-tokens",
        "2048",
        "--tool-json-report",
        str(report),
    ]


def test_builds_dsv4_thinking_command(tmp_path: Path) -> None:
    command = kve.build_pytest_command(
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        model="deepseek-ai/DeepSeek-V4-Pro",
        model_prefix="dsv4",
        report_path=tmp_path / kve.NATIVE_REPORT_FILENAME,
    )

    think_mode_index = command.index("--think-mode")
    selection_index = command.index("--selection")
    assert command[think_mode_index:selection_index] == [
        "--think-mode",
        "opensource",
        "--thinking",
    ]


@pytest.mark.parametrize(
    ("stream_status", "return_code", "expected_pass", "expected_score"),
    (
        ("passed", 0, True, 1.0),
        ("passed", 1, False, 0.0),
        ("failed", 1, False, 0.5),
    ),
)
def test_projects_upstream_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stream_status: str,
    return_code: int,
    expected_pass: bool,
    expected_score: float,
) -> None:
    output_dir = tmp_path / "output"
    native_bytes = json.dumps(_report(stream_status)).encode()
    invocation: dict[str, Any] = {}

    def fake_run(
        command: list[str], *, cwd: Path, check: bool, timeout: int
    ) -> SimpleNamespace:
        invocation.update(command=command, cwd=cwd, check=check, timeout=timeout)
        Path(command[command.index("--tool-json-report") + 1]).write_bytes(native_bytes)
        return SimpleNamespace(returncode=return_code)

    monkeypatch.setattr(kve.subprocess, "run", fake_run)

    assert (
        kve.run_evaluation(
            verifier_dir=tmp_path,
            base_url="http://localhost/v1",
            api_key="EMPTY",
            model="model-a",
            output_dir=output_dir,
        )
        is expected_pass
    )
    assert invocation["cwd"] == tmp_path
    assert invocation["check"] is False
    assert invocation["timeout"] == kve.DEFAULT_TIMEOUT_SECONDS
    assert _score(output_dir) == expected_score
    assert _n_eff(output_dir) == 2
    projected = _result(output_dir)
    assert projected["result_format"] == kve.RESULT_FORMAT
    assert projected["eval_adapter"] == kve.ADAPTER_NAME
    assert "lm_eval_version" not in projected
    assert (output_dir / kve.NATIVE_REPORT_FILENAME).read_bytes() == native_bytes


@pytest.mark.parametrize(
    ("failure", "error_type"),
    (
        (None, "FileNotFoundError"),
        ("{bad-json", "JSONDecodeError"),
        (OSError("boom"), "OSError"),
        (json.dumps(_report("skipped")), "ValueError"),
        (json.dumps(_report_with_inconsistent_counts()), "ValueError"),
        (
            subprocess.TimeoutExpired("pytest", kve.DEFAULT_TIMEOUT_SECONDS),
            "TimeoutExpired",
        ),
    ),
)
def test_collection_failures_write_zero_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str | BaseException | None,
    error_type: str,
) -> None:
    def fake_run(
        command: list[str], *, cwd: Path, check: bool, timeout: int
    ) -> SimpleNamespace:
        if isinstance(failure, BaseException):
            raise failure
        if failure is not None:
            Path(command[command.index("--tool-json-report") + 1]).write_text(failure)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(kve.subprocess, "run", fake_run)
    output_dir = tmp_path / "output"

    assert not kve.run_evaluation(
        verifier_dir=tmp_path,
        base_url="http://localhost/v1",
        api_key="EMPTY",
        model="model-a",
        output_dir=output_dir,
    )
    projected = _result(output_dir)
    assert _score(output_dir) == 0.0
    assert projected["integration_error"]["type"] == error_type
    assert _n_eff(output_dir) == 0


def test_failure_cannot_reuse_stale_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    native_report = output_dir / kve.NATIVE_REPORT_FILENAME
    native_report.write_text(json.dumps(_report()))
    (output_dir / "results_kimi_vendor_2000-01-01T00-00-00.000000.json").write_text(
        "{}"
    )
    foreign_result = output_dir / "results_other_eval.json"
    foreign_result.write_text("{}")

    def fail_collection(*args: Any, **kwargs: Any) -> SimpleNamespace:
        assert not native_report.exists()
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(kve.subprocess, "run", fail_collection)

    assert not kve.run_evaluation(
        verifier_dir=tmp_path,
        base_url="http://localhost/v1",
        api_key="EMPTY",
        model="model-a",
        output_dir=output_dir,
    )
    assert _score(output_dir) == 0.0
    assert not native_report.exists()
    assert foreign_result.exists()


def test_cli_setup_failure_writes_zero_score_artifact(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / kve.NATIVE_REPORT_FILENAME).write_text(json.dumps(_report()))
    (output_dir / "results_kimi_vendor_2000-01-01T00-00-00.000000.json").write_text(
        "{}"
    )

    assert (
        kve.main(
            [
                "--model",
                "model-a",
                "--output-dir",
                str(output_dir),
                "--integration-error",
                "checkout failed",
            ]
        )
        == 0
    )
    projected = _result(output_dir)
    assert not (output_dir / kve.NATIVE_REPORT_FILENAME).exists()
    assert _score(output_dir) == 0.0
    assert projected["integration_error"]["message"] == "checkout failed"
    assert _n_eff(output_dir) == 0
