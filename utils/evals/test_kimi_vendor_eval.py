import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kimi_vendor_eval as kve


def _native_report(*, stream_status: str = "passed") -> dict:
    statuses = ["passed", stream_status]
    by_status: dict[str, int] = {}
    for status in statuses:
        by_status[status] = by_status.get(status, 0) + 1
    return {
        "summary": {
            "total": 2,
            "by_status": by_status,
            "by_selection_reason": {"object_schema": 2},
            "by_mode": {
                "non-stream": {"passed": 1},
                "stream": {stream_status: 1},
            },
        },
        "results": [
            {"mode": "non-stream", "status": "passed"},
            {"mode": "stream", "status": stream_status},
        ],
    }


def _compatibility_file(output_dir: Path) -> Path:
    matches = list(output_dir.glob(kve.COMPATIBILITY_GLOB))
    assert len(matches) == 1
    assert re.fullmatch(
        r"results_kimi_vendor_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d{6}\.json",
        matches[0].name,
    )
    return matches[0]


def _projected(output_dir: Path) -> dict:
    return json.loads(_compatibility_file(output_dir).read_text(encoding="utf-8"))


def _score(output_dir: Path) -> float:
    return _projected(output_dir)["results"][kve.TASK_NAME][
        "exact_match,strict-match"
    ]


def test_builds_exact_upstream_pytest_command(tmp_path: Path) -> None:
    report = tmp_path / kve.NATIVE_REPORT_FILENAME

    command = kve.build_pytest_command(
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        model="test-model",
        report_path=report,
    )

    assert command == [
        sys.executable,
        "-m",
        "pytest",
        "tests/tool_call_json_schema/test_tool_call_json_schema.py",
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


def test_full_pass_projects_score_and_preserves_native_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier_dir = tmp_path / "verifier"
    verifier_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "results_kimi_vendor_2000-01-01T00-00-00.000000.json").write_text(
        "stale", encoding="utf-8"
    )
    native_bytes = (json.dumps(_native_report(), indent=2) + "\n").encode()
    invocation: dict[str, object] = {}

    def fake_run(command: list[str], *, cwd: Path, check: bool) -> SimpleNamespace:
        invocation.update(command=command, cwd=cwd, check=check)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / kve.NATIVE_REPORT_FILENAME).write_bytes(native_bytes)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(kve.subprocess, "run", fake_run)

    passed = kve.run_evaluation(
        verifier_dir=verifier_dir,
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        model="model-a",
        output_dir=output_dir,
    )

    assert passed
    assert invocation["cwd"] == verifier_dir
    assert invocation["check"] is False
    assert invocation["command"] == kve.build_pytest_command(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        model="model-a",
        report_path=(output_dir / kve.NATIVE_REPORT_FILENAME).resolve(),
    )
    assert (output_dir / kve.NATIVE_REPORT_FILENAME).read_bytes() == native_bytes
    projected = _projected(output_dir)
    assert _score(output_dir) == 1.0
    assert projected["n-samples"][kve.TASK_NAME] == {"original": 2, "effective": 2}
    assert "integration_error" not in projected


def test_one_mode_failure_projects_partial_score_and_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "output"

    def fake_run(command: list[str], *, cwd: Path, check: bool) -> SimpleNamespace:
        report_path = Path(command[command.index("--tool-json-report") + 1])
        report_path.write_text(json.dumps(_native_report(stream_status="failed")))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(kve.subprocess, "run", fake_run)

    passed = kve.run_evaluation(
        verifier_dir=tmp_path,
        base_url="http://localhost/v1",
        api_key="EMPTY",
        model="model-a",
        output_dir=output_dir,
    )

    assert not passed
    assert _score(output_dir) == 0.5


@pytest.mark.parametrize("native_contents", [None, "{not-json"])
def test_missing_or_malformed_report_writes_zero_score_with_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native_contents: str | None,
) -> None:
    output_dir = tmp_path / "output"

    def fake_run(command: list[str], *, cwd: Path, check: bool) -> SimpleNamespace:
        if native_contents is not None:
            report_path = Path(command[command.index("--tool-json-report") + 1])
            report_path.write_text(native_contents, encoding="utf-8")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(kve.subprocess, "run", fake_run)

    passed = kve.run_evaluation(
        verifier_dir=tmp_path,
        base_url="http://localhost/v1",
        api_key="EMPTY",
        model="model-a",
        output_dir=output_dir,
    )

    projected = _projected(output_dir)
    assert not passed
    assert _score(output_dir) == 0.0
    assert projected["integration_error"]["type"] in {
        "FileNotFoundError",
        "JSONDecodeError",
    }
    assert projected["integration_error"]["message"]


def test_collection_failure_cannot_project_a_stale_passing_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    native_report = output_dir / kve.NATIVE_REPORT_FILENAME
    native_report.write_text(json.dumps(_native_report()), encoding="utf-8")

    def collection_failure(
        command: list[str], *, cwd: Path, check: bool
    ) -> SimpleNamespace:
        assert not native_report.exists()
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(kve.subprocess, "run", collection_failure)

    passed = kve.run_evaluation(
        verifier_dir=tmp_path,
        base_url="http://localhost/v1",
        api_key="EMPTY",
        model="model-a",
        output_dir=output_dir,
    )

    projected = _projected(output_dir)
    assert not passed
    assert _score(output_dir) == 0.0
    assert projected["integration_error"]["type"] == "FileNotFoundError"


def test_subprocess_launch_failure_writes_zero_score_with_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_to_launch(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        raise OSError("pytest could not launch")

    monkeypatch.setattr(kve.subprocess, "run", fail_to_launch)
    output_dir = tmp_path / "output"

    passed = kve.run_evaluation(
        verifier_dir=tmp_path,
        base_url="http://localhost/v1",
        api_key="EMPTY",
        model="model-a",
        output_dir=output_dir,
    )

    projected = _projected(output_dir)
    assert not passed
    assert _score(output_dir) == 0.0
    assert projected["integration_error"] == {
        "type": "OSError",
        "message": "pytest could not launch",
    }



def test_cli_integration_error_writes_failure_without_running_pytest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_run(*args: object, **kwargs: object) -> None:
        pytest.fail("integration-error mode must not launch pytest")

    monkeypatch.setattr(kve.subprocess, "run", unexpected_run)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "results_kimi_vendor_2000-01-01T00-00-00.000000.json").write_text(
        "stale", encoding="utf-8"
    )
    (output_dir / kve.NATIVE_REPORT_FILENAME).write_text(
        json.dumps(_native_report()), encoding="utf-8"
    )

    return_code = kve.main(
        [
            "--model",
            "model-a",
            "--output-dir",
            str(output_dir),
            "--integration-error",
            "pinned verifier checkout failed",
        ]
    )

    projected = _projected(output_dir)
    assert return_code == 1
    assert not (output_dir / kve.NATIVE_REPORT_FILENAME).exists()
    assert _score(output_dir) == 0.0
    assert projected["integration_error"] == {
        "type": "RuntimeError",
        "message": "pinned verifier checkout failed",
    }