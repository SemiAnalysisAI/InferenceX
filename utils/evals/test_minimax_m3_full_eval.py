import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import minimax_m3_full_eval as full


def _write_native_run(
    output_dir: Path,
    *,
    model: str = "MiniMax-M3",
    score: float = 0.625,
    result_count: int = full.EXPECTED_RESULT_COUNT,
    failed_index: int | None = None,
) -> bytes:
    output_dir.mkdir(exist_ok=True)
    report = {
        "model": model,
        "success_count": result_count - (failed_index is not None),
        "failure_count": int(failed_index is not None),
        "success_rate": 1.0 if failed_index is None else 0.99,
        "tool_calls_match_rate": score,
        "tool_calls_successful_count": 60,
        "error_only_reasoning_rate": 0.0,
        "language_following_valid_count": 2,
        "scenario_check_pass_rate": 1.0,
    }
    report_bytes = (json.dumps(report, indent=4) + "\n").encode()
    (output_dir / full.NATIVE_REPORT_FILENAME).write_bytes(report_bytes)
    rows = [
        {
            "data_index": index,
            "status": "failed" if index == failed_index else "success",
        }
        for index in range(1, result_count + 1)
    ]
    (output_dir / full.NATIVE_RESULTS_FILENAME).write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return report_bytes


def test_prepared_source_tree_requires_every_pinned_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = b"".join(b"{}\n" for _ in range(full.EXPECTED_RESULT_COUNT))
    contents = {
        "verify.py": b"print('pinned verifier')\n",
        "sample.jsonl": sample,
        "validator/__init__.py": b"",
    }
    monkeypatch.setattr(
        full,
        "REQUIRED_SOURCE_SHA256",
        {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()},
    )

    source_dir = tmp_path / "source"
    full.prepare_source_tree(source_dir, fetcher=contents.__getitem__)
    full.verify_source_tree(source_dir)

    (source_dir / "verify.py").write_text("mutated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        full.verify_source_tree(source_dir)

def test_fetch_source_retries_transient_network_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"pinned source"
    monkeypatch.setattr(
        full,
        "REQUIRED_SOURCE_SHA256",
        {"verify.py": hashlib.sha256(content).hexdigest()},
    )
    monkeypatch.setattr(full, "DOWNLOAD_ATTEMPTS", 4)
    monkeypatch.setattr(full, "DOWNLOAD_RETRY_DELAY_SECONDS", 0.25)
    sleeps: list[float] = []

    class Response:
        status = 200
        headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def geturl(self) -> str:
            return full.source_url("verify.py")

        def read(self, _: int) -> bytes:
            return content

    class Opener:
        calls = 0

        def open(self, request, *, timeout):
            self.calls += 1
            if self.calls <= 2:
                raise full.urllib.error.URLError("transient")
            return Response()

    opener = Opener()
    monkeypatch.setattr(full, "_NO_REDIRECT_OPENER", opener)
    monkeypatch.setattr(full.time, "sleep", sleeps.append)

    assert full._fetch_source("verify.py") == content
    assert opener.calls == 3
    assert sleeps == [0.25, 0.25]


def test_fetch_source_reports_exhausted_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(full, "DOWNLOAD_ATTEMPTS", 2)
    monkeypatch.setattr(full, "DOWNLOAD_RETRY_DELAY_SECONDS", 0.5)
    sleeps: list[float] = []

    class Opener:
        calls = 0

        def open(self, request, *, timeout):
            self.calls += 1
            raise full.urllib.error.URLError("offline")

    opener = Opener()
    monkeypatch.setattr(full, "_NO_REDIRECT_OPENER", opener)
    monkeypatch.setattr(full.time, "sleep", sleeps.append)

    with pytest.raises(full.FullSuiteError, match="after 2 attempts"):
        full._fetch_source("verify.py")
    assert opener.calls == 2
    assert sleeps == [0.5]


def test_verifier_command_is_one_102_row_run_with_fixed_m3_settings(
    tmp_path: Path,
) -> None:
    command = full.build_verifier_command(
        python=Path("/runtime/python"),
        source_dir=Path("/runtime/source"),
        base_url="http://127.0.0.1:8000/v1/",
        model="MiniMax-M3",
        output_dir=tmp_path,
    )

    assert command == [
        "/runtime/python",
        "/runtime/source/verify.py",
        "/runtime/source/sample.jsonl",
        "--model",
        "MiniMax-M3",
        "--base-url",
        "http://127.0.0.1:8000/v1",
        "--api-key",
        "EMPTY",
        "--concurrency",
        "5",
        "--output",
        str(tmp_path / full.NATIVE_RESULTS_FILENAME),
        "--summary",
        str(tmp_path / full.NATIVE_REPORT_FILENAME),
        "--timeout",
        "600",
        "--retries",
        "3",
        "--extra-body",
        '{"temperature":0,"top_p":1,"max_tokens":40960}',
    ]
    assert "pass" not in " ".join(command).lower()
    assert command.count("/runtime/source/sample.jsonl") == 1


def test_projects_exactly_102_results_from_native_match_rate_without_rewriting_report(
    tmp_path: Path,
) -> None:
    report_bytes = _write_native_run(tmp_path)

    compatibility_path = full.project_native_artifacts(
        output_dir=tmp_path, model="MiniMax-M3"
    )

    compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
    assert compatibility["result_format"] == "inferencex-eval-v1"
    assert compatibility["eval_adapter"] == "minimax-provider-verifier"
    assert compatibility["results"] == {
        "minimax_m3_full": {
            "exact_match,strict-match": 0.625,
            "exact_match_stderr,strict-match": 0.0,
        }
    }
    assert compatibility["configs"]["minimax_m3_full"]["native_metric"] == (
        "tool_calls_match_rate"
    )
    assert compatibility["configs"]["minimax_m3_full"][
        "diagnostic_threshold"
    ] == 0.0
    assert compatibility["n-samples"]["minimax_m3_full"] == {
        "original": 102,
        "effective": 102,
    }
    assert (tmp_path / full.NATIVE_REPORT_FILENAME).read_bytes() == report_bytes
    assert len(
        (tmp_path / full.NATIVE_RESULTS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 102

def test_full_projection_removes_stale_smoke_result(tmp_path: Path) -> None:
    stale_smoke = tmp_path / "results_minimax_vendor_2026-01-01.json"
    stale_smoke.write_text("{}")

    compatibility_path = full._compatibility_path(tmp_path)

    assert not stale_smoke.exists()
    assert compatibility_path.name.startswith("results_minimax_vendor_full_")


def test_failure_command_replaces_stale_native_artifacts(tmp_path: Path) -> None:
    native_report = tmp_path / full.NATIVE_REPORT_FILENAME
    native_results = tmp_path / full.NATIVE_RESULTS_FILENAME
    native_report.write_text('{"stale": true}\n')
    native_results.write_text('{"stale": true}\n')

    assert (
        full.main(
            [
                "failure",
                "--model",
                "MiniMax-M3",
                "--output-dir",
                str(tmp_path),
                "--message",
                "runtime setup failed",
            ]
        )
        == 0
    )

    report = json.loads(native_report.read_text(encoding="utf-8"))
    assert report["completed"] is False
    assert report["integration_error"]["message"] == "runtime setup failed"
    result_rows = native_results.read_text(encoding="utf-8").splitlines()
    assert len(result_rows) == 1
    assert json.loads(result_rows[0])["status"] == "integration_error"


@pytest.mark.parametrize(
    ("result_count", "failed_index", "message"),
    [
        (101, None, "exactly 102 rows"),
        (102, 7, "transport failures"),
    ],
)
def test_projection_rejects_incomplete_or_transport_failed_native_results(
    tmp_path: Path,
    result_count: int,
    failed_index: int | None,
    message: str,
) -> None:
    _write_native_run(
        tmp_path,
        result_count=result_count,
        failed_index=failed_index,
    )

    with pytest.raises(full.FullSuiteError, match=message):
        full.project_native_artifacts(output_dir=tmp_path, model="MiniMax-M3")


def test_upstream_process_failure_is_nonzero_and_publishes_all_failure_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "source"
    dependency_dir = tmp_path / "deps"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    dependency_dir.mkdir()
    monkeypatch.setattr(full, "verify_source_tree", lambda _path: None)

    def failed_runner(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess([], 9)

    assert not full.run_full_suite(
        python=Path("/runtime/python"),
        source_dir=source_dir,
        dependency_dir=dependency_dir,
        base_url="http://127.0.0.1:8000/v1",
        model="MiniMax-M3",
        output_dir=output_dir,
        runner=failed_runner,
    )
    compatibility_path = next(output_dir.glob(full.COMPATIBILITY_GLOB))
    compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
    assert compatibility["integration_error"]["type"] == "FullSuiteError"
    assert compatibility["n-samples"]["minimax_m3_full"]["effective"] == 0
    assert (output_dir / full.NATIVE_REPORT_FILENAME).is_file()
    assert (output_dir / full.NATIVE_RESULTS_FILENAME).is_file()
