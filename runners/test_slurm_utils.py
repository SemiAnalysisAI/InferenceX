import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_UTILS = REPO_ROOT / "runners" / "slurm_utils.sh"


def run_bash(command: str, *args: Path | str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", command, "bash", *(str(arg) for arg in args)],
        check=False,
        capture_output=True,
        text=True,
    )


def write_executable(path: Path, body: str) -> None:
    path.write_text(f"#!/usr/bin/env bash\n{body}\n")
    path.chmod(0o755)


@pytest.mark.parametrize(
    ("accounting_record", "expected_returncode", "expected_message"),
    [
        ("4242|COMPLETED|0:0", 0, "completed successfully"),
        ("4242|FAILED|1:0", 1, "ended in FAILED with exit 1:0"),
    ],
)
def test_wait_for_slurm_job_success_reports_accounting_status(
    tmp_path: Path,
    accounting_record: str,
    expected_returncode: int,
    expected_message: str,
) -> None:
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    write_executable(mock_bin / "sacct", 'printf \'%s\\n\' "$MOCK_SACCT_RECORD"')

    result = run_bash(
        'export PATH="$3:$PATH" MOCK_SACCT_RECORD="$4" '
        "SLURM_ACCOUNTING_ATTEMPTS=1 SLURM_ACCOUNTING_INTERVAL_SECONDS=0; "
        'source "$1"; wait_for_slurm_job_success 4242',
        SLURM_UTILS,
        tmp_path,
        mock_bin,
        accounting_record,
    )

    assert result.returncode == expected_returncode
    assert expected_message in result.stdout + result.stderr


def test_copy_agentic_results_stages_only_matching_points(tmp_path: Path) -> None:
    source = tmp_path / "source"
    workspace = tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    (source / "run_conc1.json").write_text('{"conc": 1}\n')
    (source / "run_conc16.json").write_text('{"conc": 16}\n')
    (source / "other_conc1.json").write_text('{"conc": 1}\n')

    result = run_bash(
        'source "$1"; copy_agentic_results "$2" "$3" run',
        SLURM_UTILS,
        source,
        workspace,
    )

    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in workspace.iterdir()) == [
        "run_conc1.json",
        "run_conc16.json",
    ]


def test_copy_agentic_results_fails_when_aggregate_is_missing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    workspace = tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()

    result = run_bash(
        'source "$1"; copy_agentic_results "$2" "$3" run',
        SLURM_UTILS,
        source,
        workspace,
    )

    assert result.returncode != 0
    assert "no run_conc*.json results found" in result.stderr
