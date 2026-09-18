"""A completed Slurm allocation must not hide a failed benchmark."""

import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _stub(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text("#!/bin/bash\n" + body + "\n")
    path.chmod(0o755)


@pytest.mark.parametrize(
    "accounting,expected_status",
    [
        ("COMPLETED|0:0", 0),
        ("FAILED|1:0", 1),
        ("COMPLETED|0:9", 1),
        ("CANCELLED by 123|0:15", 1),
        ("DELAYED", 0),
        ("", 1),
    ],
)
def test_b200_collects_artifacts_before_returning_slurm_status(
    tmp_path: Path, accounting: str, expected_status: int
) -> None:
    """Run the actual Qwen launcher path with only external services stubbed."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name in (
        "curl",
        "uv",
        "make",
        "srtctl",
        "flock",
        "unsquashfs",
        "squeue",
        "sleep",
    ):
        _stub(binaries, name, "exit 0")
    _stub(
        binaries,
        "sacct",
        r"""
[[ " $* " == *" -X "* ]] || exit 2
count=0
[[ -f "$MOCK_SACCT_COUNT" ]] && read -r count < "$MOCK_SACCT_COUNT"
count=$((count + 1))
printf '%s\n' "$count" > "$MOCK_SACCT_COUNT"
if [[ "$MOCK_ACCOUNTING" == DELAYED ]]; then
    case "$count" in
        1) exit 0 ;;
        2) printf 'RUNNING|0:0\n' ;;
        *) printf 'COMPLETED|0:0\n' ;;
    esac
else
    printf '%s\n' "$MOCK_ACCOUNTING"
fi
""",
    )
    # Tail's long-running process is a clock collaborator; logs remain real.
    _stub(
        binaries,
        "tail",
        'for arg in "$@"; do [[ -f "$arg" ]] && cat "$arg"; done; exit 0',
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # Keep launcher-owned runtime inputs with the real launcher and helpers.
    shutil.copytree(ROOT / "runners", workspace / "runners")
    (workspace / "benchmarks").mkdir()
    shutil.copyfile(
        ROOT / "benchmarks/benchmark_lib.sh", workspace / "benchmarks/benchmark_lib.sh"
    )
    # Run the real configuration renderer regardless of editable-install state.
    (workspace / "infx").symlink_to(ROOT / "infx", target_is_directory=True)
    # Stub remote checkout and submission after loading the real shared helpers.
    with (workspace / "runners/slurm_utils.sh").open("a") as helpers:
        helpers.write(r"""
setup_srt_slurm() {
    mkdir -p "$1/recipes"
    cd "$1" || return 1
    printf 'name: fixture\n' > recipes/test.yaml
}
apply_srt_recipe() {
    mkdir -p outputs/42/logs
    cp -R "$MOCK_FIXTURE/." outputs/42/logs/
    printf '{"diagnostic":"retained"}\n' > "$GITHUB_WORKSPACE/aggregate_conc1.json"
    printf '✅ Job 42\n'
}
""")
    activation = workspace / ".venv/bin/activate"
    activation.parent.mkdir(parents=True)
    activation.write_text(":\n")
    fixture = tmp_path / "fixture"
    (fixture / "agentic/conc_1/aiperf_artifacts").mkdir(parents=True)
    (fixture / "eval_results").mkdir()
    (fixture / "sweep_42.log").write_text("benchmark diagnostics\n")
    (fixture / "agentic/conc_1/aiperf_artifacts/profile_export_aiperf.json").write_text(
        '{"metadata":{"submission_valid":false}}\n'
    )
    (fixture / "eval_results/results_eval.json").write_text('{"eval":"retained"}\n')
    env = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "MOCK_ACCOUNTING": accounting,
        "MOCK_SACCT_COUNT": str(tmp_path / "sacct-count"),
        "MOCK_FIXTURE": str(fixture),
        "GITHUB_WORKSPACE": str(workspace),
        "EVAL_ONLY": "false",
        "IS_AGENTIC": "0",
        "IS_MULTINODE": "true",
        "RUN_EVAL": "true",
        "SLURM_PARTITION": "batch_1",
        "SLURM_ACCOUNT": "benchmark",
        "FRAMEWORK": "dynamo-sglang",
        "MODEL_PREFIX": "qwen3.5",
        "PRECISION": "fp8",
        "MODEL": "Qwen/Qwen3.5-397B-A17B-FP8",
        "SPEC_DECODING": "mtp",
        "CONFIG_FILE": "recipes/test.yaml",
        "RUNNER_NAME": "b200-nscale-slurm_08",
        "IMAGE": "fixture-image",
        "B200_SQUASH_DIR": str(tmp_path / "containers"),
        "B200_SQUASH_LOCK_TIMEOUT": "1",
        "RESULT_FILENAME": "aggregate",
        "ISL": "1",
        "OSL": "1",
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            r"""
# GNU-only command options in this Linux launcher are irrelevant to the
# completion contract; keep the test runnable on macOS too.
grep() {
    if [[ "$1" == -oP ]]; then cat >/dev/null; printf '42\n'; else command grep "$@"; fi
}
sed() {
    if [[ "$1" == -i && "$(uname)" == Darwin ]]; then
        shift; command sed -i '' "$@"
    else
        command sed "$@"
    fi
}
builtin source "$1/runners/launch_b200-nscale-slurm.sh"
""",
            "bash",
            str(workspace),
        ],
        cwd=workspace,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == expected_status, result.stdout + result.stderr
    assert (
        workspace / "aggregate_conc1.json"
    ).read_text() == '{"diagnostic":"retained"}\n'
    assert (workspace / "results_eval.json").read_text() == '{"eval":"retained"}\n'
    assert (workspace / "LOGS/sweep_42.log").read_text() == "benchmark diagnostics\n"
    with tarfile.open(workspace / "multinode_server_logs.tar.gz") as archive:
        assert (
            "./agentic/conc_1/aiperf_artifacts/profile_export_aiperf.json"
            in archive.getnames()
        )
    assert not (workspace / "srt-slurm/outputs").exists()
    expected_queries = 3 if accounting == "DELAYED" else 10 if accounting == "" else 1
    assert int((tmp_path / "sacct-count").read_text()) == expected_queries
