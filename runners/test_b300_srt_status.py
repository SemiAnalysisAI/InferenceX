"""A completed Slurm allocation must not hide a failed benchmark."""

import json
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
    "accounting,expected_status,power_mode",
    [
        ("COMPLETED|0:0", 0, "off"),
        ("FAILED|1:0", 1, "off"),
        ("COMPLETED|0:9", 1, "off"),
        ("CANCELLED by 123|0:15", 1, "off"),
        ("DELAYED", 0, "off"),
        ("", 1, "off"),
        ("COMPLETED|0:0", 1, "missing"),
        ("COMPLETED|0:0", 0, "eval"),
    ],
)
def test_b300_collects_artifacts_before_returning_slurm_status(
    tmp_path: Path, accounting: str, expected_status: int, power_mode: str
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
        "squeue",
        "sleep",
    ):
        _stub(binaries, name, "exit 0")
    _stub(binaries, "unsquashfs", "exit 0")
    _stub(
        binaries,
        "sacct",
        r"""
[[ " $* " == *" -X "* ]] || exit 2
count=0
[[ -f "$MOCK_SACCT_COUNT" ]] && read -r count < "$MOCK_SACCT_COUNT"
count=$((count + 1))
printf '%s\n' "$count" > "$MOCK_SACCT_COUNT"
if [[ "$*" == *JobIDRaw* ]]; then
    printf '42|%s\n' "$MOCK_ACCOUNTING"
    exit 0
fi
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
    _stub(binaries, "sha256sum", 'printf "%064d  fixture-image\\n" 0')
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # Keep the real launcher-owned profiles and renderer in the fixture checkout.
    shutil.copytree(ROOT / "runners", workspace / "runners")
    (workspace / "benchmarks").mkdir()
    if power_mode != "off":
        recipe = workspace / "benchmarks/multi_node/srt-slurm-recipes/test.yaml"
        recipe.parent.mkdir(parents=True)
        recipe.write_text(
            "telemetry:\n  enabled: true\n  dcgm_exporter:\n    container_image: dcgm-exporter\n"
        )
    shutil.copyfile(
        ROOT / "benchmarks/benchmark_lib.sh", workspace / "benchmarks/benchmark_lib.sh"
    )
    (workspace / "infx").symlink_to(ROOT / "infx", target_is_directory=True)
    # Stub remote checkout and submission after loading the real shared helpers.
    with (workspace / "runners/slurm_utils.sh").open("a") as helpers:
        helpers.write(r"""
setup_srt_slurm() {
    mkdir -p "$1/recipes"
    cd "$1" || return 1
    printf 'name: fixture\nbenchmark:\n  type: custom\n' > recipes/test.yaml
    SRT_SLURM_COMMIT=1111111111111111111111111111111111111111
    printf '%s\n' "$SRT_SLURM_COMMIT" > "$GITHUB_WORKSPACE/power-producer-sha.txt"
}
apply_srt_recipe() {
    mkdir -p outputs/42/logs
    cp -R "$MOCK_FIXTURE/." outputs/42/logs/
    printf '{"diagnostic":"retained","disagg":true,"num_prefill_gpu":4,"num_decode_gpu":4}\n' > "$GITHUB_WORKSPACE/aggregate_conc1.json"
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
        "EVAL_ONLY": "true" if power_mode == "eval" else "false",
        "IS_AGENTIC": "1" if power_mode != "off" else "0",
        "CONC_LIST": "1",
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
        "RUNNER_NAME": "b300-dsxe_08",
        "IMAGE": "fixture-image",
        "ENROOT_IMPORT_TIME_LIMIT": "00:05:00",
        "SALLOC_TIME_LIMIT": "00:59:00",
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
# The launcher writes the fixed cluster cache directory once; the mock image
# probe does not need it, and the portable test must not touch /data.
mkdir() {
    if [[ "$*" == "-p /data/home/sa-gha-runner/squash" ]]; then return 0; fi
    command mkdir "$@"
}
test() {
    if [[ "$1" == -r && "$2" == /data/* ]]; then return 0; fi
    builtin test "$@"
}
builtin source "$1/runners/launch_b300-dsxe.sh"
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
    if power_mode != "off":
        import yaml

        emitted = yaml.safe_load((workspace / "srt-slurm/recipes/test.yaml").read_text())
        assert emitted["benchmark"]["concurrencies"] == [1]
    aggregate = json.loads((workspace / "aggregate_conc1.json").read_text())
    assert aggregate["diagnostic"] == "retained"
    if power_mode == "missing":
        validation = json.loads(
            (workspace / "LOGS/agentic/conc_1/power_validation.json").read_text()
        )
        assert validation["power_valid"] is False
        assert validation["reasons"] == ["formal_benchmark_result_missing"]
    else:
        assert not (workspace / "LOGS/agentic/conc_1/power_validation.json").exists()
    assert (workspace / "results_eval.json").read_text() == '{"eval":"retained"}\n'
    assert (workspace / "LOGS/sweep_42.log").read_text() == "benchmark diagnostics\n"
    with tarfile.open(workspace / "multinode_server_logs.tar.gz") as archive:
        assert (
            "./agentic/conc_1/aiperf_artifacts/profile_export_aiperf.json"
            in archive.getnames()
        )
    assert not (workspace / "srt-slurm/outputs").exists()
    expected_queries = (
        2
        if power_mode == "missing"
        else 3
        if accounting == "DELAYED"
        else 10
        if accounting == ""
        else 1
    )
    assert int((tmp_path / "sacct-count").read_text()) == expected_queries
