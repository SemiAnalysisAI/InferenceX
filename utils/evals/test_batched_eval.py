"""Tests for batched multi-node eval runtime and validation."""

import json
import os
import subprocess
import sys
from pathlib import Path

from validate_scores import main as validate_scores_main
from validate_scores import validate_batch_manifest


def _run_batched_eval(
    tmp_path: Path,
    *,
    failing_conc: str = "",
) -> dict:
    benchmark_lib = (
        Path(__file__).resolve().parents[2] / "benchmarks" / "benchmark_lib.sh"
    )
    trace_path = tmp_path / "eval_concs.txt"
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(benchmark_lib),
        "TRACE_PATH": str(trace_path),
        "FAILING_CONC": failing_conc,
    }
    script = r'''
source "$BENCHMARK_LIB"

run_lm_eval() {
    local results_dir=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --results-dir) results_dir="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    mkdir -p "$results_dir/nested"
    printf '%s\n' "$EVAL_CONCURRENT_REQUESTS" >> "$TRACE_PATH"
    printf '{"lm_eval_version":"0.4.0"}' \
        > "$results_dir/nested/results_test.json"
    printf '{"sample":true}\n' \
        > "$results_dir/nested/samples_test.jsonl"
    if [ "$EVAL_CONCURRENT_REQUESTS" = "$FAILING_CONC" ]; then
        return 7
    fi
}

export EVAL_CONCURRENT_REQUESTS="1 4 8"
export EVAL_MAX_MODEL_LEN=4096
export EVAL_ONLY=true
export MODEL=test-model
export MODEL_NAME=test-model
export MODEL_PREFIX=test
export RUNNER_TYPE=gb200
export FRAMEWORK=dynamo-sglang
export PRECISION=fp8
export SPEC_DECODING=none
export IS_MULTINODE=true
export ISL=8192
export OSL=1024
export PREFILL_TP=4
export PREFILL_EP=1
export PREFILL_NUM_WORKERS=1
export DECODE_TP=8
export DECODE_EP=1
export DECODE_NUM_WORKERS=2

run_eval --framework lm-eval --port 30000
export CONC="$EVAL_CONCURRENT_REQUESTS"
append_lm_eval_summary
'''
    subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    assert trace_path.read_text().splitlines() == ["1", "4", "8"]
    return json.loads((tmp_path / "meta_env.json").read_text())


def test_batched_eval_runs_every_concurrency_and_stages_results(
    tmp_path: Path,
) -> None:
    meta = _run_batched_eval(tmp_path)

    assert meta["eval_concs"] == [1, 4, 8]
    assert meta["completed_eval_concs"] == [1, 4, 8]
    assert meta["failed_eval_concs"] == []
    assert sorted(path.name for path in tmp_path.glob("results*.json")) == [
        "results_test_conc1.json",
        "results_test_conc4.json",
        "results_test_conc8.json",
    ]
    assert validate_batch_manifest(
        str(tmp_path / "meta_env.json"),
        [str(path) for path in tmp_path.glob("results*.json")],
    ) == []


def test_batched_eval_preserves_partial_results_and_records_failure(
    tmp_path: Path,
) -> None:
    meta = _run_batched_eval(tmp_path, failing_conc="4")

    assert meta["completed_eval_concs"] == [1, 8]
    assert meta["failed_eval_concs"] == [4]
    errors = validate_batch_manifest(
        str(tmp_path / "meta_env.json"),
        [str(path) for path in tmp_path.glob("results*.json")],
    )
    assert any("failed for concurrency: 4" in error for error in errors)
    assert any("missing completed concurrency: 4" in error for error in errors)


def test_batched_eval_requires_a_valid_manifest(tmp_path: Path) -> None:
    result_path = tmp_path / "results_test_conc4.json"
    result_path.write_text('{"lm_eval_version":"0.4.0"}')

    errors = validate_batch_manifest(
        str(tmp_path / "meta_env.json"),
        [str(result_path)],
    )

    assert any("unavailable or invalid" in error for error in errors)


def test_validate_scores_fails_when_expected_batch_metadata_is_unreadable(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    meta_path = tmp_path / "meta_env.json"
    meta_path.write_text("{invalid")
    result_path = tmp_path / "results_test.json"
    result_path.write_text(
        json.dumps({
            "results": {
                "gsm8k": {
                    "exact_match,strict-match": 1.0,
                },
            },
        })
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_scores.py",
            "--meta-env",
            str(meta_path),
            "--results-glob",
            str(result_path),
            "--expected-concs",
            "1 4 8",
        ],
    )

    assert validate_scores_main() == 1
    captured = capsys.readouterr()
    assert "unavailable or invalid" in captured.err


def test_workflow_concurrencies_are_independent_of_eval_metadata(
    tmp_path: Path,
) -> None:
    meta_path = tmp_path / "meta_env.json"
    meta_path.write_text(json.dumps({
        "eval_concs": [8],
        "completed_eval_concs": [8],
        "failed_eval_concs": [],
    }))
    result_path = tmp_path / "results_test_conc8.json"
    result_path.write_text('{"results": {}}')

    errors = validate_batch_manifest(
        str(meta_path),
        [str(result_path)],
        expected_concs=[1, 4, 8],
    )

    assert "batched eval metadata does not match workflow concurrencies" in errors
    assert any("missing completed concurrency: 1, 4" in error for error in errors)
    assert any("missing result files for concurrency: 1, 4" in error for error in errors)


def test_validate_scores_checks_threshold_for_every_concurrency(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "meta_env.json").write_text(json.dumps({
        "eval_concs": [1, 4],
        "completed_eval_concs": [1, 4],
        "failed_eval_concs": [],
    }))
    for conc, score in ((1, 0.9), (4, 0.8)):
        (tmp_path / f"results_test_conc{conc}.json").write_text(json.dumps({
            "results": {
                "gsm8k": {
                    "exact_match,strict-match": score,
                },
            },
        }))
    monkeypatch.setattr(sys, "argv", [
        "validate_scores.py",
        "--meta-env",
        str(tmp_path / "meta_env.json"),
        "--results-glob",
        str(tmp_path / "results*.json"),
        "--expected-concs",
        "1 4",
    ])

    assert validate_scores_main() == 1

    # Each score line is attributed to the concurrency that produced it, so a
    # failing concurrency is identifiable from the log (conc 4 here).
    captured = capsys.readouterr()
    assert "PASS: [conc=1] gsm8k exact_match,strict-match" in captured.out
    assert "FAIL: [conc=4] gsm8k exact_match,strict-match" in captured.err


def test_amd_multinode_container_forwards_eval_concurrency_list() -> None:
    job_slurm = (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "multi_node"
        / "amd_utils"
        / "job.slurm"
    )
    contents = job_slurm.read_text()

    assert r'-e \"EVAL_CONC=\$EVAL_CONC\"' in contents
    assert "-e EVAL_CONC\n" not in contents

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "benchmark-multinode-tmpl.yml"
    ).read_text()
    assert 'expected_concs="${EVAL_CONC}"' in workflow
    assert 'validate_scores.py --expected-concs "${expected_concs}"' in workflow
