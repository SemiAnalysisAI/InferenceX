"""Tests for batched multi-node eval runtime and validation."""

import json
import os
import re
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
    assert meta["eval_exit_code"] == 0
    assert sorted(path.name for path in tmp_path.glob("results*.json")) == [
        "results_test_conc1.json",
        "results_test_conc4.json",
        "results_test_conc8.json",
    ]
    assert validate_batch_manifest(
        str(tmp_path / "meta_env.json"),
        [str(path) for path in tmp_path.glob("results*.json")],
        expected_concs=[1, 4, 8],
    ) == []


def test_batched_eval_preserves_partial_results_and_records_failure(
    tmp_path: Path,
) -> None:
    meta = _run_batched_eval(tmp_path, failing_conc="4")

    assert meta["completed_eval_concs"] == [1, 8]
    assert meta["failed_eval_concs"] == [4]
    assert meta["eval_exit_code"] == 1
    errors = validate_batch_manifest(
        str(tmp_path / "meta_env.json"),
        [str(path) for path in tmp_path.glob("results*.json")],
        expected_concs=[1, 4, 8],
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


def test_batched_eval_requires_a_result_for_every_workflow_concurrency(
    tmp_path: Path,
) -> None:
    meta_path = tmp_path / "meta_env.json"
    meta_path.write_text(
        json.dumps({
            "eval_exit_code": 0,
            "eval_concs": [1, 4, 8],
            "completed_eval_concs": [1, 4, 8],
            "failed_eval_concs": [],
        })
    )
    result_files = []
    for conc in (1, 8):
        result_path = tmp_path / f"results_test_conc{conc}.json"
        result_path.write_text('{"results": {}}')
        result_files.append(str(result_path))

    errors = validate_batch_manifest(
        str(meta_path),
        result_files,
        expected_concs=[1, 4, 8],
    )

    assert any("missing result files for concurrency: 4" in error for error in errors)


def test_validate_scores_fails_when_workflow_batch_metadata_is_unreadable(
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
    assert "meta_env.json is unavailable or invalid" in captured.err


def test_validate_scores_rejects_single_result_for_workflow_batch(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    meta_path = tmp_path / "meta_env.json"
    meta_path.write_text(json.dumps({
        "eval_exit_code": 0,
        "conc": 8,
        "infmax_model_prefix": "test",
    }))
    result_path = tmp_path / "results_test.json"
    result_path.write_text(
        json.dumps({
            "results": {
                "custom_eval": {
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
    assert "workflow requested multiple eval concurrencies" in captured.err
    assert "result lacks a concurrency suffix" in captured.err
    assert "missing result files for concurrency: 1, 4, 8" in captured.err


def test_validate_scores_fails_when_any_concurrency_is_below_threshold(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    meta_path = tmp_path / "meta_env.json"
    meta_path.write_text(
        json.dumps({
            "eval_exit_code": 0,
            "eval_concs": [1, 4],
            "completed_eval_concs": [1, 4],
            "failed_eval_concs": [],
            "infmax_model_prefix": "test",
        })
    )
    for conc, score in ((1, 0.9), (4, 0.8)):
        (tmp_path / f"results_test_conc{conc}.json").write_text(
            json.dumps({
                "results": {
                    "custom_eval": {
                        "exact_match,strict-match": score,
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
            str(tmp_path / "results*.json"),
            "--expected-concs",
            "1 4",
            "--min-score",
            "0.85",
        ],
    )

    assert validate_scores_main() == 1
    captured = capsys.readouterr()
    assert "results_test_conc4.json" in captured.err
    assert "0.8000 (< 0.85 from min-score)" in captured.err


def test_validate_scores_accepts_complete_batch_above_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    meta_path = tmp_path / "meta_env.json"
    meta_path.write_text(
        json.dumps({
            "eval_exit_code": 0,
            "eval_concs": [1, 4],
            "completed_eval_concs": [1, 4],
            "failed_eval_concs": [],
            "infmax_model_prefix": "test",
        })
    )
    for conc, score in ((1, 0.9), (4, 0.86)):
        (tmp_path / f"results_test_conc{conc}.json").write_text(
            json.dumps({
                "results": {
                    "custom_eval": {
                        "exact_match,strict-match": score,
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
            str(tmp_path / "results*.json"),
            "--expected-concs",
            "1 4",
            "--min-score",
            "0.85",
        ],
    )

    assert validate_scores_main() == 0


def test_validate_scores_accepts_single_concurrency_above_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    meta_path = tmp_path / "meta_env.json"
    meta_path.write_text(json.dumps({
        "eval_exit_code": 0,
        "conc": 4,
        "infmax_model_prefix": "test",
    }))
    result_path = tmp_path / "results_test.json"
    result_path.write_text(
        json.dumps({
            "results": {
                "custom_eval": {
                    "exact_match,strict-match": 0.9,
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
            "4",
            "--min-score",
            "0.85",
        ],
    )

    assert validate_scores_main() == 0


def test_validate_scores_fails_when_a_concurrency_has_no_score_metric(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    meta_path = tmp_path / "meta_env.json"
    meta_path.write_text(
        json.dumps({
            "eval_exit_code": 0,
            "eval_concs": [1, 4],
            "completed_eval_concs": [1, 4],
            "failed_eval_concs": [],
            "infmax_model_prefix": "test",
        })
    )
    (tmp_path / "results_test_conc1.json").write_text(
        json.dumps({
            "results": {
                "custom_eval": {
                    "exact_match,strict-match": 0.9,
                },
            },
        })
    )
    (tmp_path / "results_test_conc4.json").write_text(
        json.dumps({"results": {"custom_eval": {"alias": "custom"}}})
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_scores.py",
            "--meta-env",
            str(meta_path),
            "--results-glob",
            str(tmp_path / "results*.json"),
            "--expected-concs",
            "1 4",
            "--min-score",
            "0.85",
        ],
    )

    assert validate_scores_main() == 1
    captured = capsys.readouterr()
    assert (
        "results_test_conc4.json has no numeric metrics matching prefix"
        in captured.err
    )


def test_validate_scores_fails_when_one_task_has_no_score_metric(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    meta_path = tmp_path / "meta_env.json"
    meta_path.write_text(json.dumps({
        "eval_exit_code": 0,
        "conc": 4,
        "infmax_model_prefix": "test",
    }))
    result_path = tmp_path / "results_test.json"
    result_path.write_text(json.dumps({
        "results": {
            "valid_eval": {
                "exact_match,strict-match": 0.9,
            },
            "missing_metric_eval": {
                "alias": "missing_metric_eval",
            },
        },
    }))
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
            "4",
        ],
    )

    assert validate_scores_main() == 1
    assert (
        "missing_metric_eval has no metric matching prefix"
        in capsys.readouterr().err
    )


def test_validate_scores_fails_for_non_numeric_score(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    meta_path = tmp_path / "meta_env.json"
    meta_path.write_text(json.dumps({
        "eval_exit_code": 0,
        "conc": 4,
        "infmax_model_prefix": "test",
    }))
    result_path = tmp_path / "results_test.json"
    result_path.write_text(json.dumps({
        "results": {
            "custom_eval": {
                "exact_match,strict-match": "0.99",
            },
        },
    }))
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
            "4",
        ],
    )

    assert validate_scores_main() == 1
    assert "has non-numeric value '0.99'" in capsys.readouterr().err


def test_validate_scores_fails_for_non_finite_score(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    meta_path = tmp_path / "meta_env.json"
    meta_path.write_text(json.dumps({
        "eval_exit_code": 0,
        "conc": 4,
        "infmax_model_prefix": "test",
    }))
    result_path = tmp_path / "results_test.json"
    result_path.write_text(
        '{"results":{"custom_eval":{"exact_match,strict-match":NaN}}}'
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
            "4",
        ],
    )

    assert validate_scores_main() == 1
    captured = capsys.readouterr()
    assert "exact_match,strict-match is not finite" in captured.err


def test_validate_scores_fails_for_out_of_range_score(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    meta_path = tmp_path / "meta_env.json"
    meta_path.write_text(json.dumps({
        "eval_exit_code": 0,
        "conc": 4,
        "infmax_model_prefix": "test",
    }))
    result_path = tmp_path / "results_test.json"
    result_path.write_text(json.dumps({
        "results": {
            "custom_eval": {
                "exact_match,strict-match": 1.01,
            },
        },
    }))
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
            "4",
        ],
    )

    assert validate_scores_main() == 1
    assert "is outside [0, 1]" in capsys.readouterr().err


def test_validate_scores_requires_consistent_metrics_across_concurrencies(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    meta_path = tmp_path / "meta_env.json"
    meta_path.write_text(json.dumps({
        "eval_exit_code": 0,
        "eval_concs": [1, 4],
        "completed_eval_concs": [1, 4],
        "failed_eval_concs": [],
        "infmax_model_prefix": "test",
    }))
    for conc, task in ((1, "custom_eval"), (4, "different_eval")):
        (tmp_path / f"results_test_conc{conc}.json").write_text(json.dumps({
            "results": {
                task: {
                    "exact_match,strict-match": 0.9,
                },
            },
        }))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_scores.py",
            "--meta-env",
            str(meta_path),
            "--results-glob",
            str(tmp_path / "results*.json"),
            "--expected-concs",
            "1 4",
        ],
    )

    assert validate_scores_main() == 1
    captured = capsys.readouterr()
    assert "is missing metrics present in results_test_conc1.json" in captured.err
    assert (
        "has unexpected metrics compared with results_test_conc1.json"
        in captured.err
    )


def test_single_eval_failure_is_staged_and_fails_validation(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    benchmark_lib = (
        Path(__file__).resolve().parents[2] / "benchmarks" / "benchmark_lib.sh"
    )
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(benchmark_lib),
    }
    script = r'''
source "$BENCHMARK_LIB"

run_lm_eval() {
    local results_dir
    results_dir=$(mktemp -d)
    export EVAL_RESULT_DIR="$results_dir"
    mkdir -p "$results_dir/nested"
    printf '%s' \
        '{"results":{"custom_eval":{"exact_match,strict-match":0.99}}}' \
        > "$results_dir/nested/results_test.json"
    return 7
}

export EVAL_CONCURRENT_REQUESTS=4
export EVAL_MAX_MODEL_LEN=4096
export EVAL_ONLY=true
export MODEL=test-model
export MODEL_NAME=test-model
export MODEL_PREFIX=test
export RUNNER_TYPE=h100
export FRAMEWORK=vllm
export PRECISION=fp8
export SPEC_DECODING=none
export IS_MULTINODE=false
export ISL=8192
export OSL=1024
export TP=8
export EP_SIZE=1
export CONC=4

run_eval --framework lm-eval --port 8888
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

    meta_path = tmp_path / "meta_env.json"
    assert json.loads(meta_path.read_text())["eval_exit_code"] == 7
    result_path = tmp_path / "results_test.json"
    assert result_path.is_file()

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
            "4",
            "--min-score",
            "0.85",
        ],
    )

    assert validate_scores_main() == 1
    assert "eval command failed with exit code 7" in capsys.readouterr().err


def test_append_lm_eval_summary_reports_artifact_move_failure(
    tmp_path: Path,
) -> None:
    benchmark_lib = (
        Path(__file__).resolve().parents[2] / "benchmarks" / "benchmark_lib.sh"
    )
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(benchmark_lib),
    }
    script = r'''
source "$BENCHMARK_LIB"

export EVAL_RESULT_DIR="$PWD/eval-output"
mkdir -p "$EVAL_RESULT_DIR/nested"
printf '%s' \
    '{"results":{"custom_eval":{"exact_match,strict-match":0.99}}}' \
    > "$EVAL_RESULT_DIR/nested/results_test.json"

export EVAL_RUN_EXIT_CODE=0
export MODEL=test-model
export MODEL_PREFIX=test
export RUNNER_TYPE=h100
export FRAMEWORK=vllm
export PRECISION=fp8
export SPEC_DECODING=none
export IS_MULTINODE=false
export ISL=8192
export OSL=1024
export TP=8
export EP_SIZE=1
export CONC=4

mv() {
    local arg
    for arg in "$@"; do
        if [[ "$arg" == *results_test.json ]]; then
            return 1
        fi
    done
    command mv "$@"
}

if append_lm_eval_summary; then
    exit 9
fi
test -f "$EVAL_RESULT_DIR/nested/results_test.json"
test -f "$PWD/meta_env.json"
'''
    completed = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "eval artifact staging was incomplete" in completed.stderr


def test_run_eval_rejects_duplicate_concurrency_before_execution(
    tmp_path: Path,
) -> None:
    benchmark_lib = (
        Path(__file__).resolve().parents[2] / "benchmarks" / "benchmark_lib.sh"
    )
    trace_path = tmp_path / "ran_eval"
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(benchmark_lib),
        "TRACE_PATH": str(trace_path),
    }
    script = r'''
source "$BENCHMARK_LIB"
run_lm_eval() {
    touch "$TRACE_PATH"
}
export EVAL_CONCURRENT_REQUESTS="4 4"
export EVAL_MAX_MODEL_LEN=4096
export MODEL=test-model
if run_eval --framework lm-eval --port 8888; then
    exit 9
fi
test "$EVAL_RUN_EXIT_CODE" -eq 2
'''

    subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    assert not trace_path.exists()


def test_validate_scores_rejects_invalid_threshold_config(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(
        json.dumps({"default": {"custom_eval": "0.9"}})
    )
    meta_path = tmp_path / "meta_env.json"
    meta_path.write_text(json.dumps({
        "eval_exit_code": 0,
        "conc": 4,
        "infmax_model_prefix": "test",
    }))
    result_path = tmp_path / "results_test.json"
    result_path.write_text(json.dumps({
        "results": {
            "custom_eval": {
                "exact_match,strict-match": 0.99,
            },
        },
    }))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_scores.py",
            "--thresholds",
            str(thresholds_path),
            "--meta-env",
            str(meta_path),
            "--results-glob",
            str(result_path),
            "--expected-concs",
            "4",
        ],
    )

    assert validate_scores_main() == 1
    assert "must be a finite number between 0 and 1" in capsys.readouterr().err


def test_batched_eval_rejects_duplicate_result_files(tmp_path: Path) -> None:
    meta_path = tmp_path / "meta_env.json"
    meta_path.write_text(json.dumps({
        "eval_exit_code": 0,
        "eval_concs": [4],
        "completed_eval_concs": [4],
        "failed_eval_concs": [],
    }))
    result_files = []
    for name in ("results_test_conc4.json", "results_test_conc4_2.json"):
        result_path = tmp_path / name
        result_path.write_text('{"results": {}}')
        result_files.append(str(result_path))

    errors = validate_batch_manifest(
        str(meta_path),
        result_files,
        expected_concs=[4],
    )

    assert any(
        "duplicate result files for concurrency: 4" in error
        for error in errors
    )


def test_amd_multinode_container_inherits_eval_concurrency_list() -> None:
    amd_utils = (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "multi_node"
        / "amd_utils"
    )
    job_slurm = amd_utils / "job.slurm"
    contents = job_slurm.read_text()
    submit_contents = (amd_utils / "submit.sh").read_text()

    assert r'-e \"EVAL_CONC=\$EVAL_CONC\"' in contents
    assert "\n    -e EVAL_CONC\n" not in contents
    for env_name in (
        "RUN_EVAL",
        "EVAL_ONLY",
        "EVAL_FRAMEWORK",
        "EVAL_TASKS_DIR",
        "EVAL_MAX_MODEL_LEN",
        "MODEL",
        "MAX_MODEL_LEN",
        "FRAMEWORK",
        "PRECISION",
        "MODEL_PREFIX",
        "RUNNER_TYPE",
        "RESULT_FILENAME",
        "SPEC_DECODING",
        "PREFILL_TP_SIZE",
        "PREFILL_ENABLE_EP",
        "PREFILL_ENABLE_DP",
        "DECODE_TP_SIZE",
        "DECODE_ENABLE_EP",
        "DECODE_ENABLE_DP",
        "IS_MULTINODE",
    ):
        assert rf"-e {env_name}=\${env_name}" in contents
    assert "-e MODEL_PATH=$DOCKER_MODEL_PATH" in contents
    for env_name in (
        "EVAL_CONC",
        "EVAL_FRAMEWORK",
        "EVAL_TASKS_DIR",
        "EVAL_MAX_MODEL_LEN",
        "MODEL",
        "MAX_MODEL_LEN",
    ):
        export_line = f'export {env_name}="${{{env_name}:-}}"'
        assert export_line in contents
        assert export_line in submit_contents


def test_direct_docker_launcher_forwards_workflow_metadata() -> None:
    launcher = (
        Path(__file__).resolve().parents[2] / "runners" / "launch_h100-cr.sh"
    ).read_text()

    for env_name in (
        "IMAGE",
        "MODEL_PREFIX",
        "FRAMEWORK",
        "PRECISION",
        "EP_SIZE",
        "DP_ATTENTION",
        "SPEC_DECODING",
        "DISAGG",
        "RUN_EVAL",
        "EVAL_ONLY",
        "EVAL_FRAMEWORK",
        "EVAL_TASKS_DIR",
        "EVAL_MAX_MODEL_LEN",
        "OPENAI_API_KEY",
        "RUNNER_TYPE",
        "RESULT_FILENAME",
        "SCENARIO_TYPE",
        "SCENARIO_SUBDIR",
        "IS_AGENTIC",
        "OFFLOADING",
        "TOTAL_CPU_DRAM_GB",
        "DURATION",
        "RESULT_DIR",
        "PYTHONPYCACHEPREFIX",
    ):
        assert f"    {env_name}\n" in launcher


def test_multinode_launchers_replace_container_owned_eval_artifacts() -> None:
    runners_dir = Path(__file__).resolve().parents[2] / "runners"
    launchers = (
        "launch_b200-dgxc.sh",
        "launch_b300-nv.sh",
        "launch_gb200-nv.sh",
        "launch_gb300-nv.sh",
        "launch_h100-dgxc-slurm.sh",
        "launch_h200-dgxc-slurm.sh",
        "launch_mi355x-amds.sh",
    )

    for launcher_name in launchers:
        contents = (runners_dir / launcher_name).read_text()
        assert 'eval_dest="$GITHUB_WORKSPACE/$(basename "$eval_file")"' in contents
        assert 'rm -f "$eval_dest"' in contents
        assert 'if cp "$eval_file" "$eval_dest"; then' in contents
        assert 'cp "$eval_file" "$GITHUB_WORKSPACE/"' not in contents

    amd_launcher = (runners_dir / "launch_mi355x-amds.sh").read_text()
    assert (
        '[[ "${RUN_EVAL:-false}" == "true" || '
        '"${EVAL_ONLY:-false}" == "true" ]]'
        in amd_launcher
    )


def test_atom_launcher_uses_and_records_requested_parallelism() -> None:
    server = (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "multi_node"
        / "amd_utils"
        / "server_atom.sh"
    ).read_text()

    assert 'DECODE_PARALLEL_ARGS=(-tp "$DECODE_TP_SIZE")' in server
    assert '[[ "$PREFILL_ENABLE_EP" == "true" ]]' in server
    assert '[[ "$PREFILL_ENABLE_DP" == "true" ]]' in server
    assert '[[ "$DECODE_ENABLE_EP" == "true" ]]' in server
    assert '[[ "$DECODE_ENABLE_DP" == "true" ]]' in server
    for metadata_assignment in (
        'export EP_SIZE=1',
        'export PREFILL_EP=1',
        'export DECODE_EP=1',
        'export DP_ATTENTION="${PREFILL_ENABLE_DP}"',
        'export PREFILL_DP_ATTENTION="${PREFILL_ENABLE_DP}"',
        'export DECODE_DP_ATTENTION="${DECODE_ENABLE_DP}"',
    ):
        assert metadata_assignment in server


def test_eval_workflows_pass_their_requested_concurrencies_to_validation() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    single_workflow = (
        repo_root / ".github" / "workflows" / "benchmark-tmpl.yml"
    ).read_text()
    multinode_workflow = (
        repo_root / ".github" / "workflows" / "benchmark-multinode-tmpl.yml"
    ).read_text()
    run_sweep = (
        repo_root / ".github" / "workflows" / "run-sweep.yml"
    ).read_text()

    assert '--expected-concs "${CONC}"' in single_workflow
    assert 'expected_concs="${EVAL_CONC}"' in multinode_workflow
    assert '--expected-concs "${expected_concs}"' in multinode_workflow
    verify_condition = (
        "(success() || failure()) && "
        "(inputs.run-eval || inputs.eval-only)"
    )
    assert verify_condition in single_workflow
    assert verify_condition in multinode_workflow
    single_inputs = run_sweep.split("&single-node-inputs", 1)[1].split(
        "sweep-single-node-8k1k",
        1,
    )[0]
    assert "run-eval: false" in single_inputs
    assert "run-eval: ${{ matrix.config.run-eval }}" not in run_sweep


def test_eval_regression_tests_run_for_every_inspected_runtime_path() -> None:
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "workflows"
        / "test-changelog-gate.yml"
    ).read_text()

    for watched_path in (
        '".github/workflows/benchmark-tmpl.yml"',
        '".github/workflows/benchmark-multinode-tmpl.yml"',
        '"benchmarks/benchmark_lib.sh"',
        '"benchmarks/multi_node/amd_utils/**"',
        '"benchmarks/single_node/**"',
        '"runners/launch_*.sh"',
        '"utils/evals/**"',
    ):
        assert watched_path in workflow
    assert "utils/evals/test_batched_eval.py" in workflow


def test_eval_scripts_in_audited_scope_use_shared_finalization() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    benchmark_lib = (repo_root / "benchmarks" / "benchmark_lib.sh").resolve()
    ignored_scripts = {
        "dsv4_fp4_mi355x_atom_mtp.sh",
        "dsv4_fp4_mi355x_sglang_mtp.sh",
        "dsv4_fp4_mi355x_vllm_mtp.sh",
    }
    source_pattern = re.compile(
        r'source "\$\(dirname "\$0"\)/([^"]*benchmark_lib\.sh)"'
    )
    failures = []

    for script in sorted(
        (repo_root / "benchmarks" / "single_node" / "fixed_seq_len").rglob(
            "*.sh"
        )
    ):
        if "deprecated" in script.parts or script.name in ignored_scripts:
            continue
        contents = script.read_text()
        if "run_eval --framework" not in contents:
            continue
        if "append_lm_eval_summary" not in contents:
            failures.append(f"{script}: missing append_lm_eval_summary")
        match = source_pattern.search(contents)
        if match is None:
            failures.append(f"{script}: missing relative benchmark_lib source")
            continue
        resolved = (script.parent / match.group(1)).resolve()
        if resolved != benchmark_lib:
            failures.append(f"{script}: resolves benchmark_lib to {resolved}")

    assert not failures, "\n".join(failures)


def test_slurm_container_launchers_export_the_workflow_environment() -> None:
    runners_dir = Path(__file__).resolve().parents[2] / "runners"
    checked = []

    for launcher in sorted(runners_dir.glob("launch_*.sh")):
        contents = launcher.read_text()
        if "--container-image=" not in contents:
            continue
        checked.append(launcher.name)
        assert "--export=ALL" in contents, (
            f"{launcher.name} starts a Slurm container without exporting "
            "the workflow environment"
        )

    assert checked


def test_srtctl_launchers_export_the_eval_workspace() -> None:
    runners_dir = Path(__file__).resolve().parents[2] / "runners"
    launchers = (
        "launch_b200-dgxc.sh",
        "launch_b300-nv.sh",
        "launch_gb200-nv.sh",
        "launch_gb300-nv.sh",
        "launch_h100-dgxc-slurm.sh",
        "launch_h200-dgxc-slurm.sh",
    )

    for launcher_name in launchers:
        contents = (runners_dir / launcher_name).read_text()
        assert 'export EVAL_ONLY="${EVAL_ONLY:-false}"' in contents
        assert 'export INFMAX_WORKSPACE="$GITHUB_WORKSPACE"' in contents
        assert "srtctl apply" in contents


def test_atom_logging_uses_env_and_native_router_flag_without_stream_filter() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    amd_utils = repo_root / "benchmarks" / "multi_node" / "amd_utils"
    env_contents = (amd_utils / "env_atom.sh").read_text()
    job_contents = (amd_utils / "job.slurm").read_text()
    server_contents = (amd_utils / "server_atom.sh").read_text()
    setup_contents = (amd_utils / "setup_deps.sh").read_text()

    assert 'ATOM_LOG_LEVEL="${ATOM_LOG_LEVEL:-WARNING}"' in env_contents
    assert (
        'ATOM_UVICORN_LOG_LEVEL="${ATOM_UVICORN_LOG_LEVEL:-warning}"'
        in env_contents
    )
    assert 'ATOM_UVICORN_ACCESS_LOG="${ATOM_UVICORN_ACCESS_LOG:-0}"' in env_contents
    assert 'ATOMESH_LOG_LEVEL="${ATOMESH_LOG_LEVEL:-warn}"' in env_contents
    assert r"-e ATOM_LOG_LEVEL=\${ATOM_LOG_LEVEL:-WARNING}" in job_contents
    assert (
        r"-e ATOM_UVICORN_LOG_LEVEL=\${ATOM_UVICORN_LOG_LEVEL:-warning}"
        in job_contents
    )
    assert (
        r"-e ATOM_UVICORN_ACCESS_LOG=\${ATOM_UVICORN_ACCESS_LOG:-0}"
        in job_contents
    )
    assert r"-e ATOMESH_LOG_LEVEL=\${ATOMESH_LOG_LEVEL:-warn}" in job_contents
    assert "--log-level ${ATOMESH_LOG_LEVEL}" in server_contents
    assert "filter_atom_logs.sh" not in server_contents
    assert 'os.getenv("ATOM_LOG_LEVEL", "WARNING")' in setup_contents
    assert "logger.setLevel(_atom_log_level)" in setup_contents
    assert "console_handler.setLevel(_atom_log_level)" in setup_contents
    assert "logger.propagate = False" in setup_contents
    assert "ATOM_UVICORN_ACCESS_LOG" in setup_contents
    assert 'access_log=__import__("os").getenv(' in setup_contents
    assert "verify_logger_behavior()" in setup_contents
    assert "verify_uvicorn_behavior()" in setup_contents
    assert not (amd_utils / "filter_atom_logs.sh").exists()


def test_atom_logging_patch_is_idempotent(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    setup_deps = (
        repo_root / "benchmarks" / "multi_node" / "amd_utils" / "setup_deps.sh"
    )
    atom_dir = tmp_path / "atom"
    logger_path = atom_dir / "utils" / "__init__.py"
    api_path = atom_dir / "entrypoints" / "openai" / "api_server.py"
    logger_path.parent.mkdir(parents=True)
    api_path.parent.mkdir(parents=True)
    (atom_dir / "__init__.py").write_text("")
    logger_path.write_text(
        "import logging\n"
        "import os\n\n"
        'logger = logging.getLogger("atom")\n\n'
        "def getLogger():\n"
        "    global logger\n"
        "    if not logger.handlers:\n"
        "        logger.setLevel(logging.DEBUG)\n"
        "        console_handler = logging.StreamHandler()\n"
        "        console_handler.setLevel(logging.INFO)\n"
        "        logger.addHandler(console_handler)\n"
        "    return logger\n\n"
        "logger = getLogger()\n"
    )
    api_path.write_text(
        "def main(app, args, uvicorn):\n"
        "    uvicorn.run(app, host=args.host, port=args.server_port)\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp_path),
        "SETUP_DEPS": str(setup_deps),
    }
    script = r'''
_SETUP_INSTALLED=()
eval "$(sed -n '/^patch_atom_logging_controls()/,/^}/p' "$SETUP_DEPS")"
patch_atom_logging_controls
patch_atom_logging_controls
'''

    completed = subprocess.run(
        ["bash", "-c", script],
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    logger_contents = logger_path.read_text()
    api_contents = api_path.read_text()
    assert logger_contents.count('os.getenv("ATOM_LOG_LEVEL"') == 1
    assert logger_contents.count("logger.setLevel(_atom_log_level)") == 1
    assert logger_contents.count("console_handler.setLevel(_atom_log_level)") == 1
    assert logger_contents.count("logger.propagate = False") == 1
    assert api_contents.count("ATOM_UVICORN_LOG_LEVEL") == 1
    assert api_contents.count("ATOM_UVICORN_ACCESS_LOG") == 1
    assert "engine=WARNING propagation=off uvicorn=warning access_log=off" in (
        completed.stdout
    )
    subprocess.run(
        ["python3", "-m", "py_compile", str(logger_path), str(api_path)],
        check=True,
        text=True,
        capture_output=True,
    )


def test_amd_servers_fail_when_eval_artifact_finalization_fails() -> None:
    amd_utils = (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "multi_node"
        / "amd_utils"
    )

    for server_name in ("server_atom.sh", "server_sglang.sh", "server_vllm.sh"):
        contents = (amd_utils / server_name).read_text()
        assert "append_lm_eval_summary; then" in contents
        assert (
            'if ! _copy_lm_eval_artifacts /workspace "$EVAL_COPY_DIR"; then'
            in contents
        )
        assert 'echo "ERROR: failed to finalize eval artifacts"' in contents
        assert 'echo "ERROR: failed to stage eval artifacts' in contents
