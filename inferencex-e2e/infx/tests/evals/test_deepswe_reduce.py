import importlib.util
import json
from pathlib import Path

import pytest

REDUCER = Path(__file__).resolve().parents[3] / "utils/deepswe/reduce_rewards.py"


@pytest.fixture
def reduce():
    spec = importlib.util.spec_from_file_location("reduce_rewards", REDUCER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trial(job: Path, task: str, suffix: str, reward: float | None) -> None:
    trial = job / f"{task}__{suffix}"
    (trial / "verifier").mkdir(parents=True)
    (trial / "config.json").write_text(json.dumps({"task": {"name": f"datacurve/{task}"}}))
    if reward is not None:
        (trial / "verifier" / "reward.json").write_text(
            json.dumps({"reward": reward, "f2p": 0.9, "p2p": 1.0})
        )


def _run(reduce, tmp_path: Path, job: Path, n_attempts: int) -> tuple[int, dict, dict]:
    out = tmp_path / "out"
    code = reduce.main(
        ["--job-dir", str(job), "--output-dir", str(out), "--model", "m", "--n-attempts", str(n_attempts)]
    )
    [result_path] = out.glob("results_deepswe_*.json")
    report = json.loads((out / "deepswe_report.json").read_text())
    return code, json.loads(result_path.read_text()), report


def test_complete_two_attempt_job_scores_without_integration_error(reduce, tmp_path):
    job = tmp_path / "job"
    for i in range(reduce.TOTAL_TASKS):
        _trial(job, f"task-{i:03d}", "a", 1.0 if i < 80 else 0.0)
        _trial(job, f"task-{i:03d}", "b", 1.0 if i < 60 else 0.0)
    (job / "result.json").write_text(json.dumps({"stats": {"n_errored_trials": 0}}))

    code, result, report = _run(reduce, tmp_path, job, 2)

    assert code == 0
    assert "integration_error" not in result
    scores = result["results"]["kimi_deepswe"]
    assert scores["exact_match,strict-match"] == pytest.approx(140 / 226)
    assert result["n-samples"]["kimi_deepswe"] == {"original": 226, "effective": 226}
    assert report["summary"]["pass1_by_attempt"] == pytest.approx([80 / 113, 60 / 113])
    assert report["summary"]["errored_trials"] == 0
    assert report["by_task"]["task-000"][0]["f2p"] == 0.9


def test_missing_reward_is_an_errored_trial_and_an_integration_error(reduce, tmp_path):
    job = tmp_path / "job"
    _trial(job, "abs-module-cache-flags", "x", 1.0)
    _trial(job, "abs-stepped-slices", "y", None)

    code, result, report = _run(reduce, tmp_path, job, 1)

    assert code == 1
    assert result["integration_error"]["type"] == "IncompleteRun"
    assert "1 errored" in result["integration_error"]["message"]
    assert result["n-samples"]["kimi_deepswe"] == {"original": 113, "effective": 1}
    assert report["errored_trials"] == ["abs-stepped-slices__y"]
    assert report["summary"]["accuracy"] == 1.0


def test_subset_job_named_in_the_pier_config_is_complete_and_limited(reduce, tmp_path):
    job = tmp_path / "job"
    _trial(job, "abs-module-cache-flags", "x", 0.0)
    (job / "config.json").write_text(
        json.dumps({"datasets": [{"path": "tasks", "task_names": ["abs-module-cache-flags"]}]})
    )

    code, result, report = _run(reduce, tmp_path, job, 1)

    assert code == 0
    assert "integration_error" not in result
    assert result["limit"] == 1
    assert result["n-samples"]["kimi_deepswe"] == {"original": 113, "effective": 1}
    assert report["summary"]["expected_trials"] == 1


def test_task_id_falls_back_to_the_directory_name(reduce, tmp_path):
    job = tmp_path / "job"
    trial = job / "yjs-map-conflict-detection__z" / "verifier"
    trial.mkdir(parents=True)
    (trial / "reward.json").write_text(json.dumps({"reward": 0.0}))

    _, _, report = _run(reduce, tmp_path, job, 1)

    assert list(report["by_task"]) == ["yjs-map-conflict-detection"]
