import json
import subprocess
import sys
from urllib.parse import parse_qs, urlparse

import pytest
import yaml

import infx.config
import infx.workflows.calc_success_rate as success_rate


CLUSTER = {
    "gpus-per-node": 8,
    "arch": "x86_64",
    "scheduler": "slurm",
    "slurm": {"partition": "batch", "exclusive": True},
}


def write_runners(root, cluster_ids, extra_labels=None):
    """Write a runners.yaml whose clusters each own one runner."""
    labels = {f"cluster:{cluster_id}": [f"{cluster_id}_0"] for cluster_id in cluster_ids}
    labels.update(extra_labels or {})
    runners = {"labels": labels, "clusters": {cluster_id: CLUSTER for cluster_id in cluster_ids}}
    (root / "configs").mkdir()
    (root / "configs" / "runners.yaml").write_text(yaml.safe_dump(runners, sort_keys=False))


def test_load_hardware_labels_lists_cluster_ids_only(tmp_path, monkeypatch):
    write_runners(tmp_path, ["zeta", "alpha"], {"gpu": ["zeta_0", "alpha_0"]})
    monkeypatch.setattr(infx.config, "__file__", str(tmp_path / "infx" / "config.py"))

    assert success_rate.load_hardware_labels() == ["alpha", "zeta"]


def test_extract_hardware_from_name_matches_cluster_label():
    patterns = success_rate.build_hardware_match_patterns(["b300-nv", "gb200-nv"])

    assert (
        success_rate.extract_hardware_from_name("dsv4 fp4 cluster:b300-nv vllm | tp=8", patterns)
        == "b300-nv"
    )
    assert (
        success_rate.extract_hardware_from_name("glm5 fp4 gb200-nv dynamo-sglang", patterns)
        == "gb200-nv"
    )


def test_extract_hardware_from_name_does_not_infer_broad_sku():
    patterns = success_rate.build_hardware_match_patterns(["b300-nv", "h200-dgxc"])

    assert success_rate.extract_hardware_from_name("dsv4 fp4 b300 vllm", patterns) is None
    assert success_rate.extract_hardware_from_name("dsv4 fp8 h200 sglang", patterns) is None


@pytest.mark.parametrize(
    "job_name,expected",
    [
        ("model CLUSTER:GPU.A | tp=8", "gpu.a"),
        ("model gpuXa | tp=8", None),
        ("model othergpu.a | tp=8", None),
        ("model gpu.a2 | tp=8", None),
    ],
)
def test_hardware_matching_respects_case_boundaries_and_literal_punctuation(job_name, expected):
    patterns = success_rate.build_hardware_match_patterns(["gpu.a"])

    assert success_rate.extract_hardware_from_name(job_name, patterns) == expected


def query_of(args):
    endpoint = next(arg for arg in args if arg.startswith("repos/"))
    return {key: values[0] for key, values in parse_qs(urlparse(endpoint).query).items()}


def page_of(jobs, page):
    start = (int(page) - 1) * 100
    return {"jobs": jobs[start : start + 100], "total_count": len(jobs)}


def http_error(args, status):
    return subprocess.CalledProcessError(1, args, stderr=f"gh: Server Error (HTTP {status})")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(success_rate.time, "sleep", sleeps.append)
    return sleeps


@pytest.fixture
def run_stats_environment(tmp_path, monkeypatch):
    write_runners(tmp_path, ["sample-a", "sample-b", "unused"])
    monkeypatch.setattr(infx.config, "__file__", str(tmp_path / "infx/config.py"))
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/project")
    output = tmp_path / "stats"
    monkeypatch.setattr(sys, "argv", ["calc_success_rate", str(output)])
    return output.with_suffix(".json")


def test_success_rates_include_all_pages_and_retries(run_stats_environment, monkeypatch, capsys):
    jobs = [
        {"name": f"benchmark cluster:{hardware}", "conclusion": conclusion}
        for hardware, conclusion in [
            ("sample-a", "failure"),
            ("sample-a", "success"),
            ("sample-a", "skipped"),
            ("sample-b", "cancelled"),
            ("sample-b", None),
            ("unrelated", "success"),
        ]
    ]
    jobs.extend({"name": "setup", "conclusion": "success"} for _ in range(94))
    jobs.append({"name": "benchmark cluster:sample-b", "conclusion": "success"})

    def run(args, **kwargs):
        assert "--paginate" not in args
        query = query_of(args)
        selected = jobs if query["filter"] == "all" else jobs[1:]
        return subprocess.CompletedProcess(
            args, 0, json.dumps(page_of(selected, query["page"])), ""
        )

    monkeypatch.setattr(success_rate.github.subprocess, "run", run)
    success_rate.main()

    assert json.loads(run_stats_environment.read_text()) == {
        "sample-a": {"n_success": 1, "total": 2},
        "sample-b": {"n_success": 1, "total": 3},
        "unused": {"n_success": 0, "total": 0},
    }
    table = capsys.readouterr().out
    rows = [line.split() for line in table.splitlines() if line.startswith("sample-")]
    assert rows == [["sample-a", "1", "2", "50.00", "%"], ["sample-b", "1", "3", "33.33", "%"]]
    assert "unused" not in table


@pytest.mark.parametrize(
    "response,error,match",
    [
        (401, RuntimeError, "HTTP 401"),
        ({}, RuntimeError, "unexpected shape"),
        ([{"jobs": [], "total_count": 0}], RuntimeError, "unexpected shape"),
        ({"jobs": None}, RuntimeError, "unexpected shape"),
        ({"jobs": [{}], "total_count": 1}, KeyError, "name"),
    ],
)
def test_failed_stats_do_not_publish_an_artifact(
    run_stats_environment, monkeypatch, response, error, match
):
    def run(args, **kwargs):
        if response == 401:
            raise subprocess.CalledProcessError(1, args, stderr="gh: Unauthorized (HTTP 401)")
        return subprocess.CompletedProcess(args, 0, json.dumps(response), "")

    monkeypatch.setattr(success_rate.github.subprocess, "run", run)
    with pytest.raises(error, match=match):
        success_rate.main()
    assert not run_stats_environment.exists()


def test_later_page_failure_preserves_previous_artifact(
    run_stats_environment, monkeypatch, no_sleep
):
    run_stats_environment.write_text('{"previous": true}\n')
    jobs = [{"name": "benchmark cluster:sample-a", "conclusion": "success"}] * 101

    def run(args, **kwargs):
        query = query_of(args)
        if query["page"] == "1":
            return subprocess.CompletedProcess(args, 0, json.dumps(page_of(jobs, 1)), "")
        raise http_error(args, 503)

    monkeypatch.setattr(success_rate.github.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        success_rate.main()
    assert run_stats_environment.read_text() == '{"previous": true}\n'
    # Both listings retried their failing page before giving up.
    assert no_sleep == [*success_rate.RETRY_DELAYS_SECONDS] * 2


def test_transient_page_failure_is_retried(run_stats_environment, monkeypatch, no_sleep):
    jobs = [{"name": "benchmark cluster:sample-a", "conclusion": "success"}] * 150
    failures = {"2": 2}
    calls = []

    def run(args, **kwargs):
        query = query_of(args)
        calls.append((query["filter"], query["page"]))
        if failures.get(query["page"], 0):
            failures[query["page"]] -= 1
            raise http_error(args, 502)
        return subprocess.CompletedProcess(args, 0, json.dumps(page_of(jobs, query["page"])), "")

    monkeypatch.setattr(success_rate.github.subprocess, "run", run)
    success_rate.main()

    assert json.loads(run_stats_environment.read_text())["sample-a"] == {
        "n_success": 150,
        "total": 150,
    }
    assert calls == [("all", "1"), ("all", "2"), ("all", "2"), ("all", "2")]
    assert no_sleep == list(success_rate.RETRY_DELAYS_SECONDS[:2])


def test_persistent_failure_falls_back_to_latest_attempt(
    run_stats_environment, monkeypatch, capsys
):
    every_attempt = [{"name": "benchmark cluster:sample-a", "conclusion": "failure"}] * 150
    latest = [{"name": "benchmark cluster:sample-a", "conclusion": "success"}] * 3

    def run(args, **kwargs):
        query = query_of(args)
        if query["filter"] == "latest":
            return subprocess.CompletedProcess(args, 0, json.dumps(page_of(latest, 1)), "")
        if query["page"] == "1":
            return subprocess.CompletedProcess(args, 0, json.dumps(page_of(every_attempt, 1)), "")
        raise http_error(args, 502)

    monkeypatch.setattr(success_rate.github.subprocess, "run", run)
    success_rate.main()

    assert json.loads(run_stats_environment.read_text())["sample-a"] == {"n_success": 3, "total": 3}
    assert "::warning title=calc-success-rate::" in capsys.readouterr().out


def test_timeouts_are_transient(run_stats_environment, monkeypatch):
    latest = [{"name": "benchmark cluster:sample-b", "conclusion": "success"}]

    def run(args, **kwargs):
        if query_of(args)["filter"] == "all":
            raise subprocess.TimeoutExpired(args, 60)
        return subprocess.CompletedProcess(args, 0, json.dumps(page_of(latest, 1)), "")

    monkeypatch.setattr(success_rate.github.subprocess, "run", run)
    success_rate.main()
    assert json.loads(run_stats_environment.read_text())["sample-b"] == {"n_success": 1, "total": 1}


def test_non_transient_errors_are_not_retried(run_stats_environment, monkeypatch, no_sleep):
    calls = []

    def run(args, **kwargs):
        calls.append(query_of(args)["filter"])
        raise subprocess.CalledProcessError(1, args, stderr="gh: Not Found (HTTP 404)")

    monkeypatch.setattr(success_rate.github.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="HTTP 404"):
        success_rate.main()
    assert calls == ["all"]
    assert no_sleep == []


def test_empty_job_list_still_writes_zero_counts(run_stats_environment, monkeypatch):
    monkeypatch.setattr(
        success_rate.github.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 0, '{"jobs": [], "total_count": 0}', ""
        ),
    )
    success_rate.main()
    assert json.loads(run_stats_environment.read_text()) == {
        "sample-a": {"n_success": 0, "total": 0},
        "sample-b": {"n_success": 0, "total": 0},
        "unused": {"n_success": 0, "total": 0},
    }


def test_run_stats_wait_for_every_benchmark_sweep():
    jobs = yaml.safe_load(
        (infx.config.git_repository_root() / ".github/workflows/run-sweep.yml").read_text()
    )["jobs"]

    def upstream(name):
        needs = jobs[name].get("needs", [])
        needs = [needs] if isinstance(needs, str) else needs
        return set(needs).union(*(upstream(need) for need in needs))

    sweeps = {name for name in jobs if name.startswith("sweep-") and "evals" not in name}
    assert "sweep-video" in sweeps
    assert sweeps - upstream("calc-success-rate") == set()
