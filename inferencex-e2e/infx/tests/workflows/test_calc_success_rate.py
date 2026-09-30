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
        success_rate.extract_hardware_from_name(
            "dsv4 fp4 cluster:b300-nv vllm | tp=8", patterns
        )
        == "b300-nv"
    )
    assert (
        success_rate.extract_hardware_from_name(
            "glm5 fp4 gb200-nv dynamo-sglang", patterns
        )
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


def test_success_rates_include_all_pages_and_retries(
    run_stats_environment, monkeypatch, capsys
):
    jobs = [
        {"name": f"benchmark cluster:{hardware}", "conclusion": conclusion}
        for hardware, conclusion in [
            ("sample-a", "failure"), ("sample-a", "success"),
            ("sample-a", "skipped"), ("sample-b", "cancelled"),
            ("sample-b", None), ("unrelated", "success"),
        ]
    ]
    jobs.extend({"name": "setup", "conclusion": "success"} for _ in range(94))
    jobs.append({"name": "benchmark cluster:sample-b", "conclusion": "success"})

    def run(args, **kwargs):
        endpoint = next(arg for arg in args if arg.startswith("repos/"))
        query = parse_qs(urlparse(endpoint).query)
        selected = jobs if query.get("filter") == ["all"] else jobs[1:]
        pages = [{"jobs": selected[:100], "total_count": len(selected)}]
        if "--paginate" in args:
            pages.append({"jobs": selected[100:], "total_count": len(selected)})
        return subprocess.CompletedProcess(args, 0, json.dumps(pages), "")

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


@pytest.mark.parametrize("response,error,match", [
    (401, RuntimeError, "HTTP 401"),
    ({}, RuntimeError, "unexpected shape"),
    ({"jobs": None}, RuntimeError, "unexpected shape"),
    ({"jobs": [{}], "total_count": 1}, KeyError, "name"),
])
def test_failed_stats_do_not_publish_an_artifact(
    run_stats_environment, monkeypatch, response, error, match
):
    def run(args, **kwargs):
        if response == 401:
            raise subprocess.CalledProcessError(1, args, stderr="gh: Unauthorized (HTTP 401)")
        return subprocess.CompletedProcess(args, 0, json.dumps([response]), "")

    monkeypatch.setattr(success_rate.github.subprocess, "run", run)
    with pytest.raises(error, match=match):
        success_rate.main()
    assert not run_stats_environment.exists()


def test_later_page_failure_preserves_previous_artifact(run_stats_environment, monkeypatch):
    run_stats_environment.write_text('{"previous": true}\n')

    def run(args, **kwargs):
        partial = json.dumps([{"total_count": 101, "jobs": [
            {"name": "benchmark cluster:sample-a", "conclusion": "success"}
            for _ in range(100)
        ]}])
        raise subprocess.CalledProcessError(1, args, output=partial,
                                            stderr="gh: Unavailable (HTTP 503)")

    monkeypatch.setattr(success_rate.github.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        success_rate.main()
    assert run_stats_environment.read_text() == '{"previous": true}\n'


def test_empty_job_list_still_writes_zero_counts(run_stats_environment, monkeypatch):
    monkeypatch.setattr(
        success_rate.github.subprocess, "run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 0, '[{"jobs": [], "total_count": 0}]', ""),
    )
    success_rate.main()
    assert json.loads(run_stats_environment.read_text()) == {
        "sample-a": {"n_success": 0, "total": 0},
        "sample-b": {"n_success": 0, "total": 0},
        "unused": {"n_success": 0, "total": 0},
    }
