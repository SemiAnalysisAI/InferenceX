import io
import json
import sys
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pytest
import yaml

import infx.config
import infx.workflows.calc_success_rate as success_rate


@pytest.mark.parametrize(
    "runners,expected",
    [
        (
            {
                "labels": {
                    "cluster:zeta": ["self-hosted"],
                    "gpu": ["self-hosted"],
                    "cluster:alpha": ["self-hosted"],
                },
                "hardware": {"legacy": {}},
            },
            ["alpha", "zeta"],
        ),
        ({"cluster:zeta": [], "unrelated": [], "cluster:alpha": []}, ["alpha", "zeta"]),
        ({"labels": {"gpu": []}, "hardware": {"legacy": {}}}, ["legacy"]),
        ({}, []),
    ],
    ids=["cluster-labels-take-precedence", "flat-labels", "legacy-fallback", "empty"],
)
def test_load_hardware_labels_normalizes_supported_layouts(
    tmp_path, monkeypatch, runners, expected
):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "runners.yaml").write_text(yaml.safe_dump(runners, sort_keys=False))
    monkeypatch.setattr(infx.config, "__file__", str(tmp_path / "infx" / "config.py"))

    assert success_rate.load_hardware_labels() == expected


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
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "runners.yaml").write_text(
        "labels: {cluster:sample-a: [], cluster:sample-b: [], cluster:unused: []}\n"
    )
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

    def urlopen(request, timeout):
        assert urlparse(request.full_url).path == "/repos/example/project/actions/runs/42/jobs"
        query = parse_qs(urlparse(request.full_url).query)
        selected = jobs if query.get("filter") == ["all"] else jobs[1:]
        start = (int(query["page"][0]) - 1) * int(query["per_page"][0])
        return io.BytesIO(json.dumps({"jobs": selected[start:start + 100]}).encode())

    monkeypatch.setattr(success_rate.github.urllib.request, "urlopen", urlopen)
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
    ({"jobs": [{}]}, KeyError, "name"),
])
def test_failed_stats_do_not_publish_an_artifact(
    run_stats_environment, monkeypatch, response, error, match
):
    def urlopen(request, timeout):
        if response == 401:
            raise HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO(b"denied"))
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(success_rate.github.urllib.request, "urlopen", urlopen)
    with pytest.raises(error, match=match):
        success_rate.main()
    assert not run_stats_environment.exists()


def test_later_page_failure_preserves_previous_artifact(run_stats_environment, monkeypatch):
    run_stats_environment.write_text('{"previous": true}\n')

    def urlopen(request, timeout):
        query = parse_qs(urlparse(request.full_url).query)
        if query["page"] == ["1"]:
            return io.BytesIO(json.dumps({"jobs": [
                {"name": "benchmark cluster:sample-a", "conclusion": "success"}
                for _ in range(100)
            ]}).encode())
        raise HTTPError(request.full_url, 503, "Unavailable", {}, io.BytesIO(b"unavailable"))

    monkeypatch.setattr(success_rate.github.urllib.request, "urlopen", urlopen)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        success_rate.main()
    assert run_stats_environment.read_text() == '{"previous": true}\n'


def test_empty_job_list_still_writes_zero_counts(run_stats_environment, monkeypatch):
    monkeypatch.setattr(
        success_rate.github.urllib.request, "urlopen",
        lambda request, timeout: io.BytesIO(b'{"jobs": []}'),
    )
    success_rate.main()
    assert json.loads(run_stats_environment.read_text()) == {
        "sample-a": {"n_success": 0, "total": 0},
        "sample-b": {"n_success": 0, "total": 0},
        "unused": {"n_success": 0, "total": 0},
    }
