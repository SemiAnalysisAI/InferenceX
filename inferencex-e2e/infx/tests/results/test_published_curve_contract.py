"""The published-curve snapshot stays in step with the matrix, binder and API client."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from infx.config import repository_root
from infx.klaud import api
from infx.matrix.revision import Revision, snapshot
from infx.results import published_curve as curve

POINT_FIELDS = ("recipe-fingerprint", "conc", "srt-recipe", "framework", "runner", "exp-name",
                "image", "model", "model-prefix", "precision", "scenario-type")
RUN = "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39187209967"


def row(id, disagg, **fields):
    return {"id": id, "model": "glm5.2", "hardware": "gb300", "framework": "dynamo-sglang",
            "precision": "fp4", "benchmark_type": "agentic_traces", "isl": None, "osl": None,
            "conc": 56 if disagg else 1, "run_url": RUN + "/attempts/2", "date": "2026-10-20",
            "run_started_at": "2026-10-20T08:20:56Z", "recipe_fingerprint": id * 8,
            "image": "lmsysorg/sglang:nightly-dev-20261016-a93f16f0@sha256:0b3e", "disagg": disagg,
            "num_prefill_gpu": 4 if disagg else 8, "num_decode_gpu": 8, "prefill_tp": 4,
            "prefill_ep": 1, "metrics": {"p90_full_response_itl": 0.00292, "tput_per_gpu": 2859.6},
            **fields}


OPTIONS = {
    "models": [{"name": "GLM-5.2", "dbKeys": ["glm5.2", "glm5.3"], "category": "default"}],
    "hardware": [{"key": "gb300", "costPerHour": {"h": 2.31, "r": 5}}],
}
BENCHMARKS = [row("a", False), row("b", True),
              row("c", False, model="glm5.3", benchmark_type="single_turn", isl=8192, osl=1024)]
# The run is listed only under a fork, so the requested rows have no producing commit
WORKFLOW_INFO = {"runConfigs": [
    {"github_run_id": "39187209967", "html_url": "https://github.com/example/fork/actions/runs/1",
     "head_sha": "0" * 40},
]}
EVALUATIONS = [{"model": "glm5.3", "precision": "fp4"}, {"model": "glm5.2", "precision": "fp4"}]


@pytest.fixture(scope="module")
def agentic_matrix():
    revision = Revision(repository_root())
    keys = [
        key
        for path in revision.master_configs
        for key, config in (yaml.safe_load(Path(path).read_text()) or {}).items()
        if isinstance(config, dict) and "agentic-coding" in (config.get("scenarios") or {})
    ]
    rows = revision.generate(keys, [])
    return revision, [
        {**row, "conc": conc}
        for row in rows
        for conc in (row["conc"] if isinstance(row["conc"], list) else [row["conc"]])
    ]


def test_every_agentx_point_binds_to_one_variant_that_writes_and_loads(agentic_matrix, tmp_path):
    revision, matrix = agentic_matrix
    keys = sorted({(point["recipe-fingerprint"], point["conc"]) for point in matrix})
    rows = [{"id": str(i), "conc": conc, "recipe_fingerprint": fingerprint}
            for i, (fingerprint, conc) in enumerate(keys)]

    matched, unbound = curve.bind(rows, matrix)

    assert unbound == {}
    assert sorted(len(ids) for _, ids in matched) == [1] * len(rows)
    assert {"prefill" in point for point, _ in matched} == {True, False}
    for point, ids in matched:
        found = curve.variant(revision, point)
        entry = curve.write_variant(tmp_path, "HEAD", point, found, ids)
        written = yaml.safe_load((tmp_path / entry["file"]).read_text())
        assert written == json.loads(json.dumps(found["recipe"]))
        path, _, selector = point["srt-recipe"].partition(":")
        assert (entry["path"], entry["conc"], entry["image"]) == (path, point["conc"], point["image"])
        assert not selector or entry["selector"] == selector
        assert (tmp_path / entry["source_file"]).read_text() == (revision.root / path).read_text()


def test_matrix_points_carry_the_fields_the_snapshot_reads(agentic_matrix):
    _, matrix = agentic_matrix
    for point in matrix:
        assert set(POINT_FIELDS) <= point.keys(), point.get("exp-name")


def test_snapshot_of_head_generates_points_with_fingerprints(agentic_matrix):
    _, matrix = agentic_matrix
    model, precision = matrix[0]["model-prefix"], matrix[0]["precision"]

    with snapshot("HEAD") as revision:
        points = curve.points(revision, [model], precision)

    assert points and all(point.get("recipe-fingerprint") for point in points)


class Served:
    def __init__(self, payload):
        self.body, self.status = json.dumps(payload).encode(), 200
        self.headers = {"Content-Type": "application/json"}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, size):
        return self.body[:size]


@pytest.mark.parametrize("resource,arguments,url", [
    ("options", {}, "/api/v1/views/options"),
    ("benchmarks", {"model": "GLM-5.2"}, "/api/v1/benchmarks?model=GLM-5.2"),
    ("workflow-info", {"date": "2026-10-20", "benchmark_type": "agentic_traces"},
     "/api/v1/workflow-info?date=2026-10-20&benchmarkType=agentic_traces"),
    ("workflow-info", {"date": "2026-10-20"}, "/api/v1/workflow-info?date=2026-10-20"),
])
def test_api_client_reads_each_resource_the_snapshot_uses(resource, arguments, url):
    requested = []

    def opener(request, timeout):
        requested.append(request.full_url)
        return Served({"resource": resource})

    feed = api.fetch(resource, opener=opener, **arguments)

    assert requested == [api.PUBLIC + url] and feed.url == api.PUBLIC + url
    assert feed.payload == {"resource": resource}


def run_main(monkeypatch, tmp_path, flags=("--agentic",), **served):
    payloads = {"options": OPTIONS, "benchmarks": BENCHMARKS, "evaluations": EVALUATIONS,
                "workflow-info": WORKFLOW_INFO, **served}

    def fetch(resource, **arguments):
        if isinstance(payloads[resource], Exception):
            raise payloads[resource]
        return SimpleNamespace(payload=payloads[resource], url=f"{resource}{arguments}", sha256="0")

    monkeypatch.setattr(curve, "fetch", fetch)
    code = curve.main(["--model", "GLM-5.3", "--hardware", "gb300", "--framework", "dynamo-sglang",
                       "--precision", "fp4", "--output-dir", str(tmp_path / "out"), *flags])
    snapshot = tmp_path / "out/snapshot.json"
    return code, json.loads(snapshot.read_text()) if snapshot.exists() else None


def test_missing_optional_fields_degrade_their_part_and_say_why(monkeypatch, tmp_path):
    renamed = [{**row, "spec": row.get("spec_method"), "metrics": {} if row["id"] == "b" else
                row["metrics"]} for row in BENCHMARKS]
    options = {**OPTIONS, "hardware": [{"key": "gb300"}], "added": True}

    code, snapshot = run_main(monkeypatch, tmp_path, options=options, benchmarks=renamed,
                              evaluations=api.ReadError("http-503"))

    rows = {row["id"]: row for row in snapshot["rows"]}
    assert code == 4
    assert rows["a"]["tok_per_usd"] is None and rows["a"]["on_frontier"] is True
    assert rows["b"]["on_frontier"] is None and snapshot["evaluations"] == []
    assert len(snapshot["warnings"]) == 3


def test_other_frontier_axes_leave_the_default_frontier_as_it_was(monkeypatch, tmp_path):
    _, default = run_main(monkeypatch, tmp_path / "default")
    _, other = run_main(monkeypatch, tmp_path / "other",
                        flags=["--agentic", "--frontier", "p90_ttft,tok_per_usd",
                               "--frontier", "p90_intvty,tok_per_usd"])

    assert "frontiers" not in default["rows"][0] and "frontier" not in default["target"]
    assert [row["on_frontier"] for row in other["rows"]] == [
        row["on_frontier"] for row in default["rows"]]
    assert [row["frontiers"] for row in other["rows"]] == [
        {"p90_ttft,tok_per_usd": None, "p90_intvty,tok_per_usd": row["on_frontier"]}
        for row in default["rows"]]


UNTYPED = [{k: v for k, v in row("a", False).items() if k != "benchmark_type"}]


@pytest.mark.parametrize("served,endpoint,field,flags", [
    ({"options": {"models": [{"name": "GLM-5.2"}]}}, "/api/v1/views/options", "'dbKeys'", []),
    ({"benchmarks": UNTYPED}, "/api/v1/benchmarks", "'benchmark_type'", []),
    ({"benchmarks": UNTYPED}, "/api/v1/benchmarks", "'benchmark_type'", ["--agentic"]),
])
def test_a_missing_core_field_fails_naming_the_endpoint_and_field(
        monkeypatch, tmp_path, capsys, served, endpoint, field, flags):
    code, snapshot = run_main(monkeypatch, tmp_path, flags=flags, **served)

    assert code == 3 and snapshot is None
    message = capsys.readouterr().err
    assert endpoint in message and field in message


def test_a_requested_curve_with_rows_but_none_bound_fails_saying_why(monkeypatch, tmp_path):
    code, snapshot = run_main(monkeypatch, tmp_path)

    assert code == 4 and snapshot["variants"] == []
    assert [(row["id"], row["inferencex_commit"], row["variant"], row["unbound"])
            for row in snapshot["rows"]] == [
        (id, None, None, "workflow-info lists no single head commit for the run") for id in "ab"]
    assert snapshot["evaluations"] == [EVALUATIONS[0]]
    assert run_main(monkeypatch, tmp_path / "empty", benchmarks=[])[0] == 0
