from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from infx import github
from infx.workflows import al_gate_reuse, sweep_runs


def gate(gate_id: str) -> dict:
    return {"recipe-fingerprint": "fp", "al-gate": {"id": gate_id, "curve": "dsv4_mtp"}}


def bundle(result: dict) -> bytes:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as files:
        files.writestr(f"al_gate_{result['result_filename']}.json", json.dumps(result))
        files.writestr("al_gate/metrics_after.json", "{}")
    return archive.getvalue()


def run_reuse(tmp_path: Path, plan: dict) -> tuple[dict, dict[str, dict]]:
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    argv = [
        str(tmp_path / "plan.json"), "--code-version", "abc", "--repo", "o/r", "--branch", "feat/x",
        "--output", str(tmp_path / "gates.json"), "--results", str(tmp_path / "reused"),
    ]  # fmt: skip
    assert al_gate_reuse.main(argv) == 0
    reused = {
        path.parent.name: json.loads(path.read_text())
        for path in (tmp_path / "reused").glob("*/al_gate_*.json")
    }
    return json.loads((tmp_path / "gates.json").read_text()), reused


def test_the_newest_measured_result_of_each_gate_is_reused(tmp_path, monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "token")
    plan = {"single_node": [gate("a"), gate("b")], "multi_node": [gate("c")], "unresolved": []}
    key = {g["al-gate"]["id"]: al_gate_reuse.gate_key(g, "abc") for g in
           plan["single_node"] + plan["multi_node"]}  # fmt: skip
    results = {
        "new-a": {"result_filename": "a", "status": "error"},
        "new-b": {"result_filename": "b", "status": "below"},
        "new-c": {"result_filename": "c", "status": "pass"},
        "old-a": {"result_filename": "a", "status": "pass"},
        "old-b": {"result_filename": "b", "status": "pass"},
    }
    artifacts = {
        2: [{"id": "new-a", "name": f"algate_{key['a']}"}, {"id": "new-b", "name": f"algate_{key['b']}"},
            {"id": "new-c", "name": f"algate_{key['c']}", "expired": True}],
        1: [{"id": "old-a", "name": f"algate_{key['a']}"}, {"id": "old-b", "name": f"algate_{key['b']}"},
            {"id": "plan", "name": "algate_plan"}],
    }  # fmt: skip
    monkeypatch.setattr(
        sweep_runs, "completed_pr_runs",
        lambda *_: [{"id": 1, "created_at": "2026-10-01"}, {"id": 2, "created_at": "2026-10-02"}],
    )  # fmt: skip
    monkeypatch.setattr(
        github, "paginate", lambda repo, path, *_: artifacts[int(path.split("/")[3])]
    )
    monkeypatch.setattr(
        github, "download", lambda repo, path, *_: bundle(results[path.split("/")[3]])
    )

    gates, reused = run_reuse(tmp_path, plan)

    assert reused == {key["a"]: results["old-a"], key["b"]: results["new-b"]}
    assert gates["single_node"] == []
    assert gates["multi_node"] == [
        {**gate("c"), "al-gate": {**gate("c")["al-gate"], "key": key["c"]}}
    ]


def test_a_failed_lookup_reruns_every_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "token")

    def unavailable(*_):
        raise github.APIError("/actions/workflows", "HTTP 502 (HTTP 502)")

    monkeypatch.setattr(sweep_runs, "completed_pr_runs", unavailable)
    plan = {"single_node": [gate("a")], "multi_node": [], "unresolved": []}
    gates, reused = run_reuse(tmp_path, plan)
    assert (len(gates["single_node"]), reused) == (1, {})
