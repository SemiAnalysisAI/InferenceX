from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from infx.datasets import build_weka_hf_dataset
from infx.datasets.build_weka_hf_dataset import _filter_trace_256k

STUB_SAMPLER = """
import json, os, sys
from pathlib import Path

argv = sys.argv[1:]
out = Path(argv[argv.index("--out") + 1])
Path(os.environ["STUB_SAMPLER_RECORD"]).write_text(
    json.dumps({"argv": argv, "db_url": os.environ.get("AGENTIC_PROXY_DB_URL")})
)
for n, sid in enumerate(os.environ["STUB_SESSIONS"].split(",")):
    if n == int(os.environ.get("STUB_SAMPLER_CRASH_AT", "-1")):
        sys.exit(1)
    row = {
        "timestamp": "2026-05-01T00:00:00+00:00",
        "t_sec": 0.0,
        "model": "claude-test",
        "input_tokens": 10,
        "output_tokens": 5,
        "duration_ms": 1000,
        "hash_ids": ["h1"],
    }
    if sid == os.environ.get("STUB_BAD_SESSION"):
        del row["model"]
    (out / f"{sid}.jsonl").write_text(json.dumps(row) + "\\n")
(out / "manifest.json").write_text("{}")
"""


STUB_PLOT = """
import os, sys
from pathlib import Path

if os.environ.get("STUB_PLOT_FAIL"):
    sys.exit(1)
out = Path(sys.argv[sys.argv.index("--out-dir") + 1])
(out / f"{Path(sys.argv[0]).stem}.png").write_bytes(b"png")
"""


@pytest.fixture
def build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Run the build CLI with the DB sampler stubbed to write STUB_SESSIONS and the plot
    scripts stubbed to write a PNG, or to fail when STUB_PLOT_FAIL is set."""
    sampler = tmp_path / "stub_sampler.py"
    sampler.write_text(STUB_SAMPLER)
    record = tmp_path / "sampler-record.json"
    plot = tmp_path / "stub_plot.py"
    plot.write_text(STUB_PLOT)
    monkeypatch.delenv("STUB_PLOT_FAIL", raising=False)
    monkeypatch.delenv("STUB_SAMPLER_CRASH_AT", raising=False)
    monkeypatch.delenv("STUB_BAD_SESSION", raising=False)
    monkeypatch.setattr(build_weka_hf_dataset, "SAMPLER", sampler)
    monkeypatch.setattr(build_weka_hf_dataset, "PLOT_WEKA", plot)
    monkeypatch.setattr(build_weka_hf_dataset, "PLOT_SUBAGENT", plot)
    monkeypatch.setenv("STUB_SAMPLER_RECORD", str(record))
    # setenv first so teardown also removes the value main() exports.
    monkeypatch.setenv("AGENTIC_PROXY_DB_URL", "")
    monkeypatch.delenv("AGENTIC_PROXY_DB_URL")

    def run(sessions: str, *flags: str) -> dict:
        monkeypatch.setenv("STUB_SESSIONS", sessions)
        argv = ["build", "--repo-base", "org/base", "--work-dir", str(tmp_path / "work")]
        monkeypatch.setattr(sys, "argv", [*argv, "--skip-upload", *flags])
        assert build_weka_hf_dataset.main() == 0
        return json.loads(record.read_text())

    return run


def _trace_ids(traces_jsonl: Path) -> list[str]:
    return [json.loads(line)["id"] for line in traces_jsonl.read_text().splitlines()]


def test_rebuild_in_same_work_dir_publishes_only_this_runs_sessions(build, tmp_path: Path) -> None:
    filters = ("--repo-256k", "org/256k", "--base-max-isl", "990016")
    build("old-a,old-b", *filters)
    build("new-c", *filters)

    work = tmp_path / "work"
    assert _trace_ids(work / "base/traces.jsonl") == ["new-c"]
    assert _trace_ids(work / "256k/traces.jsonl") == ["new-c"]


def test_failed_plots_do_not_republish_the_previous_builds_pngs(
    build, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build("s1")
    plots = tmp_path / "work/base/plots"
    assert [p.name for p in plots.iterdir()] == ["stub_plot.png"]

    monkeypatch.setenv("STUB_PLOT_FAIL", "1")
    build("s2")

    assert list(plots.iterdir()) == []
    assert _trace_ids(tmp_path / "work/base/traces.jsonl") == ["s2"]


def test_skip_sample_does_not_reuse_a_crashed_samplers_partial_dump(
    build, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STUB_SAMPLER_CRASH_AT", "1")
    with pytest.raises(subprocess.CalledProcessError):
        build("partial-a,partial-b")
    monkeypatch.delenv("STUB_SAMPLER_CRASH_AT")

    build("full-a,full-b", "--skip-sample")

    assert _trace_ids(tmp_path / "work/base/traces.jsonl") == ["full-a", "full-b"]


def test_skip_convert_does_not_publish_a_crashed_converters_partial_output(
    build, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STUB_BAD_SESSION", "b-bad")
    with pytest.raises(subprocess.CalledProcessError):
        build("a-good,b-bad")
    assert (tmp_path / "work/per_trace/a-good.json").is_file()

    with pytest.raises(subprocess.CalledProcessError):
        build("unused", "--skip-sample", "--skip-convert")

    assert not (tmp_path / "work/base").exists()


def test_db_url_reaches_sampler_without_being_echoed(
    build, capfd: pytest.CaptureFixture[str]
) -> None:
    url = "postgresql://reader:hunter2-secret@db.example/proxy"
    record = build("s1", "--db-url", url)

    assert record["db_url"] == url
    assert all("hunter2-secret" not in arg for arg in record["argv"])
    out, err = capfd.readouterr()
    assert "hunter2-secret" not in out + err


def test_random_sampling_forwards_seed_to_sampler_and_readme(build, tmp_path: Path) -> None:
    record = build("s1", "--sampling", "random", "--seed", "7", "--limit", "3")

    argv = record["argv"]
    assert argv[argv.index("--sampling") + 1] == "random"
    assert argv[argv.index("--seed") + 1] == "7"
    readme = (tmp_path / "work/base/README.md").read_text()
    assert "--seed 7" in readme


def test_dwbug_min_peak_without_the_filter_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["build", "--repo-base", "org/base", "--work-dir", str(tmp_path), "--dwbug-min-peak", "2"]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exit_info:
        build_weka_hf_dataset.parse_args()

    assert exit_info.value.code == 2
    assert "--dwbug-min-peak requires --exclude-dynamic-workflow-bug" in capsys.readouterr().err


def _request(t: float, api_time: float, tokens: int = 1) -> dict:
    return {
        "t": t,
        "type": "n",
        "model": "test-model",
        "in": tokens,
        "out": 0,
        "hash_ids": [],
        "api_time": api_time,
    }


def _subagent(agent_id: str, requests: list[dict]) -> dict:
    first_t = min(req["t"] for req in requests)
    last_end = max(req["t"] + req["api_time"] for req in requests)
    return {
        "t": first_t,
        "type": "subagent",
        "agent_id": agent_id,
        "duration_ms": int(round((last_end - first_t) * 1000)),
        "total_tokens": sum(req["in"] + req["out"] for req in requests),
        "requests": requests,
    }


def test_filter_without_drops_preserves_overlapping_subagents_exactly() -> None:
    trace = {
        "id": "overlap",
        "requests": [
            _request(0.0, 1.0),
            _subagent("child-a", [_request(2.0, 5.0)]),
            _subagent("child-b", [_request(3.0, 5.0)]),
        ],
    }

    filtered = _filter_trace_256k(trace, cap=256_000)

    assert filtered == trace


def test_filter_uses_uniform_shift_and_preserves_overlap() -> None:
    trace = {
        "id": "shifted-overlap",
        "requests": [
            _request(0.0, 10.0, tokens=300_000),
            _subagent("child-a", [_request(4.0, 5.0)]),
            _subagent("child-b", [_request(5.0, 5.0)]),
        ],
    }

    filtered = _filter_trace_256k(trace, cap=256_000)

    child_a, child_b = filtered["requests"]
    assert child_a["t"] == 0.0
    assert child_b["t"] == 1.0
    assert child_a["requests"][0]["t"] == 0.0
    assert child_b["requests"][0]["t"] == 1.0
    assert child_b["t"] < child_a["t"] + child_a["duration_ms"] / 1000.0


def test_partial_subagent_filter_recomputes_group_metadata() -> None:
    trace = {
        "id": "partial",
        "requests": [
            _request(0.0, 1.0),
            _subagent(
                "child",
                [
                    _request(2.0, 3.0, tokens=300_000),
                    _request(4.0, 2.0, tokens=100),
                    _request(8.0, 1.0, tokens=200),
                ],
            ),
        ],
    }

    filtered = _filter_trace_256k(trace, cap=256_000)

    child = filtered["requests"][1]
    assert [req["t"] for req in child["requests"]] == [4.0, 8.0]
    assert child["t"] == 4.0
    assert child["duration_ms"] == 5_000
    assert child["total_tokens"] == 300


def test_filter_drops_models_left_without_requests() -> None:
    opus_main = {**_request(0.0, 1.0, tokens=300_000), "model": "opus"}
    sonnet_main = {**_request(1.0, 1.0), "model": "sonnet"}
    haiku_inner = {**_request(2.0, 1.0), "model": "haiku"}
    opus_inner = {**_request(3.0, 1.0, tokens=300_000), "model": "opus"}
    child = {**_subagent("child", [haiku_inner, opus_inner]), "models": ["haiku", "opus"]}
    trace = {"id": "models", "models": ["haiku", "opus", "sonnet"]}
    trace["requests"] = [opus_main, sonnet_main, child]

    filtered = _filter_trace_256k(trace, cap=256_000)

    assert filtered["requests"][1]["models"] == ["haiku"]
    assert filtered["models"] == ["haiku", "sonnet"]
