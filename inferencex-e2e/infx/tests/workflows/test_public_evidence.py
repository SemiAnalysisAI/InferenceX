"""Exercise the verifier's public-evidence lookups against controlled feeds."""

import json

from infx.klaud.models import Feed
from infx.workflows import public_evidence

SERIES = ["--model", "m", "--precision", "fp4", "--framework", "sglang", "--hardware", "b200"]
SERIES += ["--spec-method", "mtp"]


def row(run, date, metrics, **overrides):
    return {
        "model": "m",
        "precision": "fp4",
        "framework": "sglang",
        "hardware": "b200",
        "spec_method": "mtp",
        "task": "gsm8k",
        "date": date,
        "conc": 64,
        "decode_tp": 8,
        "metrics": metrics,
        "run_url": f"https://github.com/o/r/actions/runs/{run}",
        **overrides,
    }


def run(monkeypatch, capsys, rows, argv):
    feed = Feed(url="https://x/api", retrieved_at="2026-10-24T00:00:00Z", sha256="0" * 64, payload=rows)
    calls = []

    def fetch(resource, **kwargs):
        calls.append((resource, kwargs))
        return feed

    monkeypatch.setattr(public_evidence.api, "fetch", fetch)
    public_evidence.main(argv)
    return json.loads(capsys.readouterr().out), calls


def test_evals_split_exact_and_wider_peers_newest_first_and_limited(monkeypatch, capsys):
    score = {"em_strict": 0.95, "em_strict_se": 0.006, "n_eff": 1319}
    rows = [
        row(1, "2026-10-01", score),
        row(2, "2026-10-20", score),
        row(3, "2026-10-10", score),
        row(4, "2026-10-21", score, hardware="mi355x"),
        row(5, "2026-10-22", score, spec_method="none"),
        row(6, "2026-10-23", score, precision="fp8"),
        row(7, "2026-10-23", score, task="minimax_m3_full"),
        "not-a-row",
    ]
    argv = ["evals", *SERIES, "--task", "gsm8k", "--limit", "2"]
    report, calls = run(monkeypatch, capsys, rows, argv)
    assert calls == [("evaluations", {})]
    assert [r["run_url"][-1] for r in report["exact"]] == ["2", "3"]
    assert report["exact_total"] == 3
    assert [r["run_url"][-1] for r in report["same_model_precision_task"]] == ["5", "4"]
    assert report["same_model_precision_task_total"] == 2
    assert report["exact"][0]["em_strict_se"] == 0.006
    assert report["exact"][0]["decode_tp"] == 8


def test_points_keep_only_the_requested_series_with_frontier_metrics(monkeypatch, capsys):
    metrics = {"tput_per_gpu": 812.5, "mean_intvty": 41.0, "p90_e2el": 9.5, "ttft": 1}
    rows = [
        row(1, "2026-10-01", metrics, image="lmsysorg/sglang:v1"),
        row(2, "2026-10-02", metrics, image="lmsysorg/sglang:v2"),
        row(3, "2026-10-03", metrics, framework="vllm"),
    ]
    argv = ["points", "--model-name", "M-Model", *SERIES]
    report, calls = run(monkeypatch, capsys, rows, argv)
    assert calls == [("benchmarks", {"model": "M-Model"})]
    assert report["points_total"] == 2
    assert [p["image"] for p in report["points"]] == ["lmsysorg/sglang:v2", "lmsysorg/sglang:v1"]
    assert report["points"][0]["tput_per_gpu"] == 812.5
    assert report["points"][0]["mean_intvty"] == 41.0
    assert "ttft" not in report["points"][0]
