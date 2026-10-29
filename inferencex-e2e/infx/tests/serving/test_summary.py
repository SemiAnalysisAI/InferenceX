import json

from infx.serving.summary import load, main, table


def _result(spec: str, tput: float, **extra) -> dict:
    return {
        "scenario": "greenspec-diverse", "spec_decoding": spec, "conc": 16, "num_requests_successful": 99,
        "num_requests_total": 100,
        "request_metrics": {"throughput": {"output": {"tokens_per_second": tput}},
                            "latency": {"ttft": {"p50": 0.2, "p95": 1.5}, "tpot": {"mean": 0.02}}},
        **extra,
    }


def test_speculative_row_gets_acceptance_and_speedup_over_the_base(tmp_path):
    (tmp_path / "bmk_a").mkdir()
    (tmp_path / "bmk_a/base.json").write_text(json.dumps(_result("none", 100.0)))
    (tmp_path / "bmk_a/mtp.json").write_text(json.dumps(
        _result("mtp", 180.0, spec_decode={"acceptance_length": 3.25, "acceptance_rate": 0.375, "draft_depth": 6})))
    (tmp_path / "bmk_a/other.json").write_text(json.dumps({"hello": 1}))  # not an aggregate
    lines = table(load(tmp_path)).splitlines()
    assert len(lines) == 4  # header, rule, two points
    assert "| greenspec-diverse | mtpx6 | 16 | 99/100 | 180.0 | 200 / 1,500 | 20.0 | 3.25 | 37.5% | 1.80x |" in lines
    assert "| greenspec-diverse | none | 16 | 99/100 | 100.0 | 200 / 1,500 | 20.0 | - | - | - |" in lines


def test_a_missing_number_is_a_dash_and_an_empty_dir_says_so(tmp_path, capsys):
    (tmp_path / "x.json").write_text(json.dumps({"scenario": "s", "request_metrics": {}}))
    assert table(load(tmp_path)).count(" - ") >= 6
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main([str(empty)]) == 0
    assert "No aggregate result" in capsys.readouterr().out
