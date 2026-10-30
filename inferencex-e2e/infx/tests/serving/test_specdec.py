import json

import pytest

from infx.serving import specdec

TEXT = "\n".join([
    "# HELP vllm:spec_decode_num_drafts_total drafts",
    "# TYPE vllm:spec_decode_num_drafts_total counter",
    'vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 1.5e+03',
    'vllm:spec_decode_num_drafts_total{engine="1",model_name="m"} 500',
    "vllm:spec_decode_num_accepted_tokens_total 4000",
    "vllm:spec_decode_num_draft_tokens_total 12000",
    "vllm:num_requests_running 3",
    "vllm:bad nan",
])


def test_parse_sums_label_sets_and_skips_non_finite():
    totals = specdec.parse_prometheus(TEXT)
    assert totals["vllm:spec_decode_num_drafts_total"] == 2000
    assert "vllm:bad" not in totals


def test_acceptance_is_one_plus_accepted_per_draft_event():
    before = {"vllm:spec_decode_num_accepted_tokens_total": 100.0, "vllm:spec_decode_num_drafts_total": 50.0,
              "vllm:spec_decode_num_draft_tokens_total": 300.0}
    after = {"vllm:spec_decode_num_accepted_tokens_total": 1100.0, "vllm:spec_decode_num_drafts_total": 550.0,
             "vllm:spec_decode_num_draft_tokens_total": 3300.0}
    result = specdec.acceptance(before, after)
    assert result["acceptance_length"] == pytest.approx(3.0)
    assert result["acceptance_rate"] == pytest.approx(1000 / 3000)
    assert result["spec_draft_events"] == 500 and result["draft_depth"] == 6


def test_no_series_means_no_speculative_decoding():
    assert specdec.acceptance({}, {}) is None
    assert specdec.acceptance({"vllm:num_requests_running": 1.0}, {}) is None


def test_sglang_gauge_is_read_after_the_run():
    result = specdec.acceptance({}, {"sglang:spec_accept_length": 2.4})
    assert result["acceptance_length"] == 2.4


def test_sglang_gauge_of_zero_is_no_speculative_decoding():
    assert specdec.acceptance({"sglang:spec_accept_length": 0.0}, {"sglang:spec_accept_length": 0.0}) is None


def test_summarize_and_merge(tmp_path):
    before, after, agg = tmp_path / "b.json", tmp_path / "a.json", tmp_path / "agg.json"
    assert specdec.summarize(before, after) is None
    specdec.save(before, {"vllm:spec_decode_num_drafts_total": 0.0})
    specdec.save(after, {"vllm:spec_decode_num_drafts_total": 10.0, "vllm:spec_decode_num_accepted_tokens_total": 20.0})
    agg.write_text(json.dumps({"scenario": "x"}))
    specdec.merge(agg, specdec.summarize(before, after))
    assert json.loads(agg.read_text())["spec_decode"]["acceptance_length"] == pytest.approx(3.0)
    specdec.merge(tmp_path / "missing.json", {})  # no aggregate, nothing to do
