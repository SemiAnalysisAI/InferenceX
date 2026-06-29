"""Tests for run_offline's decode-step metric computation, with the engine
and prompt construction mocked out."""

import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
_ENGINES_DIR = _THIS_DIR / "engines"
if str(_ENGINES_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINES_DIR))

import pytest

import run_offline
from latency_utils import (
    request_rounds,
    timed_max_tokens,
    tokens_per_round_estimate,
)


class TestStepHelpers:
    def test_timed_max_tokens_non_spec_is_identity(self):
        assert timed_max_tokens(256, 1.0) == 256

    def test_timed_max_tokens_scales_with_acceptance(self):
        # 255 rounds × 2.44 tokens/round + the prefill token.
        assert timed_max_tokens(256, 2.44) == 1 + round(255 * 2.44)

    def test_timed_max_tokens_never_below_steps(self):
        assert timed_max_tokens(256, 0.5) == 256

    def test_request_rounds_non_spec_equals_tokens(self):
        assert request_rounds(255, None, mtp=0) == 255

    def test_request_rounds_spec_requires_telemetry(self):
        assert request_rounds(622, None, mtp=3) is None
        assert request_rounds(622, 255, mtp=3) == 255

    def test_tokens_per_round_estimate(self):
        assert tokens_per_round_estimate(622, 255) == pytest.approx(622 / 255)
        assert tokens_per_round_estimate(0, 0) == 1.0


def _run_main(tmp_path, monkeypatch, engine_metrics, mtp):
    batch = 8
    monkeypatch.setattr(run_offline, "_load_tokenizer", lambda *a, **k: object())
    monkeypatch.setattr(
        run_offline, "build_infinitebench_prompts",
        lambda **kw: [("p", 8192, kw["output_len"])] * kw["num_prompts"])
    monkeypatch.setattr(run_offline, "_engine_run", lambda args, prompts: engine_metrics)
    monkeypatch.setattr(sys, "argv", [
        "run_offline.py",
        "--engine", "sglang",
        "--model", "fake/model",
        "--tp", "8", "--ep", "8", "--num-chips", "8",
        "--mtp", str(mtp),
        "--decode-steps", "256",
        "--batch-size", str(batch),
        "--result-dir", str(tmp_path),
        "--result-filename", "result",
    ])
    run_offline.main()
    return json.loads((tmp_path / "result.json").read_text())


@pytest.fixture
def spec_metrics():
    """8 requests, 255 decode rounds each at 19.03 ms/round, 2.44
    tokens/round."""
    n = 8
    rounds = 255
    tokens = round(rounds * 2.44)
    return {
        "warmup_seconds": 30.0,
        "timed_seconds": rounds * 0.01903 + 2.0,
        "total_output_tokens": n * (tokens + 1),
        "total_input_tokens": n * 8192,
        "ttfts_s": [1.5] * n,
        "tpots_s": [0.01903 / 2.44] * n,
        "tpot_steps_s": [0.01903] * n,
        "e2els_s": [rounds * 0.01903 + 1.5] * n,
        "decode_tokens_per_req": [tokens] * n,
        "decode_rounds_per_req": [rounds] * n,
        "decode_tokens_total": n * tokens,
        "decode_rounds_total": n * rounds,
        "warmup_tokens_per_round": 2.44,
        "timed_max_tokens": 1 + tokens,
        "latency_metrics_source": "sglang_meta_info",
        "moe_routing": "simulated_uniform_random",
    }


class TestRunOfflineStepMetrics:
    def test_headline_tpot_is_per_step(self, tmp_path, monkeypatch, spec_metrics):
        result = _run_main(tmp_path, monkeypatch, spec_metrics, mtp=3)
        assert result["tpot_unit"] == "decode_step"
        assert result["mean_tpot_ms"] == pytest.approx(19.03)
        assert result["median_tpot_ms"] == pytest.approx(19.03)

    def test_throughput_is_steps_per_second(self, tmp_path, monkeypatch, spec_metrics):
        result = _run_main(tmp_path, monkeypatch, spec_metrics, mtp=3)
        # batch=8, chips=8 -> 1 request/chip; 1/0.01903 steps/s/chip.
        assert result["output_throughput"] == pytest.approx(8 * 1000.0 / 19.03)
        assert result["decode_step_throughput_per_chip"] == pytest.approx(1000.0 / 19.03)

    def test_acceptance_is_reported_not_folded_in(self, tmp_path, monkeypatch, spec_metrics):
        result = _run_main(tmp_path, monkeypatch, spec_metrics, mtp=3)
        assert result["spec_tokens_per_step_observed"] == pytest.approx(2.44, abs=0.01)
        assert result["mtp_accepted_per_step_observed"] == pytest.approx(1.44, abs=0.01)
        # Per-token TPOT kept as a secondary metric only.
        assert result["mean_tpot_per_token_ms"] == pytest.approx(19.03 / 2.44)

    def test_result_carries_step_workload_shape(self, tmp_path, monkeypatch, spec_metrics):
        result = _run_main(tmp_path, monkeypatch, spec_metrics, mtp=3)
        assert result["engine_mode"] == "offline"
        assert result["decode_steps_target"] == 256
        assert result["benchmark_output_len"] == 256
        assert result["mtp"] == 3

    def test_spec_run_without_step_telemetry_is_labeled(self, tmp_path, monkeypatch, spec_metrics):
        spec_metrics["tpot_steps_s"] = []
        spec_metrics["decode_rounds_per_req"] = []
        spec_metrics["decode_rounds_total"] = 0
        result = _run_main(tmp_path, monkeypatch, spec_metrics, mtp=3)
        assert result["tpot_unit"] == "output_token_fallback"
        assert result["latency_metrics_source"].endswith("_partial")

    def test_non_spec_token_samples_are_step_samples(self, tmp_path, monkeypatch):
        n = 8
        metrics = {
            "warmup_seconds": 30.0,
            "timed_seconds": 10.0,
            "total_output_tokens": n * 256,
            "total_input_tokens": n * 8192,
            "ttfts_s": [1.0] * n,
            "tpots_s": [0.025] * n,
            "tpot_steps_s": [],
            "e2els_s": [7.4] * n,
            "decode_tokens_per_req": [255] * n,
            "decode_rounds_per_req": [255] * n,
            "decode_tokens_total": n * 255,
            "decode_rounds_total": n * 255,
            "latency_metrics_source": "sglang_meta_info",
        }
        result = _run_main(tmp_path, monkeypatch, metrics, mtp=0)
        # One token per round: per-token samples ARE per-step samples.
        assert result["tpot_unit"] == "decode_step"
        assert result["mean_tpot_ms"] == pytest.approx(25.0)


class TestRoutingLabel:
    def test_simulated_routing_is_carried_through(self, tmp_path, monkeypatch, spec_metrics):
        result = _run_main(tmp_path, monkeypatch, spec_metrics, mtp=3)
        assert result["moe_routing"] == "simulated_uniform_random"

    def test_engine_without_sim_support_never_claims_it(self, tmp_path, monkeypatch, spec_metrics):
        """An engine that ignores the request reports its own label (or the
        orchestrator marks it unknown) — a simulated label can only come
        from the engine driver that actually applied it."""
        spec_metrics["moe_routing"] = "real"
        result = _run_main(tmp_path, monkeypatch, spec_metrics, mtp=3)
        assert result["moe_routing"] == "real"

        spec_metrics.pop("moe_routing")
        result = _run_main(tmp_path, monkeypatch, spec_metrics, mtp=3)
        assert result["moe_routing"] == "unknown"
