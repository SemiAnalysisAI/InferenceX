"""Tests for validate_agg_result.py, covering both fixed-seq and agentic agg schemas."""
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from validate_agg_result import (
    validate,
    check_identity,
    check_numeric_finite,
    check_throughput,
    check_percentile_families,
    check_monotonicity,
)

SCRIPT = Path(__file__).parent / "validate_agg_result.py"


def fixed_seq_agg():
    """Valid fixed-seq agg: all latency families + intvty at p75..p99.9 (intvty decreasing)."""
    data = {
        "hw": "b200", "framework": "sglang", "precision": "fp8",
        "model": "test/model", "infmax_model_prefix": "tm",
        "conc": 8, "isl": 1024, "osl": 1024, "is_multinode": False,
        "tp": 8, "ep": 1, "dp_attention": "false",
        "tput_per_gpu": 1000.0, "output_tput_per_gpu": 800.0, "input_tput_per_gpu": 200.0,
        "mean_tpot": 0.01, "mean_intvty": 100.0,
    }
    for i, p in enumerate((75, 90, 95, 99, 99.9), start=1):
        k = str(int(p)) if p == int(p) else str(p)
        data[f"p{k}_ttft"] = float(i)
        data[f"p{k}_tpot"] = float(i)
        data[f"p{k}_itl"] = float(i)
        data[f"p{k}_e2el"] = float(i)
        data[f"p{k}_intvty"] = 1000.0 / (i + 9)
    return data


def agentic_agg():
    """Valid agentic agg: same families at p75/p90/p95, no isl/osl, intvty increasing."""
    data = {
        "hw": "b200", "framework": "sglang", "precision": "fp8",
        "model": "test/model", "infmax_model_prefix": "tm",
        "conc": 8, "is_multinode": False,
        "tp": 8, "ep": 1, "dp_attention": "false",
        "scenario_type": "agentic-coding",
        "tput_per_gpu": 1000.0, "output_tput_per_gpu": 800.0, "input_tput_per_gpu": 200.0,
        "mean_tpot": 0.01, "mean_intvty": 100.0,
        "theoretical_cache_hit_rate": None,
    }
    for i, p in enumerate((75, 90, 95), start=1):
        data[f"p{p}_ttft"] = float(i)
        data[f"p{p}_tpot"] = float(i)
        data[f"p{p}_itl"] = float(i)
        data[f"p{p}_e2el"] = float(i)
        data[f"p{p}_intvty"] = float(i) * 10
    return data


def test_fixed_seq_valid_passes():
    assert validate(fixed_seq_agg()) == []


def test_agentic_valid_passes():
    assert validate(agentic_agg()) == []


def test_agentic_intvty_must_increase():
    data = agentic_agg()
    data["p95_intvty"] = data["p75_intvty"] - 1.0
    assert any("intvty" in e and "non-decreasing" in e for e in check_monotonicity(data))


def test_fixed_seq_intvty_must_decrease():
    data = fixed_seq_agg()
    data["p90_intvty"] = data["p75_intvty"] + 100.0
    assert any("intvty" in e and "non-increasing" in e for e in check_monotonicity(data))


def test_missing_sibling_percentile_fails():
    data = fixed_seq_agg()
    del data["p95_e2el"]
    assert any("e2el" in e and "95" in e for e in check_percentile_families(data))


def test_intvty_must_mirror_tpot():
    data = fixed_seq_agg()
    del data["p99_intvty"]
    assert any(
        "p99_tpot present but p99_intvty missing" in e
        for e in check_percentile_families(data)
    )


def test_latency_monotonicity_fails():
    data = fixed_seq_agg()
    data["p90_ttft"] = data["p75_ttft"] - 1.0
    assert any("ttft" in e and "non-decreasing" in e for e in check_monotonicity(data))


def test_negative_percentile_value_fails():
    data = fixed_seq_agg()
    data["p90_ttft"] = -1.0
    assert any("non-negative" in e for e in check_monotonicity(data))


def test_malformed_percentile_key_flagged():
    data = fixed_seq_agg()
    data["p150_tpot"] = 1.0
    assert any("malformed" in e for e in check_percentile_families(data))


def test_throughput_positive_required():
    data = fixed_seq_agg()
    data["tput_per_gpu"] = 0.0
    assert any("tput_per_gpu" in e for e in check_throughput(data))


def test_throughput_sum_is_not_asserted():
    data = fixed_seq_agg()
    data["input_tput_per_gpu"] = 123.0  # input+output need not equal total
    assert check_throughput(data) == []


def test_nan_field_fails():
    data = fixed_seq_agg()
    data["mean_tpot"] = math.nan
    assert any("finite" in e for e in check_numeric_finite(data))


def test_missing_identity_fails():
    data = fixed_seq_agg()
    data["hw"] = ""
    assert any("hw" in e for e in check_identity(data))


def test_fixed_seq_requires_isl_osl():
    data = fixed_seq_agg()
    del data["isl"]
    assert any("isl" in e for e in check_identity(data))


def test_agentic_does_not_require_isl_osl():
    assert all("isl" not in e and "osl" not in e for e in check_identity(agentic_agg()))


def test_multinode_decode_fields_may_be_zero():
    data = fixed_seq_agg()
    data["is_multinode"] = True
    for k in ("prefill_tp", "prefill_ep", "prefill_num_workers"):
        data[k] = 4
    for k in ("decode_tp", "decode_ep", "decode_num_workers"):
        data[k] = 0
    data["prefill_dp_attention"] = "true"
    data["decode_dp_attention"] = "true"
    assert check_identity(data) == []


def _run_cli(tmp_path, payload):
    path = tmp_path / "agg.json"
    path.write_text(payload)
    return subprocess.run([sys.executable, str(SCRIPT), str(path)], capture_output=True)


def test_cli_accepts_valid(tmp_path):
    assert _run_cli(tmp_path, json.dumps(fixed_seq_agg())).returncode == 0


def test_cli_rejects_non_dict_json(tmp_path):
    assert _run_cli(tmp_path, "[]").returncode == 1


def test_cli_rejects_invalid_agg(tmp_path):
    data = fixed_seq_agg()
    data["tput_per_gpu"] = -1.0
    assert _run_cli(tmp_path, json.dumps(data)).returncode == 1
