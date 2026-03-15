"""Focused tests for summarize.py multinode throughput normalization."""

import importlib
import sys
import types

import pytest


# summarize.py depends on tabulate at import time. Stub it for unit tests.
if "tabulate" not in sys.modules:
    tabulate_stub = types.ModuleType("tabulate")
    tabulate_stub.tabulate = lambda *args, **kwargs: ""
    sys.modules["tabulate"] = tabulate_stub

summarize = importlib.import_module("summarize")


def test_get_multinode_tput_metrics_new_schema_passthrough():
    """New schema should preserve cluster + role-scoped values as-is."""
    result = {
        "output_tput_per_gpu": 1911.56,
        "input_tput_per_gpu": 1910.71,
        "output_tput_per_decode_gpu": 2867.34,
        "input_tput_per_prefill_gpu": 5732.13,
        "num_prefill_gpu": 24,
        "num_decode_gpu": 48,
    }

    output_cluster, input_cluster, output_decode, input_prefill = summarize.get_multinode_tput_metrics(result)

    assert output_cluster == pytest.approx(1911.56)
    assert input_cluster == pytest.approx(1910.71)
    assert output_decode == pytest.approx(2867.34)
    assert input_prefill == pytest.approx(5732.13)


def test_get_multinode_tput_metrics_legacy_schema_normalized_to_cluster_avg():
    """Legacy role-scoped fields should be normalized to cluster averages."""
    result = {
        # Legacy meaning: per-decode GPU and per-prefill GPU
        "output_tput_per_gpu": 2867.34,
        "input_tput_per_gpu": 5732.13,
        "num_prefill_gpu": 24,
        "num_decode_gpu": 48,
    }

    output_cluster, input_cluster, output_decode, input_prefill = summarize.get_multinode_tput_metrics(result)

    total_gpus = 24 + 48
    assert output_cluster == pytest.approx(2867.34 * (48 / total_gpus))
    assert input_cluster == pytest.approx(5732.13 * (24 / total_gpus))
    assert output_decode == pytest.approx(2867.34)
    assert input_prefill == pytest.approx(5732.13)


def test_get_multinode_tput_metrics_legacy_schema_zero_gpu_count():
    """Legacy data with zero GPU counts should not divide by zero."""
    result = {
        "output_tput_per_gpu": 1000.0,
        "input_tput_per_gpu": 500.0,
        "num_prefill_gpu": 0,
        "num_decode_gpu": 0,
    }

    output_cluster, input_cluster, output_decode, input_prefill = summarize.get_multinode_tput_metrics(result)

    assert output_cluster == pytest.approx(1000.0)
    assert input_cluster == pytest.approx(500.0)
    assert output_decode == pytest.approx(1000.0)
    assert input_prefill == pytest.approx(500.0)
