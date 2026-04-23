"""Tests for process_changelog.reduce_conc_to_endpoints."""
from process_changelog import reduce_conc_to_endpoints


_SEQ_LEN_STR = {(1024, 1024): "1k1k", (8192, 1024): "8k1k"}


def _sn_entry(conc, tp=1, isl=1024, osl=1024, run_eval=False, **overrides):
    """Build a single-node entry matching generate_test_config_sweep output."""
    entry = {
        "image": "test-image:latest",
        "model": "test-model",
        "model-prefix": "test",
        "precision": "fp8",
        "framework": "sglang",
        "runner": "h200",
        "isl": isl,
        "osl": osl,
        "tp": tp,
        "ep": 1,
        "dp-attn": False,
        "spec-decoding": "none",
        "max-model-len": isl + osl + 256,
        "exp-name": f"test_{_SEQ_LEN_STR[(isl, osl)]}",
        "disagg": False,
        "run-eval": run_eval,
        "conc": conc,
    }
    entry.update(overrides)
    return entry


def _mn_entry(conc_list, isl=1024, osl=1024):
    """Build a multi-node entry matching generate_test_config_sweep output."""
    return {
        "image": "test-image:latest",
        "model": "test-model",
        "model-prefix": "test",
        "precision": "fp4",
        "framework": "dynamo-trt",
        "runner": "gb200",
        "isl": isl,
        "osl": osl,
        "spec-decoding": "none",
        "prefill": {"num-worker": 2, "tp": 4, "ep": 4, "dp-attn": True},
        "decode": {"num-worker": 1, "tp": 8, "ep": 8, "dp-attn": True},
        "conc": conc_list,
        "max-model-len": isl + osl + 256,
        "exp-name": f"test_{_SEQ_LEN_STR[(isl, osl)]}",
        "disagg": False,
        "run-eval": False,
    }


class TestReduceConcToEndpointsSingleNode:
    def test_ascending_series_keeps_first_and_last(self):
        entries = [_sn_entry(c) for c in [4, 8, 16, 32, 64]]
        result = reduce_conc_to_endpoints(entries)
        assert [e["conc"] for e in result] == [4, 64]

    def test_descending_series_preserves_order(self):
        entries = [_sn_entry(c) for c in [32, 16, 8, 4, 2]]
        result = reduce_conc_to_endpoints(entries)
        assert [e["conc"] for e in result] == [32, 2]

    def test_two_point_series_unchanged(self):
        entries = [_sn_entry(4), _sn_entry(64)]
        result = reduce_conc_to_endpoints(entries)
        assert [e["conc"] for e in result] == [4, 64]

    def test_single_entry_unchanged(self):
        entries = [_sn_entry(32)]
        result = reduce_conc_to_endpoints(entries)
        assert [e["conc"] for e in result] == [32]

    def test_different_tp_grouped_independently(self):
        entries = (
            [_sn_entry(c, tp=1) for c in [4, 8, 16, 64]]
            + [_sn_entry(c, tp=2) for c in [8, 16, 32]]
        )
        result = reduce_conc_to_endpoints(entries)
        assert [(e["tp"], e["conc"]) for e in result] == [
            (1, 4), (1, 64), (2, 8), (2, 32),
        ]

    def test_different_seq_len_grouped_independently(self):
        entries = (
            [_sn_entry(c, isl=1024, osl=1024) for c in [4, 16, 64]]
            + [_sn_entry(c, isl=8192, osl=1024) for c in [4, 16, 64]]
        )
        result = reduce_conc_to_endpoints(entries)
        assert [(e["isl"], e["conc"]) for e in result] == [
            (1024, 4), (1024, 64), (8192, 4), (8192, 64),
        ]

    def test_run_eval_entries_not_collapsed_with_benchmark_entries(self):
        """run-eval is part of the group key, so eval and benchmark entries
        that share all other fields must remain distinct."""
        entries = [
            _sn_entry(4, run_eval=False),
            _sn_entry(64, run_eval=False),
            _sn_entry(16, run_eval=True),
            _sn_entry(64, run_eval=True),
        ]
        result = reduce_conc_to_endpoints(entries)
        assert [(e["run-eval"], e["conc"]) for e in result] == [
            (False, 4), (False, 64), (True, 16), (True, 64),
        ]

    def test_preserves_relative_group_order(self):
        """First-encountered group appears first in the output."""
        entries = (
            [_sn_entry(c, tp=2) for c in [8, 16, 32]]
            + [_sn_entry(c, tp=1) for c in [4, 16, 64]]
        )
        result = reduce_conc_to_endpoints(entries)
        assert [(e["tp"], e["conc"]) for e in result] == [
            (2, 8), (2, 32), (1, 4), (1, 64),
        ]


class TestReduceConcToEndpointsMultiNode:
    def test_trims_long_list_to_endpoints(self):
        entry = _mn_entry([500, 1000, 2000, 4000])
        result = reduce_conc_to_endpoints([entry])
        assert len(result) == 1
        assert result[0]["conc"] == [500, 4000]

    def test_preserves_descending_order(self):
        entry = _mn_entry([4000, 2000, 1000, 500])
        result = reduce_conc_to_endpoints([entry])
        assert result[0]["conc"] == [4000, 500]

    def test_two_point_list_unchanged(self):
        entry = _mn_entry([500, 4000])
        result = reduce_conc_to_endpoints([entry])
        assert result[0]["conc"] == [500, 4000]

    def test_single_element_list_unchanged(self):
        entry = _mn_entry([1000])
        result = reduce_conc_to_endpoints([entry])
        assert result[0]["conc"] == [1000]

    def test_does_not_mutate_input(self):
        original = [500, 1000, 2000, 4000]
        entry = _mn_entry(list(original))
        reduce_conc_to_endpoints([entry])
        assert entry["conc"] == original


class TestReduceConcToEndpointsMixed:
    def test_single_and_multi_node_entries_preserved_together(self):
        sn = [_sn_entry(c) for c in [4, 8, 16, 64]]
        mn = _mn_entry([500, 1000, 2000, 4000])
        result = reduce_conc_to_endpoints(sn + [mn])
        assert len(result) == 3
        assert [e.get("conc") for e in result] == [4, 64, [500, 4000]]

    def test_empty_input(self):
        assert reduce_conc_to_endpoints([]) == []
