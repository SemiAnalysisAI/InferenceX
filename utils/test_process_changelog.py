"""Tests for process_changelog.trim_conc."""
from process_changelog import trim_conc


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


class TestTrimConcSingleNode:
    def test_ascending_series_keeps_max(self):
        entries = [_sn_entry(c) for c in [4, 8, 16, 32, 64]]
        result = trim_conc(entries)
        assert [e["conc"] for e in result] == [64]

    def test_descending_series_keeps_max(self):
        entries = [_sn_entry(c) for c in [32, 16, 8, 4, 2]]
        result = trim_conc(entries)
        assert [e["conc"] for e in result] == [32]

    def test_unordered_series_keeps_max(self):
        entries = [_sn_entry(c) for c in [16, 4, 32, 8, 64]]
        result = trim_conc(entries)
        assert [e["conc"] for e in result] == [64]

    def test_two_point_series_keeps_max(self):
        entries = [_sn_entry(4), _sn_entry(64)]
        result = trim_conc(entries)
        assert [e["conc"] for e in result] == [64]

    def test_single_entry_unchanged(self):
        entries = [_sn_entry(32)]
        result = trim_conc(entries)
        assert [e["conc"] for e in result] == [32]

    def test_different_tp_grouped_independently(self):
        entries = (
            [_sn_entry(c, tp=1) for c in [4, 8, 16, 64]]
            + [_sn_entry(c, tp=2) for c in [8, 16, 32]]
        )
        result = trim_conc(entries)
        assert [(e["tp"], e["conc"]) for e in result] == [(1, 64), (2, 32)]

    def test_different_seq_len_grouped_independently(self):
        entries = (
            [_sn_entry(c, isl=1024, osl=1024) for c in [4, 16, 64]]
            + [_sn_entry(c, isl=8192, osl=1024) for c in [4, 16, 64]]
        )
        result = trim_conc(entries)
        assert [(e["isl"], e["conc"]) for e in result] == [(1024, 64), (8192, 64)]

    def test_run_eval_entries_not_collapsed_with_benchmark_entries(self):
        """run-eval is part of the group key, so eval and benchmark entries
        that share all other fields must remain distinct."""
        entries = [
            _sn_entry(4, run_eval=False),
            _sn_entry(64, run_eval=False),
            _sn_entry(16, run_eval=True),
            _sn_entry(64, run_eval=True),
        ]
        result = trim_conc(entries)
        assert [(e["run-eval"], e["conc"]) for e in result] == [
            (False, 64), (True, 64),
        ]

    def test_preserves_relative_group_order(self):
        """First-encountered group appears first in the output."""
        entries = (
            [_sn_entry(c, tp=2) for c in [8, 16, 32]]
            + [_sn_entry(c, tp=1) for c in [4, 16, 64]]
        )
        result = trim_conc(entries)
        assert [(e["tp"], e["conc"]) for e in result] == [(2, 32), (1, 64)]


class TestTrimConcMultiNode:
    def test_ascending_list_keeps_max(self):
        entry = _mn_entry([500, 1000, 2000, 4000])
        result = trim_conc([entry])
        assert len(result) == 1
        assert result[0]["conc"] == [4000]

    def test_descending_list_keeps_max(self):
        entry = _mn_entry([4000, 2000, 1000, 500])
        result = trim_conc([entry])
        assert result[0]["conc"] == [4000]

    def test_unordered_list_keeps_max(self):
        entry = _mn_entry([1000, 4000, 500, 2000])
        result = trim_conc([entry])
        assert result[0]["conc"] == [4000]

    def test_two_point_list_keeps_max(self):
        entry = _mn_entry([500, 4000])
        result = trim_conc([entry])
        assert result[0]["conc"] == [4000]

    def test_single_element_list_unchanged(self):
        entry = _mn_entry([1000])
        result = trim_conc([entry])
        assert result[0]["conc"] == [1000]

    def test_does_not_mutate_input(self):
        original = [500, 1000, 2000, 4000]
        entry = _mn_entry(list(original))
        trim_conc([entry])
        assert entry["conc"] == original


class TestTrimConcMixed:
    def test_single_and_multi_node_entries_preserved_together(self):
        sn = [_sn_entry(c) for c in [4, 8, 16, 64]]
        mn = _mn_entry([500, 1000, 2000, 4000])
        result = trim_conc(sn + [mn])
        assert len(result) == 2
        assert [e.get("conc") for e in result] == [64, [4000]]

    def test_empty_input(self):
        assert trim_conc([]) == []
