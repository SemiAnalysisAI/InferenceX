"""Comprehensive tests for generate_sweep_configs.py"""
import argparse
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from infx.matrix import generate as generate_sweep_configs
from infx.matrix.generate import (
    MIN_EVAL_CONC,
    add_multinode_node_count,
    apply_node_type_defaults,
    expand_config_keys,
    filter_exp_names,
    generate_full_sweep,
    generate_test_config_sweep,
    mark_all_eval_entries,
    mark_eval_entries,
    multinode_node_count,
    multinode_worker_pair,
    seq_len_to_str,
    trim_conc,
)


def test_aggregated_multinode_node_count_uses_explicit_num_nodes():
    entry = {
        "runner": "unknown",
        "disagg": False,
        "prefill": {},
        "decode": {},
    }

    add_multinode_node_count(entry, {}, num_nodes=3)

    assert entry["node-count"] == 3


def test_disaggregated_multinode_node_count_rejects_num_nodes():
    entry = {
        "runner": "unknown",
        "disagg": True,
        "prefill": {},
        "decode": {},
    }

    with pytest.raises(ValueError, match="num-nodes.*disaggregated"):
        add_multinode_node_count(entry, {}, num_nodes=3)


def test_disaggregated_multinode_node_count_requires_hardware_inventory():
    entry = {
        "runner": "cluster:unknown",
        "disagg": True,
        "prefill": {"num-worker": 1, "tp": 8},
        "decode": {"num-worker": 1, "tp": 8},
    }

    with pytest.raises(ValueError, match="Cannot resolve gpus-per-node"):
        add_multinode_node_count(entry, {}, num_nodes=None)


def test_aggregated_worker_expands_to_legacy_matrix_pair():
    benchmark = {
        "worker": {
            "num-worker": 2,
            "tp": 8,
            "pp": 2,
            "ep": 1,
            "dp-attn": False,
            "additional-settings": ["CONFIG_FILE=recipes/aggregate.yaml"],
        }
    }

    prefill, decode = multinode_worker_pair(benchmark, disagg=False)

    assert prefill == {
        "num-worker": 2,
        "tp": 8,
        "pp": 2,
        "dcp-size": 1,
        "pcp-size": 1,
        "ep": 1,
        "dp-attn": False,
        "additional-settings": ["CONFIG_FILE=recipes/aggregate.yaml"],
    }
    assert decode == {
        "num-worker": 0,
        "tp": 8,
        "pp": 2,
        "dcp-size": 1,
        "pcp-size": 1,
        "ep": 1,
        "dp-attn": False,
    }


def test_multinode_node_count_uses_role_gpu_footprints(sample_runner_config):
    prefill = {"num-worker": 3, "tp": 2, "pp": 1, "pcp-size": 1}
    decode = {"num-worker": 2, "tp": 8, "pp": 1, "pcp-size": 1}

    assert multinode_node_count(
        prefill, decode, "cluster:b300-nv", sample_runner_config
    ) == 3


def test_multinode_node_count_honors_explicit_role_node_settings():
    prefill = {
        "num-worker": 1,
        "tp": 8,
        "additional-settings": ["PREFILL_NODES=2"],
    }
    decode = {
        "num-worker": 1,
        "tp": 8,
        "additional-settings": ["DECODE_NODES=1"],
    }

    assert multinode_node_count(prefill, decode, "unknown", {}) == 3


def test_multinode_node_count_resolves_heterogeneous_worker_hardware(
    sample_runner_config,
):
    prefill = {"hardware": "gb200", "num-worker": 5, "tp": 4}
    decode = {"hardware": "h100", "num-worker": 1, "tp": 8}

    assert multinode_node_count(
        prefill, decode, "gb200", sample_runner_config
    ) == 6


@pytest.mark.parametrize("config_file", [
    "recipes/test.yaml",
    "benchmarks/multi_node/srt-slurm-recipes/test.yaml",
])
@pytest.mark.parametrize(("resources", "expected_nodes"), [
    ({"agg_nodes": 3}, 3),
    ({"prefill_nodes": 2, "decode_nodes": 3}, 5),
])
def test_multinode_node_count_prefers_recipe_resources(
    tmp_path, monkeypatch, config_file, resources, expected_nodes,
):
    recipe = tmp_path / "benchmarks/multi_node/srt-slurm-recipes/test.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(yaml.safe_dump({"resources": resources}))
    monkeypatch.setattr(
        generate_sweep_configs, "__file__",
        str(tmp_path / "infx/matrix/generate.py"),
    )
    prefill = {
        "num-worker": 1, "tp": 8,
        "additional-settings": [f"CONFIG_FILE={config_file}", "PREFILL_NODES=7"],
    }
    decode = {"num-worker": 1, "tp": 8, "additional-settings": ["DECODE_NODES=9"]}

    # Recipe allocation wins over role overrides, even without an inventory.
    assert multinode_node_count(prefill, decode, "unknown", {}) == expected_nodes


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_single_node_config():
    """Single node config based on dsr1-fp8-mi300x-sglang."""
    return {
        "dsr1-fp8-mi300x-sglang": {
            "image": "rocm/7.0:rocm7.0_ubuntu_22.04_sgl-dev-v0.5.2-rocm7.0-mi30x-20250915",
            "model": "deepseek-ai/DeepSeek-R1-0528",
            "model-prefix": "dsr1",
            "precision": "fp8",
            "framework": "sglang",
            "runner": "mi300x",
            "multinode": False,
            "scenarios": {
                "fixed-seq-len": [

                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [
                            {"tp": 8, "conc-start": 4, "conc-end": 64}
                        ]
                    },
                    {
                        "isl": 8192,
                        "osl": 1024,
                        "search-space": [
                            {"tp": 8, "conc-start": 4, "conc-end": 64}
                        ]
                    }
                ]
            }
        }
    }


@pytest.fixture
def sample_multinode_config():
    """Multinode config based on dsr1-fp4-gb200-dynamo-trt."""
    return {
        "dsr1-fp4-gb200-dynamo-trt": {
            "image": "nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:0.5.1-rc0.pre3",
            "model": "deepseek-r1-fp4",
            "model-prefix": "dsr1",
            "precision": "fp4",
            "framework": "dynamo-trt",
            "runner": "gb200",
            "multinode": True,
            "disagg": True,
            "kv-p2p-transfer": "nixl",
            "scenarios": {
                "fixed-seq-len": [

                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [
                            {
                                "conc-list": [2150],
                                "prefill": {
                                    "hardware": "gb200",
                                    "num-worker": 5,
                                    "tp": 4,
                                    "ep": 4,
                                    "dp-attn": True,
                                    "additional-settings": [
                                        "PREFILL_MAX_NUM_TOKENS=8448",
                                        "PREFILL_MAX_BATCH_SIZE=1",
                                    ],
                                },
                                "decode": {
                                    "hardware": "h100",
                                    "num-worker": 1,
                                    "tp": 8,
                                    "ep": 8,
                                    "dp-attn": True,
                                    "additional-settings": [
                                        "DECODE_MAX_NUM_TOKENS=256",
                                        "DECODE_MAX_BATCH_SIZE=256",
                                    ],
                                },
                            }
                        ]
                    }
                ]
            }
        }
    }


@pytest.fixture
def sample_runner_config():
    """Runner config based on configs/runners.yaml."""
    return {
        "labels": {
            "h100": ["h100-cr_0", "h100-cr_1", "h100-cw_0", "h100-cw_1"],
            "h200": ["h200-cw_0", "h200-cw_1"],
            "b200": ["b200-nvd_0", "b200-nvd_1", "b200-nscale_1"],
            "b300": ["b300-nv_0", "b300-nv_1"],
            "cluster:b300-nv": ["b300-nv_0", "b300-nv_1"],
            "mi300x": ["mi300x-amd_0", "mi300x-amd_1", "mi300x-cr_0"],
            "gb200": ["gb200-nv_0"],
        },
        "hardware": {
            "cluster:h100-dgxc": {"available-cpu-dram-mib": 2063837, "gpus-per-node": 8},
            "cluster:h200-dgxc": {"available-cpu-dram-mib": 1471356, "gpus-per-node": 8},
            "cluster:b200-nscale": {"available-cpu-dram-mib": 3774874, "gpus-per-node": 8},
            "cluster:b300-nv": {"available-cpu-dram-mib": 2964436, "gpus-per-node": 8},
            "cluster:mi300x-amd": {"available-cpu-dram-mib": 1547820, "gpus-per-node": 8},
            "cluster:mi355x-amds": {"available-cpu-dram-mib": 3095781, "gpus-per-node": 8},
            "cluster:gb200-nv": {"available-cpu-dram-mib": 860160, "gpus-per-node": 4},
        },
    }


@pytest.fixture
def full_sweep_args_single_node():
    """Args for full-sweep single-node command."""
    args = argparse.Namespace()
    args.model_prefix = None
    args.precision = None
    args.framework = None
    args.runner_type = None
    args.seq_lens = None
    args.step_size = 2
    args.min_conc = None
    args.max_conc = None
    args.max_tp = None
    args.max_ep = None
    args.runner_node_filter = None
    args.single_node = True
    args.multi_node = False
    return args


@pytest.fixture
def full_sweep_args_multi_node():
    """Args for full-sweep multi-node command."""
    args = argparse.Namespace()
    args.model_prefix = None
    args.precision = None
    args.framework = None
    args.runner_type = None
    args.seq_lens = None
    args.step_size = 2
    args.min_conc = None
    args.max_conc = None
    args.max_tp = None
    args.max_ep = None
    args.runner_node_filter = None
    args.single_node = False
    args.multi_node = True
    return args


# =============================================================================
# Test sequence length formatting
# =============================================================================

class TestSeqLenToStr:
    """Tests for seq_len_to_str function."""

    def test_known_sequence_lengths(self):
        """Known sequence lengths should return short name."""
        assert seq_len_to_str(1024, 1024) == "1k1k"
        assert seq_len_to_str(8192, 1024) == "8k1k"

    def test_unknown_sequence_lengths(self):
        """Unknown sequence lengths should return isl_osl format."""
        assert seq_len_to_str(2048, 2048) == "2048_2048"
        assert seq_len_to_str(4096, 1024) == "4096_1024"


# =============================================================================
# Test mark_eval_entries
# =============================================================================

class TestMarkEvalEntries:
    """Tests for eval matrix selection policy."""

    def test_marks_agentic_entry_for_gsm8k(self):
        matrix_values = [
            {
                "scenario-type": "agentic-coding",
                "model": "m", "runner": "b300", "framework": "vllm",
                "precision": "fp4", "tp": 8, "conc": 32,
            },
            {
                "scenario-type": "agentic-coding",
                "model": "m", "runner": "b300", "framework": "vllm",
                "precision": "fp4", "tp": 8, "conc": 64,
            },
        ]

        result = mark_eval_entries(matrix_values, include_agentic=True)

        marked = [e for e in result if e.get("run-eval")]
        assert len(marked) == 1
        assert marked[0]["conc"] == 64

    def test_marks_multinode_agentic_entry_at_highest_eligible_conc(self):
        """Multi-node agentic (SWE-bench) eval selection mirrors the
        fixed-seq-len multi-node policy: one eval row per parallelism
        topology, at its highest eligible (>= MIN_EVAL_CONC) concurrency.

        Each concurrency is its own matrix entry (chunk size 1) whose
        exp-name embeds that concurrency, unlike fixed-seq-len multi-node
        rows where exp-name never varies with conc — the grouping key must
        still treat these as the same topology.
        """
        common = {
            "scenario-type": "agentic-coding",
            "model": "m", "runner": "b300", "framework": "sglang-disagg",
            "precision": "fp4", "spec-decoding": "none", "disagg": True,
            "prefill": {"num-worker": 1, "tp": 8, "ep": 1, "dp-attn": False},
            "decode": {"num-worker": 1, "tp": 8, "ep": 1, "dp-attn": False},
        }
        matrix_values = [
            {**common, "conc": [8], "exp-name": "p1x8_d1x8_conc8"},
            {**common, "conc": [16], "exp-name": "p1x8_d1x8_conc16"},
            {**common, "conc": [32], "exp-name": "p1x8_d1x8_conc32"},
        ]

        result = mark_eval_entries(matrix_values, include_agentic=True)

        marked = [e for e in result if e.get("run-eval")]
        assert len(marked) == 1
        assert marked[0]["conc"] == [32]
        assert marked[0]["eval-conc"] == 32

    def test_multinode_agentic_groups_are_independent_per_topology(self):
        """Two distinct multi-node agentic topologies (e.g. differing by
        prefill EP/DP) must each get their own eval row."""
        base = {
            "scenario-type": "agentic-coding",
            "model": "m", "runner": "b300", "framework": "sglang-disagg",
            "precision": "fp4", "spec-decoding": "none", "disagg": True,
        }
        topology_a = {
            "prefill": {"num-worker": 1, "tp": 8, "ep": 1, "dp-attn": False},
            "decode": {"num-worker": 1, "tp": 8, "ep": 1, "dp-attn": False},
        }
        topology_b = {
            "prefill": {"num-worker": 1, "tp": 8, "ep": 8, "dp-attn": True},
            "decode": {"num-worker": 1, "tp": 8, "ep": 8, "dp-attn": True},
        }
        matrix_values = [
            {**base, **topology_a, "conc": [16], "exp-name": "a_conc16"},
            {**base, **topology_a, "conc": [32], "exp-name": "a_conc32"},
            {**base, **topology_b, "conc": [64], "exp-name": "b_conc64"},
            {**base, **topology_b, "conc": [96], "exp-name": "b_conc96"},
        ]

        result = mark_eval_entries(matrix_values, include_agentic=True)

        marked = {e["exp-name"]: e for e in result if e.get("run-eval")}
        assert set(marked) == {"a_conc32", "b_conc96"}
        assert marked["a_conc32"]["eval-conc"] == 32
        assert marked["b_conc96"]["eval-conc"] == 96

    def test_default_mode_does_not_mark_agentic(self):
        matrix_values = [
            {
                "scenario-type": "agentic-coding",
                "model": "m", "runner": "b300", "framework": "vllm",
                "precision": "fp4", "tp": 8, "conc": 32,
            },
            {
                "scenario-type": "agentic-coding",
                "model": "m", "runner": "b300", "framework": "vllm",
                "precision": "fp4", "tp": 8, "conc": 64,
            },
        ]

        result = mark_eval_entries(matrix_values)

        marked = [e for e in result if e.get("run-eval")]
        assert len(marked) == 0, (
            f"Expected 0 agentic entries marked run-eval in default mode, got {len(marked)}"
        )

    def test_default_marks_every_supported_vendor_point(self):
        matrix_values = [
            {
                "scenario-type": "agentic-coding",
                "model-prefix": model_prefix,
                "model": model_prefix,
                "runner": "b300",
                "framework": "vllm",
                "precision": "fp4",
                "tp": 8,
                "conc": conc,
            }
            for model_prefix in ("kimik3", "minimaxm3")
            for conc in (1, 64)
        ]
        matrix_values.append({
            "scenario-type": "agentic-coding",
            "model-prefix": "minimaxm3-bfcl",
            "model": "unsupported",
            "runner": "b300",
            "framework": "vllm",
            "precision": "fp4",
            "tp": 8,
            "conc": 64,
        })

        result = mark_eval_entries(matrix_values)

        expected = {
            "kimik3": ("kimi-vendor", "kimi_tool_call_schema"),
            "minimaxm3": ("minimax-vendor", "minimax_m3_smoke"),
        }
        for model_prefix, eval_spec in expected.items():
            rows = [row for row in result if row["model-prefix"] == model_prefix]
            assert {row["conc"] for row in rows} == {1, 64}
            assert all(row["run-eval"] is True for row in rows)
            assert {
                (row["eval-framework"], row["eval-suite"]) for row in rows
            } == {eval_spec}

        unsupported = result[-1]
        assert unsupported["run-eval"] is False
        assert "eval-framework" not in unsupported
        assert all(row.get("eval-framework") != "bfcl" for row in result)

    def test_default_marks_every_multinode_vendor_point(self):
        common = {
            "scenario-type": "agentic-coding",
            "model-prefix": "kimik3",
            "model": "kimi",
            "runner": "gb200",
            "framework": "sglang-disagg",
            "precision": "fp4",
            "spec-decoding": "none",
            "disagg": True,
            "prefill": {"num-worker": 1, "tp": 8},
            "decode": {"num-worker": 1, "tp": 8},
        }
        matrix_values = [
            {**common, "conc": [2], "exp-name": "kimi-conc2"},
            {**common, "conc": [32], "exp-name": "kimi-conc32"},
        ]

        result = mark_eval_entries(matrix_values)

        assert len(result) == 2
        assert all(row["run-eval"] is True for row in result)
        assert [row["eval-conc"] for row in result] == [2, 32]
        assert all(row["eval-framework"] == "kimi-vendor" for row in result)
        assert all(
            row["eval-suite"] == "kimi_tool_call_schema" for row in result
        )

    def test_fixed_sequence_eval_uses_lm_eval_metadata(self):
        matrix_values = [{
            "model": "m",
            "runner": "b200",
            "framework": "vllm",
            "precision": "fp8",
            "isl": 8192,
            "osl": 1024,
            "spec-decoding": "none",
            "dp-attn": False,
            "tp": 8,
            "conc": MIN_EVAL_CONC,
        }]

        result = mark_eval_entries(matrix_values)

        assert result[0]["run-eval"] is True
        assert result[0]["eval-framework"] == "lm-eval"
        assert result[0]["eval-suite"] == ""


    def test_single_node_skips_eval_entries_below_min_conc(self):
        """Single-node eval selection should ignore conc values below MIN_EVAL_CONC."""
        matrix_values = [
            {
                "model": "deepseek-ai/DeepSeek-R1-0528",
                "runner": "b200",
                "framework": "sglang",
                "precision": "fp8",
                "isl": 8192,
                "osl": 1024,
                "spec-decoding": "none",
                "dp-attn": False,
                "tp": 8,
                "conc": 8,
            },
            {
                "model": "deepseek-ai/DeepSeek-R1-0528",
                "runner": "b200",
                "framework": "sglang",
                "precision": "fp8",
                "isl": 8192,
                "osl": 1024,
                "spec-decoding": "none",
                "dp-attn": False,
                "tp": 8,
                "conc": MIN_EVAL_CONC,
            },
            {
                "model": "deepseek-ai/DeepSeek-R1-0528",
                "runner": "b200",
                "framework": "sglang",
                "precision": "fp8",
                "isl": 8192,
                "osl": 1024,
                "spec-decoding": "none",
                "dp-attn": False,
                "tp": 8,
                "conc": 32,
            },
            {
                "model": "deepseek-ai/DeepSeek-R1-0528",
                "runner": "b200",
                "framework": "sglang",
                "precision": "fp8",
                "isl": 8192,
                "osl": 1024,
                "spec-decoding": "none",
                "dp-attn": False,
                "tp": 8,
                "conc": 64,
            },
        ]

        result = mark_eval_entries(matrix_values)

        assert result[0]["run-eval"] is False
        assert result[1]["run-eval"] is False
        assert result[2]["run-eval"] is True
        assert result[3]["run-eval"] is True

    def test_multi_node_skips_groups_with_only_conc_below_min_conc(self):
        """Multinode eval selection should skip groups whose conc lists are all below MIN_EVAL_CONC."""
        matrix_values = [
            {
                "model": "deepseek-ai/DeepSeek-R1-0528",
                "runner": "cluster:b200-nscale",
                "framework": "dynamo-trt",
                "precision": "fp8",
                "isl": 8192,
                "osl": 1024,
                "spec-decoding": "none",
                "prefill": {
                    "num-worker": 1,
                    "tp": 8,
                    "ep": 1,
                    "dp-attn": False,
                },
                "decode": {
                    "num-worker": 1,
                    "tp": 8,
                    "ep": 1,
                    "dp-attn": False,
                },
                "conc": [1],
            }
        ]

        result = mark_eval_entries(matrix_values)

        assert result[0]["run-eval"] is False
        assert "eval-conc" not in result[0]

    def test_multi_node_marks_each_parallelism_at_highest_eligible_conc(self):
        """Each multinode parallelism should eval at its highest eligible concurrency."""
        matrix_values = [
            {
                "model": "deepseek-ai/DeepSeek-R1-0528",
                "runner": "cluster:b200-nscale",
                "framework": "dynamo-trt",
                "precision": "fp8",
                "isl": 8192,
                "osl": 1024,
                "spec-decoding": "none",
                "prefill": {
                    "num-worker": 1,
                    "tp": 8,
                    "ep": 1,
                    "dp-attn": True,
                },
                "decode": {
                    "num-worker": 4,
                    "tp": 8,
                    "ep": 1,
                    "dp-attn": False,
                },
                "conc": [8, 16, 32],
            },
            {
                "model": "deepseek-ai/DeepSeek-R1-0528",
                "runner": "cluster:b200-nscale",
                "framework": "dynamo-trt",
                "precision": "fp8",
                "isl": 8192,
                "osl": 1024,
                "spec-decoding": "none",
                "prefill": {
                    "num-worker": 2,
                    "tp": 4,
                    "ep": 1,
                    "dp-attn": True,
                },
                "decode": {
                    "num-worker": 2,
                    "tp": 4,
                    "ep": 1,
                    "dp-attn": False,
                },
                "conc": [8, 16, 64],
            },
        ]

        result = mark_eval_entries(matrix_values)

        assert result[0]["run-eval"] is True
        assert result[0]["eval-conc"] == 32
        assert result[1]["run-eval"] is True
        assert result[1]["eval-conc"] == 64

    def test_multi_node_worker_counts_define_parallelism(self):
        """Prefill and decode worker counts should each define a distinct eval target."""
        def entry(prefill_workers, decode_workers, conc):
            return {
                "model": "deepseek-ai/DeepSeek-R1-0528",
                "runner": "cluster:mi355x-amds",
                "framework": "vllm-disagg",
                "precision": "fp8",
                "isl": 8192,
                "osl": 1024,
                "spec-decoding": "none",
                "prefill": {
                    "num-worker": prefill_workers,
                    "tp": 4,
                    "ep": 1,
                    "dp-attn": False,
                },
                "decode": {
                    "num-worker": decode_workers,
                    "tp": 8,
                    "ep": 1,
                    "dp-attn": False,
                },
                "conc": [16, conc],
            }

        result = mark_eval_entries([
            entry(prefill_workers=1, decode_workers=1, conc=32),
            entry(prefill_workers=2, decode_workers=1, conc=64),
            entry(prefill_workers=1, decode_workers=2, conc=128),
        ])

        assert [(e["run-eval"], e["eval-conc"]) for e in result] == [
            (True, 32),
            (True, 64),
            (True, 128),
        ]

    def test_multi_node_split_parallelism_uses_only_highest_concurrency_entry(self):
        """Split concurrency rows for one parallelism should produce one eval job."""
        base_entry = {
            "model": "deepseek-ai/DeepSeek-R1-0528",
            "runner": "cluster:mi355x-amds",
            "framework": "sglang-disagg",
            "precision": "fp4",
            "isl": 8192,
            "osl": 1024,
            "spec-decoding": "none",
            "prefill": {
                "num-worker": 1,
                "tp": 8,
                "ep": 1,
                "dp-attn": False,
                "additional-settings": ["PREFILL_NODES=1"],
            },
            "decode": {
                "num-worker": 2,
                "tp": 8,
                "ep": 1,
                "dp-attn": False,
                "additional-settings": ["DECODE_NODES=2"],
            },
            "run-eval": False,
        }
        matrix_values = [
            {**base_entry, "conc": [2, 4, 8, 16, 32]},
            {**base_entry, "conc": [64, 128, 256]},
        ]

        result = mark_eval_entries(matrix_values)

        assert result[0]["run-eval"] is False
        assert "eval-conc" not in result[0]
        assert result[1]["run-eval"] is True
        assert result[1]["eval-conc"] == 256

    def test_marks_highest_and_median_conc(self):
        """Should mark highest and median concurrency for 8k1k entries."""
        entries = [
            {'model': 'm', 'runner': 'r', 'framework': 'f', 'precision': 'fp8',
             'isl': 8192, 'osl': 1024, 'tp': 2, 'conc': 32,
             'spec-decoding': False, 'dp-attn': False, 'run-eval': False},
            {'model': 'm', 'runner': 'r', 'framework': 'f', 'precision': 'fp8',
             'isl': 8192, 'osl': 1024, 'tp': 2, 'conc': 128,
             'spec-decoding': False, 'dp-attn': False, 'run-eval': False},
            {'model': 'm', 'runner': 'r', 'framework': 'f', 'precision': 'fp8',
             'isl': 8192, 'osl': 1024, 'tp': 2, 'conc': 512,
             'spec-decoding': False, 'dp-attn': False, 'run-eval': False},
        ]
        result = mark_eval_entries(entries)
        # conc values: [32, 128, 512]. median=128 (index 1), highest=512
        assert result[0]['run-eval'] is False   # conc=32
        assert result[1]['run-eval'] is True    # conc=128 (median)
        assert result[2]['run-eval'] is True    # conc=512 (highest)

    def test_non_8k1k_never_marked(self):
        """Entries with non-8k1k seq lengths should never be eval-marked."""
        entries = [
            {'model': 'm', 'runner': 'r', 'framework': 'f', 'precision': 'fp8',
             'isl': 1024, 'osl': 1024, 'tp': 2, 'conc': 512,
             'spec-decoding': False, 'dp-attn': False, 'run-eval': False},
        ]
        result = mark_eval_entries(entries)
        assert result[0]['run-eval'] is False


class TestMarkAllEvalEntries:
    """Tests for the all-evals selection policy."""

    def test_marks_only_8k1k_entries_and_passes_other_seq_lens_through(self):
        entries = [
            {  # 1k1k is not eligible for evals -> left unmarked
                'model': 'm', 'runner': 'r', 'framework': 'f', 'precision': 'fp8',
                'isl': 1024, 'osl': 1024, 'tp': 2, 'conc': 1,
                'spec-decoding': 'none', 'dp-attn': False, 'run-eval': False,
            },
            {  # 8k1k is eligible -> marked for eval
                'model': 'm', 'runner': 'r', 'framework': 'f', 'precision': 'fp8',
                'isl': 8192, 'osl': 1024, 'tp': 2, 'conc': 8,
                'spec-decoding': 'none', 'dp-attn': False, 'run-eval': False,
            },
        ]

        result = mark_all_eval_entries(entries)

        by_isl = {entry['isl']: entry for entry in result}
        assert by_isl[1024]['run-eval'] is False
        assert by_isl[8192]['run-eval'] is True

    def test_batches_every_multinode_concurrency_per_engine_topology(self):
        entries = [
            {
                'model': 'm', 'runner': 'r', 'framework': 'f', 'precision': 'fp8',
                'isl': 8192, 'osl': 1024, 'spec-decoding': 'none',
                'prefill': {'dp-attn': False},
                'decode': {'dp-attn': False},
                'conc': [1, 4, 8, 16],
                'run-eval': False,
            },
            {
                'model': 'm', 'runner': 'r', 'framework': 'f', 'precision': 'fp8',
                'isl': 8192, 'osl': 1024, 'spec-decoding': 'none',
                'prefill': {'dp-attn': True},
                'decode': {'dp-attn': False},
                'conc': [32],
                'run-eval': False,
            },
        ]

        result = mark_all_eval_entries(entries)

        assert len(result) == 2
        assert all(entry['run-eval'] for entry in result)
        assert [entry['conc'] for entry in result] == [
            [1, 4, 8, 16], [32],
        ]
        assert all(entry['eval-all-concs'] is True for entry in result)
        assert all('eval-conc' not in entry for entry in result)

    def test_default_eval_selection_does_not_collapse_all_evals_expansion(self):
        entries = [
            {
                'model': 'm', 'runner': 'r', 'framework': 'f', 'precision': 'fp8',
                'isl': 8192, 'osl': 1024, 'spec-decoding': 'none',
                'prefill': {'dp-attn': False},
                'decode': {'dp-attn': False},
                'conc': [1, 4, 8, 16, 32],
                'run-eval': False,
            },
        ]

        result = mark_all_eval_entries(mark_eval_entries(entries))

        assert len(result) == 1
        assert result[0]['conc'] == [1, 4, 8, 16, 32]
        assert result[0]['eval-all-concs'] is True
        assert 'eval-conc' not in result[0]
        assert result[0]['run-eval'] is True

    def test_deduplicates_overlapping_concurrency_rows_for_same_parallelism(self):
        entries = [
            {
                'model': 'm', 'runner': 'r', 'framework': 'f', 'precision': 'fp8',
                'isl': 8192, 'osl': 1024, 'spec-decoding': 'none',
                'prefill': {'dp-attn': False},
                'decode': {'dp-attn': False},
                'conc': [4, 8, 16],
                'run-eval': False,
                'eval-conc': None,
            },
            {
                'model': 'm', 'runner': 'r', 'framework': 'f', 'precision': 'fp8',
                'isl': 8192, 'osl': 1024, 'spec-decoding': 'none',
                'prefill': {'dp-attn': False},
                'decode': {'dp-attn': False},
                'conc': [16, 32],
                'run-eval': True,
                'eval-conc': 32,
            },
        ]

        result = mark_all_eval_entries(entries)

        assert len(result) == 1
        assert result[0]['conc'] == [4, 8, 16, 32]
        assert result[0]['eval-all-concs'] is True
        assert 'eval-conc' not in result[0]

    def test_excludes_1k1k_multinode_entries_from_expansion(self):
        entries = [
            {  # 1k1k multinode: left untouched, never batched or eval-marked
                'model': 'm', 'runner': 'r', 'framework': 'f', 'precision': 'fp8',
                'isl': 1024, 'osl': 1024, 'spec-decoding': 'none',
                'prefill': {'dp-attn': False},
                'decode': {'dp-attn': False},
                'conc': [4, 8, 16],
                'run-eval': False,
            },
            {  # 8k1k multinode: expanded into a batched eval row
                'model': 'm', 'runner': 'r', 'framework': 'f', 'precision': 'fp8',
                'isl': 8192, 'osl': 1024, 'spec-decoding': 'none',
                'prefill': {'dp-attn': False},
                'decode': {'dp-attn': False},
                'conc': [8, 32],
                'run-eval': False,
            },
        ]

        result = mark_all_eval_entries(entries)

        assert len(result) == 2
        one_k = next(e for e in result if e['isl'] == 1024)
        eight_k = next(e for e in result if e['isl'] == 8192)
        # 1k1k untouched: not eval-marked, not batched, concurrency unchanged
        assert one_k['run-eval'] is False
        assert 'eval-all-concs' not in one_k
        assert one_k['conc'] == [4, 8, 16]
        # 8k1k expanded into a batched eval row
        assert eight_k['run-eval'] is True
        assert eight_k['eval-all-concs'] is True
        assert eight_k['conc'] == [8, 32]

    def test_marks_agentic_entries_for_gsm8k(self):
        entries = [
            {
                'scenario-type': 'agentic-coding',
                'model': 'm',
                'runner': 'r',
                'conc': 64,
            }
        ]

        result = mark_all_eval_entries(entries)

        assert result[0]['run-eval'] is True
        assert 'eval-conc' not in result[0]

    def test_marks_multinode_agentic_entries_for_gsm8k(self):
        """Unlike fixed-seq-len multi-node evals, generic agentic rows with the
        same topology merge but select only their highest concurrency through
        eval-conc.
        """
        common = {
            'scenario-type': 'agentic-coding',
            'model': 'm', 'runner': 'r', 'framework': 'sglang-disagg',
            'precision': 'fp4', 'spec-decoding': 'none', 'disagg': True,
            'prefill': {'num-worker': 1, 'tp': 8, 'ep': 1, 'dp-attn': False},
            'decode': {'num-worker': 1, 'tp': 8, 'ep': 1, 'dp-attn': False},
        }
        entries = [
            {**common, 'conc': [2], 'exp-name': 'p1x8_d1x8_conc2'},
            {**common, 'conc': [16], 'exp-name': 'p1x8_d1x8_conc16'},
            {**common, 'conc': [32], 'exp-name': 'p1x8_d1x8_conc32'},
        ]

        result = mark_all_eval_entries(entries)

        assert len(result) == 1
        assert result[0]['run-eval'] is True
        assert result[0]['conc'] == [2, 16, 32]
        assert result[0]['eval-conc'] == 32
        assert 'eval-all-concs' not in result[0]

    def test_keeps_every_multinode_vendor_point_separate(self):
        common = {
            "scenario-type": "agentic-coding",
            "model-prefix": "minimaxm3",
            "model": "minimax",
            "runner": "gb200",
            "framework": "sglang-disagg",
            "precision": "fp4",
            "spec-decoding": "none",
            "disagg": True,
            "prefill": {"num-worker": 1, "tp": 8},
            "decode": {"num-worker": 1, "tp": 8},
        }
        entries = [
            {**common, "conc": [2], "exp-name": "minimax-conc2"},
            {**common, "conc": [32], "exp-name": "minimax-conc32"},
        ]

        result = mark_all_eval_entries(mark_eval_entries(entries))

        assert len(result) == 2
        assert [row["conc"] for row in result] == [[2], [32]]
        assert [row["eval-conc"] for row in result] == [2, 32]
        assert all(row["eval-framework"] == "minimax-vendor" for row in result)
        assert all(row["eval-suite"] == "minimax_m3_smoke" for row in result)


# =============================================================================
# Test generate_full_sweep for single-node
# =============================================================================

class TestGenerateFullSweepSingleNode:
    """Tests for generate_full_sweep with single-node configs."""

    def test_sweep_expands_each_sequence_length_across_concurrencies(self, sample_single_node_config, sample_runner_config, full_sweep_args_single_node):
        """Each input sequence pair gets the complete requested concurrency range."""
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_single_node_config,
            sample_runner_config
        )
        assert [(row["isl"], row["osl"], row["conc"]) for row in result] == [
            (isl, osl, conc)
            for isl, osl in [(1024, 1024), (8192, 1024)]
            for conc in [4, 8, 16, 32, 64]
        ]

    def test_matrix_entry_structure(self, sample_single_node_config, sample_runner_config, full_sweep_args_single_node):
        """Generated entries should have correct structure."""
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_single_node_config,
            sample_runner_config
        )
        entry = result[0]
        assert entry["image"] == "rocm/7.0:rocm7.0_ubuntu_22.04_sgl-dev-v0.5.2-rocm7.0-mi30x-20250915"
        assert entry["model"] == "deepseek-ai/DeepSeek-R1-0528"
        assert entry["precision"] == "fp8"
        assert entry["framework"] == "sglang"
        assert entry["runner"] == "mi300x"
        assert entry["tp"] == 8
        assert "exp-name" in entry
        assert "max-model-len" in entry
        assert (entry["pp"], entry["dcp-size"], entry["pcp-size"]) == (1, 1, 1)

        explicit_config = copy.deepcopy(sample_single_node_config)
        for seq_config in explicit_config["dsr1-fp8-mi300x-sglang"]["scenarios"]["fixed-seq-len"]:
            for search_entry in seq_config["search-space"]:
                search_entry.update({"pp": 2, "dcp-size": 2, "pcp-size": 2})
        explicit_result = generate_full_sweep(
            full_sweep_args_single_node,
            explicit_config,
            sample_runner_config,
        )
        assert {
            (row["pp"], row["dcp-size"], row["pcp-size"])
            for row in explicit_result
        } == {(2, 2, 2)}

    def test_filter_by_model_prefix(self, sample_single_node_config, sample_runner_config, full_sweep_args_single_node):
        """Filter by model prefix should work."""
        full_sweep_args_single_node.model_prefix = ["dsr1"]
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_single_node_config,
            sample_runner_config
        )
        assert len(result) > 0

        # Non-matching prefix should return empty
        full_sweep_args_single_node.model_prefix = ["nonexistent"]
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_single_node_config,
            sample_runner_config
        )
        assert len(result) == 0

    def test_filter_by_precision(self, sample_single_node_config, sample_runner_config, full_sweep_args_single_node):
        """Filter by precision should work."""
        full_sweep_args_single_node.precision = ["fp8"]
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_single_node_config,
            sample_runner_config
        )
        assert len(result) > 0

        full_sweep_args_single_node.precision = ["fp4"]
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_single_node_config,
            sample_runner_config
        )
        assert len(result) == 0

    def test_filter_by_framework(self, sample_single_node_config, sample_runner_config, full_sweep_args_single_node):
        """Filter by framework should work."""
        full_sweep_args_single_node.framework = ["sglang"]
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_single_node_config,
            sample_runner_config
        )
        assert len(result) > 0

        full_sweep_args_single_node.framework = ["vllm"]
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_single_node_config,
            sample_runner_config
        )
        assert len(result) == 0

    def test_filter_by_runner_type(self, sample_single_node_config, sample_runner_config, full_sweep_args_single_node):
        """Filter by runner type should work."""
        full_sweep_args_single_node.runner_type = ["mi300x"]
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_single_node_config,
            sample_runner_config
        )
        assert len(result) > 0

        full_sweep_args_single_node.runner_type = ["h100"]
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_single_node_config,
            sample_runner_config
        )
        assert len(result) == 0

    def test_invalid_runner_type_raises_error(self, sample_single_node_config, sample_runner_config, full_sweep_args_single_node):
        """Invalid runner type should raise ValueError."""
        full_sweep_args_single_node.runner_type = ["invalid_runner"]
        with pytest.raises(ValueError) as exc_info:
            generate_full_sweep(
                full_sweep_args_single_node,
                sample_single_node_config,
                sample_runner_config
            )
        assert "Invalid runner type" in str(exc_info.value)

    def test_filter_by_seq_lens(self, sample_single_node_config, sample_runner_config, full_sweep_args_single_node):
        """Filter by sequence lengths should work."""
        full_sweep_args_single_node.seq_lens = ["1k1k"]
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_single_node_config,
            sample_runner_config
        )
        # Only 1k1k entries, 5 concurrency values
        assert len(result) == 5
        assert all(entry["isl"] == 1024 and entry["osl"] == 1024 for entry in result)

    def test_max_conc_filter(self, sample_single_node_config, sample_runner_config, full_sweep_args_single_node):
        """max_conc filter should limit concurrency values."""
        full_sweep_args_single_node.max_conc = 16
        full_sweep_args_single_node.seq_lens = ["1k1k"]
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_single_node_config,
            sample_runner_config
        )
        # conc values: 4, 8, 16 (32, 64 filtered out)
        assert len(result) == 3
        assert all(entry["conc"] <= 16 for entry in result)

    def test_max_conc_creates_config_when_below_min(self, sample_single_node_config, sample_runner_config, full_sweep_args_single_node):
        """max_conc below config's min should create config with max_conc value."""
        # Config has conc-start=4, so max_conc=1 should create entry with conc=1
        full_sweep_args_single_node.max_conc = 1
        full_sweep_args_single_node.seq_lens = ["1k1k"]
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_single_node_config,
            sample_runner_config
        )
        # Should create 1 entry with conc=1
        assert len(result) == 1
        assert result[0]["conc"] == 1

    def test_max_conc_zero_or_negative_skips(self, sample_single_node_config, sample_runner_config, full_sweep_args_single_node):
        """max_conc of 0 or negative should skip configs."""
        for invalid_value in [0, -1, -100]:
            full_sweep_args_single_node.max_conc = invalid_value
            result = generate_full_sweep(
                full_sweep_args_single_node,
                sample_single_node_config,
                sample_runner_config
            )
            assert len(result) == 0, f"Expected 0 results for max_conc={invalid_value}"

    def test_max_tp_filter(self, sample_runner_config, full_sweep_args_single_node):
        """max_tp filter should SKIP configs whose tp exceeds max_tp (no clamping)."""
        config = {
            "test-max-tp": {
                "image": "test-image",
                "model": "test-model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "sglang",
                "runner": "mi300x",
                "multinode": False,
                "scenarios": {
                    "fixed-seq-len": [

                        {
                            "isl": 1024,
                            "osl": 1024,
                            "search-space": [
                                {"tp": 4, "conc-start": 4, "conc-end": 64},  # should remain
                                {"tp": 8, "conc-start": 4, "conc-end": 64},  # should be skipped
                            ],
                        }
                    ]
                },
            }
        }

        full_sweep_args_single_node.max_tp = 4
        full_sweep_args_single_node.seq_lens = ["1k1k"]

        result = generate_full_sweep(
            full_sweep_args_single_node,
            config,
            sample_runner_config,
        )

        # conc values: 4, 8, 16, 32, 64 = 5 entries from the tp=4 bmk only
        assert len(result) == 5
        assert all(entry["tp"] == 4 for entry in result)

    def test_max_tp_below_all_available_skips(self, sample_single_node_config, sample_runner_config, full_sweep_args_single_node):
        """If all available tp values are > max_tp, generator should return empty (skip)."""
        full_sweep_args_single_node.max_tp = 2
        full_sweep_args_single_node.seq_lens = ["1k1k"]

        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_single_node_config,
            sample_runner_config,
        )

        assert len(result) == 0

    def test_max_tp_zero_or_negative_skips(self, sample_single_node_config, sample_runner_config, full_sweep_args_single_node):
        """max_tp of 0 or negative should skip configs."""
        for invalid_value in [0, -1, -100]:
            full_sweep_args_single_node.max_tp = invalid_value
            result = generate_full_sweep(
                full_sweep_args_single_node,
                sample_single_node_config,
                sample_runner_config
            )
            assert len(result) == 0, f"Expected 0 results for max_tp={invalid_value}"

    def test_step_size(self, sample_single_node_config, sample_runner_config, full_sweep_args_single_node):
        """Different step sizes should affect concurrency progression."""
        full_sweep_args_single_node.step_size = 4
        full_sweep_args_single_node.seq_lens = ["1k1k"]
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_single_node_config,
            sample_runner_config
        )
        # conc: 4, 16, 64 = 3 values
        assert len(result) == 3
        conc_values = [entry["conc"] for entry in result]
        assert 4 in conc_values
        assert 16 in conc_values
        assert 64 in conc_values

    def test_exp_name_format(self, sample_single_node_config, sample_runner_config, full_sweep_args_single_node):
        """exp-name should have correct format."""
        full_sweep_args_single_node.seq_lens = ["1k1k"]
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_single_node_config,
            sample_runner_config
        )
        assert all(entry["exp-name"] == "dsr1_1k1k" for entry in result)

    def test_max_model_len_calculation(self, sample_single_node_config, sample_runner_config, full_sweep_args_single_node):
        """max-model-len should be isl + osl + 256."""
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_single_node_config,
            sample_runner_config
        )
        assert {
            (entry["isl"], entry["osl"], entry["max-model-len"])
            for entry in result
        } == {(1024, 1024, 2304), (8192, 1024, 9472)}

    def test_runner_node_filter(self, sample_single_node_config, sample_runner_config, full_sweep_args_single_node):
        """Runner node filter should expand entries to individual matching nodes."""
        full_sweep_args_single_node.runner_type = ["mi300x"]
        full_sweep_args_single_node.runner_node_filter = "amd"
        full_sweep_args_single_node.seq_lens = ["1k1k"]
        full_sweep_args_single_node.max_conc = 4  # Limit to single conc value for easier counting
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_single_node_config,
            sample_runner_config
        )
        # 2 amd nodes (mi300x-amd_0, mi300x-amd_1), 1 conc value = 2 entries
        assert len(result) == 2
        runners = [entry["runner"] for entry in result]
        assert "mi300x-amd_0" in runners
        assert "mi300x-amd_1" in runners

    def test_runner_node_filter_no_match(self, sample_single_node_config, sample_runner_config, full_sweep_args_single_node):
        """Runner node filter with no matches should skip configs (return empty)."""
        full_sweep_args_single_node.runner_type = ["mi300x"]
        full_sweep_args_single_node.runner_node_filter = "nonexistent"
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_single_node_config,
            sample_runner_config
        )
        # No nodes match, so config is skipped
        assert len(result) == 0

    def test_runner_node_filter_without_runner_type(self, sample_single_node_config, sample_runner_config, full_sweep_args_single_node):
        """Runner node filter should work without explicit runner type (uses config's runner)."""
        full_sweep_args_single_node.runner_node_filter = "amd"
        full_sweep_args_single_node.seq_lens = ["1k1k"]
        full_sweep_args_single_node.max_conc = 4
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_single_node_config,
            sample_runner_config
        )
        # Config has runner=mi300x, filter "amd" matches mi300x-amd_0 and mi300x-amd_1
        assert len(result) == 2
        assert all("amd" in entry["runner"] for entry in result)


# =============================================================================
# Test generate_full_sweep for multi-node
# =============================================================================

class TestGenerateFullSweepMultiNode:
    """Tests for generate_full_sweep with multi-node configs."""

    def test_multinode_entry_structure(self, sample_multinode_config, sample_runner_config, full_sweep_args_multi_node):
        """Multinode entries should have prefill and decode configs."""
        result = generate_full_sweep(
            full_sweep_args_multi_node,
            sample_multinode_config,
            sample_runner_config
        )
        entry = result[0]
        assert entry["prefill"]["num-worker"] == 5
        assert entry["decode"]["num-worker"] == 1
        assert entry["disagg"] is True
        assert entry["prefill"]["hardware"] == "gb200"
        assert entry["decode"]["hardware"] == "h100"
        assert (
            entry["prefill"]["pp"],
            entry["prefill"]["dcp-size"],
            entry["prefill"]["pcp-size"],
        ) == (1, 1, 1)
        assert (
            entry["decode"]["pp"],
            entry["decode"]["dcp-size"],
            entry["decode"]["pcp-size"],
        ) == (1, 1, 1)

    def test_multinode_parallelism_fields(self, sample_multinode_config, sample_runner_config, full_sweep_args_multi_node):
        explicit_config = copy.deepcopy(sample_multinode_config)
        search_entry = explicit_config["dsr1-fp4-gb200-dynamo-trt"]["scenarios"]["fixed-seq-len"][0]["search-space"][0]
        search_entry["prefill"].update({"pp": 2, "dcp-size": 2, "pcp-size": 2})
        search_entry["decode"].update({"pp": 2, "dcp-size": 4, "pcp-size": 1})

        entry = generate_full_sweep(
            full_sweep_args_multi_node,
            explicit_config,
            sample_runner_config,
        )[0]

        assert (
            entry["prefill"]["pp"],
            entry["prefill"]["dcp-size"],
            entry["prefill"]["pcp-size"],
        ) == (2, 2, 2)
        assert (
            entry["decode"]["pp"],
            entry["decode"]["dcp-size"],
            entry["decode"]["pcp-size"],
        ) == (2, 4, 1)

    def test_multinode_conc_as_list(self, sample_multinode_config, sample_runner_config, full_sweep_args_multi_node):
        """Multinode conc should be passed as list."""
        result = generate_full_sweep(
            full_sweep_args_multi_node,
            sample_multinode_config,
            sample_runner_config
        )
        entry = result[0]
        assert entry["conc"] == [2150]

    def test_single_node_flag_skips_multinode(self, sample_multinode_config, sample_runner_config, full_sweep_args_single_node):
        """Single-node flag should skip multinode configs."""
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_multinode_config,
            sample_runner_config
        )
        assert len(result) == 0

    def test_runner_node_filter_multinode(self, sample_runner_config, full_sweep_args_multi_node):
        """Runner node filter should work with multinode configs."""
        # Create a multinode config with h200 runner (which has 4 nodes)
        config = {
            "test-multinode": {
                "image": "test-image",
                "model": "test-model",
                "model-prefix": "test",
                "precision": "fp4",
                "framework": "dynamo-trt",
                "runner": "h200",
                "multinode": True,
                "disagg": True,
                "kv-p2p-transfer": "nixl",
                "scenarios": {
                    "fixed-seq-len": [

                        {
                            "isl": 1024,
                            "osl": 1024,
                            "search-space": [
                                {
                                    "conc-list": [100],
                                    "prefill": {
                                        "num-worker": 1,
                                        "tp": 4,
                                        "ep": 4,
                                        "dp-attn": False,
                                    },
                                    "decode": {
                                        "num-worker": 1,
                                        "tp": 8,
                                        "ep": 8,
                                        "dp-attn": False,
                                    },
                                }
                            ]
                        }
                    ]
                }
            }
        }
        full_sweep_args_multi_node.runner_type = ["h200"]
        full_sweep_args_multi_node.runner_node_filter = "cw"
        result = generate_full_sweep(
            full_sweep_args_multi_node,
            config,
            sample_runner_config
        )
        # Only h200-cw_0 and h200-cw_1 match "cw" filter
        assert len(result) == 2
        runners = [entry["runner"] for entry in result]
        assert "h200-cw_0" in runners
        assert "h200-cw_1" in runners


# =============================================================================
# Test edge cases and special configurations
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and special configurations."""

    def test_config_with_ep_and_dp_attn(self, sample_runner_config, full_sweep_args_single_node):
        """Config with ep and dp-attn should be handled correctly."""
        config = {
            "test-config": {
                "image": "test-image",
                "model": "test-model",
                "model-prefix": "test",
                "precision": "fp4",
                "framework": "sglang",
                "runner": "b200",
                "multinode": False,
                "scenarios": {
                    "fixed-seq-len": [

                        {
                            "isl": 1024,
                            "osl": 1024,
                            "search-space": [
                                {"tp": 4, "ep": 4, "dp-attn": True, "conc-start": 4, "conc-end": 4}
                            ]
                        }
                    ]
                }
            }
        }
        result = generate_full_sweep(
            full_sweep_args_single_node,
            config,
            sample_runner_config
        )
        assert len(result) == 1
        assert result[0]["ep"] == 4
        assert result[0]["dp-attn"] is True

    def test_config_with_spec_decoding(self, sample_runner_config, full_sweep_args_single_node):
        """Config with spec-decoding should be handled correctly."""
        config = {
            "test-config": {
                "image": "test-image",
                "model": "test-model",
                "model-prefix": "test",
                "precision": "fp4",
                "framework": "trt",
                "runner": "b200",
                "multinode": False,
                "scenarios": {
                    "fixed-seq-len": [

                        {
                            "isl": 1024,
                            "osl": 1024,
                            "search-space": [
                                {"tp": 8, "spec-decoding": "mtp", "conc-start": 4, "conc-end": 4}
                            ]
                        }
                    ]
                }
            }
        }
        result = generate_full_sweep(
            full_sweep_args_single_node,
            config,
            sample_runner_config
        )
        assert len(result) == 1
        assert result[0]["spec-decoding"] == "mtp"

    def test_conc_list_in_single_node(self, sample_runner_config, full_sweep_args_single_node):
        """Single node config with conc-list should work."""
        config = {
            "test-config": {
                "image": "test-image",
                "model": "test-model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "sglang",
                "runner": "mi300x",
                "multinode": False,
                "scenarios": {
                    "fixed-seq-len": [

                        {
                            "isl": 1024,
                            "osl": 1024,
                            "search-space": [
                                {"tp": 8, "conc-list": [4, 16, 64]}
                            ]
                        }
                    ]
                }
            }
        }
        result = generate_full_sweep(
            full_sweep_args_single_node,
            config,
            sample_runner_config
        )
        conc_values = [entry["conc"] for entry in result]
        assert conc_values == [4, 16, 64]

    def test_conc_list_in_single_node_honors_filters(
        self,
        sample_runner_config,
        full_sweep_args_single_node,
    ):
        config = {
            "test-config": {
                "image": "test-image",
                "model": "test-model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "sglang",
                "runner": "mi300x",
                "multinode": False,
                "scenarios": {
                    "fixed-seq-len": [
                        {
                            "isl": 1024,
                            "osl": 1024,
                            "search-space": [
                                {"tp": 8, "conc-list": [4, 16, 64]}
                            ],
                        }
                    ]
                },
            }
        }
        full_sweep_args_single_node.min_conc = 8
        full_sweep_args_single_node.max_conc = 32

        result = generate_full_sweep(
            full_sweep_args_single_node,
            config,
            sample_runner_config,
        )

        assert [entry["conc"] for entry in result] == [16]

    def test_step_size_must_advance(
        self,
        sample_single_node_config,
        sample_runner_config,
        full_sweep_args_single_node,
    ):
        full_sweep_args_single_node.step_size = 1

        with pytest.raises(ValueError, match="greater than 1"):
            generate_full_sweep(
                full_sweep_args_single_node,
                sample_single_node_config,
                sample_runner_config,
            )

    def test_min_conc_cannot_exceed_max_conc(
        self,
        sample_single_node_config,
        sample_runner_config,
        full_sweep_args_single_node,
    ):
        full_sweep_args_single_node.min_conc = 16
        full_sweep_args_single_node.max_conc = 8

        with pytest.raises(ValueError, match="less than or equal"):
            generate_full_sweep(
                full_sweep_args_single_node,
                sample_single_node_config,
                sample_runner_config,
            )

    def test_disagg_defaults_to_false(self, sample_runner_config, full_sweep_args_single_node):
        """disagg should default to False when not specified."""
        config = {
            "test-config": {
                "image": "test-image",
                "model": "test-model",
                "model-prefix": "test",
                "precision": "fp8",
                "framework": "sglang",
                "runner": "mi300x",
                "multinode": False,
                # No disagg field
                "scenarios": {
                    "fixed-seq-len": [

                        {
                            "isl": 1024,
                            "osl": 1024,
                            "search-space": [
                                {"tp": 8, "conc-start": 4, "conc-end": 4}
                            ]
                        }
                    ]
                }
            }
        }
        result = generate_full_sweep(
            full_sweep_args_single_node,
            config,
            sample_runner_config
        )
        assert result[0]["disagg"] is False

    def test_multinode_conc_range_expansion(self, sample_runner_config, full_sweep_args_multi_node):
        """Multinode with conc range should expand to list."""
        config = {
            "test-config": {
                "image": "test-image",
                "model": "test-model",
                "model-prefix": "test",
                "precision": "fp4",
                "framework": "dynamo-trt",
                "runner": "gb200",
                "multinode": True,
                "disagg": True,
                "kv-p2p-transfer": "nixl",
                "scenarios": {
                    "fixed-seq-len": [

                        {
                            "isl": 1024,
                            "osl": 1024,
                            "search-space": [
                                {
                                    "conc-start": 1,
                                    "conc-end": 8,
                                    "prefill": {
                                        "num-worker": 1,
                                        "tp": 4,
                                        "ep": 4,
                                        "dp-attn": False,
                                    },
                                    "decode": {
                                        "num-worker": 1,
                                        "tp": 8,
                                        "ep": 8,
                                        "dp-attn": False,
                                    },
                                }
                            ]
                        }
                    ]
                }
            }
        }
        result = generate_full_sweep(
            full_sweep_args_multi_node,
            config,
            sample_runner_config
        )
        assert len(result) == 1
        # step_size=2: 1, 2, 4, 8
        assert result[0]["conc"] == [1, 2, 4, 8]

    def test_max_ep_creates_config_when_below_min(self, sample_runner_config, full_sweep_args_single_node):
        """max_ep below config's ep should create config with max_ep value."""
        config = {
            "test-config": {
                "image": "test-image",
                "model": "test-model",
                "model-prefix": "test",
                "precision": "fp4",
                "framework": "sglang",
                "runner": "b200",
                "multinode": False,
                "scenarios": {
                    "fixed-seq-len": [

                        {
                            "isl": 1024,
                            "osl": 1024,
                            "search-space": [
                                {"tp": 8, "ep": 8, "conc-start": 4, "conc-end": 4}
                            ]
                        }
                    ]
                }
            }
        }
        full_sweep_args_single_node.max_ep = 2
        result = generate_full_sweep(
            full_sweep_args_single_node,
            config,
            sample_runner_config
        )
        # ep=8 in config, but max_ep=2, so should use ep=2
        assert len(result) == 1
        assert result[0]["ep"] == 2

    def test_max_ep_zero_or_negative_skips(self, sample_runner_config, full_sweep_args_single_node):
        """max_ep of 0 or negative should skip configs."""
        config = {
            "test-config": {
                "image": "test-image",
                "model": "test-model",
                "model-prefix": "test",
                "precision": "fp4",
                "framework": "sglang",
                "runner": "b200",
                "multinode": False,
                "scenarios": {
                    "fixed-seq-len": [

                        {
                            "isl": 1024,
                            "osl": 1024,
                            "search-space": [
                                {"tp": 8, "ep": 8, "conc-start": 4, "conc-end": 4}
                            ]
                        }
                    ]
                }
            }
        }
        for invalid_value in [0, -1, -100]:
            full_sweep_args_single_node.max_ep = invalid_value
            result = generate_full_sweep(
                full_sweep_args_single_node,
                config,
                sample_runner_config
            )
            assert len(result) == 0, f"Expected 0 results for max_ep={invalid_value}"

    def test_multinode_max_conc_zero_or_negative_skips(self, sample_runner_config, full_sweep_args_multi_node):
        """Multinode max_conc of 0 or negative should skip configs."""
        config = {
            "test-config": {
                "image": "test-image",
                "model": "test-model",
                "model-prefix": "test",
                "precision": "fp4",
                "framework": "dynamo-trt",
                "runner": "gb200",
                "multinode": True,
                "disagg": True,
                "kv-p2p-transfer": "nixl",
                "scenarios": {
                    "fixed-seq-len": [

                        {
                            "isl": 1024,
                            "osl": 1024,
                            "search-space": [
                                {
                                    "conc-list": [100, 200, 400],
                                    "prefill": {
                                        "num-worker": 1,
                                        "tp": 4,
                                        "ep": 4,
                                        "dp-attn": False,
                                    },
                                    "decode": {
                                        "num-worker": 1,
                                        "tp": 8,
                                        "ep": 8,
                                        "dp-attn": False,
                                    },
                                }
                            ]
                        }
                    ]
                }
            }
        }
        for invalid_value in [0, -1, -100]:
            full_sweep_args_multi_node.max_conc = invalid_value
            result = generate_full_sweep(
                full_sweep_args_multi_node,
                config,
                sample_runner_config
            )
            assert len(result) == 0, f"Expected 0 results for max_conc={invalid_value}"

    def test_multinode_max_conc_creates_config_when_below_min(self, sample_runner_config, full_sweep_args_multi_node):
        """Multinode max_conc below all values should create config with max_conc."""
        config = {
            "test-config": {
                "image": "test-image",
                "model": "test-model",
                "model-prefix": "test",
                "precision": "fp4",
                "framework": "dynamo-trt",
                "runner": "gb200",
                "multinode": True,
                "disagg": True,
                "kv-p2p-transfer": "nixl",
                "scenarios": {
                    "fixed-seq-len": [

                        {
                            "isl": 1024,
                            "osl": 1024,
                            "search-space": [
                                {
                                    "conc-list": [100, 200, 400],
                                    "prefill": {
                                        "num-worker": 1,
                                        "tp": 4,
                                        "ep": 4,
                                        "dp-attn": False,
                                    },
                                    "decode": {
                                        "num-worker": 1,
                                        "tp": 8,
                                        "ep": 8,
                                        "dp-attn": False,
                                    },
                                }
                            ]
                        }
                    ]
                }
            }
        }
        full_sweep_args_multi_node.max_conc = 1
        result = generate_full_sweep(
            full_sweep_args_multi_node,
            config,
            sample_runner_config
        )
        # All conc values (100, 200, 400) > max_conc (1), so should use [1]
        assert len(result) == 1
        assert result[0]["conc"] == [1]

    def test_combined_max_filters(self, sample_runner_config, full_sweep_args_single_node):
        """Multiple max filters should all apply (tp skip, ep clamp, conc clamp)."""
        config = {
            "test-config": {
                "image": "test-image",
                "model": "test-model",
                "model-prefix": "test",
                "precision": "fp4",
                "framework": "sglang",
                "runner": "b200",
                "multinode": False,
                "scenarios": {
                    "fixed-seq-len": [

                        {
                            "isl": 1024,
                            "osl": 1024,
                            "search-space": [
                                {"tp": 8, "ep": 8, "conc-start": 100, "conc-end": 200},  # should be skipped
                                {"tp": 2, "ep": 8, "conc-start": 100, "conc-end": 200},  # should remain
                            ]
                        }
                    ]
                }
            }
        }
        full_sweep_args_single_node.max_tp = 2
        full_sweep_args_single_node.max_ep = 1
        full_sweep_args_single_node.max_conc = 1

        result = generate_full_sweep(
            full_sweep_args_single_node,
            config,
            sample_runner_config
        )

        assert len(result) == 1
        assert result[0]["tp"] == 2
        assert result[0]["ep"] == 1
        assert result[0]["conc"] == 1

# =============================================================================
# Test argument parsing and defaults
# =============================================================================

class TestCommandLine:
    """Tests for CLI input loading and sweep-selection behavior."""

    @pytest.mark.parametrize("command", ["full-sweep", "test-config"])
    @pytest.mark.parametrize("invalid", [False, True])
    def test_script_from_another_directory(
        self, tmp_path, sample_single_node_config, sample_runner_config,
        command, invalid,
    ):
        """Direct scripts must resolve their own imports and caller-relative inputs."""
        (tmp_path / "master config.yaml").write_text(yaml.safe_dump(sample_single_node_config))
        (tmp_path / "runners.yaml").write_text(yaml.safe_dump(sample_runner_config))
        script = Path(__file__).with_name("generate_sweep_configs.py")
        args = [
            command, "--config-files", "master config.yaml",
            "--runner-config", "runners.yaml", "--seq-lens", "1k1k", "--no-evals",
        ]
        if command == "test-config":
            args += ["--config-keys", "*"]
        if invalid:
            args += ["--all-evals"]

        result = subprocess.run(
            [sys.executable, str(script), *args], cwd=tmp_path,
            capture_output=True, text=True, check=False,
        )

        if invalid:
            assert result.returncode == 2
            assert result.stdout == ""
            assert "--all-evals cannot be combined with --no-evals" in result.stderr
        else:
            assert result.returncode == 0, result.stderr
            assert result.stderr == ""
            rows = json.loads(result.stdout)
            assert [(r["isl"], r["osl"], r["conc"]) for r in rows] == [
                (1024, 1024, 4), (1024, 1024, 8), (1024, 1024, 16),
                (1024, 1024, 32), (1024, 1024, 64),
            ]

    @pytest.mark.parametrize("runner_file", [None, "custom runners.yaml"])
    def test_cli_uses_selected_runner_file(
        self, tmp_path, monkeypatch, sample_single_node_config,
        sample_runner_config, runner_file,
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "master.yaml").write_text(yaml.safe_dump(sample_single_node_config))
        (tmp_path / "configs").mkdir()
        # An explicit override must not fall back to the default inventory.
        (tmp_path / "configs/runners.yaml").write_text("invalid: default inventory")
        selected_file = tmp_path / (runner_file or "configs/runners.yaml")
        sample_runner_config["labels"]["mi300x"] = ["fixture-node-0", "fixture-node-1"]
        selected_file.write_text(yaml.safe_dump(sample_runner_config))
        argv = [
            "generate_sweep_configs.py", "full-sweep",
            "--config-files", "master.yaml", "--single-node", "--no-evals",
            "--runner-node-filter", "fixture-node", "--seq-lens", "1k1k",
            "--max-conc", "4",
        ]
        if runner_file is not None:
            argv.extend(["--runner-config", runner_file])
        monkeypatch.setattr(sys, "argv", argv)

        result = generate_sweep_configs.main()

        assert [(row["runner"], row["conc"]) for row in result] == [
            ("fixture-node-0", 4), ("fixture-node-1", 4),
        ]

    def test_all_evals_cli_marks_every_fixed_sequence_entry(
        self,
        monkeypatch,
        sample_single_node_config,
        sample_runner_config,
    ):
        """--all-evals bypasses the default min-conc/highest-median policy but
        still only evaluates 8k1k (1k1k entries are excluded)."""
        import sys

        from infx.matrix import generate as generate_sweep_configs

        monkeypatch.setattr(
            generate_sweep_configs,
            'load_config_files',
            lambda _: sample_single_node_config,
        )
        monkeypatch.setattr(
            generate_sweep_configs,
            'load_runner_file',
            lambda _: sample_runner_config,
        )
        monkeypatch.setattr(sys, 'argv', [
            'generate_sweep_configs.py',
            'test-config',
            '--config-files', 'dummy.yaml',
            '--config-keys', 'dsr1-fp8-mi300x-sglang',
            '--all-evals',
        ])

        result = generate_sweep_configs.main()

        # Every 8k1k concurrency is marked (5 conc values), and the 1k1k
        # entries are dropped rather than evaluated.
        assert len(result) == 5
        assert {(entry['isl'], entry['osl']) for entry in result} == {
            (8192, 1024),
        }
        assert min(entry['conc'] for entry in result) == 4
        assert all(entry['run-eval'] is True for entry in result)
        assert all(entry['eval-only'] is True for entry in result)

    def test_all_evals_composes_with_evals_only(
        self,
        monkeypatch,
        sample_single_node_config,
        sample_runner_config,
    ):
        import sys

        from infx.matrix import generate as generate_sweep_configs

        monkeypatch.setattr(
            generate_sweep_configs,
            'load_config_files',
            lambda _: sample_single_node_config,
        )
        monkeypatch.setattr(
            generate_sweep_configs,
            'load_runner_file',
            lambda _: sample_runner_config,
        )
        monkeypatch.setattr(sys, 'argv', [
            'generate_sweep_configs.py',
            'test-config',
            '--config-files', 'dummy.yaml',
            '--config-keys', 'dsr1-fp8-mi300x-sglang',
            '--evals-only',
            '--all-evals',
        ])

        result = generate_sweep_configs.main()

        assert len(result) == 5
        assert {(entry['isl'], entry['osl']) for entry in result} == {
            (8192, 1024),
        }
        assert all(entry['run-eval'] is True for entry in result)
        assert all(entry['eval-only'] is True for entry in result)

    def test_trim_conc_reduces_generated_eval_matrix(
        self,
        monkeypatch,
        sample_single_node_config,
        sample_runner_config,
    ):
        import sys

        from infx.matrix import generate as generate_sweep_configs

        monkeypatch.setattr(
            generate_sweep_configs,
            'load_config_files',
            lambda _: sample_single_node_config,
        )
        monkeypatch.setattr(
            generate_sweep_configs,
            'load_runner_file',
            lambda _: sample_runner_config,
        )
        monkeypatch.setattr(sys, 'argv', [
            'generate_sweep_configs.py',
            'test-config',
            '--config-files', 'dummy.yaml',
            '--config-keys', 'dsr1-fp8-mi300x-sglang',
            '--evals-only',
            '--all-evals',
            '--trim-conc',
        ])

        result = generate_sweep_configs.main()

        assert len(result) == 1
        assert result[0]['conc'] == 4
        assert result[0]['run-eval'] is True
        assert result[0]['eval-only'] is True

    def test_trim_conc_updates_multinode_dispatch_concurrency(self):
        low_entry = {
            'prefill': {'num-worker': 1, 'tp': 8},
            'decode': {'num-worker': 0, 'tp': 8},
            'conc': [4],
        }
        high_entry = {
            **low_entry,
            'conc': [64],
            'run-eval': True,
            'eval-conc': 64,
        }

        result = trim_conc([high_entry, low_entry])

        assert len(result) == 1
        assert result[0]['conc'] == [4]
        assert result[0]['eval-conc'] == 4
        assert result[0]['run-eval'] is True

    @pytest.mark.parametrize('entrypoint', ['cli', 'api'])
    def test_smoke_keeps_canonical_eval_instead_of_throughput_minimum(
        self, monkeypatch, sample_single_node_config, sample_runner_config, entrypoint,
    ):
        monkeypatch.setattr(generate_sweep_configs, 'load_config_files', lambda _: sample_single_node_config)
        monkeypatch.setattr(generate_sweep_configs, 'load_runner_file', lambda _: sample_runner_config)
        monkeypatch.setattr(sys, 'argv', ['generate_sweep_configs.py', 'test-config',
                                         '--config-files', 'dummy.yaml', '--config-keys',
                                         'dsr1-fp8-mi300x-sglang', '--smoke'])
        if entrypoint == 'api':
            result = generate_sweep_configs.generate_config_matrix(
                ['dsr1-fp8-mi300x-sglang'], sample_single_node_config,
                sample_runner_config, eval_mode='smoke',
            )
        else:
            result = generate_sweep_configs.main()
        benchmarks = [row for row in result if not row.get('eval-only')]
        evals = [row for row in result if row.get('eval-only')]
        assert {row['conc'] for row in benchmarks} == {4}
        assert all(not row['run-eval'] for row in benchmarks)
        assert {row['conc'] for row in evals} == {32}
        assert all(row['run-eval'] for row in evals)

    def test_multinode_smoke_preserves_representative_eval_concurrency(self):
        from infx.matrix.generate import smoke_entries
        entry = {'prefill': {'num-worker': 1, 'tp': 8}, 'decode': {'num-worker': 1, 'tp': 8},
                 'conc': [4, 32, 64], 'run-eval': True, 'eval-conc': 64}
        result = smoke_entries([entry])
        assert [(row['conc'], row.get('eval-only', False), row['run-eval']) for row in result] == [
            ([4], False, False), ([64], True, True)]
        assert result[1]['eval-conc'] == 64

    def test_all_evals_batches_each_multinode_concurrency(
        self,
        monkeypatch,
        sample_multinode_config,
        sample_runner_config,
    ):
        import sys

        from infx.matrix import generate as generate_sweep_configs

        config = sample_multinode_config
        seq_entry = (
            config['dsr1-fp4-gb200-dynamo-trt']['scenarios']
            ['fixed-seq-len'][0]
        )
        # all-evals only evaluates 8k1k, so target that sequence length.
        seq_entry['isl'] = 8192
        seq_entry['osl'] = 1024
        search_space = seq_entry['search-space']
        search_space[0]['conc-list'] = [4, 16, 64]

        monkeypatch.setattr(
            generate_sweep_configs,
            'load_config_files',
            lambda _: config,
        )
        monkeypatch.setattr(
            generate_sweep_configs,
            'load_runner_file',
            lambda _: sample_runner_config,
        )
        monkeypatch.setattr(sys, 'argv', [
            'generate_sweep_configs.py',
            'test-config',
            '--config-files', 'dummy.yaml',
            '--config-keys', 'dsr1-fp4-gb200-dynamo-trt',
            '--all-evals',
        ])

        result = generate_sweep_configs.main()

        assert len(result) == 1
        assert result[0]['conc'] == [4, 16, 64]
        assert result[0]['eval-all-concs'] is True
        assert 'eval-conc' not in result[0]
        assert all(entry['run-eval'] is True for entry in result)
        assert all(entry['eval-only'] is True for entry in result)

    def test_all_evals_cannot_combine_with_no_evals(self, monkeypatch):
        import sys

        from infx.matrix import generate as generate_sweep_configs

        monkeypatch.setattr(sys, 'argv', [
            'generate_sweep_configs.py',
            'test-config',
            '--config-files', 'dummy.yaml',
            '--config-keys', 'dummy',
            '--no-evals',
            '--all-evals',
        ])

        with pytest.raises(SystemExit):
            generate_sweep_configs.main()


# =============================================================================
# Mixed-mode fixtures
# =============================================================================

@pytest.fixture
def sample_mixed_config(sample_single_node_config, sample_multinode_config):
    """Config dict containing both single-node and multinode entries."""
    merged = {}
    merged.update(sample_single_node_config)
    merged.update(sample_multinode_config)
    return merged


@pytest.fixture
def full_sweep_args_both():
    """Args for full-sweep with both single_node and multi_node True."""
    args = argparse.Namespace()
    args.model_prefix = None
    args.precision = None
    args.framework = None
    args.runner_type = None
    args.seq_lens = None
    args.step_size = 2
    args.min_conc = None
    args.max_conc = None
    args.max_tp = None
    args.max_ep = None
    args.runner_node_filter = None
    args.single_node = True
    args.multi_node = True
    return args


# =============================================================================
# Test generate_test_config_sweep
# =============================================================================

class TestGenerateTestConfigSweep:
    """Tests for exact config-key sweep generation."""

    def test_single_node_parallelism_fields_are_generated(
        self,
        sample_single_node_config,
        sample_runner_config,
    ):
        args = argparse.Namespace(
            config_keys=["dsr1-fp8-mi300x-sglang"],
            seq_lens=["1k1k"],
            conc=[4],
            runner_node_filter=None,
        )

        default_result = generate_test_config_sweep(
            args, sample_single_node_config, sample_runner_config
        )
        assert [
            (row["pp"], row["dcp-size"], row["pcp-size"])
            for row in default_result
        ] == [(1, 1, 1)]

        explicit_config = copy.deepcopy(sample_single_node_config)
        explicit_config["dsr1-fp8-mi300x-sglang"]["scenarios"]["fixed-seq-len"][0]["search-space"][0].update(
            {"pp": 2, "dcp-size": 2, "pcp-size": 2}
        )
        explicit_result = generate_test_config_sweep(
            args, explicit_config, sample_runner_config
        )
        assert [
            (row["pp"], row["dcp-size"], row["pcp-size"])
            for row in explicit_result
        ] == [(2, 2, 2)]

    def test_multinode_parallelism_fields_are_generated(
        self,
        sample_multinode_config,
        sample_runner_config,
    ):
        args = argparse.Namespace(
            config_keys=["dsr1-fp4-gb200-dynamo-trt"],
            seq_lens=["1k1k"],
            conc=None,
            runner_node_filter=None,
        )
        explicit_config = copy.deepcopy(sample_multinode_config)
        search_entry = explicit_config["dsr1-fp4-gb200-dynamo-trt"]["scenarios"]["fixed-seq-len"][0]["search-space"][0]
        search_entry["prefill"].update({"pp": 2, "dcp-size": 2, "pcp-size": 2})
        search_entry["decode"].update({"pp": 2, "dcp-size": 4, "pcp-size": 1})

        entry = generate_test_config_sweep(
            args, explicit_config, sample_runner_config
        )[0]

        assert (
            entry["prefill"]["pp"],
            entry["prefill"]["dcp-size"],
            entry["prefill"]["pcp-size"],
        ) == (2, 2, 2)
        assert (
            entry["decode"]["pp"],
            entry["decode"]["dcp-size"],
            entry["decode"]["pcp-size"],
        ) == (2, 4, 1)

    def test_runner_node_filter_expands_config_runner(self, sample_multinode_config, sample_runner_config):
        """test-config should allow targeting one concrete runner node."""
        args = argparse.Namespace(
            config_keys=["dsr1-fp4-gb200-dynamo-trt"],
            seq_lens=None,
            conc=None,
            runner_node_filter="gb200-nv_0",
        )

        result = generate_test_config_sweep(
            args,
            sample_multinode_config,
            sample_runner_config,
        )

        assert len(result) == 1
        assert result[0]["runner"] == "gb200-nv_0"

    def test_runner_node_filter_no_match_skips_config(self, sample_multinode_config, sample_runner_config):
        """Unmatched node filters should produce no entries."""
        args = argparse.Namespace(
            config_keys=["dsr1-fp4-gb200-dynamo-trt"],
            seq_lens=None,
            conc=None,
            runner_node_filter="gb300-nv_0",
        )

        result = generate_test_config_sweep(
            args,
            sample_multinode_config,
            sample_runner_config,
        )

        assert result == []


@pytest.fixture(params=["full-sweep", "test-config"])
def agentic_mode(request):
    return request.param


@pytest.fixture
def generate_agentic_sweep(agentic_mode, full_sweep_args_single_node):
    def generate(config, runner_data, **filters):
        args = copy.copy(full_sweep_args_single_node)
        vars(args).update(
            config_keys=list(config), conc=None, multi_node=True,
            scenario_type=["agentic-coding"],
        )
        vars(args).update(filters)
        generate = generate_full_sweep if agentic_mode == "full-sweep" else generate_test_config_sweep
        return generate(args, config, runner_data)
    return generate


@pytest.fixture(params=["single", "aggregated", "disaggregated"])
def agentic_config(request, sample_single_node_config):
    config = copy.deepcopy(sample_single_node_config)
    entry = next(iter(config.values()))
    entry.update(runner="cluster:b300-nv", multinode=request.param != "single")
    if request.param == "single":
        benchmark = {"tp": 4, "kv-offloading": "none"}
    elif request.param == "aggregated":
        benchmark = {"num-nodes": 2, "worker": {"num-worker": 2, "tp": 8, "ep": 1, "dp-attn": False}}
    else:
        entry.update(disagg=True, **{"kv-p2p-transfer": "nixl"})
        benchmark = {
            "prefill": {"num-worker": 1, "tp": 8, "ep": 1, "dp-attn": False},
            "decode": {"num-worker": 1, "tp": 8, "ep": 1, "dp-attn": False},
        }
    entry["scenarios"] = {"agentic-coding": [{"search-space": [benchmark]}]}
    return config, benchmark


class TestAgenticGeneration:
    def test_point_order_and_input_preservation(
        self, agentic_config, sample_runner_config, generate_agentic_sweep,
    ):
        config, benchmark = agentic_config
        benchmark["conc-list"] = [32, 8, 32]
        original = copy.deepcopy(config)
        entries = generate_agentic_sweep(config, sample_runner_config, runner_node_filter="b300-nv_")
        if next(iter(config.values()))["multinode"]:
            expected = [
                ("b300-nv_0", [32]), ("b300-nv_0", [8]), ("b300-nv_0", [32]),
                ("b300-nv_1", [32]), ("b300-nv_1", [8]), ("b300-nv_1", [32]),
            ]
        else:
            expected = [
                ("b300-nv_0", 32), ("b300-nv_1", 32),
                ("b300-nv_0", 8), ("b300-nv_1", 8),
                ("b300-nv_0", 32), ("b300-nv_1", 32),
            ]
        assert [(e["runner"], e["conc"]) for e in entries] == expected
        assert config == original

    @pytest.mark.parametrize(("full_filters", "exact_filters", "expected"), [
        ({}, {}, [3, 6, 10]),
        ({"min_conc": 5, "max_conc": 9}, {"conc": [6, 9]}, [6]),
        ({"max_conc": 2}, {"conc": [2]}, []),
        ({"min_conc": 11}, {"conc": [11]}, []),
    ])
    def test_range_boundaries(
        self, agentic_config, sample_runner_config, generate_agentic_sweep,
        agentic_mode, full_filters, exact_filters, expected,
    ):
        config, benchmark = agentic_config
        benchmark.update({"conc-start": 3, "conc-end": 10})
        filters = full_filters if agentic_mode == "full-sweep" else exact_filters
        entries = generate_agentic_sweep(config, sample_runner_config, **filters)
        points = [entry["conc"] for entry in entries]
        assert points == ([[c] for c in expected] if next(iter(config.values()))["multinode"] else expected)

    def test_step_size_and_parallelism_caps_keep_command_semantics(
        self, agentic_config, sample_runner_config, generate_agentic_sweep, agentic_mode,
    ):
        config, benchmark = agentic_config
        benchmark.update({"conc-start": 3, "conc-end": 10})
        # Agentic rows ignore the fixed-sequence TP/EP caps. Only full-sweep
        # takes a custom range step; test-config always doubles concurrency.
        entries = generate_agentic_sweep(
            config, sample_runner_config, step_size=3, max_tp=1, max_ep=0,
        )
        expected = [3, 9, 10] if agentic_mode == "full-sweep" else [3, 6, 10]
        assert [e["conc"] for e in entries] == (
            [[c] for c in expected] if next(iter(config.values()))["multinode"] else expected
        )

    def test_runner_node_filter_expands_agentic_config_runner(self, sample_runner_config, generate_agentic_sweep):
        """Agentic entries support concrete runner targeting through both commands."""
        config = {
            "qwen-agentic-hicache": {
                "image": "sglang-rocm",
                "model": "Qwen/Qwen3.5-397B-A17B-FP8",
                "model-prefix": "qwen3.5",
                "precision": "fp8",
                "framework": "sglang",
                "runner": "cluster:b300-nv",
                "multinode": False,
                "scenarios": {
                    "agentic-coding": [
                        {
                            "dram-utilization": 0.80,
                            "search-space": [
                                {
                                    "tp": 8,
                                    "ep": 1,
                                    "kv-offloading": "dram",
                                    "kv-offload-backend": {"name": "hicache"},
                                    "conc-list": [64],
                                }
                            ],
                        }
                    ]
                },
            }
        }

        result = generate_agentic_sweep(config, sample_runner_config, runner_node_filter="b300-nv_1")

        assert len(result) == 1
        assert result[0]["runner"] == "b300-nv_1"
        assert result[0]["scenario-type"] == "agentic-coding"
        assert result[0]["total-cpu-dram-gb"] == 2399
        assert result[0]["duration"] == 3600

    def test_agentic_node_dram_uses_explicit_gpu_count(self, sample_runner_config, generate_agentic_sweep):
        config = {
            "dsv4-b300-agentic": {
                "image": "vllm/vllm-openai:v0.23.0",
                "model": "deepseek-ai/DeepSeek-V4-Pro",
                "model-prefix": "dsv4",
                "precision": "fp4",
                "framework": "vllm",
                "runner": "cluster:b300-nv",
                "multinode": False,
                "scenarios": {
                    "agentic-coding": [{
                        "dram-utilization": 0.80,
                        "search-space": [
                            {
                                "tp": 4,
                                "kv-offloading": "dram",
                                "kv-offload-backend": {"name": "native"},
                                "conc-list": [32],
                            },
                            {
                                "tp": 4,
                                "dcp-size": 2,
                                "pcp-size": 1,
                                "kv-offloading": "dram",
                                "kv-offload-backend": {"name": "native"},
                                "conc-list": [32],
                            },
                            {
                                "tp": 4,
                                "dcp-size": 1,
                                "pcp-size": 2,
                                "kv-offloading": "dram",
                                "kv-offload-backend": {"name": "native"},
                                "conc-list": [32],
                            },
                            {
                                "tp": 4,
                                "pp": 2,
                                "kv-offloading": "dram",
                                "kv-offload-backend": {"name": "native"},
                                "conc-list": [32],
                            },
                        ],
                    }],
                },
            },
        }

        result = generate_agentic_sweep(config, sample_runner_config)

        budgets = {
            (entry["pp"], entry["dcp-size"], entry["pcp-size"]): entry["total-cpu-dram-gb"]
            for entry in result
        }
        assert budgets == {
            (1, 1, 1): 1199,
            (1, 2, 1): 1199,
            (1, 1, 2): 2399,
            (2, 1, 1): 2399,
        }
        assert all(entry["duration"] == 3600 for entry in result)

    @pytest.mark.parametrize("filters", [{}, {"min_conc": 999, "conc": [999]}])
    def test_agentic_node_dram_rejects_tp_above_runner_gpus(self, sample_runner_config, generate_agentic_sweep, filters):
        config = {
            "dsv4-b300-agentic": {
                "image": "vllm/vllm-openai:v0.23.0",
                "model": "deepseek-ai/DeepSeek-V4-Pro",
                "model-prefix": "dsv4",
                "precision": "fp4",
                "framework": "vllm",
                "runner": "cluster:b300-nv",
                "multinode": False,
                "scenarios": {
                    "agentic-coding": [{
                        "dram-utilization": 0.80,
                        "search-space": [
                            {
                                "tp": 4,
                                "kv-offloading": "dram",
                                "kv-offload-backend": {"name": "native"},
                                "conc-list": [32],
                            },
                        ],
                    }],
                },
            },
        }
        runner_config = copy.deepcopy(sample_runner_config)
        runner_config["hardware"]["cluster:b300-nv"]["gpus-per-node"] = 2

        with pytest.raises(ValueError, match="exceeds gpus-per-node"):
            generate_agentic_sweep(config, runner_config, **filters)

    def test_multinode_agentic_groups_concurrencies_per_search_entry(
        self, sample_runner_config, generate_agentic_sweep
    ):
        """One server allocation should run exactly one concurrency (one task per conc)."""
        config = {
            "dsv4-agentic-2p1d": {
                "image": "vllm/vllm-openai:v0.23.0",
                "model": "deepseek-ai/DeepSeek-V4-Pro",
                "model-prefix": "dsv4",
                "precision": "fp4",
                "framework": "dynamo-vllm",
                "runner": "gb200",
                "multinode": True,
                "disagg": True,
                "kv-p2p-transfer": "nixl",
                "scenarios": {
                    "agentic-coding": [
                        {
                            "search-space": [
                                {
                                    "conc-list": [16, 32, 64, 128, 256],
                                    "prefill": {"hardware": "gb200", "num-worker": 2, "tp": 4, "pp": 2, "dcp-size": 2, "pcp-size": 2, "ep": 4, "dp-attn": False},
                                    "decode": {"hardware": "h100", "num-worker": 1, "tp": 4, "pp": 2, "dcp-size": 2, "pcp-size": 1, "ep": 1, "dp-attn": False},
                                }
                            ],
                        }
                    ]
                },
            }
        }

        result = generate_agentic_sweep(config, sample_runner_config)

        assert len(result) == 5
        assert [entry["conc"] for entry in result] == [[16], [32], [64], [128], [256]]
        assert [entry["exp-name"] for entry in result] == [
            "dsv4_p2x4ep4_d1x4_conc16",
            "dsv4_p2x4ep4_d1x4_conc32",
            "dsv4_p2x4ep4_d1x4_conc64",
            "dsv4_p2x4ep4_d1x4_conc128",
            "dsv4_p2x4ep4_d1x4_conc256",
        ]
        assert result[0]["prefill"]["pp"] == 2
        assert result[0]["prefill"]["dcp-size"] == 2
        assert result[0]["prefill"]["pcp-size"] == 2
        assert result[0]["decode"]["pp"] == 2
        assert result[0]["decode"]["dcp-size"] == 2
        assert result[0]["decode"]["pcp-size"] == 1
        assert {entry["node-count"] for entry in result} == {9}

    def test_multinode_agentic_preserves_kv_offload_fields(self, sample_runner_config, generate_agentic_sweep):
        config = {
            "dsv4-agentic-hicache": {
                "image": "sglang-rocm",
                "model": "deepseek-ai/DeepSeek-V4-Pro",
                "model-prefix": "dsv4",
                "precision": "fp4",
                "framework": "sglang-disagg",
                "runner": "cluster:mi355x-amds",
                "multinode": True,
                "disagg": True,
                "kv-p2p-transfer": "mori",
                "scenarios": {
                    "agentic-coding": [{
                        "dram-utilization": 0.80,
                        "search-space": [{
                            "conc-list": [16],
                            "kv-offloading": "dram",
                            "kv-offload-backend": {"name": "hicache"},
                            "prefill": {"num-worker": 1, "tp": 8, "ep": 1, "dp-attn": False},
                            "decode": {"num-worker": 1, "tp": 8, "ep": 1, "dp-attn": False},
                        }],
                    }],
                },
            },
        }

        result = generate_agentic_sweep(config, sample_runner_config)

        assert len(result) == 1
        assert result[0]["kv-offloading"] == "dram"
        assert result[0]["kv-offload-backend"] == {"name": "hicache"}
        assert result[0]["exp-name"] == "dsv4_p1x8_d1x8_conc16_kvdram-hicache"
        # Budget tracks the prefill worker (the only KV-offloader): tp=8 fills
        # the 8-GPU node -> full utilization share of the (MAX-capped) available
        # DRAM: 2861022 MiB * 0.80.
        assert result[0]["total-cpu-dram-gb"] == 2399

    def test_multinode_agentic_budget_ignores_decode_topology(
        self, sample_runner_config, generate_agentic_sweep
    ):
        """Only prefill offloads today, so decode's topology does not shrink it."""
        config = {
            "dsv4-agentic-hicache-asym": {
                "image": "sglang-rocm",
                "model": "deepseek-ai/DeepSeek-V4-Pro",
                "model-prefix": "dsv4",
                "precision": "fp4",
                "framework": "sglang-disagg",
                "runner": "cluster:mi355x-amds",
                "multinode": True,
                "disagg": True,
                "kv-p2p-transfer": "mori",
                "scenarios": {
                    "agentic-coding": [{
                        "dram-utilization": 0.80,
                        "search-space": [{
                            "conc-list": [16],
                            "kv-offloading": "dram",
                            "kv-offload-backend": {"name": "hicache"},
                            # prefill fills the node (8 GPUs); decode uses half.
                            "prefill": {"num-worker": 1, "tp": 8, "ep": 1, "dp-attn": False},
                            "decode": {"num-worker": 1, "tp": 4, "ep": 1, "dp-attn": False},
                        }],
                    }],
                },
            },
        }

        result = generate_agentic_sweep(config, sample_runner_config)

        assert len(result) == 1
        # prefill 8/8 -> full budget, regardless of decode tp=4.
        assert result[0]["total-cpu-dram-gb"] == 2399

    def test_multinode_agentic_rejects_node_misaligned_prefill(
        self, sample_runner_config, generate_agentic_sweep
    ):
        """A prefill worker whose GPU footprint does not tile the node is rejected."""
        config = {
            "dsv4-agentic-hicache-misaligned": {
                "image": "sglang-rocm",
                "model": "deepseek-ai/DeepSeek-V4-Pro",
                "model-prefix": "dsv4",
                "precision": "fp4",
                "framework": "sglang-disagg",
                "runner": "cluster:mi355x-amds",
                "multinode": True,
                "disagg": True,
                "kv-p2p-transfer": "mori",
                "scenarios": {
                    "agentic-coding": [{
                        "dram-utilization": 0.80,
                        "search-space": [{
                            "conc-list": [16],
                            "kv-offloading": "dram",
                            "kv-offload-backend": {"name": "hicache"},
                            # tp=6 does not divide an 8-GPU node evenly.
                            "prefill": {"num-worker": 1, "tp": 6, "ep": 1, "dp-attn": False},
                            "decode": {"num-worker": 1, "tp": 8, "ep": 1, "dp-attn": False},
                        }],
                    }],
                },
            },
        }

        with pytest.raises(ValueError, match="does not divide"):
            generate_agentic_sweep(config, sample_runner_config)


# =============================================================================
# Test apply_node_type_defaults
# =============================================================================

class TestApplyNodeTypeDefaults:
    """Tests for apply_node_type_defaults function."""

    def test_neither_flag_sets_both_true(self):
        """When neither flag is set, both should become True."""
        args = argparse.Namespace(single_node=False, multi_node=False)
        apply_node_type_defaults(args)
        assert args.single_node is True
        assert args.multi_node is True

    def test_single_only_stays_single(self):
        """When only single_node is set, it stays that way."""
        args = argparse.Namespace(single_node=True, multi_node=False)
        apply_node_type_defaults(args)
        assert args.single_node is True
        assert args.multi_node is False

    def test_multi_only_stays_multi(self):
        """When only multi_node is set, it stays that way."""
        args = argparse.Namespace(single_node=False, multi_node=True)
        apply_node_type_defaults(args)
        assert args.single_node is False
        assert args.multi_node is True

    def test_both_flags_stays_both(self):
        """When both flags are set, they stay that way."""
        args = argparse.Namespace(single_node=True, multi_node=True)
        apply_node_type_defaults(args)
        assert args.single_node is True
        assert args.multi_node is True

    def test_no_node_attrs_is_noop(self):
        """When args lacks node type attrs, nothing happens."""
        args = argparse.Namespace(command="test-config")
        apply_node_type_defaults(args)
        assert not hasattr(args, 'single_node')
        assert not hasattr(args, 'multi_node')


# =============================================================================
# Test generate_full_sweep mixed mode
# =============================================================================

class TestGenerateFullSweepMixed:
    """Tests for generate_full_sweep with both single-node and multi-node configs."""

    @pytest.mark.parametrize(("multinode", "points", "expected"), [
        (False, {"conc-start": 3, "conc-end": 10}, [5, 7]),
        (True, {"conc-start": 3, "conc-end": 10}, [[6]]),
        (False, {"conc-list": [10, 6, 3, 6]}, [6, 6]),
        (True, {"conc-list": [10, 6, 3, 6]}, [[6, 6]]),
    ])
    def test_bounds_clip_single_node_ranges_before_expansion(
        self, sample_single_node_config, sample_multinode_config,
        sample_runner_config, full_sweep_args_both, multinode, points, expected,
    ):
        config = sample_multinode_config if multinode else sample_single_node_config
        sequence = next(iter(config.values()))["scenarios"]["fixed-seq-len"][0]
        benchmark = sequence["search-space"][0]
        for name in ("conc-start", "conc-end", "conc-list"):
            benchmark.pop(name, None)
        benchmark.update(points)
        before = copy.deepcopy(config)
        vars(full_sweep_args_both).update(min_conc=5, max_conc=7, seq_lens=["1k1k"])

        rows = generate_full_sweep(full_sweep_args_both, config, sample_runner_config)

        assert [row["conc"] for row in rows] == expected
        assert config == before

    @pytest.mark.parametrize("command", ["full-sweep", "test-config"])
    @pytest.mark.parametrize("node_names", [[], ["mi300x-amd_1", "mi300x-amd_1", "mi300x-amd_0"]])
    def test_runner_filter_keeps_each_commands_label_and_duplicate_policy(
        self, sample_single_node_config, sample_runner_config,
        full_sweep_args_both, command, node_names,
    ):
        sample_runner_config["labels"]["mi300x"] = node_names
        key, config = next(iter(sample_single_node_config.items()))
        config["scenarios"]["fixed-seq-len"][0]["search-space"] = [{"tp": 8, "conc-list": [4]}]
        vars(full_sweep_args_both).update(
            config_keys=[key], runner_node_filter="mi300x", seq_lens=["1k1k"],
        )
        generate = generate_full_sweep if command == "full-sweep" else generate_test_config_sweep

        rows = generate(full_sweep_args_both, sample_single_node_config, sample_runner_config)

        if command == "full-sweep":
            expected = ["mi300x-amd_1", "mi300x-amd_1", "mi300x-amd_0"] if node_names else []
        else:
            expected = ["mi300x", "mi300x-amd_1", "mi300x-amd_0"] if node_names else ["mi300x"]
        assert [row["runner"] for row in rows] == expected

    def test_typed_full_sweep_selects_configs_and_preserves_scenario_order(
        self, sample_single_node_config, sample_runner_config,
    ):
        config = next(iter(sample_single_node_config.values()))
        config["scenarios"]["agentic-coding"] = [{"search-space": [
            {"tp": 4, "kv-offloading": "none", "conc-list": [16, 8]},
        ]}]
        excluded = copy.deepcopy(config)
        excluded["framework"] = "vllm"
        master = {"selected-vllm": excluded, "selected-sglang": config, "unselected": config}
        before = copy.deepcopy(master)

        rows = generate_sweep_configs.expand_full_sweep(
            master, sample_runner_config,
            options=generate_sweep_configs.FullSweepOptions(
                model_prefix=["selected"], framework=["sglang"], precision=["fp8"],
                runner_type=["mi300x"], runner_node_filter="amd_1",
                seq_lens=["8k1k"], min_conc=5, max_conc=10,
            ),
        )

        assert [(row.get("scenario-type", "fixed-seq-len"), row["conc"]) for row in rows] == [
            ("fixed-seq-len", 5), ("fixed-seq-len", 10), ("agentic-coding", 8),
        ]
        assert [row["runner"] for row in rows] == ["mi300x-amd_1"] * 3
        assert rows[0]["max-model-len"] == 9472
        assert master == before

    @pytest.mark.parametrize(("options", "message"), [
        ({"step_size": 1}, "step_size must be greater than 1"),
        ({"min_conc": 9, "max_conc": 3}, "min_conc must be less than or equal to max_conc"),
        ({"runner_type": ["missing"]}, "Invalid runner type"),
    ])
    def test_typed_full_sweep_validates_options_even_without_configs(
        self, sample_runner_config, options, message,
    ):
        with pytest.raises(ValueError, match=message):
            generate_sweep_configs.expand_full_sweep(
                {}, sample_runner_config,
                options=generate_sweep_configs.FullSweepOptions(**options),
            )

    def test_unbounded_reversed_multinode_range_keeps_empty_batch(
        self, sample_multinode_config, sample_runner_config, full_sweep_args_both,
    ):
        benchmark = next(iter(sample_multinode_config.values()))["scenarios"]["fixed-seq-len"][0]["search-space"][0]
        benchmark.pop("conc-list")
        benchmark.update({"conc-start": 10, "conc-end": 3})

        rows = generate_full_sweep(full_sweep_args_both, sample_multinode_config, sample_runner_config)

        assert [row["conc"] for row in rows] == [[]]
        # Applying a lower bound filters out the empty batch entirely.
        full_sweep_args_both.min_conc = 1
        assert generate_full_sweep(full_sweep_args_both, sample_multinode_config, sample_runner_config) == []

    @pytest.mark.parametrize("command", ["full-sweep", "test-config"])
    def test_unmatched_runner_defers_scenario_access_only_for_selected_keys(
        self, sample_single_node_config, sample_runner_config, full_sweep_args_both, command,
    ):
        key, config = next(iter(sample_single_node_config.items()))
        config.pop("scenarios")
        vars(full_sweep_args_both).update(config_keys=[key], runner_node_filter="missing")

        if command == "test-config":
            assert generate_test_config_sweep(full_sweep_args_both, sample_single_node_config, sample_runner_config) == []
        else:
            with pytest.raises(KeyError, match="scenarios"):
                generate_full_sweep(full_sweep_args_both, sample_single_node_config, sample_runner_config)

    def test_both_flags_generates_mixed(self, sample_mixed_config, sample_runner_config, full_sweep_args_both):
        """Both flags True should produce both single-node and multinode entries."""
        result = generate_full_sweep(
            full_sweep_args_both,
            sample_mixed_config,
            sample_runner_config
        )
        has_single = any("tp" in entry and "prefill" not in entry for entry in result)
        has_multi = any("prefill" in entry for entry in result)
        assert has_single, "Expected single-node entries in mixed output"
        assert has_multi, "Expected multinode entries in mixed output"

    def test_single_node_only_from_mixed(self, sample_mixed_config, sample_runner_config, full_sweep_args_single_node):
        """--single-node should skip multinode entries from mixed config."""
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_mixed_config,
            sample_runner_config
        )
        assert len(result) > 0
        assert all("prefill" not in entry for entry in result), "No multinode entries expected"
        assert all("tp" in entry for entry in result), "All entries should have tp field"

    def test_multi_node_only_from_mixed(self, sample_mixed_config, sample_runner_config, full_sweep_args_multi_node):
        """--multi-node should skip single-node entries from mixed config."""
        result = generate_full_sweep(
            full_sweep_args_multi_node,
            sample_mixed_config,
            sample_runner_config
        )
        assert len(result) > 0
        assert all("prefill" in entry for entry in result), "All entries should be multinode"

    def test_node_type_filters_apply_to_agentic_configs(
        self,
        sample_runner_config,
        full_sweep_args_single_node,
        full_sweep_args_multi_node,
    ):
        """--single-node and --multi-node should split agentic configs too."""
        config = {
            "qwen-agentic": {
                "image": "sglang",
                "model": "Qwen/Qwen3.5-397B-A17B-FP8",
                "model-prefix": "qwen3.5",
                "precision": "fp8",
                "framework": "sglang",
                "runner": "cluster:b300-nv",
                "multinode": False,
                "scenarios": {
                    "agentic-coding": [{
                        "search-space": [
                            {"tp": 4, "pp": 2, "kv-offloading": "none", "conc-list": [16]},
                        ],
                    }],
                },
            },
            "dsv4-agentic-multinode": {
                "image": "vllm/vllm-openai:v0.23.0",
                "model": "deepseek-ai/DeepSeek-V4-Pro",
                "model-prefix": "dsv4",
                "precision": "fp4",
                "framework": "dynamo-vllm",
                "runner": "cluster:gb200-nv",
                "multinode": True,
                "disagg": True,
                "kv-p2p-transfer": "nixl",
                "scenarios": {
                    "agentic-coding": [{
                        "search-space": [
                            {
                                "conc-list": [16],
                                "prefill": {"hardware": "gb200", "num-worker": 2, "tp": 4, "pp": 2, "dcp-size": 2, "pcp-size": 2, "ep": 4, "dp-attn": False},
                                "decode": {"hardware": "h100", "num-worker": 1, "tp": 4, "pp": 2, "dcp-size": 2, "pcp-size": 1, "ep": 1, "dp-attn": False},
                            },
                        ],
                    }],
                },
            },
        }

        single_result = generate_full_sweep(
            full_sweep_args_single_node,
            config,
            sample_runner_config,
        )
        multi_result = generate_full_sweep(
            full_sweep_args_multi_node,
            config,
            sample_runner_config,
        )

        assert len(single_result) == 1
        assert "prefill" not in single_result[0]
        assert single_result[0]["runner"] == "cluster:b300-nv"
        assert single_result[0]["pp"] == 2
        assert len(multi_result) == 1
        assert "prefill" in multi_result[0]
        assert multi_result[0]["runner"] == "cluster:gb200-nv"
        assert (
            multi_result[0]["prefill"]["pp"],
            multi_result[0]["prefill"]["dcp-size"],
            multi_result[0]["prefill"]["pcp-size"],
        ) == (2, 2, 2)
        assert (
            multi_result[0]["decode"]["pp"],
            multi_result[0]["decode"]["dcp-size"],
            multi_result[0]["decode"]["pcp-size"],
        ) == (2, 2, 1)


# =============================================================================
# Test filter_exp_names
# =============================================================================


class TestFilterExpNames:
    def test_selects_exact_names_in_matrix_order(self):
        entries = [
            {"exp-name": "deployment-a", "conc": 1},
            {"exp-name": "deployment-b", "conc": 1},
            {"exp-name": "deployment-c", "conc": 2},
        ]

        result = filter_exp_names(entries, ["deployment-b", "deployment-a"])

        assert result == entries[:2]

    @pytest.mark.parametrize(
        ("entries", "names", "message"),
        (
            ([{"exp-name": "deployment-a"}], ["missing"], "not found"),
            (
                [{"exp-name": "deployment-a"}, {"exp-name": "deployment-a"}],
                ["deployment-a"],
                "multiple rows",
            ),
            (
                [{"exp-name": "deployment-a"}],
                ["deployment-a", "deployment-a"],
                "duplicate values",
            ),
        ),
    )
    def test_rejects_missing_ambiguous_or_duplicate_names(
        self, entries, names, message
    ):
        with pytest.raises(ValueError, match=message):
            filter_exp_names(entries, names)


# =============================================================================
# Test expand_config_keys
# =============================================================================

class TestExpandConfigKeys:
    """Tests for expand_config_keys glob/wildcard matching."""

    AVAILABLE = [
        "dsr1-fp4-b200-sglang",
        "dsr1-fp8-mi300x-sglang",
        "dsr1-fp8-h200-trt",
        "gptoss-fp4-b200-vllm",
        "gptoss-fp8-b200-sglang",
    ]

    def test_exact_keys_pass_through(self):
        """Exact keys should be returned unchanged."""
        result = expand_config_keys(
            ["dsr1-fp4-b200-sglang", "dsr1-fp8-h200-trt"], self.AVAILABLE
        )
        assert result == ["dsr1-fp4-b200-sglang", "dsr1-fp8-h200-trt"]

    def test_star_sglang_matches(self):
        """*-sglang should match all keys ending with -sglang."""
        result = expand_config_keys(["*-sglang"], self.AVAILABLE)
        assert result == [
            "dsr1-fp4-b200-sglang",
            "dsr1-fp8-mi300x-sglang",
            "gptoss-fp8-b200-sglang",
        ]

    def test_prefix_glob(self):
        """dsr1* should match all keys starting with dsr1."""
        result = expand_config_keys(["dsr1*"], self.AVAILABLE)
        assert result == [
            "dsr1-fp4-b200-sglang",
            "dsr1-fp8-mi300x-sglang",
            "dsr1-fp8-h200-trt",
        ]

    def test_question_mark_wildcard(self):
        """? wildcard should match a single character."""
        result = expand_config_keys(["?sr1-fp8-mi300x-sglang"], self.AVAILABLE)
        assert result == ["dsr1-fp8-mi300x-sglang"]

    def test_no_match_pattern_raises(self):
        """Pattern matching nothing should raise ValueError."""
        with pytest.raises(ValueError, match="matched no config keys"):
            expand_config_keys(["*-b300"], self.AVAILABLE)

    def test_missing_exact_key_raises(self):
        """Missing exact key should raise ValueError."""
        with pytest.raises(ValueError, match="Config key\\(s\\) not found"):
            expand_config_keys(["nonexistent-key"], self.AVAILABLE)

    def test_mixed_exact_and_glob(self):
        """Mix of exact keys and glob patterns should work."""
        result = expand_config_keys(
            ["dsr1-fp8-h200-trt", "gptoss*"], self.AVAILABLE
        )
        assert result == [
            "dsr1-fp8-h200-trt",
            "gptoss-fp4-b200-vllm",
            "gptoss-fp8-b200-sglang",
        ]

    def test_overlapping_patterns_deduplicate(self):
        """Overlapping patterns should deduplicate while preserving order."""
        result = expand_config_keys(["dsr1*", "*-sglang"], self.AVAILABLE)
        assert result == [
            "dsr1-fp4-b200-sglang",
            "dsr1-fp8-mi300x-sglang",
            "dsr1-fp8-h200-trt",
            "gptoss-fp8-b200-sglang",
        ]


# =============================================================================
# Tests for e2e-tests.yml workflow config splitting
# =============================================================================

@pytest.fixture
def split_e2e_configs(tmp_path):
    """Run the shipped workflow step; stub only generation and priority scoring."""
    repo_root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((repo_root / ".github/workflows/e2e-tests.yml").read_text())
    step = next(step for step in workflow["jobs"]["get-jobs"]["steps"] if step.get("id") == "get-jobs")
    # Actions resolves these expressions before invoking Bash. Their values
    # are irrelevant to routing, so use a harmless nonempty command/context.
    script = re.sub(r"\$\{\{.*?\}\}", "fixture", step["run"])
    boundary_stubs = r"""#!/bin/bash
case "$*" in
  *generate_sweep_configs.py*|*infx.matrix.generate*) cat "$MATRIX_FIXTURE" ;;
  *ci_priority.py*|*infx.workflows.ci_priority*) cat ;;
  *) exit 1 ;;
esac
"""
    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "uv").write_text(boundary_stubs)
    (tools / "uv").chmod(0o755)

    def run(entries):
        matrix_file = tmp_path / "matrix.json"
        matrix_file.write_text(json.dumps(entries))
        output_file = tmp_path / "outputs"
        output_file.write_text("")
        subprocess.run(
            ["bash", "-euo", "pipefail", "-c", script],
            cwd=tmp_path, check=True, capture_output=True, text=True, timeout=30,
            env={
                **os.environ,
                "PATH": f"{tools}:{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
                "GITHUB_WORKSPACE": str(tmp_path), "GITHUB_OUTPUT": str(output_file),
                "MATRIX_FIXTURE": str(matrix_file), "PR_LABELS": "[]",
                "CHANGELOG_BASE_REF": "", "CHANGELOG_HEAD_REF": "",
                "TRIM_CONC": "false", "ALL_EVALS": "false", "EVALS_ONLY": "false",
            },
        )
        return {
            name: json.loads(value)
            for line in output_file.read_text().splitlines()
            for name, value in [line.split("=", 1)]
        }

    return run


class TestE2EConfigSplitting:
    def test_workflow_routes_benchmarks_and_evals_without_crossing_scenarios(self, split_e2e_configs):
        single = {"exp-name": "single", "run-eval": False}
        single_eval = {"exp-name": "single-eval", "run-eval": True, "recipe-fingerprint": "recipe-a"}
        single_eval_only = {"exp-name": "single-eval-only", "run-eval": True, "eval-only": True}
        multi = {"exp-name": "multi", "prefill": {}, "run-eval": True}
        multi_eval_only = {**multi, "exp-name": "multi-eval-only", "eval-only": True}
        agentic = {"exp-name": "agentic", "scenario-type": "agentic-coding", "run-eval": True}
        agentic_eval_only = {**agentic, "exp-name": "agentic-eval-only", "eval-only": True}
        multi_agentic = {**agentic, "exp-name": "multi-agentic", "prefill": {}}
        multi_agentic_eval_only = {**multi_agentic, "exp-name": "multi-agentic-eval-only", "eval-only": True}

        output = split_e2e_configs([
            single, single_eval, single_eval_only, multi, multi_eval_only,
            agentic, agentic_eval_only, multi_agentic, multi_agentic_eval_only,
        ])

        assert output == {
            "single-node-config": [single, single_eval],
            "eval-config": [single_eval, single_eval_only],
            "multi-node-config": [multi],
            "multi-node-eval-config": [multi, multi_eval_only],
            "agentic-config": [agentic],
            "agentic-eval-config": [agentic, agentic_eval_only],
            "multi-node-agentic-config": [multi_agentic],
            "multi-node-agentic-eval-config": [multi_agentic, multi_agentic_eval_only],
        }

    def test_empty_matrix_has_no_jobs(self, split_e2e_configs):
        output = split_e2e_configs([])

        assert output and all(rows == [] for rows in output.values())


@pytest.mark.parametrize("multinode", [False, True])
def test_require_power_is_scoped_to_one_fixed_sequence(multinode, sample_single_node_config,
                                                       sample_multinode_config, sample_runner_config):
    from infx.matrix.generate import expand_full_sweep, select_matrix_evals
    from infx.matrix.validation import MultiNodeSeqLenConfig, SingleNodeSeqLenConfig

    config = sample_multinode_config if multinode else sample_single_node_config
    entry = next(iter(config.values()))
    sequences = entry["scenarios"]["fixed-seq-len"]
    if multinode:
        sequences.append(copy.deepcopy(sequences[0]))
        sequences[-1]["isl"] = 8192
    before = expand_full_sweep(config, sample_runner_config)
    assert all("require-power" not in row for row in before)
    sequences[-1]["require-power"] = True
    schema = MultiNodeSeqLenConfig if multinode else SingleNodeSeqLenConfig
    schema.model_validate(sequences[-1])
    after = expand_full_sweep(config, sample_runner_config)
    assert len(before) == len(after)
    for original, row in zip(before, after):
        assert row == ({**original, "require-power": True} if original["isl"] == 8192 else original)
    evals = select_matrix_evals(copy.deepcopy(after), mode="subset")
    assert evals
    assert all("require-power" not in row for row in evals)
    sequences[0]["require-power"] = True
    with pytest.raises(ValueError, match="only fixed-sequence 8192/1024"):
        expand_full_sweep(config, sample_runner_config)
