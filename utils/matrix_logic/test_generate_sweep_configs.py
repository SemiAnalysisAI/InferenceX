"""Comprehensive tests for generate_sweep_configs.py"""
import pytest
import argparse
import copy
from generate_sweep_configs import (
    MIN_EVAL_CONC,
    seq_len_stoi,
    seq_len_itos,
    seq_len_to_str,
    generate_full_sweep,
    generate_test_config_sweep,
    mark_eval_entries,
    mark_all_eval_entries,
    apply_node_type_defaults,
    expand_config_keys,
)


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
            "h200": ["h200-cw_0", "h200-cw_1", "h200-nb_0", "h200-nb_1"],
            "b200": ["b200-nvd_0", "b200-nvd_1", "b200-dgxc_1"],
            "b300": ["b300-nv_0", "b300-nv_1"],
            "cluster:b300-nv": ["b300-nv_0", "b300-nv_1"],
            "mi300x": ["mi300x-amd_0", "mi300x-amd_1", "mi300x-cr_0"],
            "gb200": ["gb200-nv_0"],
        },
        "hardware": {
            "cluster:h100-dgxc": {"available-cpu-dram-mib": 2063837, "gpus-per-node": 8},
            "cluster:h200-dgxc": {"available-cpu-dram-mib": 1471356, "gpus-per-node": 8},
            "cluster:b200-dgxc": {"available-cpu-dram-mib": 3774874, "gpus-per-node": 8},
            "cluster:b300-nv": {"available-cpu-dram-mib": 2964436, "gpus-per-node": 8},
            "cluster:mi300x-amds": {"available-cpu-dram-mib": 2321924, "gpus-per-node": 8},
            "cluster:gb200-nv": {"available-cpu-dram-mib": 860160, "gpus-per-node": 4},
        },
    }


@pytest.fixture
def sample_single_node_agentic_config():
    """Single-node agentic config with explicit and default spec decoding."""
    return {
        "kimik2.6-fp4-b300-trt-agentic": {
            "image": "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc5",
            "model": "moonshotai/Kimi-K2.5",
            "model-prefix": "kimik2.6",
            "precision": "fp4",
            "framework": "trt",
            "runner": "cluster:b300-nv",
            "multinode": False,
            "scenarios": {
                "agentic-coding": [{
                    "search-space": [
                        {
                            "tp": 8,
                            "spec-decoding": "mtp",
                            "kv-offloading": "none",
                            "conc-list": [16],
                        },
                        {
                            "tp": 8,
                            "kv-offloading": "none",
                            "conc-list": [32],
                        },
                    ],
                }],
            },
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
# Test seq_len mappings
# =============================================================================

class TestSeqLenMappings:
    """Tests for sequence length string mappings."""

    def test_seq_len_stoi_values(self):
        """Verify seq_len_stoi has expected mappings."""
        assert seq_len_stoi["1k1k"] == (1024, 1024)
        assert seq_len_stoi["8k1k"] == (8192, 1024)

    def test_seq_len_itos_reverse_mapping(self):
        """Verify seq_len_itos is reverse of stoi."""
        assert seq_len_itos[(1024, 1024)] == "1k1k"
        assert seq_len_itos[(8192, 1024)] == "8k1k"


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

    def test_marks_agentic_entry_for_swebench(self):
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
                "runner": "b200-multinode",
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
                "runner": "b200-multinode",
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
                "runner": "b200-multinode",
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
                "runner": "mi355x-disagg",
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
            "runner": "mi355x-disagg",
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

    def test_never_marks_all_entries(self):
        """mark_eval_entries should never mark every single-node entry,
        ensuring the e2e splitting logic can distinguish default from evals-only."""
        entries = [
            {'model': 'm', 'runner': 'r', 'framework': 'f', 'precision': 'fp8',
             'isl': 8192, 'osl': 1024, 'tp': 2, 'conc': c,
             'spec-decoding': False, 'dp-attn': False, 'run-eval': False}
            for c in [32, 64, 128, 256, 512]
        ] + [
            # Non-8k1k entry that should never be marked
            {'model': 'm', 'runner': 'r', 'framework': 'f', 'precision': 'fp8',
             'isl': 1024, 'osl': 1024, 'tp': 2, 'conc': 64,
             'spec-decoding': False, 'dp-attn': False, 'run-eval': False},
        ]
        result = mark_eval_entries(entries)
        non_prefill = [x for x in result if 'prefill' not in x]
        assert not all(x['run-eval'] for x in non_prefill), \
            "mark_eval_entries must not mark all entries — would break e2e splitting"


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

    def test_marks_agentic_entries_for_swebench(self):
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
        assert 'eval-all-concs' not in result[0]


# =============================================================================
# Test generate_full_sweep for single-node
# =============================================================================

class TestGenerateFullSweepSingleNode:
    """Tests for generate_full_sweep with single-node configs."""

    def test_basic_sweep_generation(self, sample_single_node_config, sample_runner_config, full_sweep_args_single_node):
        """Basic single-node sweep should generate entries."""
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_single_node_config,
            sample_runner_config
        )
        assert len(result) > 0
        # With step_size=2, conc goes 4, 8, 16, 32, 64 = 5 values per seq-len config
        # 2 seq-len configs * 5 = 10 entries
        assert len(result) == 10

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

    def test_agentic_spec_decoding_is_propagated(
        self,
        sample_single_node_agentic_config,
        sample_runner_config,
        full_sweep_args_single_node,
    ):
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_single_node_agentic_config,
            sample_runner_config,
        )

        assert [entry["spec-decoding"] for entry in result] == ["mtp", "none"]
        assert result[0]["exp-name"].endswith("_kvnone_spec-mtp")
        assert result[1]["exp-name"].endswith("_kvnone")

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
        for entry in result:
            expected_max_model_len = entry["isl"] + entry["osl"] + 256
            assert entry["max-model-len"] == expected_max_model_len

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
        assert all("amd" in entry["runner"] for entry in result)
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

    def test_multinode_sweep_generation(self, sample_multinode_config, sample_runner_config, full_sweep_args_multi_node):
        """Multinode sweep should generate entries with prefill/decode."""
        result = generate_full_sweep(
            full_sweep_args_multi_node,
            sample_multinode_config,
            sample_runner_config
        )
        assert len(result) == 1  # One entry with conc-list

    def test_multinode_entry_structure(self, sample_multinode_config, sample_runner_config, full_sweep_args_multi_node):
        """Multinode entries should have prefill and decode configs."""
        result = generate_full_sweep(
            full_sweep_args_multi_node,
            sample_multinode_config,
            sample_runner_config
        )
        entry = result[0]
        assert "prefill" in entry
        assert "decode" in entry
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
        assert isinstance(entry["conc"], list)
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
        assert all("cw" in entry["runner"] for entry in result)
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

class TestArgumentDefaults:
    """Tests for command-line argument parsing and default values."""

    def test_runner_config_default_value(self):
        """Verify --runner-config defaults to configs/runners.yaml."""
        import sys
        from generate_sweep_configs import main

        # Save original sys.argv
        original_argv = sys.argv

        try:
            # Simulate command-line args without --runner-config flag
            sys.argv = [
                'generate_sweep_configs.py',
                'full-sweep',
                '--config-files', 'dummy.yaml',
                '--single-node'
            ]

            # Parse args using the ArgumentParser from main
            # We need to access the parser directly
            import argparse
            from generate_sweep_configs import main

            # Create the same parent parser as in main()
            parent_parser = argparse.ArgumentParser(add_help=False)
            parent_parser.add_argument(
                '--config-files',
                nargs='+',
                required=True,
                help='One or more configuration files (YAML format)'
            )
            parent_parser.add_argument(
                '--runner-config',
                default='configs/runners.yaml',
                help='Configuration file holding runner information (YAML format, defaults to configs/runners.yaml)'
            )

            # Create main parser
            parser = argparse.ArgumentParser(
                description='Generate benchmark configurations from YAML config files'
            )

            # Create subparsers
            subparsers = parser.add_subparsers(
                dest='command',
                required=True,
                help='Available commands'
            )

            # Add full-sweep subparser
            full_sweep_parser = subparsers.add_parser(
                'full-sweep',
                parents=[parent_parser],
                add_help=False,
                help='Generate full sweep configurations'
            )
            full_sweep_parser.add_argument('--single-node', action='store_true')
            full_sweep_parser.add_argument('--multi-node', action='store_true')

            # Parse the args
            args = parser.parse_args(['full-sweep', '--config-files', 'dummy.yaml', '--single-node'])

            # Verify the default value
            assert args.runner_config == 'configs/runners.yaml'

        finally:
            # Restore original sys.argv
            sys.argv = original_argv

    def test_runner_config_explicit_value(self):
        """Verify --runner-config can be explicitly set."""
        import argparse

        # Create the same parent parser as in main()
        parent_parser = argparse.ArgumentParser(add_help=False)
        parent_parser.add_argument(
            '--config-files',
            nargs='+',
            required=True,
            help='One or more configuration files (YAML format)'
        )
        parent_parser.add_argument(
            '--runner-config',
            default='configs/runners.yaml',
            help='Configuration file holding runner information (YAML format, defaults to configs/runners.yaml)'
        )

        # Create main parser
        parser = argparse.ArgumentParser(
            description='Generate benchmark configurations from YAML config files'
        )

        # Create subparsers
        subparsers = parser.add_subparsers(
            dest='command',
            required=True,
            help='Available commands'
        )

        # Add full-sweep subparser
        full_sweep_parser = subparsers.add_parser(
            'full-sweep',
            parents=[parent_parser],
            add_help=False,
            help='Generate full sweep configurations'
        )
        full_sweep_parser.add_argument('--single-node', action='store_true')

        # Parse with explicit --runner-config
        args = parser.parse_args([
            'full-sweep',
            '--config-files', 'dummy.yaml',
            '--runner-config', 'custom/path/runners.yaml',
            '--single-node'
        ])

        # Verify the explicit value
        assert args.runner_config == 'custom/path/runners.yaml'

    def test_all_evals_cli_marks_every_fixed_sequence_entry(
        self,
        monkeypatch,
        sample_single_node_config,
        sample_runner_config,
    ):
        """--all-evals bypasses the default min-conc/highest-median policy but
        still only evaluates 8k1k (1k1k entries are excluded)."""
        import sys
        import generate_sweep_configs

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
        import generate_sweep_configs

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

    def test_all_evals_batches_each_multinode_concurrency(
        self,
        monkeypatch,
        sample_multinode_config,
        sample_runner_config,
    ):
        import sys
        import generate_sweep_configs

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
        import generate_sweep_configs

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

    def test_single_node_agentic_spec_decoding_is_propagated(
        self,
        sample_single_node_agentic_config,
        sample_runner_config,
    ):
        args = argparse.Namespace(
            config_keys=["kimik2.6-fp4-b300-trt-agentic"],
            seq_lens=None,
            conc=None,
            scenario_type=["agentic-coding"],
            runner_node_filter=None,
        )

        result = generate_test_config_sweep(
            args,
            sample_single_node_agentic_config,
            sample_runner_config,
        )

        assert [entry["spec-decoding"] for entry in result] == ["mtp", "none"]
        assert result[0]["exp-name"].endswith("_kvnone_spec-mtp")
        assert result[1]["exp-name"].endswith("_kvnone")

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
        master_entry = sample_multinode_config["dsr1-fp4-gb200-dynamo-trt"]
        master_entry["router"] = {"name": "trt-router", "version": "0.20.0"}
        master_entry["kv-p2p-transfer"] = "nixl"
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
        assert result[0]["router"] == {"name": "trt-router", "version": "0.20.0"}
        assert result[0]["kv-p2p-transfer"] == "nixl"

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

    def test_runner_node_filter_expands_agentic_config_runner(self, sample_runner_config):
        """Agentic test-config entries should support concrete runner targeting."""
        config = {
            "qwen-agentic-hicache": {
                "image": "sglang-rocm",
                "model": "Qwen/Qwen3.5-397B-A17B-FP8",
                "model-prefix": "qwen3.5",
                "precision": "fp8",
                "framework": "sglang",
                "runner": "cluster:b300-nv",
                "multinode": False,
                "router": {"name": "default-router", "version": "1.0.0"},
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
        args = argparse.Namespace(
            config_keys=["qwen-agentic-hicache"],
            seq_lens=None,
            conc=None,
            scenario_type=["agentic-coding"],
            runner_node_filter="b300-nv_1",
        )

        result = generate_test_config_sweep(args, config, sample_runner_config)

        assert len(result) == 1
        assert result[0]["runner"] == "b300-nv_1"
        assert result[0]["scenario-type"] == "agentic-coding"
        assert result[0]["total-cpu-dram-gb"] == 2399
        assert result[0]["duration"] == 3600
        assert result[0]["router"] == {"name": "default-router", "version": "1.0.0"}

    def test_agentic_node_dram_uses_explicit_gpu_count(self, sample_runner_config):
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
        args = argparse.Namespace(
            config_keys=["dsv4-b300-agentic"],
            seq_lens=None,
            conc=None,
            scenario_type=["agentic-coding"],
            runner_node_filter=None,
        )

        result = generate_test_config_sweep(args, config, sample_runner_config)

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

    def test_agentic_node_dram_rejects_tp_above_runner_gpus(self, sample_runner_config):
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
        args = argparse.Namespace(
            config_keys=["dsv4-b300-agentic"],
            seq_lens=None,
            conc=None,
            scenario_type=["agentic-coding"],
            runner_node_filter=None,
        )

        with pytest.raises(ValueError, match="exceeds gpus-per-node"):
            generate_test_config_sweep(args, config, runner_config)

    def test_multinode_agentic_groups_concurrencies_per_search_entry(self):
        """One server allocation should run the selected concurrency batch."""
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
                "scenarios": {
                    "agentic-coding": [
                        {
                            "search-space": [
                                {
                                    "conc-list": [16, 32, 64, 128, 256],
                                    "router": {"name": "dynamo-router", "version": "1.3.0"},
                                    "kv-p2p-transfer": "nixl",
                                    "prefill": {"hardware": "gb200", "num-worker": 2, "tp": 4, "pp": 2, "dcp-size": 2, "pcp-size": 2, "ep": 4, "dp-attn": False},
                                    "decode": {"hardware": "h100", "num-worker": 1, "tp": 4, "pp": 2, "dcp-size": 2, "pcp-size": 1, "ep": 1, "dp-attn": False},
                                }
                            ],
                        }
                    ]
                },
            }
        }
        args = argparse.Namespace(
            config_keys=["dsv4-agentic-2p1d"],
            seq_lens=None,
            conc=[16, 32, 64, 128, 256],
            scenario_type=["agentic-coding"],
            runner_node_filter=None,
        )

        result = generate_test_config_sweep(args, config)

        assert len(result) == 2
        assert result[0]["conc"] == [16, 32, 64, 128]
        assert result[0]["exp-name"] == "dsv4_p2x4_d1x4_conc16x32x64x128"
        assert result[0]["prefill"]["pp"] == 2
        assert result[0]["prefill"]["dcp-size"] == 2
        assert result[0]["prefill"]["pcp-size"] == 2
        assert result[0]["decode"]["pp"] == 2
        assert result[0]["decode"]["dcp-size"] == 2
        assert result[0]["decode"]["pcp-size"] == 1
        assert result[1]["conc"] == [256]
        assert result[1]["exp-name"] == "dsv4_p2x4_d1x4_conc256"
        assert all(entry["router"] == {"name": "dynamo-router", "version": "1.3.0"} for entry in result)
        assert all(entry["kv-p2p-transfer"] == "nixl" for entry in result)

    def test_multinode_agentic_preserves_kv_offload_fields(self):
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
        args = argparse.Namespace(
            config_keys=["dsv4-agentic-hicache"],
            seq_lens=None,
            conc=None,
            scenario_type=["agentic-coding"],
            runner_node_filter=None,
        )

        result = generate_test_config_sweep(args, config)

        assert len(result) == 1
        assert result[0]["kv-offloading"] == "dram"
        assert result[0]["kv-offload-backend"] == {"name": "hicache"}
        assert result[0]["exp-name"] == "dsv4_p1x8_d1x8_conc16_kvdram-hicache"


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
                                "conc-list": [16, 32],
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

def _split_e2e_configs(data):
    """Replicate the splitting logic from e2e-tests.yml get-jobs step.

    Returns (SINGLE, MULTI, EVALS) lists matching the workflow filters.
    """
    single = [x for x in data if 'prefill' not in x and not x.get('eval-only', False)]
    multi = [x for x in data if 'prefill' in x and not x.get('eval-only', False)]
    evals = [x for x in data if 'prefill' not in x and x.get('run-eval', False)]
    return single, multi, evals


class TestE2EConfigSplitting:
    """Verify the e2e-tests.yml config splitting logic handles all flag
    combinations correctly: default, --no-evals, --evals-only, and
    --all-evals."""

    @pytest.fixture
    def mixed_entries(self):
        """Simulates default mode output: single-node (some eval-marked),
        plus multi-node entries."""
        return [
            {'exp-name': 'a', 'isl': 1024, 'osl': 1024, 'conc': 64, 'tp': 2, 'run-eval': False},
            {'exp-name': 'b', 'isl': 1024, 'osl': 1024, 'conc': 128, 'tp': 2, 'run-eval': False},
            {'exp-name': 'c', 'isl': 8192, 'osl': 1024, 'conc': 256, 'tp': 2, 'run-eval': True},
            {'exp-name': 'd', 'isl': 8192, 'osl': 1024, 'conc': 512, 'tp': 2, 'run-eval': True},
            {'exp-name': 'e', 'conc': 64, 'prefill': {'tp': 2, 'num-worker': 1}},
        ]

    def test_default_mode_benchmarks_all_single_node(self, mixed_entries):
        """Default: all single-node entries (including eval-marked) are benchmarked."""
        single, multi, evals = _split_e2e_configs(mixed_entries)
        assert len(single) == 4
        assert all('prefill' not in x for x in single)

    def test_default_mode_evals_only_eval_marked(self, mixed_entries):
        """Default: only eval-marked entries go to EVALS."""
        single, multi, evals = _split_e2e_configs(mixed_entries)
        assert len(evals) == 2
        assert all(x['run-eval'] for x in evals)

    def test_default_mode_eval_marked_in_both(self, mixed_entries):
        """Default: eval-marked entries appear in BOTH single and evals."""
        single, multi, evals = _split_e2e_configs(mixed_entries)
        eval_names = {x['exp-name'] for x in evals}
        single_names = {x['exp-name'] for x in single}
        assert eval_names.issubset(single_names)

    def test_no_evals_all_benchmarked(self):
        """--no-evals: mark_eval_entries is skipped, no run-eval=True entries."""
        data = [
            {'exp-name': 'a', 'conc': 64, 'tp': 2, 'run-eval': False},
            {'exp-name': 'b', 'conc': 128, 'tp': 2, 'run-eval': False},
            {'exp-name': 'c', 'conc': 256, 'tp': 2, 'run-eval': False},
        ]
        single, multi, evals = _split_e2e_configs(data)
        assert len(single) == 3
        assert len(evals) == 0

    def test_evals_only_no_benchmarks(self):
        """--evals-only: entries have eval-only flag, SINGLE must be empty."""
        data = [
            {'exp-name': 'c', 'conc': 256, 'tp': 2, 'run-eval': True, 'eval-only': True},
            {'exp-name': 'd', 'conc': 512, 'tp': 2, 'run-eval': True, 'eval-only': True},
        ]
        single, multi, evals = _split_e2e_configs(data)
        assert len(single) == 0, "evals-only should not trigger benchmarks"
        assert len(evals) == 2

    def test_all_evals_routes_every_fixed_sequence_entry_to_evals(self):
        data = [
            {'exp-name': 'a', 'isl': 1024, 'conc': 4, 'tp': 2,
             'run-eval': True, 'eval-only': True},
            {'exp-name': 'b', 'isl': 8192, 'conc': 8, 'tp': 2,
             'run-eval': True, 'eval-only': True},
        ]

        single, multi, evals = _split_e2e_configs(data)

        assert single == []
        assert multi == []
        assert evals == data

    def test_empty_config(self):
        """Empty input produces empty outputs."""
        single, multi, evals = _split_e2e_configs([])
        assert single == [] and multi == [] and evals == []

    def test_all_eval_marked_without_eval_only_flag_still_benchmarked(self):
        """Default mode where mark_eval_entries marks every entry (e.g. only
        8k1k with single conc). Without eval-only flag, SINGLE must still
        include them for benchmarking."""
        data = [
            {'exp-name': 'a', 'conc': 64, 'tp': 2, 'run-eval': True},
            {'exp-name': 'b', 'conc': 64, 'tp': 4, 'run-eval': True},
        ]
        single, multi, evals = _split_e2e_configs(data)
        assert len(single) == 2, "all-eval-marked entries must still be benchmarked in default mode"
        assert len(evals) == 2

    def test_prefill_entries_never_in_single_or_evals(self, mixed_entries):
        """Prefill (multi-node) entries only appear in MULTI."""
        single, multi, evals = _split_e2e_configs(mixed_entries)
        assert len(multi) == 1
        assert all('prefill' in x for x in multi)
        assert all('prefill' not in x for x in single)
        assert all('prefill' not in x for x in evals)
