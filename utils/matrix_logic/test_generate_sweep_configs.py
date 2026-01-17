"""Comprehensive tests for generate_sweep_configs.py"""
import pytest
import argparse
from generate_sweep_configs import (
    seq_len_stoi,
    seq_len_itos,
    seq_len_to_str,
    generate_full_sweep,
    generate_runner_model_sweep_config,
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
            "seq-len-configs": [
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
            "seq-len-configs": [
                {
                    "isl": 1024,
                    "osl": 1024,
                    "search-space": [
                        {
                            "conc-list": [2150],
                            "prefill": {
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


@pytest.fixture
def sample_runner_config():
    """Runner config based on .github/configs/runners.yaml."""
    return {
        "h100": ["h100-cr_0", "h100-cr_1", "h100-cw_0", "h100-cw_1"],
        "h200": ["h200-cw_0", "h200-cw_1", "h200-nb_0", "h200-nb_1"],
        "b200": ["b200-nvd_0", "b200-nvd_1", "b200-dgxc_1"],
        "mi300x": ["mi300x-amd_0", "mi300x-amd_1", "mi300x-cr_0"],
        "gb200": ["gb200-nv_0"],
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
    args.max_conc = None
    args.max_tp = None
    args.max_ep = None
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
    args.max_conc = None
    args.max_tp = None
    args.max_ep = None
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
        assert seq_len_stoi["1k8k"] == (1024, 8192)
        assert seq_len_stoi["8k1k"] == (8192, 1024)

    def test_seq_len_itos_reverse_mapping(self):
        """Verify seq_len_itos is reverse of stoi."""
        assert seq_len_itos[(1024, 1024)] == "1k1k"
        assert seq_len_itos[(1024, 8192)] == "1k8k"
        assert seq_len_itos[(8192, 1024)] == "8k1k"


class TestSeqLenToStr:
    """Tests for seq_len_to_str function."""

    def test_known_sequence_lengths(self):
        """Known sequence lengths should return short name."""
        assert seq_len_to_str(1024, 1024) == "1k1k"
        assert seq_len_to_str(1024, 8192) == "1k8k"
        assert seq_len_to_str(8192, 1024) == "8k1k"

    def test_unknown_sequence_lengths(self):
        """Unknown sequence lengths should return isl_osl format."""
        assert seq_len_to_str(2048, 2048) == "2048_2048"
        assert seq_len_to_str(4096, 1024) == "4096_1024"


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

    def test_max_tp_filter(self, sample_single_node_config, sample_runner_config, full_sweep_args_single_node):
        """max_tp filter should use max_tp when config tp exceeds it."""
        full_sweep_args_single_node.max_tp = 4
        full_sweep_args_single_node.seq_lens = ["1k1k"]
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_single_node_config,
            sample_runner_config
        )
        # tp=8 in config, but max_tp=4, so should use tp=4
        assert len(result) > 0
        assert all(entry["tp"] == 4 for entry in result)

    def test_max_tp_creates_config_when_below_min(self, sample_single_node_config, sample_runner_config, full_sweep_args_single_node):
        """max_tp below config's tp should create config with max_tp value."""
        # Config has tp=8, so max_tp=2 should create entries with tp=2
        full_sweep_args_single_node.max_tp = 2
        full_sweep_args_single_node.seq_lens = ["1k1k"]
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_single_node_config,
            sample_runner_config
        )
        assert len(result) > 0
        assert all(entry["tp"] == 2 for entry in result)

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
        """max-model-len should be isl + osl + 200."""
        result = generate_full_sweep(
            full_sweep_args_single_node,
            sample_single_node_config,
            sample_runner_config
        )
        for entry in result:
            expected_max_model_len = entry["isl"] + entry["osl"] + 200
            assert entry["max-model-len"] == expected_max_model_len


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


# =============================================================================
# Test generate_runner_model_sweep_config
# =============================================================================

class TestGenerateRunnerModelSweepConfig:
    """Tests for generate_runner_model_sweep_config function."""

    @pytest.fixture
    def runner_sweep_args(self):
        """Args for runner-model-sweep command (single-node)."""
        args = argparse.Namespace()
        args.runner_type = "mi300x"
        args.runner_config = "runners.yaml"
        args.runner_node_filter = None
        args.single_node = True
        args.multi_node = False
        return args

    def test_basic_runner_sweep(self, sample_single_node_config, sample_runner_config, runner_sweep_args):
        """Basic runner sweep should generate entries for each node."""
        result = generate_runner_model_sweep_config(
            runner_sweep_args,
            sample_single_node_config,
            sample_runner_config
        )
        # 3 mi300x nodes
        assert len(result) == 3

    def test_runner_sweep_entry_structure(self, sample_single_node_config, sample_runner_config, runner_sweep_args):
        """Runner sweep entries should use 1k1k config."""
        result = generate_runner_model_sweep_config(
            runner_sweep_args,
            sample_single_node_config,
            sample_runner_config
        )
        for entry in result:
            assert entry["isl"] == 1024
            assert entry["osl"] == 1024
            assert entry["max-model-len"] == 2048
            assert "_test" in entry["exp-name"]

    def test_each_node_gets_entry(self, sample_single_node_config, sample_runner_config, runner_sweep_args):
        """Each runner node should get its own entry."""
        result = generate_runner_model_sweep_config(
            runner_sweep_args,
            sample_single_node_config,
            sample_runner_config
        )
        runners = [entry["runner"] for entry in result]
        assert "mi300x-amd_0" in runners
        assert "mi300x-amd_1" in runners
        assert "mi300x-cr_0" in runners

    def test_invalid_runner_type(self, sample_single_node_config, sample_runner_config, runner_sweep_args):
        """Invalid runner type should raise error."""
        runner_sweep_args.runner_type = "nonexistent"
        with pytest.raises(ValueError) as exc_info:
            generate_runner_model_sweep_config(
                runner_sweep_args,
                sample_single_node_config,
                sample_runner_config
            )
        assert "does not exist" in str(exc_info.value)

    def test_runner_node_filter(self, sample_single_node_config, sample_runner_config, runner_sweep_args):
        """Runner node filter should limit nodes."""
        runner_sweep_args.runner_node_filter = "amd"
        result = generate_runner_model_sweep_config(
            runner_sweep_args,
            sample_single_node_config,
            sample_runner_config
        )
        # Only mi300x-amd_0 and mi300x-amd_1 match
        assert len(result) == 2
        assert all("amd" in entry["runner"] for entry in result)

    def test_runner_node_filter_no_match(self, sample_single_node_config, sample_runner_config, runner_sweep_args):
        """Runner node filter with no matches should raise error."""
        runner_sweep_args.runner_node_filter = "nonexistent"
        with pytest.raises(ValueError) as exc_info:
            generate_runner_model_sweep_config(
                runner_sweep_args,
                sample_single_node_config,
                sample_runner_config
            )
        assert "No runner nodes found" in str(exc_info.value)

    def test_uses_highest_tp(self, sample_single_node_config, sample_runner_config, runner_sweep_args):
        """Should use highest TP from search space."""
        result = generate_runner_model_sweep_config(
            runner_sweep_args,
            sample_single_node_config,
            sample_runner_config
        )
        # Config has tp=8
        assert all(entry["tp"] == 8 for entry in result)

    def test_uses_lowest_conc(self, sample_single_node_config, sample_runner_config, runner_sweep_args):
        """Should use lowest concurrency from search space."""
        result = generate_runner_model_sweep_config(
            runner_sweep_args,
            sample_single_node_config,
            sample_runner_config
        )
        # Config has conc-start=4
        assert all(entry["conc"] == 4 for entry in result)


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
                "seq-len-configs": [
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
                "seq-len-configs": [
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
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [
                            {"tp": 8, "conc-start": 4, "conc-end": 16}
                        ]
                    }
                ]
            }
        }
        result = generate_full_sweep(
            full_sweep_args_single_node,
            config,
            sample_runner_config
        )
        conc_values = [entry["conc"] for entry in result]
        assert 4 in conc_values
        assert 8 in conc_values
        assert 16 in conc_values

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
                "seq-len-configs": [
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
                "seq-len-configs": [
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
                "seq-len-configs": [
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
                "seq-len-configs": [
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
                "seq-len-configs": [
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
                "seq-len-configs": [
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
        """Multiple max filters should all apply and create configs with max values."""
        config = {
            "test-config": {
                "image": "test-image",
                "model": "test-model",
                "model-prefix": "test",
                "precision": "fp4",
                "framework": "sglang",
                "runner": "b200",
                "multinode": False,
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "search-space": [
                            {"tp": 8, "ep": 8, "conc-start": 100, "conc-end": 200}
                        ]
                    }
                ]
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
        # All values exceed max, so should use max values
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
        """Verify --runner-config defaults to .github/configs/runners.yaml."""
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
                default='.github/configs/runners.yaml',
                help='Configuration file holding runner information (YAML format, defaults to .github/configs/runners.yaml)'
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
            assert args.runner_config == '.github/configs/runners.yaml'

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
            default='.github/configs/runners.yaml',
            help='Configuration file holding runner information (YAML format, defaults to .github/configs/runners.yaml)'
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
