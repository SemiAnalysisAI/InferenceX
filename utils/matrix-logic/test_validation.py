"""Comprehensive tests for validation.py"""
import pytest
from validation import (
    Fields,
    SingleNodeMatrixEntry,
    MultiNodeMatrixEntry,
    WorkerConfig,
    SingleNodeSearchSpaceEntry,
    MultiNodeSearchSpaceEntry,
    SingleNodeSeqLenConfig,
    MultiNodeSeqLenConfig,
    SingleNodeMasterConfigEntry,
    MultiNodeMasterConfigEntry,
    validate_matrix_entry,
    validate_master_config,
    validate_runner_config,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def valid_single_node_matrix_entry():
    """Valid single node matrix entry based on dsr1-fp4-mi355x-sglang config."""
    return {
        "image": "rocm/7.0:rocm7.0_ubuntu_22.04_sgl-dev-v0.5.2-rocm7.0-mi35x-20250915",
        "model": "amd/DeepSeek-R1-0528-MXFP4-Preview",
        "model-prefix": "dsr1",
        "precision": "fp4",
        "framework": "sglang",
        "spec-decoding": "none",
        "runner": "mi355x",
        "isl": 1024,
        "osl": 1024,
        "tp": 8,
        "ep": 1,
        "dp-attn": False,
        "conc": 4,
        "max-model-len": 2248,
        "exp-name": "dsr1_1k1k",
        "disagg": False,
    }


@pytest.fixture
def valid_multinode_matrix_entry():
    """Valid multinode matrix entry based on dsr1-fp4-gb200-dynamo-trt config."""
    return {
        "image": "nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:0.5.1-rc0.pre3",
        "model": "deepseek-r1-fp4",
        "model-prefix": "dsr1",
        "precision": "fp4",
        "framework": "dynamo-trt",
        "spec-decoding": "none",
        "runner": "gb200",
        "isl": 1024,
        "osl": 1024,
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
                "DECODE_GPU_MEM_FRACTION=0.8",
                "DECODE_MTP_SIZE=0",
            ],
        },
        "conc": [2150],
        "max-model-len": 2248,
        "exp-name": "dsr1_1k1k",
        "disagg": True,
    }


@pytest.fixture
def valid_single_node_master_config():
    """Valid single node master config based on dsr1-fp8-mi300x-sglang."""
    return {
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
            }
        ]
    }


@pytest.fixture
def valid_multinode_master_config():
    """Valid multinode master config based on dsr1-fp4-gb200-dynamo-trt."""
    return {
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
                        "conc-list": [2150],
                    }
                ]
            }
        ]
    }


@pytest.fixture
def valid_runner_config():
    """Valid runner config based on .github/configs/runners.yaml."""
    return {
        "h100": ["h100-cr_0", "h100-cr_1", "h100-cw_0", "h100-cw_1"],
        "h200": ["h200-cw_0", "h200-cw_1", "h200-nb_0", "h200-nb_1"],
        "b200": ["b200-nvd_0", "b200-nvd_1", "b200-dgxc_1"],
        "mi300x": ["mi300x-amd_0", "mi300x-amd_1", "mi300x-cr_0"],
        "gb200": ["gb200-nv_0"],
    }


# =============================================================================
# Test Fields Enum
# =============================================================================

class TestFieldsEnum:
    """Tests for Fields enum."""

    def test_field_values_are_strings(self):
        """All field values should be strings."""
        for field in Fields:
            assert isinstance(field.value, str)

    def test_key_fields_exist(self):
        """Key fields should be defined."""
        assert Fields.IMAGE.value == "image"
        assert Fields.MODEL.value == "model"
        assert Fields.TP.value == "tp"
        assert Fields.MULTINODE.value == "multinode"
        assert Fields.CONC.value == "conc"
        assert Fields.SPEC_DECODING.value == "spec-decoding"
        assert Fields.PREFILL.value == "prefill"
        assert Fields.DECODE.value == "decode"


# =============================================================================
# Test WorkerConfig
# =============================================================================

class TestWorkerConfig:
    """Tests for WorkerConfig model."""

    def test_valid_worker_config(self):
        """Valid worker config should pass."""
        config = WorkerConfig(**{
            "num-worker": 5,
            "tp": 4,
            "ep": 4,
            "dp-attn": True,
        })
        assert config.num_worker == 5
        assert config.tp == 4
        assert config.ep == 4
        assert config.dp_attn is True

    def test_worker_config_with_additional_settings(self):
        """Worker config with additional settings should pass."""
        config = WorkerConfig(**{
            "num-worker": 1,
            "tp": 8,
            "ep": 8,
            "dp-attn": True,
            "additional-settings": [
                "DECODE_MAX_NUM_TOKENS=256",
                "DECODE_MAX_BATCH_SIZE=256",
                "DECODE_GPU_MEM_FRACTION=0.8",
            ],
        })
        assert len(config.additional_settings) == 3
        assert "DECODE_MAX_NUM_TOKENS=256" in config.additional_settings

    def test_worker_config_missing_required_field(self):
        """Missing required field should fail."""
        with pytest.raises(Exception):
            WorkerConfig(**{
                "num-worker": 2,
                "tp": 4,
                # Missing ep and dp-attn
            })

    def test_worker_config_extra_field_forbidden(self):
        """Extra fields should be forbidden."""
        with pytest.raises(Exception):
            WorkerConfig(**{
                "num-worker": 2,
                "tp": 4,
                "ep": 1,
                "dp-attn": False,
                "unknown-field": "value",
            })


# =============================================================================
# Test SingleNodeMatrixEntry
# =============================================================================

class TestSingleNodeMatrixEntry:
    """Tests for SingleNodeMatrixEntry model."""

    def test_valid_entry(self, valid_single_node_matrix_entry):
        """Valid entry should pass validation."""
        entry = SingleNodeMatrixEntry(**valid_single_node_matrix_entry)
        assert entry.image == "rocm/7.0:rocm7.0_ubuntu_22.04_sgl-dev-v0.5.2-rocm7.0-mi35x-20250915"
        assert entry.tp == 8
        assert entry.conc == 4
        assert entry.framework == "sglang"

    def test_conc_as_list(self, valid_single_node_matrix_entry):
        """Conc can be a list of integers."""
        valid_single_node_matrix_entry["conc"] = [4, 8, 16, 32, 64]
        entry = SingleNodeMatrixEntry(**valid_single_node_matrix_entry)
        assert entry.conc == [4, 8, 16, 32, 64]

    def test_spec_decoding_values(self, valid_single_node_matrix_entry):
        """Spec decoding should accept valid literal values."""
        for value in ["mtp", "draft_model", "none"]:
            valid_single_node_matrix_entry["spec-decoding"] = value
            entry = SingleNodeMatrixEntry(**valid_single_node_matrix_entry)
            assert entry.spec_decoding == value

    def test_invalid_spec_decoding(self, valid_single_node_matrix_entry):
        """Invalid spec decoding value should fail."""
        valid_single_node_matrix_entry["spec-decoding"] = "invalid"
        with pytest.raises(Exception):
            SingleNodeMatrixEntry(**valid_single_node_matrix_entry)

    def test_missing_required_field(self, valid_single_node_matrix_entry):
        """Missing required field should fail validation."""
        del valid_single_node_matrix_entry["model"]
        with pytest.raises(Exception):
            SingleNodeMatrixEntry(**valid_single_node_matrix_entry)

    def test_extra_field_forbidden(self, valid_single_node_matrix_entry):
        """Extra fields should be forbidden."""
        valid_single_node_matrix_entry["extra-field"] = "value"
        with pytest.raises(Exception):
            SingleNodeMatrixEntry(**valid_single_node_matrix_entry)


# =============================================================================
# Test MultiNodeMatrixEntry
# =============================================================================

class TestMultiNodeMatrixEntry:
    """Tests for MultiNodeMatrixEntry model."""

    def test_valid_entry(self, valid_multinode_matrix_entry):
        """Valid entry should pass validation."""
        entry = MultiNodeMatrixEntry(**valid_multinode_matrix_entry)
        assert entry.model == "deepseek-r1-fp4"
        assert entry.conc == [2150]
        assert entry.disagg is True

    def test_prefill_decode_worker_configs(self, valid_multinode_matrix_entry):
        """Prefill and decode should be WorkerConfig objects."""
        entry = MultiNodeMatrixEntry(**valid_multinode_matrix_entry)
        assert entry.prefill.num_worker == 5
        assert entry.prefill.tp == 4
        assert entry.decode.tp == 8
        assert entry.decode.dp_attn is True

    def test_conc_must_be_list(self, valid_multinode_matrix_entry):
        """Conc must be a list for multinode."""
        valid_multinode_matrix_entry["conc"] = 2150  # Single int, not list
        with pytest.raises(Exception):
            MultiNodeMatrixEntry(**valid_multinode_matrix_entry)

    def test_missing_prefill(self, valid_multinode_matrix_entry):
        """Missing prefill should fail."""
        del valid_multinode_matrix_entry["prefill"]
        with pytest.raises(Exception):
            MultiNodeMatrixEntry(**valid_multinode_matrix_entry)

    def test_missing_decode(self, valid_multinode_matrix_entry):
        """Missing decode should fail."""
        del valid_multinode_matrix_entry["decode"]
        with pytest.raises(Exception):
            MultiNodeMatrixEntry(**valid_multinode_matrix_entry)


# =============================================================================
# Test validate_matrix_entry function
# =============================================================================

class TestValidateMatrixEntry:
    """Tests for validate_matrix_entry function."""

    def test_valid_single_node(self, valid_single_node_matrix_entry):
        """Valid single node entry should return the entry."""
        result = validate_matrix_entry(valid_single_node_matrix_entry, is_multinode=False)
        assert result == valid_single_node_matrix_entry

    def test_valid_multinode(self, valid_multinode_matrix_entry):
        """Valid multinode entry should return the entry."""
        result = validate_matrix_entry(valid_multinode_matrix_entry, is_multinode=True)
        assert result == valid_multinode_matrix_entry

    def test_invalid_single_node_raises_valueerror(self, valid_single_node_matrix_entry):
        """Invalid single node entry should raise ValueError."""
        del valid_single_node_matrix_entry["tp"]
        with pytest.raises(ValueError) as exc_info:
            validate_matrix_entry(valid_single_node_matrix_entry, is_multinode=False)
        assert "failed validation" in str(exc_info.value)

    def test_invalid_multinode_raises_valueerror(self, valid_multinode_matrix_entry):
        """Invalid multinode entry should raise ValueError."""
        del valid_multinode_matrix_entry["prefill"]
        with pytest.raises(ValueError) as exc_info:
            validate_matrix_entry(valid_multinode_matrix_entry, is_multinode=True)
        assert "failed validation" in str(exc_info.value)


# =============================================================================
# Test SingleNodeSearchSpaceEntry
# =============================================================================

class TestSingleNodeSearchSpaceEntry:
    """Tests for SingleNodeSearchSpaceEntry model."""

    def test_valid_with_conc_range(self):
        """Valid entry with conc range should pass (like mi300x config)."""
        entry = SingleNodeSearchSpaceEntry(**{
            "tp": 8,
            "conc-start": 4,
            "conc-end": 64,
        })
        assert entry.tp == 8
        assert entry.conc_start == 4
        assert entry.conc_end == 64

    def test_valid_with_conc_list(self):
        """Valid entry with conc list should pass."""
        entry = SingleNodeSearchSpaceEntry(**{
            "tp": 4,
            "conc-list": [4, 8, 16, 32, 64, 128],
        })
        assert entry.conc_list == [4, 8, 16, 32, 64, 128]

    def test_cannot_have_both_range_and_list(self):
        """Cannot specify both conc range and list."""
        with pytest.raises(Exception) as exc_info:
            SingleNodeSearchSpaceEntry(**{
                "tp": 4,
                "conc-start": 4,
                "conc-end": 64,
                "conc-list": [4, 8, 16],
            })
        assert "Cannot specify both" in str(exc_info.value)

    def test_must_have_range_or_list(self):
        """Must specify either conc range or list."""
        with pytest.raises(Exception) as exc_info:
            SingleNodeSearchSpaceEntry(**{
                "tp": 8,
            })
        assert "Must specify either" in str(exc_info.value)

    def test_conc_start_must_be_lte_conc_end(self):
        """conc-start must be <= conc-end."""
        with pytest.raises(Exception) as exc_info:
            SingleNodeSearchSpaceEntry(**{
                "tp": 8,
                "conc-start": 64,
                "conc-end": 4,
            })
        assert "must be <=" in str(exc_info.value)

    def test_conc_list_values_must_be_positive(self):
        """conc-list values must be > 0."""
        with pytest.raises(Exception) as exc_info:
            SingleNodeSearchSpaceEntry(**{
                "tp": 4,
                "conc-list": [4, 0, 16],
            })
        assert "must be greater than 0" in str(exc_info.value)

    def test_optional_fields_defaults(self):
        """Optional fields should have correct defaults."""
        entry = SingleNodeSearchSpaceEntry(**{
            "tp": 8,
            "conc-list": [4, 8],
        })
        assert entry.ep is None
        assert entry.dp_attn is None
        assert entry.spec_decoding == "none"

    def test_with_ep_and_dp_attn(self):
        """Entry with ep and dp-attn like b200-sglang config."""
        entry = SingleNodeSearchSpaceEntry(**{
            "tp": 4,
            "ep": 4,
            "dp-attn": True,
            "conc-start": 4,
            "conc-end": 128,
        })
        assert entry.ep == 4
        assert entry.dp_attn is True

    def test_with_spec_decoding_mtp(self):
        """Entry with mtp spec decoding."""
        entry = SingleNodeSearchSpaceEntry(**{
            "tp": 8,
            "spec-decoding": "mtp",
            "conc-list": [1, 2, 4],
        })
        assert entry.spec_decoding == "mtp"


# =============================================================================
# Test MultiNodeSearchSpaceEntry
# =============================================================================

class TestMultiNodeSearchSpaceEntry:
    """Tests for MultiNodeSearchSpaceEntry model."""

    def test_valid_with_conc_list(self):
        """Valid multinode search space with list (like gb200 config)."""
        entry = MultiNodeSearchSpaceEntry(**{
            "prefill": {
                "num-worker": 5,
                "tp": 4,
                "ep": 4,
                "dp-attn": True,
                "additional-settings": ["PREFILL_MAX_NUM_TOKENS=8448"],
            },
            "decode": {
                "num-worker": 1,
                "tp": 8,
                "ep": 8,
                "dp-attn": True,
                "additional-settings": ["DECODE_MAX_NUM_TOKENS=256"],
            },
            "conc-list": [2150],
        })
        assert entry.prefill.num_worker == 5
        assert entry.decode.tp == 8

    def test_valid_with_conc_range(self):
        """Valid multinode search space with range."""
        entry = MultiNodeSearchSpaceEntry(**{
            "prefill": {
                "num-worker": 1,
                "tp": 4,
                "ep": 4,
                "dp-attn": False,
            },
            "decode": {
                "num-worker": 4,
                "tp": 8,
                "ep": 8,
                "dp-attn": False,
            },
            "conc-start": 1,
            "conc-end": 64,
        })
        assert entry.conc_start == 1
        assert entry.conc_end == 64

    def test_with_spec_decoding_mtp(self):
        """Multinode entry with mtp spec decoding."""
        entry = MultiNodeSearchSpaceEntry(**{
            "spec-decoding": "mtp",
            "prefill": {
                "num-worker": 1,
                "tp": 4,
                "ep": 4,
                "dp-attn": False,
            },
            "decode": {
                "num-worker": 4,
                "tp": 8,
                "ep": 8,
                "dp-attn": False,
            },
            "conc-list": [1, 2, 4, 8, 16, 36],
        })
        assert entry.spec_decoding == "mtp"

    def test_missing_conc_specification(self):
        """Missing conc specification should fail."""
        with pytest.raises(Exception):
            MultiNodeSearchSpaceEntry(**{
                "prefill": {
                    "num-worker": 2,
                    "tp": 4,
                    "ep": 4,
                    "dp-attn": False,
                },
                "decode": {
                    "num-worker": 2,
                    "tp": 4,
                    "ep": 4,
                    "dp-attn": False,
                },
                # Missing conc specification
            })


# =============================================================================
# Test SeqLenConfig models
# =============================================================================

class TestSeqLenConfigs:
    """Tests for sequence length config models."""

    def test_single_node_seq_len_config_1k1k(self):
        """Valid single node seq len config for 1k/1k."""
        config = SingleNodeSeqLenConfig(**{
            "isl": 1024,
            "osl": 1024,
            "search-space": [
                {"tp": 8, "conc-start": 4, "conc-end": 64}
            ]
        })
        assert config.isl == 1024
        assert config.osl == 1024
        assert len(config.search_space) == 1

    def test_single_node_seq_len_config_8k1k(self):
        """Valid single node seq len config for 8k/1k."""
        config = SingleNodeSeqLenConfig(**{
            "isl": 8192,
            "osl": 1024,
            "search-space": [
                {"tp": 8, "conc-start": 4, "conc-end": 64}
            ]
        })
        assert config.isl == 8192
        assert config.osl == 1024

    def test_multinode_seq_len_config(self):
        """Valid multinode seq len config."""
        config = MultiNodeSeqLenConfig(**{
            "isl": 1024,
            "osl": 1024,
            "search-space": [
                {
                    "prefill": {
                        "num-worker": 5,
                        "tp": 4,
                        "ep": 4,
                        "dp-attn": True,
                    },
                    "decode": {
                        "num-worker": 1,
                        "tp": 8,
                        "ep": 8,
                        "dp-attn": True,
                    },
                    "conc-list": [2150],
                }
            ]
        })
        assert config.isl == 1024
        assert config.osl == 1024


# =============================================================================
# Test MasterConfigEntry models
# =============================================================================

class TestMasterConfigEntries:
    """Tests for master config entry models."""

    def test_single_node_master_config(self, valid_single_node_master_config):
        """Valid single node master config."""
        config = SingleNodeMasterConfigEntry(**valid_single_node_master_config)
        assert config.multinode is False
        assert config.model_prefix == "dsr1"
        assert config.runner == "mi300x"
        assert config.framework == "sglang"

    def test_multinode_master_config(self, valid_multinode_master_config):
        """Valid multinode master config."""
        config = MultiNodeMasterConfigEntry(**valid_multinode_master_config)
        assert config.multinode is True
        assert config.model_prefix == "dsr1"
        assert config.runner == "gb200"
        assert config.disagg is True

    def test_single_node_cannot_have_multinode_true(self, valid_single_node_master_config):
        """Single node config must have multinode=False."""
        valid_single_node_master_config["multinode"] = True
        with pytest.raises(Exception):
            SingleNodeMasterConfigEntry(**valid_single_node_master_config)

    def test_multinode_cannot_have_multinode_false(self, valid_multinode_master_config):
        """Multinode config must have multinode=True."""
        valid_multinode_master_config["multinode"] = False
        with pytest.raises(Exception):
            MultiNodeMasterConfigEntry(**valid_multinode_master_config)

    def test_disagg_default_false(self, valid_single_node_master_config):
        """Disagg should default to False."""
        config = SingleNodeMasterConfigEntry(**valid_single_node_master_config)
        assert config.disagg is False


# =============================================================================
# Test validate_master_config function
# =============================================================================

class TestValidateMasterConfig:
    """Tests for validate_master_config function."""

    def test_valid_single_node_config(self, valid_single_node_master_config):
        """Valid single node config should pass."""
        configs = {"dsr1-fp8-mi300x-sglang": valid_single_node_master_config}
        result = validate_master_config(configs)
        assert result == configs

    def test_valid_multinode_config(self, valid_multinode_master_config):
        """Valid multinode config should pass."""
        configs = {"dsr1-fp4-gb200-dynamo-trt": valid_multinode_master_config}
        result = validate_master_config(configs)
        assert result == configs

    def test_mixed_configs(self, valid_single_node_master_config, valid_multinode_master_config):
        """Mixed single and multinode configs should pass."""
        configs = {
            "dsr1-fp8-mi300x-sglang": valid_single_node_master_config,
            "dsr1-fp4-gb200-dynamo-trt": valid_multinode_master_config,
        }
        result = validate_master_config(configs)
        assert len(result) == 2

    def test_invalid_config_raises_valueerror(self, valid_single_node_master_config):
        """Invalid config should raise ValueError with key name."""
        del valid_single_node_master_config["model"]
        configs = {"broken-config": valid_single_node_master_config}
        with pytest.raises(ValueError) as exc_info:
            validate_master_config(configs)
        assert "broken-config" in str(exc_info.value)
        assert "failed validation" in str(exc_info.value)


# =============================================================================
# Test validate_runner_config function
# =============================================================================

class TestValidateRunnerConfig:
    """Tests for validate_runner_config function."""

    def test_valid_runner_config(self, valid_runner_config):
        """Valid runner config should pass."""
        result = validate_runner_config(valid_runner_config)
        assert result == valid_runner_config

    def test_value_must_be_list(self):
        """Runner config values must be lists."""
        config = {
            "h100": "h100-cr_0",  # Not a list
        }
        with pytest.raises(ValueError) as exc_info:
            validate_runner_config(config)
        assert "must be a list" in str(exc_info.value)

    def test_list_must_contain_strings(self):
        """Runner config lists must contain only strings."""
        config = {
            "h100": ["h100-cr_0", 123],  # Contains non-string
        }
        with pytest.raises(ValueError) as exc_info:
            validate_runner_config(config)
        assert "must contain only strings" in str(exc_info.value)

    def test_list_cannot_be_empty(self):
        """Runner config lists cannot be empty."""
        config = {
            "mi355x": [],
        }
        with pytest.raises(ValueError) as exc_info:
            validate_runner_config(config)
        assert "cannot be an empty list" in str(exc_info.value)

    def test_multiple_runner_types(self, valid_runner_config):
        """Multiple runner types should work."""
        result = validate_runner_config(valid_runner_config)
        assert "h100" in result
        assert "h200" in result
        assert "mi300x" in result
        assert "gb200" in result
