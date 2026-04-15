"""Comprehensive tests for validation.py"""
import json
from pathlib import Path

import pytest
import yaml
from validation import (
    Fields,
    SingleNodeMatrixEntry,
    MultiNodeMatrixEntry,
    ISB1ReplayMatrixEntry,
    WorkerConfig,
    SingleNodeSearchSpaceEntry,
    MultiNodeSearchSpaceEntry,
    ISB1ReplaySearchSpaceEntry,
    ISB1ReplayConfigEntry,
    SingleNodeSeqLenConfig,
    MultiNodeSeqLenConfig,
    SingleNodeMasterConfigEntry,
    MultiNodeMasterConfigEntry,
    ISB1MasterConfigEntry,
    validate_matrix_entry,
    validate_isb1_matrix_entry,
    validate_master_config,
    validate_isb1_master_config,
    validate_runner_config,
    load_config_files,
    load_isb1_config_files,
    load_runner_file,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def _write_isb1_export_fixture(
    root: Path,
    relative_path: str,
    *,
    runtime_stack_id: str,
    hardware_profile_id: str,
    canonical_model_id: str,
    support_status: str,
    benchmark_certification_status: str = "dataset_replay_verified",
) -> None:
    export_path = root / relative_path
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(
        json.dumps(
            {
                "adapter_id": "inferencex_multiturn",
                "exports": [
                    {
                        "trace_id": f"{export_path.stem}-trace",
                        "runtime_stack_id": runtime_stack_id,
                        "hardware_profile_id": hardware_profile_id,
                        "canonical_model_id": canonical_model_id,
                        "support_status": support_status,
                        "benchmark_certification_status": benchmark_certification_status,
                        "session": {
                            "session_id": "fixture-session",
                            "turns": [
                                {
                                    "turn_idx": 0,
                                    "turn_id": 0,
                                    "messages": [{"role": "user", "content": "hello"}],
                                    "expected_output_tokens": 8,
                                }
                            ],
                        },
                    }
                ],
            }
        )
    )


def _write_manifest_fixture(
    root: Path,
    relative_path: str,
    *,
    export_file: str,
    max_model_len: int,
) -> None:
    manifest_path = root / relative_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": "0.1.0",
                "max_model_len": max_model_len,
                "exports": [{"export_file": export_file}],
            }
        )
    )

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
        "run-eval": False,
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
        "run-eval": False,
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
def valid_isb1_master_config():
    """Valid ISB1 replay master config for NVIDIA PR1a."""
    return {
        "image": "vllm/vllm-openai:v0.8.5",
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "model-prefix": "dsr1",
        "precision": "fp8",
        "framework": "vllm",
        "runner": "h200",
        "benchmark-type": "isb1_replay",
        "runtime-stack-id": "vllm-0.8.5-h200",
        "hardware-profile-id": "h200-8gpu",
        "canonical-model-id": "deepseek-r1-0528",
        "max-model-len": 16384,
        "replay-configs": [
            {
                "export-file": "datasets/isb1/exports/core/chat_8k1k.json",
                "request-mode": "multi-turn",
                "support-status": "supported",
                "search-space": [
                    {
                        "max-concurrency": 4,
                        "max-sessions": 2,
                        "max-turns-per-session": 6,
                        "max-output-len": 512,
                        "num-warmup-sessions": 1,
                        "ignore-waits": True,
                        "ignore-eos": False,
                    },
                    {
                        "max-concurrency": 8,
                    },
                ],
            }
        ],
    }


@pytest.fixture
def valid_isb1_matrix_entry(valid_isb1_master_config):
    """Valid ISB1 replay matrix entry."""
    return {
        "image": valid_isb1_master_config["image"],
        "model": valid_isb1_master_config["model"],
        "model-prefix": valid_isb1_master_config["model-prefix"],
        "precision": valid_isb1_master_config["precision"],
        "framework": valid_isb1_master_config["framework"],
        "runner": valid_isb1_master_config["runner"],
        "benchmark-type": valid_isb1_master_config["benchmark-type"],
        "export-file": valid_isb1_master_config["replay-configs"][0]["export-file"],
        "runtime-stack-id": valid_isb1_master_config["runtime-stack-id"],
        "hardware-profile-id": valid_isb1_master_config["hardware-profile-id"],
        "canonical-model-id": valid_isb1_master_config["canonical-model-id"],
        "support-status": valid_isb1_master_config["replay-configs"][0]["support-status"],
        "request-mode": valid_isb1_master_config["replay-configs"][0]["request-mode"],
        "max-concurrency": 4,
        "max-sessions": 2,
        "max-turns-per-session": 6,
        "max-output-len": 512,
        "num-warmup-sessions": 1,
        "ignore-waits": True,
        "ignore-eos": False,
        "max-model-len": valid_isb1_master_config["max-model-len"],
        "exp-name": "dsr1_isb1",
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
        assert Fields.BENCHMARK_TYPE.value == "benchmark-type"
        assert Fields.SUPPORT_STATUS.value == "support-status"
        assert Fields.MAX_CONCURRENCY.value == "max-concurrency"
        assert Fields.REPLAY_CONFIGS.value == "replay-configs"


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
# Test ISB1 replay models
# =============================================================================

class TestISB1ReplaySearchSpaceEntry:
    """Tests for ISB1ReplaySearchSpaceEntry model."""

    def test_valid_with_required_only(self):
        config = ISB1ReplaySearchSpaceEntry(**{
            "max-concurrency": 4,
        })
        assert config.max_concurrency == 4
        assert config.num_warmup_sessions == 0
        assert config.ignore_waits is False
        assert config.ignore_eos is False

    def test_valid_with_all_fields(self):
        config = ISB1ReplaySearchSpaceEntry(**{
            "max-concurrency": 8,
            "max-sessions": 2,
            "max-turns-per-session": 6,
            "max-output-len": 512,
            "num-warmup-sessions": 1,
            "ignore-waits": True,
            "ignore-eos": True,
        })
        assert config.max_sessions == 2
        assert config.max_turns_per_session == 6
        assert config.max_output_len == 512
        assert config.num_warmup_sessions == 1
        assert config.ignore_waits is True
        assert config.ignore_eos is True

    def test_missing_required_field(self):
        with pytest.raises(Exception):
            ISB1ReplaySearchSpaceEntry(**{
                "max-sessions": 2,
            })

    def test_extra_field_forbidden(self):
        with pytest.raises(Exception):
            ISB1ReplaySearchSpaceEntry(**{
                "max-concurrency": 4,
                "unknown-field": "value",
            })


class TestISB1ReplayConfigEntry:
    """Tests for ISB1ReplayConfigEntry model."""

    def test_valid_entry(self):
        config = ISB1ReplayConfigEntry(**{
            "export-file": "datasets/isb1/exports/core/chat_8k1k.json",
            "request-mode": "multi-turn",
            "support-status": "supported",
            "search-space": [{"max-concurrency": 4}],
        })
        assert config.export_file.endswith("chat_8k1k.json")
        assert config.request_mode == "multi-turn"
        assert config.support_status == "supported"
        assert len(config.search_space) == 1

    def test_invalid_support_status(self):
        with pytest.raises(Exception):
            ISB1ReplayConfigEntry(**{
                "export-file": "datasets/isb1/exports/core/chat_8k1k.json",
                "request-mode": "multi-turn",
                "support-status": "definitely_supported",
                "search-space": [{"max-concurrency": 4}],
            })

    def test_missing_export_file(self):
        with pytest.raises(Exception):
            ISB1ReplayConfigEntry(**{
                "request-mode": "multi-turn",
                "search-space": [{"max-concurrency": 4}],
            })

    def test_missing_request_mode(self):
        with pytest.raises(Exception):
            ISB1ReplayConfigEntry(**{
                "export-file": "datasets/isb1/exports/core/chat_8k1k.json",
                "search-space": [{"max-concurrency": 4}],
            })

    def test_empty_search_space(self):
        with pytest.raises(Exception):
            ISB1ReplayConfigEntry(**{
                "export-file": "datasets/isb1/exports/core/chat_8k1k.json",
                "request-mode": "multi-turn",
                "search-space": [],
            })


class TestISB1MasterConfigEntry:
    """Tests for ISB1MasterConfigEntry model."""

    def test_valid_isb1_master_config(self, valid_isb1_master_config):
        config = ISB1MasterConfigEntry(**valid_isb1_master_config)
        assert config.benchmark_type == "isb1_replay"
        assert config.model_prefix == "dsr1"
        assert config.runner == "h200"
        assert config.max_model_len == 16384
        assert len(config.replay_configs) == 1

    def test_max_model_len_optional(self, valid_isb1_master_config):
        del valid_isb1_master_config["max-model-len"]
        config = ISB1MasterConfigEntry(**valid_isb1_master_config)
        assert config.max_model_len is None

    def test_benchmark_type_must_match(self, valid_isb1_master_config):
        valid_isb1_master_config["benchmark-type"] = "throughput"
        with pytest.raises(Exception):
            ISB1MasterConfigEntry(**valid_isb1_master_config)

    def test_throughput_only_field_rejected(self, valid_isb1_master_config):
        valid_isb1_master_config["multinode"] = False
        with pytest.raises(Exception):
            ISB1MasterConfigEntry(**valid_isb1_master_config)

    def test_missing_required_field(self, valid_isb1_master_config):
        del valid_isb1_master_config["runtime-stack-id"]
        with pytest.raises(Exception):
            ISB1MasterConfigEntry(**valid_isb1_master_config)


class TestISB1ReplayMatrixEntry:
    """Tests for ISB1ReplayMatrixEntry model."""

    def test_valid_entry(self, valid_isb1_matrix_entry):
        entry = ISB1ReplayMatrixEntry(**valid_isb1_matrix_entry)
        assert entry.benchmark_type == "isb1_replay"
        assert entry.support_status == "supported"
        assert entry.max_concurrency == 4
        assert entry.exp_name == "dsr1_isb1"

    def test_missing_required_field(self, valid_isb1_matrix_entry):
        del valid_isb1_matrix_entry["export-file"]
        with pytest.raises(Exception):
            ISB1ReplayMatrixEntry(**valid_isb1_matrix_entry)

    def test_extra_throughput_field_forbidden(self, valid_isb1_matrix_entry):
        valid_isb1_matrix_entry["tp"] = 8
        with pytest.raises(Exception):
            ISB1ReplayMatrixEntry(**valid_isb1_matrix_entry)


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


class TestValidateISB1MasterConfig:
    """Tests for validate_isb1_master_config function."""

    def test_valid_isb1_config(self, valid_isb1_master_config):
        configs = {"dsr1-isb1-h200-vllm": valid_isb1_master_config}
        result = validate_isb1_master_config(configs)
        assert result == configs

    def test_invalid_isb1_config_raises_valueerror(self, valid_isb1_master_config):
        del valid_isb1_master_config["model"]
        configs = {"broken-isb1-config": valid_isb1_master_config}
        with pytest.raises(ValueError) as exc_info:
            validate_isb1_master_config(configs)
        assert "broken-isb1-config" in str(exc_info.value)
        assert "failed validation" in str(exc_info.value)


class TestValidateISB1MatrixEntry:
    """Tests for validate_isb1_matrix_entry function."""

    def test_valid_entry(self, valid_isb1_matrix_entry):
        result = validate_isb1_matrix_entry(valid_isb1_matrix_entry)
        assert result == valid_isb1_matrix_entry

    def test_invalid_entry_raises_valueerror(self, valid_isb1_matrix_entry):
        del valid_isb1_matrix_entry["benchmark-type"]
        with pytest.raises(ValueError) as exc_info:
            validate_isb1_matrix_entry(valid_isb1_matrix_entry)
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


# =============================================================================
# Test load_config_files
# =============================================================================

class TestLoadConfigFiles:
    """Tests for load_config_files function."""

    def test_load_single_file_with_validation(self, tmp_path, valid_single_node_master_config):
        """Should load and validate a single config file."""
        config_file = tmp_path / "config.yaml"
        import yaml
        config_file.write_text(yaml.dump({"test-config": valid_single_node_master_config}))
        result = load_config_files([str(config_file)])
        assert "test-config" in result
        assert result["test-config"]["image"] == valid_single_node_master_config["image"]

    def test_load_single_file_without_validation(self, tmp_path):
        """Should load a single config file without validation when validate=False."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
test-config:
  image: test-image
  model: test-model
""")
        result = load_config_files([str(config_file)], validate=False)
        assert "test-config" in result
        assert result["test-config"]["image"] == "test-image"

    def test_load_multiple_files(self, tmp_path):
        """Should merge multiple config files."""
        config1 = tmp_path / "config1.yaml"
        config1.write_text("""
config-one:
  value: 1
""")
        config2 = tmp_path / "config2.yaml"
        config2.write_text("""
config-two:
  value: 2
""")
        result = load_config_files([str(config1), str(config2)], validate=False)
        assert "config-one" in result
        assert "config-two" in result

    def test_duplicate_keys_raise_error(self, tmp_path):
        """Duplicate keys across files should raise error."""
        config1 = tmp_path / "config1.yaml"
        config1.write_text("""
duplicate-key:
  value: 1
""")
        config2 = tmp_path / "config2.yaml"
        config2.write_text("""
duplicate-key:
  value: 2
""")
        with pytest.raises(ValueError) as exc_info:
            load_config_files([str(config1), str(config2)], validate=False)
        assert "Duplicate configuration keys" in str(exc_info.value)

    def test_nonexistent_file_raises_error(self):
        """Nonexistent file should raise error."""
        with pytest.raises(ValueError) as exc_info:
            load_config_files(["nonexistent.yaml"])
        assert "does not exist" in str(exc_info.value)

    def test_validation_runs_by_default(self, tmp_path):
        """Validation should run by default and catch invalid configs."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
invalid-config:
  image: test-image
  # Missing required fields like model, model-prefix, precision, etc.
""")
        with pytest.raises(ValueError) as exc_info:
            load_config_files([str(config_file)])
        assert "failed validation" in str(exc_info.value)


class TestLoadISB1ConfigFiles:
    """Tests for load_isb1_config_files function."""

    def test_load_single_file_with_validation(self, tmp_path, valid_isb1_master_config):
        config_file = tmp_path / "isb1-config.yaml"
        _write_isb1_export_fixture(
            tmp_path,
            valid_isb1_master_config["replay-configs"][0]["export-file"],
            runtime_stack_id=valid_isb1_master_config["runtime-stack-id"],
            hardware_profile_id=valid_isb1_master_config["hardware-profile-id"],
            canonical_model_id=valid_isb1_master_config["canonical-model-id"],
            support_status=valid_isb1_master_config["replay-configs"][0]["support-status"],
        )

        config_file.write_text(
            yaml.dump({"dsr1-isb1-h200-vllm": valid_isb1_master_config})
        )
        result = load_isb1_config_files([str(config_file)])
        assert "dsr1-isb1-h200-vllm" in result
        assert result["dsr1-isb1-h200-vllm"]["benchmark-type"] == "isb1_replay"

    def test_export_contract_rejects_mismatched_support_status(
        self, tmp_path, valid_isb1_master_config
    ):
        config_file = tmp_path / "isb1-config.yaml"
        _write_isb1_export_fixture(
            tmp_path,
            valid_isb1_master_config["replay-configs"][0]["export-file"],
            runtime_stack_id=valid_isb1_master_config["runtime-stack-id"],
            hardware_profile_id=valid_isb1_master_config["hardware-profile-id"],
            canonical_model_id=valid_isb1_master_config["canonical-model-id"],
            support_status="reviewed_preview",
        )
        config_file.write_text(
            yaml.dump({"dsr1-isb1-h200-vllm": valid_isb1_master_config})
        )

        with pytest.raises(ValueError) as exc_info:
            load_isb1_config_files([str(config_file)])
        assert "support-status" in str(exc_info.value)
        assert "Available support tiers" in str(exc_info.value)

    def test_export_contract_requires_dataset_replay_verified_certification(
        self, tmp_path, valid_isb1_master_config
    ):
        config_file = tmp_path / "isb1-config.yaml"
        _write_isb1_export_fixture(
            tmp_path,
            valid_isb1_master_config["replay-configs"][0]["export-file"],
            runtime_stack_id=valid_isb1_master_config["runtime-stack-id"],
            hardware_profile_id=valid_isb1_master_config["hardware-profile-id"],
            canonical_model_id=valid_isb1_master_config["canonical-model-id"],
            support_status=valid_isb1_master_config["replay-configs"][0]["support-status"],
            benchmark_certification_status="pending_review",
        )
        config_file.write_text(
            yaml.dump({"dsr1-isb1-h200-vllm": valid_isb1_master_config})
        )

        with pytest.raises(ValueError) as exc_info:
            load_isb1_config_files([str(config_file)])
        assert "benchmark_certification_status" in str(exc_info.value)
        assert "dataset_replay_verified" in str(exc_info.value)

    def test_export_contract_requires_max_model_len_for_preview_style_export(
        self, tmp_path, valid_isb1_master_config
    ):
        config_file = tmp_path / "isb1-config.yaml"
        preview_config = {
            **valid_isb1_master_config,
            "replay-configs": [
                {
                    **valid_isb1_master_config["replay-configs"][0],
                    "export-file": (
                        "datasets/isb1/exports/preview/offload_core/"
                        "inferencex_multiturn__chat_hopper_blackwell_offload_core_v1__smoke.json"
                    ),
                    "support-status": "reviewed_preview",
                }
            ],
        }
        del preview_config["max-model-len"]

        _write_isb1_export_fixture(
            tmp_path,
            preview_config["replay-configs"][0]["export-file"],
            runtime_stack_id=preview_config["runtime-stack-id"],
            hardware_profile_id=preview_config["hardware-profile-id"],
            canonical_model_id=preview_config["canonical-model-id"],
            support_status="reviewed_preview",
        )
        config_file.write_text(yaml.dump({"preview-row": preview_config}))

        with pytest.raises(ValueError) as exc_info:
            load_isb1_config_files([str(config_file)])
        assert "max-model-len" in str(exc_info.value)

    def test_export_contract_accepts_preview_style_export_with_explicit_max_model_len(
        self, tmp_path, valid_isb1_master_config
    ):
        config_file = tmp_path / "isb1-config.yaml"
        preview_config = {
            **valid_isb1_master_config,
            "runtime-stack-id": "standalone:vllm",
            "hardware-profile-id": "nvidia:h100_sxm_80gb",
            "canonical-model-id": "gpt_oss_120b",
            "max-model-len": 524288,
            "replay-configs": [
                {
                    **valid_isb1_master_config["replay-configs"][0],
                    "export-file": (
                        "datasets/isb1/exports/preview/long_context_500k/"
                        "inferencex_trace_replay__coding_gptoss_xlc2_500k_preview_v1__vllm.json"
                    ),
                    "support-status": "reviewed_preview",
                }
            ],
        }

        _write_isb1_export_fixture(
            tmp_path,
            preview_config["replay-configs"][0]["export-file"],
            runtime_stack_id=preview_config["runtime-stack-id"],
            hardware_profile_id=preview_config["hardware-profile-id"],
            canonical_model_id=preview_config["canonical-model-id"],
            support_status="reviewed_preview",
        )
        config_file.write_text(yaml.dump({"preview-row": preview_config}))

        result = load_isb1_config_files([str(config_file)])
        assert "preview-row" in result

    def test_export_contract_warns_when_manifest_max_model_len_mismatches_config(
        self, tmp_path, valid_isb1_master_config
    ):
        config_file = tmp_path / "isb1-config.yaml"
        preview_config = {
            **valid_isb1_master_config,
            "runtime-stack-id": "standalone:vllm",
            "hardware-profile-id": "nvidia:h100_sxm_80gb",
            "canonical-model-id": "qwen3_5_397b_a17b",
            "max-model-len": 524288,
            "replay-configs": [
                {
                    **valid_isb1_master_config["replay-configs"][0],
                    "export-file": (
                        "datasets/isb1/exports/preview/long_context_500k/"
                        "inferencex_trace_replay__coding_qwen3.5_xlc2_500k_preview_v1__vllm.json"
                    ),
                    "support-status": "reviewed_preview",
                }
            ],
        }

        export_file = preview_config["replay-configs"][0]["export-file"]
        _write_isb1_export_fixture(
            tmp_path,
            export_file,
            runtime_stack_id=preview_config["runtime-stack-id"],
            hardware_profile_id=preview_config["hardware-profile-id"],
            canonical_model_id=preview_config["canonical-model-id"],
            support_status="reviewed_preview",
        )
        _write_manifest_fixture(
            tmp_path,
            "datasets/isb1/exports/preview/long_context_500k/manifest_qwen3.5.json",
            export_file=export_file,
            max_model_len=1048576,
        )
        config_file.write_text(yaml.dump({"preview-row": preview_config}))

        with pytest.warns(UserWarning, match="max-model-len"):
            result = load_isb1_config_files([str(config_file)])
        assert "preview-row" in result

    def test_load_single_file_without_validation(self, tmp_path):
        config_file = tmp_path / "isb1-config.yaml"
        config_file.write_text("""
test-isb1:
  image: test-image
  benchmark-type: isb1_replay
""")
        result = load_isb1_config_files([str(config_file)], validate=False)
        assert "test-isb1" in result
        assert result["test-isb1"]["benchmark-type"] == "isb1_replay"

    def test_validation_runs_by_default(self, tmp_path):
        config_file = tmp_path / "isb1-config.yaml"
        config_file.write_text("""
invalid-isb1:
  image: test-image
  benchmark-type: isb1_replay
""")
        with pytest.raises(ValueError) as exc_info:
            load_isb1_config_files([str(config_file)])
        assert "failed validation" in str(exc_info.value)

    def test_duplicate_keys_raise_error(self, tmp_path):
        config1 = tmp_path / "config1.yaml"
        config1.write_text("""
duplicate-key:
  benchmark-type: isb1_replay
""")
        config2 = tmp_path / "config2.yaml"
        config2.write_text("""
duplicate-key:
  benchmark-type: isb1_replay
""")
        with pytest.raises(ValueError) as exc_info:
            load_isb1_config_files([str(config1), str(config2)], validate=False)
        assert "Duplicate configuration keys" in str(exc_info.value)

    def test_nonexistent_file_raises_error(self):
        with pytest.raises(ValueError) as exc_info:
            load_isb1_config_files(["nonexistent-isb1.yaml"])
        assert "does not exist" in str(exc_info.value)


# =============================================================================
# Test load_runner_file
# =============================================================================

class TestLoadRunnerFile:
    """Tests for load_runner_file function."""

    def test_load_runner_file_with_validation(self, tmp_path):
        """Should load and validate runner config file."""
        runner_file = tmp_path / "runners.yaml"
        runner_file.write_text("""
h100:
- h100-node-0
- h100-node-1
""")
        result = load_runner_file(str(runner_file))
        assert "h100" in result
        assert len(result["h100"]) == 2

    def test_load_runner_file_without_validation(self, tmp_path):
        """Should load runner config file without validation when validate=False."""
        runner_file = tmp_path / "runners.yaml"
        runner_file.write_text("""
h100:
- h100-node-0
- h100-node-1
""")
        result = load_runner_file(str(runner_file), validate=False)
        assert "h100" in result
        assert len(result["h100"]) == 2

    def test_nonexistent_runner_file(self):
        """Nonexistent runner file should raise error."""
        with pytest.raises(ValueError) as exc_info:
            load_runner_file("nonexistent.yaml")
        assert "does not exist" in str(exc_info.value)

    def test_validation_runs_by_default(self, tmp_path):
        """Validation should run by default and catch invalid configs."""
        runner_file = tmp_path / "runners.yaml"
        runner_file.write_text("""
h100: not-a-list
""")
        with pytest.raises(ValueError) as exc_info:
            load_runner_file(str(runner_file))
        assert "must be a list" in str(exc_info.value)
