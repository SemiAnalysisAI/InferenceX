"""Comprehensive tests for validation.py"""
import copy

import pytest
from validation import (
    ComponentMetadata,
    Fields,
    SingleNodeMatrixEntry,
    SingleNodeAgenticMatrixEntry,
    MultiNodeMatrixEntry,
    WorkerConfig,
    SingleNodeSearchSpaceEntry,
    AgenticCodingConfig,
    AgenticCodingSearchSpaceEntry,
    MultiNodeSearchSpaceEntry,
    SingleNodeSeqLenConfig,
    MultiNodeSeqLenConfig,
    SingleNodeMasterConfigEntry,
    MultiNodeMasterConfigEntry,
    ChangelogEntry,
    ChangelogMatrixEntry,
    validate_matrix_entry,
    validate_master_config,
    validate_runner_config,
    load_config_files,
    load_runner_file,
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
        "pp": 1,
        "dcp-size": 1,
        "pcp-size": 1,
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
                "DECODE_GPU_MEM_FRACTION=0.8",
                "DECODE_MTP_SIZE=0",
            ],
        },
        "conc": [2150],
        "max-model-len": 2248,
        "exp-name": "dsr1_1k1k",
        "disagg": True,
        "kv-p2p-transfer": "nixl",
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
        "scenarios": {
            "fixed-seq-len": [

                {
                    "isl": 1024,
                    "osl": 1024,
                    "search-space": [
                        {"tp": 8, "conc-start": 4, "conc-end": 64}
                    ]
                }
            ]
        }
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
        "kv-p2p-transfer": "nixl",
        "scenarios": {
            "fixed-seq-len": [

                {
                    "isl": 1024,
                    "osl": 1024,
                    "search-space": [
                        {
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
                            "conc-list": [2150],
                        }
                    ]
                }
            ]
        }
    }


@pytest.fixture
def valid_runner_config():
    """Valid runner config based on configs/runners.yaml."""
    return {
        "labels": {
            "h100": ["h100-cr_0", "h100-cr_1", "h100-cw_0", "h100-cw_1"],
            "h200": ["h200-cw_0", "h200-cw_1", "h200-nb_0", "h200-nb_1"],
            "b200": ["b200-nvd_0", "b200-nvd_1", "b200-dgxc_1"],
            "cluster:b200-dgxc": ["b200-dgxc_1"],
            "mi300x": ["mi300x-amd_0", "mi300x-amd_1", "mi300x-cr_0"],
            "gb200": ["gb200-nv_0"],
        },
        "hardware": {
            "cluster:h100-dgxc": {"available-cpu-dram-mib": 2063837, "gpus-per-node": 8},
            "cluster:h200-dgxc": {"available-cpu-dram-mib": 1471356, "gpus-per-node": 8},
            "cluster:b200-dgxc": {"available-cpu-dram-mib": 3774874, "gpus-per-node": 8},
            "cluster:mi300x-amds": {"available-cpu-dram-mib": 2321924, "gpus-per-node": 8},
            "cluster:gb200-nv": {"available-cpu-dram-mib": 860160, "gpus-per-node": 4},
        },
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
        assert Fields.HARDWARE.value == "hardware"


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
        assert (config.pp, config.dcp_size, config.pcp_size) == (1, 1, 1)

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

    def test_worker_parallelism_fields(self):
        config = WorkerConfig(**{
            "num-worker": 2,
            "tp": 4,
            "pp": 2,
            "dcp-size": 2,
            "pcp-size": 2,
            "ep": 1,
            "dp-attn": False,
        })
        assert (config.pp, config.dcp_size, config.pcp_size) == (2, 2, 2)

    @pytest.mark.parametrize("field", ["pp", "dcp-size", "pcp-size"])
    def test_worker_parallelism_fields_must_be_positive(self, field):
        with pytest.raises(Exception, match="greater than 0"):
            WorkerConfig(**{
                "num-worker": 2,
                "tp": 4,
                field: 0,
                "ep": 1,
                "dp-attn": False,
            })

    def test_worker_dcp_size_must_divide_tp(self):
        with pytest.raises(Exception, match="must be divisible"):
            WorkerConfig(**{
                "num-worker": 2,
                "tp": 4,
                "dcp-size": 3,
                "ep": 1,
                "dp-attn": False,
            })

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

    def test_disagg_requires_multinode(self, valid_single_node_matrix_entry):
        """Single-node matrix entries cannot enable disaggregation."""
        valid_single_node_matrix_entry["disagg"] = True
        with pytest.raises(Exception, match="disagg"):
            SingleNodeMatrixEntry(**valid_single_node_matrix_entry)


# =============================================================================
# Test Agentic Matrix Entries
# =============================================================================

class TestAgenticMatrixEntries:
    """Tests for agentic coding validation models."""

    def test_arbitrary_backend_is_valid_for_single_node_agentic_entry(self):
        entry = SingleNodeAgenticMatrixEntry(**{
            "image": "cquil/vllm-openai:v0.21.0-8813c92",
            "model": "deepseek-ai/DeepSeek-V4-Pro",
            "model-prefix": "dsv4",
            "precision": "fp4",
            "framework": "vllm",
            "runner": "cluster:b200-dgxc",
            "tp": 8,
            "pp": 1,
            "dcp-size": 1,
            "pcp-size": 1,
            "ep": 1,
            "dp-attn": False,
            "conc": 1,
            "kv-offloading": "dram",
            "kv-offload-backend": {"name": "future-backend"},
            "total-cpu-dram-gb": 2949,
            "duration": 3600,
            "exp-name": "dsv4_tp8_conc1_kvdram-future-backend",
            "scenario-type": "agentic-coding",
        })
        assert entry.kv_offloading == "dram"
        assert entry.kv_offload_backend.name == "future-backend"
        assert entry.kv_offload_backend.version is None

    def test_arbitrary_backend_is_valid_for_agentic_search_space(self):
        entry = AgenticCodingSearchSpaceEntry(**{
            "tp": 8,
            "kv-offloading": "dram",
            "kv-offload-backend": {"name": "future-backend"},
            "router": {"name": "vllm-router", "version": "0.1.14"},
            "kv-p2p-transfer": "mooncake",
            "conc-list": [1, 2],
        })
        assert entry.kv_offloading == "dram"
        assert entry.kv_offload_backend.name == "future-backend"
        assert entry.kv_offload_backend.version is None
        assert entry.router.name == "vllm-router"
        assert entry.router.version == "0.1.14"
        assert entry.kv_p2p_transfer == "mooncake"

    def test_router_metadata_is_optional_for_fixed_sequence_search_space(self):
        entry = SingleNodeSearchSpaceEntry(**{
            "tp": 8,
            "router": {"name": "vllm-router", "version": "0.1.14"},
            "conc-list": [1, 2],
        })
        assert entry.router.name == "vllm-router"

    @pytest.mark.parametrize("metadata", [
        {"name": "vllm-router"},
        {"version": "0.1.14"},
        {"name": "vllm-router", "version": "0.1.14", "mode": "round-robin"},
        {"name": "", "version": "0.1.14"},
        {"name": "vllm-router", "version": ""},
    ])
    def test_component_metadata_requires_exact_non_empty_fields(self, metadata):
        with pytest.raises(Exception):
            AgenticCodingSearchSpaceEntry(**{
                "tp": 8,
                "kv-offloading": "none",
                "router": metadata,
                "conc-list": [1, 2],
            })

    @pytest.mark.parametrize("value", ["", {"name": "nixl"}])
    def test_kv_p2p_transfer_requires_a_non_empty_name(self, value):
        with pytest.raises(Exception):
            MultiNodeSearchSpaceEntry(**{
                "prefill": {
                    "num-worker": 1, "tp": 8, "ep": 1, "dp-attn": False,
                },
                "decode": {
                    "num-worker": 1, "tp": 8, "ep": 1, "dp-attn": False,
                },
                "kv-p2p-transfer": value,
                "conc-list": [1],
            })

    def test_kv_offload_backend_accepts_optional_version(self):
        entry = AgenticCodingSearchSpaceEntry(**{
            "tp": 8,
            "kv-offloading": "dram",
            "kv-offload-backend": {"name": "lmcache", "version": "0.5.1"},
            "conc-list": [1],
        })

        assert entry.kv_offload_backend.name == "lmcache"
        assert entry.kv_offload_backend.version == "0.5.1"

    def test_kv_offload_backend_rejects_unknown_metadata(self):
        with pytest.raises(Exception):
            AgenticCodingSearchSpaceEntry(**{
                "tp": 8,
                "kv-offloading": "dram",
                "kv-offload-backend": {"name": "lmcache", "mode": "cpu"},
                "conc-list": [1],
            })

    def test_kv_offload_backend_rejects_image_as_version(self):
        with pytest.raises(Exception, match="not an image reference"):
            AgenticCodingSearchSpaceEntry(**{
                "tp": 8,
                "kv-offloading": "dram",
                "kv-offload-backend": {
                    "name": "lmcache",
                    "version": "image:vllm/vllm-openai:v0.23.0",
                },
                "conc-list": [1],
            })

    def test_kv_offload_backend_requires_dram_mode(self):
        with pytest.raises(Exception, match="kv-offload-backend"):
            AgenticCodingSearchSpaceEntry(**{
                "tp": 8,
                "kv-offloading": "none",
                "kv-offload-backend": {"name": "lmcache"},
                "conc-list": [1, 2],
            })

    def test_dram_kv_offload_requires_backend(self):
        with pytest.raises(Exception, match="kv-offload-backend"):
            AgenticCodingSearchSpaceEntry(**{
                "tp": 8,
                "kv-offloading": "dram",
                "conc-list": [1, 2],
            })

    def test_single_node_agentic_requires_explicit_kv_offloading(self):
        with pytest.raises(Exception, match="kv-offloading"):
            AgenticCodingSearchSpaceEntry(**{
                "tp": 8,
                "conc-list": [1, 2],
            })

    def test_dram_kv_offload_requires_dram_utilization(self):
        with pytest.raises(Exception, match="dram-utilization"):
            AgenticCodingConfig(**{
                "search-space": [{
                    "tp": 4,
                    "kv-offloading": "dram",
                    "kv-offload-backend": {"name": "native"},
                    "conc-list": [16],
                }],
            })

    def test_agentic_search_space_rejects_total_cpu_dram_gb(self):
        with pytest.raises(Exception, match="total-cpu-dram-gb"):
            AgenticCodingSearchSpaceEntry(**{
                "tp": 8,
                "kv-offloading": "dram",
                "kv-offload-backend": {"name": "native"},
                "total-cpu-dram-gb": 1000,
                "conc-list": [1, 2],
            })

    def test_dram_kv_offload_accepts_scaled_capacity(self):
        config = AgenticCodingConfig(**{
            "dram-utilization": 0.80,
            "search-space": [{
                "tp": 4,
                "kv-offloading": "dram",
                "kv-offload-backend": {"name": "native"},
                "conc-list": [16],
            }],
        })
        assert config.dram_utilization == 0.80

    def test_gpus_per_node_is_not_a_master_config_field(self):
        with pytest.raises(Exception, match="gpus-per-node"):
            AgenticCodingConfig(**{
                "dram-utilization": 0.80,
                "gpus-per-node": 8,
                "search-space": [{
                    "tp": 4,
                    "kv-offloading": "dram",
                    "kv-offload-backend": {"name": "native"},
                    "conc-list": [16],
                }],
            })

    def test_available_cpu_dram_is_not_a_master_config_field(self):
        with pytest.raises(Exception, match="available-cpu-dram-mib"):
            AgenticCodingConfig(**{
                "available-cpu-dram-mib": 2964436,
                "dram-utilization": 0.80,
                "search-space": [{
                    "tp": 4,
                    "kv-offloading": "dram",
                    "kv-offload-backend": {"name": "native"},
                    "conc-list": [16],
                }],
            })

    def test_duration_is_an_optional_master_config_field(self):
        # Per-scenario duration override (falls back to the default 3600s in the
        # sweep generator when unset). Used for shorter agentic smoke sweeps.
        cfg = AgenticCodingConfig(**{
            "duration": 1800,
            "search-space": [{
                "tp": 8,
                "kv-offloading": "none",
                "conc-list": [16],
            }],
        })
        assert cfg.duration == 1800

    def test_duration_defaults_to_none_when_unset(self):
        cfg = AgenticCodingConfig(**{
            "search-space": [{
                "tp": 8,
                "kv-offloading": "none",
                "conc-list": [16],
            }],
        })
        assert cfg.duration is None

    def test_duration_must_be_positive(self):
        with pytest.raises(Exception, match="duration"):
            AgenticCodingConfig(**{
                "duration": 0,
                "search-space": [{
                    "tp": 8,
                    "kv-offloading": "none",
                    "conc-list": [16],
                }],
            })


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
        assert entry.prefill.hardware == "gb200"
        assert entry.decode.hardware == "h100"

    def test_disagg_allows_omitted_hardware(self, valid_multinode_matrix_entry):
        """Homogeneous disaggregated entries may omit hardware metadata."""
        del valid_multinode_matrix_entry["prefill"]["hardware"]
        del valid_multinode_matrix_entry["decode"]["hardware"]
        entry = MultiNodeMatrixEntry(**valid_multinode_matrix_entry)
        assert entry.prefill.hardware is None
        assert entry.decode.hardware is None

    @pytest.mark.parametrize("missing_worker", ["prefill", "decode"])
    def test_hardware_requires_prefill_and_decode(
        self, valid_multinode_matrix_entry, missing_worker
    ):
        """Heterogeneous hardware metadata must identify both worker pools."""
        del valid_multinode_matrix_entry[missing_worker]["hardware"]
        with pytest.raises(Exception, match="both.*prefill.*decode"):
            MultiNodeMatrixEntry(**valid_multinode_matrix_entry)

    def test_prefill_decode_worker_configs(self, valid_multinode_matrix_entry):
        """Prefill and decode should be WorkerConfig objects."""
        entry = MultiNodeMatrixEntry(**valid_multinode_matrix_entry)
        assert entry.prefill.num_worker == 5
        assert entry.prefill.tp == 4
        assert entry.decode.tp == 8
        assert entry.decode.dp_attn is True

    def test_all_eval_concurrency_batch_marker(
        self,
        valid_multinode_matrix_entry,
    ):
        valid_multinode_matrix_entry["eval-all-concs"] = True

        entry = MultiNodeMatrixEntry(**valid_multinode_matrix_entry)

        assert entry.eval_all_concs is True

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

    def test_pp_defaults_to_one(self):
        entry = SingleNodeSearchSpaceEntry(**{
            "tp": 4,
            "conc-list": [4],
        })
        assert entry.pp == 1

    def test_pp_must_be_positive_integer(self):
        with pytest.raises(Exception, match="greater than 0"):
            SingleNodeSearchSpaceEntry(**{
                "tp": 4,
                "pp": 0,
                "conc-list": [4],
            })

    def test_dcp_size_must_divide_tp(self):
        with pytest.raises(Exception, match="must be divisible"):
            SingleNodeSearchSpaceEntry(**{
                "tp": 8,
                "dcp-size": 3,
                "pcp-size": 2,
                "conc-list": [4],
            })

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

    @pytest.mark.parametrize(
        ("conc_start", "conc_end"),
        [(0, 4), (-1, 4), (1, 0)],
    )
    def test_conc_range_values_must_be_positive(self, conc_start, conc_end):
        with pytest.raises(Exception) as exc_info:
            SingleNodeSearchSpaceEntry(**{
                "tp": 4,
                "conc-start": conc_start,
                "conc-end": conc_end,
            })

        assert "must be greater than 0" in str(exc_info.value)

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
        search_entry = config.scenarios.fixed_seq_len[0].search_space[0]
        assert search_entry.prefill.hardware == "gb200"
        assert search_entry.decode.hardware == "h100"

    def test_disagg_master_config_allows_omitted_hardware(self, valid_multinode_master_config):
        """Homogeneous disaggregated master configs may omit hardware metadata."""
        search_entry = valid_multinode_master_config["scenarios"]["fixed-seq-len"][0]["search-space"][0]
        del search_entry["prefill"]["hardware"]
        del search_entry["decode"]["hardware"]
        config = MultiNodeMasterConfigEntry(**valid_multinode_master_config)
        validated_entry = config.scenarios.fixed_seq_len[0].search_space[0]
        assert validated_entry.prefill.hardware is None
        assert validated_entry.decode.hardware is None

    def test_master_hardware_requires_prefill_and_decode(self, valid_multinode_master_config):
        """Heterogeneous master configs must identify both worker pools."""
        search_entry = valid_multinode_master_config["scenarios"]["fixed-seq-len"][0]["search-space"][0]
        del search_entry["decode"]["hardware"]
        with pytest.raises(Exception, match="both.*prefill.*decode"):
            MultiNodeMasterConfigEntry(**valid_multinode_master_config)

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

    def test_single_node_rejects_kv_p2p_transfer(
        self,
        valid_single_node_master_config,
    ):
        """P2P KV transfer is reserved for multinode configurations."""
        valid_single_node_master_config["kv-p2p-transfer"] = "nixl"

        with pytest.raises(Exception, match="kv-p2p-transfer"):
            SingleNodeMasterConfigEntry(**valid_single_node_master_config)

    def test_aggregated_multinode_allows_kv_p2p_transfer(
        self,
        valid_multinode_master_config,
    ):
        """P2P transfer is not restricted to disaggregated multinode serving."""
        valid_multinode_master_config["disagg"] = False

        config = MultiNodeMasterConfigEntry(**valid_multinode_master_config)

        assert config.kv_p2p_transfer == "nixl"

    def test_component_metadata_rejects_image_as_version(self):
        """Component versions identify the component, not its container."""
        with pytest.raises(Exception, match="not an image reference"):
            ComponentMetadata(
                name="nixl",
                version="image:vllm/vllm-openai:v0.23.0",
            )

    @pytest.mark.parametrize(("field", "value"), [
        ("router", {"name": "component", "version": "1.0.0"}),
        ("kv-p2p-transfer", "nixl"),
    ])
    def test_component_metadata_rejects_mixed_scopes(
        self,
        valid_multinode_master_config,
        field,
        value,
    ):
        """One metadata field cannot be declared at both supported scopes."""
        valid_multinode_master_config[field] = value
        search_space = valid_multinode_master_config[
            "scenarios"
        ]["fixed-seq-len"][0]["search-space"]
        search_space[0][field] = value

        with pytest.raises(Exception, match=f"{field} must be declared either"):
            MultiNodeMasterConfigEntry(**valid_multinode_master_config)

    def test_component_metadata_allows_different_field_scopes(
        self,
        valid_multinode_master_config,
    ):
        """Router and KV transfer may independently choose their scope."""
        valid_multinode_master_config.pop("kv-p2p-transfer")
        valid_multinode_master_config["router"] = {
            "name": "dynamo-router",
            "version": "1.0.0",
        }
        search_space = valid_multinode_master_config[
            "scenarios"
        ]["fixed-seq-len"][0]["search-space"]
        search_space[0]["kv-p2p-transfer"] = "nixl"

        config = MultiNodeMasterConfigEntry(**valid_multinode_master_config)

        assert config.router.name == "dynamo-router"
        kv_p2p_transfer = config.scenarios.fixed_seq_len[0].search_space[0].kv_p2p_transfer
        assert kv_p2p_transfer == "nixl"

    def test_component_metadata_allows_different_search_space_values(
        self,
        valid_multinode_master_config,
    ):
        """Different search-space entries may use different components."""
        valid_multinode_master_config.pop("kv-p2p-transfer")
        search_space = valid_multinode_master_config[
            "scenarios"
        ]["fixed-seq-len"][0]["search-space"]
        search_space.append(copy.deepcopy(search_space[0]))
        search_space[0]["kv-p2p-transfer"] = "nixl"
        search_space[1]["kv-p2p-transfer"] = "mooncake"

        config = MultiNodeMasterConfigEntry(**valid_multinode_master_config)

        values = config.scenarios.fixed_seq_len[0].search_space
        assert values[0].kv_p2p_transfer == "nixl"
        assert values[1].kv_p2p_transfer == "mooncake"

    def test_disagg_requires_kv_p2p_transfer(self, valid_multinode_master_config):
        """A disaggregated config cannot omit KV transfer metadata."""
        valid_multinode_master_config.pop("kv-p2p-transfer")

        with pytest.raises(Exception, match="disagg=true requires kv-p2p-transfer"):
            MultiNodeMasterConfigEntry(**valid_multinode_master_config)

    def test_disagg_accepts_kv_p2p_transfer_on_every_search_space_entry(
        self,
        valid_multinode_master_config,
    ):
        """Per-entry KV transfer metadata is valid when every entry has it."""
        valid_multinode_master_config.pop("kv-p2p-transfer")
        search_space = valid_multinode_master_config[
            "scenarios"
        ]["fixed-seq-len"][0]["search-space"]
        search_space.append(copy.deepcopy(search_space[0]))
        for entry in search_space:
            entry["kv-p2p-transfer"] = "nixl"

        config = MultiNodeMasterConfigEntry(**valid_multinode_master_config)

        assert all(
            entry.kv_p2p_transfer == "nixl"
            for entry in config.scenarios.fixed_seq_len[0].search_space
        )

    def test_disagg_rejects_partial_search_space_kv_p2p_transfer(
        self,
        valid_multinode_master_config,
    ):
        """Per-entry KV transfer metadata cannot leave any entry unspecified."""
        valid_multinode_master_config.pop("kv-p2p-transfer")
        search_space = valid_multinode_master_config[
            "scenarios"
        ]["fixed-seq-len"][0]["search-space"]
        search_space.append(copy.deepcopy(search_space[0]))
        search_space[0]["kv-p2p-transfer"] = "nixl"

        with pytest.raises(Exception, match="disagg=true requires kv-p2p-transfer"):
            MultiNodeMasterConfigEntry(**valid_multinode_master_config)

    def test_disagg_requires_multinode(self, valid_single_node_master_config):
        """Single-node master configs cannot enable disaggregation."""
        valid_single_node_master_config["disagg"] = True
        with pytest.raises(Exception, match="disagg"):
            SingleNodeMasterConfigEntry(**valid_single_node_master_config)

    def test_single_node_agentic_master_config_requires_cluster_runner(self):
        """Single-node agentic configs must pin an exact cluster label."""
        config = {
            "image": "vllm/vllm-openai:test",
            "model": "deepseek-ai/DeepSeek-V4-Pro",
            "model-prefix": "dsv4",
            "precision": "fp4",
            "framework": "vllm",
            "runner": "b200",
            "multinode": False,
            "scenarios": {
                "agentic-coding": [
                    {
                        "search-space": [
                            {"tp": 8, "conc-list": [1], "kv-offloading": "none"}
                        ],
                    }
                ]
            },
        }

        with pytest.raises(Exception, match="Agentic master configs must use"):
            SingleNodeMasterConfigEntry(**config)

        config["runner"] = "cluster:b200-dgxc"
        assert SingleNodeMasterConfigEntry(**config).runner == "cluster:b200-dgxc"

    def test_multinode_agentic_master_config_requires_cluster_runner(self):
        """Multinode agentic configs must also pin an exact cluster label."""
        config = {
            "image": "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:test",
            "model": "deepseek-r1-fp4",
            "model-prefix": "dsr1",
            "precision": "fp4",
            "framework": "dynamo-trt",
            "runner": "b200-multinode",
            "multinode": True,
            "disagg": True,
            "kv-p2p-transfer": "nixl",
            "scenarios": {
                "agentic-coding": [
                    {
                        "search-space": [
                            {
                                "spec-decoding": "none",
                                "conc-list": [1],
                                "prefill": {
                                    "hardware": "b200",
                                    "num-worker": 1,
                                    "tp": 4,
                                    "ep": 4,
                                    "dp-attn": True,
                                },
                                "decode": {
                                    "hardware": "b200",
                                    "num-worker": 1,
                                    "tp": 8,
                                    "ep": 8,
                                    "dp-attn": False,
                                },
                            }
                        ],
                    }
                ]
            },
        }

        with pytest.raises(Exception, match="Agentic master configs must use"):
            MultiNodeMasterConfigEntry(**config)

        config["runner"] = "cluster:b200-dgxc"
        assert MultiNodeMasterConfigEntry(**config).runner == "cluster:b200-dgxc"


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
            "labels": {
                "h100": "h100-cr_0",  # Not a list
            },
        }
        with pytest.raises(ValueError) as exc_info:
            validate_runner_config(config)
        assert "must be a list" in str(exc_info.value)

    def test_list_must_contain_strings(self):
        """Runner config lists must contain only strings."""
        config = {
            "labels": {
                "h100": ["h100-cr_0", 123],  # Contains non-string
            },
        }
        with pytest.raises(ValueError) as exc_info:
            validate_runner_config(config)
        assert "must contain only strings" in str(exc_info.value)

    def test_list_cannot_be_empty(self):
        """Runner config lists cannot be empty."""
        config = {
            "labels": {
                "mi355x": [],
            },
        }
        with pytest.raises(ValueError) as exc_info:
            validate_runner_config(config)
        assert "cannot be an empty list" in str(exc_info.value)

    def test_multiple_runner_types(self, valid_runner_config):
        """Multiple runner types should work."""
        result = validate_runner_config(valid_runner_config)
        assert "h100" in result["labels"]
        assert "h200" in result["labels"]
        assert "mi300x" in result["labels"]
        assert "gb200" in result["labels"]

    def test_flat_runner_config_is_rejected(self):
        config = {
            "h100": ["h100-cr_0", "h100-cw_0"],
        }
        with pytest.raises(ValueError, match="labels mapping"):
            validate_runner_config(config)

    def test_hardware_available_dram_must_be_positive(self):
        config = {
            "labels": {"h100": ["h100-cr_0"]},
            "hardware": {"h100": {"available-cpu-dram-mib": 0, "gpus-per-node": 8}},
        }
        with pytest.raises(ValueError) as exc_info:
            validate_runner_config(config)
        assert "available-cpu-dram-mib" in str(exc_info.value)

    def test_hardware_gpus_per_node_must_be_positive(self):
        config = {
            "labels": {"h100": ["h100-cr_0"]},
            "hardware": {"h100": {"available-cpu-dram-mib": 2063837, "gpus-per-node": 0}},
        }
        with pytest.raises(ValueError) as exc_info:
            validate_runner_config(config)
        assert "gpus-per-node" in str(exc_info.value)


# =============================================================================
# Test changelog entry validation
# =============================================================================

class TestChangelogEntry:
    """Tests for changelog eval mode validation."""

    def test_all_evals_is_supported(self):
        entry = ChangelogEntry.model_validate({
            "config-keys": ["test-config"],
            "description": ["Run every eval config"],
            "pr-link": "https://github.com/SemiAnalysisAI/InferenceX/pull/1",
            "all-evals": True,
        })

        assert entry.all_evals is True
        assert entry.evals_only is False

    def test_all_evals_can_extend_evals_only(self):
        entry = ChangelogEntry.model_validate({
            "config-keys": ["test-config"],
            "description": ["Run the expanded eval-only matrix"],
            "pr-link": "https://github.com/SemiAnalysisAI/InferenceX/pull/1",
            "evals-only": True,
            "all-evals": True,
        })

        assert entry.evals_only is True
        assert entry.all_evals is True

    @pytest.mark.parametrize("scenario_type", [[], ["unsupported"]])
    def test_scenario_type_must_be_nonempty_and_supported(self, scenario_type):
        with pytest.raises(ValueError):
            ChangelogEntry.model_validate({
                "config-keys": ["test-config"],
                "description": ["Invalid scenario filter"],
                "pr-link": "https://github.com/SemiAnalysisAI/InferenceX/pull/1",
                "scenario-type": scenario_type,
            })


# =============================================================================
# Test ChangelogMatrixEntry
# =============================================================================

AGENTIC_EVAL_ROW = {
    "image": "vllm/vllm-openai:nightly", "model": "deepseek-ai/DeepSeek-V4-Pro",
    "model-prefix": "dsv4", "precision": "fp4", "framework": "vllm",
    "runner": "cluster:b300-nv", "tp": 8, "pp": 1, "dcp-size": 1,
    "pcp-size": 1, "ep": 8, "dp-attn": True, "spec-decoding": "mtp",
    "conc": 224, "kv-offloading": "none", "total-cpu-dram-gb": 0,
    "duration": 3600, "exp-name": "dsv4_tp8_conc224_kvnone_spec-mtp",
    "scenario-type": "agentic-coding", "run-eval": True, "eval-only": True,
}

CHANGELOG_METADATA = {
    "base_ref": "base", "head_ref": "head",
    "entries": [{
        "config-keys": ["test-config"], "description": ["Test entry"],
        "pr-link": "https://github.com/SemiAnalysisAI/InferenceX/pull/1",
    }],
}


class TestChangelogMatrixEntry:
    """Tests for the final search-space contract consumed by run-sweep.yml."""

    def test_agentic_eval_rows_live_in_agentic_evals_only(self):
        """`evals` is dispatched with fixed-seq-len inputs (isl/osl/
        max-model-len), so agentic rows must only validate in agentic_evals."""
        entry = ChangelogMatrixEntry.model_validate({
            "agentic_evals": [AGENTIC_EVAL_ROW],
            "changelog_metadata": CHANGELOG_METADATA,
        })
        assert entry.agentic_evals[0].run_eval is True
        assert entry.agentic_evals[0].eval_only is True

        with pytest.raises(ValueError):
            ChangelogMatrixEntry.model_validate({
                "evals": [AGENTIC_EVAL_ROW],
                "changelog_metadata": CHANGELOG_METADATA,
            })


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


# =============================================================================
# Test load_runner_file
# =============================================================================

class TestLoadRunnerFile:
    """Tests for load_runner_file function."""

    def test_load_runner_file_with_validation(self, tmp_path):
        """Should load and validate runner config file."""
        runner_file = tmp_path / "runners.yaml"
        runner_file.write_text("""
labels:
  h100:
  - h100-node-0
  - h100-node-1
hardware:
  h100:
    available-cpu-dram-mib: 2063837
    gpus-per-node: 8
""")
        result = load_runner_file(str(runner_file))
        assert "h100" in result["labels"]
        assert len(result["labels"]["h100"]) == 2

    def test_load_runner_file_without_validation(self, tmp_path):
        """Should load runner config file without validation when validate=False."""
        runner_file = tmp_path / "runners.yaml"
        runner_file.write_text("""
labels:
  h100:
  - h100-node-0
  - h100-node-1
""")
        result = load_runner_file(str(runner_file), validate=False)
        assert "h100" in result["labels"]
        assert len(result["labels"]["h100"]) == 2

    def test_nonexistent_runner_file(self):
        """Nonexistent runner file should raise error."""
        with pytest.raises(ValueError) as exc_info:
            load_runner_file("nonexistent.yaml")
        assert "does not exist" in str(exc_info.value)

    def test_validation_runs_by_default(self, tmp_path):
        """Validation should run by default and catch invalid configs."""
        runner_file = tmp_path / "runners.yaml"
        runner_file.write_text("""
labels:
  h100: not-a-list
""")
        with pytest.raises(ValueError) as exc_info:
            load_runner_file(str(runner_file))
        assert "must be a list" in str(exc_info.value)
