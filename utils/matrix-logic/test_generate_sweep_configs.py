import pytest
import yaml
from unittest.mock import patch
from generate_sweep_configs import (
    validate_master_configs_structure,
    validate_matrix_output,
    seq_len_to_str,
    generate_full_sweep,
    generate_test_config,
    generate_runner_model_sweep_config,
    generate_runner_sweep_config,
    generate_custom_test,
    load_config_files,
    main,
    MatrixEntry,
)


# Fixtures for test config files
@pytest.fixture
def sample_master_config():
    """Sample master config with valid entries."""
    return {
        "70b-fp8-vllm": {
            "image": "vllm/vllm-openai:v0.10.2",
            "model": "meta-llama/Llama-3-70b",
            "model-prefix": "70b",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "seq-len-configs": [
                {
                    "isl": 1024,
                    "osl": 1024,
                    "search-space": [
                        {"tp": 4, "conc-start": 1, "conc-end": 4},
                        {"tp": 8, "conc-start": 2, "conc-end": 8, "ep": 2, "dp-attn": True}
                    ]
                },
                {
                    "isl": 1024,
                    "osl": 8192,
                    "search-space": [
                        {"tp": 8, "conc-start": 1, "conc-end": 2}
                    ]
                }
            ]
        },
        "8b-fp4-trt": {
            "image": "nvcr.io/nvidia/tritonserver:24.01",
            "model": "meta-llama/Llama-3-8b",
            "model-prefix": "8b",
            "precision": "fp4",
            "framework": "trt",
            "runner": "h100",
            "seq-len-configs": [
                {
                    "isl": 1024,
                    "osl": 1024,
                    "search-space": [
                        {"tp": 2, "conc-start": 4, "conc-end": 16}
                    ]
                }
            ]
        },
        "gptoss-120b-fp8-vllm": {
            "image": "vllm/vllm-openai:latest",
            "model": "openai/gpt-oss-120b",
            "model-prefix": "gptoss",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200-trt",
            "seq-len-configs": [
                {
                    "isl": 1024,
                    "osl": 1024,
                    "search-space": [
                        {"tp": 8, "conc-start": 1, "conc-end": 4}
                    ]
                }
            ]
        }
    }


@pytest.fixture
def sample_runner_config():
    """Sample runner config."""
    return {
        "h200": ["h200-nv_1", "h200-nv_2"],
        "h100": ["h100-aws_1"],
        "h200-trt": ["h200-trt_1", "h200-trt_2", "h200-trt_3"]
    }


@pytest.fixture
def temp_config_files(tmp_path, sample_master_config, sample_runner_config):
    """Create temporary config files."""
    master_file = tmp_path / "master.yaml"
    runner_file = tmp_path / "runners.yaml"

    with open(master_file, 'w') as f:
        yaml.dump(sample_master_config, f)

    with open(runner_file, 'w') as f:
        yaml.dump(sample_runner_config, f)

    return str(master_file), str(runner_file)


@pytest.fixture
def invalid_master_config():
    """Master config with validation errors."""
    return {
        "missing-field": {
            "image": "test:latest",
            "model": "test/model",
            "model-prefix": "test",
            # Missing precision, framework, runner, seq-len-configs
        }
    }


# Tests for seq_len_to_str
def test_seq_len_to_str_with_mapping():
    """Test seq_len_to_str with known mappings."""
    assert seq_len_to_str(1024, 1024) == "1k1k"
    assert seq_len_to_str(1024, 8192) == "1k8k"
    assert seq_len_to_str(8192, 1024) == "8k1k"


def test_seq_len_to_str_without_mapping():
    """Test seq_len_to_str fallback for unknown mappings."""
    assert seq_len_to_str(2048, 4096) == "2048_4096"
    assert seq_len_to_str(512, 512) == "512_512"


# Tests for MatrixEntry validation
def test_matrix_entry_valid():
    """Test valid MatrixEntry."""
    entry = {
        "image": "test:latest",
        "model": "test/model",
        "precision": "fp8",
        "framework": "vllm",
        "runner": "h200",
        "isl": 1024,
        "osl": 1024,
        "tp": 8,
        "ep": 1,
        "dp-attn": False,
        "conc": 4,
        "max-model-len": 2048,
        "exp-name": "test_exp"
    }
    result = MatrixEntry(**entry)
    assert result.image == "test:latest"
    assert result.tp == 8


def test_matrix_entry_missing_field():
    """Test MatrixEntry with missing required field."""
    entry = {
        "image": "test:latest",
        "model": "test/model",
        # Missing other required fields
    }
    with pytest.raises(Exception):  # Pydantic ValidationError
        MatrixEntry(**entry)


def test_matrix_entry_wrong_type():
    """Test MatrixEntry with wrong type."""
    entry = {
        "image": "test:latest",
        "model": "test/model",
        "precision": "fp8",
        "framework": "vllm",
        "runner": "h200",
        "isl": "not_an_int",  # Wrong type
        "osl": 1024,
        "tp": 8,
        "ep": 1,
        "dp-attn": False,
        "conc": 4,
        "max-model-len": 2048,
        "exp-name": "test_exp"
    }
    with pytest.raises(Exception):  # Pydantic ValidationError
        MatrixEntry(**entry)


def test_matrix_entry_extra_field():
    """Test MatrixEntry with extra field (should be forbidden)."""
    entry = {
        "image": "test:latest",
        "model": "test/model",
        "precision": "fp8",
        "framework": "vllm",
        "runner": "h200",
        "isl": 1024,
        "osl": 1024,
        "tp": 8,
        "ep": 1,
        "dp-attn": False,
        "conc": 4,
        "max-model-len": 2048,
        "exp-name": "test_exp",
        "extra-field": "should_fail"
    }
    with pytest.raises(Exception):  # Pydantic ValidationError
        MatrixEntry(**entry)


# Tests for validate_matrix_output
def test_validate_matrix_output_valid():
    """Test validate_matrix_output with valid entries."""
    entries = [
        {
            "image": "test:latest",
            "model": "test/model",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "isl": 1024,
            "osl": 1024,
            "tp": 8,
            "ep": 1,
            "dp-attn": False,
            "conc": 4,
            "max-model-len": 2048,
            "exp-name": "test_exp"
        }
    ]
    result = validate_matrix_output(entries)
    assert result == entries


def test_validate_matrix_output_invalid():
    """Test validate_matrix_output with invalid entry."""
    entries = [
        {
            "image": "test:latest",
            "model": "test/model",
            # Missing required fields
        }
    ]
    with pytest.raises(ValueError, match="Matrix entry at index 0 failed validation"):
        validate_matrix_output(entries)


def test_validate_matrix_output_multiple_entries():
    """Test validate_matrix_output with multiple entries."""
    entries = [
        {
            "image": "test:latest",
            "model": "test/model",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "isl": 1024,
            "osl": 1024,
            "tp": 8,
            "ep": 1,
            "dp-attn": False,
            "conc": 4,
            "max-model-len": 2048,
            "exp-name": "test_exp"
        },
        {
            "image": "test2:latest",
            "model": "test2/model",
            "precision": "fp4",
            "framework": "trt",
            "runner": "h100",
            "isl": 1024,
            "osl": 1024,
            "tp": 4,
            "ep": 2,
            "dp-attn": True,
            "conc": 8,
            "max-model-len": 2048,
            "exp-name": "test_exp2"
        }
    ]
    result = validate_matrix_output(entries)
    assert len(result) == 2


# Tests for validate_master_configs_structure
def test_validate_master_configs_structure_valid(sample_master_config):
    """Test validation of valid master config."""
    validate_master_configs_structure(sample_master_config)


def test_validate_master_configs_structure_missing_field():
    """Test validation with missing required field."""
    config = {
        "test-key": {
            "image": "test:latest",
            "model-prefix": "test",
            # Missing other required fields
        }
    }
    with pytest.raises(ValueError, match="Missing required field"):
        validate_master_configs_structure(config)


def test_validate_master_configs_structure_wrong_type():
    """Test validation with wrong field type."""
    config = {
        "test-key": {
            "image": 123,  # Should be string
            "model": "test/model",
            "model-prefix": "test",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "seq-len-configs": []
        }
    }
    with pytest.raises(ValueError, match="must be str"):
        validate_master_configs_structure(config)


def test_validate_master_configs_structure_empty_seq_len_configs():
    """Test validation with empty seq-len-configs."""
    config = {
        "test-key": {
            "image": "test:latest",
            "model": "test/model",
            "model-prefix": "test",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "seq-len-configs": []
        }
    }
    with pytest.raises(ValueError, match="must be a non-empty list"):
        validate_master_configs_structure(config)


def test_validate_master_configs_structure_invalid_search_space():
    """Test validation with invalid search-space."""
    config = {
        "test-key": {
            "image": "test:latest",
            "model": "test/model",
            "model-prefix": "test",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "seq-len-configs": [
                {
                    "isl": 1024,
                    "osl": 1024,
                    "search-space": [
                        {"tp": 8}  # Missing conc-start and conc-end
                    ]
                }
            ]
        }
    }
    with pytest.raises(ValueError, match="Missing 'conc-start'"):
        validate_master_configs_structure(config)


def test_validate_master_configs_structure_missing_search_space():
    """Test validation with missing search-space."""
    config = {
        "test-key": {
            "image": "test:latest",
            "model": "test/model",
            "model-prefix": "test",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "seq-len-configs": [
                {
                    "isl": 1024,
                    "osl": 1024
                    # Missing search-space
                }
            ]
        }
    }
    with pytest.raises(ValueError, match="Missing or invalid 'search-space'"):
        validate_master_configs_structure(config)


def test_validate_master_configs_structure_search_space_not_list():
    """Test validation with search-space not being a list."""
    config = {
        "test-key": {
            "image": "test:latest",
            "model": "test/model",
            "model-prefix": "test",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "seq-len-configs": [
                {
                    "isl": 1024,
                    "osl": 1024,
                    "search-space": "not_a_list"
                }
            ]
        }
    }
    with pytest.raises(ValueError, match="Missing or invalid 'search-space'"):
        validate_master_configs_structure(config)


def test_validate_master_configs_structure_extra_fields_in_search_space():
    """Test validation with extra fields in search-space."""
    config = {
        "test-key": {
            "image": "test:latest",
            "model": "test/model",
            "model-prefix": "test",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "seq-len-configs": [
                {
                    "isl": 1024,
                    "osl": 1024,
                    "search-space": [
                        {
                            "tp": 8,
                            "conc-start": 1,
                            "conc-end": 4,
                            "invalid-field": "value"
                        }
                    ]
                }
            ]
        }
    }
    with pytest.raises(ValueError, match="Extra fields"):
        validate_master_configs_structure(config)


def test_validate_master_configs_structure_missing_isl():
    """Test validation with missing isl."""
    config = {
        "test-key": {
            "image": "test:latest",
            "model": "test/model",
            "model-prefix": "test",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "seq-len-configs": [
                {
                    "osl": 1024,
                    "search-space": [{"tp": 8, "conc-start": 1, "conc-end": 4}]
                }
            ]
        }
    }
    with pytest.raises(ValueError, match="Missing 'isl'"):
        validate_master_configs_structure(config)


def test_validate_master_configs_structure_wrong_isl_type():
    """Test validation with wrong isl type."""
    config = {
        "test-key": {
            "image": "test:latest",
            "model": "test/model",
            "model-prefix": "test",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "seq-len-configs": [
                {
                    "isl": "not_int",
                    "osl": 1024,
                    "search-space": [{"tp": 8, "conc-start": 1, "conc-end": 4}]
                }
            ]
        }
    }
    with pytest.raises(ValueError, match="'isl' must be int"):
        validate_master_configs_structure(config)


def test_validate_master_configs_structure_missing_osl():
    """Test validation with missing osl."""
    config = {
        "test-key": {
            "image": "test:latest",
            "model": "test/model",
            "model-prefix": "test",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "seq-len-configs": [
                {
                    "isl": 1024,
                    "search-space": [{"tp": 8, "conc-start": 1, "conc-end": 4}]
                }
            ]
        }
    }
    with pytest.raises(ValueError, match="Missing 'osl'"):
        validate_master_configs_structure(config)


def test_validate_master_configs_structure_wrong_osl_type():
    """Test validation with wrong osl type."""
    config = {
        "test-key": {
            "image": "test:latest",
            "model": "test/model",
            "model-prefix": "test",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "seq-len-configs": [
                {
                    "isl": 1024,
                    "osl": "not_int",
                    "search-space": [{"tp": 8, "conc-start": 1, "conc-end": 4}]
                }
            ]
        }
    }
    with pytest.raises(ValueError, match="'osl' must be int"):
        validate_master_configs_structure(config)


def test_validate_master_configs_structure_wrong_tp_type():
    """Test validation with wrong tp type."""
    config = {
        "test-key": {
            "image": "test:latest",
            "model": "test/model",
            "model-prefix": "test",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "seq-len-configs": [
                {
                    "isl": 1024,
                    "osl": 1024,
                    "search-space": [{"tp": "not_int", "conc-start": 1, "conc-end": 4}]
                }
            ]
        }
    }
    with pytest.raises(ValueError, match="'tp' must be int"):
        validate_master_configs_structure(config)


def test_validate_master_configs_structure_wrong_conc_start_type():
    """Test validation with wrong conc-start type."""
    config = {
        "test-key": {
            "image": "test:latest",
            "model": "test/model",
            "model-prefix": "test",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "seq-len-configs": [
                {
                    "isl": 1024,
                    "osl": 1024,
                    "search-space": [{"tp": 8, "conc-start": "not_int", "conc-end": 4}]
                }
            ]
        }
    }
    with pytest.raises(ValueError, match="'conc-start' must be int"):
        validate_master_configs_structure(config)


def test_validate_master_configs_structure_wrong_conc_end_type():
    """Test validation with wrong conc-end type."""
    config = {
        "test-key": {
            "image": "test:latest",
            "model": "test/model",
            "model-prefix": "test",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "seq-len-configs": [
                {
                    "isl": 1024,
                    "osl": 1024,
                    "search-space": [{"tp": 8, "conc-start": 1, "conc-end": "not_int"}]
                }
            ]
        }
    }
    with pytest.raises(ValueError, match="'conc-end' must be int"):
        validate_master_configs_structure(config)


def test_validate_master_configs_structure_wrong_ep_type():
    """Test validation with wrong ep type."""
    config = {
        "test-key": {
            "image": "test:latest",
            "model": "test/model",
            "model-prefix": "test",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "seq-len-configs": [
                {
                    "isl": 1024,
                    "osl": 1024,
                    "search-space": [{"tp": 8, "conc-start": 1, "conc-end": 4, "ep": "not_int"}]
                }
            ]
        }
    }
    with pytest.raises(ValueError, match="'ep' must be int"):
        validate_master_configs_structure(config)


def test_validate_master_configs_structure_wrong_dp_attn_type():
    """Test validation with wrong dp-attn type."""
    config = {
        "test-key": {
            "image": "test:latest",
            "model": "test/model",
            "model-prefix": "test",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "seq-len-configs": [
                {
                    "isl": 1024,
                    "osl": 1024,
                    "search-space": [{"tp": 8, "conc-start": 1, "conc-end": 4, "dp-attn": "not_bool"}]
                }
            ]
        }
    }
    with pytest.raises(ValueError, match="'dp-attn' must be bool"):
        validate_master_configs_structure(config)


# Tests for load_config_files
def test_load_config_files_valid(temp_config_files):
    """Test loading valid config files."""
    master_file, _ = temp_config_files
    result = load_config_files([master_file])
    assert len(result) == 3
    assert "70b-fp8-vllm" in result


def test_load_config_files_multiple(tmp_path, sample_master_config):
    """Test loading multiple config files."""
    file1 = tmp_path / "config1.yaml"
    file2 = tmp_path / "config2.yaml"

    config1 = {"70b-fp8-vllm": sample_master_config["70b-fp8-vllm"]}
    config2 = {"8b-fp4-trt": sample_master_config["8b-fp4-trt"]}

    with open(file1, 'w') as f:
        yaml.dump(config1, f)
    with open(file2, 'w') as f:
        yaml.dump(config2, f)

    result = load_config_files([str(file1), str(file2)])
    assert len(result) == 2


def test_load_config_files_not_found():
    """Test loading non-existent config file."""
    with pytest.raises(ValueError, match="does not exist"):
        load_config_files(["/nonexistent/file.yaml"])


def test_load_config_files_duplicate_keys(tmp_path, sample_master_config):
    """Test loading files with duplicate keys."""
    file1 = tmp_path / "config1.yaml"
    file2 = tmp_path / "config2.yaml"

    config1 = {"70b-fp8-vllm": sample_master_config["70b-fp8-vllm"]}
    config2 = {"70b-fp8-vllm": sample_master_config["70b-fp8-vllm"]}  # Duplicate

    with open(file1, 'w') as f:
        yaml.dump(config1, f)
    with open(file2, 'w') as f:
        yaml.dump(config2, f)

    with pytest.raises(ValueError, match="Duplicate configuration keys"):
        load_config_files([str(file1), str(file2)])


# Tests for generate_full_sweep
def test_generate_full_sweep_basic(sample_master_config, temp_config_files):
    """Test basic full sweep generation."""
    _, runner_file = temp_config_files

    class Args:
        model_prefix = ["70b"]
        seq_lens = ["1k1k"]
        step_size = 2
        precision = None
        framework = None
        runner_type = None
        test_mode = False
        runner_config = runner_file

    result = generate_full_sweep(Args(), sample_master_config)
    assert len(result) > 0
    assert all(entry['exp-name'].startswith('70b_1k1k') for entry in result)
    assert all(entry['isl'] == 1024 and entry['osl'] == 1024 for entry in result)


def test_generate_full_sweep_with_optionals(sample_master_config, temp_config_files):
    """Test full sweep with optional ep and dp-attn."""
    _, runner_file = temp_config_files

    class Args:
        model_prefix = ["70b"]
        seq_lens = ["1k1k"]
        step_size = 2
        precision = None
        framework = None
        runner_type = None
        test_mode = False
        runner_config = runner_file

    result = generate_full_sweep(Args(), sample_master_config)
    # Find entry with tp=8 which should have ep=2 and dp-attn=True
    tp8_entries = [e for e in result if e['tp'] == 8]
    assert len(tp8_entries) > 0
    assert all(e['ep'] == 2 for e in tp8_entries)
    assert all(e['dp-attn'] == True for e in tp8_entries)


def test_generate_full_sweep_no_matches(sample_master_config, temp_config_files):
    """Test full sweep with no matching configs."""
    _, runner_file = temp_config_files

    class Args:
        model_prefix = ["nonexistent"]
        seq_lens = ["1k1k"]
        step_size = 2
        precision = None
        framework = None
        runner_type = None
        test_mode = False
        runner_config = runner_file

    with pytest.raises(ValueError, match="No configs found matching filters"):
        generate_full_sweep(Args(), sample_master_config)


def test_generate_full_sweep_different_seq_len(sample_master_config, temp_config_files):
    """Test full sweep with different sequence length."""
    _, runner_file = temp_config_files

    class Args:
        model_prefix = ["70b"]
        seq_lens = ["1k8k"]
        step_size = 2
        precision = None
        framework = None
        runner_type = None
        test_mode = False
        runner_config = runner_file

    result = generate_full_sweep(Args(), sample_master_config)
    assert len(result) > 0
    assert all(entry['isl'] == 1024 and entry['osl'] == 8192 for entry in result)


def test_generate_full_sweep_step_size(sample_master_config, temp_config_files):
    """Test full sweep with different step size."""
    _, runner_file = temp_config_files

    class Args:
        model_prefix = ["8b"]
        seq_lens = ["1k1k"]
        step_size = 4
        precision = None
        framework = None
        runner_type = None
        test_mode = False
        runner_config = runner_file

    result = generate_full_sweep(Args(), sample_master_config)
    # Should have entries at conc=4, 8, 16 (step_size=4, conc-start=4, conc-end=16)
    conc_values = sorted(set(e['conc'] for e in result))
    assert 4 in conc_values
    assert 16 in conc_values


def test_generate_full_sweep_seq_len_not_in_config(temp_config_files):
    """Test full sweep when requested seq-len is not in config."""
    _, runner_file = temp_config_files

    config = {
        "test-fp8-vllm": {
            "image": "test:latest",
            "model": "test/model",
            "model-prefix": "test",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "seq-len-configs": [
                {
                    "isl": 8192,
                    "osl": 1024,  # Only has 8k1k, not 1k1k
                    "search-space": [
                        {"tp": 4, "conc-start": 1, "conc-end": 4}
                    ]
                }
            ]
        }
    }

    class Args:
        model_prefix = ["test"]
        seq_lens = ["1k1k"]  # Requesting 1k1k but config only has 8k1k
        step_size = 2
        precision = None
        framework = None
        runner_type = None
        test_mode = False
        runner_config = runner_file

    # Should raise error since no matching seq-len
    with pytest.raises(ValueError, match="No configs found matching filters"):
        generate_full_sweep(Args(), config)


def test_generate_full_sweep_concurrency_overshoot(temp_config_files):
    """Test full sweep when concurrency step overshoots end value."""
    _, runner_file = temp_config_files

    config = {
        "test-fp8-vllm": {
            "image": "test:latest",
            "model": "test/model",
            "model-prefix": "test",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "seq-len-configs": [
                {
                    "isl": 1024,
                    "osl": 1024,
                    "search-space": [
                        {"tp": 4, "conc-start": 1, "conc-end": 5}  # 1, 3*2=6 overshoots, clamps to 5
                    ]
                }
            ]
        }
    }

    class Args:
        model_prefix = ["test"]
        seq_lens = ["1k1k"]
        step_size = 3  # Will overshoot: 1, 3, 9 (clamped to 5)
        precision = None
        framework = None
        runner_type = None
        test_mode = False
        runner_config = runner_file

    result = generate_full_sweep(Args(), config)
    conc_values = sorted(set(e['conc'] for e in result))
    # Should have 1, 3, 5 (5 is the clamped value)
    assert conc_values == [1, 3, 5]


# Tests for generate_full_sweep with filters
def test_generate_full_sweep_no_filters(sample_master_config, temp_config_files):
    """Test filtered sweep with no filters."""
    _, runner_file = temp_config_files

    class Args:
        model_prefix = None
        precision = None
        framework = None
        runner_type = None
        seq_lens = None
        step_size = 2
        test_mode = False
        runner_config = runner_file

    result = generate_full_sweep(Args(), sample_master_config)
    assert len(result) > 0


def test_generate_full_sweep_with_filters_model_prefix(sample_master_config, temp_config_files):
    """Test filtered sweep with model prefix filter."""
    _, runner_file = temp_config_files

    class Args:
        model_prefix = ["70b"]
        precision = None
        framework = None
        runner_type = None
        seq_lens = None
        step_size = 2
        test_mode = False
        runner_config = runner_file

    result = generate_full_sweep(Args(), sample_master_config)
    assert all("70b" in entry['exp-name'] for entry in result)


def test_generate_full_sweep_with_filters_multiple_filters(sample_master_config, temp_config_files):
    """Test filtered sweep with multiple filters."""
    _, runner_file = temp_config_files

    class Args:
        model_prefix = ["70b"]
        precision = ["fp8"]
        framework = ["vllm"]
        runner_type = None
        seq_lens = ["1k1k"]
        step_size = 2
        test_mode = False
        runner_config = runner_file

    result = generate_full_sweep(Args(), sample_master_config)
    assert len(result) > 0
    assert all(entry['precision'] == 'fp8' for entry in result)
    assert all(entry['framework'] == 'vllm' for entry in result)


def test_generate_full_sweep_with_filters_test_mode(sample_master_config, temp_config_files):
    """Test filtered sweep in test mode."""
    _, runner_file = temp_config_files

    class Args:
        model_prefix = ["70b"]
        precision = None
        framework = None
        runner_type = None
        seq_lens = ["1k1k"]
        step_size = 2
        test_mode = True
        runner_config = runner_file

    result = generate_full_sweep(Args(), sample_master_config)
    # In test mode, should only get one entry per seq-len (highest TP, lowest conc)
    assert len(result) == 1  # Only one config matches 70b with 1k1k
    assert result[0]['tp'] == 8  # Highest TP
    assert '70b_1k1k' in result[0]['exp-name']


def test_generate_full_sweep_with_filters_runner_type_validation(sample_master_config, temp_config_files):
    """Test filtered sweep with invalid runner type."""
    _, runner_file = temp_config_files

    class Args:
        model_prefix = None
        precision = None
        framework = None
        runner_type = ["invalid-runner"]
        seq_lens = None
        step_size = 2
        test_mode = False
        runner_config = runner_file

    with pytest.raises(ValueError, match="Invalid runner type"):
        generate_full_sweep(Args(), sample_master_config)


def test_generate_full_sweep_with_filters_runner_type_no_config(sample_master_config):
    """Test filtered sweep with runner type but no config file."""
    class Args:
        model_prefix = None
        precision = None
        framework = None
        runner_type = ["h200"]
        seq_lens = None
        step_size = 2
        test_mode = False
        runner_config = None

    with pytest.raises(ValueError, match="runner-config is required"):
        generate_full_sweep(Args(), sample_master_config)


def test_generate_full_sweep_with_filters_multiple_runner_types(sample_master_config, temp_config_files):
    """Test filtered sweep with multiple runner types."""
    _, runner_file = temp_config_files

    class Args:
        model_prefix = None
        precision = None
        framework = None
        runner_type = ["h200", "h100"]
        seq_lens = ["1k1k"]
        step_size = 2
        test_mode = False
        runner_config = runner_file

    result = generate_full_sweep(Args(), sample_master_config)
    runners = set(entry['runner'] for entry in result)
    assert 'h200' in runners or 'h100' in runners


def test_generate_full_sweep_with_filters_no_matches(sample_master_config, temp_config_files):
    """Test filtered sweep with no matching configs."""
    _, runner_file = temp_config_files

    class Args:
        model_prefix = ["nonexistent"]
        precision = None
        framework = None
        runner_type = None
        seq_lens = None
        step_size = 2
        test_mode = False
        runner_config = runner_file

    with pytest.raises(ValueError, match="No configs found matching filters"):
        generate_full_sweep(Args(), sample_master_config)


def test_generate_full_sweep_with_filters_concurrency_overshoot(temp_config_files):
    """Test filtered sweep when concurrency step overshoots end value."""
    _, runner_file = temp_config_files

    config = {
        "test-fp8-vllm": {
            "image": "test:latest",
            "model": "test/model",
            "model-prefix": "test",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "seq-len-configs": [
                {
                    "isl": 1024,
                    "osl": 1024,
                    "search-space": [
                        {"tp": 4, "conc-start": 2, "conc-end": 7}  # 2, 8 overshoots, clamps to 7
                    ]
                }
            ]
        }
    }

    class Args:
        model_prefix = None
        precision = None
        framework = None
        runner_type = None
        seq_lens = None
        step_size = 4  # Will overshoot: 2, 8 (clamped to 7)
        test_mode = False
        runner_config = runner_file

    result = generate_full_sweep(Args(), config)
    conc_values = sorted(set(e['conc'] for e in result))
    # Should have 2, 7 (7 is the clamped value)
    assert 2 in conc_values
    assert 7 in conc_values


# Tests for generate_test_config
def test_generate_test_config_basic(sample_master_config, temp_config_files):
    """Test basic test config generation."""
    _, runner_file = temp_config_files

    class Args:
        key = "70b-fp8-vllm"
        runner_config = runner_file
        runner_node = "h200-nv_1"
        seq_lens = None
        step_size = 2
        test_mode = False

    result = generate_test_config(Args(), sample_master_config)
    assert len(result) > 0


def test_generate_test_config_test_mode(sample_master_config, temp_config_files):
    """Test test config in test mode."""
    _, runner_file = temp_config_files

    class Args:
        key = "70b-fp8-vllm"
        runner_config = runner_file
        runner_node = "h200-nv_1"
        seq_lens = ["1k1k"]
        step_size = 2
        test_mode = True

    result = generate_test_config(Args(), sample_master_config)
    # In test mode, should only use lowest concurrency
    assert all(entry['conc'] == 1 or entry['conc'] == 2 for entry in result)


def test_generate_test_config_specific_runner_node(sample_master_config, temp_config_files):
    """Test test config with specific runner node."""
    _, runner_file = temp_config_files

    class Args:
        key = "70b-fp8-vllm"
        runner_config = runner_file
        runner_node = "h200-nv_1"
        seq_lens = None
        step_size = 2
        test_mode = False

    result = generate_test_config(Args(), sample_master_config)
    assert all(entry['runner'] == 'h200-nv_1' for entry in result)


def test_generate_test_config_invalid_key(sample_master_config, temp_config_files):
    """Test test config with invalid key."""
    _, runner_file = temp_config_files

    class Args:
        key = "nonexistent-key"
        runner_config = runner_file
        runner_node = None
        seq_lens = None
        step_size = 2
        test_mode = False

    with pytest.raises(ValueError, match="does not exist in config files"):
        generate_test_config(Args(), sample_master_config)


def test_generate_test_config_invalid_runner_node(sample_master_config, temp_config_files):
    """Test test config with invalid runner node."""
    _, runner_file = temp_config_files

    class Args:
        key = "70b-fp8-vllm"
        runner_config = runner_file
        runner_node = "invalid-node"
        seq_lens = None
        step_size = 2
        test_mode = False

    with pytest.raises(ValueError, match="is not compatible"):
        generate_test_config(Args(), sample_master_config)


def test_generate_test_config_missing_runner_config(sample_master_config):
    """Test test config with missing runner config file."""
    class Args:
        key = "70b-fp8-vllm"
        runner_config = "/nonexistent/file.yaml"
        runner_node = None
        seq_lens = None
        step_size = 2
        test_mode = False

    with pytest.raises(ValueError, match="does not exist"):
        generate_test_config(Args(), sample_master_config)


def test_generate_test_config_concurrency_overshoot(temp_config_files):
    """Test test config when concurrency step overshoots end value."""
    _, runner_file = temp_config_files

    config = {
        "test-fp8-vllm": {
            "image": "test:latest",
            "model": "test/model",
            "model-prefix": "test",
            "precision": "fp8",
            "framework": "vllm",
            "runner": "h200",
            "seq-len-configs": [
                {
                    "isl": 1024,
                    "osl": 1024,
                    "search-space": [
                        {"tp": 4, "conc-start": 1, "conc-end": 6}
                    ]
                }
            ]
        }
    }

    class Args:
        key = "test-fp8-vllm"
        runner_config = runner_file
        runner_node = "h200-nv_1"
        seq_lens = None
        step_size = 4  # Will overshoot: 1, 4, 16 (clamped to 6)
        test_mode = False

    result = generate_test_config(Args(), config)
    conc_values = sorted(set(e['conc'] for e in result))
    assert 1 in conc_values
    assert 4 in conc_values
    assert 6 in conc_values


# Tests for generate_runner_model_sweep_config
def test_generate_runner_model_sweep_config(sample_master_config, temp_config_files):
    """Test runner-model sweep config generation."""
    _, runner_file = temp_config_files

    class Args:
        runner_type = "h200"
        runner_config = runner_file
        runner_node_filter = None

    result = generate_runner_model_sweep_config(Args(), sample_master_config)
    assert len(result) > 0
    # Should have entries for each runner node under h200
    runners = set(entry['runner'] for entry in result)
    assert 'h200-nv_1' in runners
    assert 'h200-nv_2' in runners


def test_generate_runner_model_sweep_config_invalid_runner(sample_master_config, temp_config_files):
    """Test runner-model sweep with invalid runner type."""
    _, runner_file = temp_config_files

    class Args:
        runner_type = "invalid-runner"
        runner_config = runner_file
        runner_node_filter = None

    with pytest.raises(ValueError, match="does not exist in runner config"):
        generate_runner_model_sweep_config(Args(), sample_master_config)


def test_generate_runner_model_sweep_config_with_node_filter(sample_master_config, temp_config_files):
    """Test runner-model sweep with runner node filter."""
    _, runner_file = temp_config_files

    class Args:
        runner_type = "h200"
        runner_config = runner_file
        runner_node_filter = "nv_1"

    result = generate_runner_model_sweep_config(Args(), sample_master_config)
    # Should only have entries for h200-nv_1
    runners = set(entry['runner'] for entry in result)
    assert 'h200-nv_1' in runners
    assert 'h200-nv_2' not in runners


def test_generate_runner_model_sweep_config_with_node_filter_multiple_matches(sample_master_config, temp_config_files):
    """Test runner-model sweep with runner node filter matching multiple nodes."""
    _, runner_file = temp_config_files

    class Args:
        runner_type = "h200"
        runner_config = runner_file
        runner_node_filter = "nv"  # Should match both nv_1 and nv_2

    result = generate_runner_model_sweep_config(Args(), sample_master_config)
    runners = set(entry['runner'] for entry in result)
    assert 'h200-nv_1' in runners
    assert 'h200-nv_2' in runners


def test_generate_runner_model_sweep_config_with_node_filter_no_matches(sample_master_config, temp_config_files):
    """Test runner-model sweep with runner node filter that matches no nodes."""
    _, runner_file = temp_config_files

    class Args:
        runner_type = "h200"
        runner_config = runner_file
        runner_node_filter = "nonexistent"

    with pytest.raises(ValueError, match="No runner nodes found matching filter"):
        generate_runner_model_sweep_config(Args(), sample_master_config)


def test_generate_runner_model_sweep_config_without_node_filter(sample_master_config, temp_config_files):
    """Test runner-model sweep without runner node filter (default behavior)."""
    _, runner_file = temp_config_files

    class Args:
        runner_type = "h200"
        runner_config = runner_file
        runner_node_filter = None

    result = generate_runner_model_sweep_config(Args(), sample_master_config)
    # Should have entries for all h200 nodes
    runners = set(entry['runner'] for entry in result)
    assert 'h200-nv_1' in runners
    assert 'h200-nv_2' in runners


# Tests for generate_runner_sweep_config
def test_generate_runner_sweep_config(sample_master_config, temp_config_files):
    """Test runner sweep config generation."""
    _, runner_file = temp_config_files

    class Args:
        model_prefix = "70b"
        runner_type = "h200"
        precision = None
        framework = None
        runner_config = runner_file

    result = generate_runner_sweep_config(Args(), sample_master_config)
    assert len(result) > 0


def test_generate_runner_sweep_config_with_filters(sample_master_config, temp_config_files):
    """Test runner sweep with precision and framework filters."""
    _, runner_file = temp_config_files

    class Args:
        model_prefix = "70b"
        runner_type = "h200"
        precision = "fp8"
        framework = "vllm"
        runner_config = runner_file

    result = generate_runner_sweep_config(Args(), sample_master_config)
    assert all(entry['precision'] == 'fp8' for entry in result)
    assert all(entry['framework'] == 'vllm' for entry in result)


def test_generate_runner_sweep_config_no_matches(sample_master_config, temp_config_files):
    """Test runner sweep with no matching configs."""
    _, runner_file = temp_config_files

    class Args:
        model_prefix = "nonexistent"
        runner_type = "h200"
        precision = None
        framework = None
        runner_config = runner_file

    with pytest.raises(ValueError, match="No configs found matching"):
        generate_runner_sweep_config(Args(), sample_master_config)


# Tests for generate_custom_test
def test_generate_custom_test(temp_config_files):
    """Test custom test generation."""
    _, runner_file = temp_config_files

    class Args:
        runner_label = "h200"
        image = "vllm/vllm-openai:latest"
        model = "test/model"
        framework = "vllm"
        precision = "fp8"
        exp_name = "custom_test"
        runner_config = runner_file

    result = generate_custom_test(Args())
    assert len(result) == 1
    assert result[0]['image'] == "vllm/vllm-openai:latest"
    assert result[0]['exp-name'] == "custom_test"


def test_generate_custom_test_invalid_runner(temp_config_files):
    """Test custom test with invalid runner label."""
    _, runner_file = temp_config_files

    class Args:
        runner_label = "invalid-runner"
        image = "vllm/vllm-openai:latest"
        model = "test/model"
        framework = "vllm"
        precision = "fp8"
        exp_name = "custom_test"
        runner_config = runner_file

    with pytest.raises(ValueError, match="Unable to find specified runner label"):
        generate_custom_test(Args())


# Tests for main function
def test_main_full_sweep(temp_config_files):
    """Test main function with full-sweep command."""
    master_file, _ = temp_config_files

    test_args = [
        "generate_sweep_configs.py",
        "full-sweep",
        "--config-files", master_file,
        "--seq-lens", "1k1k",
        "--model-prefix", "70b",
        "--step-size", "2"
    ]

    with patch('sys.argv', test_args):
        result = main()
        assert len(result) > 0


def test_main_full_sweep_with_filters(temp_config_files):
    """Test main function with full-sweep command with filters."""
    master_file, runner_file = temp_config_files

    test_args = [
        "generate_sweep_configs.py",
        "full-sweep",
        "--config-files", master_file,
        "--runner-config", runner_file,
        "--model-prefix", "70b",
        "--precision", "fp8",
        "--test-mode"
    ]

    with patch('sys.argv', test_args):
        result = main()
        assert len(result) > 0


def test_main_test_config(temp_config_files):
    """Test main function with test-config command."""
    master_file, runner_file = temp_config_files

    test_args = [
        "generate_sweep_configs.py",
        "test-config",
        "--config-files", master_file,
        "--runner-config", runner_file,
        "--key", "70b-fp8-vllm",
        "--runner-node", "h200-nv_1",
        "--test-mode"
    ]

    with patch('sys.argv', test_args):
        result = main()
        assert len(result) > 0


def test_main_runner_model_sweep(temp_config_files):
    """Test main function with runner-model-sweep command."""
    master_file, runner_file = temp_config_files

    test_args = [
        "generate_sweep_configs.py",
        "runner-model-sweep",
        "--config-files", master_file,
        "--runner-config", runner_file,
        "--runner-type", "h200"
    ]

    with patch('sys.argv', test_args):
        result = main()
        assert len(result) > 0


def test_main_runner_model_sweep_with_node_filter(temp_config_files):
    """Test main function with runner-model-sweep command with node filter."""
    master_file, runner_file = temp_config_files

    test_args = [
        "generate_sweep_configs.py",
        "runner-model-sweep",
        "--config-files", master_file,
        "--runner-config", runner_file,
        "--runner-type", "h200",
        "--runner-node-filter", "nv_1"
    ]

    with patch('sys.argv', test_args):
        result = main()
        assert len(result) > 0
        runners = set(entry['runner'] for entry in result)
        assert 'h200-nv_1' in runners
        assert 'h200-nv_2' not in runners


def test_main_runner_sweep(temp_config_files):
    """Test main function with runner-sweep command."""
    master_file, runner_file = temp_config_files

    test_args = [
        "generate_sweep_configs.py",
        "runner-sweep",
        "--config-files", master_file,
        "--runner-config", runner_file,
        "--runner-type", "h200",
        "--model-prefix", "70b"
    ]

    with patch('sys.argv', test_args):
        result = main()
        assert len(result) > 0


def test_main_custom(temp_config_files):
    """Test main function with custom command."""
    master_file, runner_file = temp_config_files

    test_args = [
        "generate_sweep_configs.py",
        "custom",
        "--config-files", master_file,
        "--runner-config", runner_file,
        "--runner-label", "h200",
        "--image", "test:latest",
        "--model", "test/model",
        "--framework", "vllm",
        "--precision", "fp8",
        "--exp-name", "custom_test"
    ]

    with patch('sys.argv', test_args):
        result = main()
        assert len(result) == 1


def test_main_invalid_config_structure(tmp_path):
    """Test main with invalid config structure."""
    invalid_file = tmp_path / "invalid.yaml"
    with open(invalid_file, 'w') as f:
        yaml.dump({"key": {"image": "test"}}, f)  # Missing required fields

    test_args = [
        "generate_sweep_configs.py",
        "full-sweep",
        "--config-files", str(invalid_file),
        "--seq-lens", "1k1k",
        "--model-prefix", "test"
    ]

    with patch('sys.argv', test_args):
        with pytest.raises(ValueError):
            main()


def test_main_validation_failure(temp_config_files, monkeypatch):
    """Test main with validation failure on output."""
    master_file, _ = temp_config_files

    # Monkey patch validate_matrix_output to always fail
    def mock_validate(entries):
        raise ValueError("Validation failed")

    monkeypatch.setattr('generate_sweep_configs.validate_matrix_output', mock_validate)

    test_args = [
        "generate_sweep_configs.py",
        "full-sweep",
        "--config-files", master_file,
        "--seq-lens", "1k1k",
        "--model-prefix", "70b"
    ]

    with patch('sys.argv', test_args):
        with pytest.raises(ValueError, match="Validation failed"):
            main()


# Edge case tests
def test_concurrency_step_reaches_exact_end(sample_master_config, temp_config_files):
    """Test that concurrency stepping reaches exact end value."""
    _, runner_file = temp_config_files

    class Args:
        model_prefix = ["8b"]
        seq_lens = ["1k1k"]
        step_size = 2
        precision = None
        framework = None
        runner_type = None
        test_mode = False
        runner_config = runner_file

    result = generate_full_sweep(Args(), sample_master_config)
    # conc-start=4, conc-end=16, step=2 should give 4,8,16
    conc_values = sorted(set(e['conc'] for e in result))
    assert 16 in conc_values


def test_multiple_model_prefixes_filtered_sweep(sample_master_config, temp_config_files):
    """Test filtered sweep with multiple model prefixes."""
    _, runner_file = temp_config_files

    class Args:
        model_prefix = ["70b", "8b"]
        precision = None
        framework = None
        runner_type = None
        seq_lens = ["1k1k"]
        step_size = 2
        test_mode = False
        runner_config = runner_file

    result = generate_full_sweep(Args(), sample_master_config)
    exp_names = [e['exp-name'] for e in result]
    assert any('70b' in name for name in exp_names)
    assert any('8b' in name for name in exp_names)


def test_seq_len_filter_multiple(sample_master_config, temp_config_files):
    """Test filtering with multiple sequence lengths."""
    _, runner_file = temp_config_files

    class Args:
        model_prefix = ["70b"]
        precision = None
        framework = None
        runner_type = None
        seq_lens = ["1k1k", "1k8k"]
        step_size = 2
        test_mode = False
        runner_config = runner_file

    result = generate_full_sweep(Args(), sample_master_config)
    seq_lens = set((e['isl'], e['osl']) for e in result)
    assert (1024, 1024) in seq_lens
    assert (1024, 8192) in seq_lens


def test_default_ep_dp_attn_values(sample_master_config, temp_config_files):
    """Test that default ep and dp-attn values are set correctly."""
    _, runner_file = temp_config_files

    class Args:
        model_prefix = ["8b"]
        seq_lens = ["1k1k"]
        step_size = 2
        precision = None
        framework = None
        runner_type = None
        test_mode = False
        runner_config = runner_file

    result = generate_full_sweep(Args(), sample_master_config)
    # 8b config doesn't specify ep/dp-attn, so should use defaults
    assert all(e['ep'] == 1 for e in result)
    assert all(e['dp-attn'] == False for e in result)


def test_max_model_len_calculation(sample_master_config, temp_config_files):
    """Test that max-model-len is calculated correctly."""
    _, runner_file = temp_config_files

    class Args:
        model_prefix = ["70b"]
        seq_lens = ["1k8k"]
        step_size = 2
        precision = None
        framework = None
        runner_type = None
        test_mode = False
        runner_config = runner_file

    result = generate_full_sweep(Args(), sample_master_config)
    # isl=1024, osl=8192, so max-model-len should be 1024+8192+200=9416
    assert all(e['max-model-len'] == 9416 for e in result)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=generate_sweep_configs", "--cov-report=term-missing"])
