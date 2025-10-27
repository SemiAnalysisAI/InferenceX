import pytest
import json
import yaml
import tempfile
import os
from pathlib import Path
from get_full_sweep_configs import main, seq_len_stoi


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary directory for config files."""
    return tmp_path


@pytest.fixture
def valid_nvidia_config():
    """Return a valid NVIDIA config structure."""
    return {
        "70b-fp4-b200-trt": {
            "image": "nvcr.io#nvidia/tensorrt-llm/release:1.1.0rc2.post2",
            "model": "nvidia/Llama-3.3-70B-Instruct-FP4",
            "runner": "b200-trt",
            "precision": "fp4",
            "framework": "trt",
            "seq-len-configs": [
                {
                    "isl": 1024,
                    "osl": 1024,
                    "bmk-space": [
                        {"tp": 1, "conc-start": 128, "conc-end": 128},
                        {"tp": 2, "conc-start": 64, "conc-end": 128},
                    ]
                },
                {
                    "isl": 1024,
                    "osl": 8192,
                    "bmk-space": [
                        {"tp": 4, "conc-start": 16, "conc-end": 128},
                    ]
                }
            ]
        }
    }


@pytest.fixture
def valid_amd_config():
    """Return a valid AMD config structure."""
    return {
        "70b-fp8-mi355x-vllm": {
            "image": "rocm/7.0:rocm7.0_ubuntu_22.04_vllm_0.10.1_instinct_20250927_rc1",
            "model": "amd/Llama-3.3-70B-Instruct-FP8-KV",
            "runner": "mi355x",
            "precision": "fp8",
            "framework": "vllm",
            "seq-len-configs": [
                {
                    "isl": 1024,
                    "osl": 1024,
                    "bmk-space": [
                        {"tp": 1, "conc-start": 32, "conc-end": 64},
                    ]
                }
            ]
        }
    }


@pytest.fixture
def config_with_optional_fields():
    """Return a config with optional ep and dp-attn fields."""
    return {
        "dsr1-fp4-b200-trt": {
            "image": "nvcr.io#nvidia/tensorrt-llm/release:1.1.0rc2.post2",
            "model": "nvidia/DeepSeek-R1-0528-FP4-V2",
            "runner": "b200-trt",
            "precision": "fp4",
            "framework": "trt",
            "seq-len-configs": [
                {
                    "isl": 1024,
                    "osl": 1024,
                    "bmk-space": [
                        {"tp": 4, "conc-start": 4, "conc-end": 32},
                        {"tp": 4, "ep": 4, "conc-start": 64, "conc-end": 128},
                        {"tp": 4, "ep": 4, "dp-attn": True, "conc-start": 256, "conc-end": 256},
                    ]
                }
            ]
        }
    }


def create_config_file(temp_dir, filename, config_data):
    """Helper to create a YAML config file."""
    config_path = temp_dir / filename
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    return str(config_path)


class TestMainFunction:
    """Test suite for the main function."""

    def test_basic_config_1k1k(self, temp_config_dir, valid_nvidia_config, monkeypatch, capsys):
        """Test basic configuration with 1k1k sequence lengths."""
        config_file = create_config_file(temp_config_dir, "nvidia.yaml", valid_nvidia_config)

        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', config_file,
            '--seq-lens', '1k1k',
            '--model-prefix', '70b'
        ])

        result = main()

        # Verify output structure
        assert isinstance(result, list)
        assert len(result) == 3  # 1 config with 128 + 2 configs (64, 128)

        # Verify all results have required fields
        for entry in result:
            assert 'image' in entry
            assert 'model' in entry
            assert 'precision' in entry
            assert 'framework' in entry
            assert 'runner' in entry
            assert 'isl' in entry
            assert 'osl' in entry
            assert 'tp' in entry
            assert 'conc' in entry
            assert entry['isl'] == 1024
            assert entry['osl'] == 1024

        # Verify JSON output to stdout
        captured = capsys.readouterr()
        json_output = json.loads(captured.out.strip())
        assert json_output == result

    def test_multiple_config_files(self, temp_config_dir, valid_nvidia_config, valid_amd_config, monkeypatch):
        """Test with multiple config files."""
        nvidia_file = create_config_file(temp_config_dir, "nvidia.yaml", valid_nvidia_config)
        amd_file = create_config_file(temp_config_dir, "amd.yaml", valid_amd_config)

        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', nvidia_file, amd_file,
            '--seq-lens', '1k1k',
            '--model-prefix', '70b'
        ])

        result = main()

        # Should have entries from both configs
        assert len(result) > 0
        runners = {entry['runner'] for entry in result}
        assert 'b200-trt' in runners
        assert 'mi355x' in runners

    def test_model_prefix_filtering(self, temp_config_dir, valid_nvidia_config, config_with_optional_fields, monkeypatch):
        """Test that model prefix filtering works correctly."""
        combined_config = {**valid_nvidia_config, **config_with_optional_fields}
        config_file = create_config_file(temp_config_dir, "combined.yaml", combined_config)

        # Filter for 70b only
        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', config_file,
            '--seq-lens', '1k1k',
            '--model-prefix', '70b'
        ])

        result = main()

        # Should only have 70b configs
        assert all('70b' in list(combined_config.keys())[0] for entry in result)
        assert len(result) == 3  # Only from 70b config

        # Filter for dsr1 only
        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', config_file,
            '--seq-lens', '1k1k',
            '--model-prefix', 'dsr1'
        ])

        result = main()

        # Should only have dsr1 configs
        # 3 bmk-space entries: [4,8,16,32] + [64,128] + [256] = 4+2+1 = 7 entries
        assert len(result) == 7

    def test_optional_fields_ep_and_dp_attn(self, temp_config_dir, config_with_optional_fields, monkeypatch):
        """Test that optional ep and dp-attn fields are included when present."""
        config_file = create_config_file(temp_config_dir, "config.yaml", config_with_optional_fields)

        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', config_file,
            '--seq-lens', '1k1k',
            '--model-prefix', 'dsr1'
        ])

        result = main()

        # Check entries without optional fields
        entries_without_ep = [e for e in result if 'ep' not in e]
        assert len(entries_without_ep) > 0
        for entry in entries_without_ep:
            assert entry['conc'] <= 32

        # Check entries with ep but without dp-attn
        entries_with_ep_no_dp = [e for e in result if 'ep' in e and 'dp-attn' not in e]
        assert len(entries_with_ep_no_dp) > 0
        for entry in entries_with_ep_no_dp:
            assert entry['ep'] == 4
            assert 64 <= entry['conc'] <= 128

        # Check entries with both ep and dp-attn
        entries_with_both = [e for e in result if 'ep' in e and 'dp-attn' in e]
        assert len(entries_with_both) > 0
        for entry in entries_with_both:
            assert entry['ep'] == 4
            assert entry['dp-attn'] is True
            assert entry['conc'] == 256

    def test_step_size_default(self, temp_config_dir, valid_nvidia_config, monkeypatch):
        """Test default step size of 2."""
        config_file = create_config_file(temp_config_dir, "nvidia.yaml", valid_nvidia_config)

        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', config_file,
            '--seq-lens', '1k1k',
            '--model-prefix', '70b'
        ])

        result = main()

        # For tp=2, conc-start=64, conc-end=128 with step=2
        # Should generate: 64, 128
        tp2_entries = [e for e in result if e['tp'] == 2]
        tp2_concs = sorted([e['conc'] for e in tp2_entries])
        assert tp2_concs == [64, 128]

    def test_step_size_custom(self, temp_config_dir, valid_nvidia_config, monkeypatch):
        """Test custom step size."""
        config_file = create_config_file(temp_config_dir, "nvidia.yaml", valid_nvidia_config)

        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', config_file,
            '--seq-lens', '1k1k',
            '--model-prefix', '70b',
            '--step-size', '4'
        ])

        result = main()

        # For tp=2, conc-start=64, conc-end=128 with step=4
        # Should generate: 64, 128 (64*4=256 > 128, so stop at 128)
        tp2_entries = [e for e in result if e['tp'] == 2]
        tp2_concs = sorted([e['conc'] for e in tp2_entries])
        assert tp2_concs == [64, 128]

    def test_conc_range_single_value(self, temp_config_dir, monkeypatch):
        """Test when conc-start equals conc-end."""
        config = {
            "test-config": {
                "image": "test-image",
                "model": "test-model",
                "runner": "test-runner",
                "precision": "fp8",
                "framework": "vllm",
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "bmk-space": [
                            {"tp": 1, "conc-start": 64, "conc-end": 64},
                        ]
                    }
                ]
            }
        }
        config_file = create_config_file(temp_config_dir, "config.yaml", config)

        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', config_file,
            '--seq-lens', '1k1k',
            '--model-prefix', 'test'
        ])

        result = main()

        assert len(result) == 1
        assert result[0]['conc'] == 64

    def test_different_seq_lens(self, temp_config_dir, valid_nvidia_config, monkeypatch):
        """Test with different sequence length configurations."""
        config_file = create_config_file(temp_config_dir, "nvidia.yaml", valid_nvidia_config)

        # Test 1k8k
        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', config_file,
            '--seq-lens', '1k8k',
            '--model-prefix', '70b'
        ])

        result = main()

        # Should match 1k8k config
        assert all(e['isl'] == 1024 and e['osl'] == 8192 for e in result)
        assert len(result) > 0

    def test_no_matching_seq_lens(self, temp_config_dir, valid_nvidia_config, monkeypatch):
        """Test when no configs match the requested sequence lengths."""
        config_file = create_config_file(temp_config_dir, "nvidia.yaml", valid_nvidia_config)

        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', config_file,
            '--seq-lens', '8k1k',  # Not in the config
            '--model-prefix', '70b'
        ])

        result = main()

        # Should return empty list
        assert result == []

    def test_no_matching_model_prefix(self, temp_config_dir, valid_nvidia_config, monkeypatch):
        """Test when no configs match the model prefix."""
        config_file = create_config_file(temp_config_dir, "nvidia.yaml", valid_nvidia_config)

        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', config_file,
            '--seq-lens', '1k1k',
            '--model-prefix', 'nonexistent'
        ])

        result = main()

        # Should return empty list
        assert result == []


class TestErrorHandling:
    """Test suite for error handling."""

    def test_missing_config_file(self, temp_config_dir, monkeypatch):
        """Test error when config file doesn't exist."""
        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', '/nonexistent/file.yaml',
            '--seq-lens', '1k1k',
            '--model-prefix', '70b'
        ])

        with pytest.raises(ValueError, match="does not exist"):
            main()

    def test_invalid_yaml(self, temp_config_dir, monkeypatch):
        """Test error when YAML is invalid."""
        config_path = temp_config_dir / "invalid.yaml"
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: content: [")

        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', str(config_path),
            '--seq-lens', '1k1k',
            '--model-prefix', '70b'
        ])

        with pytest.raises(yaml.YAMLError):
            main()

    def test_non_dict_config(self, temp_config_dir, monkeypatch):
        """Test error when config is not a dictionary."""
        config_path = temp_config_dir / "list.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(["not", "a", "dict"], f)

        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', str(config_path),
            '--seq-lens', '1k1k',
            '--model-prefix', '70b'
        ])

        with pytest.raises(AssertionError, match="must contain a dictionary"):
            main()

    def test_duplicate_keys(self, temp_config_dir, monkeypatch):
        """Test error when duplicate keys exist across config files."""
        config1 = {
            "70b-fp4-b200-trt": {
                "image": "image1",
                "model": "model1",
                "runner": "runner1",
                "precision": "fp4",
                "framework": "trt",
                "seq-len-configs": []
            }
        }
        config2 = {
            "70b-fp4-b200-trt": {  # Same key
                "image": "image2",
                "model": "model2",
                "runner": "runner2",
                "precision": "fp4",
                "framework": "trt",
                "seq-len-configs": []
            }
        }

        file1 = create_config_file(temp_config_dir, "config1.yaml", config1)
        file2 = create_config_file(temp_config_dir, "config2.yaml", config2)

        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', file1, file2,
            '--seq-lens', '1k1k',
            '--model-prefix', '70b'
        ])

        with pytest.raises(ValueError, match="Duplicate configuration keys"):
            main()

    def test_missing_seq_len_configs(self, temp_config_dir, monkeypatch):
        """Test error when seq-len-configs is missing."""
        config = {
            "70b-fp4-b200-trt": {
                "image": "test-image",
                "model": "test-model",
                "runner": "test-runner",
                "precision": "fp4",
                "framework": "trt",
                # Missing seq-len-configs
            }
        }
        config_file = create_config_file(temp_config_dir, "config.yaml", config)

        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', config_file,
            '--seq-lens', '1k1k',
            '--model-prefix', '70b'
        ])

        with pytest.raises(AssertionError, match="Missing 'seq-len-configs'"):
            main()

    def test_missing_required_fields(self, temp_config_dir, monkeypatch):
        """Test error when required fields are missing."""
        # Missing 'model' field
        config = {
            "70b-fp4-b200-trt": {
                "image": "test-image",
                # Missing model
                "runner": "test-runner",
                "precision": "fp4",
                "framework": "trt",
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "bmk-space": [
                            {"tp": 1, "conc-start": 64, "conc-end": 64}
                        ]
                    }
                ]
            }
        }
        config_file = create_config_file(temp_config_dir, "config.yaml", config)

        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', config_file,
            '--seq-lens', '1k1k',
            '--model-prefix', '70b'
        ])

        with pytest.raises(AssertionError, match="Missing required fields"):
            main()

    def test_missing_bmk_space(self, temp_config_dir, monkeypatch):
        """Test error when bmk-space is missing."""
        config = {
            "70b-fp4-b200-trt": {
                "image": "test-image",
                "model": "test-model",
                "runner": "test-runner",
                "precision": "fp4",
                "framework": "trt",
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        # Missing bmk-space
                    }
                ]
            }
        }
        config_file = create_config_file(temp_config_dir, "config.yaml", config)

        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', config_file,
            '--seq-lens', '1k1k',
            '--model-prefix', '70b'
        ])

        with pytest.raises(AssertionError, match="Missing 'bmk-space'"):
            main()

    def test_missing_bmk_space_fields(self, temp_config_dir, monkeypatch):
        """Test error when tp, conc-start, or conc-end is missing."""
        config = {
            "70b-fp4-b200-trt": {
                "image": "test-image",
                "model": "test-model",
                "runner": "test-runner",
                "precision": "fp4",
                "framework": "trt",
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "bmk-space": [
                            {"tp": 1, "conc-start": 64}  # Missing conc-end
                        ]
                    }
                ]
            }
        }
        config_file = create_config_file(temp_config_dir, "config.yaml", config)

        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', config_file,
            '--seq-lens', '1k1k',
            '--model-prefix', '70b'
        ])

        with pytest.raises(AssertionError, match="Missing 'tp', 'conc-start', or 'conc-end'"):
            main()


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_empty_config(self, temp_config_dir, monkeypatch):
        """Test with empty config file."""
        config = {}
        config_file = create_config_file(temp_config_dir, "empty.yaml", config)

        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', config_file,
            '--seq-lens', '1k1k',
            '--model-prefix', '70b'
        ])

        result = main()

        # Should return empty list
        assert result == []

    def test_large_conc_range(self, temp_config_dir, monkeypatch):
        """Test with large concurrency range."""
        config = {
            "test-config": {
                "image": "test-image",
                "model": "test-model",
                "runner": "test-runner",
                "precision": "fp8",
                "framework": "vllm",
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "bmk-space": [
                            {"tp": 1, "conc-start": 4, "conc-end": 1024},
                        ]
                    }
                ]
            }
        }
        config_file = create_config_file(temp_config_dir, "config.yaml", config)

        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', config_file,
            '--seq-lens', '1k1k',
            '--model-prefix', 'test'
        ])

        result = main()

        # With step=2: 4, 8, 16, 32, 64, 128, 256, 512, 1024
        concs = sorted([e['conc'] for e in result])
        assert concs == [4, 8, 16, 32, 64, 128, 256, 512, 1024]

    def test_conc_end_not_power_of_step(self, temp_config_dir, monkeypatch):
        """Test when conc-end is not a power of step size."""
        config = {
            "test-config": {
                "image": "test-image",
                "model": "test-model",
                "runner": "test-runner",
                "precision": "fp8",
                "framework": "vllm",
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "bmk-space": [
                            {"tp": 1, "conc-start": 10, "conc-end": 100},
                        ]
                    }
                ]
            }
        }
        config_file = create_config_file(temp_config_dir, "config.yaml", config)

        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', config_file,
            '--seq-lens', '1k1k',
            '--model-prefix', 'test'
        ])

        result = main()

        # With step=2: 10, 20, 40, 80, 100 (last value is conc-end)
        concs = sorted([e['conc'] for e in result])
        assert concs == [10, 20, 40, 80, 100]
        assert concs[-1] == 100  # Should always include conc-end

    def test_all_seq_lens_in_stoi(self):
        """Test that all defined seq_lens work correctly."""
        assert seq_len_stoi["1k1k"] == (1024, 1024)
        assert seq_len_stoi["1k8k"] == (1024, 8192)
        assert seq_len_stoi["8k1k"] == (8192, 1024)

    def test_multiple_bmk_space_entries(self, temp_config_dir, monkeypatch):
        """Test with multiple bmk-space entries."""
        config = {
            "test-config": {
                "image": "test-image",
                "model": "test-model",
                "runner": "test-runner",
                "precision": "fp8",
                "framework": "vllm",
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "bmk-space": [
                            {"tp": 1, "conc-start": 32, "conc-end": 64},
                            {"tp": 2, "conc-start": 16, "conc-end": 32},
                            {"tp": 4, "conc-start": 8, "conc-end": 16},
                        ]
                    }
                ]
            }
        }
        config_file = create_config_file(temp_config_dir, "config.yaml", config)

        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', config_file,
            '--seq-lens', '1k1k',
            '--model-prefix', 'test'
        ])

        result = main()

        # Verify all tp values are present
        tp_values = sorted(set(e['tp'] for e in result))
        assert tp_values == [1, 2, 4]

        # Verify correct conc ranges for each tp
        tp1_concs = sorted([e['conc'] for e in result if e['tp'] == 1])
        tp2_concs = sorted([e['conc'] for e in result if e['tp'] == 2])
        tp4_concs = sorted([e['conc'] for e in result if e['tp'] == 4])

        assert tp1_concs == [32, 64]
        assert tp2_concs == [16, 32]
        assert tp4_concs == [8, 16]

    def test_output_format(self, temp_config_dir, valid_nvidia_config, monkeypatch, capsys):
        """Test that output is valid JSON and matches expected format."""
        config_file = create_config_file(temp_config_dir, "nvidia.yaml", valid_nvidia_config)

        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', config_file,
            '--seq-lens', '1k1k',
            '--model-prefix', '70b'
        ])

        result = main()

        # Capture stdout
        captured = capsys.readouterr()

        # Verify it's valid JSON
        json_output = json.loads(captured.out.strip())

        # Verify it matches the result
        assert json_output == result

        # Verify each entry has the correct structure
        for entry in json_output:
            assert isinstance(entry, dict)
            assert all(isinstance(k, str) for k in entry.keys())
            assert entry['image'] == valid_nvidia_config['70b-fp4-b200-trt']['image']
            assert entry['model'] == valid_nvidia_config['70b-fp4-b200-trt']['model']
            assert entry['precision'] == valid_nvidia_config['70b-fp4-b200-trt']['precision']
            assert entry['framework'] == valid_nvidia_config['70b-fp4-b200-trt']['framework']
            assert entry['runner'] == valid_nvidia_config['70b-fp4-b200-trt']['runner']


class TestConcurrencyGeneration:
    """Test suite specifically for concurrency value generation logic."""

    def test_conc_progression_step_2(self, temp_config_dir, monkeypatch):
        """Test concurrency progression with step size 2."""
        config = {
            "test-config": {
                "image": "test-image",
                "model": "test-model",
                "runner": "test-runner",
                "precision": "fp8",
                "framework": "vllm",
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "bmk-space": [
                            {"tp": 1, "conc-start": 1, "conc-end": 16},
                        ]
                    }
                ]
            }
        }
        config_file = create_config_file(temp_config_dir, "config.yaml", config)

        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', config_file,
            '--seq-lens', '1k1k',
            '--model-prefix', 'test',
            '--step-size', '2'
        ])

        result = main()

        # Should multiply by 2 each time: 1, 2, 4, 8, 16
        concs = sorted([e['conc'] for e in result])
        assert concs == [1, 2, 4, 8, 16]

    def test_conc_progression_step_3(self, temp_config_dir, monkeypatch):
        """Test concurrency progression with step size 3."""
        config = {
            "test-config": {
                "image": "test-image",
                "model": "test-model",
                "runner": "test-runner",
                "precision": "fp8",
                "framework": "vllm",
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "bmk-space": [
                            {"tp": 1, "conc-start": 2, "conc-end": 100},
                        ]
                    }
                ]
            }
        }
        config_file = create_config_file(temp_config_dir, "config.yaml", config)

        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', config_file,
            '--seq-lens', '1k1k',
            '--model-prefix', 'test',
            '--step-size', '3'
        ])

        result = main()

        # Should multiply by 3 each time: 2, 6, 18, 54, 100
        concs = sorted([e['conc'] for e in result])
        assert concs == [2, 6, 18, 54, 100]

    def test_conc_exact_end_value(self, temp_config_dir, monkeypatch):
        """Test that conc-end is always included even if not reached by progression."""
        config = {
            "test-config": {
                "image": "test-image",
                "model": "test-model",
                "runner": "test-runner",
                "precision": "fp8",
                "framework": "vllm",
                "seq-len-configs": [
                    {
                        "isl": 1024,
                        "osl": 1024,
                        "bmk-space": [
                            {"tp": 1, "conc-start": 5, "conc-end": 50},
                        ]
                    }
                ]
            }
        }
        config_file = create_config_file(temp_config_dir, "config.yaml", config)

        monkeypatch.setattr('sys.argv', [
            'script.py',
            '--config-files', config_file,
            '--seq-lens', '1k1k',
            '--model-prefix', 'test',
            '--step-size', '2'
        ])

        result = main()

        concs = sorted([e['conc'] for e in result])
        # 5, 10, 20, 40, 50 (40*2=80 > 50, so we include 50)
        assert concs[-1] == 50
        assert 50 in concs
