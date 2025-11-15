import pytest
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, mock_open

# Import the script as a module would be tricky since it's a script
# Instead we'll test by setting env vars and running it


def create_mock_result_file(tmp_path):
    """Create a mock result JSON file."""
    result_data = {
        "max_concurrency": 10,
        "model_id": "test-model",
        "total_token_throughput": 1000.0,
        "output_throughput": 400.0,
        "ttft_ms": 50.0,
        "tpot_ms": 20.0
    }
    result_file = tmp_path / "test_result.json"
    with open(result_file, 'w') as f:
        json.dump(result_data, f)
    return str(result_file.stem)


def test_disagg_true_when_both_env_vars_set(tmp_path, monkeypatch):
    """Test that disagg=true when both PREFILL_GPUS and DECODE_GPUS are set."""
    # Create mock result file
    result_filename = create_mock_result_file(tmp_path)
    
    # Set environment variables
    monkeypatch.setenv('RUNNER_TYPE', 'h200')
    monkeypatch.setenv('TP', '8')
    monkeypatch.setenv('EP_SIZE', '1')
    monkeypatch.setenv('PREFILL_GPUS', '4')
    monkeypatch.setenv('DECODE_GPUS', '4')
    monkeypatch.setenv('DP_ATTENTION', 'false')
    monkeypatch.setenv('RESULT_FILENAME', 'test_result')
    monkeypatch.setenv('FRAMEWORK', 'vllm')
    monkeypatch.setenv('PRECISION', 'fp8')
    
    # Change to tmp_path directory
    monkeypatch.chdir(tmp_path)
    
    # Import and run the script
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "process_result",
        "/home/runner/work/InferenceMAX/InferenceMAX/utils/process_result.py"
    )
    module = importlib.util.module_from_spec(spec)
    
    # Capture stdout
    from io import StringIO
    import sys
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        spec.loader.exec_module(module)
        output = sys.stdout.getvalue()
        data = json.loads(output)
        
        # Check that disagg is true
        assert data['disagg'] == True
        # Check that num_prefill_gpu and num_decode_gpu are present
        assert data['num_prefill_gpu'] == 4
        assert data['num_decode_gpu'] == 4
    finally:
        sys.stdout = old_stdout


def test_disagg_false_when_prefill_gpus_not_set(tmp_path, monkeypatch):
    """Test that disagg=false when PREFILL_GPUS is not set."""
    # Create mock result file
    result_filename = create_mock_result_file(tmp_path)
    
    # Set environment variables (without PREFILL_GPUS)
    monkeypatch.setenv('RUNNER_TYPE', 'h200')
    monkeypatch.setenv('TP', '8')
    monkeypatch.setenv('EP_SIZE', '1')
    monkeypatch.setenv('DECODE_GPUS', '4')
    monkeypatch.setenv('DP_ATTENTION', 'false')
    monkeypatch.setenv('RESULT_FILENAME', 'test_result')
    monkeypatch.setenv('FRAMEWORK', 'vllm')
    monkeypatch.setenv('PRECISION', 'fp8')
    
    # Change to tmp_path directory
    monkeypatch.chdir(tmp_path)
    
    # Import and run the script
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "process_result",
        "/home/runner/work/InferenceMAX/InferenceMAX/utils/process_result.py"
    )
    module = importlib.util.module_from_spec(spec)
    
    # Capture stdout
    from io import StringIO
    import sys
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        spec.loader.exec_module(module)
        output = sys.stdout.getvalue()
        data = json.loads(output)
        
        # Check that disagg is false
        assert data['disagg'] == False
        # Check that num_prefill_gpu and num_decode_gpu are NOT present
        assert 'num_prefill_gpu' not in data
        assert 'num_decode_gpu' not in data
    finally:
        sys.stdout = old_stdout


def test_disagg_false_when_decode_gpus_not_set(tmp_path, monkeypatch):
    """Test that disagg=false when DECODE_GPUS is not set."""
    # Create mock result file
    result_filename = create_mock_result_file(tmp_path)
    
    # Set environment variables (without DECODE_GPUS)
    monkeypatch.setenv('RUNNER_TYPE', 'h200')
    monkeypatch.setenv('TP', '8')
    monkeypatch.setenv('EP_SIZE', '1')
    monkeypatch.setenv('PREFILL_GPUS', '4')
    monkeypatch.setenv('DP_ATTENTION', 'false')
    monkeypatch.setenv('RESULT_FILENAME', 'test_result')
    monkeypatch.setenv('FRAMEWORK', 'vllm')
    monkeypatch.setenv('PRECISION', 'fp8')
    
    # Change to tmp_path directory
    monkeypatch.chdir(tmp_path)
    
    # Import and run the script
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "process_result",
        "/home/runner/work/InferenceMAX/InferenceMAX/utils/process_result.py"
    )
    module = importlib.util.module_from_spec(spec)
    
    # Capture stdout
    from io import StringIO
    import sys
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        spec.loader.exec_module(module)
        output = sys.stdout.getvalue()
        data = json.loads(output)
        
        # Check that disagg is false
        assert data['disagg'] == False
        # Check that num_prefill_gpu and num_decode_gpu are NOT present
        assert 'num_prefill_gpu' not in data
        assert 'num_decode_gpu' not in data
    finally:
        sys.stdout = old_stdout


def test_disagg_false_when_both_env_vars_empty_strings(tmp_path, monkeypatch):
    """Test that disagg=false when both PREFILL_GPUS and DECODE_GPUS are empty strings."""
    # Create mock result file
    result_filename = create_mock_result_file(tmp_path)
    
    # Set environment variables with empty strings
    monkeypatch.setenv('RUNNER_TYPE', 'h200')
    monkeypatch.setenv('TP', '8')
    monkeypatch.setenv('EP_SIZE', '1')
    monkeypatch.setenv('PREFILL_GPUS', '')
    monkeypatch.setenv('DECODE_GPUS', '')
    monkeypatch.setenv('DP_ATTENTION', 'false')
    monkeypatch.setenv('RESULT_FILENAME', 'test_result')
    monkeypatch.setenv('FRAMEWORK', 'vllm')
    monkeypatch.setenv('PRECISION', 'fp8')
    
    # Change to tmp_path directory
    monkeypatch.chdir(tmp_path)
    
    # Import and run the script
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "process_result",
        "/home/runner/work/InferenceMAX/InferenceMAX/utils/process_result.py"
    )
    module = importlib.util.module_from_spec(spec)
    
    # Capture stdout
    from io import StringIO
    import sys
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        spec.loader.exec_module(module)
        output = sys.stdout.getvalue()
        data = json.loads(output)
        
        # Check that disagg is false
        assert data['disagg'] == False
        # Check that num_prefill_gpu and num_decode_gpu are NOT present
        assert 'num_prefill_gpu' not in data
        assert 'num_decode_gpu' not in data
    finally:
        sys.stdout = old_stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
