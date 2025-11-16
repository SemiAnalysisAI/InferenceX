import pytest
import json
import sys
import importlib.util
from io import StringIO


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


def run_process_result_script(tmp_path):
    """Helper to run process_result.py and return the output data."""
    # Create mock result file
    create_mock_result_file(tmp_path)
    
    # Get script path relative to this test file
    script_path = Path(__file__).parent / "process_result.py"
    spec = importlib.util.spec_from_file_location("process_result", script_path)
    module = importlib.util.module_from_spec(spec)
    
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        spec.loader.exec_module(module)
        output = sys.stdout.getvalue()
        return json.loads(output)
    finally:
        sys.stdout = old_stdout


def test_disagg_true_when_both_env_vars_set(tmp_path, monkeypatch):
    """Test that disagg=true when both PREFILL_GPUS and DECODE_GPUS are set."""
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
    
    # Run the script and get output
    data = run_process_result_script(tmp_path)
    
    # Check that disagg is true
    assert data['disagg'] is True
    # Check that num_prefill_gpu and num_decode_gpu are present
    assert data['num_prefill_gpu'] == 4
    assert data['num_decode_gpu'] == 4


def test_disagg_false_when_prefill_gpus_not_set(tmp_path, monkeypatch):
    """Test that disagg=false when PREFILL_GPUS is not set."""
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
    
    # Run the script and get output
    data = run_process_result_script(tmp_path)
    
    # Check that disagg is false
    assert not data['disagg']
    # Check that num_prefill_gpu and num_decode_gpu are NOT present
    assert 'num_prefill_gpu' not in data
    assert 'num_decode_gpu' not in data


def test_disagg_false_when_decode_gpus_not_set(tmp_path, monkeypatch):
    """Test that disagg=false when DECODE_GPUS is not set."""
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
    
    # Run the script and get output
    data = run_process_result_script(tmp_path)
    
    # Check that disagg is false
    assert not data['disagg']
    # Check that num_prefill_gpu and num_decode_gpu are NOT present
    assert 'num_prefill_gpu' not in data
    assert 'num_decode_gpu' not in data


def test_disagg_false_when_both_env_vars_empty_strings(tmp_path, monkeypatch):
    """Test that disagg=false when both PREFILL_GPUS and DECODE_GPUS are empty strings."""
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
    
    # Run the script and get output
    data = run_process_result_script(tmp_path)
    
    # Check that disagg is false
    assert not data['disagg']
    # Check that num_prefill_gpu and num_decode_gpu are NOT present
    assert 'num_prefill_gpu' not in data
    assert 'num_decode_gpu' not in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
