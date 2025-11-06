import pytest
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, mock_open


@pytest.fixture
def sample_benchmark_result():
    """Sample benchmark result JSON data."""
    return {
        'max_concurrency': 8,
        'model_id': 'meta-llama/Llama-3-70b',
        'total_token_throughput': 10000.0,
        'output_throughput': 3000.0,
        'ttft_ms': 150.5,
        'tpot_ms': 25.0,
        'e2e_latency_ms': 500.0,
        'decode_tpot_ms': 30.0,
        'prefill_tpot_ms': 20.0
    }


@pytest.fixture
def basic_env_vars():
    """Basic environment variables for testing."""
    return {
        'RUNNER_TYPE': 'h200',
        'TP': '8',
        'EP_SIZE': '1',
        'PREFILL_GPUS': '',
        'DECODE_GPUS': '',
        'DP_ATTENTION': 'false',
        'RESULT_FILENAME': 'test_result',
        'FRAMEWORK': 'vllm',
        'PRECISION': 'fp8',
        'MTP_MODE': ''
    }


@pytest.fixture
def temp_result_file(tmp_path, sample_benchmark_result):
    """Create a temporary benchmark result file."""
    result_file = tmp_path / 'test_result.json'
    with open(result_file, 'w') as f:
        json.dump(sample_benchmark_result, f)
    return result_file


def test_basic_processing(tmp_path, sample_benchmark_result, basic_env_vars):
    """Test basic processing of benchmark results."""
    # Create result file
    result_file = tmp_path / 'test_result.json'
    with open(result_file, 'w') as f:
        json.dump(sample_benchmark_result, f)
    
    # Change to tmp_path directory
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        with patch.dict(os.environ, basic_env_vars):
            # Import and execute the script
            exec(open('/home/runner/work/InferenceMAX/InferenceMAX/utils/process_result.py').read())
        
        # Check output file was created
        output_file = tmp_path / 'agg_test_result.json'
        assert output_file.exists()
        
        # Load and verify output
        with open(output_file) as f:
            result = json.load(f)
        
        assert result['hw'] == 'h200'
        assert result['tp'] == 8
        assert result['ep'] == 1
        assert result['dp_attention'] == 'false'
        assert result['conc'] == 8
        assert result['model'] == 'meta-llama/Llama-3-70b'
        assert result['framework'] == 'vllm'
        assert result['precision'] == 'fp8'
        assert result['tput_per_gpu'] == 10000.0 / 8
        assert result['output_tput_per_gpu'] == 3000.0 / 8
        assert result['input_tput_per_gpu'] == (10000.0 - 3000.0) / 8
        
    finally:
        os.chdir(original_dir)


def test_ms_to_seconds_conversion(tmp_path, basic_env_vars):
    """Test conversion of millisecond values to seconds."""
    benchmark_result = {
        'max_concurrency': 4,
        'model_id': 'test/model',
        'total_token_throughput': 5000.0,
        'output_throughput': 1500.0,
        'ttft_ms': 200.0,
        'e2e_latency_ms': 1000.0,
        'decode_latency_ms': 500.0
    }
    
    result_file = tmp_path / 'test_result.json'
    with open(result_file, 'w') as f:
        json.dump(benchmark_result, f)
    
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        with patch.dict(os.environ, basic_env_vars):
            exec(open('/home/runner/work/InferenceMAX/InferenceMAX/utils/process_result.py').read())
        
        output_file = tmp_path / 'agg_test_result.json'
        with open(output_file) as f:
            result = json.load(f)
        
        # Check ms values were converted to seconds
        assert result['ttft'] == 200.0 / 1000.0
        assert result['e2e_latency'] == 1000.0 / 1000.0
        assert result['decode_latency'] == 500.0 / 1000.0
        
    finally:
        os.chdir(original_dir)


def test_tpot_to_intvty_conversion(tmp_path, basic_env_vars):
    """Test conversion of tpot (time per output token) to intvty (interactivity/throughput)."""
    benchmark_result = {
        'max_concurrency': 2,
        'model_id': 'test/model',
        'total_token_throughput': 2000.0,
        'output_throughput': 500.0,
        'tpot_ms': 25.0,
        'decode_tpot_ms': 20.0,
        'prefill_tpot_ms': 30.0
    }
    
    result_file = tmp_path / 'test_result.json'
    with open(result_file, 'w') as f:
        json.dump(benchmark_result, f)
    
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        with patch.dict(os.environ, basic_env_vars):
            exec(open('/home/runner/work/InferenceMAX/InferenceMAX/utils/process_result.py').read())
        
        output_file = tmp_path / 'agg_test_result.json'
        with open(output_file) as f:
            result = json.load(f)
        
        # Check tpot values were converted to intvty
        # The logic: if 'tpot' in key, convert ms value and then intvty = 1000.0 / tpot_ms
        # So: tpot_ms: 25.0 -> tpot: 0.025 (ms to s), intvty: 1000.0/25.0 = 40.0
        assert result['tpot'] == 25.0 / 1000.0  # Converted from ms to s
        assert result['intvty'] == 1000.0 / 25.0  # intvty = 1000.0 / tpot_ms
        
        assert result['decode_tpot'] == 20.0 / 1000.0
        assert result['decode_intvty'] == 1000.0 / 20.0
        
        assert result['prefill_tpot'] == 30.0 / 1000.0
        assert result['prefill_intvty'] == 1000.0 / 30.0
        
        # Check that the intvty calculation is correct
        assert 'decode_intvty' in result
        assert 'prefill_intvty' in result
        
    finally:
        os.chdir(original_dir)


def test_mtp_mode_included(tmp_path, sample_benchmark_result, basic_env_vars):
    """Test that MTP mode is included when set."""
    env_vars = basic_env_vars.copy()
    env_vars['MTP_MODE'] = 'disaggregated'
    
    result_file = tmp_path / 'test_result.json'
    with open(result_file, 'w') as f:
        json.dump(sample_benchmark_result, f)
    
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        with patch.dict(os.environ, env_vars):
            exec(open('/home/runner/work/InferenceMAX/InferenceMAX/utils/process_result.py').read())
        
        output_file = tmp_path / 'agg_test_result.json'
        with open(output_file) as f:
            result = json.load(f)
        
        assert 'mtp' in result
        assert result['mtp'] == 'disaggregated'
        
    finally:
        os.chdir(original_dir)


def test_mtp_mode_not_included(tmp_path, sample_benchmark_result, basic_env_vars):
    """Test that MTP mode is not included when not set."""
    result_file = tmp_path / 'test_result.json'
    with open(result_file, 'w') as f:
        json.dump(sample_benchmark_result, f)
    
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        with patch.dict(os.environ, basic_env_vars):
            exec(open('/home/runner/work/InferenceMAX/InferenceMAX/utils/process_result.py').read())
        
        output_file = tmp_path / 'agg_test_result.json'
        with open(output_file) as f:
            result = json.load(f)
        
        assert 'mtp' not in result
        
    finally:
        os.chdir(original_dir)


def test_prefill_decode_gpus_explicit(tmp_path, sample_benchmark_result, basic_env_vars):
    """Test explicit prefill and decode GPU counts."""
    env_vars = basic_env_vars.copy()
    env_vars['PREFILL_GPUS'] = '4'
    env_vars['DECODE_GPUS'] = '4'
    
    result_file = tmp_path / 'test_result.json'
    with open(result_file, 'w') as f:
        json.dump(sample_benchmark_result, f)
    
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        with patch.dict(os.environ, env_vars):
            exec(open('/home/runner/work/InferenceMAX/InferenceMAX/utils/process_result.py').read())
        
        output_file = tmp_path / 'agg_test_result.json'
        with open(output_file) as f:
            result = json.load(f)
        
        # With explicit GPU counts
        assert result['output_tput_per_gpu'] == 3000.0 / 4
        assert result['input_tput_per_gpu'] == (10000.0 - 3000.0) / 4
        
    finally:
        os.chdir(original_dir)


def test_prefill_decode_gpus_defaults_to_tp(tmp_path, sample_benchmark_result, basic_env_vars):
    """Test that prefill/decode GPUs default to TP size when not specified."""
    # Default env vars have empty strings for PREFILL_GPUS and DECODE_GPUS
    result_file = tmp_path / 'test_result.json'
    with open(result_file, 'w') as f:
        json.dump(sample_benchmark_result, f)
    
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        with patch.dict(os.environ, basic_env_vars):
            exec(open('/home/runner/work/InferenceMAX/InferenceMAX/utils/process_result.py').read())
        
        output_file = tmp_path / 'agg_test_result.json'
        with open(output_file) as f:
            result = json.load(f)
        
        # Should use TP size (8) when PREFILL_GPUS and DECODE_GPUS are empty
        assert result['output_tput_per_gpu'] == 3000.0 / 8
        assert result['input_tput_per_gpu'] == (10000.0 - 3000.0) / 8
        
    finally:
        os.chdir(original_dir)


def test_different_tp_sizes(tmp_path, sample_benchmark_result, basic_env_vars):
    """Test processing with different TP sizes."""
    test_cases = [
        ('1', 1),
        ('2', 2),
        ('4', 4),
        ('8', 8),
        ('16', 16)
    ]
    
    for tp_str, tp_int in test_cases:
        env_vars = basic_env_vars.copy()
        env_vars['TP'] = tp_str
        
        result_file = tmp_path / 'test_result.json'
        with open(result_file, 'w') as f:
            json.dump(sample_benchmark_result, f)
        
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            with patch.dict(os.environ, env_vars):
                exec(open('/home/runner/work/InferenceMAX/InferenceMAX/utils/process_result.py').read())
            
            output_file = tmp_path / 'agg_test_result.json'
            with open(output_file) as f:
                result = json.load(f)
            
            assert result['tp'] == tp_int
            assert result['tput_per_gpu'] == 10000.0 / tp_int
            
        finally:
            os.chdir(original_dir)


def test_different_ep_sizes(tmp_path, sample_benchmark_result, basic_env_vars):
    """Test processing with different EP sizes."""
    test_cases = [1, 2, 4, 8]
    
    for ep_size in test_cases:
        env_vars = basic_env_vars.copy()
        env_vars['EP_SIZE'] = str(ep_size)
        
        result_file = tmp_path / 'test_result.json'
        with open(result_file, 'w') as f:
            json.dump(sample_benchmark_result, f)
        
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            with patch.dict(os.environ, env_vars):
                exec(open('/home/runner/work/InferenceMAX/InferenceMAX/utils/process_result.py').read())
            
            output_file = tmp_path / 'agg_test_result.json'
            with open(output_file) as f:
                result = json.load(f)
            
            assert result['ep'] == ep_size
            
        finally:
            os.chdir(original_dir)


def test_output_file_content_structure(tmp_path, sample_benchmark_result, basic_env_vars):
    """Test that output file has the expected structure."""
    result_file = tmp_path / 'test_result.json'
    with open(result_file, 'w') as f:
        json.dump(sample_benchmark_result, f)
    
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        with patch.dict(os.environ, basic_env_vars):
            exec(open('/home/runner/work/InferenceMAX/InferenceMAX/utils/process_result.py').read())
        
        output_file = tmp_path / 'agg_test_result.json'
        with open(output_file) as f:
            result = json.load(f)
        
        # Check required fields exist
        required_fields = [
            'hw', 'tp', 'ep', 'dp_attention', 'conc', 'model',
            'framework', 'precision', 'tput_per_gpu', 
            'output_tput_per_gpu', 'input_tput_per_gpu'
        ]
        
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
        
    finally:
        os.chdir(original_dir)


def test_complex_benchmark_result(tmp_path, basic_env_vars):
    """Test processing with a more complex benchmark result."""
    complex_result = {
        'max_concurrency': 16,
        'model_id': 'meta-llama/Llama-3-405b',
        'total_token_throughput': 50000.0,
        'output_throughput': 15000.0,
        'ttft_ms': 100.0,
        'tpot_ms': 15.0,
        'e2e_latency_ms': 2000.0,
        'decode_tpot_ms': 12.0,
        'prefill_tpot_ms': 18.0,
        'p50_latency_ms': 1500.0,
        'p90_latency_ms': 2500.0,
        'p99_latency_ms': 3000.0
    }
    
    result_file = tmp_path / 'test_result.json'
    with open(result_file, 'w') as f:
        json.dump(complex_result, f)
    
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        with patch.dict(os.environ, basic_env_vars):
            exec(open('/home/runner/work/InferenceMAX/InferenceMAX/utils/process_result.py').read())
        
        output_file = tmp_path / 'agg_test_result.json'
        with open(output_file) as f:
            result = json.load(f)
        
        # Check all ms values were converted
        assert result['ttft'] == 100.0 / 1000.0
        assert result['tpot'] == 15.0 / 1000.0
        assert result['e2e_latency'] == 2000.0 / 1000.0
        assert result['p50_latency'] == 1500.0 / 1000.0
        assert result['p90_latency'] == 2500.0 / 1000.0
        assert result['p99_latency'] == 3000.0 / 1000.0
        
        # Check tpot to intvty conversions
        assert 'intvty' in result
        assert 'decode_intvty' in result
        assert 'prefill_intvty' in result
        
    finally:
        os.chdir(original_dir)


def test_dp_attention_values(tmp_path, sample_benchmark_result, basic_env_vars):
    """Test different DP_ATTENTION values."""
    test_values = ['true', 'false', 'True', 'False']
    
    for dp_attn_value in test_values:
        env_vars = basic_env_vars.copy()
        env_vars['DP_ATTENTION'] = dp_attn_value
        
        result_file = tmp_path / 'test_result.json'
        with open(result_file, 'w') as f:
            json.dump(sample_benchmark_result, f)
        
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            with patch.dict(os.environ, env_vars):
                exec(open('/home/runner/work/InferenceMAX/InferenceMAX/utils/process_result.py').read())
            
            output_file = tmp_path / 'agg_test_result.json'
            with open(output_file) as f:
                result = json.load(f)
            
            assert result['dp_attention'] == dp_attn_value
            
        finally:
            os.chdir(original_dir)


def test_different_frameworks(tmp_path, sample_benchmark_result, basic_env_vars):
    """Test different framework values."""
    frameworks = ['vllm', 'trt', 'sglang', 'tensorrt-llm']
    
    for framework in frameworks:
        env_vars = basic_env_vars.copy()
        env_vars['FRAMEWORK'] = framework
        
        result_file = tmp_path / 'test_result.json'
        with open(result_file, 'w') as f:
            json.dump(sample_benchmark_result, f)
        
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            with patch.dict(os.environ, env_vars):
                exec(open('/home/runner/work/InferenceMAX/InferenceMAX/utils/process_result.py').read())
            
            output_file = tmp_path / 'agg_test_result.json'
            with open(output_file) as f:
                result = json.load(f)
            
            assert result['framework'] == framework
            
        finally:
            os.chdir(original_dir)


def test_different_precisions(tmp_path, sample_benchmark_result, basic_env_vars):
    """Test different precision values."""
    precisions = ['fp8', 'fp16', 'fp32', 'int8', 'int4']
    
    for precision in precisions:
        env_vars = basic_env_vars.copy()
        env_vars['PRECISION'] = precision
        
        result_file = tmp_path / 'test_result.json'
        with open(result_file, 'w') as f:
            json.dump(sample_benchmark_result, f)
        
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            with patch.dict(os.environ, env_vars):
                exec(open('/home/runner/work/InferenceMAX/InferenceMAX/utils/process_result.py').read())
            
            output_file = tmp_path / 'agg_test_result.json'
            with open(output_file) as f:
                result = json.load(f)
            
            assert result['precision'] == precision
            
        finally:
            os.chdir(original_dir)


def test_throughput_calculations(tmp_path, basic_env_vars):
    """Test throughput calculations with various values."""
    benchmark_result = {
        'max_concurrency': 10,
        'model_id': 'test/model',
        'total_token_throughput': 24000.0,
        'output_throughput': 8000.0
    }
    
    env_vars = basic_env_vars.copy()
    env_vars['TP'] = '4'
    env_vars['PREFILL_GPUS'] = '2'
    env_vars['DECODE_GPUS'] = '2'
    
    result_file = tmp_path / 'test_result.json'
    with open(result_file, 'w') as f:
        json.dump(benchmark_result, f)
    
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        with patch.dict(os.environ, env_vars):
            exec(open('/home/runner/work/InferenceMAX/InferenceMAX/utils/process_result.py').read())
        
        output_file = tmp_path / 'agg_test_result.json'
        with open(output_file) as f:
            result = json.load(f)
        
        # tput_per_gpu = total_token_throughput / tp_size
        assert result['tput_per_gpu'] == 24000.0 / 4
        
        # output_tput_per_gpu = output_throughput / decode_gpus
        assert result['output_tput_per_gpu'] == 8000.0 / 2
        
        # input_tput_per_gpu = (total_token_throughput - output_throughput) / prefill_gpus
        assert result['input_tput_per_gpu'] == (24000.0 - 8000.0) / 2
        
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
