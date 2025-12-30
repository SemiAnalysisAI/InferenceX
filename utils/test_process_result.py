"""Comprehensive tests for process_result.py

Since process_result.py executes code at module import time, we test it by:
1. Testing the get_required_env_vars function directly
2. Running the script as a subprocess with mocked environment and files
"""
import pytest
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent / "process_result.py"


# =============================================================================
# Test Fixtures - Based on real benchmark output structure
# =============================================================================

@pytest.fixture
def sample_benchmark_result():
    """Sample benchmark result JSON based on real output structure."""
    return {
        "model_id": "deepseek-ai/DeepSeek-R1-0528",
        "max_concurrency": 64,
        "total_token_throughput": 15000.5,
        "output_throughput": 12000.0,
        "ttft_p50_ms": 150.5,
        "ttft_p99_ms": 250.3,
        "tpot_p50_ms": 25.0,
        "tpot_p99_ms": 45.0,
        "e2e_latency_p50_ms": 1500.0,
        "e2e_latency_p99_ms": 2500.0,
    }


@pytest.fixture
def base_env_vars():
    """Base environment variables for single-node setup."""
    return {
        "RUNNER_TYPE": "mi300x",
        "FRAMEWORK": "sglang",
        "PRECISION": "fp8",
        "SPEC_DECODING": "none",
        "RESULT_FILENAME": "benchmark_result",
        "ISL": "1024",
        "OSL": "1024",
        "DISAGG": "false",
        "MODEL_PREFIX": "dsr1",
        "IMAGE": "test-image:latest",
    }


@pytest.fixture
def single_node_env_vars(base_env_vars):
    """Environment variables for single-node setup."""
    return {
        **base_env_vars,
        "TP": "8",
        "EP_SIZE": "1",
        "DP_ATTENTION": "false",
    }


@pytest.fixture
def multinode_env_vars(base_env_vars):
    """Environment variables for multinode setup based on gb200 config."""
    return {
        **base_env_vars,
        "RUNNER_TYPE": "gb200",
        "FRAMEWORK": "dynamo-trt",
        "PRECISION": "fp4",
        "DISAGG": "true",
        "IS_MULTINODE": "true",
        "PREFILL_GPUS": "20",
        "DECODE_GPUS": "8",
        "PREFILL_NUM_WORKERS": "5",
        "PREFILL_TP": "4",
        "PREFILL_EP": "4",
        "PREFILL_DP_ATTN": "true",
        "DECODE_NUM_WORKERS": "1",
        "DECODE_TP": "8",
        "DECODE_EP": "8",
        "DECODE_DP_ATTN": "true",
    }


def run_script(tmp_path, env, benchmark_result, result_filename="benchmark_result"):
    """Helper to run the process_result.py script."""
    result_file = tmp_path / f"{result_filename}.json"
    result_file.write_text(json.dumps(benchmark_result))

    env = env.copy()
    env["RESULT_FILENAME"] = result_filename

    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )


# =============================================================================
# Test get_required_env_vars function
# =============================================================================

class TestGetRequiredEnvVars:
    """Tests for get_required_env_vars function."""

    def test_all_vars_present(self, monkeypatch):
        """Should return dict when all vars present."""
        monkeypatch.setenv("TEST_VAR_1", "value1")
        monkeypatch.setenv("TEST_VAR_2", "value2")

        import os

        def get_required_env_vars(required_vars):
            env_values = {}
            missing_env_vars = []
            for var_name in required_vars:
                value = os.environ.get(var_name)
                if value is None:
                    missing_env_vars.append(var_name)
                env_values[var_name] = value
            if missing_env_vars:
                raise EnvironmentError(
                    f"Missing required environment variables: {', '.join(missing_env_vars)}")
            return env_values

        result = get_required_env_vars(["TEST_VAR_1", "TEST_VAR_2"])
        assert result["TEST_VAR_1"] == "value1"
        assert result["TEST_VAR_2"] == "value2"

    def test_missing_vars_raises_error(self, monkeypatch):
        """Should raise EnvironmentError when vars missing."""
        import os

        def get_required_env_vars(required_vars):
            env_values = {}
            missing_env_vars = []
            for var_name in required_vars:
                value = os.environ.get(var_name)
                if value is None:
                    missing_env_vars.append(var_name)
                env_values[var_name] = value
            if missing_env_vars:
                raise EnvironmentError(
                    f"Missing required environment variables: {', '.join(missing_env_vars)}")
            return env_values

        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)

        with pytest.raises(EnvironmentError) as exc_info:
            get_required_env_vars(["NONEXISTENT_VAR"])
        assert "NONEXISTENT_VAR" in str(exc_info.value)


# =============================================================================
# Test script execution via subprocess
# =============================================================================

class TestProcessResultScript:
    """Tests for process_result.py script execution."""

    def test_single_node_processing(self, tmp_path, sample_benchmark_result, single_node_env_vars):
        """Test single-node result processing."""
        result = run_script(tmp_path, single_node_env_vars, sample_benchmark_result)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        output_data = json.loads(result.stdout)

        # Verify base fields
        assert output_data["hw"] == "mi300x"
        assert output_data["framework"] == "sglang"
        assert output_data["precision"] == "fp8"
        assert output_data["spec_decoding"] == "none"
        assert output_data["model"] == "deepseek-ai/DeepSeek-R1-0528"
        assert output_data["conc"] == 64
        assert output_data["isl"] == 1024
        assert output_data["osl"] == 1024
        assert output_data["disagg"] is False

        # Verify single-node specific fields
        assert output_data["is_multinode"] is False
        assert output_data["tp"] == 8
        assert output_data["ep"] == 1
        assert output_data["dp_attention"] == "false"

        # Verify throughput calculations (divided by tp=8)
        assert output_data["tput_per_gpu"] == pytest.approx(15000.5 / 8)
        assert output_data["output_tput_per_gpu"] == pytest.approx(12000.0 / 8)
        assert output_data["input_tput_per_gpu"] == pytest.approx((15000.5 - 12000.0) / 8)

        # Verify latency conversions (ms to seconds)
        assert output_data["ttft_p50"] == pytest.approx(0.1505)
        assert output_data["ttft_p99"] == pytest.approx(0.2503)
        assert output_data["e2e_latency_p50"] == pytest.approx(1.5)
        assert output_data["e2e_latency_p99"] == pytest.approx(2.5)

        # Verify interactivity calculations (1000 / tpot_ms)
        assert output_data["intvty_p50"] == pytest.approx(1000.0 / 25.0)
        assert output_data["intvty_p99"] == pytest.approx(1000.0 / 45.0)

        # Verify output file created
        output_file = tmp_path / "agg_benchmark_result.json"
        assert output_file.exists()

    def test_multinode_processing(self, tmp_path, sample_benchmark_result, multinode_env_vars):
        """Test multinode result processing."""
        result = run_script(tmp_path, multinode_env_vars, sample_benchmark_result)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        output_data = json.loads(result.stdout)

        # Verify base fields
        assert output_data["hw"] == "gb200"
        assert output_data["framework"] == "dynamo-trt"
        assert output_data["precision"] == "fp4"
        assert output_data["disagg"] is True

        # Verify multinode specific fields
        assert output_data["is_multinode"] is True
        assert output_data["prefill_tp"] == 4
        assert output_data["prefill_ep"] == 4
        assert output_data["prefill_dp_attention"] == "true"
        assert output_data["prefill_num_workers"] == 5
        assert output_data["decode_tp"] == 8
        assert output_data["decode_ep"] == 8
        assert output_data["decode_dp_attention"] == "true"
        assert output_data["decode_num_workers"] == 1
        assert output_data["num_prefill_gpu"] == 20
        assert output_data["num_decode_gpu"] == 8

        # Verify throughput calculations
        total_gpus = 20 + 8  # prefill + decode
        assert output_data["tput_per_gpu"] == pytest.approx(15000.5 / total_gpus)
        assert output_data["output_tput_per_gpu"] == pytest.approx(12000.0 / 8)  # decode gpus
        assert output_data["input_tput_per_gpu"] == pytest.approx((15000.5 - 12000.0) / 20)  # prefill gpus

    def test_missing_base_env_vars(self, tmp_path, sample_benchmark_result):
        """Test that missing base env vars causes failure."""
        result_file = tmp_path / "benchmark_result.json"
        result_file.write_text(json.dumps(sample_benchmark_result))

        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=tmp_path,
            env={"PATH": "/usr/bin", "RESULT_FILENAME": "benchmark_result"},
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "Missing required environment variables" in result.stderr

    def test_missing_single_node_env_vars(self, tmp_path, sample_benchmark_result, base_env_vars):
        """Test that missing single-node env vars causes failure."""
        # base_env_vars doesn't have TP, EP_SIZE, DP_ATTENTION
        result = run_script(tmp_path, base_env_vars, sample_benchmark_result)

        assert result.returncode != 0
        assert "Missing required environment variables" in result.stderr

    def test_missing_multinode_env_vars(self, tmp_path, sample_benchmark_result, base_env_vars):
        """Test that missing multinode env vars causes failure."""
        env = base_env_vars.copy()
        env["IS_MULTINODE"] = "true"
        env["DISAGG"] = "true"
        # Missing multinode-specific vars

        result = run_script(tmp_path, env, sample_benchmark_result)

        assert result.returncode != 0
        assert "Missing required environment variables" in result.stderr

    def test_disagg_without_multinode_fails(self, tmp_path, sample_benchmark_result, single_node_env_vars):
        """Test that disagg=true without multinode raises error."""
        env = single_node_env_vars.copy()
        env["DISAGG"] = "true"  # Disagg without multinode

        result = run_script(tmp_path, env, sample_benchmark_result)

        assert result.returncode != 0
        assert "Disaggregated mode requires multinode setup" in result.stderr

    def test_missing_result_file(self, tmp_path, single_node_env_vars):
        """Test that missing result file causes failure."""
        env = single_node_env_vars.copy()
        env["RESULT_FILENAME"] = "nonexistent"

        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0


# =============================================================================
# Test latency and throughput calculations
# =============================================================================

class TestCalculations:
    """Tests for throughput and latency calculations."""

    def test_latency_ms_to_seconds_conversion(self, tmp_path, single_node_env_vars):
        """Test that _ms fields are converted to seconds."""
        benchmark_result = {
            "model_id": "test-model",
            "max_concurrency": 8,
            "total_token_throughput": 1000.0,
            "output_throughput": 800.0,
            "custom_metric_ms": 500.0,  # Should become custom_metric = 0.5
        }

        result = run_script(tmp_path, single_node_env_vars, benchmark_result)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        output_data = json.loads(result.stdout)
        assert output_data["custom_metric"] == pytest.approx(0.5)

    def test_tpot_to_interactivity_conversion(self, tmp_path, single_node_env_vars):
        """Test that tpot fields are converted to interactivity."""
        benchmark_result = {
            "model_id": "test-model",
            "max_concurrency": 8,
            "total_token_throughput": 1000.0,
            "output_throughput": 800.0,
            "tpot_p50_ms": 20.0,  # Should become intvty_p50 = 50
            "tpot_p99_ms": 50.0,  # Should become intvty_p99 = 20
        }

        result = run_script(tmp_path, single_node_env_vars, benchmark_result)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        output_data = json.loads(result.stdout)
        assert output_data["intvty_p50"] == pytest.approx(50.0)
        assert output_data["intvty_p99"] == pytest.approx(20.0)

    def test_throughput_per_gpu_single_node(self, tmp_path, single_node_env_vars):
        """Test throughput per GPU calculation for single node."""
        benchmark_result = {
            "model_id": "test-model",
            "max_concurrency": 8,
            "total_token_throughput": 8000.0,
            "output_throughput": 6000.0,
        }

        env = single_node_env_vars.copy()
        env["TP"] = "4"

        result = run_script(tmp_path, env, benchmark_result)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        output_data = json.loads(result.stdout)
        assert output_data["tput_per_gpu"] == pytest.approx(2000.0)  # 8000 / 4
        assert output_data["output_tput_per_gpu"] == pytest.approx(1500.0)  # 6000 / 4
        assert output_data["input_tput_per_gpu"] == pytest.approx(500.0)  # (8000 - 6000) / 4

    def test_throughput_per_gpu_multinode(self, tmp_path, multinode_env_vars):
        """Test throughput per GPU calculation for multinode."""
        benchmark_result = {
            "model_id": "test-model",
            "max_concurrency": 64,
            "total_token_throughput": 28000.0,  # Will be divided by total GPUs
            "output_throughput": 16000.0,  # Will be divided by decode GPUs
        }

        env = multinode_env_vars.copy()
        env["PREFILL_GPUS"] = "20"
        env["DECODE_GPUS"] = "8"

        result = run_script(tmp_path, env, benchmark_result)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        output_data = json.loads(result.stdout)
        assert output_data["tput_per_gpu"] == pytest.approx(1000.0)  # 28000 / 28
        assert output_data["output_tput_per_gpu"] == pytest.approx(2000.0)  # 16000 / 8
        assert output_data["input_tput_per_gpu"] == pytest.approx(600.0)  # (28000 - 16000) / 20


# =============================================================================
# Test output file generation
# =============================================================================

class TestOutputFile:
    """Tests for output file generation."""

    def test_output_file_created(self, tmp_path, sample_benchmark_result, single_node_env_vars):
        """Test that aggregated output file is created."""
        result = run_script(tmp_path, single_node_env_vars, sample_benchmark_result)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        output_file = tmp_path / "agg_benchmark_result.json"
        assert output_file.exists()

        # Verify content matches stdout
        with open(output_file) as f:
            file_content = json.load(f)

        stdout_content = json.loads(result.stdout)
        assert file_content == stdout_content

    def test_output_file_has_correct_prefix(self, tmp_path, sample_benchmark_result, single_node_env_vars):
        """Test that output file has 'agg_' prefix."""
        result = run_script(tmp_path, single_node_env_vars, sample_benchmark_result, "my_custom_result")
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        output_file = tmp_path / "agg_my_custom_result.json"
        assert output_file.exists()


# =============================================================================
# Test edge cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_boolean_disagg_parsing_false(self, tmp_path, sample_benchmark_result, single_node_env_vars):
        """Test that DISAGG env var is parsed as boolean correctly for false values."""
        for disagg_value in ["false", "False", "FALSE"]:
            env = single_node_env_vars.copy()
            env["DISAGG"] = disagg_value

            result = run_script(tmp_path, env, sample_benchmark_result)
            assert result.returncode == 0, f"Script failed for DISAGG={disagg_value}: {result.stderr}"

            output_data = json.loads(result.stdout)
            assert output_data["disagg"] is False

    def test_boolean_disagg_parsing_true_requires_multinode(self, tmp_path, sample_benchmark_result, single_node_env_vars):
        """Test that DISAGG=true without multinode fails."""
        for disagg_value in ["true", "True", "TRUE"]:
            env = single_node_env_vars.copy()
            env["DISAGG"] = disagg_value

            result = run_script(tmp_path, env, sample_benchmark_result)
            assert result.returncode != 0

    def test_is_multinode_default_false(self, tmp_path, sample_benchmark_result, single_node_env_vars):
        """Test that IS_MULTINODE defaults to false when not set."""
        # Don't set IS_MULTINODE
        result = run_script(tmp_path, single_node_env_vars, sample_benchmark_result)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        output_data = json.loads(result.stdout)
        assert output_data["is_multinode"] is False

    def test_integer_conversion(self, tmp_path, single_node_env_vars):
        """Test that numeric env vars are converted to integers."""
        benchmark_result = {
            "model_id": "test-model",
            "max_concurrency": 32,
            "total_token_throughput": 5000.0,
            "output_throughput": 4000.0,
        }

        env = single_node_env_vars.copy()
        env["ISL"] = "8192"
        env["OSL"] = "1024"

        result = run_script(tmp_path, env, benchmark_result)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        output_data = json.loads(result.stdout)
        assert output_data["isl"] == 8192
        assert output_data["osl"] == 1024
        assert isinstance(output_data["isl"], int)
        assert isinstance(output_data["osl"], int)

    def test_conc_from_benchmark_result(self, tmp_path, single_node_env_vars):
        """Test that conc is read from benchmark result max_concurrency."""
        benchmark_result = {
            "model_id": "test-model",
            "max_concurrency": 128,
            "total_token_throughput": 5000.0,
            "output_throughput": 4000.0,
        }

        result = run_script(tmp_path, single_node_env_vars, benchmark_result)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        output_data = json.loads(result.stdout)
        assert output_data["conc"] == 128
